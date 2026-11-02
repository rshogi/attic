//! `cargo xtask build-pgo`: a release build guided by a profile of the engine
//! searching a fixed set of positions.
//!
//! Three stages, all run in one go:
//!
//! 1. build `attic` with `-Cprofile-generate` into a target directory of its own;
//! 2. drive that binary over USI through the training positions, so it writes
//!    `.profraw` files on exit, then merge them with `llvm-profdata`;
//! 3. build `attic` again, in the ordinary target directory, with
//!    `-Cprofile-use` pointing at the merged profile.
//!
//! The rustflags handling is the part that is easy to get wrong. Cargo reads
//! `build.rustflags` from `.cargo/config.toml` (here `-C target-cpu=native`)
//! only when no `RUSTFLAGS` environment variable is set; as soon as one is, the
//! config value is dropped, not merged. So the profile flag is added in
//! whichever place the caller already uses: appended to `RUSTFLAGS` when that is
//! set, or given as a `--config 'build.rustflags=[...]'` array otherwise (cargo
//! concatenates the arrays, so `-C target-cpu=native` stays in effect).

use std::collections::VecDeque;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command as ProcessCommand, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use attic_state::parse_sfen;

use super::{resolve_under, workspace_root};

/// Training positions shipped with the repository, relative to the workspace
/// root.
const POSITIONS_DEFAULT: &str = "crates/xtask/pgo-positions.sfen";
const EVAL_DIR_DEFAULT: &str = "eval";
/// The file the engine loads from `EvalDir`.
const EVAL_FILE_NAME: &str = "nn.bin";
/// The package the profile-guided binary is built from, and its binary name.
const ENGINE_PACKAGE: &str = "attic";

/// Separator `CARGO_ENCODED_RUSTFLAGS` uses between flags.
const ENCODED_FLAG_SEPARATOR: char = '\x1f';

#[derive(clap::Args, Debug)]
pub struct BuildPgoArgs {
    /// Cargo features to enable on `attic`. May be repeated or comma-separated,
    /// as for `cargo build --features`.
    #[arg(long)]
    features: Vec<String>,
    /// Do not enable `attic`'s default features (`cargo build --no-default-features`).
    #[arg(long)]
    no_default_features: bool,
    /// Directory holding the evaluation network (`nn.bin`) the training run
    /// loads; the same meaning as the engine's `EvalDir` option. A relative path
    /// resolves against the workspace root.
    #[arg(long, default_value = EVAL_DIR_DEFAULT)]
    eval_dir: PathBuf,
    /// Node limit (`go nodes N`) of each training search.
    #[arg(long, default_value_t = 100_000, value_parser = clap::value_parser!(u64).range(1..))]
    nodes: u64,
    /// File of training positions, one SFEN per line (a leading `sfen ` is
    /// accepted; blank lines and lines starting with `#` are skipped). A relative
    /// path resolves against the workspace root. Defaults to the set shipped in
    /// `crates/xtask/pgo-positions.sfen`.
    #[arg(long)]
    positions: Option<PathBuf>,
    /// Keep the raw `.profraw` files under `<target>/pgo/data` after merging
    /// them. The merged profile is always kept.
    #[arg(long)]
    keep_profile: bool,
}

/// What a finished run produced, for the closing report.
#[derive(Debug)]
pub struct PgoOutcome {
    pub binary: PathBuf,
    pub profile: PathBuf,
    pub positions: usize,
    pub nodes: u64,
}

pub fn build_pgo(args: &BuildPgoArgs) -> Result<PgoOutcome, String> {
    let root = workspace_root()?;

    // Everything that can be checked cheaply is checked before the first build,
    // so a missing input does not surface after minutes of compiling.
    let positions_path = match &args.positions {
        Some(p) => resolve_under(&root, p),
        None => root.join(POSITIONS_DEFAULT),
    };
    let positions = load_positions(&positions_path)?;

    let eval_dir = resolve_under(&root, &args.eval_dir);
    let eval_file = eval_dir.join(EVAL_FILE_NAME);
    if !eval_file.is_file() {
        return Err(format!(
            "evaluation network not found at {}; the PGO training run loads it and a \
             profile-guided build cannot be made without it. Put `{EVAL_FILE_NAME}` there \
             or point `--eval-dir` at the directory that holds it",
            eval_file.display()
        ));
    }

    let llvm_profdata = find_llvm_profdata(&root)?;

    let target_dir = resolve_target_dir(&root)?;
    let pgo_dir = target_dir.join("pgo");
    let instrumented_dir = pgo_dir.join("instrumented");
    let data_dir = pgo_dir.join("data");
    let merged = pgo_dir.join("merged.profdata");

    // A stale profile of an older binary must not leak into this merge.
    remove_dir_if_exists(&data_dir)?;
    if merged.exists() {
        fs::remove_file(&merged)
            .map_err(|e| format!("could not remove {}: {e}", merged.display()))?;
    }

    let mut lap = Instant::now();
    println!(
        "[1/4] building the instrumented binary into {}",
        instrumented_dir.display()
    );
    cargo_build(
        &root,
        &instrumented_dir,
        &format!("-Cprofile-generate={}", utf8(&data_dir)?),
        args,
    )?;
    let instrumented_binary = instrumented_dir.join("release").join(ENGINE_PACKAGE);
    if !instrumented_binary.is_file() {
        return Err(format!(
            "expected the instrumented binary at {} but it was not produced",
            instrumented_binary.display()
        ));
    }

    finish_stage(&mut lap);

    // The flag reaches the build scripts too (cargo applies rustflags to them
    // when there is no `--target`), and they write `.profraw` files into the
    // same directory as they run during the build. Those belong to no binary
    // that is shipped, and would let an engine that wrote nothing pass the
    // "no .profraw" check below, so only what the training run writes is kept.
    remove_dir_if_exists(&data_dir)?;
    fs::create_dir_all(&data_dir)
        .map_err(|e| format!("could not create {}: {e}", data_dir.display()))?;

    println!(
        "[2/4] training run: {} positions x go nodes {}",
        positions.len(),
        args.nodes
    );
    train(&instrumented_binary, &eval_dir, &positions, args.nodes)?;
    finish_stage(&mut lap);

    println!("[3/4] merging the profile into {}", merged.display());
    merge_profile(&llvm_profdata, &data_dir, &merged, args.keep_profile)?;
    finish_stage(&mut lap);

    println!(
        "[4/4] building the profile-guided binary into {}",
        target_dir.display()
    );
    cargo_build(
        &root,
        &target_dir,
        &format!("-Cprofile-use={}", utf8(&merged)?),
        args,
    )?;
    let binary = target_dir.join("release").join(ENGINE_PACKAGE);
    if !binary.is_file() {
        return Err(format!(
            "expected the profile-guided binary at {} but it was not produced",
            binary.display()
        ));
    }
    finish_stage(&mut lap);

    Ok(PgoOutcome {
        binary,
        profile: merged,
        positions: positions.len(),
        nodes: args.nodes,
    })
}

/// Print how long the stage that just ended took and start timing the next.
fn finish_stage(lap: &mut Instant) {
    println!("      done in {:.1}s", lap.elapsed().as_secs_f64());
    *lap = Instant::now();
}

/// The target directory the profile-guided binary lands in: `CARGO_TARGET_DIR`
/// when set (a relative value is taken against the directory the user ran
/// from, as cargo itself would), `<workspace>/target` otherwise. It is passed to
/// every cargo invocation explicitly so the path reported at the end is the one
/// that was built into.
fn resolve_target_dir(root: &Path) -> Result<PathBuf, String> {
    match env::var_os("CARGO_TARGET_DIR") {
        Some(v) if !v.is_empty() => {
            let p = PathBuf::from(v);
            if p.is_absolute() {
                Ok(p)
            } else {
                let cwd = env::current_dir()
                    .map_err(|e| format!("could not read the current directory: {e}"))?;
                Ok(cwd.join(p))
            }
        }
        _ => Ok(root.join("target")),
    }
}

fn remove_dir_if_exists(dir: &Path) -> Result<(), String> {
    match fs::remove_dir_all(dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("could not remove {}: {e}", dir.display())),
    }
}

fn utf8(p: &Path) -> Result<&str, String> {
    p.to_str()
        .ok_or_else(|| format!("path {} is not valid UTF-8", p.display()))
}

// ---------------------------------------------------------------------------
// Training positions
// ---------------------------------------------------------------------------

fn load_positions(path: &Path) -> Result<Vec<String>, String> {
    let text = fs::read_to_string(path).map_err(|e| {
        format!(
            "could not read the training positions {}: {e}",
            path.display()
        )
    })?;
    parse_positions(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// One SFEN per line, validated with the workspace parser so a malformed line
/// is reported with its line number instead of making the engine search a
/// position other than the one intended.
fn parse_positions(text: &str) -> Result<Vec<String>, String> {
    let mut positions = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let sfen = line.strip_prefix("sfen ").unwrap_or(line).trim();
        parse_sfen(sfen).map_err(|e| format!("line {}: invalid SFEN ({e:?}): {sfen}", i + 1))?;
        positions.push(sfen.to_string());
    }
    if positions.is_empty() {
        return Err("no positions in the file".to_string());
    }
    Ok(positions)
}

// ---------------------------------------------------------------------------
// cargo
// ---------------------------------------------------------------------------

/// How one extra rustc flag reaches cargo, given the rustflags environment the
/// caller already has.
#[derive(Debug, PartialEq, Eq)]
enum RustflagPlan {
    /// Set `CARGO_ENCODED_RUSTFLAGS` to this value (it outranks `RUSTFLAGS`).
    EncodedEnv(String),
    /// Set `RUSTFLAGS` to this value.
    PlainEnv(String),
    /// Pass this value as `--config`, adding to `build.rustflags` from the
    /// config files instead of replacing it.
    Config(String),
}

/// Decide how to add `flag` without losing the rustflags already in effect.
///
/// Cargo uses the first of `CARGO_ENCODED_RUSTFLAGS`, `RUSTFLAGS` and the
/// config files' `build.rustflags` that is set, and ignores the rest. A set but
/// empty variable counts as set. `RUSTFLAGS` is split on whitespace, so a flag
/// that itself contains whitespace cannot be appended to it.
fn plan_rustflag(
    encoded: Option<&str>,
    plain: Option<&str>,
    flag: &str,
) -> Result<RustflagPlan, String> {
    if let Some(encoded) = encoded {
        let mut value = encoded.to_string();
        if !value.is_empty() {
            value.push(ENCODED_FLAG_SEPARATOR);
        }
        value.push_str(flag);
        return Ok(RustflagPlan::EncodedEnv(value));
    }
    if let Some(plain) = plain {
        if flag.contains(char::is_whitespace) {
            return Err(format!(
                "`{flag}` contains whitespace and cannot be appended to RUSTFLAGS, which cargo \
                 splits on whitespace; use a target directory without spaces in its path, or \
                 unset RUSTFLAGS"
            ));
        }
        let mut value = plain.trim_end().to_string();
        if !value.is_empty() {
            value.push(' ');
        }
        value.push_str(flag);
        return Ok(RustflagPlan::PlainEnv(value));
    }
    Ok(RustflagPlan::Config(format!(
        "build.rustflags=[{}]",
        toml_basic_string(flag)
    )))
}

/// A TOML basic string literal for `s`.
fn toml_basic_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04X}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn env_string(name: &str) -> Result<Option<String>, String> {
    match env::var(name) {
        Ok(v) => Ok(Some(v)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(_)) => Err(format!("{name} is not valid UTF-8")),
    }
}

fn cargo_program() -> OsString {
    env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

/// `cargo build --release -p attic <feature arguments>` into `target_dir`, with
/// `flag` added to the rustc flags.
fn cargo_build(
    root: &Path,
    target_dir: &Path,
    flag: &str,
    args: &BuildPgoArgs,
) -> Result<(), String> {
    let plan = plan_rustflag(
        env_string("CARGO_ENCODED_RUSTFLAGS")?.as_deref(),
        env_string("RUSTFLAGS")?.as_deref(),
        flag,
    )?;

    let mut cmd = ProcessCommand::new(cargo_program());
    // Run from the workspace root so `.cargo/config.toml` and the pinned
    // toolchain are found whatever directory the user started from.
    cmd.current_dir(root)
        .arg("build")
        .arg("--release")
        .args(["-p", ENGINE_PACKAGE])
        .env("CARGO_TARGET_DIR", target_dir)
        .stdin(Stdio::null());
    if args.no_default_features {
        cmd.arg("--no-default-features");
    }
    for features in &args.features {
        cmd.args(["--features", features]);
    }
    match &plan {
        RustflagPlan::EncodedEnv(v) => {
            cmd.env("CARGO_ENCODED_RUSTFLAGS", v);
        }
        RustflagPlan::PlainEnv(v) => {
            cmd.env("RUSTFLAGS", v);
        }
        RustflagPlan::Config(c) => {
            cmd.args(["--config", c]);
        }
    }

    let status = cmd
        .status()
        .map_err(|e| format!("failed to spawn cargo: {e}"))?;
    if !status.success() {
        return Err(format!("`cargo build` with `{flag}` exited with {status}"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// llvm-profdata
// ---------------------------------------------------------------------------

fn rustc_program() -> OsString {
    env::var_os("RUSTC").unwrap_or_else(|| OsString::from("rustc"))
}

fn rustc_output(root: &Path, args: &[&str]) -> Result<String, String> {
    let out = ProcessCommand::new(rustc_program())
        .current_dir(root)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("failed to run `rustc {}`: {e}", args.join(" ")))?;
    if !out.status.success() {
        return Err(format!(
            "`rustc {}` exited with {}: {}",
            args.join(" "),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    String::from_utf8(out.stdout)
        .map_err(|_| format!("`rustc {}` printed non-UTF-8 output", args.join(" ")))
}

/// The host triple from the output of `rustc -vV`.
fn parse_host_triple(version_verbose: &str) -> Option<&str> {
    version_verbose
        .lines()
        .find_map(|l| l.strip_prefix("host: "))
        .map(str::trim)
        .filter(|h| !h.is_empty())
}

/// `llvm-profdata` of the pinned toolchain: the `llvm-tools` rustup component
/// puts it under the sysroot, not on `PATH`.
fn find_llvm_profdata(root: &Path) -> Result<PathBuf, String> {
    let sysroot = rustc_output(root, &["--print", "sysroot"])?;
    let version = rustc_output(root, &["-vV"])?;
    let host = parse_host_triple(&version)
        .ok_or_else(|| "could not find the host triple in `rustc -vV`".to_string())?;
    let path = Path::new(sysroot.trim())
        .join("lib/rustlib")
        .join(host)
        .join("bin/llvm-profdata");
    if !path.is_file() {
        return Err(format!(
            "llvm-profdata not found at {}; install it with `rustup component add llvm-tools` \
             (rust-toolchain.toml lists the component, so rustup normally installs it with the \
             toolchain)",
            path.display()
        ));
    }
    Ok(path)
}

fn merge_profile(
    llvm_profdata: &Path,
    data_dir: &Path,
    merged: &Path,
    keep_raw: bool,
) -> Result<(), String> {
    let mut raw: Vec<PathBuf> = fs::read_dir(data_dir)
        .map_err(|e| format!("could not read {}: {e}", data_dir.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "profraw"))
        .collect();
    if raw.is_empty() {
        return Err(format!(
            "the training run left no .profraw file in {}; the instrumented binary did not \
             write a profile (the -Cprofile-generate flag may not have taken effect, or the \
             engine did not exit normally)",
            data_dir.display()
        ));
    }
    raw.sort();

    let status = ProcessCommand::new(llvm_profdata)
        .arg("merge")
        .arg("-o")
        .arg(merged)
        .args(&raw)
        .stdin(Stdio::null())
        .status()
        .map_err(|e| format!("failed to spawn {}: {e}", llvm_profdata.display()))?;
    if !status.success() {
        return Err(format!("`llvm-profdata merge` exited with {status}"));
    }
    if !merged.is_file() {
        return Err(format!(
            "`llvm-profdata merge` succeeded but {} does not exist",
            merged.display()
        ));
    }

    if !keep_raw {
        for p in &raw {
            fs::remove_file(p).map_err(|e| format!("could not remove {}: {e}", p.display()))?;
        }
        // Only an empty directory is removed; anything else in there stays.
        let _ = fs::remove_dir(data_dir);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Training run
// ---------------------------------------------------------------------------

/// Options set before `isready`: single thread, the small hash the manual
/// procedure used, no book (a book hit would replace the search) and no PV
/// throttling.
const ENGINE_OPTIONS: &[(&str, &str)] = &[
    ("Threads", "1"),
    ("USI_Hash", "16"),
    ("MultiPV", "1"),
    ("BookFile", "no_book"),
    ("USI_OwnBook", "false"),
    ("PvInterval", "0"),
];

const STDERR_TAIL_LINES: usize = 20;

/// A USI engine process with its stdout read on a background thread, so every
/// wait has a deadline.
struct Engine {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<String>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
}

impl Engine {
    fn spawn(binary: &Path) -> Result<Engine, String> {
        let mut child = ProcessCommand::new(binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to spawn {}: {e}", binary.display()))?;
        let stdin = child.stdin.take();
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "failed to capture engine stdout".to_string())?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| "failed to capture engine stderr".to_string())?;

        let (tx, lines) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });

        // Drained continuously: an engine that fills a full pipe would block.
        let stderr_tail = Arc::new(Mutex::new(VecDeque::new()));
        let tail = Arc::clone(&stderr_tail);
        thread::spawn(move || drain_stderr(stderr, &tail));

        Ok(Engine {
            child,
            stdin,
            lines,
            stderr_tail,
        })
    }

    fn send(&mut self, line: &str) -> Result<(), String> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| "engine stdin is already closed".to_string())?;
        writeln!(stdin, "{line}")
            .and_then(|()| stdin.flush())
            .map_err(|e| self.failure(&format!("could not send `{line}` to the engine ({e})")))
    }

    /// Wait for a line whose first token is `token` and return it.
    ///
    /// An `info string` line that says something `failed` ends the wait at once:
    /// the engine reports a network it could not load that way and then never
    /// answers `isready`, so waiting out the timeout would only delay the
    /// message that explains it.
    fn expect(&mut self, token: &str, timeout: Duration) -> Result<String, String> {
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match self.lines.recv_timeout(remaining) {
                Ok(line) => {
                    if line.split_ascii_whitespace().next() == Some(token) {
                        return Ok(line);
                    }
                    if line.starts_with("info string") && line.contains("failed") {
                        return Err(self.failure(&format!(
                            "while waiting for `{token}` the engine reported: {line}"
                        )));
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    return Err(self.failure(&format!(
                        "no `{token}` from the engine within {}s",
                        timeout.as_secs()
                    )));
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(self.failure(&format!(
                        "the engine closed its output before sending `{token}`"
                    )));
                }
            }
        }
    }

    /// Send `quit`, wait for the engine to exit by itself and require a zero
    /// status: the profile is written by the exit handler, so a killed or
    /// crashed process leaves nothing usable behind.
    fn quit(&mut self, timeout: Duration) -> Result<(), String> {
        self.send("quit")?;
        self.stdin = None;
        let deadline = Instant::now() + timeout;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) if status.success() => return Ok(()),
                Ok(Some(status)) => {
                    return Err(self.failure(&format!("the engine exited with {status}")));
                }
                Ok(None) if Instant::now() >= deadline => {
                    return Err(self.failure(&format!(
                        "the engine did not exit within {}s of `quit`",
                        timeout.as_secs()
                    )));
                }
                Ok(None) => thread::sleep(Duration::from_millis(20)),
                Err(e) => return Err(format!("could not wait on the engine: {e}")),
            }
        }
    }

    /// `message`, followed by the engine's exit status when it has already gone
    /// and the tail of its stderr.
    fn failure(&mut self, message: &str) -> String {
        let mut out = message.to_string();
        if let Ok(Some(status)) = self.child.try_wait() {
            out.push_str(&format!(" (the engine exited with {status})"));
        }
        // The stderr thread may still be reading the last lines of a process
        // that has just died.
        thread::sleep(Duration::from_millis(50));
        if let Ok(tail) = self.stderr_tail.lock()
            && !tail.is_empty()
        {
            out.push_str("\nengine stderr (tail):");
            for l in tail.iter() {
                out.push_str("\n  ");
                out.push_str(l);
            }
        }
        out
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // No-ops once the process has been waited for.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn drain_stderr<R: Read>(stderr: R, tail: &Mutex<VecDeque<String>>) {
    let mut reader = BufReader::new(stderr);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                let line = String::from_utf8_lossy(&buf).trim_end().to_string();
                if let Ok(mut tail) = tail.lock() {
                    if tail.len() == STDERR_TAIL_LINES {
                        tail.pop_front();
                    }
                    tail.push_back(line);
                }
            }
        }
    }
}

fn train(binary: &Path, eval_dir: &Path, positions: &[String], nodes: u64) -> Result<(), String> {
    let eval_dir = utf8(eval_dir)?;
    let mut engine = Engine::spawn(binary)?;

    engine.send("usi")?;
    engine.expect("usiok", Duration::from_secs(60))?;
    for (name, value) in ENGINE_OPTIONS {
        engine.send(&format!("setoption name {name} value {value}"))?;
    }
    engine.send(&format!("setoption name EvalDir value {eval_dir}"))?;
    engine.send("isready")?;
    // The network is read here.
    engine.expect("readyok", Duration::from_secs(300))?;

    // Generous on purpose: this only has to catch a hang. The instrumented
    // binary is slower than the real one, and the node limit sets the scale.
    let search_timeout = Duration::from_secs(60 + nodes / 1000);

    for (i, sfen) in positions.iter().enumerate() {
        engine.send("usinewgame")?;
        engine.send("isready")?;
        engine.expect("readyok", Duration::from_secs(60))?;
        engine.send(&format!("position sfen {sfen}"))?;
        engine.send(&format!("go nodes {nodes}"))?;
        let bestmove = engine
            .expect("bestmove", search_timeout)
            .map_err(|e| format!("position {} ({sfen}): {e}", i + 1))?;
        if bestmove.split_ascii_whitespace().nth(1).is_none() {
            return Err(format!(
                "position {} ({sfen}): malformed reply `{bestmove}`",
                i + 1
            ));
        }
        let done = i + 1;
        if done % 32 == 0 || done == positions.len() {
            println!("  searched {done}/{} positions", positions.len());
        }
    }

    engine.quit(Duration::from_secs(60))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SFEN_A: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
    const SFEN_B: &str =
        "ln2gk1nl/1r5b1/p1ppsp1+Rp/5sp2/1p2P4/2PP5/PPS2PP1P/2G1G4/LN2K1SNL w GPb2p 32";

    #[test]
    fn rustflag_goes_into_config_when_no_rustflags_env_is_set() {
        let plan = plan_rustflag(None, None, "-Cprofile-generate=/t/pgo/data").unwrap();
        assert_eq!(
            plan,
            RustflagPlan::Config("build.rustflags=[\"-Cprofile-generate=/t/pgo/data\"]".into())
        );
    }

    #[test]
    fn rustflag_is_appended_to_rustflags_when_it_is_set() {
        let plan =
            plan_rustflag(None, Some("-C target-cpu=x86-64-v3"), "-Cprofile-use=/m").unwrap();
        assert_eq!(
            plan,
            RustflagPlan::PlainEnv("-C target-cpu=x86-64-v3 -Cprofile-use=/m".into())
        );
    }

    #[test]
    fn an_empty_rustflags_still_counts_as_set() {
        // Cargo drops `build.rustflags` for an empty `RUSTFLAGS` too, so the
        // value must not go through `--config` there.
        let plan = plan_rustflag(None, Some(""), "-Cprofile-use=/m").unwrap();
        assert_eq!(plan, RustflagPlan::PlainEnv("-Cprofile-use=/m".into()));
    }

    #[test]
    fn encoded_rustflags_outrank_plain_rustflags() {
        let plan = plan_rustflag(
            Some("-C\x1ftarget-cpu=native"),
            Some("-C target-cpu=x86-64-v3"),
            "-Cprofile-use=/m",
        )
        .unwrap();
        assert_eq!(
            plan,
            RustflagPlan::EncodedEnv("-C\x1ftarget-cpu=native\x1f-Cprofile-use=/m".into())
        );
    }

    #[test]
    fn rustflag_with_whitespace_cannot_join_rustflags() {
        let err =
            plan_rustflag(None, Some("-C target-cpu=native"), "-Cprofile-use=/a b/m").unwrap_err();
        assert!(err.contains("whitespace"), "got: {err}");
        // The config route has no such limit.
        assert!(matches!(
            plan_rustflag(None, None, "-Cprofile-use=/a b/m").unwrap(),
            RustflagPlan::Config(_)
        ));
    }

    #[test]
    fn toml_strings_escape_quotes_and_backslashes() {
        assert_eq!(toml_basic_string(r#"a"b\c"#), r#""a\"b\\c""#);
        assert_eq!(toml_basic_string("a\tb"), "\"a\\tb\"");
    }

    #[test]
    fn host_triple_is_read_from_rustc_version_output() {
        let v = "rustc 1.98.1 (abc 2026-01-01)\nbinary: rustc\nhost: x86_64-unknown-linux-gnu\nrelease: 1.98.1\n";
        assert_eq!(parse_host_triple(v), Some("x86_64-unknown-linux-gnu"));
        assert_eq!(parse_host_triple("rustc 1.98.1\n"), None);
    }

    #[test]
    fn positions_accept_an_optional_sfen_prefix_and_skip_comments() {
        let text = format!("# header\n\n{SFEN_A}\nsfen {SFEN_B}\n  \n");
        assert_eq!(
            parse_positions(&text).unwrap(),
            vec![SFEN_A.to_string(), SFEN_B.to_string()]
        );
    }

    #[test]
    fn positions_report_the_line_of_a_malformed_sfen() {
        let text = format!("{SFEN_A}\nnot a position\n");
        let err = parse_positions(&text).unwrap_err();
        assert!(err.contains("line 2"), "got: {err}");
    }

    #[test]
    fn positions_reject_an_empty_set() {
        assert!(parse_positions("# nothing\n\n").is_err());
    }

    #[test]
    fn shipped_positions_all_parse() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("pgo-positions.sfen");
        let text = fs::read_to_string(&path).unwrap();
        assert_eq!(parse_positions(&text).unwrap().len(), 256);
    }
}
