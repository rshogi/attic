//! Search parity against YaneuraOu 9.80 (KP256), through the USI front end.
//!
//! For every position in the fixture the engine is driven the way a GUI does it
//! (`usinewgame`, `isready`, wait for `readyok`, `position`, `go nodes 100000`,
//! wait for `bestmove`) and its output is compared with what YaneuraOu 9.80
//! printed for the same position and the same options:
//!
//! * every `info depth` line, with `time` / `nps` / `hashfull` removed — so the
//!   depth, seldepth, score, bound, node count and PV of every iteration,
//! * the `bestmove` line, ponder move included.
//!
//! One differing node anywhere fails the test. `usinewgame` clears the
//! transposition table before every position, so each search is independent of
//! the one before it.
//!
//! # Configuration
//!
//! The parity holds for one build configuration only, so the file compiles only
//! there and is otherwise absent (`running 0 tests`):
//!
//! ```text
//! cargo test --release -p attic-protocol --test yaneuraou_parity \
//!     --no-default-features --features nnue_kp_256x2_32_32,ref_check_time
//! ```
//!
//! * `--no-default-features` leaves out `quick-draw`: the reference here is the
//!   `make normal` build, which does not define `ENABLE_QUICK_DRAW`.
//! * `ref_check_time` makes `go nodes N` stop where the reference does; see
//!   `crates/attic-search/Cargo.toml`.
//!
//! # Inputs
//!
//! * The network: `ATTIC_KP256_NN`, else `eval/nn.bin` at the workspace root. It
//!   must be the file the fixture was captured with (`eval_sha256` in
//!   `manifest.json`). When it is absent the test prints a notice and passes,
//!   like the other real-network tests, unless `ATTIC_REQUIRE_PARITY=1`, which
//!   turns the absence into a failure.
//! * The fixture: `tests/fixtures/search-yaneuraou-kp256-nodes100000/` (a subset
//!   of the 16,384 captured positions), or the directory in
//!   `ATTIC_YO_FIXTURE_DIR` holding a `fixture.jsonl` and a `manifest.json` —
//!   which is how the full set is run; see that directory's `README.md`.

#![cfg(all(
    feature = "nnue_kp_256x2_32_32",
    feature = "ref_check_time",
    not(feature = "quick-draw")
))]

mod common;

use std::path::{Path, PathBuf};

use common::{StreamHarness, bestmove_lines};
use serde::Deserialize;

/// A generous bound for one `go nodes 100000` search plus the table clear.
const SEARCH_TIMEOUT_MS: u64 = 120_000;
/// How many mismatching positions to print before summarising.
const SHOWN_MISMATCHES: usize = 5;

#[derive(Debug, Deserialize)]
struct Record {
    idx: u64,
    sfen: String,
    info: Vec<String>,
    bestmove: String,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    nodes: u64,
    eval_sha256: String,
    /// The USI options the reference ran with, name to value.
    options: serde_json::Map<String, serde_json::Value>,
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_dir() -> PathBuf {
    match std::env::var_os("ATTIC_YO_FIXTURE_DIR") {
        Some(d) => PathBuf::from(d),
        None => workspace_root().join("tests/fixtures/search-yaneuraou-kp256-nodes100000"),
    }
}

fn network_path() -> PathBuf {
    match std::env::var_os("ATTIC_KP256_NN") {
        Some(p) => PathBuf::from(p),
        None => workspace_root().join("eval/nn.bin"),
    }
}

fn load_records(dir: &Path) -> Vec<Record> {
    let path = dir.join("fixture.jsonl");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
    raw.lines()
        .enumerate()
        .map(|(n, l)| {
            serde_json::from_str(l).unwrap_or_else(|e| panic!("{}:{}: {e}", path.display(), n + 1))
        })
        .collect()
}

fn load_manifest(dir: &Path) -> Manifest {
    let path = dir.join("manifest.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read manifest {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse manifest: {e}"))
}

/// Drop the `time`, `nps` and `hashfull` fields of an `info` line: they depend on
/// the machine, not on the search.
fn normalise(line: &str) -> String {
    let mut kept: Vec<&str> = Vec::new();
    let mut tokens = line.split_whitespace().peekable();
    while let Some(tok) = tokens.next() {
        let is_numeric = |v: &&str| v.bytes().all(|b| b.is_ascii_digit());
        if matches!(tok, "time" | "nps" | "hashfull") && tokens.peek().is_some_and(is_numeric) {
            tokens.next();
            continue;
        }
        kept.push(tok);
    }
    kept.join(" ")
}

/// The `info depth` lines among `lines`, normalised.
fn info_lines<'a>(lines: impl Iterator<Item = &'a str>) -> Vec<String> {
    lines
        .filter(|l| l.starts_with("info depth"))
        .map(normalise)
        .collect()
}

/// A description of the first place `got` and `want` differ, or `None`.
fn first_difference(got: &[String], want: &[String]) -> Option<String> {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g != w {
            return Some(format!("info line {i}:\n    engine: {g}\n    want:   {w}"));
        }
    }
    (got.len() != want.len()).then(|| {
        format!(
            "info line count: engine printed {}, want {}",
            got.len(),
            want.len()
        )
    })
}

/// The comparison is the whole point of the test, so check that it can fail: a
/// single node, a different bound, or a missing line must all be seen.
#[test]
fn comparison_detects_a_one_node_difference() {
    let want = vec![
        "info depth 9 seldepth 14 multipv 1 score cp 120 nodes 41233 pv 7g7f 3c3d".to_string(),
        "info depth 10 seldepth 16 multipv 1 score cp 118 nodes 77001 pv 7g7f 3c3d".to_string(),
    ];
    assert_eq!(first_difference(&want, &want), None);

    let mut one_node = want.clone();
    one_node[1] = one_node[1].replace("nodes 77001", "nodes 77002");
    assert!(first_difference(&one_node, &want).is_some());

    let mut bound = want.clone();
    bound[0] = bound[0].replace("score cp 120", "score cp 120 lowerbound");
    assert!(first_difference(&bound, &want).is_some());

    assert!(first_difference(&want[..1], &want).is_some());

    // time / nps / hashfull are the only things allowed to differ.
    assert_eq!(
        normalise(
            "info depth 3 seldepth 4 multipv 1 score cp 5 nodes 99 nps 1500000 hashfull 3 time 61 pv 7g7f"
        ),
        "info depth 3 seldepth 4 multipv 1 score cp 5 nodes 99 pv 7g7f"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn search_matches_yaneuraou_9_80_kp256() {
    let nn = network_path();
    if !nn.exists() {
        let msg = format!(
            "{} is not present (set ATTIC_KP256_NN to the nn.bin the fixture was captured with)",
            nn.display()
        );
        assert!(
            std::env::var_os("ATTIC_REQUIRE_PARITY").is_none(),
            "ATTIC_REQUIRE_PARITY is set but {msg}"
        );
        eprintln!("skipping search_matches_yaneuraou_9_80_kp256: {msg}");
        return;
    }

    let dir = fixture_dir();
    let manifest = load_manifest(&dir);
    let records = load_records(&dir);
    assert!(!records.is_empty(), "fixture {} is empty", dir.display());
    assert!(
        records.iter().all(|r| !r.info.is_empty()),
        "a fixture record has no info lines"
    );

    let evaldir = nn.parent().expect("nn.bin has a parent directory");
    let h = StreamHarness::start();
    h.send("usi");
    for (name, value) in &manifest.options {
        let value = match value {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        h.send(&format!("setoption name {name} value {value}"));
    }
    h.send(&format!(
        "setoption name EvalDir value {}",
        evaldir.display()
    ));
    h.send("isready");
    assert!(
        h.wait_until(60_000, |o| o.contains("readyok")),
        "the network must load and the engine answer readyok:\n{}",
        h.output()
    );

    let mut mismatches: Vec<String> = Vec::new();
    let mut consumed = 0usize; // bytes of output already attributed to earlier positions
    for (n, rec) in records.iter().enumerate() {
        let readyoks = h.output().matches("readyok").count();
        h.send("usinewgame");
        h.send("isready");
        assert!(
            h.wait_until(60_000, |o| o.matches("readyok").count() > readyoks),
            "no readyok after usinewgame (position idx {})",
            rec.idx
        );
        h.send(&format!("position {}", rec.sfen));
        h.send(&format!("go nodes {}", manifest.nodes));
        assert!(
            h.wait_until(SEARCH_TIMEOUT_MS, |o| bestmove_lines(o).len() == n + 1),
            "no bestmove for position idx {}:\n{}",
            rec.idx,
            h.output()
        );

        let out = h.output();
        let mine = &out[consumed..];
        consumed = out.len();

        let got_info = info_lines(mine.lines());
        let want_info: Vec<String> = rec.info.iter().map(|l| normalise(l)).collect();
        let got_best = bestmove_lines(mine).last().map(|s| format!("bestmove {s}"));

        let mut why = first_difference(&got_info, &want_info);
        if why.is_none() && got_best.as_deref() != Some(rec.bestmove.as_str()) {
            why = Some(format!(
                "bestmove: engine {got_best:?}, want {:?}",
                rec.bestmove
            ));
        }
        if let Some(why) = why {
            mismatches.push(format!("idx {} ({}):\n  {why}", rec.idx, rec.sfen));
        }
    }
    h.quit_join();

    if !mismatches.is_empty() {
        let shown = mismatches
            .iter()
            .take(SHOWN_MISMATCHES)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "{} of {} positions differ from YaneuraOu 9.80 (nn.bin must be the file with sha256 {}):\n{shown}",
            mismatches.len(),
            records.len(),
            manifest.eval_sha256
        );
    }
}
