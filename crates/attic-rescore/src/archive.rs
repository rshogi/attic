//! Raw, pre-selection PV archives and their pack commit journal.
//!
//! A worker finishes and syncs one zstd frame per chunk before returning its
//! receipt. The pack writer commits only receipts whose raw bytes are durable.
//! Each restart creates new attempt files; old frames, including incomplete
//! frames from interrupted searches, are never overwritten or removed.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use attic_search::{PvBound, PvInfo, RootKind, RootOutcome};
use attic_state::{Move, format_usi_move as format_normal_move};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use super::{Format, Job, hex, record_len};

fn format_usi_move(mv: Move) -> String {
    if mv == Move::resign() {
        "resign".to_string()
    } else if mv == Move::win() {
        "win".to_string()
    } else if mv == Move::none() {
        "none".to_string()
    } else {
        format_normal_move(mv)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(super) struct Identity {
    version: u32,
    #[serde(default)]
    flip_turn: bool,
    /// `rescore --qsearch`: the positions searched, and written, are the
    /// leaves of the input's. Absent from checkpoints written before the flag
    /// existed, which were all `false`.
    #[serde(default)]
    qsearch: bool,
    /// What the part is written as, by [`Format::name`]. Absent from checkpoints written before
    /// there was a second layout, which were all `.pack`s; a part is resumed only as what it
    /// started as.
    #[serde(default = "legacy_format")]
    format: String,
    source: String,
    source_size: u64,
    source_range_sha256: String,
    network_sha256: String,
    program_sha256: String,
    start: u64,
    end: u64,
    depth: i32,
    multipv: usize,
    jobs: usize,
    hash_mb: usize,
    fv_scale: i32,
    quick_draw: bool,
    entering_king: String,
}

/// What a checkpoint without a `format` key was written as.
fn legacy_format() -> String {
    Format::Pack.name().to_string()
}

impl Identity {
    pub(super) fn new(job: &Job<'_>, end: u64) -> Result<Self, String> {
        let mut input = File::open(job.input).map_err(|e| e.to_string())?;
        let source_size = input.metadata().map_err(|e| e.to_string())?.len();
        input
            .seek(SeekFrom::Start(job.start * job.input_width() as u64))
            .map_err(|e| e.to_string())?;
        let source_range_sha256 =
            hash_reader(input.take((end - job.start) * job.input_width() as u64))?;
        let network_sha256 = hash_reader(File::open(job.network_path).map_err(|e| e.to_string())?)?;
        let executable = std::env::current_exe().map_err(|e| e.to_string())?;
        let program_sha256 = hash_reader(File::open(executable).map_err(|e| e.to_string())?)?;
        Ok(Self {
            version: 1,
            flip_turn: job.flip_turn,
            qsearch: job.qsearch,
            format: job.format.name().to_string(),
            source: job
                .input
                .file_name()
                .ok_or("source has no file name")?
                .to_string_lossy()
                .into_owned(),
            source_size,
            source_range_sha256,
            network_sha256,
            program_sha256,
            start: job.start,
            end,
            depth: job.depth,
            multipv: job.multipv,
            jobs: job.jobs,
            hash_mb: job.hash_mb,
            fv_scale: attic_search::fv_scale(),
            quick_draw: super::quick_draw(),
            entering_king: "CSARule27".to_string(),
        })
    }
}

fn hash_reader(mut reader: impl Read) -> Result<String, String> {
    let mut hash = Sha256::new();
    let mut bytes = [0; 64 * 1024];
    loop {
        let n = reader.read(&mut bytes).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&bytes[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Whether `bytes` are what a chunk of `records` records of this part's format looks like.
fn chunk_holds(format: Format<'_>, identity: &Identity, bytes: &[u8], records: u64) -> bool {
    match format {
        Format::Pack => {
            bytes.len() as u64
                == records * (record_len(identity.multipv) + usize::from(identity.flip_turn)) as u64
        }
        Format::Other(format) => format.count_records(bytes) == Some(records),
    }
}

pub(super) fn attempt() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{}-{nanos}", std::process::id())
}

struct HashFile {
    file: File,
    hash: Sha256,
}
impl Write for HashFile {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let n = self.file.write(bytes)?;
        self.hash.update(&bytes[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

type Encoder = zstd::stream::write::Encoder<'static, HashFile>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Receipt {
    file: String,
    offset: u64,
    length: u64,
    sha256: String,
    pub(super) start: u64,
    pub(super) records: u64,
    pv_events: u64,
}

pub(super) struct Recorder {
    file: Option<File>,
    encoder: Option<Encoder>,
    line: Vec<u8>,
    name: String,
    identity: Identity,
    worker: usize,
    attempt: String,
    offset: u64,
    start: u64,
    records: u64,
    completed: u64,
    index: Option<u64>,
    sequence: u64,
    snapshot: u64,
    pv_events: u64,
}

impl Recorder {
    pub(super) fn new(
        dir: &Path,
        identity: &Identity,
        attempt: &str,
        worker: usize,
    ) -> Result<Self, String> {
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let name = format!("attempt-{attempt}-worker-{worker:04}.jsonl.zst");
        let file = File::options()
            .write(true)
            .create_new(true)
            .open(dir.join(&name))
            .map_err(|e| e.to_string())?;
        File::open(dir)
            .and_then(|dir| dir.sync_all())
            .map_err(|e| e.to_string())?;
        Ok(Self {
            file: Some(file),
            encoder: None,
            line: Vec::with_capacity(1024),
            name,
            identity: identity.clone(),
            worker,
            attempt: attempt.to_string(),
            offset: 0,
            start: 0,
            records: 0,
            completed: 0,
            index: None,
            sequence: 0,
            snapshot: 0,
            pv_events: 0,
        })
    }

    fn event(&mut self, event: serde_json::Value) -> Result<(), String> {
        let encoder = self
            .encoder
            .as_mut()
            .ok_or("raw archive has no active frame")?;
        // Serialize one event before compression: streaming individual JSON
        // punctuation bytes into zstd makes a C call for every tiny write.
        self.line.clear();
        serde_json::to_writer(&mut self.line, &event).map_err(|e| format!("raw archive: {e}"))?;
        self.line.push(b'\n');
        encoder
            .write_all(&self.line)
            .map_err(|e| format!("raw archive: {e}"))
    }

    pub(super) fn begin_chunk(&mut self, start: u64, records: usize) -> Result<(), String> {
        if self.encoder.is_some() {
            return Err("raw chunk already active".into());
        }
        let file = self.file.take().ok_or("raw archive file unavailable")?;
        self.offset = file.metadata().map_err(|e| e.to_string())?.len();
        self.encoder = Some(
            Encoder::new(
                HashFile {
                    file,
                    hash: Sha256::new(),
                },
                3,
            )
            .map_err(|e| e.to_string())?,
        );
        self.start = start;
        self.records = records as u64;
        self.completed = 0;
        self.index = None;
        self.pv_events = 0;
        self.event(
            json!({"event":"chunk_begin", "attempt":self.attempt, "worker":self.worker,
            "start":start, "records":records, "settings":self.identity}),
        )
    }

    pub(super) fn begin_record(
        &mut self,
        index: u64,
        input: &[u8],
        sfen: &str,
    ) -> Result<(), String> {
        if self.index.is_some() || index != self.start + self.completed {
            return Err("raw record order mismatch".into());
        }
        self.index = Some(index);
        self.sequence = 0;
        self.snapshot = 0;
        self.event(json!({"event":"record_begin", "record_index":index, "input_psv_hex":hex(input), "sfen":sfen}))
    }

    /// `--qsearch`: the variation walked from the input to the position this
    /// record's searches ran at, which `record_begin` names only by its sfen.
    pub(super) fn walk(&mut self, index: u64, pv: &[Move]) -> Result<(), String> {
        if self.index != Some(index) {
            return Err("qsearch walk without its record_begin".into());
        }
        let usi: Vec<String> = pv.iter().map(|&mv| format_usi_move(mv)).collect();
        let moves: Vec<u16> = pv.iter().map(|mv| mv.move16()).collect();
        self.event(json!({"event":"qsearch_walk", "record_index":index, "pv":usi, "move16":moves}))
    }

    pub(super) fn pv(&mut self, info: &PvInfo) -> Result<(), String> {
        let index = self.index.ok_or("PV output without record_begin")?;
        if info.multipv == 1 {
            self.snapshot += 1;
        }
        let bound = match info.bound {
            PvBound::Exact => "exact",
            PvBound::Lower => "lower",
            PvBound::Upper => "upper",
        };
        let usi: Vec<String> = info.pv.iter().map(|&mv| format_usi_move(mv)).collect();
        let moves: Vec<u16> = info.pv.iter().map(|mv| mv.move16()).collect();
        self.event(
            json!({"event":"pv", "record_index":index, "sequence":self.sequence,
            "snapshot":self.snapshot, "depth":info.depth, "sel_depth":info.sel_depth,
            "multipv":info.multipv, "score":info.score, "bound":bound, "nodes":info.nodes,
            "pv":usi, "move16":moves}),
        )?;
        self.sequence += 1;
        self.pv_events += 1;
        Ok(())
    }

    pub(super) fn skip_record(&mut self, index: u64, packed: &[u8]) -> Result<(), String> {
        if self.index != Some(index) {
            return Err("skip order mismatch".into());
        }
        self.event(json!({"event":"record_end", "record_index":index, "skip":"original_in_check", "pack_hex":hex(packed)}))?;
        self.index = None;
        self.completed += 1;
        Ok(())
    }

    pub(super) fn end_record(
        &mut self,
        index: u64,
        outcome: &RootOutcome,
        packed: &[u8],
    ) -> Result<(), String> {
        if self.index != Some(index) {
            return Err("raw record_end mismatch".into());
        }
        let kind = match outcome.kind {
            RootKind::Normal => "normal",
            RootKind::Resign => "resign",
            RootKind::DeclarationWin => "declaration_win",
        };
        let pv: Vec<String> = outcome.pv.iter().map(|&mv| format_usi_move(mv)).collect();
        let moves: Vec<u16> = outcome.pv.iter().map(|mv| mv.move16()).collect();
        self.event(json!({"event":"record_end", "record_index":index, "pv_events":self.sequence,
            "selected_snapshot":self.snapshot, "root":{"kind":kind, "score":outcome.score,
            "stored_score":super::stored_cp(outcome.score), "best_move":format_usi_move(outcome.best_move),
            "best_move16":outcome.best_move.move16(), "pv":pv, "move16":moves,
            "depth":outcome.depth, "sel_depth":outcome.sel_depth, "nodes":outcome.nodes},
            "pack_hex":hex(packed)}))?;
        self.index = None;
        self.completed += 1;
        Ok(())
    }

    pub(super) fn finish_chunk(&mut self) -> Result<Receipt, String> {
        if self.index.is_some() || self.completed != self.records {
            return Err("raw chunk is incomplete".into());
        }
        self.event(json!({"event":"chunk_end", "start":self.start, "records":self.records, "pv_events":self.pv_events}))?;
        let encoder = self.encoder.take().ok_or("raw frame missing")?;
        let mut output = encoder
            .finish()
            .map_err(|e| format!("finish raw frame: {e}"))?;
        output
            .file
            .flush()
            .and_then(|()| output.file.sync_all())
            .map_err(|e| format!("sync raw frame: {e}"))?;
        let length = output.file.metadata().map_err(|e| e.to_string())?.len() - self.offset;
        let receipt = Receipt {
            file: self.name.clone(),
            offset: self.offset,
            length,
            sha256: format!("{:x}", output.hash.finalize()),
            start: self.start,
            records: self.records,
            pv_events: self.pv_events,
        };
        self.file = Some(output.file);
        Ok(receipt)
    }
}

#[derive(Serialize, Deserialize)]
struct Commit {
    start: u64,
    records: u64,
    pack_offset: u64,
    pack_length: u64,
    pack_sha256: String,
    raw: Receipt,
}

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    identity: Identity,
    next_record: u64,
    pack_bytes: u64,
    journal_bytes: u64,
}

pub(super) struct Ledger<'a> {
    format: Format<'a>,
    _lock: File,
    final_pack: PathBuf,
    partial: PathBuf,
    checkpoint: PathBuf,
    pub(super) raw_dir: PathBuf,
    state: Checkpoint,
    pack: File,
    journal: File,
}

fn sidecar(pack: &Path, suffix: &str) -> PathBuf {
    let mut name = pack.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn atomic_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let tmp = sidecar(path, &format!(".{}.tmp", attempt()));
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .map_err(|e| e.to_string())?;
    serde_json::to_writer(&mut file, value).map_err(|e| e.to_string())?;
    file.write_all(b"\n")
        .and_then(|()| file.sync_all())
        .map_err(|e| e.to_string())?;
    fs::rename(&tmp, path).map_err(|e| e.to_string())?;
    sync_parent(path)
}

fn sync_parent(path: &Path) -> Result<(), String> {
    File::open(path.parent().ok_or("file has no parent")?)
        .and_then(|dir| dir.sync_all())
        .map_err(|e| e.to_string())
}

fn preserve_tail(path: &Path, committed: u64) -> Result<(), String> {
    let length = fs::metadata(path).map_err(|e| e.to_string())?.len();
    if length < committed {
        return Err(format!("{} is shorter than its checkpoint", path.display()));
    }
    if length == committed {
        return Ok(());
    }
    let preserved = sidecar(path, &format!(".uncommitted-{}", attempt()));
    let mut input = File::open(path).map_err(|e| e.to_string())?;
    input
        .seek(SeekFrom::Start(committed))
        .map_err(|e| e.to_string())?;
    let mut output = File::options()
        .write(true)
        .create_new(true)
        .open(&preserved)
        .map_err(|e| e.to_string())?;
    std::io::copy(&mut input, &mut output).map_err(|e| e.to_string())?;
    output.sync_all().map_err(|e| e.to_string())?;
    sync_parent(&preserved)?;
    let file = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    file.set_len(committed)
        .and_then(|()| file.sync_all())
        .map_err(|e| e.to_string())?;
    eprintln!(
        "rescore: retained uncommitted tail as {}",
        preserved.display()
    );
    Ok(())
}

fn verify_raw(dir: &Path, receipt: &Receipt) -> Result<(), String> {
    let name = Path::new(&receipt.file);
    if name.components().count() != 1 || name.file_name().is_none() {
        return Err("unsafe raw archive name".into());
    }
    let mut file = File::open(dir.join(name)).map_err(|e| format!("raw archive missing: {e}"))?;
    if file.metadata().map_err(|e| e.to_string())?.len()
        < receipt
            .offset
            .checked_add(receipt.length)
            .ok_or("raw offset overflow")?
    {
        return Err("raw archive is shorter than its receipt".into());
    }
    file.seek(SeekFrom::Start(receipt.offset))
        .map_err(|e| e.to_string())?;
    if hash_reader(file.take(receipt.length))? != receipt.sha256 {
        return Err("raw archive checksum mismatch".into());
    }
    Ok(())
}

impl<'a> Ledger<'a> {
    pub(super) fn open(
        final_pack: &Path,
        identity: Identity,
        format: Format<'a>,
    ) -> Result<Self, String> {
        debug_assert_eq!(
            identity.format,
            format.name(),
            "the identity of a part names the format it is written as"
        );
        let header = format.header();
        let lock_path = sidecar(final_pack, ".lock");
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|e| e.to_string())?;
        lock.try_lock()
            .map_err(|e| format!("rescore part is locked or filesystem locking failed: {e}"))?;
        let partial = sidecar(final_pack, ".partial");
        let checkpoint = sidecar(final_pack, ".checkpoint.json");
        let journal_path = sidecar(final_pack, ".receipts.jsonl");
        let raw_dir = sidecar(final_pack, ".analysis");
        let state = if checkpoint.exists() {
            let state: Checkpoint =
                serde_json::from_reader(File::open(&checkpoint).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            if state.identity != identity {
                return Err("rescore settings/input/network differ from the raw checkpoint".into());
            }
            state
        } else {
            if final_pack.exists() || partial.exists() || journal_path.exists() || raw_dir.exists()
            {
                return Err(
                    "existing rescore output has no raw checkpoint; use a new output directory"
                        .into(),
                );
            }
            fs::create_dir(&raw_dir).map_err(|e| e.to_string())?;
            // A part opens with its header, which the checkpoint counts as written.
            File::options()
                .write(true)
                .create_new(true)
                .open(&partial)
                .and_then(|mut f| f.write_all(&header).and_then(|()| f.sync_all()))
                .map_err(|e| e.to_string())?;
            File::options()
                .write(true)
                .create_new(true)
                .open(&journal_path)
                .and_then(|f| f.sync_all())
                .map_err(|e| e.to_string())?;
            let state = Checkpoint {
                next_record: identity.start,
                pack_bytes: header.len() as u64,
                identity,
                journal_bytes: 0,
            };
            atomic_json(&checkpoint, &state)?;
            state
        };
        if state.next_record < state.identity.start || state.next_record > state.identity.end {
            return Err("raw checkpoint record range is invalid".into());
        }
        let path = if final_pack.exists() {
            final_pack
        } else {
            &partial
        };
        if final_pack.exists() && state.next_record != state.identity.end {
            return Err("final pack has an incomplete checkpoint".into());
        }
        if final_pack.exists() && partial.exists() {
            return Err("both final and partial packs exist".into());
        }
        if final_pack.exists()
            && fs::metadata(path).map_err(|e| e.to_string())?.len() != state.pack_bytes
        {
            return Err("final pack size differs from raw checkpoint".into());
        }
        preserve_tail(&journal_path, state.journal_bytes)?;
        if !final_pack.exists() {
            preserve_tail(&partial, state.pack_bytes)?;
        }
        let mut pack = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        let mut journal = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&journal_path)
            .map_err(|e| e.to_string())?;
        let mut next = state.identity.start;
        let mut pack_bytes = header.len() as u64;
        let mut covered = 0;
        if !header.is_empty() {
            // The part has to open with the header this run would have written it with.
            let mut opening = vec![0; header.len()];
            pack.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
            pack.read_exact(&mut opening)
                .map_err(|e| format!("{} part header: {e}", format.name()))?;
            if opening != header {
                return Err(format!(
                    "{} part does not open with this run's header",
                    format.name()
                ));
            }
        }
        for line in BufReader::new((&mut journal).take(state.journal_bytes)).lines() {
            let line = line.map_err(|e| e.to_string())?;
            let commit: Commit = serde_json::from_str(&line).map_err(|e| e.to_string())?;
            if commit.start != next
                || commit.records == 0
                || commit.raw.start != commit.start
                || commit.raw.records != commit.records
                || commit.pack_offset != pack_bytes
            {
                return Err("raw receipt journal is not a contiguous pack prefix".into());
            }
            verify_raw(&raw_dir, &commit.raw)?;
            pack.seek(SeekFrom::Start(commit.pack_offset))
                .map_err(|e| e.to_string())?;
            let mut chunk = Vec::new();
            (&mut pack)
                .take(commit.pack_length)
                .read_to_end(&mut chunk)
                .map_err(|e| e.to_string())?;
            // The chunk is as long as the journal says and holds the records it says: for a
            // fixed-width format that is arithmetic, for any other it is a walk of the records.
            if chunk.len() as u64 != commit.pack_length
                || !chunk_holds(format, &state.identity, &chunk, commit.records)
            {
                return Err("raw receipt journal is not a contiguous pack prefix".into());
            }
            if hash_bytes(&chunk) != commit.pack_sha256 {
                return Err("pack chunk checksum mismatch".into());
            }
            next += commit.records;
            pack_bytes += commit.pack_length;
            covered += line.len() as u64 + 1;
        }
        if next != state.next_record
            || pack_bytes != state.pack_bytes
            || covered != state.journal_bytes
        {
            return Err("raw checkpoint and receipt coverage disagree".into());
        }
        pack.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
        journal.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
        Ok(Self {
            format,
            _lock: lock,
            final_pack: final_pack.to_owned(),
            partial,
            checkpoint,
            raw_dir,
            state,
            pack,
            journal,
        })
    }

    pub(super) fn next_record(&self) -> u64 {
        self.state.next_record
    }
    pub(super) fn identity(&self) -> &Identity {
        &self.state.identity
    }

    #[cfg(test)]
    pub(super) fn commit(&mut self, bytes: &[u8], receipt: Receipt) -> Result<(), String> {
        self.commit_batch(&[(bytes.to_vec(), receipt)])
    }

    pub(super) fn commit_batch(&mut self, chunks: &[(Vec<u8>, Receipt)]) -> Result<(), String> {
        if chunks.is_empty() {
            return Ok(());
        }
        let mut next = self.state.next_record;
        // Every raw frame must already be durable before any pack byte is written.
        for (bytes, receipt) in chunks {
            if receipt.start != next
                || receipt.records == 0
                || !chunk_holds(self.format, &self.state.identity, bytes, receipt.records)
            {
                return Err("pack chunk does not match its durable raw receipt".into());
            }
            next += receipt.records;
            if next > self.state.identity.end {
                return Err("raw chunk exceeds part range".into());
            }
            verify_raw(&self.raw_dir, receipt)?;
        }
        let mut pack_bytes = self.state.pack_bytes;
        for (bytes, receipt) in chunks {
            self.pack.write_all(bytes).map_err(|e| e.to_string())?;
            let commit = Commit {
                start: receipt.start,
                records: receipt.records,
                pack_offset: pack_bytes,
                pack_length: bytes.len() as u64,
                pack_sha256: hash_bytes(bytes),
                raw: receipt.clone(),
            };
            serde_json::to_writer(&mut self.journal, &commit).map_err(|e| e.to_string())?;
            self.journal.write_all(b"\n").map_err(|e| e.to_string())?;
            pack_bytes += commit.pack_length;
        }
        self.pack
            .sync_all()
            .map_err(|e| format!("sync pack batch: {e}"))?;
        self.journal.sync_all().map_err(|e| e.to_string())?;
        let state = Checkpoint {
            identity: self.state.identity.clone(),
            next_record: next,
            pack_bytes,
            journal_bytes: self.journal.stream_position().map_err(|e| e.to_string())?,
        };
        atomic_json(&self.checkpoint, &state)?;
        self.state = state;
        Ok(())
    }

    pub(super) fn finish(self) -> Result<(), String> {
        if self.state.next_record != self.state.identity.end {
            return Err("raw part is not complete".into());
        }
        self.pack.sync_all().map_err(|e| e.to_string())?;
        self.journal.sync_all().map_err(|e| e.to_string())?;
        if self.final_pack.exists() {
            return Ok(());
        }
        fs::rename(&self.partial, &self.final_pack).map_err(|e| e.to_string())?;
        sync_parent(&self.final_pack)
    }
}

// Reject unrelated/source-turn outputs and incorrect corpus search conditions.
pub(super) fn validate_source(job: &Job<'_>) -> Result<(), String> {
    let path = PathBuf::from(format!("{}.checkpoint.json", job.input.display()));
    let value: serde_json::Value =
        serde_json::from_reader(File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?)
            .map_err(|e| e.to_string())?;
    let id = &value["identity"];
    for (field, expected) in [
        ("depth", 9),
        ("multipv", 5),
        ("hash_mb", 16),
        ("fv_scale", 28),
    ] {
        if id[field].as_i64() != Some(expected) {
            return Err(format!("source metadata mismatch: {field}"));
        }
    }
    if id["quick_draw"] != true || id["entering_king"] != "CSARule27" || id["flip_turn"] == true {
        return Err("source search conditions mismatch".into());
    }
    // A `--qsearch` pack is rows about leaves. The sidecar this source feeds is
    // row for row about the corpus's own positions, so it cannot be built on one.
    if id["qsearch"] == true {
        return Err("source was written with --qsearch: its positions are leaves".into());
    }
    // The sidecar is row for row about a 60-byte `.pack`. A part of any other layout has no such
    // rows, and a checkpoint without the key was written before there was another.
    if let Some(format) = id["format"].as_str()
        && format != Format::Pack.name()
    {
        return Err(format!(
            "source is a {format} part: a flip sidecar is built on a --format pack part"
        ));
    }
    let hash = hash_reader(File::open(job.network_path).map_err(|e| e.to_string())?)?;
    if id["network_sha256"] != hash {
        return Err("source NNUE checksum mismatch".into());
    }
    let rows = fs::metadata(job.input).map_err(|e| e.to_string())?.len() / 60;
    if id["end"]
        .as_u64()
        .and_then(|end| id["start"].as_u64().map(|start| end.saturating_sub(start)))
        != Some(rows)
    {
        return Err("source row count mismatch".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RecordFormat, Searched};
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "raw-rescore-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn identity() -> Identity {
        Identity {
            version: 1,
            flip_turn: false,
            qsearch: false,
            format: "pack".into(),
            source: "x.bin".into(),
            source_size: 80,
            source_range_sha256: "input".into(),
            network_sha256: "network".into(),
            program_sha256: "program".into(),
            start: 0,
            end: 2,
            depth: 9,
            multipv: 5,
            jobs: 1,
            hash_mb: 16,
            fv_scale: 28,
            quick_draw: true,
            entering_king: "CSARule27".into(),
        }
    }
    fn outcome() -> RootOutcome {
        let position = attic_state::parse_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        )
        .unwrap();
        let mv = attic_state::parse_usi_move("7g7f", &position).unwrap();
        RootOutcome {
            best_move: mv,
            score: 10,
            nodes: 100,
            pv: vec![mv],
            depth: 9,
            sel_depth: 12,
            kind: RootKind::Normal,
        }
    }
    fn frame(dir: &Path, start: u64, count: usize, attempt: &str) -> Receipt {
        let mut recorder = Recorder::new(dir, &identity(), attempt, 0).unwrap();
        recorder.begin_chunk(start, count).unwrap();
        for index in start..start + count as u64 {
            recorder.begin_record(index, &[0; 40], "position").unwrap();
            recorder.end_record(index, &outcome(), &[0; 60]).unwrap();
        }
        recorder.finish_chunk().unwrap()
    }
    /// A layout whose records are not all one width, laid out as an `MPK1` stream is: a header,
    /// then twelve bytes with the number of candidates at 10..12, a 96-byte position, and eight
    /// bytes a candidate. The ledger knows nothing of it but what [`RecordFormat`] says.
    struct Wide;
    const WIDE_HEADER: [u8; 16] = *b"WIDE\0\0\0\0\0\0\0\0\x1c\0\0\0";
    impl RecordFormat for Wide {
        fn name(&self) -> &str {
            "wide"
        }
        fn extension(&self) -> &str {
            "wide"
        }
        fn header(&self) -> Vec<u8> {
            WIDE_HEADER.to_vec()
        }
        fn max_record_len(&self, multipv: usize) -> usize {
            12 + 96 + 8 * multipv
        }
        fn count_records(&self, bytes: &[u8]) -> Option<u64> {
            let mut at = 0usize;
            let mut records = 0u64;
            while at < bytes.len() {
                let count = bytes.get(at + 10..at + 12)?;
                at += 12 + 96 + 8 * usize::from(u16::from_le_bytes([count[0], count[1]]));
                if at > bytes.len() {
                    return None;
                }
                records += 1;
            }
            Some(records)
        }
        fn append(&self, _: &Searched<'_>, _: &mut Vec<u8>) -> Result<(), String> {
            unreachable!("the ledger is handed chunks that are already written")
        }
    }
    const WIDE: Format<'static> = Format::Other(&Wide);

    /// The identity of a `wide` part of two records.
    fn wide_identity() -> Identity {
        Identity {
            format: "wide".into(),
            ..identity()
        }
    }

    /// Records as `rescore` writes them in this layout: `n` candidates each, so that they are
    /// not all one width.
    fn wide_chunk(candidates: &[usize]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &n in candidates {
            let mut record = vec![0u8; 12 + 96 + 8 * n];
            record[10..12].copy_from_slice(&(n as u16).to_le_bytes());
            bytes.extend_from_slice(&record);
        }
        bytes
    }

    fn wide_frame(dir: &Path, start: u64, count: usize, attempt: &str) -> Receipt {
        let mut recorder = Recorder::new(dir, &wide_identity(), attempt, 0).unwrap();
        recorder.begin_chunk(start, count).unwrap();
        for index in start..start + count as u64 {
            recorder.begin_record(index, &[0; 40], "position").unwrap();
            recorder.end_record(index, &outcome(), &[0; 60]).unwrap();
        }
        recorder.finish_chunk().unwrap()
    }

    #[test]
    fn a_chunk_of_a_wide_layout_is_counted_by_walking_it() {
        // 5, 0 and 3 candidates: 148, 108 and 132 bytes.
        let chunk = wide_chunk(&[5, 0, 3]);
        assert_eq!(chunk.len(), 148 + 108 + 132);
        let held = |format, identity: &Identity, bytes: &[u8], records| {
            chunk_holds(format, identity, bytes, records)
        };
        assert!(held(WIDE, &wide_identity(), &chunk, 3));
        assert!(!held(WIDE, &wide_identity(), &chunk, 2));
        assert!(held(WIDE, &wide_identity(), &[], 0));
        // Not on a record boundary: one byte short, and the middle of the fixed part.
        assert!(!held(WIDE, &wide_identity(), &chunk[..chunk.len() - 1], 3));
        assert!(!held(WIDE, &wide_identity(), &chunk[..5], 1));
        // The format decides which rule a chunk is held to.
        assert!(
            !held(Format::Pack, &identity(), &chunk, 3),
            "a pack is arithmetic"
        );
        assert!(held(Format::Pack, &identity(), &[0; 120], 2));
    }

    #[test]
    fn a_part_opens_with_its_header_and_the_checkpoint_counts_it() {
        let dir = Scratch::new();
        let path = dir.0.join("x.wide");
        let ledger = Ledger::open(&path, wide_identity(), WIDE).unwrap();
        assert_eq!(ledger.state.pack_bytes, WIDE_HEADER.len() as u64);
        assert_eq!(fs::read(&ledger.partial).unwrap(), WIDE_HEADER);
        // And a pack part has nothing before its first record.
        let pack = dir.0.join("y.pack");
        let ledger = Ledger::open(&pack, identity(), Format::Pack).unwrap();
        assert_eq!(ledger.state.pack_bytes, 0);
        assert_eq!(fs::metadata(&ledger.partial).unwrap().len(), 0);
    }

    #[test]
    fn a_part_resumes_at_a_record_boundary_and_cuts_a_torn_tail() {
        let dir = Scratch::new();
        let path = dir.0.join("x.wide");
        let mut ledger = Ledger::open(&path, wide_identity(), WIDE).unwrap();
        let first = wide_chunk(&[5]);
        let receipt = wide_frame(&ledger.raw_dir, 0, 1, "first");
        ledger.commit(&first, receipt).unwrap();
        let committed = WIDE_HEADER.len() + first.len();
        assert_eq!(ledger.state.pack_bytes, committed as u64);
        // A second chunk written to the part but never committed, cut in the middle of a
        // record: the shape a stop leaves behind.
        let torn = wide_chunk(&[5, 5]);
        ledger.pack.write_all(&torn[..torn.len() - 40]).unwrap();
        ledger.pack.sync_all().unwrap();
        drop(ledger);

        let mut ledger = Ledger::open(&path, wide_identity(), WIDE).unwrap();
        assert_eq!(ledger.next_record(), 1);
        assert_eq!(
            fs::metadata(&ledger.partial).unwrap().len(),
            committed as u64,
            "cut back to the end of the last committed record"
        );
        assert!(fs::read_dir(&dir.0).unwrap().any(|e| {
            e.unwrap()
                .file_name()
                .to_string_lossy()
                .contains("uncommitted")
        }));
        // And it carries on from there to a part that reads as one stream.
        let second = wide_chunk(&[3]);
        let receipt = wide_frame(&ledger.raw_dir, 1, 1, "second");
        ledger.commit(&second, receipt).unwrap();
        ledger.finish().unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.len(), WIDE_HEADER.len() + first.len() + second.len());
        assert_eq!(bytes[..WIDE_HEADER.len()], WIDE_HEADER);
        assert_eq!(Wide.count_records(&bytes[WIDE_HEADER.len()..]), Some(2));
        let reopened = Ledger::open(&path, wide_identity(), WIDE).unwrap();
        assert_eq!(reopened.next_record(), 2);
    }

    #[test]
    fn a_chunk_that_is_not_the_records_its_receipt_names_is_refused() {
        let dir = Scratch::new();
        let path = dir.0.join("x.wide");
        let mut ledger = Ledger::open(&path, wide_identity(), WIDE).unwrap();
        let receipt = wide_frame(&ledger.raw_dir, 0, 2, "first");
        // Two records named, one given.
        assert!(
            ledger
                .commit(&wide_chunk(&[5]), receipt.clone())
                .unwrap_err()
                .contains("does not match")
        );
        // Two given but cut short of the second's end.
        let short = wide_chunk(&[5, 5]);
        assert!(
            ledger
                .commit(&short[..short.len() - 1], receipt)
                .unwrap_err()
                .contains("does not match")
        );
        assert_eq!(
            ledger.state.pack_bytes,
            WIDE_HEADER.len() as u64,
            "nothing was written"
        );
    }

    #[test]
    fn a_part_is_resumed_only_as_the_format_it_started_as() {
        let dir = Scratch::new();
        let path = dir.0.join("x.wide");
        drop(Ledger::open(&path, wide_identity(), WIDE).unwrap());
        assert!(
            Ledger::open(&path, identity(), Format::Pack)
                .err()
                .unwrap()
                .contains("differ")
        );
        // A checkpoint from before the key existed was a .pack.
        let mut old = serde_json::to_value(identity()).unwrap();
        old.as_object_mut().unwrap().remove("format").unwrap();
        let read: Identity = serde_json::from_value(old).unwrap();
        assert_eq!(read.format, "pack");
        assert_eq!(read, identity());
    }

    #[test]
    fn a_part_that_does_not_open_with_this_runs_header_is_refused() {
        let dir = Scratch::new();
        let path = dir.0.join("x.wide");
        let ledger = Ledger::open(&path, wide_identity(), WIDE).unwrap();
        let partial = ledger.partial.clone();
        drop(ledger);
        let mut bytes = fs::read(&partial).unwrap();
        bytes[12] ^= 0xFF; // the scale, which a run states
        fs::write(&partial, bytes).unwrap();
        assert!(
            Ledger::open(&path, wide_identity(), WIDE)
                .err()
                .unwrap()
                .contains("header")
        );
    }

    #[test]
    fn a_flip_sidecar_is_not_built_on_a_part_of_another_layout() {
        let dir = Scratch::new();
        let mut job = flip_source(&dir.0, Some(false));
        job.format = Format::Pack;
        assert!(validate_source(&job).is_ok());
        // The checkpoint of such a part says so, and a sidecar about its rows has no
        // 60-byte rows to be about.
        let path = dir.0.join("part.pack.checkpoint.json");
        let mut value: serde_json::Value =
            serde_json::from_reader(File::open(&path).unwrap()).unwrap();
        value["identity"]["format"] = "wide".into();
        fs::write(&path, value.to_string()).unwrap();
        let error = validate_source(&job).unwrap_err();
        assert!(error.contains("wide"), "{error}");
    }

    /// A checkpoint written before `--qsearch` existed has no such key, and has
    /// to read back as the plain run it was or every old part would refuse to
    /// resume.
    #[test]
    fn a_checkpoint_from_before_the_qsearch_flag_reads_back_as_plain() {
        let mut old = serde_json::to_value(identity()).unwrap();
        old.as_object_mut().unwrap().remove("qsearch").unwrap();
        let read: Identity = serde_json::from_value(old).unwrap();
        assert!(!read.qsearch);
        assert_eq!(read, identity());
        // And a leaf run is not the plain run of the same part: it will not be
        // resumed on top of it.
        let leaf = Identity {
            qsearch: true,
            ..identity()
        };
        assert_ne!(leaf, identity());
        let dir = Scratch::new();
        let path = dir.0.join("x.pack");
        drop(Ledger::open(&path, identity(), Format::Pack).unwrap());
        assert!(
            Ledger::open(&path, leaf, Format::Pack)
                .err()
                .unwrap()
                .contains("differ")
        );
    }

    #[test]
    fn the_walk_is_kept_beside_its_record_in_the_raw_archive() {
        let dir = Scratch::new();
        let mut recorder = Recorder::new(&dir.0, &identity(), "walk", 0).unwrap();
        recorder.begin_chunk(0, 1).unwrap();
        let root = outcome();
        // Not allowed before the record it belongs to has begun.
        assert!(recorder.walk(0, &root.pv).is_err());
        recorder.begin_record(0, &[0; 40], "position").unwrap();
        recorder.walk(0, &root.pv).unwrap();
        assert!(
            recorder.walk(1, &root.pv).is_err(),
            "nor for another record"
        );
        recorder.end_record(0, &root, &[0; 60]).unwrap();
        let receipt = recorder.finish_chunk().unwrap();
        let decoded =
            zstd::stream::decode_all(File::open(dir.0.join(&receipt.file)).unwrap()).unwrap();
        let walks: Vec<serde_json::Value> = decoded
            .split(|&b| b == b'\n')
            .filter(|b| !b.is_empty())
            .map(|b| serde_json::from_slice::<serde_json::Value>(b).unwrap())
            .filter(|event| event["event"] == "qsearch_walk")
            .collect();
        assert_eq!(walks.len(), 1);
        assert_eq!(walks[0]["record_index"], 0);
        assert_eq!(walks[0]["pv"][0], "7g7f");
    }

    /// A source for `--flip-turn` to be built on, as the identity of the run
    /// that wrote it describes it.
    fn flip_source(dir: &Path, qsearch: Option<bool>) -> Job<'static> {
        let network = dir.join("nn.bin");
        fs::write(&network, b"network").unwrap();
        let input = dir.join("part.pack");
        fs::write(&input, [0u8; 120]).unwrap();
        let mut id = serde_json::json!({
            "depth": 9, "multipv": 5, "hash_mb": 16, "fv_scale": 28,
            "quick_draw": true, "entering_king": "CSARule27", "flip_turn": false,
            "network_sha256": hash_reader(File::open(&network).unwrap()).unwrap(),
            "start": 0, "end": 2,
        });
        if let Some(qsearch) = qsearch {
            id["qsearch"] = qsearch.into();
        }
        fs::write(
            dir.join("part.pack.checkpoint.json"),
            serde_json::json!({ "identity": id }).to_string(),
        )
        .unwrap();
        // Leaked: a `Job` borrows its paths and the test is over in a moment.
        let leak = |path: PathBuf| -> &'static Path { Box::leak(path.into_boxed_path()) };
        Job {
            input: leak(input),
            network_path: leak(network),
            out: leak(dir.join("out")),
            start: 0,
            count: None,
            depth: 9,
            multipv: 1,
            flip_turn: true,
            format: Format::Pack,
            qsearch: false,
            jobs: 1,
            hash_mb: 16,
        }
    }

    #[test]
    fn a_flip_sidecar_is_not_built_on_a_pack_of_leaves() {
        let dir = Scratch::new();
        // A pack from before the flag, and one that says it is not a leaf run.
        assert!(validate_source(&flip_source(&dir.0, None)).is_ok());
        assert!(validate_source(&flip_source(&dir.0, Some(false))).is_ok());
        let error = validate_source(&flip_source(&dir.0, Some(true))).unwrap_err();
        assert!(error.contains("--qsearch"), "{error}");
    }

    #[test]
    fn terminal_outcomes_preserve_special_move_names() {
        let dir = Scratch::new();
        let mut recorder = Recorder::new(&dir.0, &identity(), "terminal", 0).unwrap();
        recorder.begin_chunk(0, 2).unwrap();
        for (index, mv, kind) in [
            (0, Move::resign(), RootKind::Resign),
            (1, Move::win(), RootKind::DeclarationWin),
        ] {
            let mut root = outcome();
            root.best_move = mv;
            root.pv = vec![mv];
            root.kind = kind;
            recorder.begin_record(index, &[0; 40], "position").unwrap();
            recorder.end_record(index, &root, &[0; 60]).unwrap();
        }
        let receipt = recorder.finish_chunk().unwrap();
        let decoded =
            zstd::stream::decode_all(File::open(dir.0.join(receipt.file)).unwrap()).unwrap();
        let ends: Vec<serde_json::Value> = decoded
            .split(|&b| b == b'\n')
            .filter(|b| !b.is_empty())
            .map(|b| serde_json::from_slice::<serde_json::Value>(b).unwrap())
            .filter(|event| event["event"] == "record_end")
            .collect();
        assert_eq!(ends[0]["root"]["best_move"], "resign");
        assert_eq!(ends[0]["root"]["pv"][0], "resign");
        assert_eq!(ends[1]["root"]["best_move"], "win");
        assert_eq!(ends[1]["root"]["pv"][0], "win");
    }

    #[test]
    fn all_emitted_pvs_survive_including_superseded_rankings() {
        let dir = Scratch::new();
        let mut recorder = Recorder::new(&dir.0, &identity(), "first", 0).unwrap();
        recorder.begin_chunk(0, 1).unwrap();
        recorder.begin_record(0, &[0; 40], "position").unwrap();
        let root = outcome();
        for (rank, score) in [(1, 10), (2, 0), (1, 20), (2, 5)] {
            recorder
                .pv(&PvInfo {
                    depth: 9,
                    sel_depth: 12,
                    multipv: rank,
                    score,
                    bound: PvBound::Exact,
                    nodes: 100,
                    nps: 0,
                    hashfull: 0,
                    time_ms: 1,
                    pv: root.pv.clone(),
                })
                .unwrap();
        }
        recorder.end_record(0, &root, &[0; 60]).unwrap();
        let receipt = recorder.finish_chunk().unwrap();
        verify_raw(&dir.0, &receipt).unwrap();
        let decoded =
            zstd::stream::decode_all(File::open(dir.0.join(&receipt.file)).unwrap()).unwrap();
        let events: Vec<serde_json::Value> = decoded
            .split(|&b| b == b'\n')
            .filter(|b| !b.is_empty())
            .map(|b| serde_json::from_slice(b).unwrap())
            .collect();
        let pv: Vec<_> = events.iter().filter(|e| e["event"] == "pv").collect();
        assert_eq!(pv.len(), 4);
        assert_eq!(pv[0]["score"], 10);
        assert_eq!(pv[2]["score"], 20);
        assert_eq!(pv[2]["snapshot"], 2);
        assert_eq!(pv[0]["pv"][0], "7g7f");
        assert_eq!(
            events.iter().filter(|e| e["event"] == "record_end").count(),
            1
        );
    }
    #[test]
    fn resume_requires_raw_coverage_and_preserves_uncommitted_tails() {
        let dir = Scratch::new();
        let path = dir.0.join("x.pack");
        let mut ledger = Ledger::open(&path, identity(), Format::Pack).unwrap();
        let receipt = frame(&ledger.raw_dir, 0, 1, "first");
        ledger.commit(&[0; 60], receipt.clone()).unwrap();
        ledger.pack.write_all(&[7; 13]).unwrap();
        ledger.pack.sync_all().unwrap();
        ledger.journal.write_all(b"uncommitted").unwrap();
        ledger.journal.sync_all().unwrap();
        drop(ledger);
        let mut ledger = Ledger::open(&path, identity(), Format::Pack).unwrap();
        assert_eq!(ledger.next_record(), 1);
        assert_eq!(fs::metadata(&ledger.partial).unwrap().len(), 60);
        assert!(fs::read_dir(&dir.0).unwrap().any(|e| {
            e.unwrap()
                .file_name()
                .to_string_lossy()
                .contains("uncommitted")
        }));
        let next = frame(&ledger.raw_dir, 1, 1, "second");
        ledger.commit(&[0; 60], next).unwrap();
        ledger.finish().unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), 120);
        let ledger = Ledger::open(&path, identity(), Format::Pack).unwrap();
        assert_eq!(ledger.next_record(), 2);
        drop(ledger);
        fs::remove_file(sidecar(&path, ".analysis").join(receipt.file)).unwrap();
        assert!(
            Ledger::open(&path, identity(), Format::Pack)
                .err()
                .unwrap()
                .contains("raw archive missing")
        );
    }
    #[test]
    fn a_batched_checkpoint_covers_every_raw_receipt() {
        let dir = Scratch::new();
        let path = dir.0.join("batch.pack");
        let mut ledger = Ledger::open(&path, identity(), Format::Pack).unwrap();
        let first = frame(&ledger.raw_dir, 0, 1, "first");
        let second = frame(&ledger.raw_dir, 1, 1, "second");
        ledger
            .commit_batch(&[(vec![0; 60], first), (vec![1; 60], second)])
            .unwrap();
        ledger.finish().unwrap();
        let reopened = Ledger::open(&path, identity(), Format::Pack).unwrap();
        assert_eq!(reopened.next_record(), 2);
        assert_eq!(
            fs::read(&path).unwrap(),
            [vec![0; 60], vec![1; 60]].concat()
        );
    }

    #[test]
    fn a_bad_batch_receipt_prevents_all_pack_writes() {
        let dir = Scratch::new();
        let path = dir.0.join("batch.pack");
        let mut ledger = Ledger::open(&path, identity(), Format::Pack).unwrap();
        let first = frame(&ledger.raw_dir, 0, 1, "first");
        let second = frame(&ledger.raw_dir, 1, 1, "second");
        fs::remove_file(ledger.raw_dir.join(&second.file)).unwrap();
        assert!(
            ledger
                .commit_batch(&[(vec![0; 60], first), (vec![1; 60], second)])
                .is_err()
        );
        assert_eq!(fs::metadata(&ledger.partial).unwrap().len(), 0);
        assert_eq!(ledger.next_record(), 0);
    }

    #[test]
    fn raw_failure_cannot_commit_pack_and_legacy_partials_are_refused() {
        let dir = Scratch::new();
        let path = dir.0.join("x.pack");
        let mut ledger = Ledger::open(&path, identity(), Format::Pack).unwrap();
        let receipt = frame(&ledger.raw_dir, 0, 1, "first");
        fs::remove_file(ledger.raw_dir.join(&receipt.file)).unwrap();
        assert!(ledger.commit(&[0; 60], receipt).is_err());
        assert_eq!(fs::metadata(&ledger.partial).unwrap().len(), 0);
        let old = dir.0.join("legacy.pack");
        fs::write(sidecar(&old, ".partial"), [0; 60]).unwrap();
        assert!(
            Ledger::open(&old, identity(), Format::Pack)
                .err()
                .unwrap()
                .contains("no raw checkpoint")
        );
    }
    #[test]
    fn raw_only_crash_and_changed_settings_do_not_skip_records() {
        let dir = Scratch::new();
        let path = dir.0.join("x.pack");
        let ledger = Ledger::open(&path, identity(), Format::Pack).unwrap();
        let receipt = frame(&ledger.raw_dir, 0, 1, "before-crash");
        let old_archive = ledger.raw_dir.join(&receipt.file);
        let old_hash = hash_reader(File::open(&old_archive).unwrap()).unwrap();
        drop(ledger);
        let ledger = Ledger::open(&path, identity(), Format::Pack).unwrap();
        assert_eq!(ledger.next_record(), 0);
        assert_eq!(
            hash_reader(File::open(&old_archive).unwrap()).unwrap(),
            old_hash
        );
        drop(ledger);
        let mut changed = identity();
        changed.depth = 10;
        assert!(
            Ledger::open(&path, changed, Format::Pack)
                .err()
                .unwrap()
                .contains("differ")
        );
    }

    #[test]
    fn damaged_raw_frames_and_concurrent_writers_are_refused() {
        let dir = Scratch::new();
        let path = dir.0.join("x.pack");
        let mut ledger = Ledger::open(&path, identity(), Format::Pack).unwrap();
        assert!(
            Ledger::open(&path, identity(), Format::Pack)
                .err()
                .unwrap()
                .contains("locked")
        );
        let receipt = frame(&ledger.raw_dir, 0, 1, "first");
        ledger.commit(&[0; 60], receipt.clone()).unwrap();
        let raw_path = ledger.raw_dir.join(&receipt.file);
        drop(ledger);
        let mut damaged = OpenOptions::new().write(true).open(raw_path).unwrap();
        damaged.write_all(&[0; 4]).unwrap();
        damaged.sync_all().unwrap();
        assert!(
            Ledger::open(&path, identity(), Format::Pack)
                .err()
                .unwrap()
                .contains("checksum")
        );
    }

    #[test]
    fn multiple_chunk_frames_in_one_attempt_remain_independently_readable() {
        let dir = Scratch::new();
        let mut recorder = Recorder::new(&dir.0, &identity(), "first", 0).unwrap();
        let mut receipts = Vec::new();
        for index in 0..2 {
            recorder.begin_chunk(index, 1).unwrap();
            recorder.begin_record(index, &[0; 40], "position").unwrap();
            recorder.end_record(index, &outcome(), &[0; 60]).unwrap();
            receipts.push(recorder.finish_chunk().unwrap());
        }
        for receipt in &receipts {
            verify_raw(&dir.0, receipt).unwrap();
        }
        let all =
            zstd::stream::decode_all(File::open(dir.0.join(&receipts[0].file)).unwrap()).unwrap();
        assert_eq!(
            String::from_utf8(all)
                .unwrap()
                .matches("record_end")
                .count(),
            2
        );
    }
}
