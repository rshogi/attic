//! Put a fixed-depth score on every record of a PSV file.
//!
//! The input is the 40-byte `PackedSfenValue` layout that upstream's learner
//! and every public YaneuraOu-format corpus share. Only the position, the ply
//! and the game result are kept from it; the score and move it carried came
//! from another engine at another depth and are replaced by what `attic`
//! finds at `--depth` with `--multipv` lines, the same search `attic-gensfen`
//! runs when it produces the V4 corpus.
//!
//! The output is `pack` by default: `attic-gensfen`'s layout, the 40-byte
//! prefix followed by `--multipv` `(move16, cp)` pairs, fixed width, which is
//! what `--flip-turn` reads. A host that wants another layout supplies it as a
//! [`RecordFormat`] and the run writes that instead: the position and its lines
//! reach it as a [`Searched`], everything about how a part is checkpointed,
//! resumed and verified stays here. Either way records stay in input order,
//! which keeps record `i` of the output about record `start + i` of the input.
//!
//! With `--qsearch` the search is not run at the record's own position but at
//! the end of the principal variation of a quiescence search from it: the
//! position the record would have been before any exchange it was caught in
//! the middle of is settled. The output then carries *that* position — the leaf
//! — with its ply and game result brought along (see [`leaf_result`]), and the
//! `--multipv` lines are the leaf's. The walk is [`find_leaf`] and [`play_pv`].
//!
//! The walk runs on a `QSearch` of its own, with its own table, and the depth
//! search on another. Sharing one looked free and was not: the depth search
//! leaves its history tables behind, the quiescence search orders its moves by
//! them, and equal-valued variations are told apart by that order — so which
//! leaf a record was walked to depended on which records the same worker had
//! searched before it, and the positions written changed with `--jobs` and
//! from one run to the next. Apart, the walk sees only the walks before it,
//! and the leaves are the same under any `--jobs`. The price is one more
//! `QSearch` per worker, about 51MB and `--hash-mb`.
//!
//! A file of hundreds of millions of positions is days of search, so one run
//! covers `--start`/`--count` records of it and a machine can be handed a
//! range. The output is appended to a `.partial` file as it is produced; a run
//! that finds one picks up after its last whole record instead of starting
//! over.

#[cfg(test)]
use std::fs::OpenOptions;
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use attic_numa::{NumaAutoPolicy, NumaConfig};
use attic_search::{
    EnteringKingConfig, EnteringKingRule, PvInfo, PvOutputConfig, PvSink, QSearch, RootKind,
};
use attic_state::{PACKED_SFEN_LEN, PackedSfen, Position, format_sfen, sfen_pack, sfen_unpack};
use attic_storage::TranspositionTable;

mod archive;
mod leaf;

pub use leaf::{find_leaf, play_pv};

/// One input record: `PackedSfenValue`.
pub const PSV_LEN: usize = 40;

/// Records handed out at a time.
///
/// Small, because the workers take from one shared queue and the writer puts
/// the results back in order: a depth-9 search varies by an order of magnitude
/// from one position to the next, and the chunk is the unit the writer may be
/// kept waiting on.
const CHUNK: usize = 16;

/// How often the progress line is printed.
const REPORT_EVERY: Duration = Duration::from_secs(60);

/// Scores reach the root as `VALUE_MATE - ply` at most; anything wider is a
/// sentinel. The same clamp `attic-gensfen` stores with.
const CP_LIMIT: i32 = 32000;

/// The entering-king rule the V4 corpus was generated under
/// (`attic-gensfen`'s default).
const ENTERING_KING: EnteringKingRule = EnteringKingRule::Point27;

/// Whether a record that packs differently has been shown yet; one is enough
/// to see what differs, and a whole file of them would bury the progress.
static SHOWN_REPACKED: AtomicBool = AtomicBool::new(false);

/// What the search found at one position, as a [`RecordFormat`] is handed it.
pub struct Searched<'a> {
    /// The position the search ran at — for `--qsearch`, the leaf.
    pub position: &'a Position,
    /// The ply it is written at.
    pub ply: u16,
    /// The root score, in centipawns, already clamped to what a corpus stores.
    pub cp: i16,
    /// The root's best move as `move16`, or 0 where the search had none.
    pub best: u16,
    /// The MultiPV lines in rank order, each its first move (`move16`) and
    /// score. Empty for a position the search never entered.
    pub lines: &'a [(u16, i16)],
}

/// A layout for the records of a part, other than `pack`.
///
/// A part is `header()` followed by one record per input record. The run
/// checkpoints a part chunk by chunk and has to be able to tell, from a chunk's
/// bytes alone, how many records it holds; that is `count_records`.
pub trait RecordFormat: Sync {
    /// The name a part's checkpoint states, lower case. A part is resumed only
    /// as the format it started as, so this is compared; it is never `pack`.
    fn name(&self) -> &str;
    /// The extension the part is named with, without the dot.
    fn extension(&self) -> &str;
    /// The bytes a part opens with, which the checkpoint counts as written.
    /// Empty for a layout with no header.
    fn header(&self) -> Vec<u8>;
    /// The size of the longest record at `multipv` lines, which a chunk's
    /// buffer is sized for.
    fn max_record_len(&self, multipv: usize) -> usize;
    /// How many whole records `bytes` hold, or `None` when they do not end on a
    /// record boundary. For a layout whose records are not all one width this is
    /// a walk of them.
    fn count_records(&self, bytes: &[u8]) -> Option<u64>;
    /// Append the record of one searched position to `out`.
    fn append(&self, searched: &Searched<'_>, out: &mut Vec<u8>) -> Result<(), String>;
}

/// What the parts of a run are written as.
#[derive(Clone, Copy)]
pub enum Format<'a> {
    /// `attic-gensfen`'s `pack`: the 40-byte `PackedSfenValue` prefix and one
    /// `(move16, cp)` pair per line, fixed width. The only layout `--flip-turn`
    /// reads and writes.
    Pack,
    /// A layout the host supplies.
    Other(&'a dyn RecordFormat),
}

impl Format<'_> {
    /// The name a part's checkpoint states.
    pub fn name(&self) -> &str {
        match self {
            Format::Pack => "pack",
            Format::Other(format) => format.name(),
        }
    }

    /// The extension a part is named with, without the dot.
    pub fn extension(&self) -> &str {
        match self {
            Format::Pack => "pack",
            Format::Other(format) => format.extension(),
        }
    }

    /// The bytes a part opens with.
    fn header(&self) -> Vec<u8> {
        match self {
            Format::Pack => Vec::new(),
            Format::Other(format) => format.header(),
        }
    }
}

/// What one run is asked to do.
pub struct Job<'a> {
    pub input: &'a Path,
    pub network_path: &'a Path,
    pub out: &'a Path,
    pub start: u64,
    /// `None` runs to the end of the file.
    pub count: Option<u64>,
    pub depth: i32,
    pub multipv: usize,
    pub flip_turn: bool,
    /// What the parts are written as.
    pub format: Format<'a>,
    /// Search the leaf of each position's quiescence search instead of the
    /// position, and write the leaf.
    pub qsearch: bool,
    pub jobs: usize,
    pub hash_mb: usize,
}

impl Job<'_> {
    fn input_width(&self) -> usize {
        if self.flip_turn { 60 } else { PSV_LEN }
    }
    /// A record's size, or for a layout of records of different widths its size
    /// with every line present, which is what a chunk's buffer is sized for.
    fn output_width(&self) -> usize {
        match self.format {
            Format::Pack => record_len(self.multipv) + usize::from(self.flip_turn),
            Format::Other(format) => format.max_record_len(self.multipv),
        }
    }
}

/// Width of one `pack` record.
pub fn record_len(multipv: usize) -> usize {
    PSV_LEN + 4 * multipv
}

/// Whether the search this build runs treats the second occurrence of a
/// position as a draw (`ENABLE_QUICK_DRAW`). Decided when `attic-state` is
/// built, and what a position starts with.
fn quick_draw() -> bool {
    attic_state::RepetitionRule::DEFAULT == attic_state::RepetitionRule::QuickDraw
}

/// What the searches came to.
///
/// `same_move` is the check that the input was read the way it was written:
/// the file's own move is another engine's choice, so it agrees with V4's
/// often but not always, and a rate near zero means the positions were decoded
/// wrongly rather than that the engines disagree. It is taken over `with_move`
/// only, since some PSV files leave the move empty on every record (Hao's
/// `shuffled_*.bin` do), and those have nothing to check against.
#[derive(Clone, Copy, Default)]
pub struct Tally {
    pub records: u64,
    pub declared: u64,
    pub mated: u64,
    /// Records with fewer than `--multipv` lines, i.e. fewer legal moves.
    pub short: u64,
    /// Records whose input carries a move at all.
    pub with_move: u64,
    pub same_move: u64,
    /// Records whose position packs to other bytes than the input's.
    pub repacked: u64,
    pub nodes: u64,
    /// `--qsearch`: records whose quiescence search was run, i.e. all of them.
    pub qs_records: u64,
    /// `--qsearch`: records whose position was replaced by a leaf.
    pub qs_moved: u64,
    /// `--qsearch`: moves walked, over all of them.
    pub qs_plies: u64,
    /// `--qsearch`: the longest variation walked.
    pub qs_deepest: usize,
    min_cp: i16,
    max_cp: i16,
}

impl Tally {
    fn score(&mut self, cp: i16) {
        if self.records == 1 {
            self.min_cp = cp;
            self.max_cp = cp;
        } else {
            self.min_cp = self.min_cp.min(cp);
            self.max_cp = self.max_cp.max(cp);
        }
    }

    fn merge(&mut self, other: Tally) {
        if other.records > 0 {
            if self.records == 0 {
                self.min_cp = other.min_cp;
                self.max_cp = other.max_cp;
            } else {
                self.min_cp = self.min_cp.min(other.min_cp);
                self.max_cp = self.max_cp.max(other.max_cp);
            }
        }
        self.records += other.records;
        self.declared += other.declared;
        self.mated += other.mated;
        self.short += other.short;
        self.with_move += other.with_move;
        self.same_move += other.same_move;
        self.repacked += other.repacked;
        self.nodes += other.nodes;
        self.qs_records += other.qs_records;
        self.qs_moved += other.qs_moved;
        self.qs_plies += other.qs_plies;
        self.qs_deepest = self.qs_deepest.max(other.qs_deepest);
    }

    /// Count one record's quiescence walk of `walked` moves.
    fn walk(&mut self, walked: usize) {
        self.qs_records += 1;
        if walked > 0 {
            self.qs_moved += 1;
            self.qs_plies += walked as u64;
            self.qs_deepest = self.qs_deepest.max(walked);
        }
    }

    pub fn summary(&self) -> String {
        let per = |n: u64| {
            if self.records == 0 {
                0.0
            } else {
                n as f64 / self.records as f64
            }
        };
        let same_move = if self.qs_records > 0 {
            // The file's move belongs to the position the file stored, and the
            // position searched is the leaf; there is nothing to compare.
            "n/a (--qsearch searched the leaf, not the input's position)".to_owned()
        } else if self.with_move == 0 {
            "n/a (the input carries no moves)".to_owned()
        } else {
            format!(
                "{:.1}% of {}",
                100.0 * self.same_move as f64 / self.with_move as f64,
                self.with_move,
            )
        };
        let mut line = format!(
            "records={} declared={} mated={} short={} same_move={} repacked={} nodes/record={:.0} cp={}..{}",
            self.records,
            self.declared,
            self.mated,
            self.short,
            same_move,
            self.repacked,
            per(self.nodes),
            self.min_cp,
            self.max_cp,
        );
        if self.qs_records > 0 {
            // A run where nothing moved did not walk, and a run where
            // everything moved found no position quiet; neither shows in the
            // output file, so the counts are printed with the rest.
            let moved_share = 100.0 * self.qs_moved as f64 / self.qs_records as f64;
            let mean = if self.qs_moved == 0 {
                0.0
            } else {
                self.qs_plies as f64 / self.qs_moved as f64
            };
            line.push_str(&format!(
                " qs_moved={} ({moved_share:.1}%) qs_mean_pv={mean:.2} qs_deepest={}",
                self.qs_moved, self.qs_deepest,
            ));
        }
        line
    }
}

/// Collects only the last complete MultiPV output snapshot.
///
/// `emit_pv` sends each snapshot in rank order, starting with rank 1. A new
/// snapshot replaces the whole previous one, regardless of reported depths,
/// so ranks from different root orderings can never be combined.
#[derive(Default)]
struct LastPvs {
    lines: Vec<PvInfo>,
    raw: Option<archive::Recorder>,
    error: Option<String>,
}

struct Collector(Arc<Mutex<LastPvs>>);

impl PvSink for Collector {
    fn emit(&mut self, info: &PvInfo) {
        let mut slot = self.0.lock().expect("pv collector mutex poisoned");
        // The original output is recorded before any snapshot selection.
        // PvSink cannot return an error; retain it and refuse this entire chunk.
        if slot.error.is_none()
            && let Some(raw) = slot.raw.as_mut()
            && let Err(error) = raw.pv(info)
        {
            slot.error = Some(error);
        }
        if info.multipv == 1 {
            slot.lines.clear();
        }
        slot.lines.push(info.clone());
    }
}

type Work = (u64, Vec<[u8; PSV_LEN]>);
type Done = (u64, Result<(Vec<u8>, Tally, archive::Receipt), String>);

/// Where a range's output goes: named by the input and the range, so the
/// ranges of one file can share a directory.
///
/// `--qsearch` writes the same ranges of the same file, but not the same
/// positions, so its stem carries `-qs` — with no dot, because `rescore-parquet`
/// and `pack` read everything after the last dot of the stem as the record range.
/// The extension says what the part is.
pub fn destination(
    input: &Path,
    out: &Path,
    start: u64,
    end: u64,
    qsearch: bool,
    format: Format<'_>,
) -> PathBuf {
    let stem = input
        .file_stem()
        .map_or_else(|| "records".into(), |s| s.to_string_lossy());
    let tag = if qsearch { "-qs" } else { "" };
    out.join(format!(
        "{stem}{tag}.{start:010}-{end:010}.{}",
        format.extension()
    ))
}

/// The game result byte of a record whose position was walked on `walked` moves.
///
/// A PSV record states the result from the point of view of the side to move
/// at the position it stores, as a signed byte: 1, 0 or -1. Every move of the
/// walk hands the move to the other side, so an odd walk turns the result
/// around and an even one leaves it. The value is flipped with
/// [`i8::wrapping_neg`] rather than negated, so a byte that is not a result at
/// all (-128) comes out the way it went in instead of becoming a result.
pub fn leaf_result(result: u8, walked: usize) -> u8 {
    if walked % 2 == 1 {
        (result as i8).wrapping_neg() as u8
    } else {
        result
    }
}

/// The ply of a leaf `walked` moves on from a record at `ply`, held at the top
/// of the field instead of wrapping to the start — the clamp
/// [`crate::leaf`] applies to a stream's `ply`, and upstream's `qsearch_psv`.
pub fn leaf_ply(ply: u16, walked: usize) -> u16 {
    ply.saturating_add(u16::try_from(walked).unwrap_or(u16::MAX))
}

/// How many whole records a `.partial` already holds, cutting off a record
/// that was being written when the last run stopped.
#[cfg(test)]
fn resume_point(partial: &Path, record_len: usize) -> Result<u64, String> {
    let len = match fs::metadata(partial) {
        Ok(meta) => meta.len(),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(err) => return Err(format!("{}: {err}", partial.display())),
    };
    let whole = len / record_len as u64;
    if whole * record_len as u64 != len {
        OpenOptions::new()
            .write(true)
            .open(partial)
            .and_then(|f| f.set_len(whole * record_len as u64))
            .map_err(|e| format!("{}: {e}", partial.display()))?;
    }
    Ok(whole)
}

pub fn run(job: &Job<'_>, base: &attic_search::Search) -> Result<Tally, String> {
    let size = fs::metadata(job.input)
        .map_err(|e| format!("{}: {e}", job.input.display()))?
        .len();
    if job.qsearch && job.flip_turn {
        return Err("--qsearch and --flip-turn cannot be combined".into());
    }
    if job.flip_turn && !matches!(job.format, Format::Pack) {
        return Err("--flip-turn writes a .pack sidecar; it needs --format pack".into());
    }
    if job.flip_turn {
        if job.depth != 9
            || job.multipv != 1
            || job.hash_mb != 16
            || attic_search::fv_scale() != 28
            || !quick_draw()
        {
            return Err("flip-turn requires depth9/multipv1/hash16/fv-scale28/quick-draw".into());
        }
        archive::validate_source(job)?;
    }
    if size % job.input_width() as u64 != 0 {
        return Err(format!(
            "{}: {size} bytes is not a whole number of {PSV_LEN}-byte records",
            job.input.display()
        ));
    }
    let total = size / job.input_width() as u64;
    if job.start >= total {
        return Err(format!(
            "{}: --start {} is past its {total} records",
            job.input.display(),
            job.start
        ));
    }
    let end = match job.count {
        Some(count) => job.start.saturating_add(count).min(total),
        None => total,
    };

    fs::create_dir_all(job.out).map_err(|e| format!("{}: {e}", job.out.display()))?;
    let destination = destination(job.input, job.out, job.start, end, job.qsearch, job.format);
    let identity = archive::Identity::new(job, end)?;
    let mut ledger = archive::Ledger::open(&destination, identity, job.format)?;
    let from = ledger.next_record();
    if from == end {
        ledger.finish()?;
        eprintln!(
            "rescore: {} and all raw receipts are verified complete",
            destination.display()
        );
        return Ok(Tally::default());
    }
    if from > job.start {
        eprintln!(
            "rescore: resuming {} at record {from}, with verified raw coverage",
            destination.display()
        );
    }
    let attempt = archive::attempt();
    let raw_dir = ledger.raw_dir.clone();
    let identity = ledger.identity().clone();

    let numa = NumaConfig::from_system(&NumaAutoPolicy::SystemNuma, true);
    let nodes = numa.num_numa_nodes().max(1).min(job.jobs);
    let per_node = job.jobs / nodes;
    let extra = job.jobs % nodes;
    eprintln!(
        "rescore: {} records {}..{end} depth={} multipv={} jobs={} numa_nodes={nodes} hash_mb={}",
        job.input.display(),
        from,
        job.depth,
        job.multipv,
        job.jobs,
        job.hash_mb,
    );

    // One shared queue rather than one per worker: a round that waited for
    // every worker would run at the pace of the slowest search in it.
    let (work_tx, work_rx) = sync_channel::<Work>(job.jobs * 4);
    let work_rx = Arc::new(Mutex::new(work_rx));
    let (done_tx, done_rx) = sync_channel::<Done>(job.jobs * 4);

    let started = Instant::now();
    let outcome = std::thread::scope(|outer| -> Result<Tally, String> {
        // A bounded result channel alone does not bound the reorder map: a
        // slow early chunk can let later chunks accumulate indefinitely.
        let (permit_tx, permit_rx) = sync_channel::<()>(job.jobs * 4);
        for _ in 0..job.jobs * 4 {
            permit_tx.send(()).expect("initial in-flight permits fit");
        }
        for node in 0..nodes {
            let numa = &numa;
            let here = per_node + usize::from(node < extra);
            let work_rx = Arc::clone(&work_rx);
            let done_tx = done_tx.clone();
            let raw_dir = &raw_dir;
            let identity = &identity;
            let attempt = &attempt;
            let first_worker = node * per_node + node.min(extra);
            outer.spawn(move || {
                // Bind before replicating so the copy lands in this node's
                // memory, as `attic-gensfen` does.
                numa.bind_current_thread_to_numa_node(node);
                let local = base.replicate();
                let local = &local;
                std::thread::scope(|inner| {
                    for worker_index in 0..here {
                        let work_rx = Arc::clone(&work_rx);
                        let done_tx = done_tx.clone();
                        inner.spawn(move || {
                            numa.bind_current_thread_to_numa_node(node);
                            let raw = archive::Recorder::new(
                                raw_dir,
                                identity,
                                attempt,
                                first_worker + worker_index,
                            );
                            match raw {
                                Ok(raw) => worker(local.network(), job, &work_rx, &done_tx, raw),
                                Err(error) => {
                                    let _ = done_tx.send((from, Err(error)));
                                }
                            }
                        });
                    }
                });
            });
        }
        drop(done_tx);

        let input = job.input;
        let reader = outer.spawn(move || {
            read_range_stride(input, from, end, &work_tx, &permit_rx, job.input_width())
        });

        let result = (|| -> Result<Tally, String> {
            let mut tally = Tally::default();
            let mut pending = std::collections::BTreeMap::new();
            let batch_limit = (job.jobs * 4).min(64);
            let mut batch = Vec::with_capacity(batch_limit);
            let mut next = from;
            let mut written = 0u64;
            let mut last_report = Instant::now();
            for (at, result) in &done_rx {
                let (bytes, part, receipt) = result?;
                tally.merge(part);
                if pending.insert(at, (bytes, receipt)).is_some() {
                    return Err(format!("duplicate result chunk at {at}"));
                }
                while let Some((bytes, receipt)) = pending.remove(&next) {
                    // The chunk's own count: a record is not a fixed number of bytes in
                    // every format.
                    let records = receipt.records;
                    batch.push((bytes, receipt));
                    next += records;
                    if batch.len() >= batch_limit || next == end {
                        // Raw frames are durable already. Sync pack/journal and
                        // checkpoint once for the group rather than per 16 rows.
                        ledger.commit_batch(&batch)?;
                        written += batch
                            .iter()
                            .map(|(_, receipt)| receipt.records)
                            .sum::<u64>();
                        let committed_chunks = batch.len();
                        batch.clear();
                        for _ in 0..committed_chunks {
                            // The reader may already have finished.
                            let _ = permit_tx.send(());
                        }
                    }
                }
                if last_report.elapsed() >= REPORT_EVERY {
                    last_report = Instant::now();
                    let rate = written as f64 / started.elapsed().as_secs_f64().max(1e-9);
                    let committed = ledger.next_record();
                    let left = (end - committed) as f64 / rate.max(1e-9);
                    eprintln!(
                        "rescore: {committed}/{end} ({rate:.0}/s, {:.1} h left) {}",
                        left / 3600.0,
                        tally.summary()
                    );
                }
            }
            if next != end {
                return Err(format!("wrote up to record {next}, expected {end}"));
            }
            Ok(tally)
        })();
        // Unblock every possible wait before scoped threads are joined, also
        // when raw/pack writes fail: reader permits, queue sends and results.
        drop(permit_tx);
        drop(done_rx);
        drop(work_rx);
        let read_result = reader
            .join()
            .map_err(|_| "the reader panicked".to_string())?;
        let tally = result?;
        read_result?;
        Ok(tally)
    });

    let tally = outcome?;
    ledger.finish()?;
    let elapsed = started.elapsed().as_secs_f64().max(1e-9);
    eprintln!(
        "rescore: {} ({:.0} records/s, {:.1} ms/record/job)",
        destination.display(),
        tally.records as f64 / elapsed,
        1000.0 * elapsed * job.jobs as f64 / (tally.records.max(1) as f64),
    );
    Ok(tally)
}

/// Feed records `from..end` to the workers in chunks, numbered by the index of
/// their first record.
#[cfg(test)]
fn read_range(
    path: &Path,
    from: u64,
    end: u64,
    work: &SyncSender<Work>,
    permits: &Receiver<()>,
) -> Result<(), String> {
    read_range_stride(path, from, end, work, permits, PSV_LEN)
}

fn read_range_stride(
    path: &Path,
    from: u64,
    end: u64,
    work: &SyncSender<Work>,
    permits: &Receiver<()>,
    stride: usize,
) -> Result<(), String> {
    let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    file.seek(SeekFrom::Start(from * stride as u64))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut file = BufReader::with_capacity(1 << 20, file);
    let mut at = from;
    while at < end {
        if permits.recv().is_err() {
            return Ok(());
        }
        let take = (end - at).min(CHUNK as u64) as usize;
        let mut chunk = vec![[0u8; PSV_LEN]; take];
        for record in &mut chunk {
            file.read_exact(record)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            let mut tail = [0; 20];
            if stride > PSV_LEN {
                file.read_exact(&mut tail[..stride - PSV_LEN])
                    .map_err(|e| e.to_string())?;
            }
        }
        if work.send((at, chunk)).is_err() {
            // The writer gave up; its error is the one worth reporting.
            return Ok(());
        }
        at += take as u64;
    }
    Ok(())
}

fn worker(
    net: &attic_eval::NnueNetwork,
    job: &Job<'_>,
    work: &Mutex<Receiver<Work>>,
    done: &SyncSender<Done>,
    raw: archive::Recorder,
) {
    let mut tt = TranspositionTable::new();
    tt.resize(job.hash_mb);
    // Built once per worker: `QSearch::new` zero-fills a 51MB history table.
    let collected = Arc::new(Mutex::new(LastPvs {
        raw: Some(raw),
        ..LastPvs::default()
    }));
    let mut qs = QSearch::new(net, &tt);
    // The walk's own search and table; see the module comment for why they are
    // not the depth search's. Written and never read (`find_leaf` runs with
    // `read_tt = false`), so the table is only here because the search wants one.
    let mut walk_tt = TranspositionTable::new();
    if job.qsearch {
        walk_tt.resize(job.hash_mb);
    }
    let mut walker = job.qsearch.then(|| QSearch::new(net, &walk_tt));
    qs.set_multi_pv(job.multipv);
    qs.set_pv_output(
        PvOutputConfig {
            multi_pv: job.multipv,
            pv_interval: Duration::ZERO,
            consideration_mode: false,
            output_fail_lh_pv: false,
            start_time: Instant::now(),
        },
        Box::new(Collector(Arc::clone(&collected))),
    );
    let width = job.output_width();
    loop {
        let next = work.lock().map(|rx| rx.recv());
        let Ok(Ok((at, records))) = next else {
            return;
        };
        let mut out = Vec::with_capacity(records.len() * width);
        let mut tally = Tally::default();
        let result = (|| {
            {
                let mut slot = collected.lock().expect("pv collector mutex poisoned");
                slot.error = None;
                slot.raw
                    .as_mut()
                    .expect("worker has a raw recorder")
                    .begin_chunk(at, records.len())?;
            }
            for (i, record) in records.iter().enumerate() {
                let index = at + i as u64;
                rescore(
                    &mut qs,
                    walker.as_mut(),
                    &collected,
                    job,
                    index,
                    record,
                    &mut out,
                    &mut tally,
                )
                .map_err(|e| format!("record {index}: {e}"))?;
            }
            let mut slot = collected.lock().expect("pv collector mutex poisoned");
            let receipt = slot
                .raw
                .as_mut()
                .expect("worker has a raw recorder")
                .finish_chunk()?;
            Ok((out, tally, receipt))
        })();
        let failed = result.is_err();
        if done.send((at, result)).is_err() || failed {
            return;
        }
    }
}

/// Search record `index` and append its output record to `out`.
// The search, the walk's search, what they report into, the run, and the one record: each is
// a thing the function needs and none is a part of another.
#[allow(clippy::too_many_arguments)]
fn rescore(
    qs: &mut QSearch<'_>,
    walker: Option<&mut QSearch<'_>>,
    collected: &Mutex<LastPvs>,
    job: &Job<'_>,
    index: u64,
    record: &[u8; PSV_LEN],
    out: &mut Vec<u8>,
    tally: &mut Tally,
) -> Result<(), String> {
    let packed: PackedSfen = record[..PACKED_SFEN_LEN]
        .try_into()
        .expect("a PSV record opens with a packed position");
    let mut position = sfen_unpack(&packed).map_err(|err| format!("{err:?}"))?;
    // The decoder takes the hand pieces in whatever order they come, so bytes
    // that differ from ours can still be the same position; it is `same_move`,
    // where the file carries moves, that says whether it was read right. They
    // are counted, and the first is shown so the difference can be looked at.
    let ours = sfen_pack(&position);
    if ours != packed {
        tally.repacked += 1;
        if !SHOWN_REPACKED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "rescore: record {index} packs differently here: input {} ours {} ({})",
                hex(&packed),
                hex(&ours),
                format_sfen(&position),
            );
        }
    }
    let ply = u16::from_le_bytes([record[36], record[37]]);
    let their_move = u16::from_le_bytes([record[34], record[35]]);
    position.set_ply(ply);

    if job.flip_turn {
        if position.in_check() {
            let start = out.len();
            out.extend_from_slice(record);
            out.extend_from_slice(&[0, 0, 0, 0, 1]); // skip=1, no flipped score
            tally.records += 1;
            let mut slot = collected.lock().expect("pv collector mutex poisoned");
            if let Some(raw) = slot.raw.as_mut() {
                raw.begin_record(index, record, &format_sfen(&position))?;
                raw.skip_record(index, &out[start..])?;
            }
            return Ok(());
        }
        position = flip_position(&position)?;
    }
    // `--qsearch`: what is searched, and written, is the leaf of the quiescence
    // search from this position. `position` becomes the leaf here, so everything
    // below — the entering-king thresholds, the archive's `sfen`, the search
    // itself — is about the leaf without knowing it.
    let mut walked = 0;
    let mut walk_pv = Vec::new();
    if let Some(walker) = walker {
        let found = find_leaf(walker, &mut position);
        play_pv(&mut position, &found.pv);
        walked = found.pv.len();
        walk_pv = found.pv;
        tally.walk(walked);
    }
    qs.set_entering_king(EnteringKingConfig::new(ENTERING_KING, &position));
    {
        let mut slot = collected.lock().expect("pv collector mutex poisoned");
        slot.lines.clear();
        if let Some(raw) = slot.raw.as_mut() {
            raw.begin_record(index, record, &format_sfen(&position))?;
            if job.qsearch {
                raw.walk(index, &walk_pv)?;
            }
        }
    }
    let outcome = qs.run_root(&position, job.depth);
    {
        let slot = collected.lock().expect("pv collector mutex poisoned");
        if let Some(error) = &slot.error {
            return Err(error.clone());
        }
    }
    tally.records += 1;
    tally.nodes += outcome.nodes;

    // The two exits that never search emit no line: a declaration win and a
    // position with no legal move. Their score is still the right one
    // (`mate_in(1)` / `mated_in(1)`), there is just no move to go with it.
    let lines = match outcome.kind {
        RootKind::DeclarationWin => {
            tally.declared += 1;
            Vec::new()
        }
        RootKind::Resign => {
            tally.mated += 1;
            Vec::new()
        }
        RootKind::Normal => {
            let mut slot = collected.lock().expect("pv collector mutex poisoned");
            let mut lines = std::mem::take(&mut slot.lines);
            lines.sort_by_key(|info| info.multipv);
            lines.retain(|info| !info.pv.is_empty());
            lines
        }
    };
    if lines.len() < job.multipv {
        tally.short += 1;
    }

    let cp = stored_cp(outcome.score);
    tally.score(cp);
    let best = match outcome.kind {
        RootKind::Normal => outcome.best_move.move16(),
        RootKind::DeclarationWin | RootKind::Resign => 0,
    };
    // The move the file stored belongs to the position the file stored; at a
    // leaf there is nothing for it to agree with.
    if their_move != 0 && !job.qsearch {
        tally.with_move += 1;
        if best == their_move {
            tally.same_move += 1;
        }
    }

    let packed_start = out.len();
    match job.format {
        Format::Pack => {
            if job.flip_turn {
                out.extend_from_slice(record);
            } else {
                // A position that did not move keeps the bytes it came in with, as it
                // does without `--qsearch`; one that did is packed here.
                if walked == 0 {
                    out.extend_from_slice(&packed);
                } else {
                    out.extend_from_slice(&sfen_pack(&position));
                }
                out.extend_from_slice(&cp.to_le_bytes());
                out.extend_from_slice(&best.to_le_bytes());
                out.extend_from_slice(&leaf_ply(ply, walked).to_le_bytes());
                // The game result, turned around by an odd walk, and the padding byte
                // as the input had it.
                out.push(leaf_result(record[38], walked));
                out.push(record[39]);
            }
            for slot in 0..job.multipv {
                let (mv, cp) = match lines.get(slot) {
                    Some(info) => (info.pv[0].move16(), stored_cp(info.score)),
                    None => (0, 0),
                };
                out.extend_from_slice(&mv.to_le_bytes());
                out.extend_from_slice(&cp.to_le_bytes());
            }
            if job.flip_turn {
                // Root scores also cover declaration/no-legal-move exits without PV.
                let end = out.len();
                out[end - 4..end - 2].copy_from_slice(&best.to_le_bytes());
                out[end - 2..end].copy_from_slice(&cp.to_le_bytes());
                out.push(0);
            }
        }
        Format::Other(format) => {
            // The lines in rank order, as the first move of each and its score. The ranks that
            // came back short are not padded: a record has as many candidates as it has lines.
            let pairs: Vec<(u16, i16)> = lines
                .iter()
                .take(job.multipv)
                .map(|info| (info.pv[0].move16(), stored_cp(info.score)))
                .collect();
            format.append(
                &Searched {
                    position: &position,
                    ply: leaf_ply(ply, walked),
                    cp,
                    best,
                    lines: &pairs,
                },
                out,
            )?;
        }
    }
    let mut slot = collected.lock().expect("pv collector mutex poisoned");
    if let Some(raw) = slot.raw.as_mut() {
        raw.end_record(index, &outcome, &out[packed_start..])?;
    }
    Ok(())
}

fn flip_position(position: &attic_state::Position) -> Result<attic_state::Position, String> {
    let mut changed = position.clone();
    changed.set_side_to_move(position.side_to_move().flip());
    let rebuilt = attic_state::parse_sfen(&format_sfen(&changed)).map_err(|e| e.to_string())?;
    if rebuilt.side_to_move() != position.side_to_move().flip() || rebuilt.key() == position.key() {
        return Err("flipped side/hash reconstruction failed".into());
    }
    Ok(rebuilt)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn stored_cp(cp: i32) -> i16 {
    cp.clamp(-CP_LIMIT, CP_LIMIT) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A layout that exists to have a name and an extension.
    struct Elsewhere;
    impl RecordFormat for Elsewhere {
        fn name(&self) -> &str {
            "elsewhere"
        }
        fn extension(&self) -> &str {
            "els"
        }
        fn header(&self) -> Vec<u8> {
            Vec::new()
        }
        fn max_record_len(&self, _: usize) -> usize {
            0
        }
        fn count_records(&self, _: &[u8]) -> Option<u64> {
            Some(0)
        }
        fn append(&self, _: &Searched<'_>, _: &mut Vec<u8>) -> Result<(), String> {
            Ok(())
        }
    }

    fn pv(depth: i32, rank: usize, usi: &str, score: i32) -> PvInfo {
        let position = attic_state::parse_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        )
        .unwrap();
        PvInfo {
            depth,
            sel_depth: depth,
            multipv: rank,
            score,
            bound: attic_search::PvBound::Exact,
            nodes: 1,
            nps: 0,
            hashfull: 0,
            time_ms: 1,
            pv: vec![attic_state::parse_usi_move(usi, &position).unwrap()],
        }
    }

    #[test]
    fn the_final_snapshot_replaces_earlier_rankings() {
        let collected = Arc::new(Mutex::new(LastPvs::default()));
        let mut sink = Collector(Arc::clone(&collected));
        sink.emit(&pv(9, 1, "7g7f", -100));
        sink.emit(&pv(9, 2, "2g2f", -200));
        sink.emit(&pv(9, 1, "2g2f", -90));
        sink.emit(&pv(9, 2, "7g7f", -110));
        let slot = collected.lock().unwrap();
        assert_eq!(slot.lines.len(), 2);
        assert_eq!(slot.lines[0].pv[0], pv(9, 1, "2g2f", -90).pv[0]);
        assert_eq!(slot.lines[0].score, -90);
        assert_eq!(slot.lines[1].pv[0], pv(9, 2, "7g7f", -110).pv[0]);
        assert_eq!(slot.lines[1].score, -110);
        assert_ne!(slot.lines[0].pv[0], slot.lines[1].pv[0]);
    }

    #[test]
    fn the_last_snapshot_keeps_its_own_depths_and_drops_old_ranks() {
        let collected = Arc::new(Mutex::new(LastPvs::default()));
        let mut sink = Collector(Arc::clone(&collected));
        sink.emit(&pv(9, 1, "7g7f", 10));
        sink.emit(&pv(9, 2, "2g2f", 0));
        sink.emit(&pv(8, 1, "2g2f", 20));
        assert_eq!(collected.lock().unwrap().lines.len(), 1);
        sink.emit(&pv(7, 2, "7g7f", 10));
        let slot = collected.lock().unwrap();
        assert_eq!(slot.lines.len(), 2);
        assert_eq!(slot.lines[0].depth, 8);
        assert_eq!(slot.lines[1].depth, 7);
        assert_eq!(slot.lines[0].pv[0], pv(8, 1, "2g2f", 20).pv[0]);
        assert_eq!(slot.lines[1].pv[0], pv(7, 2, "7g7f", 10).pv[0]);
    }

    #[test]
    fn a_torn_record_is_cut_off_on_resume() {
        let dir = std::env::temp_dir().join(format!("rescore-resume-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let partial = dir.join("x.pack.partial");
        fs::write(&partial, vec![7u8; 2 * 60 + 13]).unwrap();
        assert_eq!(resume_point(&partial, 60).unwrap(), 2);
        assert_eq!(fs::metadata(&partial).unwrap().len(), 120);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn no_partial_means_starting_from_the_top() {
        let missing = std::env::temp_dir().join("rescore-no-such-file.partial");
        assert_eq!(resume_point(&missing, 60).unwrap(), 0);
    }

    #[test]
    fn a_file_without_moves_has_no_agreement_rate() {
        let tally = Tally {
            records: 2000,
            ..Tally::default()
        };
        let summary = tally.summary();
        assert!(summary.contains("same_move=n/a"), "{summary}");
    }

    #[test]
    fn agreement_is_taken_over_the_records_that_carry_a_move() {
        let tally = Tally {
            records: 2000,
            with_move: 1000,
            same_move: 450,
            ..Tally::default()
        };
        let summary = tally.summary();
        assert!(summary.contains("same_move=45.0% of 1000"), "{summary}");
    }

    #[test]
    fn reader_cannot_run_ahead_of_uncommitted_chunk_permits() {
        let path = std::env::temp_dir().join(format!("rescore-permits-{}", std::process::id()));
        fs::write(&path, vec![0; 4 * CHUNK * PSV_LEN]).unwrap();
        let (work_tx, work_rx) = sync_channel(4);
        let (permit_tx, permit_rx) = sync_channel(1);
        permit_tx.send(()).unwrap();
        let input = path.clone();
        std::thread::scope(|scope| {
            let reader = scope
                .spawn(move || read_range(&input, 0, (4 * CHUNK) as u64, &work_tx, &permit_rx));
            let (at, records) = work_rx.recv().unwrap();
            assert_eq!(at, 0);
            assert_eq!(records.len(), CHUNK);
            assert!(work_rx.try_recv().is_err());
            drop(permit_tx);
            reader.join().unwrap().unwrap();
        });
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn ranges_of_one_file_get_distinct_names() {
        let a = destination(
            Path::new("/in/shuffled_01.bin"),
            Path::new("/out"),
            0,
            100,
            false,
            Format::Pack,
        );
        let b = destination(
            Path::new("/in/shuffled_01.bin"),
            Path::new("/out"),
            100,
            200,
            false,
            Format::Pack,
        );
        assert_ne!(a, b);
        assert_eq!(a, Path::new("/out/shuffled_01.0000000000-0000000100.pack"));
    }

    /// The same range of the same file, with other positions in it, must not be
    /// able to land on the other's name.
    #[test]
    fn a_qsearch_range_is_named_apart_from_the_plain_one() {
        let at = |qsearch, format| {
            destination(
                Path::new("/in/x.bin"),
                Path::new("/out"),
                0,
                100,
                qsearch,
                format,
            )
        };
        let plain = at(false, Format::Pack);
        let leaf = at(true, Format::Pack);
        assert_ne!(plain, leaf);
        // The extension says what the part is, so the two formats of one range are not the
        // same file either.
        assert_eq!(
            at(false, Format::Other(&Elsewhere)),
            Path::new("/out/x.0000000000-0000000100.els")
        );
        assert_eq!(
            at(true, Format::Other(&Elsewhere)),
            Path::new("/out/x-qs.0000000000-0000000100.els")
        );
        assert_eq!(leaf, Path::new("/out/x-qs.0000000000-0000000100.pack"));
        // The stem gains no dot: `rescore-parquet` takes the range from after
        // the last one.
        let stem = leaf.file_name().unwrap().to_str().unwrap();
        let body = stem.strip_suffix(".pack").unwrap();
        let (before, range) = body.rsplit_once('.').unwrap();
        assert_eq!((before, range), ("x-qs", "0000000000-0000000100"));
    }

    #[test]
    fn an_odd_walk_turns_the_game_result_around_and_an_even_one_leaves_it() {
        for walked in [0, 2, 4, 26] {
            for result in [1u8, 0, 255] {
                assert_eq!(leaf_result(result, walked), result, "walk {walked}");
            }
        }
        for walked in [1, 3, 25] {
            assert_eq!(leaf_result(1, walked), 255, "a win is a loss one move on");
            assert_eq!(leaf_result(255, walked), 1, "and a loss is a win");
            assert_eq!(leaf_result(0, walked), 0, "a draw stays a draw");
        }
    }

    /// 0x80 is `i8::MIN`, which negates to itself under `wrapping_neg` and to
    /// 127 under a saturating one; a byte that is no result should not become one.
    #[test]
    fn a_byte_that_is_no_result_is_not_made_into_one() {
        assert_eq!(leaf_result(0x80, 1), 0x80);
        assert_eq!(leaf_result(0x80, 2), 0x80);
    }

    #[test]
    fn the_leaf_ply_follows_the_walk_and_stops_at_the_top_of_the_field() {
        assert_eq!(leaf_ply(37, 0), 37);
        assert_eq!(leaf_ply(37, 3), 40);
        assert_eq!(leaf_ply(u16::MAX - 1, 5), u16::MAX);
        assert_eq!(leaf_ply(10, usize::MAX), u16::MAX);
    }

    #[test]
    fn the_walk_is_counted_apart_from_the_search_and_only_when_asked_for() {
        let mut plain = Tally {
            records: 4,
            ..Tally::default()
        };
        assert!(!plain.summary().contains("qs_"), "{}", plain.summary());

        let mut tally = Tally::default();
        for walked in [0, 2, 0, 5] {
            tally.walk(walked);
        }
        assert_eq!(tally.qs_records, 4);
        assert_eq!(tally.qs_moved, 2);
        assert_eq!(tally.qs_plies, 7);
        assert_eq!(tally.qs_deepest, 5);
        tally.records = 4;
        let summary = tally.summary();
        assert!(
            summary.contains("qs_moved=2 (50.0%) qs_mean_pv=3.50 qs_deepest=5"),
            "{summary}"
        );
        // Nothing to compare the file's move with at a leaf, and the summary
        // says so instead of reporting an agreement rate of nothing.
        assert!(summary.contains("same_move=n/a (--qsearch"), "{summary}");

        let mut other = Tally::default();
        other.walk(8);
        plain.merge(other);
        tally.merge(plain);
        assert_eq!((tally.qs_records, tally.qs_moved), (5, 3));
        assert_eq!(tally.qs_plies, 15);
        assert_eq!(tally.qs_deepest, 8, "the deepest is the deeper of the two");
    }

    /// A run that walked nothing and a run that did not walk have to be told
    /// apart by the log, and an empty tally must not divide by zero.
    #[test]
    fn a_walk_tally_of_nothing_says_so_without_dividing_by_zero() {
        let mut tally = Tally::default();
        tally.walk(0);
        assert!(
            tally
                .summary()
                .contains("qs_moved=0 (0.0%) qs_mean_pv=0.00 qs_deepest=0"),
            "{}",
            tally.summary()
        );
    }

    /// The ones that want the real search; the evaluation directory is named in
    /// `ATTIC_RESCORE_EVAL_DIR`, and holds the `nn.bin` of the shape the tests were built for.
    mod with_a_network {
        use attic_search::Search;

        use super::*;

        /// Black's rook and white's bishop are trading on 8h.
        const EXCHANGE: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1b5R1/LNSGKGSNL b - 1";

        #[test]
        #[ignore = "wants an evaluation directory: set ATTIC_RESCORE_EVAL_DIR"]
        fn the_ply_of_the_position_follows_the_walk_it_was_given() {
            let dir = std::env::var("ATTIC_RESCORE_EVAL_DIR")
                .expect("set ATTIC_RESCORE_EVAL_DIR to an evaluation directory");
            let search = Search::from_network_file(&Path::new(&dir).join("nn.bin"))
                .expect("the network should load");
            let mut tt = TranspositionTable::new();
            tt.resize(16);
            let mut qs = QSearch::new(search.network(), &tt);

            let mut position = attic_state::parse_sfen(EXCHANGE).unwrap();
            position.set_ply(37);
            let before = format_sfen(&position);
            let found = find_leaf(&mut qs, &mut position);
            assert_eq!(
                format_sfen(&position),
                before,
                "the search takes back every move it plays"
            );
            assert!(!found.pv.is_empty(), "a live exchange has a walk");

            play_pv(&mut position, &found.pv);
            let walked = found.pv.len();
            assert_eq!(usize::from(position.ply()), 37 + walked);
            assert_eq!(position.ply(), leaf_ply(37, walked));
            // The leaf is another position, and packs to other bytes.
            assert_ne!(format_sfen(&position), before);
        }
    }
}

#[cfg(test)]
mod flip_tests {
    use super::*;
    #[test]
    fn flip_rebuilds_hash_and_preserves_board() {
        let p = attic_state::parse_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        )
        .unwrap();
        let q = flip_position(&p).unwrap();
        assert_ne!(p.key(), q.key());
        assert_eq!(
            format_sfen(&p).split_whitespace().next(),
            format_sfen(&q).split_whitespace().next()
        );
        assert_eq!(flip_position(&q).unwrap().key(), p.key());
    }
    #[test]
    fn packed_input_skips_pv_tail() {
        let path = std::env::temp_dir().join(format!("flip-stride-{}", std::process::id()));
        let mut bytes = vec![9; 120];
        bytes[..40].fill(1);
        bytes[60..100].fill(2);
        fs::write(&path, bytes).unwrap();
        let (tx, rx) = sync_channel(1);
        let (pt, pr) = sync_channel(1);
        pt.send(()).unwrap();
        read_range_stride(&path, 0, 2, &tx, &pr, 60).unwrap();
        let (_, rows) = rx.recv().unwrap();
        assert_eq!(rows, vec![[1; 40], [2; 40]]);
        fs::remove_file(path).unwrap();
    }
}
