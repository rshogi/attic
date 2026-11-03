//! One worker: plays games back to back and appends them to its own file.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use attic_search::{PvOutputConfig, QSearch, SearchControl};
use attic_storage::TranspositionTable;

use crate::game::{Collector, DeepestPvs, Rng, Start, play_game};
use crate::governor::Governor;
use crate::options::{Format, Options};
use crate::recover::{Resume, Shard, Source, record_len, resume_from, rewrite_results};

/// Each worker writes its own file rather than sharing one behind a lock: a
/// game is a single long line, and interleaving two writers mid-line corrupts
/// both. The consumer reads the directory, not one file.
#[allow(clippy::too_many_arguments)]
pub fn run(
    net: &attic_eval::NnueNetwork,
    tt: &TranspositionTable,
    options: &Options,
    governor: &'static Governor,
    seed: u64,
    index: usize,
    games_played: &Arc<AtomicU64>,
    positions_written: &Arc<AtomicU64>,
    source: Option<&Source>,
) {
    let suffix = match options.format {
        Format::Jsonl => "jsonl",
        Format::Psv | Format::Pack => "bin",
    };
    let path = options.out.with_extension(format!("w{index:03}.{suffix}"));
    let mut file = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(file) => file,
        Err(err) => {
            eprintln!("worker {index}: cannot open {}: {err}", path.display());
            return;
        }
    };

    // Built once, not per move or per game: `QSearch::new` zero-fills a 51MB
    // continuation-history table, which dominated everything else when it sat
    // in the move loop.
    let collected = Arc::new(Mutex::new(DeepestPvs::default()));
    let mut qs = QSearch::new(net, tt);
    qs.set_multi_pv(options.multipv);
    if options.nodes > 0 {
        // `go depth D nodes N`: whichever bound is reached first ends the
        // search. Left unset, `run_root` is bounded by depth alone.
        qs.set_control(SearchControl {
            node_limit: Some(options.nodes),
            ..SearchControl::default()
        });
    }
    qs.set_pv_output(
        PvOutputConfig {
            multi_pv: options.multipv,
            pv_interval: Duration::ZERO,
            consideration_mode: false,
            output_fail_lh_pv: false,
            start_time: Instant::now(),
        },
        Box::new(Collector(Arc::clone(&collected))),
    );

    let mut rng = Rng::new(seed);
    let started = Instant::now();
    let mut mine = 0u64;
    let mut played = 0u64;
    // Held across games: one shard is read to its end before the next is
    // claimed, so a worker opens ~one file rather than one per game.
    let mut shard: Option<Shard> = None;

    loop {
        // Both quotas are fleet-wide and checked before starting a game, so the
        // run overshoots by at most one game per worker rather than cutting a
        // game off mid-record.
        if options.games > 0 && games_played.load(Ordering::Relaxed) >= options.games {
            break;
        }
        if options.positions > 0 && positions_written.load(Ordering::Relaxed) >= options.positions {
            break;
        }

        // Checked between games so a shrink never discards one in flight. A
        // parked worker keeps its engine and history tables — rebuilding those
        // costs more than the idle thread does, and the machine filling up is
        // usually temporary.
        while !governor.admits(index) {
            std::thread::sleep(Duration::from_secs(2));
            if options.games > 0 && games_played.load(Ordering::Relaxed) >= options.games {
                return;
            }
            // A recovery run is bounded by its input, not by a quota. Without
            // this a parked worker waits for an admission that would only let
            // it find there is nothing left.
            if shard.is_none() && source.is_some_and(Source::exhausted) {
                return;
            }
        }

        // Empty unless this is a recovery run, where it holds the records kept
        // from the original game and the position to play on from.
        let resume = match source {
            None => None,
            Some(source) => match next_resume(source, &mut shard, index) {
                Some(resume) => Some(resume),
                None => break,
            },
        };
        let (start, prefix) = match resume {
            Some(resume) => (
                Some(Start {
                    position: resume.position,
                    played: resume.played,
                }),
                resume.prefix,
            ),
            None => (None, Vec::new()),
        };
        let game = play_game(&mut qs, &collected, options, &mut rng, start);
        // The kept records carry the original game's results, and the
        // continuation may have ended the other way round, so they are stamped
        // with this game's result before the new records are appended.
        let wrote = match options.format {
            Format::Jsonl => {
                let line = game.to_json("attic", options.depth, options.multipv);
                writeln!(file, "{line}")
            }
            Format::Psv => {
                let mut buf = prefix;
                rewrite_results(&mut buf, game.result, record_len(0));
                game.to_psv(&mut buf);
                file.write_all(&buf)
            }
            Format::Pack => {
                let mut buf = prefix;
                rewrite_results(&mut buf, game.result, record_len(options.multipv));
                game.to_pack(&mut buf, options.multipv);
                file.write_all(&buf)
            }
        };
        if let Err(err) = wrote {
            eprintln!("worker {index}: write failed: {err}");
            return;
        }

        mine += 1;
        played += game.records.len() as u64;
        positions_written.fetch_add(game.records.len() as u64, Ordering::Relaxed);
        let total = games_played.fetch_add(1, Ordering::Relaxed) + 1;
        if mine.is_multiple_of(10) {
            let elapsed = started.elapsed().as_secs_f64();
            let rate = elapsed / mine as f64;
            match source {
                // Positions per game is what sizes a recovery run: the games
                // start deep, so it is far from a full game's worth and is the
                // number the whole-corpus estimate hangs on.
                Some(_) => eprintln!(
                    "worker {index}: {mine} games, {rate:.1} s/game, \
                     {:.1} new positions/game (fleet {total})",
                    played as f64 / mine as f64
                ),
                None => eprintln!("worker {index}: {mine} games, {rate:.1} s/game (fleet {total})"),
            }
        }
    }
}

/// The next game to resume, or `None` once every shard has been read.
///
/// Shards are claimed whole and read to the end before the next is taken. A
/// record that will not decode is reported and its shard abandoned rather than
/// stopping the worker — one bad offset should not cost the other shards.
fn next_resume(source: &Source, shard: &mut Option<Shard>, index: usize) -> Option<Resume> {
    let len = record_len(source.multipv);
    loop {
        if shard.is_none() {
            let path = source.claim()?;
            match Shard::open(path, source.multipv) {
                Ok(open) => *shard = Some(open),
                Err(err) => {
                    eprintln!("worker {index}: {err}");
                    continue;
                }
            }
        }
        let open = shard.as_mut().expect("a shard was just opened");
        match open.next_game() {
            None => *shard = None,
            Some(Err(err)) => {
                eprintln!("worker {index}: {err}");
                *shard = None;
            }
            Some(Ok(records)) => match resume_from(&records, len) {
                Some(Ok(resume)) => return Some(resume),
                Some(Err(err)) => eprintln!("worker {index}: {err}"),
                None => {}
            },
        }
    }
}
