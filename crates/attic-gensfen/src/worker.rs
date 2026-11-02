//! One worker: plays games back to back and appends them to its own file.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use attic_search::{PvOutputConfig, QSearch, SearchControl};
use attic_storage::TranspositionTable;

use crate::game::{Collector, DeepestPvs, Rng, play_game};
use crate::options::Options;

/// Each worker writes its own file rather than sharing one behind a lock: a
/// game is a single long line, and interleaving two writers mid-line corrupts
/// both. The consumer reads the directory, not one file.
pub fn run(
    net: &attic_eval::NnueNetwork,
    tt: &TranspositionTable,
    options: &Options,
    seed: u64,
    index: usize,
    games_played: &Arc<AtomicU64>,
    positions_written: &Arc<AtomicU64>,
) {
    let path = options.out.with_extension(format!("w{index:03}.jsonl"));
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

        let game = play_game(&mut qs, &collected, options, &mut rng);
        let line = game.to_json("Attic/NAGISA_V4", options.depth, options.multipv);
        if let Err(err) = writeln!(file, "{line}") {
            eprintln!("worker {index}: write failed: {err}");
            return;
        }

        mine += 1;
        positions_written.fetch_add(game.records.len() as u64, Ordering::Relaxed);
        let total = games_played.fetch_add(1, Ordering::Relaxed) + 1;
        if mine.is_multiple_of(10) {
            let elapsed = started.elapsed().as_secs_f64();
            eprintln!(
                "worker {index}: {mine} games, {:.1} s/game (fleet {total})",
                elapsed / mine as f64
            );
        }
    }
}
