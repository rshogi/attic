//! One worker of the tree: takes boards off the frontier and searches them.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use attic_search::{EnteringKingConfig, PvOutputConfig, QSearch, SearchControl};
use attic_state::{format_sfen, parse_sfen};
use attic_storage::TranspositionTable;

use super::Frontier;
use crate::game::{Collector, DeepestPvs, drain_candidates, within_margin};
use crate::governor::Governor;
use crate::options::Options;

/// Takes boards off the frontier and searches them until the last level is
/// written.
pub fn run(
    net: &attic_eval::NnueNetwork,
    tt: &TranspositionTable,
    options: &Options,
    governor: &'static Governor,
    frontier: &Frontier,
    index: usize,
) {
    let collected = Arc::new(Mutex::new(DeepestPvs::default()));
    let mut qs = QSearch::new(net, tt);
    qs.set_multi_pv(options.multipv);
    if options.nodes > 0 {
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

    loop {
        // Checked before claiming rather than while holding a board: a parked
        // worker still holding one would keep its level from ever closing.
        while !governor.admits(index) {
            std::thread::sleep(Duration::from_secs(2));
        }

        let Some(task) = frontier.claim() else {
            return;
        };
        let Ok(mut pos) = parse_sfen(&task.sfen) else {
            eprintln!("worker {index}: cannot parse {}", task.sfen);
            frontier.finish(None, Vec::new());
            continue;
        };

        // Before the first move, as `play_game` does it: the searches copy the
        // position, so the rule travels into every one of them. Left to the
        // parse default, `--quick-draw` would reach the games and not the book.
        pos.set_repetition_rule(options.repetition_rule);
        qs.set_entering_king(EnteringKingConfig::new(options.entering_king, &pos));
        collected
            .lock()
            .expect("pv collector mutex poisoned")
            .reset();
        let outcome = qs.run_root(&pos, options.depth);
        let candidates = drain_candidates(&collected);
        let root_cp = outcome.score;

        // A mated board is not a book position. The band is not judged here:
        // it is measured from the level's median, which no single search
        // knows.
        if candidates.is_empty() {
            frontier.finish(None, Vec::new());
            continue;
        }

        let mut children = Vec::new();
        if task.expand {
            let admitted = within_margin(&candidates, options.move_margin);
            for &(mv, _) in &candidates[..admitted] {
                let undo = pos.do_move(mv);
                children.push((pos.key(), format_sfen(&pos), task.sfen.clone()));
                pos.undo_move(mv, undo);
            }
        }
        frontier.finish(Some((root_cp, task.sfen, task.parent)), children);
    }
}
