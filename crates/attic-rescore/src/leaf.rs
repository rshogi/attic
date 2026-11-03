//! Walking a position to the leaf of its quiescence search.
//!
//! A record taken from the middle of an exchange has an evaluation that swings
//! by a piece depending on whose turn it is to recapture, and a value head
//! trained on such records learns the swing rather than the position. The
//! answer is the one upstream's `qsearch_psv` gives: run the quiescence search,
//! walk its principal variation to the end, and store *that* position.
//!
//! This module is the two steps of it that need the search and nothing else,
//! so that every tool that walks to a leaf walks by the same calls.

use attic_search::{QSearch, QSearchOutcome};
use attic_state::{Move, Position};
use attic_storage::Value;

/// The reference's `VALUE_INFINITE` (`types.h`), the widest window a search
/// can be asked for. Spelled out here rather than imported because
/// `attic-search` keeps its copy private, and `attic-protocol` keeps a third
/// for the same reason.
const VALUE_INFINITE: Value = 32001;

/// Run the quiescence search on `position` and return what it found.
///
/// The variation in the outcome is the walk to the leaf; [`play_pv`] takes it
/// there. `position` is handed back as it was: the search plays its moves and
/// takes every one of them back.
///
/// `pv_node = true` or no principal variation comes back at all
/// (`QSearchOutcome::pv` is empty for a non-PV run or a fail-low). `read_tt =
/// false` so that one record's leaf never depends on what an earlier record
/// happened to leave in the table.
pub fn find_leaf(qs: &mut QSearch, position: &mut Position) -> QSearchOutcome {
    qs.run(position, -VALUE_INFINITE, VALUE_INFINITE, true, false)
}

/// Play `pv` out from `position`, which then is the leaf.
///
/// The ply counter of the position follows: each move is one more, so a
/// position that came in at ply `n` leaves at `n + pv.len()`. Callers that
/// carry a ply of their own beside the position read it off from here or add
/// the length themselves, and have to agree.
pub fn play_pv(position: &mut Position, pv: &[Move]) {
    for &mv in pv {
        debug_assert!(
            position.pseudo_legal(mv, true),
            "the search returned a variation it cannot play"
        );
        position.do_move(mv);
    }
}
