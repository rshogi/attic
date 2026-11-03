//! The dedup filter and the move margin, played out on a network of zeros.
//!
//! Gated like [`resumed_budget_tests`](super::resumed_budget_tests), whose zero
//! network and helpers these share.

use super::resumed_budget_tests::{options, play, play_seen};
use super::*;

/// Ten plies against a flat network: every evaluation is 0, so the search
/// has nothing to prefer and plays the same move every time. That is what
/// makes "the second game retraced the first" assertable — without the
/// filter it is exactly what happens.
const BUDGET: u16 = 10;

fn boards(game: &Game) -> Vec<String> {
    game.records.iter().map(|r| r.sfen.clone()).collect()
}

#[test]
fn two_games_retrace_each_other_when_no_table_is_lent() {
    let options = options(BUDGET);
    let first = play(None, &options);
    let second = play(None, &options);

    assert_eq!(boards(&first), boards(&second));
    assert_eq!(first.repeats, 0);
    assert_eq!(second.repeats, 0);
}

#[test]
fn the_second_game_diverges_where_it_starts_repeating() {
    let options = options(BUDGET);
    let seen = Seen::new(1).expect("1 MB is a table");
    let first = play_seen(None, &options, Some(&seen));
    let second = play_seen(None, &options, Some(&seen));

    // The first game meets an empty table and is the game above.
    assert_eq!(boards(&first), boards(&play(None, &options)));
    assert_eq!(first.repeats, 0);

    // The second meets the first game's own opening and leaves it.
    assert!(second.repeats > 0, "the opening was already in the table");
    assert_ne!(boards(&first), boards(&second));
}

#[test]
fn a_repeated_ply_is_played_but_not_recorded() {
    let options = options(BUDGET);
    let seen = Seen::new(1).expect("1 MB is a table");
    let first = play_seen(None, &options, Some(&seen));
    let second = play_seen(None, &options, Some(&seen));

    // Both games are the same length — a repeat drops the *record*, not
    // the ply, and the search still runs — so the records the second one
    // is missing are its repeats, one for one. `--max-ply` ends both.
    assert_eq!(first.reason, "max_ply");
    assert_eq!(second.reason, "max_ply");
    assert_eq!(first.records.len(), usize::from(BUDGET));
    assert_eq!(
        second.records.len() + second.repeats as usize,
        usize::from(BUDGET),
    );
}

#[test]
fn a_margin_draws_among_the_candidates_it_admits() {
    // The test network is flat, so every candidate scores 0 and any margin
    // admits all of them. The greedy game plays the best move every ply;
    // the same game under a margin draws instead, and leaves that line
    // without a random move and without a table.
    // The helper asks for one PV line; a margin has nothing to draw from
    // until the search reports more than the best move.
    let mut greedy = options(BUDGET);
    greedy.multipv = 4;
    let mut drawn = options(BUDGET);
    drawn.multipv = 4;
    drawn.move_margin = 100;

    let first = play(None, &greedy);
    let second = play(None, &drawn);

    assert_ne!(boards(&first), boards(&second));
    assert_eq!(first.records.len(), second.records.len());
    assert_eq!(second.repeats, 0, "no table was lent");
}

#[test]
fn the_admitted_moves_are_the_ones_within_the_margin() {
    let mv = attic_state::Move::none();
    let candidates = [(mv, 120), (mv, 100), (mv, 20), (mv, -30)];

    assert_eq!(within_margin(&candidates, 100), 3);
    assert_eq!(within_margin(&candidates, 20), 2);
    assert_eq!(within_margin(&candidates, 0), 4, "the margin is off");
    assert_eq!(within_margin(&[], 100), 0);
}

#[test]
fn the_ply_bound_keeps_the_table_out_of_the_game_entirely() {
    let mut options = options(BUDGET);
    options.dedup_max_ply = 0;
    let seen = Seen::new(1).expect("1 MB is a table");
    let first = play_seen(None, &options, Some(&seen));
    let second = play_seen(None, &options, Some(&seen));

    // A window of zero plies: the table is never read and never written,
    // so the two games are the pair from the first test.
    assert_eq!(boards(&first), boards(&second));
    assert_eq!(first.repeats, 0);
    assert_eq!(second.repeats, 0);
}
