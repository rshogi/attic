//! The forward pass and the accumulator.

use attic_state::{Color, Position, parse_sfen};

use super::super::{
    Accumulator, FinnyCache, SECOND_LAYER, evaluate_pre, evaluate_value, value_from_sums,
};
use super::fixtures::{assert_close, assert_close_relative, game, random_game, synthetic};
use super::reference::reference_value;
use crate::MoveDelta;
#[cfg(feature = "sym_token")]
use crate::active_features;

/// How far a pair sum built by deltas may be from one built from scratch, as a
/// fraction of its size (or of one, for a sum smaller than that). The sums are
/// in the hundreds in the log-sum-exp channels, where one `f32` step is a few
/// millionths of the sum, and a game of 160 plies adds and takes out a few
/// thousand rows of them.
const PAIR_DRIFT: f32 = 2e-5;

/// The same for the second moment and the square field. Those sums are in the
/// tens, and are carried in `f64` and rounded to `f32` once, so what separates a
/// derived sum from a rebuilt one is that one rounding and not the games.
const SUM_DRIFT: f32 = 2e-5;

/// The value is a number in `[-1, 1]` and is the `tanh` of the pre-activation
/// it reports, on a position with hands and promoted pieces as well as the
/// opening.
#[test]
fn the_value_is_finite_bounded_and_the_tanh_of_the_pre_activation() {
    let net = synthetic();
    let mixed =
        parse_sfen("ln2b4/5skgl/p4pn1r/2+P1p2pp/1sP3p2/2GSKP1GP/P1NPS2PN/1+pRpP2+bL/L4G3 w Pp 94")
            .expect("a valid sfen");
    for pos in [Position::startpos(), mixed] {
        let value = evaluate_value(net, &pos);
        let pre = evaluate_pre(net, &pos);
        assert!(
            value.is_finite() && (-1.0..=1.0).contains(&value),
            "{value}"
        );
        assert_eq!(value, pre.tanh());
    }
}

/// The forward pass agrees with [`reference_value`] over a played-out game,
/// which has drops, captures, promotions and a king that moves.
#[test]
fn the_forward_pass_matches_a_reference_written_from_the_definition() {
    let net = synthetic();
    let mut worst = 0.0f64;
    for (pos, mv) in game(40) {
        let mut after = pos.clone();
        let _ = after.do_move(mv);
        for p in [pos, after] {
            let got = f64::from(evaluate_value(net, &p));
            worst = worst.max((got - reference_value(net, &p)).abs());
        }
    }
    assert!(
        worst < 1e-4,
        "f32 forward is {worst} from the f64 reference"
    );
}

/// The second layer is read, not skipped: moving its bias moves the value.
#[test]
fn the_second_hidden_layer_changes_the_value() {
    if !SECOND_LAYER {
        return;
    }
    let net = synthetic();
    let pos = Position::startpos();
    let before = evaluate_pre(net, &pos);
    let mut shifted = net.replicate();
    for b in &mut shifted.b2 {
        *b += 0.5;
    }
    assert_ne!(evaluate_pre(&shifted, &pos), before);
}

/// The incremental path and a rebuild agree over a played-out game, by both of
/// the entry points the search has: a derive from the previous accumulator, and
/// the one-off `update_after_move` at the root.
///
/// The sums are not the same *bits*, since f32 addition is not associative and
/// the two arrive at each in a different order, so this is a tolerance. It is
/// a game rather than a position so that drift would show.
///
/// The tolerance is absolute, which is only sound because the network is
/// synthetic: its embedding is at most 0.6 in magnitude, so a power sum stays in
/// the tens and the f32 rounding of one is far under `1e-3`. A trained
/// network's sums reach thousands, and `tests/sym_network.rs` scales its bound
/// to the sum for that reason.
#[test]
fn the_incremental_path_matches_a_rebuild() {
    let net = synthetic();
    let mut cache = FinnyCache::new();
    let mut pos = Position::startpos();
    let mut acc = Accumulator::new();
    acc.refresh(net, &pos);
    let mut derived = Accumulator::new();

    for (before, mv) in game(40) {
        assert_eq!(before.side_to_move(), pos.side_to_move());
        let delta = MoveDelta::from_move(&pos, mv);
        let root = acc.update_after_move(net, &mut pos, mv);
        let _ = pos.do_move(mv);
        Accumulator::derive_into_cached(&acc, &mut derived, net, &pos, &delta, &mut cache);

        let mut rebuilt = Accumulator::new();
        rebuilt.refresh(net, &pos);
        for color in [Color::Black, Color::White] {
            assert_close(
                "derive",
                derived.perspective(color),
                rebuilt.perspective(color),
                1e-3,
            );
            assert_close(
                "update_after_move",
                root.perspective(color),
                rebuilt.perspective(color),
                1e-3,
            );
            assert_close_relative(
                "derive pair",
                &derived.pair(color),
                &rebuilt.pair(color),
                PAIR_DRIFT,
            );
            assert_close_relative(
                "update_after_move pair",
                &root.pair(color),
                &rebuilt.pair(color),
                PAIR_DRIFT,
            );
            for (what, got, want) in [
                ("derive", &derived, &rebuilt),
                ("update_after_move", &root, &rebuilt),
            ] {
                assert_close_relative(what, &got.moment(color), &want.moment(color), SUM_DRIFT);
                assert_close_relative(what, &got.field(color), &want.field(color), SUM_DRIFT);
            }
        }
        let want = evaluate_value(net, &pos);
        let got = value_from_sums(net, &derived, pos.side_to_move());
        assert!((want - got).abs() < 1e-4, "value {got} against {want}");
        acc.clone_from(&derived);
    }
}

/// The same, over games that reach what the opening line does not: captures,
/// drops of every kind of piece, promotions, and a king that walks. Several of
/// them and long, so that drift would show, and an accumulator carried the whole
/// way down each game by derivation alone — never rebuilt — so that a rebuild
/// is not what makes it agree.
///
/// Where the token shapes keep the set of tokens they hold, that is checked too:
/// the set is what a table over pairs of tokens would be read from.
#[test]
fn the_incremental_path_survives_long_random_games() {
    let net = synthetic();
    let mut cache = FinnyCache::new();
    for seed in 1..=4 {
        let game = random_game(seed, 160);
        let mut acc = Accumulator::new();
        acc.refresh(net, &game[0].0);
        let mut derived = Accumulator::new();
        let mut post = game[0].0.clone();
        for (before, mv) in &game {
            let delta = MoveDelta::from_move(before, *mv);
            post.clone_from(before);
            let _ = post.do_move(*mv);
            Accumulator::derive_into_cached(&acc, &mut derived, net, &post, &delta, &mut cache);

            let mut rebuilt = Accumulator::new();
            rebuilt.refresh(net, &post);
            for color in [Color::Black, Color::White] {
                assert_close(
                    "derive",
                    derived.perspective(color),
                    rebuilt.perspective(color),
                    1e-3,
                );
                assert_close_relative(
                    "derive pair",
                    &derived.pair(color),
                    &rebuilt.pair(color),
                    PAIR_DRIFT,
                );
                assert_close_relative(
                    "derive moment",
                    &derived.moment(color),
                    &rebuilt.moment(color),
                    SUM_DRIFT,
                );
                assert_close_relative(
                    "derive field",
                    &derived.field(color),
                    &rebuilt.field(color),
                    SUM_DRIFT,
                );
                #[cfg(feature = "sym_token")]
                {
                    let mut held = derived.tokens(color).to_vec();
                    let mut want = active_features(&post, color);
                    held.sort_unstable();
                    want.sort_unstable();
                    assert_eq!(held, want, "seed {seed}, {color:?}: the tokens held");
                }
            }
            acc.clone_from(&derived);
        }
    }
}
