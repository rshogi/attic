//! The forward pass and the accumulator.

use attic_state::{Color, Position, parse_sfen};

use super::super::forward::{dense_clip, features_from_sums};
use super::super::{
    Accumulator, BLOCK_LEN, DIM, FEATURE_LEN, FIELD_AT, FIELD_LEN, FinnyCache, HAS_FIELD,
    HAS_MOMENT, HAS_PAIR, HIDDEN, HIDDEN2, MOMENT_AT, MOMENT_LEN, ORDER, PAIR, PAIR_AT,
    W1Precision, evaluate_pre, evaluate_value, value_from_sums,
};
use super::fixtures::{assert_close, assert_close_relative, game, least, random_game, synthetic};
use super::reference::reference_value;
use crate::MoveDelta;
#[cfg(feature = "sym_token")]
use crate::active_features;

/// How far a pair sum built by deltas may be from one built from scratch, as a
/// fraction of its size (or of one, for a sum smaller than that). A game of 160
/// plies adds and takes out a few thousand rows of it.
const PAIR_DRIFT: f32 = 2e-5;

/// The same for the second moment and the square field. Those sums are in the
/// tens, and are carried in `f64` and rounded to `f32` once, so what separates a
/// derived sum from a rebuilt one is that one rounding and not the games.
const SUM_DRIFT: f32 = 2e-5;

/// The value is finite and within the reach of its weights, on a position with
/// hands and promoted pieces as well as the opening. There is no `tanh`: the value
/// is the value row's output as it is, so `evaluate_pre` is the same number, and
/// what bounds it is the row over activations clipped to `[0, 1]`.
#[test]
fn the_value_is_finite_and_within_the_reach_of_its_weights() {
    let net = synthetic();
    let mixed =
        parse_sfen("ln2b4/5skgl/p4pn1r/2+P1p2pp/1sP3p2/2GSKP1GP/P1NPS2PN/1+pRpP2+bL/L4G3 w Pp 94")
            .expect("a valid sfen");
    // Every activation is in `[0, 1]`, so no row of weights over them can reach
    // further than the weights' own sum of magnitudes.
    let reach = net.value_b.abs() + net.value_w.iter().map(|w| w.abs()).sum::<f32>();
    for pos in [Position::startpos(), mixed] {
        let value = evaluate_value(net, &pos);
        let pre = evaluate_pre(net, &pos);
        assert!(value.is_finite() && value.abs() <= reach, "{value}");
        assert_eq!(value, pre);
    }
}

/// The value is linear: nothing saturates it. Shifting the value row's bias by a
/// hundred moves the value by a hundred, where a `tanh` would pin it near one.
#[test]
fn the_value_is_linear() {
    let net = synthetic();
    let pos = Position::startpos();
    let before = evaluate_value(net, &pos);
    let mut shifted = net.replicate();
    shifted.value_b += 100.0;
    let after = evaluate_value(&shifted, &pos);
    assert!((after - before - 100.0).abs() < 1e-3, "{before} -> {after}");
    assert!(after > 1.0, "{after}");
}

/// Both hidden layers clip at both ends, read off a net built so that the answer
/// is a handful of numbers rather than a sum over hundreds. The first layer
/// reads nothing (`w1 = 0`), so its units are its biases, and the second reads
/// one unit each at a tenth of its weight.
///
/// In the first scenario the second layer has no bias, so what it sees of the
/// first is its clip: a unit of `3` arrives as `1`, and of `-3` as `0`. In the
/// second the second layer's own bias drives it past each rail from either side,
/// which the first scenario's small products never reach.
#[test]
fn a_hidden_layer_clips_at_both_ends() {
    let pos = Position::startpos();
    let clip = |x: f64| x.clamp(0.0, 1.0);
    let units = least(HIDDEN, HIDDEN2);
    for (name, first, second) in [
        (
            "first layer",
            [-3.0f32, 0.25, 0.5, 3.0],
            [0.0f32, 0.0, 0.0, 0.0],
        ),
        (
            "second layer",
            [3.0f32, 0.5, 3.0, -3.0],
            [-3.0f32, 0.0, 3.0, 0.0],
        ),
    ] {
        let mut net = synthetic().replicate();
        net.w1.fill(0.0);
        net.w2.fill(0.0);
        net.value_w.fill(1.0);
        net.value_b = 0.0;
        for (i, b) in net.b1.iter_mut().enumerate() {
            *b = first[i % 4];
        }
        for (i, b) in net.b2.iter_mut().enumerate() {
            *b = second[i % 4];
        }
        for unit in 0..units {
            net.w2[unit * HIDDEN + unit] = 0.1;
        }
        let want: f64 = (0..HIDDEN2)
            .map(|i| {
                let h = clip(f64::from(net.b1[i]));
                let w = if i < units { 0.1 } else { 0.0 };
                clip(f64::from(w * h as f32) + f64::from(net.b2[i]))
            })
            .sum();
        let got = f64::from(evaluate_value(&net, &pos));
        assert!((got - want).abs() < 1e-5, "{name}: {got} against {want}");
    }
}

/// What the head reads: `(name, start, length)` of every block in one
/// perspective's share, at the offsets `shape.rs` gives.
fn blocks() -> Vec<(String, usize, usize)> {
    let mut blocks = Vec::new();
    for k in 0..ORDER {
        blocks.push((format!("e{}", k + 1), k * DIM, DIM));
    }
    if HAS_PAIR {
        blocks.push(("pair".to_string(), PAIR_AT, PAIR));
    }
    if HAS_MOMENT {
        blocks.push(("moment".to_string(), MOMENT_AT, MOMENT_LEN));
    }
    if HAS_FIELD {
        blocks.push(("field".to_string(), FIELD_AT, FIELD_LEN));
    }
    blocks
}

/// The test net has to be one the clip can be seen in. A block that sits on
/// a rail everywhere is a block whose arithmetic no comparison can tell from any
/// other, and a clip that was never reached (or only ever reached from one side)
/// is a clip nobody tested. So: every block has entries strictly inside `(0, 1)`
/// in a good share of its positions, and every block reaches both rails.
#[test]
fn a_test_net_exercises_the_clip_in_every_block_and_layer() {
    let net = synthetic();
    let mut tally: Vec<(String, usize, usize, usize)> = blocks()
        .into_iter()
        .map(|(name, _, _)| (name, 0, 0, 0))
        .collect();
    let layout = blocks();
    // Hidden layers: low, inside, high.
    let mut hidden_tally = [[0usize; 3]; 2];
    let bucket = |v: f32| match v {
        v if v <= 0.0 => 0,
        v if v >= 1.0 => 2,
        _ => 1,
    };
    for (pos, _) in game(30) {
        let mut acc = Accumulator::new();
        acc.refresh(net, &pos);
        let mut features = [0.0f32; FEATURE_LEN];
        features_from_sums(net, &acc, pos.side_to_move(), &mut features);
        for half in 0..2 {
            let share = &features[half * BLOCK_LEN..(half + 1) * BLOCK_LEN];
            for ((_, start, len), (_, low, inside, high)) in layout.iter().zip(&mut tally) {
                for &v in &share[*start..*start + *len] {
                    match bucket(v) {
                        0 => *low += 1,
                        1 => *inside += 1,
                        _ => *high += 1,
                    }
                }
            }
        }
        let mut hidden = [0.0f32; HIDDEN];
        dense_clip(&net.w1, &net.b1, &features, &mut hidden);
        for &v in &hidden {
            hidden_tally[0][bucket(v)] += 1;
        }
        let mut second = [0.0f32; HIDDEN2];
        dense_clip(&net.w2, &net.b2, &hidden, &mut second);
        for &v in &second {
            hidden_tally[1][bucket(v)] += 1;
        }
    }
    for (name, low, inside, high) in &tally {
        println!("{name}: {low} at 0, {inside} inside, {high} at 1");
        let total = low + inside + high;
        assert!(*inside * 10 >= total, "{name}: nearly all on a rail");
        assert!(*low > 0 && *high > 0, "{name}: a rail was never reached");
    }
    for (layer, [low, inside, high]) in hidden_tally.iter().enumerate() {
        println!("hidden {layer}: {low} at 0, {inside} inside, {high} at 1");
        let total = low + inside + high;
        assert!(
            *inside * 10 >= total,
            "hidden {layer}: nearly all on a rail"
        );
        assert!(
            *low > 0 && *high > 0,
            "hidden {layer}: a rail was never reached"
        );
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

/// Each integer first layer is the one the network runs once it is chosen, and
/// lands near the `f32` value: the `i16` layer within a hundredth of a unit, the
/// `i8` one within a tenth, on every position of a game. Neither is the `f32`
/// value to the bit, which is what shows the choice reached the forward pass.
#[test]
fn an_integer_first_layer_runs_in_place_of_w1_and_stays_near_it() {
    let base = synthetic();
    for (precision, tolerance) in [(W1Precision::I16, 0.01f32), (W1Precision::I8, 0.1)] {
        let mut net = base.replicate();
        net.use_first_layer(precision);
        let (mut worst, mut differs) = (0.0f32, false);
        for (pos, _) in random_game(3, 120) {
            let (exact, got) = (evaluate_value(base, &pos), evaluate_value(&net, &pos));
            worst = worst.max((got - exact).abs());
            differs |= got.to_bits() != exact.to_bits();
        }
        assert!(
            differs,
            "{precision:?}: the value is the f32 one to the bit"
        );
        assert!(
            worst < tolerance,
            "{precision:?}: {worst} from the f32 value"
        );
        net.use_first_layer(W1Precision::F32);
        for (pos, _) in random_game(3, 40) {
            assert_eq!(
                evaluate_value(&net, &pos).to_bits(),
                evaluate_value(base, &pos).to_bits(),
                "{precision:?} switched back to f32 is the f32 layer"
            );
        }
    }
}

#[test]
fn the_precision_is_read_from_its_name() {
    for (name, want) in [
        (None, W1Precision::I16),
        (Some("i16"), W1Precision::I16),
        (Some("f32"), W1Precision::F32),
        (Some("F32"), W1Precision::F32),
        (Some("i8"), W1Precision::I8),
        (Some("I8"), W1Precision::I8),
        (Some("u8"), W1Precision::I16),
        (Some(""), W1Precision::I16),
    ] {
        assert_eq!(W1Precision::from_name(name), want, "{name:?}");
    }
}

/// The second layer is read, not skipped: moving its bias moves the value.
#[test]
fn the_second_hidden_layer_changes_the_value() {
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
