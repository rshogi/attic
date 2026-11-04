//! The forward pass and the accumulator.

use attic_state::{Color, Position, parse_sfen};

use super::super::{
    Accumulator, DIM, FEATURE_LEN, FinnyCache, ORDER, RMS_EPS, SECOND_LAYER, SymNetwork,
    evaluate_pre, evaluate_value, value_from_sums,
};
use super::fixtures::{assert_close, game, synthetic};
use crate::features::{MoveDelta, active_features};

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

/// The forward pass written a second way, in `f64`, from the definition.
///
/// The elementary symmetric polynomial `e_k` is the coefficient of `x^k` in
/// `∏ (1 + v_i x)`, so this builds it by multiplying out the product rather
/// than by Newton's identities from power sums — a different route to the same
/// number, which is what makes agreement mean something. The norm, the hidden
/// layers and the head are then the plain textbook forms.
fn reference_value(net: &SymNetwork, pos: &Position) -> f64 {
    let mut features = Vec::with_capacity(FEATURE_LEN);
    for color in [pos.side_to_move(), pos.side_to_move().flip()] {
        let active = active_features(pos, color);
        let mut elementary = vec![[0.0f64; ORDER + 1]; DIM];
        for (channel, e) in elementary.iter_mut().enumerate() {
            e[0] = 1.0;
            for &token in &active {
                let v = f64::from(net.embed[token as usize * DIM + channel]);
                for k in (1..=ORDER).rev() {
                    e[k] += e[k - 1] * v;
                }
            }
        }
        for k in 1..=ORDER {
            let block: Vec<f64> = elementary.iter().map(|e| e[k]).collect();
            let mean_square = block.iter().map(|v| v * v).sum::<f64>() / DIM as f64;
            let scale = 1.0 / (mean_square + f64::from(RMS_EPS)).sqrt();
            for (channel, v) in block.iter().enumerate() {
                let gamma = f64::from(net.gamma[(k - 1) * DIM + channel]);
                features.push(v * scale * gamma);
            }
        }
    }

    let dense = |w: &[f32], b: &[f32], x: &[f64]| -> Vec<f64> {
        b.iter()
            .enumerate()
            .map(|(row, b)| {
                let dot: f64 = w[row * x.len()..(row + 1) * x.len()]
                    .iter()
                    .zip(x)
                    .map(|(w, x)| f64::from(*w) * x)
                    .sum();
                (f64::from(*b) + dot).max(0.0)
            })
            .collect()
    };
    let mut head = dense(&net.w1, &net.b1, &features);
    if SECOND_LAYER {
        head = dense(&net.w2, &net.b2, &head);
    }
    let dot: f64 = net
        .value_w
        .iter()
        .zip(&head)
        .map(|(w, h)| f64::from(*w) * h)
        .sum();
    (f64::from(net.value_b) + dot).tanh()
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
        }
        let want = evaluate_value(net, &pos);
        let got = value_from_sums(net, &derived, pos.side_to_move());
        assert!((want - got).abs() < 1e-4, "value {got} against {want}");
        acc.clone_from(&derived);
    }
}
