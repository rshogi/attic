//! Loading and evaluating one of Manaka's own networks from a file.
//!
//! Gated on `ATTIC_SYM_MNW` naming a SYNN `nn.bin` that matches
//! whichever `synn_*` shape the crate was built with. These are training
//! artefacts and none of them live in the repository; `src/manaka/tests/` is
//! what covers every shape without one.
//!
//! The variable keeps the name it had when the networks were `SYM…` files: what it
//! names is the `nn.bin` that `manaka-train export-nnue` writes.

#![cfg(feature = "sym")]
// `PAIR`, `MOMENT_LEN` and `FIELD_LEN` are zero in a shape without that block,
// and this file is compiled for every shape. A comparison with one or a range
// ending at one is then decided before the code runs, which is what these three
// lints report; each use below is guarded by the constant it would be absurd
// without.
#![allow(
    clippy::absurd_extreme_comparisons,
    clippy::chunks_exact_to_as_chunks,
    clippy::reversed_empty_ranges
)]

mod sym_common;

use attic_eval::manaka::{
    Accumulator, DIM, FEATURE_LEN, FIELD, FIELD_LEN, HEAD, HIDDEN, HIDDEN2, MOMENT, MOMENT_LEN,
    ORDER, PAIR, PAIR_ROWS, TOKENS, evaluate_value, load_network,
};
use attic_state::{Color, Position};
use sym_common::{network_path, relative_drift};

/// Every array is the length the header said it would be, and the file ends
/// exactly where the last one does — the loader rejects a trailing byte, so
/// reaching this assertion at all means the layout matched.
#[test]
fn a_trained_network_loads_at_the_declared_shape() {
    let Some(path) = network_path() else {
        eprintln!("set ATTIC_SYM_MNW to a Manaka nn.bin to run this");
        return;
    };
    let net = load_network(&path).expect("the file loads");
    assert_eq!(net.embed.len(), TOKENS * DIM);
    assert_eq!(net.gamma.len(), ORDER * DIM);
    assert_eq!(net.w1.len(), HIDDEN * FEATURE_LEN);
    assert_eq!(net.b1.len(), HIDDEN);
    assert_eq!(net.w2.len(), HIDDEN2 * HIDDEN);
    assert_eq!(net.b2.len(), HIDDEN2);
    assert_eq!(net.value_w.len(), HEAD);
    assert_eq!(net.pair_gamma.len(), PAIR);
    assert_eq!(net.pair.len(), PAIR_ROWS * PAIR);
    assert_eq!(net.moment_gamma.len(), MOMENT_LEN);
    assert_eq!(net.moment_embed.len(), TOKENS * MOMENT);
    assert_eq!(net.field_gamma.len(), FIELD);
    assert_eq!(net.field.len(), TOKENS * FIELD_LEN);
}

/// The forward pass runs on the opening position and lands inside what bounds it.
/// A network that read its weights at the wrong stride would still produce a
/// number, but almost never a finite one inside the bound.
///
/// Every activation is clipped to `[0, 1]`, so the value is at most the sum of the
/// value row's magnitudes and the bias.
#[test]
fn the_forward_pass_produces_a_bounded_value() {
    let Some(path) = network_path() else {
        return;
    };
    let net = load_network(&path).expect("the file loads");
    let pos = Position::startpos();
    let value = evaluate_value(&net, &pos);
    assert!(value.is_finite(), "value was {value}");
    let reach = net.value_b.abs() + net.value_w.iter().map(|w| w.abs()).sum::<f32>();
    assert!(value.abs() <= reach, "value was {value}, reach {reach}");
}

/// The incremental path and a rebuild agree, over a played-out game.
///
/// The power sums are a sum over tokens, so `prev − removed + added` should be
/// the same set of numbers a from-scratch refresh produces. It is not the same
/// *bits*: f32 addition is not associative and the two arrive at each sum in a
/// different order, so this holds them to a tolerance rather than to equality.
/// Drift that accumulates would show as the bound failing deeper into the
/// game, which is why this walks thirty plies rather than one.
///
/// The tolerance is relative to the largest `|p_k|` in the same block, not an
/// absolute number. A trained network's sums are large — up to 6.9e3 in the
/// HalfKP shapes measured — and the rounding of an f32 sum is a fraction of its
/// own size, so an absolute `1e-3` fails on a network that is behaving: the
/// largest error seen there, 1.5e-3, was 5e-7 of its block's largest sum, a few
/// times the f32 epsilon of 1.2e-7. A real fault (a row added and not removed)
/// is of the size of a row, which is orders of magnitude over this.
#[test]
fn the_incremental_path_matches_a_rebuild() {
    let Some(path) = network_path() else {
        return;
    };
    let net = load_network(&path).expect("the file loads");
    let mut pos = Position::startpos();
    let mut acc = Accumulator::new();
    acc.refresh(&net, &pos);

    // Deterministic rather than random: the same game every run, so a failure
    // is reproducible from the ply number alone.
    let mut legal = Vec::new();
    for ply in 0..30 {
        legal.clear();
        pos.generate_legal_all(&mut legal);
        if legal.is_empty() {
            break;
        }
        let mv = legal[ply % legal.len()];

        let derived = acc.update_after_move(&net, &mut pos, mv);
        let undo = pos.do_move(mv);
        let mut rebuilt = Accumulator::new();
        rebuilt.refresh(&net, &pos);

        for color in [Color::Black, Color::White] {
            let a = derived.perspective(color);
            let b = rebuilt.perspective(color);
            for block in 0..ORDER {
                let span = block * DIM..(block + 1) * DIM;
                let scale = b[span.clone()].iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let worst = a[span.clone()]
                    .iter()
                    .zip(&b[span])
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    worst <= 2e-6 * scale.max(1.0),
                    "ply {ply}, {color:?}, block {}: incremental drifted {worst} from a \
                     rebuild, against a largest sum of {scale}",
                    block + 1
                );
            }
            let (pa, pb) = (derived.pair(color), rebuilt.pair(color));
            for channel in 0..PAIR {
                let worst = (pa[channel] - pb[channel]).abs();
                assert!(
                    worst <= 2e-6 * pb[channel].abs().max(1.0),
                    "ply {ply}, {color:?}, pair channel {channel}: incremental drifted \
                     {worst} from a rebuild, against a sum of {}",
                    pb[channel]
                );
            }
            for (what, a, b) in [
                (
                    "moment",
                    &derived.moment(color)[..],
                    &rebuilt.moment(color)[..],
                ),
                (
                    "field",
                    &derived.field(color)[..],
                    &rebuilt.field(color)[..],
                ),
            ] {
                let worst = relative_drift(a, b);
                assert!(
                    worst <= 2e-6,
                    "ply {ply}, {color:?}, {what}: incremental drifted {worst} from a rebuild"
                );
            }
        }
        let _ = undo;
        acc = derived;
    }
}
