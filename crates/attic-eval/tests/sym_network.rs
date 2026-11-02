//! Loading and evaluating one of Manaka's own `MNW1` networks.
//!
//! Gated on `ATTIC_SYM_MNW` naming a file that matches whichever of
//! `sym_kp_128x2_512` / `sym_halfkp_128x2_512` the crate was built with. These
//! are training artefacts and none of them live in the repository.

#![cfg(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512"))]

use std::path::PathBuf;

use attic_eval::manaka::{
    Accumulator, DIM, FEATURE_LEN, HIDDEN, ORDER, TOKENS, evaluate_value, load_network,
};
use attic_state::{Color, Position};

fn network_path() -> Option<PathBuf> {
    std::env::var_os("ATTIC_SYM_MNW").map(PathBuf::from)
}

/// Every array is the length the header said it would be, and the file ends
/// exactly where the last one does — the loader rejects a trailing byte, so
/// reaching this assertion at all means the layout matched.
#[test]
fn a_trained_network_loads_at_the_declared_shape() {
    let Some(path) = network_path() else {
        eprintln!("set ATTIC_SYM_MNW to a Manaka .mnw to run this");
        return;
    };
    let net = load_network(&path).expect("the file loads");
    assert_eq!(net.embed.len(), TOKENS * DIM);
    assert_eq!(net.gamma.len(), ORDER * DIM);
    assert_eq!(net.w1.len(), HIDDEN * FEATURE_LEN);
    assert_eq!(net.b1.len(), HIDDEN);
    assert_eq!(net.value_w.len(), HIDDEN);
}

/// The forward pass runs on the opening position and lands inside `tanh`'s
/// range. A network that read its weights at the wrong stride would still
/// produce a number, but almost never a finite one in `[-1, 1]`.
#[test]
fn the_forward_pass_produces_a_bounded_value() {
    let Some(path) = network_path() else {
        return;
    };
    let net = load_network(&path).expect("the file loads");
    let pos = Position::startpos();
    let value = evaluate_value(&net, &pos);
    assert!(value.is_finite(), "value was {value}");
    assert!((-1.0..=1.0).contains(&value), "value was {value}");
}

/// The incremental path and a rebuild agree, over a played-out game.
///
/// The power sums are a sum over tokens, so `prev − removed + added` should be
/// the same set of numbers a from-scratch refresh produces. It is not the same
/// *bits*: f32 addition is not associative and the two arrive at each sum in a
/// different order, so this holds them to a tolerance rather than to equality.
/// Drift that accumulates would show as the bound failing deeper into the
/// game, which is why this walks thirty plies rather than one.
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
            let worst = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst < 1e-3,
                "ply {ply}, {color:?}: incremental drifted {worst} from a rebuild"
            );
        }
        let _ = undo;
        acc = derived;
    }
}

/// What the quantisation cost, on one position.
///
/// Both files come from the same weights, so any difference here is the
/// rounding and nothing else. Point `ATTIC_SYM_NN_F32` at the `--f32` export
/// of whatever `ATTIC_SYM_MNW` names. The assertion is loose on purpose — this
/// is a measurement, and the number it prints is the point; a ladder is what
/// says whether that number costs rating.
#[test]
fn the_quantisation_moves_the_value_only_slightly() {
    let (Some(q), Some(f)) = (
        network_path(),
        std::env::var_os("ATTIC_SYM_NN_F32").map(PathBuf::from),
    ) else {
        eprintln!("set ATTIC_SYM_MNW and ATTIC_SYM_NN_F32 to compare encodings");
        return;
    };
    let quantised = load_network(&q).expect("the quantised file loads");
    let float = load_network(&f).expect("the f32 file loads");
    let pos = Position::startpos();
    let a = evaluate_value(&quantised, &pos);
    let b = evaluate_value(&float, &pos);
    eprintln!("quantised={a:.6} f32={b:.6} delta={:.6}", a - b);
    assert!(
        (a - b).abs() < 0.05,
        "quantisation moved the value from {b} to {a}"
    );
}

/// The opening is not priced at a saturation.
///
/// `tanh` hides a great deal: weights read at the wrong stride tend to blow
/// the pre-activation sum up until the value pins to ±1, which
/// [`the_forward_pass_produces_a_bounded_value`] would still accept. A network
/// that is actually reading its own table lands well inside the range on a
/// position it was trained to call roughly even.
#[test]
fn the_opening_is_not_a_saturated_value() {
    let Some(path) = network_path() else {
        return;
    };
    let net = load_network(&path).expect("the file loads");
    let value = evaluate_value(&net, &Position::startpos());
    assert!(value.abs() < 0.999, "value saturated at {value}");
}
