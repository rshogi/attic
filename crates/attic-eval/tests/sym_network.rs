//! Loading and evaluating one of Manaka's own networks from a file.
//!
//! Gated on `ATTIC_SYM_MNW` naming a `ModelType=SYM…` file that matches
//! whichever `sym_*` shape the crate was built with. These are training
//! artefacts and none of them live in the repository; `src/manaka/tests/` is
//! what covers every shape without one.
//!
//! The variable keeps its name from when the file was Manaka's `MNW1`: what it
//! names now is the `nn.bin` that `manaka-train export-nnue` writes.

#![cfg(feature = "sym")]

use std::path::PathBuf;

use attic_eval::manaka::{
    Accumulator, DIM, FEATURE_LEN, HEAD, HIDDEN, HIDDEN2, ORDER, TOKENS, evaluate_value,
    load_network,
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
                    "ply {ply}, {color:?}, order {}: incremental drifted {worst} from a \
                     rebuild, against a largest sum of {scale}",
                    block + 1
                );
            }
        }
        let _ = undo;
        acc = derived;
    }
}

/// What the quantisation cost, on one position.
///
/// Both files come from the same weights, so any difference here is the
/// rounding and nothing else. Point `ATTIC_SYM_NN_F32` at the `--f32` export
/// of whatever `ATTIC_SYM_MNW` names.
///
/// This measures and does not judge unless asked to, because the answer depends
/// on the encoding. The first one (`ModelType=SYM`) fails any bound worth
/// stating on the trained nets measured: their dense weights reach 28 where an
/// `i8` at scale 64 stops at 1.98, and the value row sums terms hundreds large
/// into a result near one, so a clamp anywhere in it moves the answer. A
/// `SYMscaled` file keeps within 0.03 of the f32 one on the same nets over 4,000
/// positions. `ATTIC_SYM_QUANT_TOLERANCE` sets a bound for a file that is
/// expected to meet one.
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
    assert!(a.is_finite() && b.is_finite());
    if let Some(tolerance) = std::env::var("ATTIC_SYM_QUANT_TOLERANCE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
    {
        assert!(
            (a - b).abs() < tolerance,
            "quantisation moved the value from {b} to {a}"
        );
    }
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
