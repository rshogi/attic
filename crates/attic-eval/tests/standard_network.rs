//! Loading the two standard YaneuraOu shapes from a real `nn.bin`.
//!
//! Gated on an environment variable naming the file, the same way `loader.rs`'s
//! `ATTIC_HALFKA2304_NN` unit test is: these networks are large and none of
//! them live in the repository.
//!
//! One variable per shape, `ATTIC_KP256_NN` and `ATTIC_HALFKP256_NN`, rather
//! than one for both: a single name means whichever shape is not the one the
//! crate was built with points the loader at a file of the wrong size, and
//! running the shapes in a loop with the variable exported once fails on every
//! shape but one.

#![cfg(any(feature = "nnue_kp_256x2_32_32", feature = "nnue_halfkp_256x2_32_32"))]

use std::path::PathBuf;

use attic_eval::{HIDDEN_SIZE, LAYER_STACKS, NUM_FEATURES, load_network_with_warnings};

#[cfg(feature = "nnue_kp_256x2_32_32")]
const NETWORK_ENV: &str = "ATTIC_KP256_NN";
#[cfg(feature = "nnue_halfkp_256x2_32_32")]
const NETWORK_ENV: &str = "ATTIC_HALFKP256_NN";

fn network_path() -> Option<PathBuf> {
    std::env::var_os(NETWORK_ENV).map(PathBuf::from)
}

/// The whole point of the exercise: a file YaneuraOu wrote is read back with
/// nothing to complain about.
///
/// Warnings are what a wrong hash produces — the loader does not stop for one —
/// so asserting they are absent is the only way this test can fail usefully.
#[test]
fn a_real_network_loads_without_warnings() {
    let Some(path) = network_path() else {
        eprintln!("set {NETWORK_ENV} to a matching nn.bin to run this");
        return;
    };
    let (net, warnings) = load_network_with_warnings(&path).expect("the file loads");
    assert!(warnings.is_empty(), "hash mismatches: {warnings:?}");
    assert_eq!(net.stacks.len(), LAYER_STACKS, "one stack, no bucketing");
    assert_eq!(net.ft_biases.len(), HIDDEN_SIZE);
    assert_eq!(net.ft_weights.len(), HIDDEN_SIZE * NUM_FEATURES);
}

/// Every parameter array is 64-byte aligned, because the SIMD kernels load
/// from them without checking.
#[test]
fn every_array_is_aligned() {
    let Some(path) = network_path() else {
        return;
    };
    let (net, _) = load_network_with_warnings(&path).expect("the file loads");
    let aligned = |ptr: *const u8| (ptr as usize).is_multiple_of(64);
    assert!(aligned(net.ft_biases.as_ptr() as *const u8));
    assert!(aligned(net.ft_weights.as_ptr() as *const u8));
    for stack in &net.stacks {
        assert!(aligned(stack.fc_0_biases().as_ptr() as *const u8));
        assert!(aligned(stack.fc_0_weights().as_ptr() as *const u8));
        assert!(aligned(stack.fc_1_biases().as_ptr() as *const u8));
        assert!(aligned(stack.fc_1_weights().as_ptr() as *const u8));
        assert!(aligned(stack.fc_2_biases().as_ptr() as *const u8));
        assert!(aligned(stack.fc_2_weights().as_ptr() as *const u8));
    }
}

/// The five positions YaneuraOu's own `e` command prints, and what it prints.
///
/// `e` (`usi.cpp`, the single letter — `eval` routes to `trace_eval()`, whose
/// print line is commented out) reports `Eval::evaluate(pos)` for the position
/// last set. Run against `YaneuraOu-by-gcc` built as `NNUE KP256 9.70git
/// 64AVX2` with the 2019 KP256 distribution, it answers 125, 345, 421, -542
/// and -817 for these five. They are the start position followed by the four
/// `benchmark.cpp` defaults.
///
/// Two loader bugs lived behind the absence of this assertion. The feature
/// transformer's parameters were being doubled on load for every shape, when
/// the doubling belongs to the element-wise-multiply transformer alone; and
/// `MoveDelta::from_move` mirrored its squares by the `hm` rule that this
/// shape's `king_code` does not use. Nothing here checked a number, so both
/// showed up only as an engine that played differently.
#[cfg(feature = "nnue_kp_256x2_32_32")]
#[test]
fn kp256_static_evals_match_yaneuraou() {
    use attic_eval::evaluate;
    use attic_state::parse_sfen;

    const EXPECTED: [(&str, i32); 5] = [
        (
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            125,
        ),
        (
            "lnsgkgsnl/1r7/p1ppp1bpp/1p3pp2/7P1/2P6/PP1PPPP1P/1B3S1R1/LNSGKG1NL b - 9",
            345,
        ),
        (
            "l4S2l/4g1gs1/5p1p1/pr2N1pkp/4Gn3/PP3PPPP/2GPP4/1K7/L3r+s2L w BS2N5Pb 1",
            421,
        ),
        (
            "6n1l/2+S1k4/2lp4p/1np1B2b1/3PP4/1N1S3rP/1P2+pPP+p1/1p1G5/3KG2r1 b GSN2L4Pgs2p 1",
            -542,
        ),
        (
            "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1",
            -817,
        ),
    ];

    let Some(path) = network_path() else {
        eprintln!("set {NETWORK_ENV} to the KP256 nn.bin to run this");
        return;
    };
    let (net, _) = load_network_with_warnings(&path).expect("the file loads");
    for (sfen, expected) in EXPECTED {
        let pos = parse_sfen(sfen).expect("sfen parses");
        assert_eq!(evaluate(&net, &pos), expected, "static eval of `{sfen}`");
    }
}

/// The incremental accumulator must equal a full refresh, on every ply of a
/// long random game, for both perspectives.
///
/// [`crate::incremental_parity`]'s equivalent is gated to the SFNN shapes, so
/// the standard ones had no such gate — and a delta that mirrored its squares
/// when this shape's feature set does not went unnoticed for as long as the
/// search kept agreeing with itself. The first divergence there was a capturing
/// pawn push 56 plies deep, with the perspective's king on file 5 or higher;
/// 40 games of 60 plies is enough walking to reach that kind of position many
/// times over.
#[test]
fn incremental_update_equals_refresh() {
    use attic_eval::{Accumulator, MoveDelta, evaluate, evaluate_with};
    use attic_state::{Color, STARTPOS_SFEN, format_sfen, format_usi_move, parse_sfen};

    let Some(path) = network_path() else {
        eprintln!("set {NETWORK_ENV} to a matching nn.bin to run this");
        return;
    };
    let (net, _) = load_network_with_warnings(&path).expect("the file loads");

    // xorshift64*, so the walk is the same one on every machine and no `rand`
    // dependency is pulled in for a test.
    let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    for _ in 0..40 {
        let mut pos = parse_sfen(STARTPOS_SFEN).expect("startpos parses");
        let mut acc = Accumulator::new();
        acc.refresh(&net, &pos);
        for _ in 0..60 {
            let mut moves = Vec::new();
            pos.generate_legal_all(&mut moves);
            if moves.is_empty() {
                break;
            }
            let mv = moves[(next() % moves.len() as u64) as usize];
            let delta = MoveDelta::from_move(&pos, mv);
            let _undo = pos.do_move(mv);

            let mut derived = Accumulator::new();
            Accumulator::derive_into(&acc, &mut derived, &net, &pos, &delta);
            let mut fresh = Accumulator::new();
            fresh.refresh(&net, &pos);
            for c in [Color::Black, Color::White] {
                assert_eq!(
                    derived.perspective(c),
                    fresh.perspective(c),
                    "{c:?} half after {} in {}",
                    format_usi_move(mv),
                    format_sfen(&pos),
                );
            }
            assert_eq!(
                evaluate_with(&net, &derived, &pos),
                evaluate(&net, &pos),
                "eval after {} in {}",
                format_usi_move(mv),
                format_sfen(&pos),
            );
            acc = derived;
        }
    }
}
