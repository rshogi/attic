//! A trained network carried down long random games by the search's own path.
//!
//! Gated on `ATTIC_SYM_MNW` naming a `ModelType=SYM…` file that matches
//! whichever `sym_*` shape the crate was built with, as `sym_network.rs` is, and
//! ignored by default: it is the slowest test here, and it needs a file none of
//! which live in the repository.

#![cfg(feature = "sym")]
// `PAIR`, `MEAN`, `MOMENT_LEN` and `FIELD_LEN` are zero in a shape without that
// block, and this file is compiled for every shape. A comparison with one or a
// range ending at one is then decided before the code runs, which is what these
// lints report; each use below is guarded by the constant it would be absurd
// without.
#![allow(
    clippy::absurd_extreme_comparisons,
    clippy::chunks_exact_to_as_chunks,
    clippy::reversed_empty_ranges
)]

mod sym_common;

use attic_eval::manaka::{
    Accumulator, DIM, FIELD_LEN, GROUPS, MEAN, MOMENT_LEN, ORDER, PAIR, evaluate_value,
    evaluate_value_with, load_network,
};
use attic_eval::{FinnyCache, MoveDelta, evaluate, evaluate_with};
use attic_state::{Color, Position};
use sym_common::{network_path, relative_drift};

/// The search's own path on a trained network: an accumulator carried down long
/// random games by `derive_into_cached` alone, each step from the move's
/// `MoveDelta`, against a rebuild of the same position.
///
/// The test above walks a fixed line with `update_after_move` for thirty plies.
/// This one is what the search does at every node, and it is random so that it
/// reaches what a fixed line does not: captures of promoted pieces, drops of
/// every kind of piece and of later copies, promotions, and a king that walks.
/// Three games of up to four hundred plies (a random game can end sooner, in
/// mate), none of them rebuilt on the way, so that drift would show as a bound
/// failing late in a game and not early. The search rebuilds at its root every
/// iteration and so never carries a sum nearly that far; this is the worst it
/// could be. `ATTIC_SYM_PLIES` sets the length.
///
/// A shape with a pair table is held to two more things, and prints how close
/// it came. Its sums are compared as the search sees them, and its log-sum-exp
/// channels are compared after the logarithm, which is where a large `exp θ`
/// that came and went would show as an error in a small sum left behind:
/// `docs/network.md` asks for under `1e-4` there. A shape with a second moment or
/// a square field has those sums held to the same bound and prints how close
/// they came too.
///
/// The bound is relative to the largest sum of its block, for the reason the
/// test above gives. It also compares what the search would be handed, the
/// integer score, which may differ by the rounding of the last bit of a float
/// and not by more.
///
/// Ignored because it needs a trained file, which is what
/// `ATTIC_SYM_MNW=/path/to/nn.bin cargo test -p attic-eval --features <shape>
/// --test sym_network -- --ignored` supplies.
#[test]
#[ignore = "needs ATTIC_SYM_MNW to name a trained nn.bin"]
fn the_search_path_stays_with_a_rebuild_over_long_random_games() {
    let path = network_path().expect("set ATTIC_SYM_MNW to a Manaka nn.bin");
    let net = load_network(&path).expect("the file loads");
    let mut cache = FinnyCache::new();
    let (mut worst_relative, mut worst_score) = (0.0f32, 0i32);
    let (mut worst_pair, mut worst_ln, mut worst_value) = (0.0f32, 0.0f32, 0.0f32);
    let (mut worst_moment, mut worst_field) = (0.0f32, 0.0f32);
    let mut moves = 0usize;
    let plies = std::env::var("ATTIC_SYM_PLIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);

    for seed in 1..=3u64 {
        let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03);
        let mut pos = Position::startpos();
        let mut acc = Accumulator::new();
        acc.refresh(&net, &pos);
        let mut derived = Accumulator::new();
        let mut legal = Vec::new();

        for ply in 0..plies {
            legal.clear();
            pos.generate_legal_all(&mut legal);
            if legal.is_empty() {
                break;
            }
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            let mv = legal[(rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as usize % legal.len()];

            let delta = MoveDelta::from_move(&pos, mv);
            let _ = pos.do_move(mv);
            Accumulator::derive_into_cached(&acc, &mut derived, &net, &pos, &delta, &mut cache);
            moves += 1;

            let mut rebuilt = Accumulator::new();
            rebuilt.refresh(&net, &pos);
            for color in [Color::Black, Color::White] {
                let a = derived.perspective(color);
                let b = rebuilt.perspective(color);
                for block in 0..GROUPS * ORDER {
                    let span = block * DIM..(block + 1) * DIM;
                    let scale = b[span.clone()].iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    let drift = a[span.clone()]
                        .iter()
                        .zip(&b[span])
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0f32, f32::max);
                    let relative = drift / scale.max(1.0);
                    worst_relative = worst_relative.max(relative);
                    assert!(
                        relative <= 1e-5,
                        "seed {seed}, ply {ply}, {color:?}, block {}: derived drifted {drift} \
                         from a rebuild, against a largest sum of {scale}",
                        block + 1
                    );
                }
                let (pa, pb) = (derived.pair(color), rebuilt.pair(color));
                for channel in 0..PAIR {
                    let drift = (pa[channel] - pb[channel]).abs();
                    let relative = drift / pb[channel].abs().max(1.0);
                    worst_pair = worst_pair.max(relative);
                    assert!(
                        relative <= 1e-5,
                        "seed {seed}, ply {ply}, {color:?}, pair channel {channel}: derived \
                         drifted {drift} from a rebuild, against a sum of {}",
                        pb[channel]
                    );
                    if channel >= MEAN {
                        let ln = (pa[channel].max(1e-6).ln() - pb[channel].max(1e-6).ln()).abs();
                        worst_ln = worst_ln.max(ln);
                        assert!(
                            ln < 1e-4,
                            "seed {seed}, ply {ply}, {color:?}, channel {channel}: ln drifted {ln}"
                        );
                    }
                }
            }
            // The moment and the field are carried the same way, and held to the
            // same bound.
            for color in [Color::Black, Color::White] {
                let moment = relative_drift(&derived.moment(color), &rebuilt.moment(color));
                let field = relative_drift(&derived.field(color), &rebuilt.field(color));
                worst_moment = worst_moment.max(moment);
                worst_field = worst_field.max(field);
                assert!(
                    moment <= 1e-5 && field <= 1e-5,
                    "seed {seed}, ply {ply}, {color:?}: moment drifted {moment}, field {field}"
                );
            }
            let value =
                (evaluate_value(&net, &pos) - evaluate_value_with(&net, &derived, &pos)).abs();
            worst_value = worst_value.max(value);
            let score = (evaluate_with(&net, &derived, &pos) - evaluate(&net, &pos)).abs();
            worst_score = worst_score.max(score);
            assert!(
                score <= 2,
                "seed {seed}, ply {ply}: scores differ by {score}"
            );
            acc.clone_from(&derived);
        }
    }
    eprintln!(
        "{moves} moves; worst drift {worst_relative:.3e} of the block's largest sum; \
         worst score difference {worst_score}"
    );
    if PAIR > 0 {
        eprintln!(
            "pair sums: worst relative drift {worst_pair:.3e}; log-sum-exp ln drift \
             {worst_ln:.3e}; value drift {worst_value:.3e}"
        );
    }
    if MOMENT_LEN > 0 {
        eprintln!("moment sums: worst relative drift {worst_moment:.3e}");
    }
    if FIELD_LEN > 0 {
        eprintln!("field sums: worst relative drift {worst_field:.3e}");
    }
}
