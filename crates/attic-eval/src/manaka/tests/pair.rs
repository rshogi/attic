//! The token-pair helper against the definition: which row a pair is.

use super::super::pair::pair_row;
use super::super::{HAS_PAIR, PAIR_ROWS, TOKENS};

/// The pair rows are the upper triangle read row by row, so walking it in that
/// order must count 0, 1, 2, … without a gap or a repeat, to exactly `C(2344, 2)`,
/// whichever way round the two tokens are given.
#[test]
fn pair_rows_number_the_upper_triangle_in_order() {
    let mut next = 0;
    for lo in 0..TOKENS as u32 {
        for hi in lo + 1..TOKENS as u32 {
            assert_eq!(pair_row(lo, hi), next, "({lo}, {hi})");
            assert_eq!(pair_row(hi, lo), next, "({hi}, {lo})");
            next += 1;
        }
    }
    assert_eq!(next, 2_745_996);
    // The table's own size, in a shape that has one; the others have no rows.
    assert_eq!(PAIR_ROWS, if HAS_PAIR { next } else { 0 });
}

/// The totals the derive adds to a pair sum, from the path the build runs, are the
/// plain ones to the last bit: the same rows, in the same two totals, in the same
/// order. Every count of other tokens from none to thirty-nine, so that an odd
/// last row and an empty set are both met, against tokens drawn over the whole
/// table.
#[test]
fn the_row_totals_are_the_plain_ones_to_the_last_bit() {
    use super::super::pair::row_totals_scalar;
    use super::fixtures::synthetic;
    use crate::tokens::FeatureIndex;

    if !HAS_PAIR {
        return;
    }
    let net = synthetic();
    let mut state = 5u64;
    let mut pick = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) as usize % TOKENS) as FeatureIndex
    };
    for count in 0..40 {
        let token = pick();
        let mut others = Vec::new();
        while others.len() < count {
            let other = pick();
            if other != token && !others.contains(&other) {
                others.push(other);
            }
        }
        let want = row_totals_scalar(&net.pair, token, &others);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            // SAFETY: the `cfg` is the build enabling AVX2.
            let got = unsafe { super::super::pair::row_totals_avx2(&net.pair, token, &others) };
            for (channel, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{count} others, channel {channel}"
                );
            }
        }
        // Against a sum of the rows in any order: within an f32 rounding of each row.
        for (channel, &w) in want.iter().enumerate() {
            let plain: f64 = others
                .iter()
                .map(|&o| {
                    f64::from(
                        net.pair
                            [super::super::pair::pair_row(token, o) * super::super::PAIR + channel],
                    )
                })
                .sum();
            assert!(
                (f64::from(w) - plain).abs() <= 1e-5 * (count as f64).max(1.0),
                "{count} others, channel {channel}: {w} against {plain}"
            );
        }
    }
}
