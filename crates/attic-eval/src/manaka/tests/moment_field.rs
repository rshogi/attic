//! What a token adds to the second moment and to the square field, against the
//! definition: which entry of the triangle is which, and which slot of the field.

use super::super::field::touch_field;
use super::super::moment::touch_moment;
use super::super::shape::SQUARES;
use super::super::{FIELD, FIELD_LEN, HAS_FIELD, HAS_MOMENT, MOMENT, MOMENT_LEN, TOKENS};
use super::fixtures::{synthetic, upto};

/// A few tokens spread over the space: the first and the last row, one in the
/// middle, and two next to each other.
const TOKENS_TRIED: [u32; 5] = [0, 1, 1170, 1171, TOKENS as u32 - 1];

/// The position of entry `(a, b)`, `a ≤ b`, in the triangle read row by row: `a`
/// full rows of `m, m−1, …` come first, and `b − a` is the offset into row `a`.
/// Counted from this closed form and not from the running offset the
/// accumulator keeps, so that the two are two routes to one number.
fn triangle_index(a: usize, b: usize) -> usize {
    a * MOMENT - a * a.saturating_sub(1) / 2 + (b - a)
}

/// One token's outer product lands entry by entry where the upper triangle puts
/// it: `(0,0) (0,1) … (0,m−1) (1,1) … (m−1,m−1)`, each the product of the two
/// channels it names, and the diagonal once.
#[test]
fn a_token_adds_the_upper_triangle_of_its_outer_product() {
    if !HAS_MOMENT {
        return;
    }
    let net = synthetic();
    for token in TOKENS_TRIED {
        let mut sums = vec![0.0f64; MOMENT_LEN];
        touch_moment(&mut sums, net, token, 1.0);
        let row = &net.moment_embed[token as usize * MOMENT..][..MOMENT];
        for a in upto(MOMENT) {
            for b in a..MOMENT {
                let want = f64::from(row[a]) * f64::from(row[b]);
                assert_eq!(sums[triangle_index(a, b)], want, "token {token}, ({a},{b})");
            }
        }
        // The last entry of the triangle is the last diagonal one.
        assert_eq!(triangle_index(MOMENT - 1, MOMENT - 1), MOMENT_LEN - 1);
    }
}

/// A moment is a sum over tokens: it does not matter in which order they arrive,
/// and a token that leaves takes back exactly what it added. Several tokens go
/// in and two come out again, and the result is the sum of the three left.
#[test]
fn the_moment_is_a_sum_over_tokens() {
    if !HAS_MOMENT {
        return;
    }
    let net = synthetic();
    let mut sums = vec![0.0f64; MOMENT_LEN];
    for token in TOKENS_TRIED {
        touch_moment(&mut sums, net, token, 1.0);
    }
    touch_moment(&mut sums, net, TOKENS_TRIED[1], -1.0);
    touch_moment(&mut sums, net, TOKENS_TRIED[3], -1.0);

    let mut kept = vec![0.0f64; MOMENT_LEN];
    for token in [TOKENS_TRIED[4], TOKENS_TRIED[0], TOKENS_TRIED[2]] {
        touch_moment(&mut kept, net, token, 1.0);
    }
    for (entry, (got, want)) in sums.iter().zip(&kept).enumerate() {
        assert!(
            (got - want).abs() <= 1e-12,
            "entry {entry}: {got} against {want}"
        );
    }
}

/// A token adds its row of the field to the sum slot for slot: the entry for
/// square `s` and channel `c` is at `token · 81 · f + s · f + c` in the table and
/// at `s · f + c` in the sum.
#[test]
fn a_token_adds_its_row_of_the_field_slot_for_slot() {
    if !HAS_FIELD {
        return;
    }
    let net = synthetic();
    for token in TOKENS_TRIED {
        let mut sums = vec![0.0f64; FIELD_LEN];
        touch_field(&mut sums, net, token, 1.0);
        for square in upto(SQUARES) {
            for channel in upto(FIELD) {
                let at = token as usize * FIELD_LEN + square * FIELD + channel;
                let want = f64::from(net.field[at]);
                assert_eq!(
                    sums[square * FIELD + channel],
                    want,
                    "token {token}, square {square}, channel {channel}"
                );
            }
        }
        // And back out again: the sum is a sum.
        touch_field(&mut sums, net, token, -1.0);
        assert!(sums.iter().all(|v| *v == 0.0), "token {token}: not undone");
    }
}
