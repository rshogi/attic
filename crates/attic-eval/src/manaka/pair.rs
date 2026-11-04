//! What a table over pairs of tokens needs from a token's number: which row a
//! pair is, which side owns a token, and the add-or-subtract of the rows of one
//! token against all the others present.
//!
//! The numbering itself is `tokens.rs`'s; this reads it and does not extend it,
//! so that the one file that knows how a token is numbered stays the only one.

use crate::tokens::{BOARD_ROWS, FeatureIndex, HAND_SIDE, SQUARES};

use super::accumulator::PairSum;
use super::{PAIR, SymNetwork, TOKENS};

/// The row of the pair `{a, b}` in the pair table.
///
/// The upper triangle in lexicographic order, `(0,1) (0,2) … (0,2343) (1,2) …`:
/// the pair with the lower token `lo` and the higher `hi` is `lo` full rows of
/// the triangle in, `TOKENS − 1`, `TOKENS − 2`, … long, then `hi − lo − 1` into
/// its own. The division is exact, since `lo` and `2 · TOKENS − lo − 1` are not
/// both odd, and the order of the arguments does not matter.
///
/// # Panics
/// Panics, in a debug build, if the two tokens are the same one, which a
/// position never has: a token is a distinct row in a given frame.
#[inline]
pub(super) fn pair_row(a: FeatureIndex, b: FeatureIndex) -> usize {
    debug_assert!(a != b, "a token does not pair with itself");
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    let (lo, hi) = (lo as usize, hi as usize);
    lo * (2 * TOKENS - lo - 1) / 2 + (hi - lo - 1)
}

/// Whether the opponent of the frame owns this token: `0` for the frame's own
/// side, `1` for the other. The ownership bit is in the number already, since a
/// token is numbered relative to the frame it is read from.
///
/// This is `split`'s group: the power sums of the frame's own pieces and of the
/// other side's are kept apart. Hand tokens are owned like any other.
#[inline]
pub(super) fn theirs(token: FeatureIndex) -> usize {
    let token = token as usize;
    if token < BOARD_ROWS {
        // `(kind · 2 + theirs) · 81 + square`.
        (token / SQUARES) % 2
    } else {
        // `BOARD_ROWS + theirs · 38 + offset`.
        (token - BOARD_ROWS) / HAND_SIDE
    }
}

/// Add (`sign = 1`) or subtract (`sign = −1`) the pair rows of `token` against
/// every one of `others`, into `sums`.
///
/// `others` must not contain `token`. The caller keeps that by taking a token
/// out of the set before subtracting its pairs and putting it in after adding
/// them, which is why the accumulator removes before it touches a row.
#[inline]
pub(super) fn touch_pairs(
    sums: &mut [PairSum],
    net: &SymNetwork,
    token: FeatureIndex,
    others: &[FeatureIndex],
    sign: PairSum,
) {
    if PAIR == 0 {
        return;
    }
    let sums: &mut [PairSum; PAIR] = sums.try_into().expect("a pair sum is PAIR long");
    for &other in others {
        let at = pair_row(token, other) * PAIR;
        let row: &[f32; PAIR] = net.pair[at..at + PAIR]
            .try_into()
            .expect("a pair row is PAIR long");
        for (sum, value) in sums.iter_mut().zip(row) {
            *sum += sign * PairSum::from(*value);
        }
    }
}
