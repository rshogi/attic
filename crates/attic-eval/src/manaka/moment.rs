//! What the second moment needs from a token: its share of the upper triangle of
//! `M = Σ v vᵀ`, where `v` is the token's row of `moment_embed`.
//!
//! Like the power sums, and unlike the pair table, the moment is a sum over
//! tokens and nothing else: a token that arrives adds its outer product and one
//! that leaves subtracts the same, whichever tokens are present, so there is no
//! order the accumulator has to respect.

use crate::tokens::FeatureIndex;

use super::accumulator::MomentSum;
use super::{MOMENT, MOMENT_LEN, SymNetwork};

/// Add (`sign = 1`) or subtract (`sign = −1`) the outer product of `token`'s
/// moment row with itself into `sums`.
///
/// Only the upper triangle is kept, row by row — `(0,0) (0,1) … (0,m−1) (1,1) …
/// (m−1,m−1)` — because that is the order the head's block and `moment_gamma` are
/// in, and the entries below the diagonal would be the same numbers again. The
/// diagonal is in once.
///
/// The product of two `f32` is exact in `f64`, so what is rounded is the sum and
/// not the term.
#[inline]
pub(super) fn touch_moment(
    sums: &mut [MomentSum],
    net: &SymNetwork,
    token: FeatureIndex,
    sign: MomentSum,
) {
    if MOMENT == 0 {
        return;
    }
    let sums: &mut [MomentSum; MOMENT_LEN] = sums.try_into().expect("a moment is MOMENT_LEN long");
    let at = token as usize * MOMENT;
    let row: &[f32; MOMENT] = net.moment_embed[at..at + MOMENT]
        .try_into()
        .expect("a moment row is MOMENT long");
    let mut start = 0;
    for (a, &lead) in row.iter().enumerate() {
        let lead = sign * MomentSum::from(lead);
        let width = MOMENT - a;
        for (sum, &value) in sums[start..start + width].iter_mut().zip(&row[a..]) {
            *sum += lead * MomentSum::from(value);
        }
        start += width;
    }
}
