//! What a table over pairs of tokens needs from a token's number: which row a
//! pair is, and the add-or-subtract of the rows of one token against all the
//! others present.
//!
//! The numbering itself is `tokens.rs`'s; this reads it and does not extend it,
//! so that the one file that knows how a token is numbered stays the only one.

use crate::tokens::FeatureIndex;

use super::accumulator::PairSum;
use super::{PAIR, PAIR_ROWS, SymNetwork, TOKENS};

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

/// The group of channels the AVX2 path adds at once: a register of `f32`.
const LANES: usize = 8;

const _: () = assert!(
    PAIR.is_multiple_of(LANES),
    "pair channels come in whole groups of eight"
);

/// Add (`sign = 1`) or subtract (`sign = −1`) the pair rows of `token` against
/// every one of `others`, into `sums`.
///
/// `others` must not contain `token`. The caller keeps that by taking a token
/// out of the set before subtracting its pairs and putting it in after adding
/// them, which is why the accumulator removes before it touches a row.
///
/// The rows are added up in `f32` first, at most thirty-nine of them, and that
/// total goes into the `f64` sum once. Adding each row to the `f64` sum as it was
/// read made every row wait for the one before it to be written back, and that
/// chain was most of the time a pair16 net spent here. The running sum stays
/// `f64`, so what a derive leaves in it is still one rounding per token touched,
/// as [`Accumulator`](super::Accumulator) describes; only that rounding is now of
/// the token's total and not of each row.
///
/// The rows go to two totals in turn, even-numbered rows to one and odd to the
/// other, a last odd row to the first, and the two are added at the end. Two
/// totals let two rows be added at once. [`row_totals_scalar`] is that order
/// written plainly, and the AVX2 path is held to it to the last bit.
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
    let totals = row_totals(&net.pair, token, others);
    for (sum, total) in sums.iter_mut().zip(totals) {
        *sum += sign * PairSum::from(total);
    }
}

/// The rows of `token` against each of `others`, added up channel by channel in
/// the order [`touch_pairs`] gives.
#[inline]
fn row_totals(table: &[f32], token: FeatureIndex, others: &[FeatureIndex]) -> [f32; PAIR] {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: the `cfg` above is the build enabling AVX2 for the CPU it runs on.
        unsafe { row_totals_avx2(table, token, others) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        row_totals_scalar(table, token, others)
    }
}

/// [`row_totals`] in plain `f32`: the reference the AVX2 path is held to.
#[allow(dead_code)]
pub(super) fn row_totals_scalar(
    table: &[f32],
    token: FeatureIndex,
    others: &[FeatureIndex],
) -> [f32; PAIR] {
    let row = |other: FeatureIndex| -> &[f32; PAIR] {
        let at = pair_row(token, other) * PAIR;
        table[at..at + PAIR]
            .try_into()
            .expect("a pair row is PAIR long")
    };
    let (mut even, mut odd) = ([0.0f32; PAIR], [0.0f32; PAIR]);
    let (twos, last) = others.as_chunks::<2>();
    for &[a, b] in twos {
        for (total, value) in even.iter_mut().zip(row(a)) {
            *total += value;
        }
        for (total, value) in odd.iter_mut().zip(row(b)) {
            *total += value;
        }
    }
    for &other in last {
        for (total, value) in even.iter_mut().zip(row(other)) {
            *total += value;
        }
    }
    std::array::from_fn(|channel| even[channel] + odd[channel])
}

/// [`row_totals_scalar`] with eight channels of a total in one register.
///
/// The table is [`Aligned64`](crate::Aligned64) and a pair16 row is sixty-four
/// bytes, so a row is one cache line; the loads are unaligned all the same, so a
/// table of another width is still read correctly.
///
/// # Safety
/// The running CPU must support AVX2. The `cfg` that compiles this in names it,
/// and the build is for the host CPU, so it does.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn row_totals_avx2(
    table: &[f32],
    token: FeatureIndex,
    others: &[FeatureIndex],
) -> [f32; PAIR] {
    use std::arch::x86_64::*;

    const GROUPS: usize = PAIR / LANES;
    // Every row read is `pair_row(token, other)`, which is below `PAIR_ROWS`, and
    // the table is that many rows long.
    assert_eq!(table.len(), PAIR_ROWS * PAIR);
    let base = table.as_ptr();
    let row = |other: FeatureIndex| -> *const f32 {
        // SAFETY: the row starts `pair_row · PAIR` in, and the whole row is in the
        // table (asserted above).
        unsafe { base.add(pair_row(token, other) * PAIR) }
    };
    let mut even = [_mm256_setzero_ps(); GROUPS];
    let mut odd = [_mm256_setzero_ps(); GROUPS];
    // Group by group of eight channels: `group · 8 + 8 <= PAIR`, inside the row.
    // The loads are unaligned.
    let add_row = |totals: &mut [__m256; GROUPS], row: *const f32| {
        for (group, total) in totals.iter_mut().enumerate() {
            // SAFETY: see above; the row is in the table.
            *total = _mm256_add_ps(*total, unsafe { _mm256_loadu_ps(row.add(group * LANES)) });
        }
    };
    let (twos, last) = others.as_chunks::<2>();
    for &[a, b] in twos {
        add_row(&mut even, row(a));
        add_row(&mut odd, row(b));
    }
    for &other in last {
        add_row(&mut even, row(other));
    }
    let mut out = [0.0f32; PAIR];
    for ((chunk, e), o) in out.as_chunks_mut::<LANES>().0.iter_mut().zip(even).zip(odd) {
        // SAFETY: `chunk` is eight `f32`s long; the store is unaligned.
        unsafe { _mm256_storeu_ps(chunk.as_mut_ptr(), _mm256_add_ps(e, o)) };
    }
    out
}
