//! What a table over pairs of tokens needs from a token's number: which row a
//! pair is, and the add-or-subtract of the rows of one token against all the
//! others present, in both frames at once.
//!
//! The numbering itself is `tokens.rs`'s; this reads it and does not extend it,
//! so that the one file that knows how a token is numbered stays the only one.

use crate::tokens::FeatureIndex;

use super::accumulator::PairSum;
use super::{PAIR, PAIR_ROWS, PairTable, SymNetwork, TOKENS};

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

/// Halves in one row of the table: both frames' channels.
const ROW: usize = 2 * PAIR;

const _: () = assert!(
    PAIR.is_multiple_of(LANES),
    "pair channels come in whole groups of eight"
);

/// Add (`sign = 1`) or subtract (`sign = −1`) the pair rows of `token` against
/// every one of `others`, into both frames' sums.
///
/// `token` and `others` are numbered as the black frame numbers them, and
/// `sums` is `[black, white]`. A row of the table holds both frames' channels for
/// the pair (see [`PairTable`]), so the white frame's sum comes from the same
/// reads. That is the white frame's own computation and not an approximation of
/// it: the white frame's tokens are the black frame's through `other_frame`, in
/// the same order (`tokens.rs`), so the rows it would read and the order it
/// would add them in are the ones read here.
///
/// `others` must not contain `token`. The caller keeps that by taking a token
/// out of the set before subtracting its pairs and putting it in after adding
/// them, which is why the accumulator removes before it touches a row.
///
/// The rows are added up in `f32` first, at most thirty-nine of them, and that
/// total goes into the `f64` sum once, so the `f64` sum takes one rounding per
/// token touched. The rows go to two totals in turn, even-numbered rows to one
/// and odd to the other, a last odd row to the first, and the two are added at
/// the end; two totals let two rows be added at once. [`row_totals_scalar`] is
/// that order written plainly, and the AVX2 path is held to it to the last bit.
#[inline]
pub(super) fn touch_pairs(
    sums: &mut [Vec<PairSum>; 2],
    net: &SymNetwork,
    token: FeatureIndex,
    others: &[FeatureIndex],
    sign: PairSum,
) {
    if PAIR == 0 {
        return;
    }
    let totals = row_totals(&net.pair, token, others);
    for (frame, sum) in sums.iter_mut().enumerate() {
        let sum: &mut [PairSum; PAIR] = sum
            .as_mut_slice()
            .try_into()
            .expect("a pair sum is PAIR long");
        for (s, total) in sum
            .iter_mut()
            .zip(&totals[frame * PAIR..(frame + 1) * PAIR])
        {
            *s += sign * PairSum::from(*total);
        }
    }
}

/// The rows of `token` against each of `others`, both frames' channels, added up
/// in the order [`touch_pairs`] gives.
#[inline]
fn row_totals(table: &PairTable, token: FeatureIndex, others: &[FeatureIndex]) -> [f32; ROW] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "f16c"
    ))]
    {
        // SAFETY: the `cfg` above is the build enabling AVX2 and F16C for the CPU
        // it runs on.
        unsafe { row_totals_avx2(table, token, others) }
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "f16c"
    )))]
    {
        row_totals_scalar(table, token, others)
    }
}

/// [`row_totals`] in plain `f32`, a half widened by the reader's own conversion:
/// the reference the AVX2 path is held to.
#[cfg(any(
    test,
    not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "f16c"
    ))
))]
pub(super) fn row_totals_scalar(
    table: &PairTable,
    token: FeatureIndex,
    others: &[FeatureIndex],
) -> [f32; ROW] {
    use super::reader::half_to_f32;

    let halves = table.halves();
    let row = |other: FeatureIndex| -> &[u16] {
        let at = pair_row(token, other) * ROW;
        &halves[at..at + ROW]
    };
    let widen = |half: u16| half_to_f32(half).expect("the reader found every half finite");
    let (mut even, mut odd) = ([0.0f32; ROW], [0.0f32; ROW]);
    let (twos, last) = others.as_chunks::<2>();
    for &[a, b] in twos {
        for (total, &half) in even.iter_mut().zip(row(a)) {
            *total += widen(half);
        }
        for (total, &half) in odd.iter_mut().zip(row(b)) {
            *total += widen(half);
        }
    }
    for &other in last {
        for (total, &half) in even.iter_mut().zip(row(other)) {
            *total += widen(half);
        }
    }
    std::array::from_fn(|channel| even[channel] + odd[channel])
}

/// [`row_totals_scalar`] with eight channels of a total in one register, each
/// group of eight halves widened by `vcvtph2ps`, which is exact as the scalar
/// widening is.
///
/// A pair16 row is thirty-two halves, sixty-four bytes, and the table is
/// [`Aligned64`](crate::Aligned64), so a row is one cache line. The loads are
/// unaligned all the same, so a table of another width is still read correctly.
///
/// # Safety
/// The running CPU must support AVX2 and F16C. The `cfg` that compiles this in
/// names both, and the build is for the host CPU, so it does.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "f16c"
))]
#[target_feature(enable = "avx2,f16c")]
pub(super) unsafe fn row_totals_avx2(
    table: &PairTable,
    token: FeatureIndex,
    others: &[FeatureIndex],
) -> [f32; ROW] {
    use std::arch::x86_64::*;

    const GROUPS: usize = ROW / LANES;
    let halves = table.halves();
    // Every row read is `pair_row(token, other)`, which is below `PAIR_ROWS`, and
    // the table is that many rows long.
    assert_eq!(halves.len(), PAIR_ROWS * ROW);
    let base = halves.as_ptr();
    let row = |other: FeatureIndex| -> *const u16 {
        // SAFETY: the row starts `pair_row · ROW` in, and the whole row is in the
        // table (asserted above).
        unsafe { base.add(pair_row(token, other) * ROW) }
    };
    let mut even = [_mm256_setzero_ps(); GROUPS];
    let mut odd = [_mm256_setzero_ps(); GROUPS];
    // Group by group of eight halves: `group · 8 + 8 <= ROW`, inside the row.
    let add_row = |totals: &mut [__m256; GROUPS], row: *const u16| {
        for (group, total) in totals.iter_mut().enumerate() {
            // SAFETY: see above; the row is in the table, and the load is unaligned.
            let eight = unsafe { _mm_loadu_si128(row.add(group * LANES).cast()) };
            *total = _mm256_add_ps(*total, _mm256_cvtph_ps(eight));
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
    let mut out = [0.0f32; ROW];
    for ((chunk, e), o) in out.as_chunks_mut::<LANES>().0.iter_mut().zip(even).zip(odd) {
        // SAFETY: `chunk` is eight `f32`s long; the store is unaligned.
        unsafe { _mm256_storeu_ps(chunk.as_mut_ptr(), _mm256_add_ps(e, o)) };
    }
    out
}
