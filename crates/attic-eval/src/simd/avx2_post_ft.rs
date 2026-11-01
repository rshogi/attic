//! AVX2 post-feature-transformer element-wise kernels.
//!
//! The 256-bit counterparts of [`crate::simd::avx512_post_ft`]'s element-wise
//! stages. Unlike the feature-transformer kernels ([`crate::simd::avx2`]),
//! these are not a mechanical halving, because AVX-512's narrowing converts
//! (`cvtepi16_epi8` and friends) have no AVX2 equivalent.
//!
//! ## The pack/permute idiom
//!
//! `_mm256_packus_epi16(a, b)` packs **within each 128-bit lane**, so the result
//! reads `[a_lo, b_lo, a_hi, b_hi]` in 64-bit groups — not the straight
//! concatenation `cvtepi16_epi8` produces. Every use here follows the pack with
//! `_mm256_permute4x64_epi64::<0b11_01_10_00>()`, which reorders those groups to
//! `[a_lo, a_hi, b_lo, b_hi]`. Forgetting it yields plausible-but-wrong bytes
//! that no coarse test would catch, which is why the parity tests below compare
//! against the scalar baseline element by element.
//!
//! ## Why saturation is safe here
//!
//! `packus` saturates to `[0, 255]` where `cvtepi16_epi8` truncates. They agree
//! only while the values already fit. They do: `ewm_one_perspective`'s product
//! is `(sum0 * sum1) >> 9` with both inputs clamped to `EWM_CLAMP = 254`, so its
//! maximum is `(254 * 254) >> 9 = 126`, and the clipped-ReLU stages clamp to
//! `[0, 127]` before packing.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::simd::scalar_post_ft;
use crate::types::HIDDEN_SIZE;

const I16_LANES: usize = 16;
const EWM_HALF: usize = HIDDEN_SIZE / 2;
const EWM_CLAMP_I16: i16 = 127 * 2;
// Same mulhi trick as the AVX-512 path: mulhi(sum0 << 7, sum1) == (sum0 * sum1) >> 9.
const EWM_PRESHIFT: i32 = 7;
const WEIGHT_SCALE_BITS: u32 = 6;

/// Reorder `packus`/`packs` output from lane-local `[a_lo, b_lo, a_hi, b_hi]`
/// into the straight `[a_lo, a_hi, b_lo, b_hi]` a narrowing convert would give.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn fix_pack_lanes(packed: __m256i) -> __m256i {
    _mm256_permute4x64_epi64::<0b11_01_10_00>(packed)
}

/// Pairwise element-wise multiply for one perspective.
///
/// Lane `j` pairs with lane `j + HIDDEN_SIZE/2`; both are clamped to
/// `[0, EWM_CLAMP]`, multiplied, and shifted right by 9. Reads `HIDDEN_SIZE`
/// `i16` from `half`, writes `HIDDEN_SIZE / 2` `u8` to `out`.
///
/// # Safety
/// The running CPU must support `avx2`. `half` must be readable for
/// `HIDDEN_SIZE` `i16` and `out` writable for `EWM_HALF` bytes.
#[target_feature(enable = "avx2")]
pub unsafe fn ewm_one_perspective_ptr(half: *const i16, out: *mut u8) {
    let zero = _mm256_setzero_si256();
    let cap = _mm256_set1_epi16(EWM_CLAMP_I16);

    // Two 16-lane halves per iteration so one `packus` fills a whole 32-byte
    // store, matching the AVX-512 kernel's store granularity.
    let chunks = EWM_HALF / (I16_LANES * 2);
    for chunk in 0..chunks {
        let off = chunk * I16_LANES * 2;

        let mut packed = [zero; 2];
        for (i, slot) in packed.iter_mut().enumerate() {
            let at = off + i * I16_LANES;
            // SAFETY: at + I16_LANES <= EWM_HALF, and the second load is
            // EWM_HALF further into a buffer of 2 * EWM_HALF i16.
            let (in0, in1) = unsafe {
                (
                    _mm256_loadu_si256(half.add(at).cast::<__m256i>()),
                    _mm256_loadu_si256(half.add(at + EWM_HALF).cast::<__m256i>()),
                )
            };
            let sum0 = _mm256_max_epi16(_mm256_min_epi16(in0, cap), zero);
            let sum1 = _mm256_max_epi16(_mm256_min_epi16(in1, cap), zero);
            let shifted = _mm256_slli_epi16::<EWM_PRESHIFT>(sum0);
            *slot = _mm256_mulhi_epi16(shifted, sum1);
        }

        // SAFETY: guarded by this function's own target_feature.
        let bytes = unsafe { fix_pack_lanes(_mm256_packus_epi16(packed[0], packed[1])) };
        // SAFETY: off + 32 <= EWM_HALF keeps the 32-byte store in range.
        unsafe { _mm256_storeu_si256(out.add(off).cast::<__m256i>(), bytes) };
    }
}

/// [`ewm_one_perspective_ptr`] over slices.
///
/// # Safety
/// The running CPU must support `avx2`.
#[target_feature(enable = "avx2")]
pub unsafe fn ewm_one_perspective(half: &[i16], out: &mut [u8]) {
    debug_assert_eq!(half.len(), HIDDEN_SIZE);
    debug_assert_eq!(out.len(), EWM_HALF);
    // SAFETY: lengths checked above; caller guarantees avx2.
    unsafe { ewm_one_perspective_ptr(half.as_ptr(), out.as_mut_ptr()) }
}

/// `output = clamp(input >> WEIGHT_SCALE_BITS, 0, 127)`.
///
/// Processes 32 `i32` per iteration (a full 32-byte store), then 16, so the
/// two-stage narrowing (`i32` -> `i16` -> `u8`) never wastes a register half.
/// A tail shorter than 16 falls back to scalar,
/// which every current caller avoids — `fc_0`'s output is 16 wide and `fc_1`'s
/// is 32 or 64 — but the tail keeps this correct for any length.
///
/// # Safety
/// The running CPU must support `avx2`. `input` and `output` must be the same
/// length.
#[target_feature(enable = "avx2")]
pub unsafe fn clipped_relu(input: &[i32], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    let lanes = 16;
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let zero = _mm256_setzero_si256();
    let cap = _mm256_set1_epi32(127);

    // 32 lanes per step: shift, narrow `i32` -> `i16` -> `i8` with saturation,
    // and raise the floor to zero. The two saturating narrowings are monotone
    // and clip to `[-32768, 32767]` and then `[-128, 127]`, so with the floor
    // applied the result equals `clamp(0, 127)` for every `i32`. Each pack is
    // lane-local and leaves the dwords as `[a0..4, b0..4, c0..4, d0..4,
    // a4..8, b4..8, c4..8, d4..8]`; `permutevar8x32` puts them back in
    // `a, b, c, d` order.
    let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let mut done = 0;
    while done + 2 * lanes <= input.len() {
        // SAFETY: done + 32 <= input.len() == output.len() for the four loads
        // and the store.
        let (a, b, c, d) = unsafe {
            (
                _mm256_loadu_si256(in_ptr.add(done).cast::<__m256i>()),
                _mm256_loadu_si256(in_ptr.add(done + 8).cast::<__m256i>()),
                _mm256_loadu_si256(in_ptr.add(done + 16).cast::<__m256i>()),
                _mm256_loadu_si256(in_ptr.add(done + 24).cast::<__m256i>()),
            )
        };
        let a = _mm256_srai_epi32::<{ WEIGHT_SCALE_BITS as i32 }>(a);
        let b = _mm256_srai_epi32::<{ WEIGHT_SCALE_BITS as i32 }>(b);
        let c = _mm256_srai_epi32::<{ WEIGHT_SCALE_BITS as i32 }>(c);
        let d = _mm256_srai_epi32::<{ WEIGHT_SCALE_BITS as i32 }>(d);
        let bytes = _mm256_max_epi8(
            _mm256_packs_epi16(_mm256_packs_epi32(a, b), _mm256_packs_epi32(c, d)),
            zero,
        );
        let bytes = _mm256_permutevar8x32_epi32(bytes, order);
        // SAFETY: done + 32 <= output.len().
        unsafe { _mm256_storeu_si256(out_ptr.add(done).cast::<__m256i>(), bytes) };
        done += 2 * lanes;
    }

    // Then 16 lanes at a time, then the scalar tail.
    let chunks = input.len() / lanes;
    for k in done / lanes..chunks {
        let off = k * lanes;
        // SAFETY: chunks*16 <= input.len(); both 8-lane loads stay in range.
        let (a, b) = unsafe {
            (
                _mm256_loadu_si256(in_ptr.add(off).cast::<__m256i>()),
                _mm256_loadu_si256(in_ptr.add(off + 8).cast::<__m256i>()),
            )
        };
        let ca = _mm256_min_epi32(
            _mm256_max_epi32(_mm256_srai_epi32::<{ WEIGHT_SCALE_BITS as i32 }>(a), zero),
            cap,
        );
        let cb = _mm256_min_epi32(
            _mm256_max_epi32(_mm256_srai_epi32::<{ WEIGHT_SCALE_BITS as i32 }>(b), zero),
            cap,
        );

        // i32 -> i16, then i16 -> u8. Each pack is lane-local, so each needs its
        // own reorder; values are already in [0, 127] so neither saturates.
        // SAFETY: guarded by this function's own target_feature.
        let words = unsafe { fix_pack_lanes(_mm256_packs_epi32(ca, cb)) };
        // SAFETY: same.
        let bytes = unsafe { fix_pack_lanes(_mm256_packus_epi16(words, words)) };
        // The useful half is the low 128 bits: `packus` duplicated the input.
        let low = _mm256_castsi256_si128(bytes);
        // SAFETY: off + 16 <= input.len() == output.len().
        unsafe { _mm_storeu_si128(out_ptr.add(off).cast::<__m128i>(), low) };
    }

    for i in chunks * lanes..input.len() {
        output[i] = (input[i] >> WEIGHT_SCALE_BITS).clamp(0, 127) as u8;
    }
}

/// Byte lanes in a 256-bit register.
const U8_LANES: usize = 32;

/// The one release-mode check the pointer loads in the affine kernels rely on:
/// every slice is as long as the shape says. One `assert!` over a combined
/// condition, so the hot path has a single not-taken branch and a single cold
/// panic site; `checked_mul` also rules out an `output.len() * padded_in` that
/// wraps around to a matching `weights.len()`.
#[inline(always)]
fn assert_affine_shape(
    output_len: usize,
    biases_len: usize,
    weights_len: usize,
    input_len: usize,
    in_dims: usize,
    padded_in: usize,
) {
    assert!(
        output_len == biases_len
            && in_dims <= padded_in
            && input_len >= in_dims
            && output_len.checked_mul(padded_in) == Some(weights_len),
        "affine kernel: slice lengths do not match the layer shape"
    );
}

/// Horizontal sum of the eight `i32` lanes of `v` (wrapping, like every other
/// `i32` add in this file).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_epi32(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_hadd_epi32(s, s);
    let s = _mm_hadd_epi32(s, s);
    _mm_cvtsi128_si32(s)
}

/// Integer affine transform for **7-bit inputs**: the same function as
/// [`crate::simd::scalar_post_ft::affine`], restricted to `input[i] <= 127`.
///
/// `output[j] = biases[j] + sum_i weights[j][i] * input[i]` with `i32`
/// wrapping, weights row-major `[output][padded_in]`.
///
/// ## Why the restriction
///
/// `_mm256_maddubs_epi16` multiplies unsigned bytes by signed bytes and adds
/// adjacent pairs **with signed 16-bit saturation**. With `input <= 127` and
/// `weight >= -128`, a pair is at worst `2 * 127 * -128 = -32512` and at best
/// `2 * 127 * 127 = 32258`, both inside `i16`, so nothing saturates and the
/// result is the exact pair sum the scalar loop computes. A byte above 127
/// (the SFNN chains feed `u8` up to 255) could saturate, which is why the
/// general `affine` does not use this kernel. The KP/HalfKP chain feeds
/// `clamp(0, 127)` bytes into every layer, which is the case this serves.
///
/// Each pair sum is widened to `i32` with `madd_epi16(_, 1)` before it is
/// accumulated, so the running sum never lives in `i16`. `i32` addition is
/// associative modulo 2^32, so reordering the sum into per-lane partials
/// leaves the bits unchanged.
///
/// Four output rows are processed together so each 32-byte input chunk is
/// loaded once for four weight rows. A trailing `output.len() % 4` rows go one
/// at a time. An `in_dims` that is not a multiple of 32 falls back to the
/// scalar kernel (no current shape has one).
///
/// # Safety
/// The running CPU must support `avx2`. Every `input[..in_dims]` byte must be
/// at most 127 (checked by `debug_assert!`); a larger byte may saturate inside
/// `maddubs` and change the result, though it cannot read out of bounds: the
/// slice lengths are asserted on every call.
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn affine_u7(
    output: &mut [i32],
    biases: &[i32],
    weights: &[i8],
    input: &[u8],
    in_dims: usize,
    padded_in: usize,
) {
    // A real assert, not a debug one: the loads below go through raw pointers
    // and trust these lengths.
    assert_affine_shape(
        output.len(),
        biases.len(),
        weights.len(),
        input.len(),
        in_dims,
        padded_in,
    );
    // The 7-bit bound is a value property, not a length: it would cost a pass
    // over the input, so it stays a debug-only check.
    debug_assert!(
        input[..in_dims].iter().all(|&x| x <= 127),
        "affine_u7 needs 7-bit inputs: maddubs saturates above 127"
    );

    if in_dims % U8_LANES != 0 {
        scalar_post_ft::affine(output, biases, weights, input, in_dims, padded_in);
        return;
    }

    let chunks = in_dims / U8_LANES;
    let rows = output.len();
    let ones = _mm256_set1_epi16(1);
    let in_ptr = input.as_ptr();
    let w_ptr = weights.as_ptr();
    let b_ptr = biases.as_ptr();
    let out_ptr = output.as_mut_ptr();

    let mut j = 0;
    while j + 4 <= rows {
        // SAFETY: j + 4 <= rows and weights.len() == rows * padded_in, so the
        // four row bases are in range.
        let (r0, r1, r2, r3) = unsafe {
            (
                w_ptr.add(j * padded_in),
                w_ptr.add((j + 1) * padded_in),
                w_ptr.add((j + 2) * padded_in),
                w_ptr.add((j + 3) * padded_in),
            )
        };
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut a2 = _mm256_setzero_si256();
        let mut a3 = _mm256_setzero_si256();
        for c in 0..chunks {
            let off = c * U8_LANES;
            // SAFETY: off + 32 <= in_dims <= padded_in, so each 32-byte load
            // is inside the input and inside its weight row.
            unsafe {
                let x = _mm256_loadu_si256(in_ptr.add(off).cast::<__m256i>());
                let w0 = _mm256_loadu_si256(r0.add(off).cast::<__m256i>());
                let w1 = _mm256_loadu_si256(r1.add(off).cast::<__m256i>());
                let w2 = _mm256_loadu_si256(r2.add(off).cast::<__m256i>());
                let w3 = _mm256_loadu_si256(r3.add(off).cast::<__m256i>());
                a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w0), ones));
                a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w1), ones));
                a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w2), ones));
                a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w3), ones));
            }
        }
        // Reduce four 8-lane accumulators to [sum0, sum1, sum2, sum3]:
        // `hadd` works per 128-bit lane, so the low and high halves of the
        // final vector each hold the four partials for the same four rows.
        let s01 = _mm256_hadd_epi32(a0, a1);
        let s23 = _mm256_hadd_epi32(a2, a3);
        let s = _mm256_hadd_epi32(s01, s23);
        let sums = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256::<1>(s));
        // SAFETY: j + 4 <= rows == biases.len() == output.len().
        unsafe {
            let bias = _mm_loadu_si128(b_ptr.add(j).cast::<__m128i>());
            _mm_storeu_si128(out_ptr.add(j).cast::<__m128i>(), _mm_add_epi32(sums, bias));
        }
        j += 4;
    }

    while j < rows {
        // SAFETY: j < rows and weights.len() == rows * padded_in.
        let r = unsafe { w_ptr.add(j * padded_in) };
        let mut acc = _mm256_setzero_si256();
        for c in 0..chunks {
            let off = c * U8_LANES;
            // SAFETY: off + 32 <= in_dims <= padded_in.
            unsafe {
                let x = _mm256_loadu_si256(in_ptr.add(off).cast::<__m256i>());
                let w = _mm256_loadu_si256(r.add(off).cast::<__m256i>());
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w), ones));
            }
        }
        // SAFETY: guarded by this function's own target_feature.
        let sum = unsafe { hsum_epi32(acc) };
        // SAFETY: j < rows == biases.len() == output.len().
        unsafe { *out_ptr.add(j) = (*b_ptr.add(j)).wrapping_add(sum) };
        j += 1;
    }
}

/// [`affine_u7`] for weights that passed
/// `scalar_post_ft::pair_sums_fit_i16`: 8 chunks of `maddubs` results are
/// added as `i16` before one widening `madd`, instead of widening each.
///
/// Per 4 rows and 8 chunks that is 32 `maddubs` + 32 `add_epi16` + 4 `madd` + 4
/// `add_epi32`, against `affine_u7`'s 32 + 32 + 32 -- about a quarter fewer
/// vector ops on the layer that dominates an evaluation (fc_0, 512 inputs).
///
/// ## Why it is still exact
///
/// `add_epi16` wraps, so the sum is right only if no lane leaves `i16`.
/// `pair_sums_fit_i16` proves that for every input in `0..=127` (it bounds the
/// positive and the negative weight mass of each row, lane and group), and the
/// caller must have run it on exactly these `weights`. Given that, every
/// `maddubs` pair sum, every partial `i16` sum and the group total equal the
/// true integers, so the result is the scalar one. If the check is skipped or
/// the weights changed, the memory accesses are unchanged but the numbers can
/// be wrong -- hence the `debug_assert!`.
///
/// # Safety
/// The running CPU must support `avx2`. `in_dims` must be a multiple of 32, and
/// every `input[..in_dims]` byte at most 127.
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn affine_u7_wide(
    output: &mut [i32],
    biases: &[i32],
    weights: &[i8],
    input: &[u8],
    in_dims: usize,
    padded_in: usize,
) {
    // Real asserts, not debug ones: the loads below go through raw pointers and
    // trust these lengths (and that the width is whole 32-byte chunks).
    assert_affine_shape(
        output.len(),
        biases.len(),
        weights.len(),
        input.len(),
        in_dims,
        padded_in,
    );
    assert!(in_dims % U8_LANES == 0, "affine_u7_wide needs whole chunks");
    // Value properties would cost a pass over the data, so they are checked in
    // debug builds only; `NetworkStack` proves the weights' once, at build.
    debug_assert!(
        input[..in_dims].iter().all(|&x| x <= 127),
        "affine_u7_wide needs 7-bit inputs: maddubs saturates above 127"
    );
    debug_assert!(
        scalar_post_ft::pair_sums_fit_i16(weights, output.len(), in_dims, padded_in),
        "affine_u7_wide called on weights whose i16 pair sums can overflow"
    );

    let chunks = in_dims / U8_LANES;
    let rows = output.len();
    let group = scalar_post_ft::U7_WIDE_GROUP_CHUNKS;
    let ones = _mm256_set1_epi16(1);
    let in_ptr = input.as_ptr();
    let w_ptr = weights.as_ptr();
    let b_ptr = biases.as_ptr();
    let out_ptr = output.as_mut_ptr();

    let mut j = 0;
    while j + 4 <= rows {
        // SAFETY: j + 4 <= rows and weights.len() == rows * padded_in (asserted).
        let (r0, r1, r2, r3) = unsafe {
            (
                w_ptr.add(j * padded_in),
                w_ptr.add((j + 1) * padded_in),
                w_ptr.add((j + 2) * padded_in),
                w_ptr.add((j + 3) * padded_in),
            )
        };
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut a2 = _mm256_setzero_si256();
        let mut a3 = _mm256_setzero_si256();
        let mut first = 0;
        while first < chunks {
            let last = (first + group).min(chunks);
            let mut s0 = _mm256_setzero_si256();
            let mut s1 = _mm256_setzero_si256();
            let mut s2 = _mm256_setzero_si256();
            let mut s3 = _mm256_setzero_si256();
            for c in first..last {
                let off = c * U8_LANES;
                // SAFETY: off + 32 <= in_dims <= padded_in (asserted), so each
                // 32-byte load is inside the input and inside its weight row.
                unsafe {
                    let x = _mm256_loadu_si256(in_ptr.add(off).cast::<__m256i>());
                    s0 = _mm256_add_epi16(
                        s0,
                        _mm256_maddubs_epi16(x, _mm256_loadu_si256(r0.add(off).cast::<__m256i>())),
                    );
                    s1 = _mm256_add_epi16(
                        s1,
                        _mm256_maddubs_epi16(x, _mm256_loadu_si256(r1.add(off).cast::<__m256i>())),
                    );
                    s2 = _mm256_add_epi16(
                        s2,
                        _mm256_maddubs_epi16(x, _mm256_loadu_si256(r2.add(off).cast::<__m256i>())),
                    );
                    s3 = _mm256_add_epi16(
                        s3,
                        _mm256_maddubs_epi16(x, _mm256_loadu_si256(r3.add(off).cast::<__m256i>())),
                    );
                }
            }
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(s0, ones));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(s1, ones));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(s2, ones));
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(s3, ones));
            first = last;
        }
        // Same reduction as `affine_u7`: [sum0, sum1, sum2, sum3] + biases.
        let s01 = _mm256_hadd_epi32(a0, a1);
        let s23 = _mm256_hadd_epi32(a2, a3);
        let s = _mm256_hadd_epi32(s01, s23);
        let sums = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256::<1>(s));
        // SAFETY: j + 4 <= rows == biases.len() == output.len().
        unsafe {
            let bias = _mm_loadu_si128(b_ptr.add(j).cast::<__m128i>());
            _mm_storeu_si128(out_ptr.add(j).cast::<__m128i>(), _mm_add_epi32(sums, bias));
        }
        j += 4;
    }

    while j < rows {
        // SAFETY: j < rows and weights.len() == rows * padded_in.
        let r = unsafe { w_ptr.add(j * padded_in) };
        let mut acc = _mm256_setzero_si256();
        let mut first = 0;
        while first < chunks {
            let last = (first + group).min(chunks);
            let mut s = _mm256_setzero_si256();
            for c in first..last {
                let off = c * U8_LANES;
                // SAFETY: off + 32 <= in_dims <= padded_in.
                unsafe {
                    let x = _mm256_loadu_si256(in_ptr.add(off).cast::<__m256i>());
                    let w = _mm256_loadu_si256(r.add(off).cast::<__m256i>());
                    s = _mm256_add_epi16(s, _mm256_maddubs_epi16(x, w));
                }
            }
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(s, ones));
            first = last;
        }
        // SAFETY: guarded by this function's own target_feature.
        let sum = unsafe { hsum_epi32(acc) };
        // SAFETY: j < rows == biases.len() == output.len().
        unsafe { *out_ptr.add(j) = (*b_ptr.add(j)).wrapping_add(sum) };
        j += 1;
    }
}

/// Reduce four 8-lane `i32` accumulators to `[sum(a0), sum(a1), sum(a2),
/// sum(a3)]`. `hadd` works per 128-bit lane, so the low and high halves of the
/// final vector each hold the four partials for the same four accumulators.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn reduce4_epi32(a0: __m256i, a1: __m256i, a2: __m256i, a3: __m256i) -> __m128i {
    let s01 = _mm256_hadd_epi32(a0, a1);
    let s23 = _mm256_hadd_epi32(a2, a3);
    let s = _mm256_hadd_epi32(s01, s23);
    _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256::<1>(s))
}

/// The dot products of `R` consecutive weight rows with `input`, as `R`
/// 8-lane `i32` accumulators (each lane a partial sum of 4-byte groups).
///
/// `vpdpbusd` multiplies unsigned input bytes by signed weight bytes, adds the
/// four products of each 32-bit lane and accumulates into `i32`, **without the
/// `i16` saturation `maddubs` has**: one instruction replaces `maddubs` +
/// `madd` + `add`. Its accumulate wraps modulo 2^32 like the scalar loop's
/// `wrapping_add`, so the sum is the same for any input byte, 0..=255.
///
/// # Safety
/// The CPU must support `avx2` and `avxvnni`. Every row pointer must be valid
/// for `chunks * 32` bytes, and `input` likewise.
#[inline]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn vnni_rows<const R: usize>(
    rows: [*const i8; R],
    input: *const u8,
    chunks: usize,
) -> [__m256i; R] {
    let mut acc = [_mm256_setzero_si256(); R];
    for c in 0..chunks {
        let off = c * U8_LANES;
        // SAFETY: off + 32 <= chunks * 32, in range for the input and each row.
        unsafe {
            let x = _mm256_loadu_si256(input.add(off).cast::<__m256i>());
            for r in 0..R {
                let w = _mm256_loadu_si256(rows[r].add(off).cast::<__m256i>());
                acc[r] = _mm256_dpbusd_avx_epi32(acc[r], x, w);
            }
        }
    }
    acc
}

/// [`affine_u7`] with AVX-VNNI: the same row-major layout, the same result,
/// and no 7-bit restriction (the products never saturate), but `vpdpbusd` does
/// in one instruction what `maddubs` + `madd` + `add` do in three.
///
/// Eight output rows are taken together, which gives the `vpdpbusd` chains
/// enough independent accumulators to cover its latency; a trailing 4 and then
/// single rows follow. An `in_dims` that is not a multiple of 32 falls back to
/// the scalar kernel.
///
/// # Safety
/// The running CPU must support `avx2` and `avxvnni`.
#[inline]
#[target_feature(enable = "avx2,avxvnni")]
pub unsafe fn affine_u7_vnni(
    output: &mut [i32],
    biases: &[i32],
    weights: &[i8],
    input: &[u8],
    in_dims: usize,
    padded_in: usize,
) {
    // A real assert: the loads below trust these lengths.
    assert_affine_shape(
        output.len(),
        biases.len(),
        weights.len(),
        input.len(),
        in_dims,
        padded_in,
    );

    if in_dims % U8_LANES != 0 {
        scalar_post_ft::affine(output, biases, weights, input, in_dims, padded_in);
        return;
    }

    let chunks = in_dims / U8_LANES;
    let rows = output.len();
    let in_ptr = input.as_ptr();
    let w_ptr = weights.as_ptr();
    let b_ptr = biases.as_ptr();
    let out_ptr = output.as_mut_ptr();

    let mut j = 0;
    while j + 8 <= rows {
        // SAFETY: j + 8 <= rows and weights.len() == rows * padded_in.
        let acc = unsafe {
            vnni_rows::<8>(
                std::array::from_fn(|k| w_ptr.add((j + k) * padded_in)),
                in_ptr,
                chunks,
            )
        };
        // SAFETY: guarded by this function's own target_feature.
        let (lo, hi) = unsafe {
            (
                reduce4_epi32(acc[0], acc[1], acc[2], acc[3]),
                reduce4_epi32(acc[4], acc[5], acc[6], acc[7]),
            )
        };
        // SAFETY: j + 8 <= rows == biases.len() == output.len().
        unsafe {
            let (b_lo, b_hi) = (
                _mm_loadu_si128(b_ptr.add(j).cast::<__m128i>()),
                _mm_loadu_si128(b_ptr.add(j + 4).cast::<__m128i>()),
            );
            _mm_storeu_si128(out_ptr.add(j).cast::<__m128i>(), _mm_add_epi32(lo, b_lo));
            _mm_storeu_si128(
                out_ptr.add(j + 4).cast::<__m128i>(),
                _mm_add_epi32(hi, b_hi),
            );
        }
        j += 8;
    }
    if j + 4 <= rows {
        // SAFETY: j + 4 <= rows and weights.len() == rows * padded_in.
        let acc = unsafe {
            vnni_rows::<4>(
                std::array::from_fn(|k| w_ptr.add((j + k) * padded_in)),
                in_ptr,
                chunks,
            )
        };
        // SAFETY: guarded by this function's own target_feature.
        let sums = unsafe { reduce4_epi32(acc[0], acc[1], acc[2], acc[3]) };
        // SAFETY: j + 4 <= rows == biases.len() == output.len().
        unsafe {
            let bias = _mm_loadu_si128(b_ptr.add(j).cast::<__m128i>());
            _mm_storeu_si128(out_ptr.add(j).cast::<__m128i>(), _mm_add_epi32(sums, bias));
        }
        j += 4;
    }
    while j < rows {
        // SAFETY: j < rows and weights.len() == rows * padded_in.
        let acc = unsafe { vnni_rows::<1>([w_ptr.add(j * padded_in)], in_ptr, chunks) };
        // SAFETY: guarded by this function's own target_feature; j < rows.
        unsafe {
            let sum = hsum_epi32(acc[0]);
            *out_ptr.add(j) = (*b_ptr.add(j)).wrapping_add(sum);
        }
        j += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Skip unless the CPU has AVX-VNNI (this build need not enable it: the
    /// kernel is compiled regardless, so the tests run it wherever it exists).
    macro_rules! require_avxvnni {
        () => {
            if !(std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("avxvnni"))
            {
                eprintln!("skipping AVX-VNNI parity test: avxvnni unavailable");
                return;
            }
        };
    }

    macro_rules! require_avx2 {
        () => {
            if !std::arch::is_x86_feature_detected!("avx2") {
                eprintln!("skipping AVX2 parity test: avx2 unavailable");
                return;
            }
        };
    }

    /// Values spanning the clamp boundaries in both directions, so the test
    /// covers the saturating and the pass-through paths rather than a middle
    /// band where every implementation agrees.
    fn seeded_half(seed: i16) -> Vec<i16> {
        (0..HIDDEN_SIZE)
            .map(|i| {
                let v = (i as i32 * 37 + seed as i32) % 700 - 200;
                v as i16
            })
            .collect()
    }

    #[test]
    fn ewm_matches_scalar() {
        require_avx2!();
        for seed in [0i16, 13, 101] {
            let half = seeded_half(seed);
            let mut want = vec![0u8; EWM_HALF];
            let mut got = vec![0u8; EWM_HALF];
            scalar_post_ft::ewm_one_perspective(&half, &mut want);
            // SAFETY: guarded by `require_avx2!`.
            unsafe { ewm_one_perspective(&half, &mut got) };
            assert_eq!(want, got, "seed {seed}");
        }
    }

    #[test]
    fn clipped_relu_matches_scalar() {
        require_avx2!();
        // Spans both clamp boundaries and the shift's sign behaviour, and
        // includes a length that is not a multiple of 16 to exercise the tail.
        for len in [16usize, 32, 64, 19] {
            let input: Vec<i32> = (0..len)
                .map(|i| (i as i32 * 977 - 4000).wrapping_mul(13))
                .collect();
            let mut want = vec![0u8; len];
            let mut got = vec![0u8; len];
            scalar_post_ft::clipped_relu(&input, &mut want);
            // SAFETY: guarded by `require_avx2!`.
            unsafe { clipped_relu(&input, &mut got) };
            assert_eq!(want, got, "len {len}");
        }
    }

    /// The pack reorder is the easiest thing to get wrong and the hardest to
    /// see, so pin it directly: a pattern whose every byte is distinct fails
    /// loudly if the 64-bit groups come back in the wrong order.
    #[test]
    fn pack_lane_fix_produces_straight_concatenation() {
        require_avx2!();
        let a: [i16; 16] = std::array::from_fn(|i| i as i16);
        let b: [i16; 16] = std::array::from_fn(|i| (i + 16) as i16);
        // SAFETY: guarded by `require_avx2!`.
        let got: [u8; 32] = unsafe {
            let va = _mm256_loadu_si256(a.as_ptr().cast::<__m256i>());
            let vb = _mm256_loadu_si256(b.as_ptr().cast::<__m256i>());
            let packed = fix_pack_lanes(_mm256_packus_epi16(va, vb));
            let mut out = [0u8; 32];
            _mm256_storeu_si256(out.as_mut_ptr().cast::<__m256i>(), packed);
            out
        };
        let want: [u8; 32] = std::array::from_fn(|i| i as u8);
        assert_eq!(got, want);
    }
    /// Deterministic xorshift64*, so a failure reproduces from the seed alone.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: u64) -> u64 {
            (self.next() >> 11) % n
        }
    }

    /// Weights drawn so the extremes (-128 and 127) are common rather than
    /// vanishingly rare: a third of the weights are pinned to one of them.
    fn random_weights(rng: &mut Rng, n: usize) -> Vec<i8> {
        (0..n)
            .map(|_| match rng.below(6) {
                0 => -128i8,
                1 => 127i8,
                _ => rng.next() as i8,
            })
            .collect()
    }

    /// Inputs in `0..=127` with 0 and 127 over-represented.
    fn random_inputs(rng: &mut Rng, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| match rng.below(6) {
                0 => 0u8,
                1 => 127u8,
                _ => rng.below(128) as u8,
            })
            .collect()
    }

    fn random_biases(rng: &mut Rng, n: usize) -> Vec<i32> {
        (0..n)
            .map(|_| match rng.below(5) {
                0 => i32::MAX,
                1 => i32::MIN,
                _ => rng.next() as i32,
            })
            .collect()
    }

    /// Every shape the KP/HalfKP chain uses (fc_0 512 -> 32, fc_1 32 -> 32,
    /// fc_2 32 -> 1), plus ones whose row count is not a multiple of four and
    /// whose input is padded, so the single-row path and the padding stride are
    /// both exercised.
    const AFFINE_SHAPES: [(usize, usize, usize); 8] = [
        (32, 512, 512),
        (32, 32, 32),
        (1, 32, 32),
        (7, 64, 64),
        (5, 96, 128),
        (4, 32, 64),
        (3, 32, 32),
        (9, 160, 160),
    ];

    #[test]
    fn affine_u7_matches_scalar_on_random_data() {
        require_avx2!();
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for &(rows, in_dims, padded_in) in &AFFINE_SHAPES {
            for _ in 0..200 {
                let weights = random_weights(&mut rng, rows * padded_in);
                let input = random_inputs(&mut rng, in_dims);
                let biases = random_biases(&mut rng, rows);
                let mut want = vec![0i32; rows];
                let mut got = vec![0i32; rows];
                scalar_post_ft::affine(&mut want, &biases, &weights, &input, in_dims, padded_in);
                // SAFETY: guarded by `require_avx2!`; the inputs are all <= 127.
                unsafe { affine_u7(&mut got, &biases, &weights, &input, in_dims, padded_in) };
                assert_eq!(want, got, "rows {rows} in {in_dims} padded {padded_in}");
            }
        }
    }

    /// The worst cases for `maddubs`' saturating pair sum: every product at its
    /// most negative (127 * -128) and at its most positive (127 * 127), with the
    /// bias at both ends of `i32` so the wrapping add is exercised too.
    #[test]
    fn affine_u7_is_exact_at_the_saturation_boundaries() {
        require_avx2!();
        for (w, x) in [(-128i8, 127u8), (127, 127), (-128, 0), (127, 0), (-1, 127)] {
            for bias in [0, i32::MAX, i32::MIN, -1, 1] {
                for &(rows, in_dims, padded_in) in &AFFINE_SHAPES {
                    let weights = vec![w; rows * padded_in];
                    let input = vec![x; in_dims];
                    let biases = vec![bias; rows];
                    let mut want = vec![0i32; rows];
                    let mut got = vec![0i32; rows];
                    scalar_post_ft::affine(
                        &mut want, &biases, &weights, &input, in_dims, padded_in,
                    );
                    // SAFETY: guarded by `require_avx2!`; the inputs are all <= 127.
                    unsafe { affine_u7(&mut got, &biases, &weights, &input, in_dims, padded_in) };
                    assert_eq!(
                        want, got,
                        "w {w} x {x} bias {bias} rows {rows} in {in_dims}"
                    );
                }
            }
        }
    }

    /// `affine_u7` must read exactly `in_dims` input bytes and the matching
    /// weight columns: bytes past `in_dims` in the weight rows (padding) must
    /// not leak into the sum.
    #[test]
    fn affine_u7_ignores_weight_padding() {
        require_avx2!();
        let (rows, in_dims, padded_in) = (4usize, 32usize, 64usize);
        let mut weights = vec![0i8; rows * padded_in];
        for r in 0..rows {
            for i in 0..in_dims {
                weights[r * padded_in + i] = (r as i8 + 1) * ((i % 5) as i8 - 2);
            }
            for i in in_dims..padded_in {
                weights[r * padded_in + i] = 127;
            }
        }
        let input: Vec<u8> = (0..in_dims).map(|i| (i * 4 % 128) as u8).collect();
        let biases = vec![11i32; rows];
        let mut want = vec![0i32; rows];
        let mut got = vec![0i32; rows];
        scalar_post_ft::affine(&mut want, &biases, &weights, &input, in_dims, padded_in);
        // SAFETY: guarded by `require_avx2!`; the inputs are all <= 127.
        unsafe { affine_u7(&mut got, &biases, &weights, &input, in_dims, padded_in) };
        assert_eq!(want, got);
    }

    #[test]
    fn clipped_relu_matches_scalar_on_random_i32() {
        require_avx2!();
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        for len in [16usize, 19, 32, 33, 48, 64, 100] {
            for round in 0..200 {
                let input: Vec<i32> = (0..len)
                    .map(|_| match rng.below(8) {
                        0 => i32::MIN,
                        1 => i32::MAX,
                        2 => 127 << 6,
                        3 => (127 << 6) - 1,
                        4 => -1,
                        // Around the clamp, where the shift and both packs
                        // decide the answer.
                        5 => (rng.below(300) as i32 - 100) << 6,
                        _ => rng.next() as i32,
                    })
                    .collect();
                let mut want = vec![0u8; len];
                let mut got = vec![0u8; len];
                scalar_post_ft::clipped_relu(&input, &mut want);
                // SAFETY: guarded by `require_avx2!`.
                unsafe { clipped_relu(&input, &mut got) };
                assert_eq!(want, got, "len {len} round {round}");
            }
        }
    }
    /// Weights small enough that `pair_sums_fit_i16` holds for a 512-input
    /// row (which `affine_u7_wide` requires), still hitting the extremes of the
    /// range the check allows.
    fn random_small_weights(rng: &mut Rng, n: usize, magnitude: u64) -> Vec<i8> {
        (0..n)
            .map(|_| (rng.below(2 * magnitude + 1) as i64 - magnitude as i64) as i8)
            .collect()
    }

    #[test]
    fn affine_u7_wide_matches_scalar_when_the_weights_qualify() {
        require_avx2!();
        let mut rng = Rng(0x0DDB_1A5E_5BAD_5EED);
        let shapes: [(usize, usize, usize); 6] = [
            (32, 512, 512),
            (32, 256, 256),
            (6, 288, 320),
            (5, 32, 32),
            (4, 64, 64),
            (1, 512, 512),
        ];
        let mut accepted = 0;
        for &(rows, in_dims, padded_in) in &shapes {
            for magnitude in [3u64, 10, 20, 40] {
                for _ in 0..100 {
                    let weights = random_small_weights(&mut rng, rows * padded_in, magnitude);
                    if !scalar_post_ft::pair_sums_fit_i16(&weights, rows, in_dims, padded_in) {
                        continue;
                    }
                    accepted += 1;
                    let input = random_inputs(&mut rng, in_dims);
                    let biases = random_biases(&mut rng, rows);
                    let mut want = vec![0i32; rows];
                    let mut got = vec![0i32; rows];
                    scalar_post_ft::affine(
                        &mut want, &biases, &weights, &input, in_dims, padded_in,
                    );
                    // SAFETY: guarded by `require_avx2!`; inputs <= 127 and the
                    // weights passed `pair_sums_fit_i16` just above.
                    unsafe {
                        affine_u7_wide(&mut got, &biases, &weights, &input, in_dims, padded_in)
                    };
                    assert_eq!(want, got, "rows {rows} in {in_dims} magnitude {magnitude}");
                }
            }
        }
        assert!(accepted > 500, "the generator rarely met the precondition");
    }

    /// The worst input for the weights the check admits: 127 where the weight is
    /// positive and 0 where it is negative drives the `i16` sum to its maximum
    /// (and the reverse to its minimum), so a lane sitting exactly at the limit
    /// must still come out exact.
    #[test]
    fn affine_u7_wide_is_exact_at_the_i16_limit() {
        require_avx2!();
        let (rows, in_dims, padded_in) = (4usize, 512usize, 512usize);
        // Lane 0 of every chunk of a group carries 16 weights; 258 in total
        // (the largest sum 127 * w can hold) is 16 weights of 16 plus 2.
        for sign in [1i32, -1] {
            let mut weights = vec![0i8; rows * padded_in];
            for r in 0..rows {
                for chunk in 0..16 {
                    for k in 0..2 {
                        weights[r * padded_in + chunk * 32 + k] = (sign * 16) as i8;
                    }
                }
                // Two chunk groups of 16 weights each; add 2 to the first.
                weights[r * padded_in] = (sign * 18) as i8;
            }
            assert!(scalar_post_ft::pair_sums_fit_i16(
                &weights, rows, in_dims, padded_in
            ));
            for adversarial in [true, false] {
                let input: Vec<u8> = weights_row_input(&weights[..padded_in], sign, adversarial);
                let biases = vec![0, i32::MAX, i32::MIN, 7];
                let mut want = vec![0i32; rows];
                let mut got = vec![0i32; rows];
                scalar_post_ft::affine(&mut want, &biases, &weights, &input, in_dims, padded_in);
                // SAFETY: guarded by `require_avx2!`; inputs <= 127 and the
                // weights passed `pair_sums_fit_i16` above.
                unsafe { affine_u7_wide(&mut got, &biases, &weights, &input, in_dims, padded_in) };
                assert_eq!(want, got, "sign {sign} adversarial {adversarial}");
            }
        }
    }

    /// 127 on the bytes whose weight pushes the sum toward its extreme, 0
    /// elsewhere (`adversarial`), or the plain all-127 vector.
    fn weights_row_input(row: &[i8], sign: i32, adversarial: bool) -> Vec<u8> {
        row.iter()
            .map(|&w| {
                if !adversarial || (w as i32) * sign > 0 {
                    127u8
                } else {
                    0u8
                }
            })
            .collect()
    }
    /// Inputs over the full byte range, 0 and 255 over-represented: unlike
    /// `maddubs`, `vpdpbusd` does not care about the 7-bit bound.
    fn random_full_range_inputs(rng: &mut Rng, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| match rng.below(6) {
                0 => 0u8,
                1 => 255u8,
                _ => rng.next() as u8,
            })
            .collect()
    }

    /// Shapes the dense chain uses plus ones that exercise each of the 8-row,
    /// 4-row and single-row paths and a padded stride.
    const VNNI_SHAPES: [(usize, usize, usize); 11] = [
        (32, 512, 512),
        (32, 32, 32),
        (1, 32, 32),
        (7, 64, 64),
        (8, 64, 64),
        (13, 96, 128),
        (12, 32, 64),
        (4, 32, 64),
        (3, 32, 32),
        (9, 160, 160),
        (17, 224, 256),
    ];

    #[test]
    fn affine_u7_vnni_matches_scalar_on_random_data() {
        require_avxvnni!();
        let mut rng = Rng(0xC0FF_EE15_600D_F00D);
        for &(rows, in_dims, padded_in) in &VNNI_SHAPES {
            for round in 0..200 {
                let weights = random_weights(&mut rng, rows * padded_in);
                // Half the rounds use the full byte range, which `affine_u7`
                // may not be given and this kernel may.
                let input = if round % 2 == 0 {
                    random_inputs(&mut rng, in_dims)
                } else {
                    random_full_range_inputs(&mut rng, in_dims)
                };
                let biases = random_biases(&mut rng, rows);
                let mut want = vec![0i32; rows];
                let mut got = vec![0i32; rows];
                scalar_post_ft::affine(&mut want, &biases, &weights, &input, in_dims, padded_in);
                // SAFETY: guarded by `require_avxvnni!`.
                unsafe { affine_u7_vnni(&mut got, &biases, &weights, &input, in_dims, padded_in) };
                assert_eq!(want, got, "rows {rows} in {in_dims} padded {padded_in}");
            }
        }
    }

    /// Every extreme product (127 * -128, 127 * 127, 255 * -128, 255 * 127,
    /// zeros) with the bias at both ends of `i32`, so the wrapping accumulate
    /// is exercised on every shape.
    #[test]
    fn affine_u7_vnni_is_exact_at_the_extremes() {
        require_avxvnni!();
        for (w, x) in [
            (-128i8, 127u8),
            (127, 127),
            (-128, 255),
            (127, 255),
            (-128, 0),
            (127, 0),
            (-1, 255),
        ] {
            for bias in [0, i32::MAX, i32::MIN, -1, 1] {
                for &(rows, in_dims, padded_in) in &VNNI_SHAPES {
                    let weights = vec![w; rows * padded_in];
                    let input = vec![x; in_dims];
                    let biases = vec![bias; rows];
                    let mut want = vec![0i32; rows];
                    let mut got = vec![0i32; rows];
                    scalar_post_ft::affine(
                        &mut want, &biases, &weights, &input, in_dims, padded_in,
                    );
                    // SAFETY: guarded by `require_avxvnni!`.
                    unsafe {
                        affine_u7_vnni(&mut got, &biases, &weights, &input, in_dims, padded_in)
                    };
                    assert_eq!(
                        want, got,
                        "w {w} x {x} bias {bias} rows {rows} in {in_dims}"
                    );
                }
            }
        }
    }

    /// Padding bytes past `in_dims` in a weight row must not reach the sum, and
    /// an `in_dims` that is not a multiple of 32 must take the scalar fallback.
    #[test]
    fn affine_u7_vnni_ignores_padding_and_handles_ragged_widths() {
        require_avxvnni!();
        let mut rng = Rng(0x5EED_0000_1234_ABCD);
        for &(rows, in_dims, padded_in) in &[(4usize, 32usize, 64usize), (5, 48, 64), (9, 40, 64)] {
            let mut weights = random_weights(&mut rng, rows * padded_in);
            for r in 0..rows {
                for i in in_dims..padded_in {
                    weights[r * padded_in + i] = 127;
                }
            }
            let input = random_inputs(&mut rng, in_dims);
            let biases = random_biases(&mut rng, rows);
            let mut want = vec![0i32; rows];
            let mut got = vec![0i32; rows];
            scalar_post_ft::affine(&mut want, &biases, &weights, &input, in_dims, padded_in);
            // SAFETY: guarded by `require_avxvnni!`.
            unsafe { affine_u7_vnni(&mut got, &biases, &weights, &input, in_dims, padded_in) };
            assert_eq!(want, got, "rows {rows} in {in_dims} padded {padded_in}");
        }
    }
    /// The kernels read through raw pointers, so a slice shorter than the shape
    /// says must stop them before the first load, in release builds too (the
    /// test profile here is release). Each case below would otherwise read past
    /// the end of the weights or the input.
    #[test]
    fn affine_kernels_refuse_inconsistent_lengths() {
        require_avx2!();
        let (rows, in_dims, padded_in) = (8usize, 64usize, 64usize);
        let weights = vec![1i8; rows * padded_in];
        let input = vec![1u8; in_dims];
        let biases = vec![0i32; rows];

        type Kernel = unsafe fn(&mut [i32], &[i32], &[i8], &[u8], usize, usize);
        let mut kernels: Vec<(&str, Kernel)> =
            vec![("affine_u7", affine_u7), ("affine_u7_wide", affine_u7_wide)];
        if std::arch::is_x86_feature_detected!("avxvnni") {
            kernels.push(("affine_u7_vnni", affine_u7_vnni));
        }

        for (name, kernel) in kernels {
            // Each closure is a malformed call; none may return.
            let cases: [(&str, Box<dyn Fn()>); 4] = [
                ("short weights", {
                    let (b, i) = (biases.clone(), input.clone());
                    let w = weights[..weights.len() - 1].to_vec();
                    Box::new(move || {
                        let mut out = vec![0i32; rows];
                        // SAFETY: avx2 checked above; the kernel must reject the call.
                        unsafe { kernel(&mut out, &b, &w, &i, in_dims, padded_in) };
                    })
                }),
                ("short input", {
                    let (b, w) = (biases.clone(), weights.clone());
                    let i = input[..in_dims - 1].to_vec();
                    Box::new(move || {
                        let mut out = vec![0i32; rows];
                        // SAFETY: as above.
                        unsafe { kernel(&mut out, &b, &w, &i, in_dims, padded_in) };
                    })
                }),
                ("short biases", {
                    let (w, i) = (weights.clone(), input.clone());
                    let b = biases[..rows - 1].to_vec();
                    Box::new(move || {
                        let mut out = vec![0i32; rows];
                        // SAFETY: as above.
                        unsafe { kernel(&mut out, &b, &w, &i, in_dims, padded_in) };
                    })
                }),
                ("in_dims past padded_in", {
                    let (b, w, i) = (biases.clone(), weights.clone(), input.clone());
                    Box::new(move || {
                        let mut out = vec![0i32; rows];
                        // SAFETY: as above.
                        unsafe { kernel(&mut out, &b, &w, &i, in_dims + 32, padded_in) };
                    })
                }),
            ];
            for (what, call) in cases {
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(call));
                assert!(outcome.is_err(), "{name} accepted {what}");
            }
        }
    }
    /// `output.len() * padded_in` wrapping around to the weights' length must
    /// not pass for a match: 2 * 2^63 wraps to 0, the length of an empty slice.
    #[test]
    fn affine_shape_check_rejects_a_wrapping_product() {
        let wrapped = std::panic::catch_unwind(|| {
            assert_affine_shape(2, 2, 0, 64, 32, 1usize << (usize::BITS - 1));
        });
        assert!(wrapped.is_err(), "a wrapping rows * padded_in was accepted");
        // And a consistent shape still passes.
        assert_affine_shape(32, 32, 32 * 512, 512, 512, 512);
    }
}
