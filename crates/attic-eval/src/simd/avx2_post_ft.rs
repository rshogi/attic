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
/// Processes 16 `i32` per iteration so the two-stage narrowing (`i32` -> `i16`
/// -> `u8`) fills a 16-byte store. A tail shorter than 16 falls back to scalar,
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
    let chunks = input.len() / lanes;
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let zero = _mm256_setzero_si256();
    let cap = _mm256_set1_epi32(127);

    for k in 0..chunks {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::scalar_post_ft;

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
}
