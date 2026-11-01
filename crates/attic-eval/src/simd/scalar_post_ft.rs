//! Scalar output-transform and layer kernels — the always-available baseline
//! the AVX-512 ones in [`crate::simd::avx512_post_ft`] must match bit for bit.
//!
//! The reference exposes a whole-`Accumulator` `transformer_ewm`. This crate's
//! `Accumulator` keeps the two perspectives in colour-indexed buffers and does
//! the side-to-move reordering itself, so the output transform is factored here
//! as a per-perspective helper the caller invokes twice.

use crate::types::HIDDEN_SIZE;

/// Right-shift applied to the affine output before the clipped ReLU
/// (`kWeightScaleBits`).
const WEIGHT_SCALE_BITS: i32 = 6;

/// Right-shift applied to the squared affine output.
const SQR_SHIFT: i32 = 2 * WEIGHT_SCALE_BITS + 7;

/// FT biases and weights are scaled ×2 at load time, so the post-scale
/// accumulator bound is `[0, 254]`, not `[0, 127]`.
const EWM_CLAMP: i32 = 127 * 2;

/// Right-shift applied to the lane product in the output transform (`/ 512`).
const EWM_SHIFT: i32 = 9;

/// Integer affine transform: `output[j] = bias[j] + Σ weights[j][i] * input[i]`.
///
/// Weights are row-major `[output][padded_in]`; only the first `in_dims`
/// columns of each row are consumed (padding lanes are never read).
pub fn affine(
    output: &mut [i32],
    biases: &[i32],
    weights: &[i8],
    input: &[u8],
    in_dims: usize,
    padded_in: usize,
) {
    debug_assert_eq!(output.len(), biases.len());
    debug_assert_eq!(weights.len(), output.len() * padded_in);
    debug_assert!(in_dims <= padded_in);
    debug_assert!(input.len() >= in_dims);

    for (j, out_slot) in output.iter_mut().enumerate() {
        let row = j * padded_in;
        let mut acc = biases[j];
        for i in 0..in_dims {
            acc = acc.wrapping_add(weights[row + i] as i32 * input[i] as i32);
        }
        *out_slot = acc;
    }
}

/// [`affine`] for inputs that are all at most 127.
///
/// The scalar loop has no use for the bound; it exists so the AVX2 kernel,
/// which does (`maddubs` saturates above 127), has a same-signature baseline to
/// match bit for bit, and so a build without AVX2 can call one name.
pub fn affine_u7(
    output: &mut [i32],
    biases: &[i32],
    weights: &[i8],
    input: &[u8],
    in_dims: usize,
    padded_in: usize,
) {
    debug_assert!(input[..in_dims].iter().all(|&x| x <= 127));
    affine(output, biases, weights, input, in_dims, padded_in);
}

/// Input bytes one 256-bit `maddubs` consumes.
pub const U7_CHUNK_BYTES: usize = 32;

/// Chunks whose `maddubs` results the wide affine adds as `i16` before
/// widening (see [`pair_sums_fit_i16`]).
pub const U7_WIDE_GROUP_CHUNKS: usize = 8;

/// Whether [`affine_u7_wide`](crate::simd::post_ft_kernel::affine_u7_wide)'s
/// `i16` accumulation is exact for these weights, for **every** input in
/// `0..=127`.
///
/// `maddubs` yields one `i16` per adjacent byte pair, `x[2p]*w[2p] +
/// x[2p+1]*w[2p+1]`. The wide kernel adds [`U7_WIDE_GROUP_CHUNKS`] such results
/// (same lane `p`, consecutive 32-byte chunks) in `i16` and only then widens to
/// `i32`, saving a widen and an `i32` add per chunk. That is exact exactly when
/// no lane's running sum leaves `i16`. For inputs in `0..=127` the sum over a
/// set of weights ranges over `[127 * (sum of the negative weights), 127 * (sum
/// of the positive weights)]`, and every partial sum of the lane (a subset of
/// the terms) lies inside the full group's range, so it is enough to check the
/// two ends for each row, lane and group of chunks.
///
/// A trailing group shorter than [`U7_WIDE_GROUP_CHUNKS`] is checked as it is.
/// Only the first `in_dims` bytes of each `padded_in`-byte row are read, as in
/// [`affine`]; `in_dims` must be a multiple of [`U7_CHUNK_BYTES`].
pub fn pair_sums_fit_i16(weights: &[i8], rows: usize, in_dims: usize, padded_in: usize) -> bool {
    if in_dims % U7_CHUNK_BYTES != 0 || in_dims > padded_in || weights.len() != rows * padded_in {
        return false;
    }
    let chunks = in_dims / U7_CHUNK_BYTES;
    for row in 0..rows {
        let r = &weights[row * padded_in..row * padded_in + in_dims];
        for first in (0..chunks).step_by(U7_WIDE_GROUP_CHUNKS) {
            let last = (first + U7_WIDE_GROUP_CHUNKS).min(chunks);
            for lane in 0..U7_CHUNK_BYTES / 2 {
                let (mut pos, mut neg) = (0i32, 0i32);
                for chunk in first..last {
                    for w in &r[chunk * U7_CHUNK_BYTES + 2 * lane..][..2] {
                        if *w > 0 {
                            pos += *w as i32;
                        } else {
                            neg += *w as i32;
                        }
                    }
                }
                if pos * 127 > i16::MAX as i32 || neg * 127 < i16::MIN as i32 {
                    return false;
                }
            }
        }
    }
    true
}

/// [`affine_u7`] for weights that passed [`pair_sums_fit_i16`]: the portable
/// build has no wider accumulator to exploit, so this is the plain loop.
pub fn affine_u7_wide(
    output: &mut [i32],
    biases: &[i32],
    weights: &[i8],
    input: &[u8],
    in_dims: usize,
    padded_in: usize,
) {
    affine_u7(output, biases, weights, input, in_dims, padded_in);
}

/// Clipped ReLU: `output = clamp(input >> WEIGHT_SCALE_BITS, 0, 127)`.
pub fn clipped_relu(input: &[i32], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = (inp >> WEIGHT_SCALE_BITS).clamp(0, 127) as u8;
    }
}

/// Squared clipped ReLU: `output = clamp(input² >> SQR_SHIFT, 0, 127)`, whose
/// `i64` intermediate avoids overflow on the `i32` square.
pub fn sqr_clipped_relu(input: &[i32], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        let sq = (*inp as i64) * (*inp as i64);
        *out = (sq >> SQR_SHIFT).clamp(0, 127) as u8;
    }
}

/// Pairwise element-wise multiply for one perspective: lane `j` pairs with lane
/// `j + HIDDEN_SIZE/2`, each clamped to `[0, EWM_CLAMP]`, multiplied, then
/// shifted right by [`EWM_SHIFT`].
pub fn ewm_one_perspective(half: &[i16], out: &mut [u8]) {
    const HALF: usize = HIDDEN_SIZE / 2;
    debug_assert_eq!(half.len(), HIDDEN_SIZE);
    debug_assert_eq!(out.len(), HALF);
    for j in 0..HALF {
        let s0 = (half[j] as i32).clamp(0, EWM_CLAMP);
        let s1 = (half[j + HALF] as i32).clamp(0, EWM_CLAMP);
        out[j] = ((s0 * s1) >> EWM_SHIFT) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipped_relu_matches_reference_on_boundary_values() {
        let inputs = [
            -1_000_000i32,
            -64,
            -1,
            0,
            1,
            63,
            64,
            8_000,
            8_127,
            8_128,
            i32::MAX,
        ];
        let mut out = [0u8; 11];
        clipped_relu(&inputs, &mut out);
        assert_eq!(out, [0u8, 0, 0, 0, 0, 0, 1, 125, 126, 127, 127]);
    }

    #[test]
    fn sqr_clipped_relu_matches_reference_formula() {
        let inputs = [
            0i32,
            1,
            724,
            725,
            8_191,
            8_192,
            100_000,
            -724,
            -725,
            -100_000,
            i32::MAX,
            i32::MIN,
        ];
        let mut out = [0u8; 12];
        sqr_clipped_relu(&inputs, &mut out);
        assert_eq!(out, [0u8, 0, 0, 1, 127, 127, 127, 0, 1, 127, 127, 127]);
    }

    #[test]
    fn affine_accumulates_bias_plus_dot_product() {
        let biases = [10i32, -5];
        let weights = [2i8, 3, 99, -1, 4, 88];
        let input = [5u8, 7, 0];
        let mut out = [0i32; 2];
        affine(&mut out, &biases, &weights, &input, 2, 3);
        assert_eq!(out[0], 41);
        assert_eq!(out[1], 18);
    }

    #[test]
    fn pair_sums_fit_i16_accepts_the_exact_limit_and_rejects_one_past_it() {
        // One row of 256 inputs = 8 chunks = one group. Lane 0 covers bytes
        // 0 and 1 of every chunk, so put the weights there.
        let make = |lane0: i8, lane1: i8| {
            let mut w = vec![0i8; 256];
            for chunk in 0..8 {
                w[chunk * 32] = lane0;
                w[chunk * 32 + 1] = lane1;
            }
            w
        };
        // 16 weights of 16 sum to 256 -> 256 * 127 = 32512, inside i16.
        assert!(pair_sums_fit_i16(&make(16, 16), 1, 256, 256));
        // 16 weights summing to 258: 258 * 127 = 32766, the last value that fits.
        let mut edge = make(16, 16);
        edge[0] = 18;
        assert!(pair_sums_fit_i16(&edge, 1, 256, 256));
        // 259 * 127 = 32893 does not.
        edge[1] = 17;
        assert!(!pair_sums_fit_i16(&edge, 1, 256, 256));
        // The negative end: -258 * 127 = -32766 fits, -259 * 127 does not.
        let mut neg = make(-16, -16);
        neg[0] = -18;
        assert!(pair_sums_fit_i16(&neg, 1, 256, 256));
        neg[1] = -17;
        assert!(!pair_sums_fit_i16(&neg, 1, 256, 256));
    }

    #[test]
    fn pair_sums_fit_i16_splits_a_row_into_groups_of_eight_chunks() {
        // 16 chunks: the same large weight in lane 0 of chunk 0 and chunk 8
        // lands in different groups, so each group's sum stays small.
        let mut w = vec![0i8; 512];
        for chunk in [0usize, 8] {
            for k in 0..2 {
                w[chunk * 32 + k] = 100;
            }
        }
        assert!(pair_sums_fit_i16(&w, 1, 512, 512));
        // In one group (chunks 0 and 1) they would sum to 400 * 127 > i16::MAX.
        let mut w = vec![0i8; 512];
        for chunk in [0usize, 1] {
            for k in 0..2 {
                w[chunk * 32 + k] = 100;
            }
        }
        assert!(!pair_sums_fit_i16(&w, 1, 512, 512));
    }

    #[test]
    fn pair_sums_fit_i16_rejects_shapes_it_cannot_vouch_for() {
        assert!(
            !pair_sums_fit_i16(&[0i8; 48], 1, 48, 48),
            "not a multiple of 32"
        );
        assert!(!pair_sums_fit_i16(&[0i8; 10], 1, 32, 32), "wrong length");
    }

    #[test]
    fn ewm_one_perspective_clamps_and_shifts() {
        const HALF: usize = HIDDEN_SIZE / 2;
        let mut half = [0i16; HIDDEN_SIZE];
        half[0] = 200;
        half[HALF] = 100;
        half[1] = -1_000;
        half[HALF + 1] = 200;
        half[2] = 30_000;
        half[HALF + 2] = 30_000;

        let mut out = [0u8; HALF];
        ewm_one_perspective(&half, &mut out);
        assert_eq!(out[0], 39);
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 126);
        assert!(out[3..].iter().all(|&x| x == 0));
    }
}
