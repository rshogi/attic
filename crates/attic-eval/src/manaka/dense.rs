//! The dense layers' inner product, eight lanes wide.
//!
//! A dot product summed through one running total is a serial dependency chain:
//! every add waits for the one before it, and `f32` addition is not associative,
//! so the compiler may not break it up however wide the registers are. For the
//! first layer here that is `HIDDEN × FEATURE_LEN` adds, each behind the last,
//! and it was the whole of the evaluation's cost.
//!
//! The grouping below is Manaka's own (`manaka_core::eval::ops::dot`): eight
//! independent lanes, `lanes[l] += a[i] · b[i]` for `i ≡ l (mod 8)`, then the
//! lanes added in order from zero, then the tail — what is left after the last
//! whole group of eight — one element at a time. It is the same arithmetic the
//! trainer's forward pass does, so this evaluator and Manaka's now sum in the
//! same order and agree to the last bit wherever the rest of the arithmetic does.
//! The lane count is part of that and is not tunable.
//!
//! The AVX2 path is the scalar one with the eight lanes in one register: a
//! multiply and then an add, never a fused multiply-add, because a fused one
//! rounds once where the lanes round twice and would be a different number. It
//! takes [`ROWS`] rows of the weight matrix at a time against the one input, so
//! that [`ROWS`] independent chains are in flight and the input is loaded once
//! for all of them. Each row's own arithmetic is untouched by that.

/// The lane count: an AVX2 register of `f32`.
const LANES: usize = 8;

/// How many rows of the matrix one pass handles. Eight chains cover the latency of
/// the add behind them, and the input register is loaded once for all of them.
const ROWS: usize = 8;

/// `Σ a[i] · b[i]` in Manaka's grouping, for `R` rows `rows[r]` against one `input`.
///
/// Portable: plain `f32` arithmetic, and the reference the AVX2 path is held to.
#[allow(dead_code)]
fn dots_scalar<const R: usize>(rows: [&[f32]; R], input: &[f32]) -> [f32; R] {
    let n = input.len();
    let chunks = n / LANES;
    let mut out = [0.0f32; R];
    for r in 0..R {
        debug_assert_eq!(rows[r].len(), n);
        let mut lanes = [0.0f32; LANES];
        for chunk in 0..chunks {
            let at = chunk * LANES;
            for lane in 0..LANES {
                lanes[lane] += rows[r][at + lane] * input[at + lane];
            }
        }
        let mut total = 0.0f32;
        for lane in lanes {
            total += lane;
        }
        for i in chunks * LANES..n {
            total += rows[r][i] * input[i];
        }
        out[r] = total;
    }
    out
}

/// [`dots_scalar`] with the lanes in a 256-bit register.
///
/// # Safety
/// The running CPU must support AVX2. The `cfg` that compiles this in names it,
/// and the build is for the host CPU, so it does.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
unsafe fn dots_avx2<const R: usize>(rows: [&[f32]; R], input: &[f32]) -> [f32; R] {
    use std::arch::x86_64::*;

    let n = input.len();
    let chunks = n / LANES;
    for row in rows {
        assert_eq!(row.len(), n, "a row is as long as the input");
    }
    let mut acc = [_mm256_setzero_ps(); R];
    for chunk in 0..chunks {
        let at = chunk * LANES;
        // SAFETY: `at + LANES <= chunks * LANES <= n`, and the input and every row
        // are `n` long (asserted above). The loads are unaligned, so they ask
        // nothing of the address.
        unsafe {
            let x = _mm256_loadu_ps(input.as_ptr().add(at));
            for r in 0..R {
                let w = _mm256_loadu_ps(rows[r].as_ptr().add(at));
                acc[r] = _mm256_add_ps(acc[r], _mm256_mul_ps(w, x));
            }
        }
    }
    let mut out = [0.0f32; R];
    for r in 0..R {
        let mut lanes = [0.0f32; LANES];
        // SAFETY: `lanes` is eight `f32`, which is what one register stores.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc[r]) };
        let mut total = 0.0f32;
        for lane in lanes {
            total += lane;
        }
        for i in chunks * LANES..n {
            total += rows[r][i] * input[i];
        }
        out[r] = total;
    }
    out
}

/// Whichever of the two this build compiled for.
#[inline]
fn dots<const R: usize>(rows: [&[f32]; R], input: &[f32]) -> [f32; R] {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: the `cfg` above is the build enabling AVX2 for the CPU it runs on.
        unsafe { dots_avx2(rows, input) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        dots_scalar(rows, input)
    }
}

/// `Σ a[i] · b[i]` in Manaka's grouping.
pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    dots([a], b)[0]
}

/// One dense layer and its clip: `out[row] = clamp(bias[row] + weights[row] · input, 0, 1)`.
///
/// `weights` is `[out.len()][input.len()]`, and every row's sum is [`dot`]'s.
pub(super) fn dense_clip(weights: &[f32], bias: &[f32], input: &[f32], out: &mut [f32]) {
    let n = input.len();
    debug_assert_eq!(weights.len(), out.len() * n);
    debug_assert_eq!(bias.len(), out.len());
    let row = |at: usize| &weights[at * n..(at + 1) * n];
    let mut at = 0;
    while at + ROWS <= out.len() {
        let sums = dots::<ROWS>(std::array::from_fn(|r| row(at + r)), input);
        for (r, sum) in sums.into_iter().enumerate() {
            out[at + r] = (bias[at + r] + sum).clamp(0.0, 1.0);
        }
        at += ROWS;
    }
    while at < out.len() {
        out[at] = (bias[at] + dot(row(at), input)).clamp(0.0, 1.0);
        at += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed stream of values spread over a few orders of magnitude, so that a
    /// regrouped sum rounds differently from the right one.
    fn values(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = ((state >> 40) as f32) / (1u64 << 24) as f32 - 0.5;
                let scale = [1.0e-3, 1.0, 40.0][((state >> 20) % 3) as usize];
                unit * scale
            })
            .collect()
    }

    /// Every length the shapes use, and the awkward ones either side of a whole
    /// number of lanes: the tail is where a SIMD path most easily goes wrong.
    const LENGTHS: [usize; 11] = [0, 1, 7, 8, 9, 32, 33, 598, 768, 800, 1196];

    #[test]
    fn the_avx2_path_is_the_scalar_one_to_the_last_bit() {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        for n in LENGTHS {
            let input = values(n, 1);
            let data = values(4 * n, 2);
            let rows: [&[f32]; 4] = std::array::from_fn(|r| &data[r * n..(r + 1) * n]);
            let want = dots_scalar(rows, &input);
            // SAFETY: the `cfg` is the build enabling AVX2.
            let got = unsafe { dots_avx2(rows, &input) };
            for r in 0..4 {
                assert_eq!(got[r].to_bits(), want[r].to_bits(), "n={n} row {r}");
            }
        }
    }

    /// Manaka's own `dot`, typed out again so this file does not import it.
    fn manaka_dot(a: &[f32], b: &[f32]) -> f32 {
        let mut lanes = [0.0f32; 8];
        let chunks = a.len() / 8;
        for chunk in 0..chunks {
            let base = chunk * 8;
            for lane in 0..8 {
                lanes[lane] += a[base + lane] * b[base + lane];
            }
        }
        let mut total = 0.0f32;
        for lane in lanes {
            total += lane;
        }
        for index in chunks * 8..a.len() {
            total += a[index] * b[index];
        }
        total
    }

    #[test]
    fn dot_is_manakas_grouping() {
        for n in LENGTHS {
            let (a, b) = (values(n, 3), values(n, 4));
            assert_eq!(dot(&a, &b).to_bits(), manaka_dot(&a, &b).to_bits(), "n={n}");
        }
    }

    /// The rows go through in blocks of eight and then one at a time, and a block
    /// must not change what any row's sum is.
    #[test]
    fn a_layer_is_one_dot_per_row_whatever_the_block() {
        for (rows, n) in [
            (1, 33),
            (4, 8),
            (5, 768),
            (7, 40),
            (32, 768),
            (32, 32),
            (33, 9),
        ] {
            let weights = values(rows * n, 5);
            let bias = values(rows, 6);
            let input = values(n, 7);
            let mut out = vec![0.0f32; rows];
            dense_clip(&weights, &bias, &input, &mut out);
            for row in 0..rows {
                let want = (bias[row] + manaka_dot(&weights[row * n..(row + 1) * n], &input))
                    .clamp(0.0, 1.0);
                assert_eq!(out[row].to_bits(), want.to_bits(), "{rows}x{n} row {row}");
            }
        }
    }
}
