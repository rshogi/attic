//! From an accumulator to a value: the polynomials, the clipped blocks (the power
//! sums, and the pair, moment and field blocks a shape has), the two hidden layers
//! and the value row.
//!
//! Nothing here is incremental. The accumulator carries sums over tokens, and
//! everything below is what cannot be kept as one — a polynomial of the sums, a
//! clip, a matrix product — so it runs once per evaluation.
//!
//! The arithmetic after the sums is NNUE's. Each block is multiplied by its gain
//! and clipped to `[0, 1]`, each hidden layer clips to `[0, 1]`, and the value is
//! the row's output as it is: there is no norm, no `tanh` and no unbounded
//! activation anywhere in it.

use attic_state::Color;

use super::dense;
use super::{
    Accumulator, BLOCK_LEN, DIM, FEATURE_LEN, FIELD_AT, FIELD_LEN, FirstLayer, HAS_FIELD,
    HAS_MOMENT, HAS_PAIR, HIDDEN, HIDDEN2, MOMENT_AT, MOMENT_LEN, ORDER, PAIR, PAIR_AT, POWER_LEN,
    SymNetwork,
};

pub(super) use dense::dense_clip;

/// The group of channels `power_blocks` works on at once: an AVX2 register of `f32`.
const LANES: usize = 8;

const _: () = assert!(
    DIM.is_multiple_of(LANES),
    "channels come in whole groups of eight"
);

/// Newton's identities, then each order's gain and clip.
///
/// `k·e_k = Σ_{j=1..k} (−1)^{j−1} · e_{k−j} · p_j`, with `e_0 = 1`, and the block
/// that goes on to the first hidden layer is `clamp(e_k · gamma_k, 0, 1)`.
///
/// Not incrementally updatable, which is the reason the accumulator keeps `p` and
/// this runs once per evaluation rather than once per token change.
///
/// One pass over groups of eight channels: a group takes its `p_1 … p_ORDER` in,
/// works every order out in registers and writes the clipped blocks, where doing
/// each order and the clip as a pass of their own is `ORDER` times the loads and
/// stores. Every channel does the operations the channel-at-a-time form did, in
/// the same order — `acc` from zero, `+= sign · e_{k−j} · p_j` for `j = 1..k`, then
/// `/ k`, then `· gamma` and the clip — so the result is the same bits, and the
/// tests hold both paths to that form.
#[inline]
fn power_blocks(p: &[f32], gamma: &[f32], out: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: the `cfg` above is the build enabling AVX2 for the CPU it runs on.
        unsafe { power_blocks_avx2(p, gamma, out) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        power_blocks_scalar(p, gamma, out);
    }
}

/// [`power_blocks`] in plain `f32`: the reference the AVX2 path is held to.
///
/// Index loops throughout, on purpose: this is the plainest thing that says what
/// the identities say, which is what makes agreeing with it mean something.
#[allow(dead_code, clippy::needless_range_loop)]
fn power_blocks_scalar(p: &[f32], gamma: &[f32], out: &mut [f32]) {
    assert_eq!(p.len(), ORDER * DIM);
    assert_eq!(gamma.len(), ORDER * DIM);
    assert_eq!(out.len(), ORDER * DIM);
    for at in (0..DIM).step_by(LANES) {
        // `e[0]` is one in every channel.
        let mut e = [[1.0f32; LANES]; ORDER + 1];
        for k in 1..=ORDER {
            let mut acc = [0.0f32; LANES];
            let mut sign = 1.0f32;
            for j in 1..=k {
                for lane in 0..LANES {
                    acc[lane] += sign * e[k - j][lane] * p[(j - 1) * DIM + at + lane];
                }
                sign = -sign;
            }
            for lane in 0..LANES {
                e[k][lane] = acc[lane] / k as f32;
            }
        }
        for k in 1..=ORDER {
            for lane in 0..LANES {
                let base = (k - 1) * DIM + at + lane;
                out[base] = (e[k][lane] * gamma[base]).clamp(0.0, 1.0);
            }
        }
    }
}

/// [`power_blocks_scalar`] with the eight channels of a group in one register.
///
/// `sign · e · p` is `±(e · p)` exactly, so the odd terms are added and the even
/// ones subtracted, which is the same sum with one multiplication fewer. The clip
/// is `f32::clamp`'s: below zero is zero, above one is one, and anything else —
/// negative zero, `NaN` — is left as it is, which is what a compare and a blend
/// do and what `_mm256_max_ps` and `_mm256_min_ps` do not.
///
/// # Safety
/// The running CPU must support AVX2. The `cfg` that compiles this in names it,
/// and the build is for the host CPU, so it does.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(dead_code, clippy::needless_range_loop)]
#[target_feature(enable = "avx2")]
unsafe fn power_blocks_avx2(p: &[f32], gamma: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    assert_eq!(p.len(), ORDER * DIM);
    assert_eq!(gamma.len(), ORDER * DIM);
    assert_eq!(out.len(), ORDER * DIM);
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    for at in (0..DIM).step_by(LANES) {
        let mut e = [one; ORDER + 1];
        for k in 1..=ORDER {
            let mut acc = zero;
            for j in 1..=k {
                // SAFETY: `(j − 1) · DIM + at + 8 <= ORDER · DIM`, which is what `p`
                // holds (asserted above); the load is unaligned.
                let power = unsafe { _mm256_loadu_ps(p.as_ptr().add((j - 1) * DIM + at)) };
                // `e_0` is one, so its term is `p_k` as it is.
                let term = if j == k {
                    power
                } else {
                    _mm256_mul_ps(e[k - j], power)
                };
                acc = if j % 2 == 1 {
                    _mm256_add_ps(acc, term)
                } else {
                    _mm256_sub_ps(acc, term)
                };
            }
            e[k] = divide_by(acc, k);
        }
        for k in 1..=ORDER {
            let base = (k - 1) * DIM + at;
            // SAFETY: `base + 8 <= ORDER · DIM`, which is what `gamma` and `out`
            // hold (asserted above); the access is unaligned.
            unsafe {
                let x = _mm256_mul_ps(e[k], _mm256_loadu_ps(gamma.as_ptr().add(base)));
                let at_least_zero = _mm256_blendv_ps(x, zero, _mm256_cmp_ps::<_CMP_LT_OQ>(x, zero));
                let at_most_one = _mm256_blendv_ps(
                    at_least_zero,
                    one,
                    _mm256_cmp_ps::<_CMP_GT_OQ>(at_least_zero, one),
                );
                _mm256_storeu_ps(out.as_mut_ptr().add(base), at_most_one);
            }
        }
    }
}

/// `acc / k` in every lane, without a division where one is not needed.
///
/// A division by one is `acc` itself, and a division by any other power of two
/// is a multiplication by its reciprocal, which is exact: both are the same real
/// number correctly rounded, so the bits are the division's, for zeros,
/// subnormals and `NaN` as well. Only the other `k` (three, in the shapes there
/// are) divide. A division is several times the latency and a fraction of the
/// throughput of a multiplication, and this runs `ORDER` times for every group of
/// eight channels of both perspectives.
///
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
fn divide_by(acc: std::arch::x86_64::__m256, k: usize) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    if k == 1 {
        acc
    } else if k.is_power_of_two() {
        _mm256_mul_ps(acc, _mm256_set1_ps(1.0 / k as f32))
    } else {
        _mm256_div_ps(acc, _mm256_set1_ps(k as f32))
    }
}

/// What the accumulator holds for one perspective, as the forward pass reads it.
/// Each is empty in a shape that lacks the block.
struct Sums<'a> {
    power: &'a [f32],
    pair: &'a [f32],
    moment: &'a [f32],
    field: &'a [f32],
}

/// `values` multiplied by `gamma` and clipped to `[0, 1]` into `out`: what every
/// block is put through before the first hidden layer.
///
/// The product is `sum × gamma` and nothing else: nothing is divided by a
/// root-mean-square first. It is Manaka's `(e × gamma).clamp(0, 1)` from the
/// `nnue` arm of `eval.rs`, in the same order.
fn clip_by_gamma(values: &[f32], gamma: &[f32], out: &mut [f32]) {
    for ((slot, &sum), &gamma) in out.iter_mut().zip(values).zip(gamma) {
        *slot = (sum * gamma).clamp(0.0, 1.0);
    }
}

/// [`clip_by_gamma`] for a block laid out `[square][channel]`, whose gain is one
/// per channel and so repeats for every square. The width is `gamma`'s length and
/// not the constant it is, because a shape without a field is compiled with this
/// call in it and a constant chunk size of zero is something clippy refuses to
/// see.
fn clip_by_channel_gamma(values: &[f32], gamma: &[f32], out: &mut [f32]) {
    let width = gamma.len();
    for (slots, sums) in out.chunks_exact_mut(width).zip(values.chunks_exact(width)) {
        clip_by_gamma(sums, gamma, slots);
    }
}

/// One perspective's share of the head's input, in the layout `shape.rs` gives:
/// each order's block, then the pair channels, the second moment and the square
/// field, each scaled by its own gamma and clipped to `[0, 1]`.
fn block_from_sums(net: &SymNetwork, sums: &Sums<'_>, out: &mut [f32]) {
    let Sums {
        power,
        pair,
        moment,
        field,
    } = *sums;
    debug_assert_eq!(power.len(), POWER_LEN);
    debug_assert_eq!(pair.len(), PAIR);
    debug_assert_eq!(moment.len(), MOMENT_LEN);
    debug_assert_eq!(field.len(), FIELD_LEN);
    debug_assert_eq!(out.len(), BLOCK_LEN);

    power_blocks(power, &net.gamma, &mut out[..POWER_LEN]);

    if HAS_PAIR {
        clip_by_gamma(pair, &net.pair_gamma, &mut out[PAIR_AT..PAIR_AT + PAIR]);
    }

    if HAS_MOMENT {
        // One gain for each entry of the triangle. The diagonal is in once, as it
        // is in the triangle.
        let out = &mut out[MOMENT_AT..MOMENT_AT + MOMENT_LEN];
        clip_by_gamma(moment, &net.moment_gamma, out);
    }

    if HAS_FIELD {
        // One gain for each channel, which every one of the 81 squares shares.
        let out = &mut out[FIELD_AT..FIELD_AT + FIELD_LEN];
        clip_by_channel_gamma(field, &net.field_gamma, out);
    }
}

/// What the first hidden layer reads: both perspectives' blocks, side to move
/// first, each in the layout `shape.rs` gives. Split out of [`value_from_sums`] so
/// that a test can look at the head's input without a second implementation of
/// how it is assembled.
pub(super) fn features_from_sums(
    net: &SymNetwork,
    acc: &Accumulator,
    stm: Color,
    features: &mut [f32; FEATURE_LEN],
) {
    // Side to move first: the head was trained reading its own frame in the
    // first half, and swapping them is a different network.
    for (half, frame) in [stm, stm.flip()].into_iter().enumerate() {
        let (pair, moment, field) = (acc.pair(frame), acc.moment(frame), acc.field(frame));
        let sums = Sums {
            power: acc.perspective(frame),
            pair: &pair,
            moment: &moment,
            field: &field,
        };
        block_from_sums(
            net,
            &sums,
            &mut features[half * BLOCK_LEN..(half + 1) * BLOCK_LEN],
        );
    }
}

/// The head, from an accumulator: the clipped blocks, the two clipped hidden
/// layers, and the value row's output as it is.
///
/// It is not bounded by one. What bounds it is the same thing that bounds NNUE's
/// output: a row of weights over activations that are themselves clipped to
/// `[0, 1]`.
pub(super) fn value_from_sums(net: &SymNetwork, acc: &Accumulator, stm: Color) -> f32 {
    let mut features = [0.0f32; FEATURE_LEN];
    features_from_sums(net, acc, stm, &mut features);

    let mut hidden = [0.0f32; HIDDEN];
    match &net.first {
        FirstLayer::F32 => dense_clip(&net.w1, &net.b1, &features, &mut hidden),
        FirstLayer::I16(q) => q.layer_clip(&features, &net.b1, &mut hidden),
        FirstLayer::I8(q) => q.layer_clip(&features, &net.b1, &mut hidden),
    }

    let mut second = [0.0f32; HIDDEN2];
    dense_clip(&net.w2, &net.b2, &hidden, &mut second);

    // The value row's sum is the dense layers' too: eight lanes, as Manaka's is.
    net.value_b + dense::dot(&net.value_w, &second)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Newton identities as they were written with the channel as the outer
    /// loop, then the gain and the clip as a pass of their own: the form the
    /// fused pass replaced, kept to hold it to the same bits.
    fn channel_outer(p: &[f32], gamma: &[f32]) -> [f32; ORDER * DIM] {
        let mut out = [0.0f32; ORDER * DIM];
        for channel in 0..DIM {
            let mut e = [1.0f32; ORDER + 1];
            for k in 1..=ORDER {
                let mut acc = 0.0f32;
                let mut sign = 1.0f32;
                for j in 1..=k {
                    acc += sign * e[k - j] * p[(j - 1) * DIM + channel];
                    sign = -sign;
                }
                e[k] = acc / k as f32;
                let at = (k - 1) * DIM + channel;
                out[at] = (e[k] * gamma[at]).clamp(0.0, 1.0);
            }
        }
        out
    }

    #[test]
    fn the_fused_blocks_are_the_channel_at_a_time_ones_to_the_last_bit() {
        let mut state = 9u64;
        let mut next = |scale: f32| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Sums over forty tokens, so a few units either way, with the
            // occasional exact zero and negative zero.
            match (state >> 33) % 17 {
                0 => 0.0,
                1 => -0.0,
                _ => ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale,
            }
        };
        for round in 0..50 {
            let p: Vec<f32> = (0..ORDER * DIM).map(|_| next(12.0)).collect();
            let gamma: Vec<f32> = (0..ORDER * DIM).map(|_| next(3.0)).collect();
            let want = channel_outer(&p, &gamma);
            let mut scalar = [0.0f32; ORDER * DIM];
            power_blocks_scalar(&p, &gamma, &mut scalar);
            let mut dispatched = [0.0f32; ORDER * DIM];
            power_blocks(&p, &gamma, &mut dispatched);
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            let avx2 = {
                let mut out = [0.0f32; ORDER * DIM];
                // SAFETY: the `cfg` is the build enabling AVX2.
                unsafe { power_blocks_avx2(&p, &gamma, &mut out) };
                out
            };
            for (at, w) in want.iter().enumerate() {
                assert_eq!(
                    scalar[at].to_bits(),
                    w.to_bits(),
                    "scalar, round {round} entry {at}"
                );
                assert_eq!(
                    dispatched[at].to_bits(),
                    w.to_bits(),
                    "dispatched, round {round} entry {at}"
                );
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                assert_eq!(
                    avx2[at].to_bits(),
                    w.to_bits(),
                    "avx2, round {round} entry {at}"
                );
            }
        }
    }
}
