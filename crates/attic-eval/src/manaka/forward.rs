//! From an accumulator to a value: the polynomials, the norms, the hidden
//! layers and the value row.
//!
//! Nothing here is incremental. The accumulator carries sums over tokens, and
//! everything below is what cannot be kept as one — a polynomial of the sums, a
//! root-mean-square, a matrix product — so it runs once per evaluation.

use attic_state::Color;

use super::{Accumulator, DIM, FEATURE_LEN, HIDDEN, HIDDEN2, ORDER, SECOND_LAYER, SymNetwork};

/// Guard on the root-mean-square, matching `manaka_core::eval::RMS_EPS`. Both
/// forwards floor at the same number or they are not the same function.
pub(super) const RMS_EPS: f32 = 1e-6;

/// Newton's identities, channel by channel: `k·e_k = Σ_{j=1..k} (−1)^{j−1} ·
/// e_{k−j} · p_j`, with `e_0 = 1`.
///
/// Not incrementally updatable, which is the reason the accumulator keeps `p`
/// and this runs once per evaluation rather than once per token change.
fn elementary_from_power(p: &[f32]) -> [f32; ORDER * DIM] {
    debug_assert_eq!(p.len(), ORDER * DIM);
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
            out[(k - 1) * DIM + channel] = e[k];
        }
    }
    out
}

/// One perspective's contribution to the head's input: each order block
/// RMS-normalised, then scaled by its own gamma.
fn blocks_from_sums(net: &SymNetwork, sums: &[f32], out: &mut [f32]) {
    debug_assert_eq!(out.len(), ORDER * DIM);
    let e = elementary_from_power(sums);
    for k in 0..ORDER {
        let block = &e[k * DIM..(k + 1) * DIM];
        let mean_square: f32 = block.iter().map(|v| v * v).sum::<f32>() / DIM as f32;
        let scale = 1.0 / (mean_square + RMS_EPS).sqrt();
        let gamma = &net.gamma[k * DIM..(k + 1) * DIM];
        let out = &mut out[k * DIM..(k + 1) * DIM];
        for channel in 0..DIM {
            // Left to right, the same grouping Manaka's own forward uses. f32
            // addition is not associative and neither is this product, so
            // folding `scale * gamma` first would be a different number.
            out[channel] = block[channel] * scale * gamma[channel];
        }
    }
}

/// One dense layer and its ReLU: `out[row] = max(0, bias[row] + weights[row] · input)`.
///
/// The sum is left to right, the same grouping the first layer had before there
/// was a second, so a shape with one layer produces the numbers it always did.
fn dense_relu(weights: &[f32], bias: &[f32], input: &[f32], out: &mut [f32]) {
    debug_assert_eq!(weights.len(), out.len() * input.len());
    for (row, slot) in out.iter_mut().enumerate() {
        let weights = &weights[row * input.len()..(row + 1) * input.len()];
        let sum: f32 = weights.iter().zip(input).map(|(w, x)| w * x).sum();
        *slot = (bias[row] + sum).max(0.0);
    }
}

/// The head up to the `tanh`, from an accumulator: polynomials, norm, the
/// hidden layers, and the value row's pre-activation.
pub(super) fn pre_from_sums(net: &SymNetwork, acc: &Accumulator, stm: Color) -> f32 {
    let mut features = [0.0f32; FEATURE_LEN];
    // Side to move first: the head was trained reading its own frame in the
    // first half, and swapping them is a different network.
    for (half, frame) in [stm, stm.flip()].into_iter().enumerate() {
        blocks_from_sums(
            net,
            acc.perspective(frame),
            &mut features[half * ORDER * DIM..(half + 1) * ORDER * DIM],
        );
    }

    let mut hidden = [0.0f32; HIDDEN];
    dense_relu(&net.w1, &net.b1, &features, &mut hidden);

    // Zero-sized, and never written, when the shape has one layer: `HIDDEN2`
    // is a constant, so the branch below is decided before the code is.
    let mut second = [0.0f32; HIDDEN2];
    let head: &[f32] = if SECOND_LAYER {
        dense_relu(&net.w2, &net.b2, &hidden, &mut second);
        &second
    } else {
        &hidden
    };

    let sum: f32 = net.value_w.iter().zip(head).map(|(w, h)| w * h).sum();
    net.value_b + sum
}

/// The head, from an accumulator: [`pre_from_sums`] and a `tanh`.
pub(super) fn value_from_sums(net: &SymNetwork, acc: &Accumulator, stm: Color) -> f32 {
    pre_from_sums(net, acc, stm).tanh()
}
