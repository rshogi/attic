//! From an accumulator to a value: the polynomials, the norms, the pair, moment,
//! field and kept-magnitude blocks, the hidden layers and the value row.
//!
//! Nothing here is incremental. The accumulator carries sums over tokens, and
//! everything below is what cannot be kept as one — a polynomial of the sums, a
//! root-mean-square, a logarithm, a matrix product — so it runs once per
//! evaluation.

use attic_state::Color;

use super::{
    Accumulator, BLOCK_LEN, DIM, FEATURE_LEN, FIELD_AT, FIELD_LEN, GROUPS, HAS_FIELD, HAS_MOMENT,
    HAS_PAIR, HIDDEN, HIDDEN2, KS, MEAN, MOMENT_AT, MOMENT_LEN, ORDER, PAIR, PAIR_AT, POWER_LEN,
    SCALE_AT, SCALE_LEN, SECOND_LAYER, SymNetwork,
};

/// Guard on the root-mean-square, matching `manaka_core::eval::RMS_EPS`. Both
/// forwards floor at the same number or they are not the same function.
pub(super) const RMS_EPS: f32 = 1e-6;

/// Floor under a log-sum-exp channel before its logarithm, matching
/// `manaka_core::params::LSE_FLOOR`. A table that was never exponentiated could
/// otherwise hand `ln` a negative sum, and the result is a NaN that poisons
/// everything downstream.
pub(super) const LSE_FLOOR: f32 = 1e-6;

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

/// The mean of the squares of `values`, plus the guard, rooted: what a norm
/// divides by, and what `ks` keeps the logarithm of.
fn root_mean_square(values: &[f32]) -> f32 {
    let mean_square: f32 = values.iter().map(|v| v * v).sum::<f32>() / values.len() as f32;
    (mean_square + RMS_EPS).sqrt()
}

/// What the accumulator holds for one perspective, as the forward pass reads it.
/// Each is empty in a shape that lacks the block.
struct Sums<'a> {
    power: &'a [f32],
    pair: &'a [f32],
    moment: &'a [f32],
    field: &'a [f32],
}

/// `values` scaled by `scale` and by `gamma` into `out`, in the order the
/// reference multiplies them: `sum × (1/r) × gamma`, left to right.
fn scale_by_gamma(values: &[f32], scale: f32, gamma: &[f32], out: &mut [f32]) {
    for ((slot, &sum), &gamma) in out.iter_mut().zip(values).zip(gamma) {
        *slot = sum * scale * gamma;
    }
}

/// [`scale_by_gamma`] for a block laid out `[square][channel]`, whose gain is one
/// per channel and so repeats for every square.
///
/// The width is `gamma`'s length and not the constant it is, because a shape
/// without a field is compiled with this call in it and a constant chunk size of
/// zero is something clippy refuses to see.
fn scale_by_channel_gamma(values: &[f32], scale: f32, gamma: &[f32], out: &mut [f32]) {
    let width = gamma.len();
    for (slots, sums) in out.chunks_exact_mut(width).zip(values.chunks_exact(width)) {
        scale_by_gamma(sums, scale, gamma, slots);
    }
}

/// One perspective's share of the head's input, in the layout `shape.rs` gives:
/// each group's order blocks RMS-normalised and scaled by their own gamma, then
/// the pair channels, the second moment and the square field, then the kept
/// magnitudes.
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

    // The magnitudes the norms divide away, in the order the blocks are
    // emitted: every order of the first group, then of the second, then the
    // pair's, the moment's and the field's. Read only by `ks`.
    let mut kept = [0.0f32; SCALE_LEN];
    let mut blocks = 0;

    for group in 0..GROUPS {
        let start = group * ORDER * DIM;
        let e = elementary_from_power(&power[start..start + ORDER * DIM]);
        for k in 0..ORDER {
            let block = &e[k * DIM..(k + 1) * DIM];
            let rms = root_mean_square(block);
            let scale = 1.0 / rms;
            if KS {
                kept[blocks] = rms;
                blocks += 1;
            }
            let gamma = &net.gamma[k * DIM..(k + 1) * DIM];
            let out = &mut out[start + k * DIM..start + (k + 1) * DIM];
            for channel in 0..DIM {
                // Left to right, the same grouping Manaka's own forward uses.
                // f32 addition is not associative and neither is this product,
                // so folding `scale * gamma` first would be a different number.
                out[channel] = block[channel] * scale * gamma[channel];
            }
        }
    }

    if HAS_PAIR {
        // The mean channels are a sum of 780 table rows and are normalised
        // like an order block. The log-sum-exp ones are not, and must not be:
        // `ln Σ exp` is already bounded and its magnitude is the half of the
        // answer a norm would discard. They are not in the norm's mean either,
        // so a shape with both kinds prices each on its own terms.
        let (mean_out, lse_out) = out[PAIR_AT..PAIR_AT + PAIR].split_at_mut(MEAN);
        if MEAN != 0 {
            let rms = root_mean_square(&pair[..MEAN]);
            let scale = 1.0 / rms;
            if KS {
                kept[blocks] = rms;
                blocks += 1;
            }
            for ((slot, &sum), &gamma) in mean_out.iter_mut().zip(pair).zip(&net.pair_gamma) {
                *slot = sum * scale * gamma;
            }
        }
        let lse_in = pair[MEAN..].iter().zip(&net.pair_gamma[MEAN..]);
        for (slot, (&sum, &gamma)) in lse_out.iter_mut().zip(lse_in) {
            *slot = sum.max(LSE_FLOOR).ln() * gamma;
        }
    }

    if HAS_MOMENT {
        // One norm over all the entries of the triangle, and a gain for each
        // entry. The diagonal is in the mean once, as it is in the triangle.
        let rms = root_mean_square(moment);
        if KS {
            kept[blocks] = rms;
            blocks += 1;
        }
        let out = &mut out[MOMENT_AT..MOMENT_AT + MOMENT_LEN];
        scale_by_gamma(moment, 1.0 / rms, &net.moment_gamma, out);
    }

    if HAS_FIELD {
        // One norm over all 81 squares at once, and a gain for each channel
        // that every square shares.
        let rms = root_mean_square(field);
        if KS {
            kept[blocks] = rms;
            blocks += 1;
        }
        let out = &mut out[FIELD_AT..FIELD_AT + FIELD_LEN];
        scale_by_channel_gamma(field, 1.0 / rms, &net.field_gamma, out);
    }

    if KS {
        debug_assert_eq!(blocks, SCALE_LEN);
        // What each norm above divided away. Forty tokens go in whatever the
        // position, so this is not the set-size correction a norm usually
        // removes: it is the position moving pieces between the board and the
        // hands, which the head would otherwise never be shown. The logarithm
        // is of the root-mean-square of the *true-scale* sum, which these are:
        // the readers divide every table by its scale on the way in.
        let out = &mut out[SCALE_AT..SCALE_AT + SCALE_LEN];
        for (slot, (&rms, &gamma)) in out.iter_mut().zip(kept.iter().zip(&net.scale_gamma)) {
            *slot = rms.ln() * gamma;
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
