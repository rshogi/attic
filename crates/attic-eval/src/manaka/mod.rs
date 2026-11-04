//! Manaka's own evaluator, in the shared `nn.bin` container.
//!
//! The file is the same container every network here arrives in; what differs
//! is the architecture string, which leads with `ModelType=SYM…` and spells a
//! structure YaneuraOu has no reader for. That is the whole identification —
//! one format, and the header says which shape is inside.
//!
//! Nothing below it is NNUE. The shapes served here (`sym_*`, listed with the
//! dimensions of each in `shape.rs`) borrow YaneuraOu's *feature numbering* (see
//! `features.rs`, shared with the standard shapes) and nothing else: the
//! accumulator carries power sums rather than a single sum, and each block is
//! divided by its own root-mean-square where NNUE clips. The `sym_token_*`
//! shapes borrow not even that: they number Manaka's own token space, in
//! `tokens.rs`, and update their accumulator from the move's delta instead of
//! by diffing two feature lists.
//!
//! ## Three encodings of one layout
//!
//! `ModelType` says how the numbers are stored, and `loader.rs` has a reader for
//! each: `SYMscaled` (`i16` arrays with a scale per group, what the exporter
//! writes by default), `SYMf32` (the trainer's own numbers) and `SYM`, the
//! first encoding, which fixed one scale per kind of array and clamps any
//! trained weight past 1.98. All three are turned into the same `f32` tables at
//! load, so the forward pass below does not know which it was given, and runs
//! no integer arithmetic in any of them.
//!
//! ## Adding a shape
//!
//! A shape that differs from these in dimensions alone is one `sym_*` entry in
//! `Cargo.toml` (pass-through included, in every crate that re-exports the
//! shapes), one row in `DIMS` (`shape.rs`), and one name in the guard in `types.rs`. Nothing
//! else here changes: every length below is derived from the row, and the
//! architecture string is assembled from the same constants. A new feature
//! *space* is more than that — `sym_token` had to touch `loader.rs` (the
//! `Features=` word) and `accumulator.rs` (how a move is applied) as well — but
//! another shape in a space that exists is not.
//!
//! ## Why the legacy encoding needs no per-order correction
//!
//! Its `embed` is stored at `EMBED_SCALE` (`loader.rs`), so `p_k` arrives at
//! `S^k` — a different scale per order, which would normally have to be tracked
//! and undone one at a time. It does not, because the next thing each block
//! meets is a division by its own root-mean-square, and a constant factor
//! divides out of that exactly: `e_k / rms(e_k)` is the same number whether
//! `e_k` was built from `v` or from `v·S`. Every array is therefore divisible by
//! the one constant it was multiplied by, which is what that reader does on the
//! way in. `SYMscaled` has a scale per embedding *channel*, which does not divide
//! out of a sum over channels, so its reader divides each element by its own
//! channel's scale at load and the accumulator never sees a scale at all.
//!
//! ## The forward pass
//!
//! Per perspective, over the forty active features:
//!
//! 1. Power sums `p_k[c] = Σ_i embed[token_i][c]^k` for `k = 1..=order`.
//! 2. Newton's identities turn those into elementary symmetric polynomials:
//!    `k·e_k = Σ_{j=1..k} (−1)^{j−1} · e_{k−j} · p_j`, with `e_0 = 1`.
//! 3. Each `e_k` block is RMS-normalised and scaled by its own `gamma`.
//!
//! The two perspectives' blocks are concatenated — side to move first — into
//! `2 · order · dim` values, which one or two hidden layers (each followed by a
//! ReLU) and a `tanh` value row reduce to a scalar in `[-1, 1]`. The value row
//! reads whichever hidden layer is last, which is [`HEAD`] wide.
//!
//! Ported from `manaka-core`: `params.rs` (the file), `accum/newton.rs` (step
//! 2) and `eval.rs` (steps 1 and 3, and the head).

mod accumulator;
mod loader;
mod reader;
mod shape;
#[cfg(test)]
mod tests;

use attic_state::{Color, Position};

pub use accumulator::{Accumulator, FinnyCache};
pub use loader::{load_network, load_network_with_warnings};
pub use shape::{DIM, FEATURE_LEN, HEAD, HIDDEN, HIDDEN2, ORDER, SECOND_LAYER, TOKENS};

/// Guard on the root-mean-square, matching `manaka_core::eval::RMS_EPS`. Both
/// forwards floor at the same number or they are not the same function.
const RMS_EPS: f32 = 1e-6;

/// A loaded network, with the tables this shape actually reads.
///
/// The pair, moment and field tables are absent by construction — Manaka's
/// shape grammar refuses them outside its own token space, because each wants
/// a row per token and these spaces have up to 125,389 of them — so the file
/// carries them at length zero and this does not name them at all.
#[derive(Debug)]
pub struct SymNetwork {
    /// `[TOKENS][DIM]`.
    pub embed: Vec<f32>,
    /// `[ORDER][DIM]` — the per-channel scale each normalised block is given.
    pub gamma: Vec<f32>,
    /// `[HIDDEN][FEATURE_LEN]`.
    pub w1: Vec<f32>,
    /// `[HIDDEN]`.
    pub b1: Vec<f32>,
    /// `[HIDDEN2][HIDDEN]`. Empty when the shape has one hidden layer.
    pub w2: Vec<f32>,
    /// `[HIDDEN2]`. Empty when the shape has one hidden layer.
    pub b2: Vec<f32>,
    /// `[HEAD]`.
    pub value_w: Vec<f32>,
    pub value_b: f32,
}

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
fn pre_from_sums(net: &SymNetwork, acc: &Accumulator, stm: Color) -> f32 {
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
fn value_from_sums(net: &SymNetwork, acc: &Accumulator, stm: Color) -> f32 {
    pre_from_sums(net, acc, stm).tanh()
}

/// Static evaluation of `pos` from the side to move's perspective, in
/// `[-1, 1]`, rebuilding the accumulator from scratch.
///
/// # Panics
/// Panics if `pos` is missing either king (via feature extraction).
pub fn evaluate_value(net: &SymNetwork, pos: &Position) -> f32 {
    let mut acc = Accumulator::new();
    acc.refresh(net, pos);
    value_from_sums(net, &acc, pos.side_to_move())
}

/// The value row's pre-activation, before the `tanh`: [`evaluate_value`] is its
/// `tanh`. For measuring a network against its trainer, where the `tanh` hides
/// the difference between two large values.
///
/// # Panics
/// Panics if `pos` is missing either king.
pub fn evaluate_pre(net: &SymNetwork, pos: &Position) -> f32 {
    let mut acc = Accumulator::new();
    acc.refresh(net, pos);
    pre_from_sums(net, &acc, pos.side_to_move())
}

/// What a unit value is worth before [`crate::fv_scale`] divides it.
///
/// `127 · 64`, which is what NNUE's output layer carries: its `i32` is a
/// unit-scaled score times the activation scale times the weight scale, and
/// `ComputeScore` divides that by `FV_SCALE`. Matching it here is what puts
/// the two evaluators on one scale — at the default 16, both hand the search
/// about 508 per unit — so a search tuned against one is not retuned for the
/// other, and two networks of different kinds can be compared on a ladder.
pub const RAW_SCALE: f32 = 127.0 * 64.0;

/// Static evaluation of `pos`, positive when the side to move is better.
///
/// `tanh` bounds the value at one, so what the search sees is bounded too:
/// `±RAW_SCALE / fv_scale()`. That is the whole of the difference from NNUE,
/// whose output is unbounded and whose scale therefore has more work to do.
///
/// # Panics
/// Panics if `pos` is missing either king.
pub fn evaluate(net: &SymNetwork, pos: &Position) -> i32 {
    let raw = evaluate_value(net, pos) * RAW_SCALE;
    (raw / crate::fv_scale() as f32) as i32
}

impl SymNetwork {
    /// A second copy for another NUMA node.
    ///
    /// A plain clone: these tables are ordinary `Vec`s rather than views into
    /// a shared arena, so there is no backing allocation to duplicate by hand.
    pub fn replicate(&self) -> Self {
        SymNetwork {
            embed: self.embed.clone(),
            gamma: self.gamma.clone(),
            w1: self.w1.clone(),
            b1: self.b1.clone(),
            w2: self.w2.clone(),
            b2: self.b2.clone(),
            value_w: self.value_w.clone(),
            value_b: self.value_b,
        }
    }
}

/// Static evaluation from an already-built accumulator.
///
/// The entry point for search: no feature extraction and no power sums, just
/// the polynomials, the norm and the head.
pub fn evaluate_with(net: &SymNetwork, acc: &Accumulator, pos: &Position) -> i32 {
    let raw = value_from_sums(net, acc, pos.side_to_move()) * RAW_SCALE;
    (raw / crate::fv_scale() as f32) as i32
}
