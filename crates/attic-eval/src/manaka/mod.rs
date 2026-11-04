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
//! A shape that differs from these in dimensions or in which of the pair,
//! log-sum-exp, kept-magnitude and split knobs it turns on is one `sym_*` entry
//! in `Cargo.toml` (pass-through included, in every crate that re-exports the
//! shapes), and one row in `DIMS` and one name in the count beside it
//! (`shape.rs`). Nothing else here changes: every length below is derived from
//! the row, and the architecture string is assembled from the same constants. A new
//! feature *space* is more than that — `sym_token` had to touch `loader.rs` (the
//! `Features=` word) and `accumulator.rs` (how a move is applied) as well — and
//! so is a new *kind of block*, as the pair table was: that is a table in
//! `SymNetwork`, a section in each reader, a sum in the accumulator and a step
//! in `forward.rs`.
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
//! 4. A shape with a pair table adds its sum, `Σ_{a<b} pair[row(a, b)]` over the
//!    forty tokens' 780 pairs, which is normalised like an order block in its
//!    first `MEAN` channels and read as `ln Σ exp θ` in its last `LSE`. With
//!    `ks` each norm's divisor is also reported, as a logarithm.
//!
//! The two perspectives' blocks are concatenated — side to move first — into
//! `FEATURE_LEN` values, which one or two hidden layers (each followed by a
//! ReLU) and a `tanh` value row reduce to a scalar in `[-1, 1]`. The value row
//! reads whichever hidden layer is last, which is [`HEAD`] wide. A `split` shape
//! takes step 1 twice, over the frame's own pieces and over the opponent's.
//!
//! Ported from `manaka-core`: `params.rs` (the file), `accum/newton.rs` (step
//! 2) and `eval.rs` (steps 1, 3 and 4, and the head). The format is specified
//! in tideborn's `docs/network.md`.

mod accumulator;
mod forward;
mod loader;
#[cfg(feature = "sym_token")]
mod pair;
mod reader;
mod shape;
#[cfg(test)]
mod tests;

use attic_state::Position;

pub use accumulator::{Accumulator, FinnyCache};
use forward::{pre_from_sums, value_from_sums};
pub use loader::{load_network, load_network_with_warnings};
pub use shape::{
    BLOCK_LEN, DIM, FEATURE_LEN, GROUPS, HAS_PAIR, HEAD, HIDDEN, HIDDEN2, KS, LSE, MEAN, ORDER,
    PAIR, PAIR_AT, PAIR_ROWS, POWER_LEN, SCALE_AT, SCALE_LEN, SECOND_LAYER, SPLIT, TOKENS,
};

/// A loaded network, with the tables this shape actually reads.
///
/// The moment and field tables are absent by construction, as is the pair table
/// in every shape that has none: the file carries a table that is off at length
/// zero and this holds it as an empty `Vec`. Manaka's shape grammar refuses all
/// three outside its own token space, because each wants a row per token (or per
/// pair of them) and the other spaces have up to 125,389 tokens.
#[derive(Debug)]
pub struct SymNetwork {
    /// `[TOKENS][DIM]`.
    pub embed: Vec<f32>,
    /// `[ORDER][DIM]` — the per-channel scale each normalised block is given,
    /// shared by both groups of a `split` shape.
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
    /// `[PAIR]` — the gain on each pair channel, mean and log-sum-exp alike.
    pub pair_gamma: Vec<f32>,
    /// `[PAIR_ROWS][PAIR]`. In the last `LSE` channels of a row it holds
    /// `exp θ`, which is what the accumulator adds: a `SYMscaled` file stores
    /// `θ` and the reader exponentiates it once at load. Empty without a pair
    /// table.
    pub pair: Vec<f32>,
    /// `[SCALE_LEN]` — the gain on each kept magnitude. Empty without `ks`.
    pub scale_gamma: Vec<f32>,
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
            pair_gamma: self.pair_gamma.clone(),
            pair: self.pair.clone(),
            scale_gamma: self.scale_gamma.clone(),
        }
    }
}

/// The value in `[-1, 1]` from an already-built accumulator, before it is turned
/// into a score: what [`evaluate_with`] scales, for measuring an accumulator
/// that was carried by derivation against one that was rebuilt.
pub fn evaluate_value_with(net: &SymNetwork, acc: &Accumulator, pos: &Position) -> f32 {
    value_from_sums(net, acc, pos.side_to_move())
}

/// Static evaluation from an already-built accumulator.
///
/// The entry point for search: no feature extraction and no power sums, just
/// the polynomials, the norm and the head.
pub fn evaluate_with(net: &SymNetwork, acc: &Accumulator, pos: &Position) -> i32 {
    let raw = evaluate_value_with(net, acc, pos) * RAW_SCALE;
    (raw / crate::fv_scale() as f32) as i32
}
