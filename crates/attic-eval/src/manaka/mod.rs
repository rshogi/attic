//! Manaka's own evaluator, `SYNN`, in the shared `nn.bin` container.
//!
//! SYNN puts SYM's front half in front of NNUE's back half. The front is
//! Manaka's: a token embedding and its power sums, and optionally a pair table, a
//! second moment and a square field. The back is NNUE's: each block scaled by its
//! gain and clipped to `[0, 1]`, two clipped dense layers, and a linear value.
//!
//! The file is the same container every network here arrives in; what differs
//! is the architecture string, which leads with `ModelType=SYNN…` and spells a
//! structure YaneuraOu has no reader for. That is the whole identification —
//! one format, and the header says which shape is inside.
//!
//! Nothing below it is NNUE's feature transformer. The shapes served here
//! (`synn_*`, listed with the dimensions of each in `shape.rs`) borrow
//! YaneuraOu's *feature numbering* (see `features.rs`, shared with the standard
//! shapes) and nothing else: the accumulator carries power sums rather than a
//! single sum. The `synn_token_*` shapes borrow not even that: they number
//! Manaka's own token space, in `tokens.rs`, and update their accumulator from
//! the move's delta instead of by diffing two feature lists.
//!
//! ## Two encodings of one layout
//!
//! `ModelType` says how the numbers are stored, and `body.rs` has a reader for
//! each: `SYNNscaled` (`i16` arrays with a scale per group, what the exporter
//! writes by default) and `SYNNf32` (the trainer's own numbers). Both are turned
//! into the same `f32` tables at load, so the forward pass below does not know
//! which it was given, and runs no integer arithmetic in either.
//!
//! ## Adding a shape
//!
//! A shape that differs from these in dimensions or in which of the pair, moment
//! and field knobs it turns on is one `synn_*` entry in `Cargo.toml` (pass-through
//! included, in every crate that re-exports the shapes), and one row in `DIMS` and
//! one name in the count beside it (`shape.rs`). Nothing else here changes: every
//! length below is derived from the row, and the architecture string is assembled
//! from the same constants. A new feature *space* is more than that —
//! `sym_token` had to touch `loader.rs` (the `Features=` word) and
//! `accumulator.rs` (how a move is applied) as well — and so is a new *kind of
//! block*, as the pair table was: that is a table in `SymNetwork`, a section in
//! each reader, a sum in the accumulator and a step in `forward.rs`.
//!
//! ## The forward pass
//!
//! Per perspective, over the forty active features:
//!
//! 1. Power sums `p_k[c] = Σ_i embed[token_i][c]^k` for `k = 1..=order`.
//! 2. Newton's identities turn those into elementary symmetric polynomials:
//!    `k·e_k = Σ_{j=1..k} (−1)^{j−1} · e_{k−j} · p_j`, with `e_0 = 1`.
//! 3. Each `e_k` block is scaled by its own `gamma` and clipped to `[0, 1]`.
//! 4. A shape with a pair table adds its sum, `Σ_{a<b} pair[row(a, b)]` over the
//!    forty tokens' 780 pairs, scaled by its gains and clipped like an order block.
//! 5. A shape with a second moment adds the upper triangle of `Σ v vᵀ`, where `v`
//!    is a token's `MOMENT`-channel row of a table of its own: the cross-channel
//!    terms the element-wise power sums cannot reach. It is scaled and clipped as
//!    one block, with a gain per entry.
//! 6. A shape with a square field adds `A[s][c] = Σ field[token][s·FIELD + c]`,
//!    81 slots of `FIELD` channels, scaled and clipped as one block with a gain per
//!    channel that all 81 slots share.
//!
//! The two perspectives' blocks are concatenated — side to move first — into
//! `FEATURE_LEN` values, which two hidden layers (each clipped to `[0, 1]`) and a
//! linear value row reduce to a scalar. The value row reads the second layer,
//! which is [`HEAD`] wide. There is no norm and no `tanh`.
//!
//! Its value is in NNUE's units. The row's output is multiplied by [`RAW_SCALE`]
//! and divided by `fv_scale` as NNUE's is, so one unit is about 508 at the default
//! scale. Nothing clamps it here, and NNUE's `evaluate_with` does not clamp its
//! score either: what bounds it is the same thing, a row of weights over
//! activations that are themselves clipped to `[0, 1]`, so it is at most
//! `|value_b| + Σ|value_w|`. Where the search clamps a static evaluation, it does
//! so on what it reads from `evaluate_with` (`to_corrected_static_eval` in
//! `attic-search`), the same for every shape.
//!
//! Ported from `manaka-core`: `params.rs` (the file), `accum/newton.rs` (step
//! 2) and `eval.rs` (steps 1 and 3 to 6, and the head, in its `nnue` arm). The
//! format is specified in tideborn's `docs/network.md`.

mod accumulator;
mod body;
mod cache;
mod dense;
#[cfg(feature = "sym_token")]
mod field;
mod forward;
mod loader;
#[cfg(feature = "sym_token")]
mod moment;
#[cfg(feature = "sym_token")]
mod pair;
mod qdense;
mod reader;
mod shape;
#[cfg(test)]
mod tests;

use attic_state::Position;

pub use accumulator::Accumulator;
pub use cache::FinnyCache;
use forward::value_from_sums;
pub use loader::{load_network, load_network_with_warnings};
pub use qdense::QuantW1;
pub use shape::{
    BLOCK_LEN, DIM, FEATURE_LEN, FIELD, FIELD_AT, FIELD_LEN, HAS_FIELD, HAS_MOMENT, HAS_PAIR, HEAD,
    HIDDEN, HIDDEN2, MOMENT, MOMENT_AT, MOMENT_LEN, ORDER, PAIR, PAIR_AT, PAIR_ROWS, POWER_LEN,
    TOKENS,
};

/// A loaded network, with the tables this shape actually reads.
///
/// The pair, moment and field tables are absent in every shape that has none of
/// them: the file carries a table that is off at length zero and this holds it as
/// an empty `Vec`. Manaka's shape grammar refuses all three outside its own token
/// space, because each wants a row per token (or per pair of them) and the other
/// spaces have up to 125,389 tokens.
#[derive(Debug)]
pub struct SymNetwork {
    /// `[TOKENS][DIM]`.
    pub embed: Vec<f32>,
    /// `[ORDER][DIM]` — the per-channel gain each block is given before it is
    /// clipped.
    pub gamma: Vec<f32>,
    /// `[HIDDEN][FEATURE_LEN]`.
    pub w1: Vec<f32>,
    /// `[HIDDEN]`.
    pub b1: Vec<f32>,
    /// `[HIDDEN2][HIDDEN]`.
    pub w2: Vec<f32>,
    /// `[HIDDEN2]`.
    pub b2: Vec<f32>,
    /// `[HEAD]`.
    pub value_w: Vec<f32>,
    pub value_b: f32,
    /// `[PAIR]` — the gain on each pair channel.
    pub pair_gamma: Vec<f32>,
    /// `[PAIR_ROWS][PAIR]`. Empty without a pair table.
    pub pair: Vec<f32>,
    /// `[MOMENT_LEN]` — the gain on each entry of the second moment's upper
    /// triangle. Empty without a second moment.
    pub moment_gamma: Vec<f32>,
    /// `[TOKENS][MOMENT]`. Empty without a second moment.
    pub moment_embed: Vec<f32>,
    /// `[FIELD]` — the gain on each channel of the square field, shared by all 81
    /// squares. Empty without a field.
    pub field_gamma: Vec<f32>,
    /// `[TOKENS][SQUARES][FIELD]`: the entry for token `t`, square slot `s` and
    /// channel `c` is at `t · FIELD_LEN + s · FIELD + c`. Empty without a field.
    pub field: Vec<f32>,
    /// `w1` as `i16`, when the first layer is to run on an integer dot product.
    /// `None` is the `f32` layer, which is the definition and the default; see
    /// [`SymNetwork::use_i16_w1`].
    pub qw1: Option<QuantW1>,
}

/// Static evaluation of `pos` from the side to move's perspective, in NNUE's
/// unit before [`RAW_SCALE`]: the value row's output as it is, which is not
/// confined to `[-1, 1]`. Rebuilds the accumulator from scratch.
///
/// # Panics
/// Panics if `pos` is missing either king (via feature extraction).
pub fn evaluate_value(net: &SymNetwork, pos: &Position) -> f32 {
    let mut acc = Accumulator::new();
    acc.refresh(net, pos);
    value_from_sums(net, &acc, pos.side_to_move())
}

/// The value row's pre-activation. There is no `tanh` after it, so this is the
/// same number [`evaluate_value`] returns; it stays for the tools that print the
/// two as separate columns (`examples/sym_dump.rs`, `scripts/manaka_compare.py`).
///
/// # Panics
/// Panics if `pos` is missing either king.
pub fn evaluate_pre(net: &SymNetwork, pos: &Position) -> f32 {
    evaluate_value(net, pos)
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
/// The value is linear and is not bounded at one, only by its own weights, as
/// NNUE's output is.
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
            moment_gamma: self.moment_gamma.clone(),
            moment_embed: self.moment_embed.clone(),
            field_gamma: self.field_gamma.clone(),
            field: self.field.clone(),
            qw1: self.qw1.clone(),
        }
    }

    /// Run the first layer on `i16` weights and an integer dot product (`true`), or
    /// back on `f32` (`false`).
    ///
    /// The `i16` layer is the faster one and not the same number: it carries the
    /// rounding of the weights to a row's levels and of the input to 1/4096, about
    /// 0.3 cp on average on trained weights, so it is a switch and not the
    /// definition. The `f32` weights stay in the network either way.
    pub fn use_i16_w1(&mut self, on: bool) {
        self.qw1 = on.then(|| QuantW1::new(&self.w1, HIDDEN, FEATURE_LEN));
    }
}

/// The value from an already-built accumulator, before it is turned into a
/// score: what [`evaluate_with`] scales, for measuring an accumulator that was
/// carried by derivation against one that was rebuilt.
pub fn evaluate_value_with(net: &SymNetwork, acc: &Accumulator, pos: &Position) -> f32 {
    value_from_sums(net, acc, pos.side_to_move())
}

/// Static evaluation from an already-built accumulator.
///
/// The entry point for search: no feature extraction and no power sums, just
/// the polynomials, the clips and the head.
pub fn evaluate_with(net: &SymNetwork, acc: &Accumulator, pos: &Position) -> i32 {
    let raw = evaluate_value_with(net, acc, pos) * RAW_SCALE;
    (raw / crate::fv_scale() as f32) as i32
}
