//! The sections after the header, in each of the three encodings.
//!
//! Each encoding has a reader that takes the body from the first section to the
//! last, so a new encoding is one more function here and one more arm of the
//! `match` in `loader::parse_network`; nothing else knows how a number is
//! stored. Whatever the encoding, the result is the same `f32` tables, which is
//! what keeps the forward pass free of any of them.

use crate::types::NnueError;

use super::loader::{EMBED_SCALE, WEIGHT_SCALE};
use super::reader::Reader;
use super::shape::SQUARES;
use super::{
    DIM, FEATURE_LEN, FIELD, HAS_FIELD, HAS_MOMENT, HAS_PAIR, HEAD, HIDDEN, HIDDEN2, KS, MEAN,
    MOMENT, MOMENT_LEN, ORDER, PAIR, PAIR_ROWS, SCALE_LEN, SymNetwork, TOKENS,
};

/// `SYMscaled`: section 1 is the embedding and section 2 is everything dense,
/// with a scale per group on the arrays that have one. `gamma`, the biases and
/// the value row are `f32`: the value row because its entries multiply hidden
/// units that are not bounded, so a step that is invisible in `W1` is amplified
/// there, and the rest because they are a few thousand numbers.
///
/// The optional sections follow in the order of their numbers, each only when its
/// knob is on: the pair table (3, with its gains), the second moment (4), the
/// square field (5) and the kept magnitudes' gains (6).
///
/// The pair table is a `SYMscaled` array like the embedding, with a scale per
/// channel, and its log-sum-exp channels are stored as `θ`, not the `exp θ` the
/// accumulator adds: `exp θ` spans `e⁻⁶ … e⁶`, which no linear integer scale
/// holds at both ends, while `θ` is clamped to ±6 by the trainer and quantises
/// like any other channel. They are exponentiated here once, so that from this
/// point both encodings carry `exp θ`.
pub(super) fn read_scaled(r: &mut Reader<'_>, hash: u32) -> Result<SymNetwork, NnueError> {
    r.expect_hash(hash.wrapping_add(1), "feature transformer")?;
    let embed = r.scaled_per_column(TOKENS, DIM, "embed")?;

    r.expect_hash(hash.wrapping_add(2), "dense stack")?;
    let gamma = r.f32s(ORDER * DIM)?;
    let w1 = r.scaled_per_row(HIDDEN, FEATURE_LEN, "w1")?;
    let b1 = r.f32s(HIDDEN)?;
    let w2 = r.scaled_per_row(HIDDEN2, HIDDEN, "w2")?;
    let b2 = r.f32s(HIDDEN2)?;
    let value_w = r.f32s(HEAD)?;
    let value_b = r.f32s(1)?[0];

    let (mut pair_gamma, mut pair) = (Vec::new(), Vec::new());
    if HAS_PAIR {
        r.expect_hash(hash.wrapping_add(3), "pair table")?;
        pair_gamma = r.f32s(PAIR)?;
        pair = r.scaled_per_column(PAIR_ROWS, PAIR, "pair")?;
        exponentiate_thetas(&mut pair, PAIR, MEAN);
    }
    let (moment_gamma, moment_embed) = read_moment(r, hash, true)?;
    let (field_gamma, field) = read_field(r, hash, true)?;
    let scale_gamma = read_scale_gamma(r, hash)?;
    Ok(SymNetwork {
        embed,
        gamma,
        w1,
        b1,
        w2,
        b2,
        value_w,
        value_b,
        pair_gamma,
        pair,
        moment_gamma,
        moment_embed,
        field_gamma,
        field,
        scale_gamma,
    })
}

/// `θ → exp θ` in the channels from `first` on of every `width`-wide row of
/// `table`, which is what the accumulator adds.
///
/// The width is an argument and not the constant it always is, because a shape
/// whose width is zero is compiled with this call in it, and a constant chunk
/// size of zero is something clippy refuses to see.
fn exponentiate_thetas(table: &mut [f32], width: usize, first: usize) {
    for row in table.chunks_exact_mut(width) {
        for theta in &mut row[first..] {
            *theta = theta.exp();
        }
    }
}

/// Section 4, the second moment: the gain on each entry of the upper triangle
/// (`f32`) and the embedding the moment is built from, `[tokens][m]`, which is
/// `SYMscaled` with a scale per channel when `scaled`. Absent, and so empty,
/// without a moment.
fn read_moment(
    r: &mut Reader<'_>,
    hash: u32,
    scaled: bool,
) -> Result<(Vec<f32>, Vec<f32>), NnueError> {
    if !HAS_MOMENT {
        return Ok((Vec::new(), Vec::new()));
    }
    r.expect_hash(hash.wrapping_add(4), "second moment")?;
    let gamma = r.f32s(MOMENT_LEN)?;
    let embed = if scaled {
        r.scaled_per_column(TOKENS, MOMENT, "moment_embed")?
    } else {
        r.f32s(TOKENS * MOMENT)?
    };
    Ok((gamma, embed))
}

/// Section 5, the square field: a gain per channel (`f32`, one for all 81
/// squares) and the table `[token][square][channel]`.
///
/// A `SYMscaled` field has a scale per *channel*, not per entry of a row: the
/// group of element `i` is `i mod f`. Read as `tokens · 81` rows of `f` columns it
/// is exactly what `scaled_per_column` reads, and a reader that gave each of the
/// `81 · f` positions in a row a scale of its own would be reading a file that
/// does not exist.
fn read_field(
    r: &mut Reader<'_>,
    hash: u32,
    scaled: bool,
) -> Result<(Vec<f32>, Vec<f32>), NnueError> {
    if !HAS_FIELD {
        return Ok((Vec::new(), Vec::new()));
    }
    r.expect_hash(hash.wrapping_add(5), "square field")?;
    let gamma = r.f32s(FIELD)?;
    let table = if scaled {
        r.scaled_per_column(TOKENS * SQUARES, FIELD, "field")?
    } else {
        r.f32s(TOKENS * SQUARES * FIELD)?
    };
    Ok((gamma, table))
}

/// Section 6, the gains on the kept magnitudes, which are `f32` in every
/// encoding that has the section. Absent, and so empty, without `ks`.
fn read_scale_gamma(r: &mut Reader<'_>, hash: u32) -> Result<Vec<f32>, NnueError> {
    if !KS {
        return Ok(Vec::new());
    }
    r.expect_hash(hash.wrapping_add(6), "kept magnitudes")?;
    r.f32s(SCALE_LEN)
}

/// `SYMf32`: the trainer's own numbers, verbatim.
pub(super) fn read_f32(r: &mut Reader<'_>, hash: u32) -> Result<SymNetwork, NnueError> {
    r.expect_hash(hash.wrapping_add(1), "feature transformer")?;
    let embed = r.f32s(TOKENS * DIM)?;

    r.expect_hash(hash.wrapping_add(2), "dense stack")?;
    let gamma = r.f32s(ORDER * DIM)?;
    // The second layer's arrays sit between the first's and the value row, and
    // are zero long for a shape with one layer, so they read nothing there.
    let w1 = r.f32s(HIDDEN * FEATURE_LEN)?;
    let b1 = r.f32s(HIDDEN)?;
    let w2 = r.f32s(HIDDEN2 * HIDDEN)?;
    let b2 = r.f32s(HIDDEN2)?;
    let value_w = r.f32s(HEAD)?;
    let value_b = r.f32s(1)?[0];

    // Already `exp θ` in the last channels: the trainer bakes the exponential in
    // so that its accumulator can stay a plain add, and this file is its table.
    let (mut pair_gamma, mut pair) = (Vec::new(), Vec::new());
    if HAS_PAIR {
        r.expect_hash(hash.wrapping_add(3), "pair table")?;
        pair_gamma = r.f32s(PAIR)?;
        pair = r.f32s(PAIR_ROWS * PAIR)?;
    }
    let (moment_gamma, moment_embed) = read_moment(r, hash, false)?;
    let (field_gamma, field) = read_field(r, hash, false)?;
    let scale_gamma = read_scale_gamma(r, hash)?;
    Ok(SymNetwork {
        embed,
        gamma,
        w1,
        b1,
        w2,
        b2,
        value_w,
        value_b,
        pair_gamma,
        pair,
        moment_gamma,
        moment_embed,
        field_gamma,
        field,
        scale_gamma,
    })
}

/// `SYM`, the first quantised encoding: one scale per kind of array, as NNUE
/// does. An `i8` at 64 holds ±1.98 and trained weights reach 28, so the writer
/// clamped them; this reads what is in the file and cannot know what was lost.
///
/// `embed` is stored at [`EMBED_SCALE`], so `p_k` arrives at `S^k` — a
/// different scale per order, which would normally have to be tracked and
/// undone one at a time. It does not, because the next thing each block meets
/// is a division by its own root-mean-square, and a constant factor divides
/// out of that exactly. Dividing each array by the one constant it was
/// multiplied by on the way in is therefore all there is to do.
pub(super) fn read_legacy(r: &mut Reader<'_>, hash: u32) -> Result<SymNetwork, NnueError> {
    r.expect_hash(hash.wrapping_add(1), "feature transformer")?;
    let embed = r.i16s_scaled(TOKENS * DIM, EMBED_SCALE)?;

    r.expect_hash(hash.wrapping_add(2), "dense stack")?;
    // `gamma` was never quantised in any encoding: it applies immediately
    // after the norm, the one place in the forward where a rounding error is
    // not divided back out again.
    let gamma = r.f32s(ORDER * DIM)?;
    let w1 = r.i8s_scaled(HIDDEN * FEATURE_LEN, WEIGHT_SCALE)?;
    let b1 = r.i32s_scaled(HIDDEN, WEIGHT_SCALE)?;
    let w2 = r.i8s_scaled(HIDDEN2 * HIDDEN, WEIGHT_SCALE)?;
    let b2 = r.i32s_scaled(HIDDEN2, WEIGHT_SCALE)?;
    let value_w = r.i8s_scaled(HEAD, WEIGHT_SCALE)?;
    let value_b = r.i32s_scaled(1, WEIGHT_SCALE)?[0];
    // No pair table, moment, field or kept magnitudes: `Encoding::exists` refuses
    // the encoding for a shape that has any, so this is only reached without.
    Ok(SymNetwork {
        embed,
        gamma,
        w1,
        b1,
        w2,
        b2,
        value_w,
        value_b,
        pair_gamma: Vec::new(),
        pair: Vec::new(),
        moment_gamma: Vec::new(),
        moment_embed: Vec::new(),
        field_gamma: Vec::new(),
        field: Vec::new(),
        scale_gamma: Vec::new(),
    })
}
