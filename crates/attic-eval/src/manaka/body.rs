//! The sections after the header, in each of the two encodings.
//!
//! Each encoding has a reader that takes the body from the first section to the
//! last, so a new encoding is one more function here and one more arm of the
//! `match` in `loader::parse_network`; nothing else knows how a number is
//! stored. Whatever the encoding, the result is the same `f32` tables, which is
//! what keeps the forward pass free of any of them.

use crate::types::NnueError;

use super::reader::Reader;
use super::shape::SQUARES;
use super::{
    DIM, FEATURE_LEN, FIELD, HAS_FIELD, HAS_MOMENT, HAS_PAIR, HEAD, HIDDEN, HIDDEN2, MOMENT,
    MOMENT_LEN, ORDER, PAIR, PAIR_ROWS, SymNetwork, TOKENS,
};

/// `SYNNscaled`: section 1 is the embedding and section 2 is everything dense,
/// with a scale per group on the arrays that have one. `gamma`, the biases and
/// the value row are `f32`: the value row because its entries multiply hidden
/// units, so a step that is invisible in `W1` is amplified there, and the rest
/// because they are a few thousand numbers.
///
/// The optional sections follow in the order of their numbers, each only when its
/// knob is on: the pair table (3, with its gains), the second moment (4) and the
/// square field (5).
///
/// The pair table is a `SYNNscaled` array like the embedding, with a scale per
/// channel.
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
    }
    let (moment_gamma, moment_embed) = read_moment(r, hash, true)?;
    let (field_gamma, field) = read_field(r, hash, true)?;
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
        qw1: None,
    })
}

/// Section 4, the second moment: the gain on each entry of the upper triangle
/// (`f32`) and the embedding the moment is built from, `[tokens][m]`, which is
/// `SYNNscaled` with a scale per channel when `scaled`. Absent, and so empty,
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
/// A `SYNNscaled` field has a scale per *channel*, not per entry of a row: the
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

/// `SYNNf32`: the trainer's own numbers, verbatim.
pub(super) fn read_f32(r: &mut Reader<'_>, hash: u32) -> Result<SymNetwork, NnueError> {
    r.expect_hash(hash.wrapping_add(1), "feature transformer")?;
    let embed = r.f32s(TOKENS * DIM)?;

    r.expect_hash(hash.wrapping_add(2), "dense stack")?;
    let gamma = r.f32s(ORDER * DIM)?;
    let w1 = r.f32s(HIDDEN * FEATURE_LEN)?;
    let b1 = r.f32s(HIDDEN)?;
    let w2 = r.f32s(HIDDEN2 * HIDDEN)?;
    let b2 = r.f32s(HIDDEN2)?;
    let value_w = r.f32s(HEAD)?;
    let value_b = r.f32s(1)?[0];

    let (mut pair_gamma, mut pair) = (Vec::new(), Vec::new());
    if HAS_PAIR {
        r.expect_hash(hash.wrapping_add(3), "pair table")?;
        pair_gamma = r.f32s(PAIR)?;
        pair = r.f32s(PAIR_ROWS * PAIR)?;
    }
    let (moment_gamma, moment_embed) = read_moment(r, hash, false)?;
    let (field_gamma, field) = read_field(r, hash, false)?;
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
        qw1: None,
    })
}
