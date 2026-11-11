//! The sections after the header.
//!
//! There is one way to store a number, so there is one reader. The arrays a
//! position's evaluation reads a row of or multiplies by a vector — `embed`, `W1`,
//! `W2`, `pair`, `moment_embed` and `field` — are halves, two bytes each with
//! nothing in front of them. Every other number — the gains, the biases and the
//! value row — is `f32`. All of it but the pair table is turned into `f32` tables
//! at load, so the forward pass below does not know how a number was stored, and
//! runs no integer arithmetic on the way in. The pair table stays halves, with both
//! frames to a row (`pair_table.rs`), and is widened as a derive adds it up.

use crate::types::NnueError;

use super::pair_table::PairTable;
use super::reader::Reader;
use super::shape::SQUARES;
use super::{
    DIM, FEATURE_LEN, FIELD, FirstLayer, HAS_FIELD, HAS_MOMENT, HAS_PAIR, HEAD, HIDDEN, HIDDEN2,
    MOMENT, MOMENT_LEN, ORDER, PAIR, SymNetwork, TOKENS,
};

/// The body of the file, from the first section to the last.
///
/// Section 1 is the embedding, as halves. Section 2 is everything dense: `gamma`
/// (`f32`), `W1` (halves), `b1` (`f32`), `W2` (halves), `b2` (`f32`), the value row
/// (`f32`) and its bias (`f32`). The optional sections follow in the order of their
/// numbers, each only when its knob is on: the pair table (3, its gains and then its
/// table), the second moment (4) and the square field (5).
///
/// The pair table is read as it is. A shape here has no log-sum-exp channel, so
/// every channel is a mean and is added as it stands.
pub(super) fn read_body(r: &mut Reader<'_>, hash: u32) -> Result<SymNetwork, NnueError> {
    r.expect_hash(hash.wrapping_add(1), "feature transformer")?;
    let embed = r.halves(TOKENS * DIM, "embed")?;

    r.expect_hash(hash.wrapping_add(2), "dense stack")?;
    let gamma = r.f32s(ORDER * DIM)?;
    let w1 = r.halves(HIDDEN * FEATURE_LEN, "w1")?;
    let b1 = r.f32s(HIDDEN)?;
    let w2 = r.halves(HIDDEN2 * HIDDEN, "w2")?;
    let b2 = r.f32s(HIDDEN2)?;
    let value_w = r.f32s(HEAD)?;
    let value_b = r.f32s(1)?[0];

    let (mut pair_gamma, mut pair) = (Vec::new(), PairTable::empty());
    if HAS_PAIR {
        r.expect_hash(hash.wrapping_add(3), "pair table")?;
        pair_gamma = r.f32s(PAIR)?;
        pair = read_pair_table(r)?;
    }
    let (moment_gamma, moment_embed) = read_moment(r, hash)?;
    let (field_gamma, field) = read_field(r, hash)?;
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
        first: FirstLayer::F32,
    })
}

/// The pair table's halves, kept as halves with both frames to a row.
#[cfg(feature = "sym_token")]
fn read_pair_table(r: &mut Reader<'_>) -> Result<PairTable, NnueError> {
    Ok(PairTable::from_le_bytes(
        r.half_bytes(super::PAIR_ROWS * PAIR, "pair")?,
    ))
}

/// Outside the token space there is no pair table to read: `HAS_PAIR` is false
/// there, so this is never called.
#[cfg(not(feature = "sym_token"))]
fn read_pair_table(_: &mut Reader<'_>) -> Result<PairTable, NnueError> {
    unreachable!("a pair table exists only in the token space")
}

/// Section 4, the second moment: the gain on each entry of the upper triangle
/// (`f32`) and the embedding the moment is built from, `[tokens][m]` (halves).
/// Absent, and so empty, without a moment.
fn read_moment(r: &mut Reader<'_>, hash: u32) -> Result<(Vec<f32>, Vec<f32>), NnueError> {
    if !HAS_MOMENT {
        return Ok((Vec::new(), Vec::new()));
    }
    r.expect_hash(hash.wrapping_add(4), "second moment")?;
    let gamma = r.f32s(MOMENT_LEN)?;
    let embed = r.halves(TOKENS * MOMENT, "moment_embed")?;
    Ok((gamma, embed))
}

/// Section 5, the square field: a gain per channel (`f32`, one for all 81 squares)
/// and the table `[token][square][channel]` (halves).
fn read_field(r: &mut Reader<'_>, hash: u32) -> Result<(Vec<f32>, Vec<f32>), NnueError> {
    if !HAS_FIELD {
        return Ok((Vec::new(), Vec::new()));
    }
    r.expect_hash(hash.wrapping_add(5), "square field")?;
    let gamma = r.f32s(FIELD)?;
    let table = r.halves(TOKENS * SQUARES * FIELD, "field")?;
    Ok((gamma, table))
}
