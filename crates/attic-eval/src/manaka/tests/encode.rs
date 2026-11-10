//! A network written out as a file, the way Manaka's exporter writes one, in each
//! of the two encodings.
//!
//! The scales are computed here from the format's description (`docs/network.md`,
//! "How an array's numbers are stored") and not by the reader, so that a reader
//! which disagreed with the description would be caught by a file it did not
//! write.

use std::sync::OnceLock;

use super::super::loader::{Encoding, VERSION, arch_hash, expected_arch};
use super::super::{
    DIM, FEATURE_LEN, FIELD, HAS_FIELD, HAS_MOMENT, HAS_PAIR, HIDDEN, MOMENT, PAIR, SymNetwork,
};
use super::fixtures::synthetic;

/// The column an element of a row-major array of `width` columns is in. The
/// width is an argument and not the constant it is: `i % PAIR` written out is a
/// remainder by zero in a shape without a pair table, which rustc refuses to
/// compile even in a branch that shape never takes.
pub(super) fn column_of(width: usize) -> impl Fn(usize) -> usize {
    move |i| i % width
}

/// The encoded file of the synthetic net, once per encoding. A pair table makes
/// each of these tens to hundreds of megabytes, and nearly every test wants one.
pub(super) fn encoded(encoding: Encoding) -> &'static [u8] {
    static FILES: [OnceLock<Vec<u8>>; 2] = [OnceLock::new(), OnceLock::new()];
    let slot = Encoding::ALL
        .iter()
        .position(|e| *e == encoding)
        .expect("a known encoding");
    FILES[slot].get_or_init(|| encode(synthetic(), encoding))
}

/// The scale `export-nnue` gives one group of a `SYNNscaled` array: the largest
/// `f32` for which the group's largest element still fits `i16`, so that
/// element lands on ±32767 and nothing clamps. An all-zero group has scale 1.
pub(super) fn group_scale<'a>(group: impl Iterator<Item = &'a f32>) -> f32 {
    let max = group.fold(0.0f32, |m, v| m.max(v.abs()));
    if max == 0.0 {
        return 1.0;
    }
    let mut scale = 32767.0 / max;
    while max * scale > 32767.0 {
        scale = f32::from_bits(scale.to_bits() - 1);
    }
    scale
}

/// The scales of a `rows × cols` array grouped by column (`i mod cols`).
///
/// One pass over the rows rather than a strided pass per column: the pair table
/// is `PAIR` columns of 2.7 million, and walking it `PAIR` times is the slowest
/// thing a test here would do.
pub(super) fn column_scales(values: &[f32], cols: usize) -> Vec<f32> {
    let mut largest = vec![0.0f32; cols];
    for row in values.chunks_exact(cols) {
        for (max, v) in largest.iter_mut().zip(row) {
            *max = max.max(v.abs());
        }
    }
    largest
        .into_iter()
        .map(|max| group_scale(std::iter::once(&max)))
        .collect()
}

/// The scales of a `rows × cols` array grouped by row (`i div cols`).
pub(super) fn row_scales(values: &[f32], cols: usize) -> Vec<f32> {
    values
        .chunks_exact(cols)
        .map(|r| group_scale(r.iter()))
        .collect()
}

fn put_f32s(out: &mut Vec<u8>, values: &[f32]) {
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// A `SYNNscaled` array: its scales, then `round(x · scale)` as `i16`, half away
/// from zero. Nothing at all for an empty array.
fn put_scaled(out: &mut Vec<u8>, values: &[f32], scales: &[f32], group: impl Fn(usize) -> usize) {
    if values.is_empty() {
        return;
    }
    put_f32s(out, scales);
    for (i, v) in values.iter().enumerate() {
        let q = (v * scales[group(i)]).round();
        assert!(
            q.abs() <= 32767.0,
            "a scale that clamps is not the exporter's"
        );
        out.extend_from_slice(&(q as i16).to_le_bytes());
    }
}

/// `net` written the way `export-nnue` writes it in `encoding`.
pub(super) fn encode(net: &SymNetwork, encoding: Encoding) -> Vec<u8> {
    let arch = expected_arch(encoding);
    let hash = arch_hash(&arch);
    let mut out = Vec::new();
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&hash.to_le_bytes());
    out.extend_from_slice(&(arch.len() as u32).to_le_bytes());
    out.extend_from_slice(arch.as_bytes());

    out.extend_from_slice(&hash.wrapping_add(1).to_le_bytes());
    match encoding {
        Encoding::Scaled => {
            put_scaled(&mut out, &net.embed, &column_scales(&net.embed, DIM), |i| {
                i % DIM
            })
        }
        Encoding::F32 => put_f32s(&mut out, &net.embed),
    }

    out.extend_from_slice(&hash.wrapping_add(2).to_le_bytes());
    put_f32s(&mut out, &net.gamma);
    match encoding {
        Encoding::Scaled => {
            put_scaled(&mut out, &net.w1, &row_scales(&net.w1, FEATURE_LEN), |i| {
                i / FEATURE_LEN
            });
            put_f32s(&mut out, &net.b1);
            put_scaled(&mut out, &net.w2, &row_scales(&net.w2, HIDDEN), |i| {
                i / HIDDEN
            });
            put_f32s(&mut out, &net.b2);
            put_f32s(&mut out, &net.value_w);
            put_f32s(&mut out, &[net.value_b]);
        }
        Encoding::F32 => {
            put_f32s(&mut out, &net.w1);
            put_f32s(&mut out, &net.b1);
            put_f32s(&mut out, &net.w2);
            put_f32s(&mut out, &net.b2);
            put_f32s(&mut out, &net.value_w);
            put_f32s(&mut out, &[net.value_b]);
        }
    }

    let scaled = encoding == Encoding::Scaled;
    if HAS_PAIR {
        out.extend_from_slice(&hash.wrapping_add(3).to_le_bytes());
        put_f32s(&mut out, &net.pair_gamma);
        put_table(&mut out, &net.pair, PAIR, scaled);
    }
    if HAS_MOMENT {
        out.extend_from_slice(&hash.wrapping_add(4).to_le_bytes());
        put_f32s(&mut out, &net.moment_gamma);
        put_table(&mut out, &net.moment_embed, MOMENT, scaled);
    }
    if HAS_FIELD {
        out.extend_from_slice(&hash.wrapping_add(5).to_le_bytes());
        put_f32s(&mut out, &net.field_gamma);
        // `[token][square][channel]`, and the group is the channel: the
        // innermost index, whatever the table's rows are taken to be.
        put_table(&mut out, &net.field, FIELD, scaled);
    }
    out
}

/// A table whose group is its last index, `cols` wide: `f32`, or `SYNNscaled` with
/// a scale per column.
fn put_table(out: &mut Vec<u8>, values: &[f32], cols: usize, scaled: bool) {
    if scaled {
        put_scaled(out, values, &column_scales(values, cols), column_of(cols));
    } else {
        put_f32s(out, values);
    }
}
