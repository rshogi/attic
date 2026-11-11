//! A network written out as a file, the way Manaka's exporter writes one.
//!
//! Written from the format's description (`docs/network.md`, "How an array's
//! numbers are stored") and not from the reader, and the rounding is done from the
//! definition of a half rather than by shifting bits: the number of steps of the
//! half's own size that the value is, rounded to the nearest whole one with ties
//! to even. So a reader that disagreed with the description is caught by a file it
//! did not write.

use std::sync::OnceLock;

use super::super::loader::{VERSION, arch_hash, expected_arch};
use super::super::{HAS_FIELD, HAS_MOMENT, HAS_PAIR, SymNetwork};
use super::fixtures::synthetic;

/// The column an element of a row-major array of `width` columns is in. The
/// width is an argument and not the constant it is: `i % FIELD` written out is a
/// remainder by zero in a shape without a field, which rustc refuses to compile
/// even in a branch that shape never takes.
pub(super) fn column_of(width: usize) -> impl Fn(usize) -> usize {
    move |i| i % width
}

/// The encoded file of the synthetic net. A pair table makes it tens to hundreds
/// of megabytes, and nearly every test wants one.
pub(super) fn encoded() -> &'static [u8] {
    static FILE: OnceLock<Vec<u8>> = OnceLock::new();
    FILE.get_or_init(|| encode(synthetic()))
}

/// The nearest half to `x`, ties to even, as its bits.
///
/// A half of binade `e` (that is, of magnitude in `[2^e, 2^(e+1))`) is a multiple
/// of `2^(e−10)`, and below `2^−14` every half is a multiple of `2^−24`. So `x` is
/// divided by that step, which is exact in a double, rounded to a whole number of
/// steps with ties to even, and the whole number is the fraction (with the
/// implicit one) or, in the subnormals, the whole of the bits. A value that rounds
/// up to the top of its binade carries into the next, which this handles by
/// reading the result as `2048` steps and calling it `1024` of the binade above.
///
/// # Panics
/// Panics on a value a file may not hold: not finite, or `65520` and over, which
/// rounds to infinity. The exporter refuses those before the file exists, and a
/// test that made one would be testing a file the exporter cannot write.
pub(super) fn to_half(x: f32) -> u16 {
    assert!(x.is_finite(), "{x} is not a number a file holds");
    let sign = if x.is_sign_negative() { 0x8000u16 } else { 0 };
    let a = f64::from(x.abs());
    assert!(a < 65520.0, "{x} rounds to infinity as a half");
    if a == 0.0 {
        return sign;
    }
    // The binade of `a`, from the double's own exponent field: exact.
    let binade = ((a.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let binade = binade.max(-14);
    let step = f64::from_bits(((binade - 10 + 1023) as u64) << 52);
    let steps = (a / step).round_ties_even() as u32;
    if binade == -14 && steps < 1024 {
        // Zero or a subnormal: the bits are the number of `2^−24`s.
        return sign | steps as u16;
    }
    let (binade, steps) = if steps == 2048 {
        (binade + 1, 1024)
    } else {
        (binade, steps)
    };
    assert!(binade <= 15, "{x} rounds past the largest half");
    sign | (((binade + 15) as u16) << 10) | (steps - 1024) as u16
}

fn put_f32s(out: &mut Vec<u8>, values: &[f32]) {
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// An array of halves: two bytes each, little-endian, and nothing in front.
fn put_halves(out: &mut Vec<u8>, values: &[f32]) {
    for &v in values {
        out.extend_from_slice(&to_half(v).to_le_bytes());
    }
}

/// `net` written the way `export-nnue` writes it.
pub(super) fn encode(net: &SymNetwork) -> Vec<u8> {
    let arch = expected_arch();
    let hash = arch_hash(&arch);
    let mut out = Vec::new();
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&hash.to_le_bytes());
    out.extend_from_slice(&(arch.len() as u32).to_le_bytes());
    out.extend_from_slice(arch.as_bytes());

    out.extend_from_slice(&hash.wrapping_add(1).to_le_bytes());
    put_halves(&mut out, &net.embed);

    out.extend_from_slice(&hash.wrapping_add(2).to_le_bytes());
    put_f32s(&mut out, &net.gamma);
    put_halves(&mut out, &net.w1);
    put_f32s(&mut out, &net.b1);
    put_halves(&mut out, &net.w2);
    put_f32s(&mut out, &net.b2);
    put_f32s(&mut out, &net.value_w);
    put_f32s(&mut out, &[net.value_b]);

    if HAS_PAIR {
        out.extend_from_slice(&hash.wrapping_add(3).to_le_bytes());
        put_f32s(&mut out, &net.pair_gamma);
        put_halves(&mut out, &net.pair);
    }
    if HAS_MOMENT {
        out.extend_from_slice(&hash.wrapping_add(4).to_le_bytes());
        put_f32s(&mut out, &net.moment_gamma);
        put_halves(&mut out, &net.moment_embed);
    }
    if HAS_FIELD {
        out.extend_from_slice(&hash.wrapping_add(5).to_le_bytes());
        put_f32s(&mut out, &net.field_gamma);
        // `[token][square][channel]`, flattened as it is held.
        put_halves(&mut out, &net.field);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::to_half;

    /// The writer against halves worked out by hand, so that the test's own
    /// rounding is what the description says before anything is read with it: ties
    /// go to the even fraction, a carry moves into the next binade, the subnormals
    /// are multiples of `2^−24` and anything at or under `2^−25` is zero.
    #[test]
    fn the_writer_rounds_to_nearest_even() {
        let ulp = 2f32.powi(-10);
        for (x, bits) in [
            (0.0f32, 0x0000u16),
            (-0.0, 0x8000),
            (1.0, 0x3c00),
            (-2.0, 0xc000),
            // Halfway between 1 and 1 + 2^−10: the even fraction, 0, is 1.
            (1.0 + ulp / 2.0, 0x3c00),
            // Halfway between 1 + 2^−10 and 1 + 2^−9: the even fraction, 2.
            (1.0 + 1.5 * ulp, 0x3c02),
            // Just over the first of those is the next half up.
            (1.0 + ulp / 2.0 + 2f32.powi(-20), 0x3c01),
            // The top of a binade carries into the next one: 2 − 2^−11 is halfway
            // between 2 − 2^−10 and 2, and the even one is 2.
            (2.0 - 2f32.powi(-11), 0x4000),
            (65504.0, 0x7bff),
            (65519.0, 0x7bff),
            (2f32.powi(-14), 0x0400),
            (2f32.powi(-24), 0x0001),
            (1023.0 * 2f32.powi(-24), 0x03ff),
            // Under the smallest subnormal: halfway is a tie to zero, over it is not.
            (2f32.powi(-25), 0x0000),
            (1.5 * 2f32.powi(-25), 0x0001),
            // Halfway between 1 and 2 `2^−24`s is 1.5: the even number of steps is 2.
            (1.5 * 2f32.powi(-24), 0x0002),
            // The largest subnormal and the smallest normal are neighbours, and
            // halfway between them rounds to the even one, the normal.
            (1023.5 * 2f32.powi(-24), 0x0400),
        ] {
            assert_eq!(to_half(x), bits, "{x:e}");
        }
    }

    /// A value a file may not hold is refused by the writer, as the exporter
    /// refuses it before the file exists.
    #[test]
    #[should_panic(expected = "rounds to infinity")]
    fn the_writer_refuses_what_rounds_to_infinity() {
        let _ = to_half(65520.0);
    }
}
