//! The cursor the reader uses: little-endian words, `f32` arrays, and the `f16`
//! arrays that hold the tables and the weights. It knows how a number is stored
//! and nothing about which array comes next; that is `body.rs`.

use crate::types::NnueError;

pub(super) struct Reader<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) pos: usize,
}

/// An IEEE 754 binary16, widened to `f32`. Exact: every half is an `f32`.
///
/// A half is one sign bit, five exponent bits `e` (bias 15) and ten fraction bits
/// `f`, and its value is
///
/// ```text
/// 0 < e < 31   (−1)^sign · (1 + f/1024) · 2^(e−15)
/// e = 0        (−1)^sign · f · 2^−24                 zero and the subnormals
/// e = 31       infinity or NaN
/// ```
///
/// The last is `None`: a file never holds one, and a reader that took one in
/// would hand every product it met an infinity or a NaN.
///
/// Written out of that table and not by shifting the fields into an `f32`'s, so
/// that the two are two routes to the same number: the subnormal case is a
/// multiple of `2^−24`, which an `f32` holds exactly.
fn half_to_f32(bits: u16) -> Option<f32> {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = f32::from(bits & 0x3ff);
    let magnitude = match exponent {
        // `f · 2^−24`: `f` is below 2^10, so this is exact in an `f32`, and so is the
        // power of two (`2^−24` is the `f32` with exponent field 103).
        0 => fraction * f32::from_bits(103 << 23),
        31 => return None,
        // `(1024 + f) · 2^(e−25)`: `1024 + f` is below 2^11 and exact, and the power of
        // two is built from its bits (`e − 25 + 127` in the exponent field) so that
        // no `powi` rounds it. It is the same number as `(1 + f/1024) · 2^(e−15)`.
        e => (1024.0 + fraction) * f32::from_bits(((e + 102) as u32) << 23),
    };
    Some(sign * magnitude)
}

impl<'a> Reader<'a> {
    pub(super) fn take(&mut self, n: usize) -> Result<&'a [u8], NnueError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| NnueError::InvalidFormat {
                reason: "read offset overflows usize".to_string(),
            })?;
        if end > self.bytes.len() {
            return Err(NnueError::SizeMismatch {
                expected: end,
                got: self.bytes.len(),
            });
        }
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    pub(super) fn u32_le(&mut self) -> Result<u32, NnueError> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    pub(super) fn f32s(&mut self, n: usize) -> Result<Vec<f32>, NnueError> {
        let s = self.take(n * 4)?;
        let (words, _) = s.as_chunks::<4>();
        Ok(words.iter().map(|c| f32::from_le_bytes(*c)).collect())
    }

    /// `n` halves, widened. Two bytes an element and nothing in front: no scale and
    /// no padding. An infinity or a NaN is a refusal, naming the array and the
    /// element, because a file that holds one is not a file the exporter wrote.
    pub(super) fn halves(&mut self, n: usize, what: &str) -> Result<Vec<f32>, NnueError> {
        let bytes = n.checked_mul(2).ok_or_else(|| NnueError::InvalidFormat {
            reason: format!("{what} is too long to address"),
        })?;
        let s = self.take(bytes)?;
        let (pairs, _) = s.as_chunks::<2>();
        let mut out = Vec::with_capacity(n);
        for (i, c) in pairs.iter().enumerate() {
            let bits = u16::from_le_bytes(*c);
            out.push(half_to_f32(bits).ok_or_else(|| NnueError::InvalidFormat {
                reason: format!("{what}[{i}] is {bits:#06x}, an infinity or a NaN"),
            })?);
        }
        Ok(out)
    }

    pub(super) fn expect_hash(&mut self, want: u32, section: &str) -> Result<(), NnueError> {
        let got = self.u32_le()?;
        if got == want {
            return Ok(());
        }
        Err(NnueError::InvalidFormat {
            reason: format!("{section} hash is {got:#010x}, this build expects {want:#010x}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::half_to_f32;

    /// Every one of the 65,536 bit patterns against the formula written as a
    /// double: finite halves are exact, and the 2,048 infinities and NaNs are
    /// refused.
    #[test]
    fn every_half_is_what_the_formula_says() {
        let mut refused = 0;
        for bits in 0..=u16::MAX {
            let sign = if bits >> 15 == 1 { -1.0f64 } else { 1.0 };
            let (e, f) = (i32::from((bits >> 10) & 0x1f), f64::from(bits & 0x3ff));
            let want = match e {
                0 => Some(sign * f * 2f64.powi(-24)),
                31 => None,
                e => Some(sign * (1.0 + f / 1024.0) * 2f64.powi(e - 15)),
            };
            match (half_to_f32(bits), want) {
                (Some(got), Some(want)) => {
                    assert_eq!(f64::from(got), want, "{bits:#06x}");
                    // The sign of a zero is kept: `-0.0` is a number a file can hold.
                    assert_eq!(got.is_sign_negative(), bits >> 15 == 1, "{bits:#06x}");
                }
                (None, None) => refused += 1,
                (got, want) => panic!("{bits:#06x}: {got:?} against {want:?}"),
            }
        }
        assert_eq!(refused, 2 * 1024, "e = 31 is 1,024 patterns of each sign");
    }

    /// The numbers the format's description names: the smallest subnormal, the
    /// largest subnormal, the smallest normal, one, and the largest half.
    #[test]
    fn the_named_halves() {
        for (bits, want) in [
            (0x0000, 0.0f32),
            (0x0001, 2f32.powi(-24)),
            (0x03ff, 1023.0 * 2f32.powi(-24)),
            (0x0400, 2f32.powi(-14)),
            (0x3c00, 1.0),
            (0xbc00, -1.0),
            (0x7bff, 65504.0),
            (0xfbff, -65504.0),
        ] {
            assert_eq!(half_to_f32(bits), Some(want), "{bits:#06x}");
        }
    }
}
