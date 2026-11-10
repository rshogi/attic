//! The cursor the readers share: little-endian words, `f32` arrays, and the
//! `SYNNscaled` arrays with a scale per group. It knows how a number is stored
//! and nothing about which array comes next; that is each encoding's reader.

use crate::types::NnueError;

pub(super) struct Reader<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) pos: usize,
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

    /// `n` scales: the head of a `SYNNscaled` array. A scale is read, never
    /// computed, and a file whose scale is not a positive number would turn
    /// every element of its group into an infinity or a NaN, so that is a
    /// refusal here rather than a value that looks like a network.
    pub(super) fn group_scales(&mut self, n: usize, what: &str) -> Result<Vec<f32>, NnueError> {
        let scales = self.f32s(n)?;
        match scales.iter().find(|s| !(s.is_finite() && **s > 0.0)) {
            Some(bad) => Err(NnueError::InvalidFormat {
                reason: format!("{what} has a scale of {bad}"),
            }),
            None => Ok(scales),
        }
    }

    /// A `SYNNscaled` array of `rows × cols` whose group is the **column**: one
    /// scale per column, then the `i16` elements row-major, element `i` read as
    /// `q[i] / scale[i mod cols]`. The embedding is stored this way, a scale per
    /// channel. An empty array stores nothing, not even a scale.
    pub(super) fn scaled_per_column(
        &mut self,
        rows: usize,
        cols: usize,
        what: &str,
    ) -> Result<Vec<f32>, NnueError> {
        if rows * cols == 0 {
            return Ok(Vec::new());
        }
        let scales = self.group_scales(cols, what)?;
        let s = self.take(rows * cols * 2)?;
        let (pairs, _) = s.as_chunks::<2>();
        let mut out = Vec::with_capacity(rows * cols);
        for row in pairs.chunks_exact(cols) {
            out.extend(
                row.iter()
                    .zip(&scales)
                    .map(|(c, scale)| f32::from(i16::from_le_bytes(*c)) / scale),
            );
        }
        Ok(out)
    }

    /// A `SYNNscaled` array of `rows × cols` whose group is the **row**: one scale
    /// per row, element `i` read as `q[i] / scale[i div cols]`. The dense
    /// weights are stored this way, a scale per output unit.
    pub(super) fn scaled_per_row(
        &mut self,
        rows: usize,
        cols: usize,
        what: &str,
    ) -> Result<Vec<f32>, NnueError> {
        if rows * cols == 0 {
            return Ok(Vec::new());
        }
        let scales = self.group_scales(rows, what)?;
        let s = self.take(rows * cols * 2)?;
        let (pairs, _) = s.as_chunks::<2>();
        let mut out = Vec::with_capacity(rows * cols);
        for (row, scale) in pairs.chunks_exact(cols).zip(&scales) {
            out.extend(
                row.iter()
                    .map(|c| f32::from(i16::from_le_bytes(*c)) / scale),
            );
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
