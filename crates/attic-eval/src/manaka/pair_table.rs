//! The pair table as the derive reads it: halves, both frames to a row.
//!
//! The file holds one table, `[PAIR_ROWS][PAIR]` halves, and both frames read it:
//! the row of a pair is `pair_row` of the two tokens as that frame numbers them.
//! A move touches the same pieces in both frames, so a derive reads, for every
//! pair it touches, one row for the black frame and another for the white one,
//! far apart in the table. This keeps the two together: row `r` here is the
//! file's row `r` (the pair as the black frame numbers it) followed by the file's
//! row for the same two pieces as the white frame numbers them. One read of
//! `2 · PAIR` halves serves both frames.
//!
//! The values stay halves, as the file has them, and are widened when they are
//! added (`pair.rs`). That is exact, so what the sums see is what an `f32` table
//! widened at load would give, and the table is half the size: `2 · PAIR` halves a
//! row is the `PAIR` `f32`s a row of the single-frame table took.

use crate::Aligned64;

use super::PAIR;
#[cfg(feature = "sym_token")]
use super::PAIR_ROWS;
use super::reader::half_to_f32;

/// `[PAIR_ROWS][2][PAIR]` halves: for each pair as the black frame numbers it,
/// that pair's row of the file, then the row of the same two pieces as the white
/// frame numbers them. Empty in a shape without a pair table.
#[derive(Debug)]
pub struct PairTable {
    halves: Aligned64<u16>,
}

impl PairTable {
    /// The table of a shape without one.
    pub(super) fn empty() -> PairTable {
        PairTable {
            halves: Aligned64::zeroed(0),
        }
    }

    /// The table from the file's, `PAIR_ROWS × PAIR` little-endian halves that the
    /// reader has already found finite.
    ///
    /// Each pair `(lo, hi)` of the black frame, in the file's row order, takes
    /// its own row and then the row of `(other_frame(lo), other_frame(hi))`.
    #[cfg(feature = "sym_token")]
    pub(super) fn from_le_bytes(file: &[u8]) -> PairTable {
        use super::TOKENS;
        use super::pair::pair_row;
        use crate::tokens::{FeatureIndex, other_frame};

        assert_eq!(file.len(), PAIR_ROWS * PAIR * 2, "the file's pair table");
        let read = |row: usize, out: &mut [u16]| {
            let bytes = &file[row * PAIR * 2..(row + 1) * PAIR * 2];
            for (slot, two) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
                *slot = u16::from_le_bytes(*two);
            }
        };
        let mut halves = Aligned64::zeroed(PAIR_ROWS * 2 * PAIR);
        let mut row = 0;
        for lo in 0..TOKENS as FeatureIndex {
            for hi in lo + 1..TOKENS as FeatureIndex {
                let (black, white) =
                    halves[row * 2 * PAIR..(row + 1) * 2 * PAIR].split_at_mut(PAIR);
                read(row, black);
                read(pair_row(other_frame(lo), other_frame(hi)), white);
                row += 1;
            }
        }
        PairTable { halves }
    }

    /// Rows, one per pair: `PAIR_ROWS`, or none without a pair table.
    pub fn rows(&self) -> usize {
        if PAIR == 0 {
            0
        } else {
            self.halves.len() / (2 * PAIR)
        }
    }

    /// Every half, `[rows][2][PAIR]`.
    #[cfg(any(test, feature = "sym_token"))]
    pub(super) fn halves(&self) -> &[u16] {
        &self.halves
    }

    /// The file's entry at `row` and `channel`, widened: the table both frames
    /// read, as the file has it.
    ///
    /// # Panics
    /// Panics if `channel` is not below `PAIR` or `row` not below [`rows`](Self::rows).
    pub fn value(&self, row: usize, channel: usize) -> f32 {
        let channels = PAIR;
        assert!(channel < channels, "channel {channel} of {channels}");
        half_to_f32(self.halves[row * 2 * PAIR + channel])
            .expect("the reader found every half finite")
    }

    /// The file's table, `[PAIR_ROWS][PAIR]`, widened: for the tests that write a
    /// network out and read it back.
    #[cfg(test)]
    pub(super) fn file_table(&self) -> Vec<f32> {
        (0..self.rows())
            .flat_map(|row| (0..PAIR).map(move |channel| (row, channel)))
            .map(|(row, channel)| self.value(row, channel))
            .collect()
    }

    /// A second copy, for another NUMA node.
    pub(super) fn replicate(&self) -> PairTable {
        PairTable {
            halves: Aligned64::from_slice(&self.halves),
        }
    }
}
