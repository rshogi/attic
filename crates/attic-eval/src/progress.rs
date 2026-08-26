//! Progress-bucket selection for the `progress8ek` layer-stack mode.
//!
//! Stock SFNN-1536 picks its layer stack from the two kings' ranks
//! (`king3_by_king3`, nine buckets). The shape NAGISA V4 ships declares
//! `NNUE_SFNN_HAND_BUCKETS 1`, `NNUE_SFNN_KING_BUCKETS 1` and
//! `NNUE_SFNN_PROGRESS_BUCKETS 9`, which collapses `stack_index_for_nnue` to
//! the progress bucket alone — the kings' ranks never enter it.
//!
//! Buckets 0..=7 are a game-progress estimate; bucket 8 is reserved for mutual
//! entering-king positions, which have a different material and progress
//! profile from everything else and were trained as their own stack.
//!
//! The coefficients live outside `nn.bin` (`NNUE_SFNN_PROGRESS_EXTERNAL 1`), in
//! a `progress.bin` alongside it: a raw `f64[81][1548]` with no header and no
//! bias, rounded to Q16 on load. That is also why `nn.bin`'s hash does not
//! cover them.
//!
//! Ported from `eval/nnue/evaluate_nnue.cpp` in the shielune tree
//! (`Progress::Parameters::{Value0To255, BucketIndex, BucketIndexWithEnteringKing}`
//! and `IsMutualEnteringKing`).

use std::path::Path;
use std::sync::OnceLock;

use attic_state::{Color, PieceKind, Position, Square};

use crate::features::{FE_END, HAND_KINDS, board_plane, from_persp, hand_plane};

/// Squares on the board — the first axis of the coefficient table.
const SQ_NB: usize = Square::COUNT;

/// Piece slots the progress sum walks. The reference iterates
/// `i < PIECE_NUMBER_KING`, i.e. all forty slots minus the two kings.
const PROGRESS_PIECE_SLOTS: usize = 38;

/// `BONA_PIECE_ZERO`, the code an empty piece slot carries. Absent pieces still
/// contribute `w[sq_k][0]` to the sum, so the padding is not a no-op.
const BONA_PIECE_ZERO: usize = 0;

/// How many distinct progress values `Value0To255` can return.
const PROGRESS_VALUE_COUNT: i64 = 256;

/// Number of progress buckets, the last of which is the entering-king stack.
pub const PROGRESS_BUCKETS: usize = 9;

/// Q16 fixed-point coefficients, `[square][bona_piece]`.
///
/// Boxed because the table is `81 * 1548 * 4` bytes and would otherwise be a
/// half-megabyte value moved by copy.
pub struct ProgressParameters {
    weights_q16: Box<[i32]>,
}

impl ProgressParameters {
    /// Number of coefficients in a `progress.bin`.
    pub const WEIGHT_COUNT: usize = SQ_NB * FE_END;

    /// Reads a `progress.bin`: `f64[81][1548]`, little-endian, no header.
    ///
    /// Each coefficient is `round(v * 65536)` clamped into `i32`, matching
    /// `Parameters::ReadExternalCoefficients`. A file whose length is not
    /// exactly `WEIGHT_COUNT * 8` is rejected rather than padded — a truncated
    /// table would silently shift every bucket.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let want = Self::WEIGHT_COUNT * std::mem::size_of::<f64>();
        if bytes.len() != want {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "{}: expected {want} bytes of f64[{SQ_NB}][{FE_END}], found {}",
                    path.display(),
                    bytes.len()
                ),
            ));
        }

        let mut weights = Vec::with_capacity(Self::WEIGHT_COUNT);
        for chunk in bytes.chunks_exact(8) {
            let raw = f64::from_le_bytes(chunk.try_into().expect("chunks_exact(8) yields 8 bytes"));
            let scaled = (raw * 65536.0).round();
            weights.push(scaled.clamp(i32::MIN as f64, i32::MAX as f64) as i32);
        }

        Ok(ProgressParameters {
            weights_q16: weights.into_boxed_slice(),
        })
    }

    #[inline]
    fn weight(&self, sq: usize, bona: usize) -> i64 {
        debug_assert!(sq < SQ_NB && bona < FE_END);
        self.weights_q16[sq * FE_END + bona] as i64
    }

    /// The progress estimate, `0..=255`.
    ///
    /// `sum_q16` is the bias (always zero for an external table) plus one
    /// coefficient per non-king piece slot, taken from Black's king square in
    /// Black's frame and White's king square in White's frame.
    pub fn value_0_to_255(&self, pos: &Position) -> usize {
        let sq_bk = king_square(pos, Color::Black).index() as usize;
        // `Inv(sq_wk)` in the reference: White's king viewed from White.
        let sq_wk = from_persp(king_square(pos, Color::White), Color::White).index() as usize;

        let mut black = [BONA_PIECE_ZERO; PROGRESS_PIECE_SLOTS];
        let mut white = [BONA_PIECE_ZERO; PROGRESS_PIECE_SLOTS];
        bona_pieces(pos, Color::Black, &mut black);
        bona_pieces(pos, Color::White, &mut white);

        let mut sum_q16: i64 = 0;
        for i in 0..PROGRESS_PIECE_SLOTS {
            sum_q16 += self.weight(sq_bk, black[i]);
            sum_q16 += self.weight(sq_wk, white[i]);
        }

        // How many thresholds the sum clears; the table is sorted ascending.
        thresholds_q16().partition_point(|&t| t <= sum_q16)
    }

    /// The layer-stack index under `progress8ek`.
    ///
    /// Mutual entering king takes the last stack outright; everything else is
    /// split across the remaining eight by progress.
    pub fn bucket_index(&self, pos: &Position) -> usize {
        if is_mutual_entering_king(pos) {
            return PROGRESS_BUCKETS - 1;
        }
        let usable = PROGRESS_BUCKETS - 1;
        let progress = self.value_0_to_255(pos) as i64;
        let bucket = progress * usable as i64 / PROGRESS_VALUE_COUNT;
        (bucket.clamp(0, usable as i64 - 1)) as usize
    }
}

/// Both kings past the middle rank, in opposite directions.
///
/// Ranks are zero-based here, so the reference's `RANK_5` is rank index 4.
fn is_mutual_entering_king(pos: &Position) -> bool {
    let black = king_square(pos, Color::Black).rank() as usize;
    let white = king_square(pos, Color::White).rank() as usize;
    black <= 4 && white >= 4
}

/// The `piece_list_fb` / `piece_list_fw` equivalent: one BonaPiece code per
/// non-king slot, in `perspective`'s frame.
///
/// This is deliberately *not* [`crate::features::active_features`]. That one
/// mirrors the whole position into files 1–5 and folds both kings into a shared
/// plane, because its output indexes a `45 * 1629` feature space. The progress
/// table is `81 * 1548` and never mirrors, so only the plane arithmetic is
/// shared.
fn bona_pieces(pos: &Position, perspective: Color, out: &mut [usize; PROGRESS_PIECE_SLOTS]) {
    let mut n = 0;

    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).expect("index < Square::COUNT is valid");
        let Some(piece) = pos.board().get(sq) else {
            continue;
        };
        if piece.kind == PieceKind::King {
            continue;
        }
        let is_friend = piece.color == perspective;
        let sq_code = from_persp(sq, perspective).index() as usize;
        out[n] = board_plane(piece.kind, piece.promoted, is_friend) + sq_code;
        n += 1;
    }

    for hand_color in [Color::Black, Color::White] {
        let is_friend = hand_color == perspective;
        let hand = pos.hand(hand_color);
        for kind in HAND_KINDS {
            let base = hand_plane(kind, is_friend);
            for i in 1..=hand.count(kind) as usize {
                out[n] = base + i;
                n += 1;
            }
        }
    }

    // Remaining slots keep BONA_PIECE_ZERO from the caller's initialiser.
    debug_assert!(n <= PROGRESS_PIECE_SLOTS);
}

fn king_square(pos: &Position, color: Color) -> Square {
    pos.king_square(color)
        .expect("every position fed to the evaluator has both kings")
}

/// `log(p / (1 - p)) * 65536` for `p = i/256`, `i` in `1..256`, ascending.
///
/// Built once: 255 `f64` logs per call would otherwise sit in the search's hot
/// path.
fn thresholds_q16() -> &'static [i64; 255] {
    static THRESHOLDS: OnceLock<[i64; 255]> = OnceLock::new();
    THRESHOLDS.get_or_init(|| {
        let mut table = [0i64; 255];
        for (slot, i) in table.iter_mut().zip(1..PROGRESS_VALUE_COUNT) {
            let p = i as f64 / PROGRESS_VALUE_COUNT as f64;
            let scaled = (p / (1.0 - p)).ln() * 65536.0;
            *slot = scaled.round().clamp(i64::MIN as f64, i64::MAX as f64) as i64;
        }
        table
    })
}

/// The process-wide coefficient table, installed once at startup.
static PARAMETERS: OnceLock<ProgressParameters> = OnceLock::new();

/// Installs the table read from `path`. Later calls are ignored, matching the
/// reference's load-once behaviour.
pub fn load_parameters(path: &Path) -> std::io::Result<()> {
    let params = ProgressParameters::load(path)?;
    let _ = PARAMETERS.set(params);
    Ok(())
}

/// The installed table, if [`load_parameters`] has run.
pub fn parameters() -> Option<&'static ProgressParameters> {
    PARAMETERS.get()
}

/// The layer-stack index for `pos`.
///
/// Falls back to bucket 0 when no table is installed: a caller that forgot to
/// load `progress.bin` gets one consistent stack rather than a panic mid-search,
/// and [`parameters`] tells it apart from a real bucket 0.
pub fn bucket_index(pos: &Position) -> usize {
    parameters().map_or(0, |p| p.bucket_index(pos))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_are_ascending_and_centred() {
        let t = thresholds_q16();
        for w in t.windows(2) {
            assert!(w[0] < w[1], "thresholds must be strictly ascending");
        }
        // p = 128/256 = 0.5 -> log(1) = 0, the 128th threshold (index 127).
        assert_eq!(t[127], 0);
    }

    #[test]
    fn bucket_is_zero_without_a_table() {
        let pos = Position::startpos();
        assert_eq!(bucket_index(&pos), 0);
    }

    /// Prints what a real `progress.bin` makes of a few positions.
    ///
    /// Skipped unless `ATTIC_PROGRESS_BIN` names a file. Diagnostic only — it
    /// asserts the shape of the answer, not the answer itself, because the
    /// reference has no command to print its own bucket.
    #[test]
    fn reports_buckets_for_real_coefficients() {
        let Ok(path) = std::env::var("ATTIC_PROGRESS_BIN") else {
            return;
        };
        let params = ProgressParameters::load(Path::new(&path)).expect("progress.bin should load");

        let pos = Position::startpos();
        let value = params.value_0_to_255(&pos);
        let bucket = params.bucket_index(&pos);
        eprintln!(
            "startpos: progress={value} bucket={bucket} entering_king={}",
            is_mutual_entering_king(&pos)
        );
        assert!(value <= 255);
        assert!(bucket < PROGRESS_BUCKETS);
    }
}
