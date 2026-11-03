//! Progress-bucket selection for the `progress8ek` and `progress8kpabs`
//! layer-stack modes.
//!
//! Stock SFNN-1536 picks its layer stack from the two kings' ranks
//! (`king3_by_king3`, nine buckets). The shape NAGISA V4 ships declares
//! `NNUE_SFNN_HAND_BUCKETS 1`, `NNUE_SFNN_KING_BUCKETS 1` and
//! `NNUE_SFNN_PROGRESS_BUCKETS 9`, which collapses `stack_index_for_nnue` to
//! the progress bucket alone — the kings' ranks never enter it.
//!
//! Buckets 0..=7 are a game-progress estimate in both modes. They part ways on
//! mutual entering-king positions, which have a different material and progress
//! profile from everything else:
//!
//! - `progress8ek` (NAGISA V4, 2304) trained them as their own stack, bucket 8.
//! - `progress8kpabs` (NAGISA V3, 1024) did not. V3 was trained with eight
//!   buckets, so it never looks for entering kings; those positions get the
//!   progress bucket like any other, and the ninth stack is in the file but
//!   nothing selects it. Sending them to bucket 8 anyway would evaluate them
//!   with weights that were never trained.
//!
//! The two rules are separate pure functions below, [`bucket_progress8ek`] and
//! [`bucket_progress8kpabs`], so that both are compiled and tested in every
//! build. [`ProgressParameters::bucket_index`] is the only place that chooses
//! between them, by the `sfnn_progress_ek` feature, rather than being split into
//! one method per mode.
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

/// Layer stacks in a progress network's file. The last is the entering-king
/// stack under `progress8ek`; under `progress8kpabs` it is present and unused.
pub const PROGRESS_BUCKETS: usize = 9;

/// The buckets game progress alone is cut into, in both modes: every stack but
/// the last.
const PROGRESS_ONLY_BUCKETS: usize = PROGRESS_BUCKETS - 1;

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
        let (chunks, _) = bytes.as_chunks::<8>();
        for chunk in chunks {
            let raw = f64::from_le_bytes(*chunk);
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

        let sum_q16 = self.sum_q16(pos, sq_bk, sq_wk);

        // How many thresholds the sum clears; the table is sorted ascending.
        thresholds_q16().partition_point(|&t| t <= sum_q16)
    }

    /// The coefficient sum [`Self::value_0_to_255`] compares against its
    /// thresholds.
    ///
    /// The same sum [`bona_pieces`] describes, taken without the two 38-slot
    /// arrays: addition does not care what order the slots come in, so each
    /// piece adds both frames' coefficients as it is found, walking the
    /// occupied squares rather than all eighty-one, and the slots the position
    /// leaves empty are added at the end as the `BONA_PIECE_ZERO` coefficient
    /// they would have carried. A position with more than 38 non-king pieces and
    /// held pieces cannot exist -- the pieces are forty, two of them kings --
    /// so that count never goes negative.
    #[inline]
    fn sum_q16(&self, pos: &Position, sq_bk: usize, sq_wk: usize) -> i64 {
        let board = pos.board();
        let mut sum_q16: i64 = 0;
        let mut slots = 0usize;

        for sq in board.occupied() {
            let Some(piece) = board.get(sq) else {
                continue;
            };
            if piece.kind == PieceKind::King {
                continue;
            }
            let black_code = board_plane(piece.kind, piece.promoted, piece.color == Color::Black)
                + from_persp(sq, Color::Black).index() as usize;
            let white_code = board_plane(piece.kind, piece.promoted, piece.color == Color::White)
                + from_persp(sq, Color::White).index() as usize;
            sum_q16 += self.weight(sq_bk, black_code) + self.weight(sq_wk, white_code);
            slots += 1;
        }

        for hand_color in [Color::Black, Color::White] {
            let hand = pos.hand(hand_color);
            for kind in HAND_KINDS {
                let held = hand.count(kind) as usize;
                let black_base = hand_plane(kind, hand_color == Color::Black);
                let white_base = hand_plane(kind, hand_color == Color::White);
                for i in 1..=held {
                    sum_q16 +=
                        self.weight(sq_bk, black_base + i) + self.weight(sq_wk, white_base + i);
                }
                slots += held;
            }
        }

        debug_assert!(slots <= PROGRESS_PIECE_SLOTS);
        let empty = (PROGRESS_PIECE_SLOTS - slots) as i64;
        sum_q16
            + empty * (self.weight(sq_bk, BONA_PIECE_ZERO) + self.weight(sq_wk, BONA_PIECE_ZERO))
    }

    /// The layer-stack index for `pos`.
    ///
    /// Under `sfnn_progress_ek` ([`bucket_progress8ek`]) mutual entering king
    /// takes the last stack outright and everything else is split across the
    /// remaining eight by progress. Otherwise ([`bucket_progress8kpabs`]) there
    /// is no entering-king case at all and progress alone decides.
    ///
    /// `cfg!` rather than `#[cfg]`: both arms stay compiled, and so type-checked
    /// and tested, in every build, and the dead one costs nothing.
    pub fn bucket_index(&self, pos: &Position) -> usize {
        if cfg!(feature = "sfnn_progress_ek") {
            // The entering-king test comes first and the progress sum is only
            // taken when it fails: the sum is the expensive half.
            bucket_progress8ek(is_mutual_entering_king(pos), || self.value_0_to_255(pos))
        } else {
            bucket_progress8kpabs(self.value_0_to_255(pos))
        }
    }
}

/// `Parameters::BucketIndex(pos, bucket_count)` once the progress is known:
/// `progress * bucket_count / 256`, clamped into `0..bucket_count`.
///
/// Both modes cut progress into [`PROGRESS_ONLY_BUCKETS`] buckets with this.
fn bucket_by_progress(progress: usize, bucket_count: usize) -> usize {
    let bucket = progress as i64 * bucket_count as i64 / PROGRESS_VALUE_COUNT;
    bucket.clamp(0, bucket_count as i64 - 1) as usize
}

/// The layer-stack index under `progress8ek`.
///
/// Mutual entering king takes the last stack outright; everything else is
/// split across the remaining eight by progress. `progress` is a closure so
/// that an entering-king position, which never needs it, does not pay for the
/// sum.
pub fn bucket_progress8ek(mutual_entering_king: bool, progress: impl FnOnce() -> usize) -> usize {
    if mutual_entering_king {
        return PROGRESS_BUCKETS - 1;
    }
    bucket_by_progress(progress(), PROGRESS_ONLY_BUCKETS)
}

/// The layer-stack index under `progress8kpabs`.
///
/// Progress alone, in `0..=7`. Whether the kings have entered each other's camp
/// is not asked, and the ninth stack is never returned.
pub fn bucket_progress8kpabs(progress: usize) -> usize {
    bucket_by_progress(progress, PROGRESS_ONLY_BUCKETS)
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
#[cfg(test)]
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
    use attic_state::parse_sfen;

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

    /// The sum as the slot arrays define it: one `BonaPiece` code per non-king
    /// piece and held piece in each frame, `BONA_PIECE_ZERO` in the rest.
    fn reference_sum_q16(params: &ProgressParameters, pos: &Position) -> i64 {
        let sq_bk = king_square(pos, Color::Black).index() as usize;
        let sq_wk = from_persp(king_square(pos, Color::White), Color::White).index() as usize;
        let mut black = [BONA_PIECE_ZERO; PROGRESS_PIECE_SLOTS];
        let mut white = [BONA_PIECE_ZERO; PROGRESS_PIECE_SLOTS];
        bona_pieces(pos, Color::Black, &mut black);
        bona_pieces(pos, Color::White, &mut white);
        let mut sum_q16: i64 = 0;
        for i in 0..PROGRESS_PIECE_SLOTS {
            sum_q16 += params.weight(sq_bk, black[i]);
            sum_q16 += params.weight(sq_wk, white[i]);
        }
        sum_q16
    }

    #[test]
    fn the_single_pass_sum_matches_the_slot_arrays() {
        // Coefficients that differ everywhere, including column 0 (the padding
        // code), so a miscounted empty slot or a swapped frame cannot cancel.
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let weights_q16: Vec<i32> = (0..ProgressParameters::WEIGHT_COUNT)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as i32) >> 4
            })
            .collect();
        let params = ProgressParameters {
            weights_q16: weights_q16.into_boxed_slice(),
        };

        for sfen in [
            // Start position, then both sides to move.
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 2",
            // Pieces in hand on both sides.
            "l1sgk2nl/6sb1/n1p1pp2p/p1bp2p2/7p1/Pr1P1PP1P/2P1P1N2/1SG2G1R1/LN2K1SGL b 2Pp 35",
            "ln1gk1snl/2s3g2/1rp1pp1pp/pp4p2/3p3P1/PP3P3/NGP1G1P1P/1BKS3R1/L1S4NL b Pbp 35",
            // Promoted pieces and a crowded hand.
            "4k4/9/9/9/9/9/+P+L+N+S+B+R3/9/4K4 b 2G2S2N2L9PBR2gsnl8p 1",
            "9/4k4/9/3+r+b4/9/9/9/4K4/9 w RB4G4S4N4L18P 90",
            // Almost bare: nearly all of the 38 slots are padding.
            "4k4/9/9/9/9/9/9/9/4K4 b - 1",
        ] {
            let pos = attic_state::parse_sfen(sfen).expect("test sfen parses");
            assert_eq!(
                params.sum_q16(
                    &pos,
                    king_square(&pos, Color::Black).index() as usize,
                    from_persp(king_square(&pos, Color::White), Color::White).index() as usize,
                ),
                reference_sum_q16(&params, &pos),
                "single-pass sum disagrees with the slot arrays for {sfen}",
            );
        }
    }

    /// Black's king on the fifth rank and White's on the ninth: past the middle
    /// in opposite directions, which [`is_mutual_entering_king`] counts.
    const MUTUAL_ENTERING_KING: &str = "9/9/9/9/4K4/9/9/9/4k4 b - 1";

    /// A table that gives every piece slot the same weight, so the progress sum
    /// is `76 * weight` for any position and the result does not depend on where
    /// the pieces are. The weights decide the progress: `i32::MIN` clears no
    /// threshold (progress 0), zero clears the 128 up to and including the one
    /// at log-odds 0 (progress 128), and `i32::MAX` clears all 255 (progress 255).
    fn flat_table(weight: i32) -> ProgressParameters {
        ProgressParameters {
            weights_q16: vec![weight; ProgressParameters::WEIGHT_COUNT].into_boxed_slice(),
        }
    }

    #[test]
    fn mutual_entering_king_is_told_from_startpos() {
        assert!(is_mutual_entering_king(
            &parse_sfen(MUTUAL_ENTERING_KING).unwrap()
        ));
        assert!(!is_mutual_entering_king(&Position::startpos()));
    }

    /// `progress8kpabs` is `progress * 8 / 256` and nothing else: eight buckets
    /// of 32, none of them the ninth.
    #[test]
    fn kpabs_cuts_progress_into_eight_and_never_returns_the_ninth() {
        for progress in 0..PROGRESS_VALUE_COUNT as usize {
            let bucket = bucket_progress8kpabs(progress);
            assert_eq!(bucket, progress / 32, "progress {progress}");
            assert!(bucket <= 7, "progress {progress} gave bucket {bucket}");
        }
    }

    /// The same cut under `progress8ek` for everything that is not a mutual
    /// entering king, so the two modes agree wherever the ninth stack is not in
    /// play and a 2304 position outside it reads what it always did.
    #[test]
    fn ek_agrees_with_kpabs_when_the_kings_have_not_entered() {
        for progress in 0..PROGRESS_VALUE_COUNT as usize {
            assert_eq!(
                bucket_progress8ek(false, || progress),
                bucket_progress8kpabs(progress),
                "progress {progress}"
            );
        }
    }

    /// The case the two modes exist to differ on: a mutual entering king is the
    /// ninth stack under `ek` whatever its progress, and under `kpabs` it is
    /// bucketed by progress like any other position and stays in `0..=7`.
    #[test]
    fn mutual_entering_king_is_bucket_8_under_ek_and_not_under_kpabs() {
        let pos = parse_sfen(MUTUAL_ENTERING_KING).unwrap();
        assert!(is_mutual_entering_king(&pos));
        for progress in 0..PROGRESS_VALUE_COUNT as usize {
            assert_eq!(
                bucket_progress8ek(is_mutual_entering_king(&pos), || progress),
                PROGRESS_BUCKETS - 1,
                "ek, progress {progress}"
            );
            let kpabs = bucket_progress8kpabs(progress);
            assert!(
                (0..=7).contains(&kpabs),
                "kpabs, progress {progress} gave {kpabs}"
            );
        }
    }

    /// A mutual entering king never needs its progress under `ek`, and the sum is
    /// the expensive half of a bucket lookup.
    #[test]
    fn ek_does_not_compute_progress_for_a_mutual_entering_king() {
        let bucket = bucket_progress8ek(true, || panic!("progress was computed"));
        assert_eq!(bucket, PROGRESS_BUCKETS - 1);
    }

    /// `bucket_index` end to end on a position the two modes disagree about,
    /// through a table whose progress is known. This is the one test here that
    /// follows the build: the expected bucket is the mode's, and picking the
    /// wrong arm of the `cfg!` shows up as the other number.
    #[test]
    fn bucket_index_follows_the_mode_of_the_build() {
        let pos = parse_sfen(MUTUAL_ENTERING_KING).unwrap();
        let ek = cfg!(feature = "sfnn_progress_ek");

        // Progress 128: bucket 4 by progress, 8 when the kings have entered.
        let mid = flat_table(0).bucket_index(&pos);
        assert_eq!(mid, if ek { 8 } else { 4 });

        // The extremes of progress, where `kpabs` has to stay inside 0..=7.
        let lowest = flat_table(i32::MIN).bucket_index(&pos);
        assert_eq!(lowest, if ek { 8 } else { 0 });
        let highest = flat_table(i32::MAX).bucket_index(&pos);
        assert_eq!(highest, if ek { 8 } else { 7 });

        // An ordinary position is bucketed by progress in either mode.
        let startpos = Position::startpos();
        assert_eq!(flat_table(0).bucket_index(&startpos), 4);
        assert_eq!(flat_table(i32::MIN).bucket_index(&startpos), 0);
        assert_eq!(flat_table(i32::MAX).bucket_index(&startpos), 7);
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
