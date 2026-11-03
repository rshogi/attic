//! `HalfKA_hm2` active-feature index extraction, ported from
//! `eval/nnue/features/half_ka_hm2.{h,cpp}`. An `attic-state::Square` index is
//! `file * 9 + rank`, byte-identical to the reference's `Square` numbering, so
//! the index arithmetic carries over unchanged.
//!
//! A feature index is `E_KING * sq_k_code + p_adj`, where:
//! - `sq_k_code` is the perspective's own-king square, horizontally mirrored
//!   into files 1–5 (`0..45`) — the `hm` canonicalization mirrors the whole
//!   position around the king's file;
//! - `p_adj` selects a plane and a square or count within it: hand planes
//!   `[0, 90)`, board planes `[90, 1548)` interleaving friend and enemy per
//!   piece type, and one shared king plane `[1548, 1629)` both kings collapse
//!   into.
//!
//! The accumulator itself lives in [`crate::transformer`].

use attic_state::{Color, Move, Piece, PieceKind, Position, Square};

use crate::types::NUM_FEATURES;

// The tables above (`BOARD_PLANE`, `HAND_PLANE`, `FE_END`) are BonaPiece's own
// numbering and are shared by all four shapes — the promoted minors fold into
// gold in every one of them. What differs is only how a piece code is crossed
// with the king, which is `king_code` / `king_plane` / `encode_feature` below.
//
// A compile-time check that the two agree, because the equality used to live
// in a test: under `kp256` the table is 1,710 rows and an index that still
// reached past 73,000 would be read, not rejected. `features.rs`'s own bound
// check is a `debug_assert!` and disappears from the builds that matter.
#[cfg(any(
    feature = "sfnn_halfkahm2_1536_15_32_k3k3",
    feature = "sfnn_halfkahm2_2304_15_64_progress8ek"
))]
const _: () = assert!(FEATURE_DIMENSION == NUM_FEATURES);
#[cfg(feature = "nnue_kp_256x2_32_32")]
const _: () = assert!(NUM_FEATURES == FE_END + 2 * SQ_NB);
#[cfg(feature = "nnue_halfkp")]
const _: () = assert!(NUM_FEATURES == SQ_NB * FE_END);

/// A single active input-feature index into the `HalfKA_hm2` feature space.
pub type FeatureIndex = u32;

/// The feature-space origin the reference initialises empty piece slots to
/// (`EvalList::clear()`), used to pad sparse positions up to 40 slots.
const BONA_PIECE_ZERO: usize = 0;

// Hand planes. `F_*` is the friend plane base, `E_*` the enemy one. Each span
// holds one index per possible held count, with slot 0 the unused "zero" pad,
// so a pawn's plane is 19 wide.
const F_HAND_PAWN: usize = 0;
const E_HAND_PAWN: usize = F_HAND_PAWN + 19;
const F_HAND_LANCE: usize = E_HAND_PAWN + 19;
const E_HAND_LANCE: usize = F_HAND_LANCE + 5;
const F_HAND_KNIGHT: usize = E_HAND_LANCE + 5;
const E_HAND_KNIGHT: usize = F_HAND_KNIGHT + 5;
const F_HAND_SILVER: usize = E_HAND_KNIGHT + 5;
const E_HAND_SILVER: usize = F_HAND_SILVER + 5;
const F_HAND_GOLD: usize = E_HAND_SILVER + 5;
const E_HAND_GOLD: usize = F_HAND_GOLD + 5;
const F_HAND_BISHOP: usize = E_HAND_GOLD + 5;
const E_HAND_BISHOP: usize = F_HAND_BISHOP + 3;
const F_HAND_ROOK: usize = E_HAND_BISHOP + 3;
const E_HAND_ROOK: usize = F_HAND_ROOK + 3;
const FE_HAND_END: usize = E_HAND_ROOK + 3;

// Board planes: one 81-square plane per (side, effective piece type), in the
// reference's `BonaPiece` order. Promoted minors collapse onto the gold plane;
// promoted bishop and rook land on the horse and dragon planes.
const F_PAWN: usize = FE_HAND_END;
const E_PAWN: usize = F_PAWN + 81;
const F_LANCE: usize = E_PAWN + 81;
const E_LANCE: usize = F_LANCE + 81;
const F_KNIGHT: usize = E_LANCE + 81;
const E_KNIGHT: usize = F_KNIGHT + 81;
const F_SILVER: usize = E_KNIGHT + 81;
const E_SILVER: usize = F_SILVER + 81;
const F_GOLD: usize = E_SILVER + 81;
const E_GOLD: usize = F_GOLD + 81;
const F_BISHOP: usize = E_GOLD + 81;
const E_BISHOP: usize = F_BISHOP + 81;
const F_HORSE: usize = E_BISHOP + 81;
const E_HORSE: usize = F_HORSE + 81;
const F_ROOK: usize = E_HORSE + 81;
const E_ROOK: usize = F_ROOK + 81;
const F_DRAGON: usize = E_ROOK + 81;
const E_DRAGON: usize = F_DRAGON + 81;
pub(crate) const FE_END: usize = E_DRAGON + 81;

const SQ_NB: usize = Square::COUNT;
/// Size of one king-plane block. Both kings share the `[FE_END, E_KING)` span,
/// so a king-plane is `FE_END + 81` wide.
// Manaka's shapes cross nothing on a king plane, so outside the tests they
// never read it.
#[cfg(any(
    test,
    not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512"))
))]
const E_KING: usize = FE_END + SQ_NB;

/// Number of distinct mirrored king squares (files 1–5 × 9 ranks).
#[cfg(any(
    feature = "sfnn_halfkahm2_1536_15_32_k3k3",
    feature = "sfnn_halfkahm2_2304_15_64_progress8ek"
))]
const SQ_K_COUNT: usize = 5 * 9;

/// Total `HalfKA_hm2` feature dimension, kept equal to
/// [`crate::types::NUM_FEATURES`].
#[cfg(any(
    feature = "sfnn_halfkahm2_1536_15_32_k3k3",
    feature = "sfnn_halfkahm2_2304_15_64_progress8ek"
))]
pub const FEATURE_DIMENSION: usize = SQ_K_COUNT * E_KING;

/// Maximum number of simultaneously active features (`PIECE_NUMBER_NB`). Every
/// legal shogi position has exactly 40 pieces across board and hands.
#[cfg(any(
    feature = "sfnn_halfkahm2_1536_15_32_k3k3",
    feature = "sfnn_halfkahm2_2304_15_64_progress8ek",
    feature = "nnue_kp_256x2_32_32",
    feature = "sym_kp_128x2_512",
    feature = "sym_halfkp_128x2_512"
))]
pub const MAX_ACTIVE_FEATURES: usize = 40;

/// HalfKP stops at `PIECE_NUMBER_KING`.
///
/// The kings are the two slots this feature set does not have: crossing a
/// piece on the perspective's own king says nothing when the piece *is* that
/// king, so the reference walks thirty-eight and leaves them out. Manaka
/// numbers forty and points both kings at a padding row, which is the row its
/// table has and this one does not — see [`crate::types::NUM_FEATURES`].
#[cfg(feature = "nnue_halfkp")]
pub const MAX_ACTIVE_FEATURES: usize = 38;

/// The seven piece kinds that can sit in a hand, in `BonaPiece` plane order.
pub(crate) const HAND_KINDS: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

/// Rotate a square 180° when viewing the board from `persp`'s side, so that the
/// mirror maths downstream works in an own-side-forward frame.
#[inline]
pub(crate) fn from_persp(sq: Square, persp: Color) -> Square {
    match persp {
        Color::Black => sq,
        Color::White => Square::from_index((SQ_NB as u8 - 1) - sq.index())
            .expect("180-degree rotation of a valid square is valid"),
    }
}

/// The `hm` trigger: mirror when the (perspective-relative) king sits on files
/// 6–9, i.e. file index `>= 5`. Matches the C++ `sq_k >= SQ_61` test.
#[cfg(any(
    feature = "sfnn_halfkahm2_1536_15_32_k3k3",
    feature = "sfnn_halfkahm2_2304_15_64_progress8ek"
))]
#[inline]
fn needs_mirror(king_sq_persp: Square) -> bool {
    king_sq_persp.file() >= 5
}

/// Horizontal file mirror: file `f -> 8 - f`, rank unchanged.
#[inline]
fn mirror_if_needed(sq: Square, mirror: bool) -> Square {
    if mirror {
        Square::new(Square::FILES - 1 - sq.file(), sq.rank())
            .expect("file mirror of a valid square is valid")
    } else {
        sq
    }
}

/// A `(friend, enemy)` board-plane base pair for a king slot — never indexed in
/// production (kings use the shared king plane), so a sentinel that would blow
/// past `E_KING` and trip [`encode_feature`]'s `debug_assert!` if ever used.
const NO_BOARD_PLANE: (usize, usize) = (usize::MAX, usize::MAX);

/// Friend/enemy board-plane bases indexed by `[PieceKind::index()][promoted]`.
/// The king row is a sentinel: kings have no board plane, the caller routing
/// them to the shared one instead.
const BOARD_PLANE: [[(usize, usize); 2]; PieceKind::COUNT] = {
    let mut table = [[NO_BOARD_PLANE; 2]; PieceKind::COUNT];
    table[PieceKind::Pawn.index()] = [(F_PAWN, E_PAWN), (F_GOLD, E_GOLD)];
    table[PieceKind::Lance.index()] = [(F_LANCE, E_LANCE), (F_GOLD, E_GOLD)];
    table[PieceKind::Knight.index()] = [(F_KNIGHT, E_KNIGHT), (F_GOLD, E_GOLD)];
    table[PieceKind::Silver.index()] = [(F_SILVER, E_SILVER), (F_GOLD, E_GOLD)];
    table[PieceKind::Gold.index()] = [(F_GOLD, E_GOLD), (F_GOLD, E_GOLD)];
    table[PieceKind::Bishop.index()] = [(F_BISHOP, E_BISHOP), (F_HORSE, E_HORSE)];
    table[PieceKind::Rook.index()] = [(F_ROOK, E_ROOK), (F_DRAGON, E_DRAGON)];
    table
};

/// Friend/enemy hand-plane bases indexed by `[PieceKind::index()]`. The king
/// row is a sentinel — kings are never held in hand.
const HAND_PLANE: [(usize, usize); PieceKind::COUNT] = {
    let mut table = [NO_BOARD_PLANE; PieceKind::COUNT];
    table[PieceKind::Pawn.index()] = (F_HAND_PAWN, E_HAND_PAWN);
    table[PieceKind::Lance.index()] = (F_HAND_LANCE, E_HAND_LANCE);
    table[PieceKind::Knight.index()] = (F_HAND_KNIGHT, E_HAND_KNIGHT);
    table[PieceKind::Silver.index()] = (F_HAND_SILVER, E_HAND_SILVER);
    table[PieceKind::Gold.index()] = (F_HAND_GOLD, E_HAND_GOLD);
    table[PieceKind::Bishop.index()] = (F_HAND_BISHOP, E_HAND_BISHOP);
    table[PieceKind::Rook.index()] = (F_HAND_ROOK, E_HAND_ROOK);
    table
};

/// Friend/enemy board-plane base for a board piece. Kings have no board plane,
/// so passing one is a caller bug.
#[inline]
pub(crate) fn board_plane(kind: PieceKind, promoted: bool, is_friend: bool) -> usize {
    debug_assert!(
        kind != PieceKind::King,
        "king has no board plane; caller handles it"
    );
    let (friend, enemy) = BOARD_PLANE[kind.index()][promoted as usize];
    if is_friend { friend } else { enemy }
}

/// Friend/enemy hand-plane base for a held piece kind. Kings are never held in
/// hand, so passing one is a caller bug.
#[inline]
pub(crate) fn hand_plane(kind: PieceKind, is_friend: bool) -> usize {
    debug_assert!(kind != PieceKind::King, "king is never held in hand");
    let (friend, enemy) = HAND_PLANE[kind.index()];
    if is_friend { friend } else { enemy }
}

/// Combine the mirrored king code with a plane-adjusted piece code.
#[cfg(any(
    feature = "sfnn_halfkahm2_1536_15_32_k3k3",
    feature = "sfnn_halfkahm2_2304_15_64_progress8ek"
))]
#[inline]
fn encode_feature(sq_k_code: usize, p_adj: usize) -> FeatureIndex {
    debug_assert!(sq_k_code < SQ_K_COUNT);
    debug_assert!(p_adj < E_KING);
    let idx = E_KING * sq_k_code + p_adj;
    debug_assert!(idx < NUM_FEATURES);
    idx as FeatureIndex
}

/// Locate `color`'s king.
#[cfg(any(feature = "nnue_halfkp", feature = "sym_halfkp_128x2_512"))]
#[inline]
fn encode_feature(sq_k_code: usize, p_adj: usize) -> FeatureIndex {
    debug_assert!(sq_k_code < SQ_NB);
    debug_assert!(p_adj < FE_END);
    let idx = FE_END * sq_k_code + p_adj;
    debug_assert!(idx < NUM_FEATURES);
    idx as FeatureIndex
}
#[cfg(any(feature = "nnue_kp_256x2_32_32", feature = "sym_kp_128x2_512"))]
#[inline]
fn encode_feature(_sq_k_code: usize, p_adj: usize) -> FeatureIndex {
    debug_assert!(p_adj < NUM_FEATURES);
    p_adj as FeatureIndex
}

/// The king code pieces are crossed on, and whether the board is mirrored to
/// keep that king on the left five files.
#[cfg(any(
    feature = "sfnn_halfkahm2_1536_15_32_k3k3",
    feature = "sfnn_halfkahm2_2304_15_64_progress8ek"
))]
fn king_code(pos: &Position, perspective: Color) -> (usize, bool) {
    let own = from_persp(king_square(pos, perspective), perspective);
    let mirror = needs_mirror(own);
    (mirror_if_needed(own, mirror).index() as usize, mirror)
}
#[cfg(any(feature = "nnue_halfkp", feature = "sym_halfkp_128x2_512"))]
fn king_code(pos: &Position, perspective: Color) -> (usize, bool) {
    (
        from_persp(king_square(pos, perspective), perspective).index() as usize,
        false,
    )
}
#[cfg(any(feature = "nnue_kp_256x2_32_32", feature = "sym_kp_128x2_512"))]
fn king_code(_pos: &Position, _perspective: Color) -> (usize, bool) {
    (0, false)
}

/// The feature index a king on `sq_code` contributes, or `None` when the
/// feature set has no slot for one.
///
/// Above the piece-code level because not every shape's king lands somewhere
/// `encode_feature` can reach: Manaka's HalfKP points both kings at a padding
/// row past the end of the crossed table, which is an index and not a plane.
#[cfg(any(
    feature = "sfnn_halfkahm2_1536_15_32_k3k3",
    feature = "sfnn_halfkahm2_2304_15_64_progress8ek"
))]
#[inline]
fn king_feature(_is_friend: bool, sq_code: usize, sq_k_code: usize) -> Option<FeatureIndex> {
    Some(encode_feature(sq_k_code, FE_END + sq_code))
}
#[cfg(any(feature = "nnue_kp_256x2_32_32", feature = "sym_kp_128x2_512"))]
#[inline]
fn king_feature(is_friend: bool, sq_code: usize, sq_k_code: usize) -> Option<FeatureIndex> {
    let side = if is_friend { 0 } else { SQ_NB };
    Some(encode_feature(sq_k_code, FE_END + side + sq_code))
}
#[cfg(feature = "nnue_halfkp")]
#[inline]
fn king_feature(_is_friend: bool, _sq_code: usize, _sq_k_code: usize) -> Option<FeatureIndex> {
    None
}
#[cfg(feature = "sym_halfkp_128x2_512")]
#[inline]
fn king_feature(_is_friend: bool, _sq_code: usize, _sq_k_code: usize) -> Option<FeatureIndex> {
    Some((NUM_FEATURES - 1) as FeatureIndex)
}

// The KP shapes cross pieces on no king at all (`king_code` ignores the
// position), so only the tests read the king's square there.
///
/// # Panics
/// Panics if `color` has no king: every position fed to the evaluator must have
/// both, as the reference's `pos.king_square` contract requires.
#[cfg(any(
    test,
    not(any(feature = "nnue_kp_256x2_32_32", feature = "sym_kp_128x2_512"))
))]
pub(crate) fn king_square(pos: &Position, color: Color) -> Square {
    pos.king_square(color)
        .unwrap_or_else(|| panic!("position has no {color:?} king"))
}

/// An 81-square mailbox scan, the oracle for [`king_square`].
#[cfg(test)]
fn king_square_scan(pos: &Position, color: Color) -> Square {
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).expect("index < Square::COUNT is valid");
        if let Some(piece) = pos.board().get(sq)
            && piece.kind == PieceKind::King
            && piece.color == color
        {
            return sq;
        }
    }
    panic!("position has no {color:?} king");
}

/// Active `HalfKA_hm2` feature indices for one `perspective` of `pos`: one per
/// board piece, one per held-piece instance, then `BONA_PIECE_ZERO` padding for
/// every empty piece slot. Always exactly [`MAX_ACTIVE_FEATURES`] indices, each
/// `< FEATURE_DIMENSION`, with the padding features the only duplicates.
///
/// The padding is load-bearing. The reference iterates a *fixed* 40 piece-number
/// slots and initialises every unused one to `BONA_PIECE_ZERO`, so each absent
/// piece contributes the feature `E_KING * sq_k_code + 0`. The accumulator sums
/// feature columns, so those repeated zero-features shift the result: omitting
/// them diverges from ground truth on any sparse position.
///
/// # Panics
/// Panics if `pos` is missing the `perspective` side's king.
pub fn active_features(pos: &Position, perspective: Color) -> Vec<FeatureIndex> {
    let mut list = Vec::with_capacity(MAX_ACTIVE_FEATURES);
    active_features_into(pos, perspective, &mut list);
    list
}

/// [`active_features`] writing into a caller-owned buffer, so that the
/// finny-table cache's king-move arm does not allocate per node.
///
/// # Panics
/// Panics if `pos` is missing the `perspective` side's king.
pub(crate) fn active_features_into(
    pos: &Position,
    perspective: Color,
    list: &mut Vec<FeatureIndex>,
) {
    list.clear();
    list.reserve(MAX_ACTIVE_FEATURES);

    let (sq_k_code, mirror) = king_code(pos, perspective);
    scan_active(pos, perspective, sq_k_code, mirror, |feature| {
        list.push(feature)
    });

    // A legal 40-piece position is already full and skips this loop.
    debug_assert!(list.len() <= MAX_ACTIVE_FEATURES);
    while list.len() < MAX_ACTIVE_FEATURES {
        list.push(encode_feature(sq_k_code, BONA_PIECE_ZERO));
    }
}

/// Hand every real active feature of `pos` seen from `perspective` to `emit`:
/// one per board piece in square order, then one per held-piece instance. The
/// `BONA_PIECE_ZERO` padding is the caller's business; this emits none.
///
/// `(sq_k_code, mirror)` is `king_code(pos, perspective)`, passed in because
/// callers need `sq_k_code` for the padding as well.
#[inline(always)]
fn scan_active(
    pos: &Position,
    perspective: Color,
    sq_k_code: usize,
    mirror: bool,
    mut emit: impl FnMut(FeatureIndex),
) {
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).expect("index < Square::COUNT is valid");
        let Some(piece) = pos.board().get(sq) else {
            continue;
        };
        let is_friend = piece.color == perspective;
        let sq_persp = mirror_if_needed(from_persp(sq, perspective), mirror);
        let sq_code = sq_persp.index() as usize;

        if piece.kind == PieceKind::King {
            // `None` is YaneuraOu's HalfKP, which numbers no king at all.
            if let Some(feature) = king_feature(is_friend, sq_code, sq_k_code) {
                emit(feature);
            }
            continue;
        }
        let p_adj = board_plane(piece.kind, piece.promoted, is_friend) + sq_code;
        emit(encode_feature(sq_k_code, p_adj));
    }

    // The k-th held piece of a kind lands at `base + k`.
    for hand_color in [Color::Black, Color::White] {
        let is_friend = hand_color == perspective;
        let hand = pos.hand(hand_color);
        for kind in HAND_KINDS {
            let count = hand.count(kind);
            let base = hand_plane(kind, is_friend);
            for i in 1..=count as usize {
                emit(encode_feature(sq_k_code, base + i));
            }
        }
    }
}

/// Active features for both perspectives, indexed by [`Color::index`].
pub fn active_features_both(pos: &Position) -> [Vec<FeatureIndex>; Color::COUNT] {
    [
        active_features(pos, Color::Black),
        active_features(pos, Color::White),
    ]
}

/// Whether `perspective`'s accumulator half must be fully refreshed after `mv`
/// rather than updated incrementally.
///
/// A `HalfKA_hm2` or HalfKP index embeds the perspective's own-king `sq_k_code`,
/// so when that king moves *every* index shifts — and the mirror flag may flip
/// with it — and no add/sub delta can express the change. The reference's
/// separate mirror-boundary case is subsumed here: only a king move can change
/// the perspective's own-king file. The KP shapes embed no king, so they never
/// need this.
pub fn requires_full_refresh(mv: Move, perspective: Color) -> bool {
    // A KP index does not embed the own king at all (`king_code` is a constant
    // there and the kings are ordinary features of the shared king plane), so
    // there a king move is an add/sub like any other.
    const INDEX_EMBEDS_KING: bool = !cfg!(any(
        feature = "nnue_kp_256x2_32_32",
        feature = "sym_kp_128x2_512"
    ));
    if !INDEX_EMBEDS_KING || mv.is_drop() {
        return false;
    }
    // A king never promotes, so `moved_piece_after` reports it directly.
    let piece = mv.moved_piece_after();
    piece.kind == PieceKind::King && piece.color == perspective
}

/// Offsets a feature can have within one own-king bucket: every shape's index
/// is `bucket_base + offset` with the offset below this (`encode_feature` adds
/// `p_adj < FE_END + 2 * SQ_NB` to a base that depends only on the own king).
#[cfg(not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512")))]
const BUCKET_FEATURES: usize = FE_END + 2 * SQ_NB;
#[cfg(not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512")))]
const BUCKET_WORDS: usize = BUCKET_FEATURES.div_ceil(64);
#[cfg(not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512")))]
const _: () = assert!(E_KING <= BUCKET_FEATURES);

/// The active-feature multiset of one position and perspective, as a bitset over
/// the offsets within its own-king bucket, plus the count of padding copies.
///
/// The multiset difference of two such sets is a few word operations, where the
/// list form had to sort both lists first. That is possible because the only
/// repeated feature is the `BONA_PIECE_ZERO` padding (see
/// [`active_features`]), which is offset `0` and counted rather than stored; a
/// real feature never has offset `0`, and no two real ones coincide.
///
/// Only two sets from the **same bucket** (same perspective and own-king
/// square, hence the same `base`) can be compared, which is exactly how the
/// finny table uses them.
#[cfg(not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512")))]
#[derive(Clone, Debug)]
pub(crate) struct FeatureSet {
    bits: [u64; BUCKET_WORDS],
    /// Index of offset `0`: the bucket's `BONA_PIECE_ZERO` feature.
    base: FeatureIndex,
    /// Copies of the offset-`0` padding feature in the multiset.
    padding: u32,
}

#[cfg(not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512")))]
impl FeatureSet {
    pub(crate) fn new() -> Self {
        FeatureSet {
            bits: [0; BUCKET_WORDS],
            base: 0,
            padding: 0,
        }
    }

    /// Overwrite `self` with the multiset [`active_features`] would list for
    /// `pos` and `perspective`.
    ///
    /// # Panics
    /// Panics if `pos` is missing `perspective`'s king, or (an invariant
    /// violation) if a feature falls outside its bucket's offset range.
    pub(crate) fn fill(&mut self, pos: &Position, perspective: Color) {
        let (sq_k_code, mirror) = king_code(pos, perspective);
        let base = encode_feature(sq_k_code, BONA_PIECE_ZERO);
        let bits = &mut self.bits;
        *bits = [0; BUCKET_WORDS];
        let mut real = 0usize;
        scan_active(pos, perspective, sq_k_code, mirror, |feature| {
            // An index below `base` wraps to a huge offset and trips the bounds
            // check below rather than setting a wrong bit.
            let offset = feature.wrapping_sub(base) as usize;
            debug_assert!(offset != 0, "a real feature coincides with the padding");
            debug_assert_eq!(
                bits[offset / 64] >> (offset % 64) & 1,
                0,
                "two real features coincide"
            );
            bits[offset / 64] |= 1u64 << (offset % 64);
            real += 1;
        });
        debug_assert!(real <= MAX_ACTIVE_FEATURES);
        self.base = base;
        self.padding = MAX_ACTIVE_FEATURES.saturating_sub(real) as u32;
    }

    /// The multiset as an ascending list, padding first -- the order
    /// [`changed_indices_into`] works in.
    #[cfg(test)]
    pub(crate) fn to_indices(&self) -> Vec<FeatureIndex> {
        let mut out = vec![self.base; self.padding as usize];
        for (word_index, &word) in self.bits.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                let offset = word_index * 64 + word.trailing_zeros() as usize;
                out.push(self.base + offset as FeatureIndex);
                word &= word - 1;
            }
        }
        out
    }
}

/// Multiset difference of two [`FeatureSet`]s of the same bucket:
/// `removed = before - after`, `added = after - before`. Both come out in
/// ascending index order with the padding first, which is exactly what
/// [`changed_indices_into`] returns for the two equivalent lists, so a caller
/// can swap one for the other without the result changing in any way.
///
/// # Panics
/// Panics if the sets are from different buckets.
#[cfg(not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512")))]
pub(crate) fn diff_feature_sets(
    before: &FeatureSet,
    after: &FeatureSet,
    removed: &mut Vec<FeatureIndex>,
    added: &mut Vec<FeatureIndex>,
) {
    assert_eq!(
        before.base, after.base,
        "feature sets from different buckets"
    );
    removed.clear();
    added.clear();
    let base = before.base;
    for _ in after.padding..before.padding {
        removed.push(base);
    }
    for _ in before.padding..after.padding {
        added.push(base);
    }
    for word_index in 0..BUCKET_WORDS {
        let (b, a) = (before.bits[word_index], after.bits[word_index]);
        let mut gone = b & !a;
        let mut new = a & !b;
        while gone != 0 {
            let offset = word_index * 64 + gone.trailing_zeros() as usize;
            removed.push(base + offset as FeatureIndex);
            gone &= gone - 1;
        }
        while new != 0 {
            let offset = word_index * 64 + new.trailing_zeros() as usize;
            added.push(base + offset as FeatureIndex);
            new &= new - 1;
        }
    }
}

/// Multiset difference of two active-feature lists: `(removed, added)` such
/// that applying them to the `before` multiset yields `after`.
///
/// The inputs carry their `BONA_PIECE_ZERO` padding, but a legal move conserves
/// total piece count, so for any perspective updated incrementally the padding
/// multiplicity is equal on both sides and cancels.
#[cfg(any(
    test,
    not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512"))
))]
pub(crate) fn changed_indices(
    before: &[FeatureIndex],
    after: &[FeatureIndex],
) -> (Vec<FeatureIndex>, Vec<FeatureIndex>) {
    let mut scratch = DiffScratch::default();
    changed_indices_into(before, after, &mut scratch);
    (scratch.removed, scratch.added)
}

/// Reusable buffers for [`changed_indices_into`], owned by the finny-table cache
/// so that a cached rebuild allocates nothing.
#[derive(Debug, Default)]
pub(crate) struct DiffScratch {
    sorted_before: Vec<FeatureIndex>,
    sorted_after: Vec<FeatureIndex>,
    /// Features present in `before` but not in `after` (multiset semantics).
    pub(crate) removed: Vec<FeatureIndex>,
    /// Features present in `after` but not in `before` (multiset semantics).
    pub(crate) added: Vec<FeatureIndex>,
}

/// [`changed_indices`] writing into caller-owned buffers. This is the single
/// implementation; `changed_indices` is a thin allocating wrapper.
pub(crate) fn changed_indices_into(
    before: &[FeatureIndex],
    after: &[FeatureIndex],
    scratch: &mut DiffScratch,
) {
    let DiffScratch {
        sorted_before,
        sorted_after,
        removed,
        added,
    } = scratch;

    sorted_before.clear();
    sorted_before.extend_from_slice(before);
    sorted_after.clear();
    sorted_after.extend_from_slice(after);
    sorted_before.sort_unstable();
    sorted_after.sort_unstable();
    removed.clear();
    added.clear();

    let (mut i, mut j) = (0usize, 0usize);
    while i < sorted_before.len() && j < sorted_after.len() {
        match sorted_before[i].cmp(&sorted_after[j]) {
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                removed.push(sorted_before[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                added.push(sorted_after[j]);
                j += 1;
            }
        }
    }
    removed.extend_from_slice(&sorted_before[i..]);
    added.extend_from_slice(&sorted_after[j..]);
}

// The hot search path cannot afford [`active_features`]'s full 40-slot scan and
// merge at every `do_move`, so the reference instead tracks the handful of
// "dirty" pieces a move touches and rewrites only their feature columns.
// [`MoveDelta::from_move`] reads those straight off the move and the pre-move
// position.
//
// The result is bit-exact, not approximate: the only `active_features` entries
// that differ between the pre- and post-move positions are exactly these dirty
// pieces, every legal move conserving the total piece count so that the
// `BONA_PIECE_ZERO` padding multiplicity cancels.

/// One board or hand slot a move touches, in a perspective-independent form.
#[derive(Clone, Copy)]
enum Dirty {
    /// A piece sitting on `sq` (a king is encoded via the shared king plane).
    Board { sq: Square, piece: Piece },
    /// The `count`-th (1-based) held piece of `kind` in `color`'s hand.
    Hand {
        color: Color,
        kind: PieceKind,
        count: usize,
    },
}

impl Dirty {
    /// Encode this slot into the feature index for `persp`, whose own-king code
    /// is `(sq_k_code, mirror)`.
    #[inline]
    fn encode(self, persp: Color, sq_k_code: usize, mirror: bool) -> Option<FeatureIndex> {
        match self {
            Dirty::Board { sq, piece } => {
                let sq_code = mirror_if_needed(from_persp(sq, persp), mirror).index() as usize;
                if piece.kind == PieceKind::King {
                    return king_feature(piece.color == persp, sq_code, sq_k_code);
                }
                let p_adj = board_plane(piece.kind, piece.promoted, piece.color == persp) + sq_code;
                Some(encode_feature(sq_k_code, p_adj))
            }
            Dirty::Hand { color, kind, count } => Some(encode_feature(
                sq_k_code,
                hand_plane(kind, color == persp) + count,
            )),
        }
    }
}

/// The add/sub feature delta for one perspective across a move. At most two
/// columns change per side, so the lists are fixed two-slot arrays.
///
/// Plain data, written field by field and read back field by field at the same
/// width. It used to be built zero-initialised, counted in `usize`s, wrapped in
/// an `Option` and moved by value, and the compiled code copied it with 32-byte
/// vector moves out of a stack slot that had just been written with 4-byte
/// stores -- a load spanning several recent stores cannot be store-forwarded
/// (the old profile's hottest instructions in `from_move` were exactly those
/// loads).
#[derive(Clone, Copy)]
pub struct PerspectiveDelta {
    removed: [FeatureIndex; 2],
    added: [FeatureIndex; 2],
    n_removed: u8,
    n_added: u8,
    /// This perspective's own king moved: no add/sub delta exists and its half
    /// must be rebuilt from scratch.
    rebuild: bool,
}

impl PerspectiveDelta {
    /// The marker for a half that is rebuilt rather than updated.
    const REBUILD: PerspectiveDelta = PerspectiveDelta {
        removed: [0; 2],
        added: [0; 2],
        n_removed: 0,
        n_added: 0,
        rebuild: true,
    };

    /// Feature columns to subtract from the pre-move accumulator half.
    #[inline]
    pub fn removed(&self) -> &[FeatureIndex] {
        &self.removed[..self.n_removed as usize]
    }

    /// Feature columns to add to the pre-move accumulator half.
    #[inline]
    pub fn added(&self) -> &[FeatureIndex] {
        &self.added[..self.n_added as usize]
    }
}

/// The per-perspective feature delta a move induces, computed straight from the
/// pre-move position without the [`active_features`] scan.
pub struct MoveDelta {
    halves: [PerspectiveDelta; Color::COUNT],
}

impl MoveDelta {
    /// This move's delta for `perspective`, or `None` if that perspective's own
    /// king moved and its half must be refreshed from scratch.
    #[inline]
    pub fn half(&self, perspective: Color) -> Option<&PerspectiveDelta> {
        let half = &self.halves[perspective.index()];
        if half.rebuild { None } else { Some(half) }
    }

    /// Compute the delta for `mv` from the **pre-move** `pos`, which is not
    /// mutated.
    ///
    /// # Panics
    /// Panics if a non-drop move has no piece on its origin square, or if an
    /// incrementally-updated perspective is missing its king.
    #[inline]
    pub fn from_move(pos: &Position, mv: Move) -> MoveDelta {
        let mover = pos.side_to_move();
        let to = mv.to_sq();

        // The three shapes a move can take are spelled out so that each
        // `assemble` sees its slots, and their arities, as constants.
        if mv.is_drop() {
            // The top held slot leaves the hand and an unpromoted board piece
            // appears at `to`.
            let kind = mv.dropped_piece_kind();
            let count = pos.hand(mover).count(kind) as usize;
            return Self::assemble(
                pos,
                mv,
                [Dirty::Hand {
                    color: mover,
                    kind,
                    count,
                }],
                [Dirty::Board {
                    sq: to,
                    piece: Piece::new(kind, mover),
                }],
            );
        }

        let from = mv.from_sq();
        // The `Move` encoding already carries the moving piece, so there is no
        // need to re-read the board. `before` undoes any promotion.
        let after = mv.moved_piece_after();
        let before = if mv.is_promote() {
            Piece {
                promoted: false,
                ..after
            }
        } else {
            after
        };
        debug_assert_eq!(
            pos.board().get(from),
            Some(before),
            "Move encoding disagrees with the board on the mover at {from:?}",
        );
        let lifted = Dirty::Board {
            sq: from,
            piece: before,
        };
        let placed = Dirty::Board {
            sq: to,
            piece: after,
        };

        match pos.board().get(to) {
            None => Self::assemble(pos, mv, [lifted], [placed]),
            // The victim leaves the board and enters the mover's hand as its
            // unpromoted kind, at the next count slot.
            Some(captured) => {
                let base = captured.kind;
                let count = pos.hand(mover).count(base) as usize + 1;
                Self::assemble(
                    pos,
                    mv,
                    [
                        lifted,
                        Dirty::Board {
                            sq: to,
                            piece: captured,
                        },
                    ],
                    [
                        placed,
                        Dirty::Hand {
                            color: mover,
                            kind: base,
                            count,
                        },
                    ],
                )
            }
        }
    }

    /// Encode the dirty slots for both perspectives.
    #[inline(always)]
    fn assemble<const R: usize, const A: usize>(
        pos: &Position,
        mv: Move,
        removed: [Dirty; R],
        added: [Dirty; A],
    ) -> MoveDelta {
        MoveDelta {
            halves: [
                Self::encode_half(pos, mv, Color::Black, &removed, &added),
                Self::encode_half(pos, mv, Color::White, &removed, &added),
            ],
        }
    }

    #[inline(always)]
    fn encode_half<const R: usize, const A: usize>(
        pos: &Position,
        mv: Move,
        persp: Color,
        removed: &[Dirty; R],
        added: &[Dirty; A],
    ) -> PerspectiveDelta {
        if requires_full_refresh(mv, persp) {
            return PerspectiveDelta::REBUILD;
        }
        // The perspective's own king did not move, so its `(sq_k_code,
        // mirror)` is the same pre- and post-move; read it from `pos`.
        //
        // Through `king_code`, not `needs_mirror` spelled out again: which
        // king code a shape crosses its pieces on, and whether it mirrors
        // at all, is exactly what differs between the feature sets. This
        // used to recompute the `hm` mirror unconditionally, so the KP
        // shapes -- whose `king_code` is `(0, false)` and whose
        // `active_features` therefore never mirrors -- got a delta whose
        // squares were file-mirrored whenever that perspective's king stood
        // on files 6-9. The incremental accumulator then walked away from a
        // full refresh, silently, from that move on.
        let (sq_k_code, mirror) = king_code(pos, persp);

        let mut pd = PerspectiveDelta {
            removed: [0; 2],
            added: [0; 2],
            n_removed: 0,
            n_added: 0,
            rebuild: false,
        };
        // A slot that encodes to nothing is dropped rather than counted.
        // Only HalfKP produces one, and only for a king -- whose move that
        // space answers with a full refresh anyway.
        for d in removed {
            if let Some(index) = d.encode(persp, sq_k_code, mirror) {
                pd.removed[pd.n_removed as usize] = index;
                pd.n_removed += 1;
            }
        }
        for d in added {
            if let Some(index) = d.encode(persp, sq_k_code, mirror) {
                pd.added[pd.n_added as usize] = index;
                pd.n_added += 1;
            }
        }
        pd
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use attic_state::{parse_sfen, parse_usi_move};
    use std::collections::HashSet;

    const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
    // Sparse and hand-heavy: 6 board pieces and 6 held ones.
    const DROP_HEAVY: &str = "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1";

    /// Startpos puts forty pieces on distinct squares, so the list carries
    /// forty distinct features — except under Manaka's HalfKP, where both kings
    /// land on the one padding row it appends and the duplicate is deliberate.
    #[cfg(not(feature = "sym_halfkp_128x2_512"))]
    const DISTINCT_AT_STARTPOS: usize = MAX_ACTIVE_FEATURES;
    #[cfg(feature = "sym_halfkp_128x2_512")]
    const DISTINCT_AT_STARTPOS: usize = MAX_ACTIVE_FEATURES - 1;

    /// How many distinct feature values those twelve pieces produce. The two
    /// that move between shapes are the kings: YaneuraOu's HalfKP has no row
    /// for them at all, and Manaka's HalfKP sends both to the same padding row.
    #[cfg(not(any(feature = "nnue_halfkp", feature = "sym_halfkp_128x2_512")))]
    const DISTINCT_REAL_IN_DROP_HEAVY: usize = 12;
    #[cfg(feature = "nnue_halfkp")]
    const DISTINCT_REAL_IN_DROP_HEAVY: usize = 10;
    #[cfg(feature = "sym_halfkp_128x2_512")]
    const DISTINCT_REAL_IN_DROP_HEAVY: usize = 11;

    fn sorted(list: &[FeatureIndex]) -> Vec<FeatureIndex> {
        let mut v = list.to_vec();
        v.sort_unstable();
        v
    }

    fn distinct(list: &[FeatureIndex]) -> usize {
        list.iter().copied().collect::<HashSet<_>>().len()
    }

    // Against `NUM_FEATURES`, the width of the transformer's weight table, and
    // not `FEATURE_DIMENSION`, which is `HalfKA_hm2`'s crossed size and is far
    // wider than the table the KP shapes actually index.
    fn assert_in_bounds(list: &[FeatureIndex]) {
        for &idx in list {
            assert!(
                (idx as usize) < NUM_FEATURES,
                "feature index {idx} out of bounds",
            );
        }
    }

    /// Horizontal mirror of `pos`: every board piece file `f -> 8 - f`, ranks
    /// and hands unchanged (a file mirror leaves held pieces untouched).
    ///
    /// Only `HalfKA_hm2` canonicalizes on the mirror; K, P and HalfKP all
    /// number the squares absolutely, so a mirrored position is simply a
    /// different position to them.
    #[cfg(any(
        feature = "sfnn_halfkahm2_1536_15_32_k3k3",
        feature = "sfnn_halfkahm2_2304_15_64_progress8ek"
    ))]
    fn mirror_position(pos: &Position) -> Position {
        let mut mirrored = Position::empty();
        for index in 0..Square::COUNT as u8 {
            let sq = Square::from_index(index).unwrap();
            if let Some(piece) = pos.board().get(sq) {
                let msq = Square::new(Square::FILES - 1 - sq.file(), sq.rank()).unwrap();
                mirrored.board_mut().set(msq, Some(piece));
            }
        }
        for color in [Color::Black, Color::White] {
            for kind in HAND_KINDS {
                for _ in 0..pos.hand(color).count(kind) {
                    mirrored.hand_mut(color).increment(kind);
                }
            }
        }
        mirrored.set_side_to_move(pos.side_to_move());
        mirrored
    }

    #[test]
    fn king_square_matches_scan_oracle() {
        for sfen in [STARTPOS, DROP_HEAVY] {
            let pos = parse_sfen(sfen).unwrap();
            for color in [Color::Black, Color::White] {
                assert_eq!(
                    king_square(&pos, color),
                    king_square_scan(&pos, color),
                    "sfen `{sfen}` color {color:?}",
                );
            }
        }
    }

    #[test]
    fn plane_constants_match_reference() {
        // BonaPiece's own numbering. Every shape shares it; what they differ on
        // is how a piece code is crossed with the king.
        assert_eq!(FE_HAND_END, 90);
        assert_eq!(FE_END, 1_548);
        assert_eq!(E_KING, 1_629);

        // `FEATURE_DIMENSION` is the crossed `HalfKA_hm2` table, so only that
        // shape's `NUM_FEATURES` is it. The other three size their tables in
        // `types.rs`, and `num_features_matches_shape` there pins each one.
        #[cfg(any(
            feature = "sfnn_halfkahm2_1536_15_32_k3k3",
            feature = "sfnn_halfkahm2_2304_15_64_progress8ek"
        ))]
        {
            assert_eq!(FEATURE_DIMENSION, NUM_FEATURES);
            assert_eq!(FEATURE_DIMENSION, SQ_K_COUNT * E_KING);
        }
    }

    #[test]
    fn startpos_has_forty_features_per_perspective() {
        let pos = parse_sfen(STARTPOS).unwrap();
        for persp in [Color::Black, Color::White] {
            let list = active_features(&pos, persp);
            assert_eq!(
                list.len(),
                MAX_ACTIVE_FEATURES,
                "perspective {persp:?}: expected 40 active features",
            );
            assert_eq!(
                distinct(&list),
                DISTINCT_AT_STARTPOS,
                "perspective {persp:?}: wrong number of distinct features",
            );
            assert_in_bounds(&list);
        }
    }

    #[cfg(any(
        feature = "sfnn_halfkahm2_1536_15_32_k3k3",
        feature = "sfnn_halfkahm2_2304_15_64_progress8ek"
    ))]
    #[test]
    fn startpos_both_kings_share_one_plane() {
        let pos = parse_sfen(STARTPOS).unwrap();
        for persp in [Color::Black, Color::White] {
            let (sq_k_code, _mirror) = king_code(&pos, persp);
            let plane_lo = (E_KING * sq_k_code + FE_END) as FeatureIndex;
            let plane_hi = (E_KING * sq_k_code + E_KING) as FeatureIndex;

            let list = active_features(&pos, persp);
            let kings = list
                .iter()
                .filter(|&&f| f >= plane_lo && f < plane_hi)
                .count();
            assert_eq!(kings, 2, "both kings should land in the shared king plane");
        }
    }

    /// The K half of `FeatureSet<K, P>`: two planes of eighty-one appended to
    /// BonaPiece's 1,548, friend below enemy. That second plane is what makes
    /// the table 1,710 wide rather than 1,629, and it is the one place the KP
    /// shapes and `HalfKA_hm2` disagree about where a king goes.
    #[cfg(any(feature = "nnue_kp_256x2_32_32", feature = "sym_kp_128x2_512"))]
    #[test]
    fn startpos_kings_take_one_slot_in_each_king_plane() {
        let pos = parse_sfen(STARTPOS).unwrap();
        for persp in [Color::Black, Color::White] {
            let list = active_features(&pos, persp);
            let in_plane = |lo: usize, hi: usize| {
                list.iter()
                    .filter(|&&f| (f as usize) >= lo && (f as usize) < hi)
                    .count()
            };
            assert_eq!(in_plane(FE_END, FE_END + SQ_NB), 1, "{persp:?}: own king");
            assert_eq!(
                in_plane(FE_END + SQ_NB, FE_END + 2 * SQ_NB),
                1,
                "{persp:?}: enemy king",
            );
        }
    }

    #[test]
    fn active_features_is_deterministic() {
        let pos = parse_sfen(STARTPOS).unwrap();
        assert_eq!(
            active_features(&pos, Color::Black),
            active_features(&pos, Color::Black),
        );
    }

    #[test]
    fn both_perspectives_helper_matches_single() {
        let pos = parse_sfen(DROP_HEAVY).unwrap();
        let both = active_features_both(&pos);
        assert_eq!(
            both[Color::Black.index()],
            active_features(&pos, Color::Black)
        );
        assert_eq!(
            both[Color::White.index()],
            active_features(&pos, Color::White)
        );
    }

    #[test]
    fn drop_heavy_extracts_cleanly_and_indexes_hands() {
        let pos = parse_sfen(DROP_HEAVY).unwrap();
        for persp in [Color::Black, Color::White] {
            let list = active_features(&pos, persp);
            assert_eq!(list.len(), MAX_ACTIVE_FEATURES, "perspective {persp:?}");
            assert_in_bounds(&list);

            // The real features are distinct; every remaining slot repeats the
            // single BONA_PIECE_ZERO feature. Through `king_code` and
            // `encode_feature`, so the arithmetic is the shape's own rather
            // than `HalfKA_hm2`'s spelled out a second time.
            let (sq_k_code, _mirror) = king_code(&pos, persp);
            let pad = encode_feature(sq_k_code, BONA_PIECE_ZERO);
            // Twenty-eight in every shape: the two HalfKP leaves out are two
            // it does not walk either, so both sides of the subtraction move.
            assert_eq!(
                list.iter().filter(|&&f| f == pad).count(),
                28,
                "perspective {persp:?}: expected 28 padding features",
            );
            assert_eq!(
                distinct(&list),
                DISTINCT_REAL_IN_DROP_HEAVY + 1,
                "perspective {persp:?}: distinct real features + 1 padding value",
            );
        }
    }

    #[test]
    fn every_position_yields_exactly_forty_features() {
        // A full and a sparse position must both produce exactly
        // PIECE_NUMBER_NB features per perspective.
        for sfen in [STARTPOS, DROP_HEAVY] {
            let pos = parse_sfen(sfen).unwrap();
            for persp in [Color::Black, Color::White] {
                assert_eq!(
                    active_features(&pos, persp).len(),
                    MAX_ACTIVE_FEATURES,
                    "sfen `{sfen}` perspective {persp:?}",
                );
            }
        }
    }

    #[test]
    fn hand_piece_indices_span_consecutive_slots() {
        // Each side holds two pawns and a gold, so from Black's perspective the
        // friend pawns occupy `F_HAND_PAWN+{1,2}` and the enemy ones
        // `E_HAND_PAWN+{1,2}`, all offset by the same king block.
        let pos = parse_sfen(DROP_HEAVY).unwrap();
        let (sq_k_code, _mirror) = king_code(&pos, Color::Black);

        let list: HashSet<_> = active_features(&pos, Color::Black).into_iter().collect();
        for expected in [
            encode_feature(sq_k_code, F_HAND_PAWN + 1),
            encode_feature(sq_k_code, F_HAND_PAWN + 2),
            encode_feature(sq_k_code, F_HAND_GOLD + 1),
            encode_feature(sq_k_code, E_HAND_PAWN + 1),
            encode_feature(sq_k_code, E_HAND_PAWN + 2),
            encode_feature(sq_k_code, E_HAND_GOLD + 1),
        ] {
            assert!(list.contains(&expected), "missing hand feature {expected}");
        }
    }

    #[cfg(any(
        feature = "sfnn_halfkahm2_1536_15_32_k3k3",
        feature = "sfnn_halfkahm2_2304_15_64_progress8ek"
    ))]
    #[test]
    fn mirror_property_preserves_feature_set() {
        // Both kings sit off the centre file, so the mirror flag flips between
        // a position and its mirror and the two share one feature set per
        // perspective. A centre-file king is a genuinely distinct position.
        let sfen = "2k6/9/4+R4/8P/9/1n7/9/9/6K2 b B2Pgp 1";
        let pos = parse_sfen(sfen).unwrap();
        let mirrored = mirror_position(&pos);
        for persp in [Color::Black, Color::White] {
            assert_eq!(
                sorted(&active_features(&pos, persp)),
                sorted(&active_features(&mirrored, persp)),
                "perspective {persp:?}: mirror changed the feature set",
            );
        }
    }

    /// Reconstruct the `after` multiset from `before + added - removed`, an
    /// independent oracle for [`changed_indices`].
    fn apply_multiset(
        before: &[FeatureIndex],
        removed: &[FeatureIndex],
        added: &[FeatureIndex],
    ) -> Vec<FeatureIndex> {
        let mut counts: std::collections::BTreeMap<FeatureIndex, i32> =
            std::collections::BTreeMap::new();
        for &x in before {
            *counts.entry(x).or_insert(0) += 1;
        }
        for &x in removed {
            *counts.entry(x).or_insert(0) -= 1;
        }
        for &x in added {
            *counts.entry(x).or_insert(0) += 1;
        }
        let mut out = Vec::new();
        for (&k, &c) in &counts {
            assert!(c >= 0, "negative multiplicity for {k}");
            for _ in 0..c {
                out.push(k);
            }
        }
        out
    }

    /// Diff the pre- and post-move feature lists for a perspective, as the
    /// accumulator does, and confirm the delta reconstructs the post-move
    /// multiset.
    fn check_diff_invariant(sfen: &str, usi: &str) {
        let mut pos = parse_sfen(sfen).unwrap();
        let mv = parse_usi_move(usi, &pos).unwrap();
        for persp in [Color::Black, Color::White] {
            assert!(
                !requires_full_refresh(mv, persp),
                "test move {usi} must be incremental for {persp:?}"
            );
            let before = active_features(&pos, persp);
            let undo = pos.do_move(mv);
            let after = active_features(&pos, persp);
            pos.undo_move(mv, undo);

            let (removed, added) = changed_indices(&before, &after);
            assert_eq!(
                apply_multiset(&before, &removed, &added),
                sorted(&after),
                "{persp:?}: diff does not reconstruct post-move features for {usi}",
            );
        }
    }

    #[test]
    fn requires_full_refresh_flags_only_own_king_moves() {
        let pos = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").unwrap();
        let king_move = parse_usi_move("5i5h", &pos).unwrap();
        // Where the index embeds the own king the mover's half rebuilds; where
        // it does not (KP) a king move is an ordinary add/sub.
        let embeds_king = !cfg!(any(
            feature = "nnue_kp_256x2_32_32",
            feature = "sym_kp_128x2_512"
        ));
        assert_eq!(requires_full_refresh(king_move, Color::Black), embeds_king);
        assert!(!requires_full_refresh(king_move, Color::White));

        let start = parse_sfen(STARTPOS).unwrap();
        let pawn = parse_usi_move("7g7f", &start).unwrap();
        assert!(!requires_full_refresh(pawn, Color::Black));
        assert!(!requires_full_refresh(pawn, Color::White));
    }

    #[test]
    fn requires_full_refresh_drops_are_never_refresh() {
        let pos = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1").unwrap();
        let drop = parse_usi_move("P*5f", &pos).unwrap();
        assert!(!requires_full_refresh(drop, Color::Black));
        assert!(!requires_full_refresh(drop, Color::White));
    }

    #[test]
    fn changed_indices_reconstructs_after_for_all_move_types() {
        check_diff_invariant(STARTPOS, "7g7f");
        check_diff_invariant("4k4/1p7/9/9/9/9/9/1R7/4K4 b - 1", "8h8b");
        check_diff_invariant("4k4/9/9/9/9/9/9/9/4K4 b P 1", "P*5e");
        check_diff_invariant("4k4/9/9/1P7/9/9/9/9/4K4 b - 1", "8d8c+");
        check_diff_invariant("4k4/2p6/9/9/9/9/9/1B7/4K4 b - 1", "8h7b+");
    }

    /// Drive [`MoveDelta::from_move`] — the path the search actually takes —
    /// against the `active_features` scan.
    ///
    /// [`check_diff_invariant`] above drives `changed_indices`, which is handed
    /// two lists the scan produced; it cannot see a delta that encodes its own
    /// squares wrongly. Nothing covered `from_move` itself, and that is the gap
    /// the mirror bug walked through: it recomputed the `hm` file mirror rather
    /// than taking it from `king_code`, so under a shape that does not mirror,
    /// a perspective whose own king stood on files 6 to 9 accumulated
    /// file-mirrored columns and never came back.
    fn check_move_delta_invariant(sfen: &str, usi: &str) {
        let mut pos = parse_sfen(sfen).unwrap();
        let mv = parse_usi_move(usi, &pos).unwrap();

        let delta = MoveDelta::from_move(&pos, mv);
        let before = active_features_both(&pos);
        let undo = pos.do_move(mv);
        let after = active_features_both(&pos);
        pos.undo_move(mv, undo);

        for persp in [Color::Black, Color::White] {
            let i = persp.index();
            match delta.half(persp) {
                Some(half) => assert_eq!(
                    apply_multiset(&before[i], half.removed(), half.added()),
                    sorted(&after[i]),
                    "{persp:?}: delta does not reconstruct post-move features for {usi}",
                ),
                None => assert!(
                    requires_full_refresh(mv, persp),
                    "{persp:?}: refused a delta for {usi} without an own-king move",
                ),
            }
        }
    }

    #[test]
    fn move_delta_reconstructs_after_for_all_move_types() {
        // Quiet, capture, drop, promotion, capture-promotion, king move.
        check_move_delta_invariant(STARTPOS, "7g7f");
        check_move_delta_invariant("4k4/1p7/9/9/9/9/9/1R7/4K4 b - 1", "8h8b");
        check_move_delta_invariant("4k4/9/9/9/9/9/9/9/4K4 b P 1", "P*5e");
        check_move_delta_invariant("4k4/9/9/1P7/9/9/9/9/4K4 b - 1", "8d8c+");
        check_move_delta_invariant("4k4/2p6/9/9/9/9/9/1B7/4K4 b - 1", "8h7b+");
        check_move_delta_invariant(STARTPOS, "5i5h");
    }

    /// Every legal move of a spread of positions, not a hand-picked few: both
    /// sides to move, hands full and empty, promoted pieces, drops, captures
    /// (including a capture of a promoted piece), promotions and king moves.
    /// Each is held against the `active_features` scan by the same invariant.
    ///
    /// Every position must be legal for the side it names: with the other king
    /// en prise the generator offers to capture it, and the shapes that read
    /// the king square (HalfKP, SFNN) then panic on a position with no king.
    #[test]
    fn move_delta_reconstructs_after_for_every_legal_move() {
        let positions = [
            STARTPOS,
            DROP_HEAVY,
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
            "pnsg1k2l/l1rg2s2/+P1p3n1p/Pp2pp3/L2p2pp1/1B2P2P1/NP1PGPPRP/3S1K2L/3G1S1N1 b B 55",
            "pnsg1k2l/l1rg2s2/+P1p3n1p/Pp2pp3/L2p2pp1/1B2P2P1/NP1PGPPRP/3S1K2L/3G1S1N1 w B 56",
            "4k4/2pP5/9/3+r5/9/6+b2/9/1R1+B5/4K4 b 2G2S2N2L8Pgsnl 1",
            "4k4/2pP5/9/3+r5/9/6+b2/9/1R1+B5/4K4 w 2G2S2N2L8Pgsnl 1",
            "k8/9/9/9/9/9/9/9/8K b RB2G2S2N2L9Prb2g2s2n2l9p 1",
            "1k7/9/9/9/9/9/9/9/7K1 w 3P3p 1",
        ];
        let mut checked = 0;
        for sfen in positions {
            let pos = parse_sfen(sfen).unwrap();
            let mut moves = Vec::new();
            pos.generate_legal_all(&mut moves);
            assert!(!moves.is_empty(), "no legal moves in {sfen}");
            for mv in moves {
                check_move_delta_invariant(sfen, &attic_state::format_usi_move(mv));
                checked += 1;
            }
        }
        assert!(checked > 600, "only {checked} moves were checked");
    }

    /// The bitset difference is the sorted-merge difference, element for element
    /// and in the same order -- on real games (full 40-piece positions, same
    /// bucket revisited many times) and on sparse positions, where the padding
    /// count differs between the two sides.
    #[cfg(not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512")))]
    #[test]
    fn feature_set_diff_equals_the_sorted_merge() {
        use std::collections::HashMap;

        fn check(persp: Color, before: &Position, after: &Position, what: &str) {
            let (kb, ka) = (before.king_square(persp), after.king_square(persp));
            assert_eq!(kb, ka, "test bug: not the same bucket");
            let (list_b, list_a) = (
                active_features(before, persp),
                active_features(after, persp),
            );
            let (want_removed, want_added) = changed_indices(&list_b, &list_a);

            let (mut set_b, mut set_a) = (FeatureSet::new(), FeatureSet::new());
            set_b.fill(before, persp);
            set_a.fill(after, persp);
            assert_eq!(set_b.to_indices(), sorted(&list_b), "{what}: before set");
            assert_eq!(set_a.to_indices(), sorted(&list_a), "{what}: after set");

            let (mut removed, mut added) = (Vec::new(), Vec::new());
            diff_feature_sets(&set_b, &set_a, &mut removed, &mut added);
            assert_eq!(removed, want_removed, "{what}: removed");
            assert_eq!(added, want_added, "{what}: added");
        }

        // 1. Random games: remember the last position seen per (perspective,
        // own-king square) and diff against it whenever the bucket recurs.
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut compared = 0;
        let mut last: HashMap<(usize, u8), Position> = HashMap::new();
        for _ in 0..30 {
            let mut pos = parse_sfen(STARTPOS).unwrap();
            for ply in 0..120 {
                for persp in [Color::Black, Color::White] {
                    let king = pos.king_square(persp).unwrap().index();
                    if let Some(prev) = last.get(&(persp.index(), king)) {
                        check(persp, prev, &pos, &format!("game ply {ply}"));
                        compared += 1;
                    }
                    last.insert((persp.index(), king), pos.clone());
                }
                let mut moves = Vec::new();
                pos.generate_legal_all(&mut moves);
                if moves.is_empty() {
                    break;
                }
                let mv = moves[(next() % moves.len() as u64) as usize];
                let _ = pos.do_move(mv);
            }
        }
        assert!(
            compared > 500,
            "only {compared} same-bucket pairs were compared"
        );

        // 2. Sparse positions with the kings fixed: the padding count differs
        // (5 pieces vs 12 vs 2), pieces appear and vanish, hands change.
        let sparse = [
            "4k4/9/9/9/9/9/9/9/4K4 b - 1",
            "4k4/9/9/9/9/9/9/9/4K4 b P 1",
            "4k4/9/9/9/9/9/9/9/4K4 w 3P3p 1",
            "4k4/2p+P5/9/3+r5/9/6+b2/9/1R1+B5/4K4 w 2G2S2N2L8Pgsnl 1",
            "4k4/9/4p4/9/9/9/4P4/9/4K4 b 2G 1",
            "4k4/9/9/9/9/9/9/9/4K4 b RB2G2S2N2L9Prb2g2s2n2l9p 1",
        ];
        let positions: Vec<Position> = sparse.iter().map(|f| parse_sfen(f).unwrap()).collect();
        for persp in [Color::Black, Color::White] {
            for a in &positions {
                for b in &positions {
                    check(persp, a, b, "sparse pair");
                }
            }
        }
    }

    #[test]
    fn move_delta_holds_with_a_king_off_the_left_five_files() {
        // Black's king on 8i: file index 7, which is what `needs_mirror` fires
        // on. Under `HalfKA_hm2` the mirror is real and both sides of this
        // equality carry it; under K, P and HalfKP there is no mirror, and a
        // delta that applied one anyway would land on the wrong columns here
        // and nowhere else. Both kings are covered: White's is on 5a, a file
        // that mirrors from its own perspective too.
        check_move_delta_invariant("4k4/9/9/9/9/9/9/9/1K7 b P 1", "P*5e");
        check_move_delta_invariant("1k7/9/9/9/9/9/9/9/7K1 b P 1", "P*5e");

        // A real position from the KP256 divergence hunt, where the two engines
        // first disagreed.
        check_move_delta_invariant(
            "pnsg1k2l/l1rg2s2/+P1p3n1p/Pp2pp3/L2p2pp1/1B2P2P1/NP1PGPPRP/3S1K2L/3G1S1N1 b B 55",
            "2f2e",
        );
    }

    #[test]
    fn changed_indices_of_identical_lists_is_empty() {
        let pos = parse_sfen(STARTPOS).unwrap();
        let feats = active_features(&pos, Color::Black);
        let (removed, added) = changed_indices(&feats, &feats);
        assert!(removed.is_empty() && added.is_empty());
    }

    #[cfg(any(
        feature = "sfnn_halfkahm2_1536_15_32_k3k3",
        feature = "sfnn_halfkahm2_2304_15_64_progress8ek"
    ))]
    #[test]
    fn mirror_property_needs_mirror_flag_actually_flips() {
        // Exactly one of the two positions must trigger the mirror path for each
        // perspective, so that the test above is not a trivial no-op.
        let sfen = "2k6/9/4+R4/8P/9/1n7/9/9/6K2 b B2Pgp 1";
        let pos = parse_sfen(sfen).unwrap();
        let mirrored = mirror_position(&pos);
        for persp in [Color::Black, Color::White] {
            let m_pos = needs_mirror(from_persp(king_square(&pos, persp), persp));
            let m_mir = needs_mirror(from_persp(king_square(&mirrored, persp), persp));
            assert_ne!(
                m_pos, m_mir,
                "perspective {persp:?}: mirror flag did not flip"
            );
        }
    }
}
