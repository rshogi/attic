//! The two hand-crafted evaluations, Material (17) and Mobility (164).
//!
//! Both are a dot product over a feature vector counted off the board, and
//! both were fitted by Ridge against the corpus V4 trained on
//! (`crates/manaka-pack/src/bin/manaka-hce-fit.rs` in the parent repository).
//! The coefficients arrive as a generated source file rather than as something
//! read at startup, which is what the C++ engines do too: seventeen and a
//! hundred and sixty-four numbers are not worth a file format.
//!
//! ## Why this sits behind NNUE's names
//!
//! [`crate::manaka`] already established that the search binds to a handful of
//! names — `Accumulator`, `FinnyCache`, `MoveDelta`, `NnueNetwork`,
//! `evaluate_with` — and that an evaluator which is not NNUE can be compiled
//! in behind them. This is the third such evaluator and it borrows the same
//! seam, so `qsearch` needs no edit to run it.
//!
//! ## No accumulator
//!
//! One move changes the attack map over a wide part of the board — every
//! slider whose ray the mover crossed, and every drop square the mover freed
//! or filled — so there is no small delta to carry. [`Accumulator`] therefore
//! holds nothing, `derive_into_cached` copies nothing, and [`evaluate_with`]
//! recomputes the whole vector. That is a real cost and it is paid knowingly:
//! the first thing to establish is that the numbers match the C++ engines,
//! and the price of that is measurable with `bench` afterwards.
//!
//! ## Not divided by `FV_SCALE`
//!
//! NNUE and Manaka both produce a fixed-point quantity that `FV_SCALE` turns
//! into YaneuraOu's internal units. These weights were fitted **in** internal
//! units (`PawnValue = 90`), so the sum is already the value the search wants
//! and [`crate::fv_scale`] is deliberately not consulted here.
//!
//! ## What is not carried over
//!
//! `evaluate_material.cpp` also writes its learned values into `PieceValue` /
//! `CapturePieceValue` / `ProDiffPieceValue`, so that SEE and the futility
//! margins prune on the same ruler the evaluation uses. Attic's equivalents
//! are `attic-state`'s `see.rs` table and `qsearch.rs`'s `PAWN_VALUE`, both
//! `const`, and both already on the same 90-per-pawn scale. They keep
//! YaneuraOu's default piece values rather than the fitted ones; that is a
//! difference from the C++ side and it is left alone here on purpose, so that
//! the first comparison changes one thing.

use std::path::Path;
use std::sync::LazyLock;

use attic_state::bitboard::{
    bishop_attacks, dragon_attacks, gold_attacks, horse_attacks, king_attacks, knight_attacks,
    lance_attacks, pawn_attacks, rook_attacks, silver_attacks,
};
use attic_state::{Bitboard, Color, Move, Piece, PieceKind, Position, Square};

use crate::types::NnueError;

// The shape says which file, and `hce_fit_qsearched` says which fit of that
// shape. A modifier rather than two more shapes, because the arrays are the
// same length and the extraction is the same code -- only the numbers differ,
// and both fits have to stay in the tree at once for the two binaries to be
// buildable from one checkout.
#[cfg(all(feature = "hce_material_17", not(feature = "hce_fit_qsearched")))]
#[path = "hce_weights_material.rs"]
mod weights;
#[cfg(all(feature = "hce_material_17", feature = "hce_fit_qsearched"))]
#[path = "hce_weights_material_qs.rs"]
mod weights;
#[cfg(all(feature = "hce_mobility_107_57", not(feature = "hce_fit_qsearched")))]
#[path = "hce_weights_mobility.rs"]
mod weights;
#[cfg(all(feature = "hce_mobility_107_57", feature = "hce_fit_qsearched"))]
#[path = "hce_weights_mobility_qs.rs"]
mod weights;

/// Mobility: 107 board features then 57 hand features.
pub const NUM_MOBILITY: usize = 164;
const NUM_BOARD_FEATURES: usize = 107;

/// Material: 10 board slots (no king, the four gold-shaped promotions merged)
/// then 7 hand slots.
pub const NUM_MATERIAL: usize = 17;

/// How many coefficients the compiled-in shape has.
#[cfg(feature = "hce_material_17")]
pub const DIM: usize = NUM_MATERIAL;
#[cfg(feature = "hce_mobility_107_57")]
pub const DIM: usize = NUM_MOBILITY;

/// What `evaluate_mobility.cpp:54-56` clamps its sum to, in internal units.
/// Material has no such clamp, and neither does this.
#[cfg(feature = "hce_mobility_107_57")]
const MOBILITY_CLAMP: f64 = 3000.0;

// ---------------------------------------------------------------------------
// Piece numbering
// ---------------------------------------------------------------------------

/// The fourteen piece types as one index, in the order the fit used.
///
/// `attic-state` spells a piece as `(kind, promoted)` where YaneuraOu spells
/// it as one of fourteen values, and the index tables below are keyed by the
/// latter. Written as a match rather than as arithmetic so that a reader can
/// check it against `humanlike_eval.cpp` line by line.
fn pt_index(kind: PieceKind, promoted: bool) -> usize {
    match (kind, promoted) {
        (PieceKind::King, _) => 0,
        (PieceKind::Rook, false) => 1,
        (PieceKind::Bishop, false) => 2,
        (PieceKind::Gold, _) => 3,
        (PieceKind::Silver, false) => 4,
        (PieceKind::Knight, false) => 5,
        (PieceKind::Lance, false) => 6,
        (PieceKind::Pawn, false) => 7,
        (PieceKind::Rook, true) => 8,
        (PieceKind::Bishop, true) => 9,
        (PieceKind::Silver, true) => 10,
        (PieceKind::Knight, true) => 11,
        (PieceKind::Lance, true) => 12,
        (PieceKind::Pawn, true) => 13,
    }
}

const NUM_PIECE_TYPES: usize = 14;

/// Attacks of `piece` standing on `sq` under occupancy `occ`, own pieces not
/// excluded — `effects_from` does not exclude them either, so a piece
/// defending its own is counted.
fn attacks_from(piece: Piece, sq: Square, occ: Bitboard) -> Bitboard {
    match (piece.kind, piece.promoted) {
        (PieceKind::Pawn, false) => pawn_attacks(piece.color, sq),
        (PieceKind::Lance, false) => lance_attacks(piece.color, sq, occ),
        (PieceKind::Knight, false) => knight_attacks(piece.color, sq),
        (PieceKind::Silver, false) => silver_attacks(piece.color, sq),
        (PieceKind::Gold, _) => gold_attacks(piece.color, sq),
        // The four gold-shaped promotions move as gold, which is also why they
        // share gold's feature slots below.
        (PieceKind::Pawn | PieceKind::Lance | PieceKind::Knight | PieceKind::Silver, true) => {
            gold_attacks(piece.color, sq)
        }
        (PieceKind::Bishop, false) => bishop_attacks(sq, occ),
        (PieceKind::Rook, false) => rook_attacks(sq, occ),
        (PieceKind::Bishop, true) => horse_attacks(sq, occ),
        (PieceKind::Rook, true) => dragon_attacks(sq, occ),
        (PieceKind::King, _) => king_attacks(piece.color, sq),
    }
}

/// The seven hand kinds, in Material's and Mobility's shared 歩香桂銀金角飛
/// order.
const HAND_KINDS: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

/// The occupancy bitboard, rebuilt from the mailbox.
///
/// `Board::occupied` is crate-private to `attic-state`, and the scan is
/// eighty-one reads once per evaluation — next to the drop scan below it does
/// not show up.
fn occupancy(pos: &Position) -> Bitboard {
    let mut occ = Bitboard::empty();
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).expect("index < Square::COUNT is valid");
        if pos.board().get(sq).is_some() {
            occ.set(sq);
        }
    }
    occ
}

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------

/// Material's board slots. The king has none, and と金・成香・成桂・成銀 share
/// one (`mat_board_index` in `evaluate_material.cpp:37-52`).
fn material_board_slot(kind: PieceKind, promoted: bool) -> Option<usize> {
    Some(match (kind, promoted) {
        (PieceKind::Pawn, false) => 0,
        (PieceKind::Lance, false) => 1,
        (PieceKind::Knight, false) => 2,
        (PieceKind::Silver, false) => 3,
        (PieceKind::Gold, _) => 4,
        (PieceKind::Bishop, false) => 5,
        (PieceKind::Rook, false) => 6,
        (PieceKind::Pawn | PieceKind::Lance | PieceKind::Knight | PieceKind::Silver, true) => 7,
        (PieceKind::Bishop, true) => 8,
        (PieceKind::Rook, true) => 9,
        (PieceKind::King, _) => return None,
    })
}

/// Material's 17 counts, from Black's point of view whichever side is to move.
pub fn extract_material(pos: &Position) -> [f32; NUM_MATERIAL] {
    let mut mat = [0.0f32; NUM_MATERIAL];

    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).expect("index < Square::COUNT is valid");
        let Some(piece) = pos.board().get(sq) else {
            continue;
        };
        if let Some(at) = material_board_slot(piece.kind, piece.promoted) {
            mat[at] += sign_of(piece.color);
        }
    }

    for color in [Color::Black, Color::White] {
        let sign = sign_of(color);
        let hand = pos.hand(color);
        for (slot, &kind) in HAND_KINDS.iter().enumerate() {
            mat[10 + slot] += sign * f32::from(hand.count(kind));
        }
    }

    mat
}

#[inline]
fn sign_of(color: Color) -> f32 {
    match color {
        Color::Black => 1.0,
        Color::White => -1.0,
    }
}

// ---------------------------------------------------------------------------
// Mobility
// ---------------------------------------------------------------------------

/// `df` is taken absolute and `dr` is signed, both within a board's reach.
const OFFSET_RANGE: usize = 17;
const OFFSET_BIAS: i32 = 8;

/// `(piece, |df|, dr)` to feature index, or -1. One table for a piece standing
/// on the board and one for a piece dropped onto an empty square.
struct IndexTables {
    board: Vec<i16>,
    hand: Vec<i16>,
}

fn slot(pt: usize, df: i32, dr: i32) -> usize {
    (pt * OFFSET_RANGE + (df.abs() + OFFSET_BIAS) as usize) * OFFSET_RANGE
        + (dr + OFFSET_BIAS) as usize
}

/// Builds the tables in shielune's registration order
/// (`humanlike_eval.cpp:82-194`).
///
/// The order **is** the layout: a fit against a different order produces
/// weights that are wrong in a way no number in the output would show. So it
/// is written out in the same sequence as the C++ rather than derived from the
/// attack patterns, which would renumber silently if the two ever disagreed
/// about one of them.
fn index_tables() -> IndexTables {
    let mut board = vec![-1i16; NUM_PIECE_TYPES * OFFSET_RANGE * OFFSET_RANGE];
    let mut hand = vec![-1i16; NUM_PIECE_TYPES * OFFSET_RANGE * OFFSET_RANGE];
    let mut idx: i16 = 0;

    {
        let mut add = |pt: usize, df: i32, dr: i32| {
            let at = slot(pt, df, dr);
            if board[at] < 0 {
                board[at] = idx;
                idx += 1;
            }
        };
        let pawn = pt_index(PieceKind::Pawn, false);
        let lance = pt_index(PieceKind::Lance, false);
        let knight = pt_index(PieceKind::Knight, false);
        let silver = pt_index(PieceKind::Silver, false);
        let gold = pt_index(PieceKind::Gold, false);
        let king = pt_index(PieceKind::King, false);
        let bishop = pt_index(PieceKind::Bishop, false);
        let rook = pt_index(PieceKind::Rook, false);
        let horse = pt_index(PieceKind::Bishop, true);
        let dragon = pt_index(PieceKind::Rook, true);

        add(pawn, 0, -1);
        for d in 1..=8 {
            add(lance, 0, -d);
        }
        add(knight, 1, -2);
        add(silver, 0, -1);
        add(silver, 1, -1);
        add(silver, 1, 1);
        for (df, dr) in [(0, -1), (1, -1), (1, 0), (0, 1)] {
            add(gold, df, dr);
        }
        add(king, 0, -1);
        add(king, 1, -1);
        add(king, 1, 0);
        add(king, 1, 1);
        add(king, 0, 1);
        for d in 1..=8 {
            add(bishop, d, -d);
            add(bishop, d, d);
        }
        for d in 1..=8 {
            add(rook, 0, -d);
            add(rook, 0, d);
            add(rook, d, 0);
        }
        for d in 1..=8 {
            add(horse, d, -d);
            add(horse, d, d);
        }
        add(horse, 0, -1);
        add(horse, 0, 1);
        add(horse, 1, 0);
        for d in 1..=8 {
            add(dragon, 0, -d);
            add(dragon, 0, d);
            add(dragon, d, 0);
        }
        add(dragon, 1, -1);
        add(dragon, 1, 1);
    }
    assert_eq!(idx as usize, NUM_BOARD_FEATURES, "board feature count");

    // と金・成香・成桂・成銀 have no slots of their own; they share gold's four
    // (`alias_to` in `humanlike_eval.cpp:102-105`). Gold's row holds exactly
    // those four entries, so copying the row is the aliasing.
    let gold = pt_index(PieceKind::Gold, false);
    for kind in [
        PieceKind::Pawn,
        PieceKind::Lance,
        PieceKind::Knight,
        PieceKind::Silver,
    ] {
        let to = pt_index(kind, true);
        for k in 0..OFFSET_RANGE * OFFSET_RANGE {
            board[to * OFFSET_RANGE * OFFSET_RANGE + k] =
                board[gold * OFFSET_RANGE * OFFSET_RANGE + k];
        }
    }

    {
        let mut add = |pt: usize, df: i32, dr: i32| {
            let at = slot(pt, df, dr);
            if hand[at] < 0 {
                hand[at] = idx;
                idx += 1;
            }
        };
        let pawn = pt_index(PieceKind::Pawn, false);
        let lance = pt_index(PieceKind::Lance, false);
        let knight = pt_index(PieceKind::Knight, false);
        let silver = pt_index(PieceKind::Silver, false);
        let gold = pt_index(PieceKind::Gold, false);
        let bishop = pt_index(PieceKind::Bishop, false);
        let rook = pt_index(PieceKind::Rook, false);

        add(pawn, 0, -1);
        for d in 1..=8 {
            add(lance, 0, -d);
        }
        add(knight, 1, -2);
        add(silver, 0, -1);
        add(silver, 1, -1);
        add(silver, 1, 1);
        for (df, dr) in [(0, -1), (1, -1), (1, 0), (0, 1)] {
            add(gold, df, dr);
        }
        for d in 1..=8 {
            add(bishop, d, -d);
            add(bishop, d, d);
        }
        for d in 1..=8 {
            add(rook, 0, -d);
            add(rook, 0, d);
            add(rook, d, 0);
        }
    }
    assert_eq!(idx as usize, NUM_MOBILITY, "total feature count");

    IndexTables { board, hand }
}

static TABLES: LazyLock<IndexTables> = LazyLock::new(index_tables);

/// A pawn or lance may not sit on the last rank, a knight not on the last two.
///
/// Nothing else about the drop is checked, matching
/// `humanlike_eval.cpp:229-236`: no drop-pawn-mate test, and no legality of
/// the resulting position.
#[inline]
fn drop_legal_rank(color: Color, kind: PieceKind, rank: i32) -> bool {
    let rel = match color {
        Color::Black => rank,
        Color::White => 8 - rank,
    };
    match kind {
        PieceKind::Pawn | PieceKind::Lance => rel != 0,
        PieceKind::Knight => rel != 0 && rel != 1,
        _ => true,
    }
}

/// Mobility's 164 counts, from Black's point of view whichever side is to
/// move.
///
/// Black's pieces count positive either way; the caller negates the result
/// instead, which is what `evaluate_mobility.cpp:48-58` does at the other end.
pub fn extract_mobility(pos: &Position) -> [f32; NUM_MOBILITY] {
    let mut mob = [0.0f32; NUM_MOBILITY];
    let tables = &*TABLES;
    let occ = occupancy(pos);

    // Files each side already has a pawn on, so a pawn drop can skip them.
    let mut pawn_files = [[false; 9]; Color::COUNT];

    for sq in occ.squares() {
        let piece = pos.board().get(sq).expect("occupancy came from the board");
        let sign = sign_of(piece.color);

        if piece.kind == PieceKind::Pawn && !piece.promoted {
            pawn_files[piece.color.index()][sq.file() as usize] = true;
        }

        let row = pt_index(piece.kind, piece.promoted) * OFFSET_RANGE * OFFSET_RANGE;
        let (sf, sr) = (i32::from(sq.file()), i32::from(sq.rank()));
        for target in attacks_from(piece, sq, occ).squares() {
            let df = i32::from(target.file()) - sf;
            let mut dr = i32::from(target.rank()) - sr;
            if piece.color == Color::White {
                dr = -dr;
            }
            let at = tables.board[row
                + (df.abs() + OFFSET_BIAS) as usize * OFFSET_RANGE
                + (dr + OFFSET_BIAS) as usize];
            if at >= 0 {
                mob[at as usize] += sign;
            }
        }
    }

    for color in [Color::Black, Color::White] {
        let sign = sign_of(color);
        let pawn_files = &pawn_files[color.index()];
        let hand = pos.hand(color);

        for &kind in &HAND_KINDS {
            let count = hand.count(kind);
            if count == 0 {
                continue;
            }
            let piece = Piece::new(kind, color);
            let row = pt_index(kind, false) * OFFSET_RANGE * OFFSET_RANGE;

            for index in 0..Square::COUNT as u8 {
                let sq = Square::from_index(index).expect("index < Square::COUNT is valid");
                if occ.test(sq) {
                    continue;
                }
                let (sf, sr) = (i32::from(sq.file()), i32::from(sq.rank()));
                if !drop_legal_rank(color, kind, sr) {
                    continue;
                }
                if kind == PieceKind::Pawn && pawn_files[sq.file() as usize] {
                    continue;
                }
                for target in attacks_from(piece, sq, occ).squares() {
                    let df = i32::from(target.file()) - sf;
                    let mut dr = i32::from(target.rank()) - sr;
                    if color == Color::White {
                        dr = -dr;
                    }
                    let at = tables.hand[row
                        + (df.abs() + OFFSET_BIAS) as usize * OFFSET_RANGE
                        + (dr + OFFSET_BIAS) as usize];
                    if at >= 0 {
                        mob[at as usize] += sign * f32::from(count);
                    }
                }
            }
        }
    }

    mob
}

// ---------------------------------------------------------------------------
// The search-facing surface
// ---------------------------------------------------------------------------

/// The compiled-in coefficients, wrapped so the search's `NnueNetwork` name
/// has something to bind to.
///
/// There is nothing to own: the weights are a `const` in the binary. The type
/// exists so that `Search::new(net)`, `replicate` and the rest read the same
/// whichever evaluator is compiled in.
#[derive(Debug, Clone, Copy, Default)]
pub struct HceNetwork;

impl HceNetwork {
    /// The coefficients this shape was built with.
    pub fn weights(&self) -> &'static [f32; DIM] {
        &weights::WEIGHTS
    }

    /// A deep copy, which for a network with no storage is a copy.
    pub fn replicate(&self) -> Self {
        *self
    }
}

/// Loads nothing and succeeds.
///
/// The weights are compiled in, so there is no `EvalDir` to honour. It still
/// returns `Ok` rather than being removed because the driver refuses to answer
/// `readyok` unless a load succeeded (`isready_heavy_job` in `attic-protocol`),
/// and a shape that carries no file should not have to change that contract.
pub fn load_network(_path: &Path) -> Result<HceNetwork, NnueError> {
    Ok(HceNetwork)
}

/// As [`load_network`], with the empty warning list the driver prints before
/// `readyok`.
pub fn load_network_with_warnings(path: &Path) -> Result<(HceNetwork, Vec<String>), NnueError> {
    Ok((load_network(path)?, Vec::new()))
}

/// The dot product, from the side to move's point of view.
///
/// Summed in `f64` and truncated toward zero, as both C++ engines do, so that
/// the two agree on the last unit rather than nearly agreeing.
pub fn evaluate(net: &HceNetwork, pos: &Position) -> i32 {
    #[cfg(feature = "hce_material_17")]
    let (features, clamp) = (extract_material(pos), None::<f64>);
    #[cfg(feature = "hce_mobility_107_57")]
    let (features, clamp) = (extract_mobility(pos), Some(MOBILITY_CLAMP));

    let mut sum = 0.0f64;
    for (w, f) in net.weights().iter().zip(features.iter()) {
        sum += f64::from(*w) * f64::from(*f);
    }
    if let Some(limit) = clamp {
        sum = sum.clamp(-limit, limit);
    }
    let score = sum as i32;
    match pos.side_to_move() {
        Color::Black => score,
        Color::White => -score,
    }
}

/// Static evaluation from an already-built accumulator.
///
/// There is no accumulator to read: this is [`evaluate`], and the argument is
/// here because the search's seam has it.
pub fn evaluate_with(net: &HceNetwork, _acc: &Accumulator, pos: &Position) -> i32 {
    evaluate(net, pos)
}

/// What NNUE carries down the tree and this carries nothing in.
///
/// A move changes the attack map far beyond the squares it touches, so there
/// is no small delta to accumulate and nothing worth caching between plies.
/// The type stays because the search declares a stack of them.
#[derive(Debug, Clone, Copy, Default)]
pub struct Accumulator;

impl Accumulator {
    pub fn new() -> Self {
        Accumulator
    }

    /// Rebuilds nothing.
    pub fn refresh(&mut self, _net: &HceNetwork, _pos: &Position) {}

    /// The accumulator for the position after `mv`, which is the same nothing.
    ///
    /// `pos` is left byte-for-byte unchanged, as the NNUE and Manaka forms
    /// promise: here that is easy, because it is never touched.
    pub fn update_after_move(
        &self,
        _net: &HceNetwork,
        _pos: &mut Position,
        _mv: Move,
    ) -> Accumulator {
        Accumulator
    }

    /// Moves `dst` to `post_pos`, which takes no work.
    pub fn derive_into_cached(
        _src: &Accumulator,
        _dst: &mut Accumulator,
        _net: &HceNetwork,
        _post_pos: &Position,
        _delta: &MoveDelta,
        _cache: &mut FinnyCache,
    ) {
    }
}

/// Where NNUE keeps its king-bucket refresh cache. There is no refresh to
/// cache; the search allocates one per worker and hands it to every derive, so
/// the name stays and the storage does not.
#[derive(Debug, Clone, Copy, Default)]
pub struct FinnyCache;

impl FinnyCache {
    pub fn new() -> Box<Self> {
        Box::new(FinnyCache)
    }
}

/// The add/sub feature delta a move induces, for a shape that has none.
#[derive(Debug, Clone, Copy, Default)]
pub struct PerspectiveDelta;

impl PerspectiveDelta {
    pub fn removed(&self) -> &[u32] {
        &[]
    }

    pub fn added(&self) -> &[u32] {
        &[]
    }
}

/// As [`PerspectiveDelta`], for both sides at once.
#[derive(Debug, Clone, Copy, Default)]
pub struct MoveDelta;

impl MoveDelta {
    /// Always `None`: every perspective is "refreshed", and a refresh is free.
    pub fn half(&self, _perspective: Color) -> Option<&PerspectiveDelta> {
        None
    }

    /// Reads neither the position nor the move.
    pub fn from_move(_pos: &Position, _mv: Move) -> MoveDelta {
        MoveDelta
    }
}
