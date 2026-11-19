//! The threat features of Sirius's network (`ThreatProfile 23`): which piece attacks which,
//! numbered into the 218,016 rows of the int8 threat block, plus the king-count rows.
//!
//! ## What is a feature
//!
//! An edge is an attacker (any piece but a king) and a target (any piece but a king) the
//! attacker reaches on the board as it stands: a step piece by its steps, a lance, bishop,
//! rook, horse or dragon along its rays up to and including the first piece in the way. For
//! each perspective an edge is active unless attacker and target both belong to the other
//! side. The edge's row depends on the attacker's and the target's kind and owner and on the
//! two squares *as the perspective sees the board*: turned half a revolution for White, and
//! then mirrored left to right when the perspective's own king stands on files 6 to 9 (the
//! rule `HalfKA_hm2` uses for the pieces).
//!
//! ```text
//! row = block_start[as][ac][ds][dc] + from_offset[pattern][s'] + order[pattern][s'][t']
//! ```
//!
//! `as` and `ds` say whether the attacker and the target belong to the other side, `ac` and
//! `dc` are their classes ([`class_of`]), `pattern` is `ac + 9 * as` (the squares a piece of
//! that class reaches on an empty board, moving toward rank one for the perspective's own
//! pieces and toward rank nine for the other side's), and `s'`, `t'` are the squares in the
//! perspective's frame. `from_offset` counts the empty-board moves of the squares before `s'`
//! and `order` ranks `t'` among the moves from `s'` in ascending square order. There are 36
//! blocks of 6,020 rows.
//!
//! Sirius stores its rows in an order of its own (even slots for Black's attackers, odd for
//! White's counted down, and an exclusive-or with the perspective), and a permutation moves
//! the rows of the file into it as it loads. This module numbers an edge by the file's
//! order directly, which `SPEC.md` shows to be the same thing, so nothing is permuted.
//!
//! ## The cap
//!
//! A perspective's list holds at most [`MAX_EDGES`] edges, taken in the order below. The two
//! king-count rows are added after the cap. Which edges survive when a position has more is
//! part of the network's behaviour, and depends on whether the two perspectives mirror the
//! same way:
//!
//! * They differ: each list takes its own first 128 active edges.
//! * They agree: one pass serves both, and an edge is taken only while *both* lists have room,
//!   so both stop growing when the first of them is full.
//!
//! The order is ascending attacker square, then ascending target square, in the board's own
//! coordinates (not the perspective's). A list therefore always has strictly increasing
//! `attacker * 81 + target` keys, which is what makes two lists cheap to compare.
//!
//! ## King-count rows
//!
//! 1,296 more rows: for each perspective one for its own king's square and one for the other
//! king's, each paired with a count bucket (0 to 7) of how many pieces that side has in hand
//! and in the enemy camp.

use std::sync::OnceLock;

use attic_state::bitboard::{
    bishop_attacks, dragon_attacks, gold_attacks, horse_attacks, knight_attacks, lance_attacks,
    pawn_attacks, promotion_zone, rook_attacks, silver_attacks,
};
use attic_state::{Bitboard, Color, Piece, PieceKind, Position, Square};

/// Rows that stand for an attacker and a target: 36 blocks of [`ATTACKS_PER_CLASS`]'s total.
pub const EDGE_ROWS: usize = 216_720;
/// Rows of the king-square-by-count-bucket features.
pub const KING_COUNT_ROWS: usize = 1_296;
/// Edges a perspective's list holds at most; the king rows come on top.
pub const MAX_EDGES: usize = 128;
/// The longest a list can be: the cap and the two king rows.
pub const MAX_LIST: usize = MAX_EDGES + 2;
/// The most rows a difference of two lists can name on either side: every row of the old
/// list going out, or every row of the new one coming in.
pub const MAX_CHANGES: usize = MAX_LIST;

/// First row of the "own king" king-count features; the "other king" ones follow 648 later.
const KING_ROWS_OWN: u32 = EDGE_ROWS as u32;
const KING_ROWS_OTHER: u32 = (EDGE_ROWS + KING_COUNT_ROWS / 2) as u32;

/// Moves a piece of each class makes on an empty board, summed over the 81 squares (a piece
/// moving toward rank one; the other direction has as many).
const ATTACKS_PER_CLASS: [usize; 9] = [72, 324, 112, 328, 416, 816, 1296, 1104, 1552];

/// Bits of a list entry that hold the row; the key is above them.
const ROW_BITS: u32 = 18;
const ROW_MASK: u32 = (1 << ROW_BITS) - 1;

/// The class of a piece that can attack or be attacked (anything but a king): pawn 0, lance
/// 1, knight 2, silver 3, gold and the four small promotions 4, bishop 5, rook 6, horse 7,
/// dragon 8.
#[inline]
fn class_of(piece: Piece) -> usize {
    match (piece.kind, piece.promoted) {
        (PieceKind::Pawn, false) => 0,
        (PieceKind::Lance, false) => 1,
        (PieceKind::Knight, false) => 2,
        (PieceKind::Silver, false) => 3,
        (PieceKind::Gold, _)
        | (PieceKind::Pawn | PieceKind::Lance | PieceKind::Knight | PieceKind::Silver, true) => 4,
        (PieceKind::Bishop, false) => 5,
        (PieceKind::Rook, false) => 6,
        (PieceKind::Bishop, true) => 7,
        (PieceKind::Rook, true) => 8,
        (PieceKind::King, _) => unreachable!("a king is neither an attacker nor a target"),
    }
}

/// The squares `piece` on `sq` attacks with `occ` on the board, blockers included.
#[inline]
fn attacks_of(piece: Piece, sq: Square, occ: Bitboard) -> Bitboard {
    let c = piece.color;
    match (piece.kind, piece.promoted) {
        (PieceKind::Pawn, false) => pawn_attacks(c, sq),
        (PieceKind::Lance, false) => lance_attacks(c, sq, occ),
        (PieceKind::Knight, false) => knight_attacks(c, sq),
        (PieceKind::Silver, false) => silver_attacks(c, sq),
        (PieceKind::Gold, _)
        | (PieceKind::Pawn | PieceKind::Lance | PieceKind::Knight | PieceKind::Silver, true) => {
            gold_attacks(c, sq)
        }
        (PieceKind::Bishop, false) => bishop_attacks(sq, occ),
        (PieceKind::Rook, false) => rook_attacks(sq, occ),
        (PieceKind::Bishop, true) => horse_attacks(sq, occ),
        (PieceKind::Rook, true) => dragon_attacks(sq, occ),
        (PieceKind::King, _) => Bitboard::EMPTY,
    }
}

/// `mir(sq)`: file `f` to `8 - f`, rank unchanged.
#[inline]
const fn mirror(sq: usize) -> usize {
    (8 - sq / 9) * 9 + sq % 9
}

/// The numbering tables, built once.
struct Tables {
    /// `pattern_row[pattern * 81 * 81 + from * 81 + to]`: `from_offset + order` for a move
    /// of `pattern` (= `class + 9 * as`) from `from` to `to` in the perspective's frame. The
    /// entries for moves the pattern does not make are never read.
    pattern_row: Box<[u16]>,
    /// `block_start[as][ac][ds][dc]`.
    block_start: [[[[u32; 9]; 2]; 9]; 2],
    /// `frame[perspective][mirrored][sq]`: the square as the perspective sees it.
    frame: [[[u8; 81]; 2]; 2],
}

/// The squares a piece of `class` moving toward rank one (`toward_nine = false`) or toward
/// rank nine reaches from `from` on an empty board, ascending.
fn empty_board_targets(class: usize, toward_nine: bool, from: usize) -> Vec<usize> {
    let (file, rank) = ((from / 9) as i32, (from % 9) as i32);
    let fwd = if toward_nine { 1 } else { -1 };
    let mut steps: Vec<(i32, i32)> = Vec::new();
    let mut rays: Vec<(i32, i32)> = Vec::new();
    match class {
        0 => steps.push((0, fwd)),
        1 => rays.push((0, fwd)),
        2 => steps.extend([(-1, 2 * fwd), (1, 2 * fwd)]),
        3 => steps.extend([(-1, fwd), (0, fwd), (1, fwd), (-1, -fwd), (1, -fwd)]),
        4 => steps.extend([(-1, fwd), (0, fwd), (1, fwd), (-1, 0), (1, 0), (0, -fwd)]),
        5 => rays.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)]),
        6 => rays.extend([(-1, 0), (1, 0), (0, -1), (0, 1)]),
        7 => {
            rays.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)]);
            steps.extend([(-1, 0), (1, 0), (0, -1), (0, 1)]);
        }
        8 => {
            rays.extend([(-1, 0), (1, 0), (0, -1), (0, 1)]);
            steps.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)]);
        }
        _ => unreachable!("nine classes"),
    }
    let mut out = Vec::new();
    for (df, dr) in steps {
        let (f, r) = (file + df, rank + dr);
        if (0..9).contains(&f) && (0..9).contains(&r) {
            out.push((f * 9 + r) as usize);
        }
    }
    for (df, dr) in rays {
        let (mut f, mut r) = (file + df, rank + dr);
        while (0..9).contains(&f) && (0..9).contains(&r) {
            out.push((f * 9 + r) as usize);
            f += df;
            r += dr;
        }
    }
    out.sort_unstable();
    out
}

impl Tables {
    fn build() -> Tables {
        let mut pattern_row = vec![u16::MAX; 18 * 81 * 81].into_boxed_slice();
        for pattern in 0..18 {
            let (class, toward_nine) = (pattern % 9, pattern >= 9);
            let mut offset = 0usize;
            for from in 0..81 {
                let targets = empty_board_targets(class, toward_nine, from);
                for (order, &to) in targets.iter().enumerate() {
                    pattern_row[(pattern * 81 + from) * 81 + to] = (offset + order) as u16;
                }
                offset += targets.len();
            }
            assert_eq!(
                offset, ATTACKS_PER_CLASS[class],
                "pattern {pattern} makes {offset} moves on an empty board"
            );
        }

        // Blocks run as, ac, ds, dc (each `as`/`ds` 0 then 1, each class 0 to 8), each as
        // long as the attacker's class makes moves on an empty board.
        let mut block_start = [[[[0u32; 9]; 2]; 9]; 2];
        let mut next = 0usize;
        for per_side in block_start.iter_mut() {
            for (ac, per_class) in per_side.iter_mut().enumerate() {
                for per_target_side in per_class.iter_mut() {
                    for slot in per_target_side.iter_mut() {
                        *slot = next as u32;
                        next += ATTACKS_PER_CLASS[ac];
                    }
                }
            }
        }
        assert_eq!(next, EDGE_ROWS);

        let mut frame = [[[0u8; 81]; 2]; 2];
        for (perspective, per_perspective) in frame.iter_mut().enumerate() {
            for (mirrored, table) in per_perspective.iter_mut().enumerate() {
                for (sq, slot) in table.iter_mut().enumerate() {
                    let turned = if perspective == 0 { sq } else { 80 - sq };
                    *slot = if mirrored == 1 {
                        mirror(turned)
                    } else {
                        turned
                    } as u8;
                }
            }
        }

        Tables {
            pattern_row,
            block_start,
            frame,
        }
    }
}

fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(Tables::build)
}

/// One perspective's active threat rows: the capped edges and the two king-count rows.
///
/// Plain data, `Copy`, a few hundred bytes: an accumulator keeps one per perspective, the
/// lists its sums were built from, so that the next position's list can be compared with it.
#[derive(Clone, Copy, Debug)]
pub struct ThreatList {
    /// `(attacker * 81 + target) << ROW_BITS | row`, strictly ascending; only the first
    /// `len` are live.
    edges: [u32; MAX_EDGES],
    len: u32,
    king: [u32; 2],
}

/// Two lists are equal when they name the same rows in the same order; what lies in the
/// unused part of `edges` does not count.
impl PartialEq for ThreatList {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
            && self.king == other.king
            && self.edges[..self.len as usize] == other.edges[..other.len as usize]
    }
}

impl Eq for ThreatList {}

impl ThreatList {
    /// A list with no edges and both king rows at row zero. Only a placeholder until a
    /// position has filled it in.
    pub const EMPTY: ThreatList = ThreatList {
        edges: [0; MAX_EDGES],
        len: 0,
        king: [0; 2],
    };

    /// How many rows the list names: its edges and the two king rows.
    #[inline]
    pub fn row_count(&self) -> usize {
        self.len as usize + 2
    }

    /// How many of the rows are edges (the king rows are the other two).
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.len as usize
    }

    /// The rows, edges first (in enumeration order) and then the king rows, written to the
    /// front of `out`; returns how many.
    #[inline]
    pub fn rows_into(&self, out: &mut [u32; MAX_LIST]) -> usize {
        let n = self.len as usize;
        for (slot, &entry) in out.iter_mut().zip(&self.edges[..n]) {
            *slot = entry & ROW_MASK;
        }
        out[n] = self.king[0];
        out[n + 1] = self.king[1];
        n + 2
    }

    /// The edges in enumeration order as `(attacker, target, row)`, the squares being board
    /// indices (`file * 9 + rank`). For tests and tools.
    pub fn edges(&self) -> impl Iterator<Item = (usize, usize, u32)> + '_ {
        self.edges[..self.len as usize].iter().map(|&entry| {
            let key = (entry >> ROW_BITS) as usize;
            (key / 81, key % 81, entry & ROW_MASK)
        })
    }

    /// The two king-count rows: the perspective's own king's, then the other king's.
    pub fn king_rows(&self) -> [u32; 2] {
        self.king
    }

    /// The rows, as a vector (for tests and tools; the search uses [`Self::rows_into`]).
    pub fn rows(&self) -> Vec<u32> {
        let mut buf = [0u32; MAX_LIST];
        let n = self.rows_into(&mut buf);
        buf[..n].to_vec()
    }
}

/// The rows to take away and to put in to turn the sums of one list into those of another.
pub struct ThreatDiff {
    removed: [u32; MAX_CHANGES],
    added: [u32; MAX_CHANGES],
    n_removed: usize,
    n_added: usize,
}

impl ThreatDiff {
    /// An empty difference, to be filled by [`ThreatDiff::between`].
    #[inline]
    pub fn new() -> ThreatDiff {
        ThreatDiff {
            removed: [0; MAX_CHANGES],
            added: [0; MAX_CHANGES],
            n_removed: 0,
            n_added: 0,
        }
    }

    /// Rows in `old` and not in `new`.
    #[inline]
    pub fn removed(&self) -> &[u32] {
        &self.removed[..self.n_removed]
    }

    /// Rows in `new` and not in `old`.
    #[inline]
    pub fn added(&self) -> &[u32] {
        &self.added[..self.n_added]
    }

    /// Fill `self` with the difference from `old` to `new`.
    ///
    /// Both lists run in ascending edge key, so they are walked together. An edge present in
    /// both under the same row costs nothing; under a different row (the attacker promoted,
    /// or the target changed) the old row goes out and the new one comes in. The king rows
    /// are compared position by position.
    #[inline]
    pub fn between(&mut self, old: &ThreatList, new: &ThreatList) {
        let (mut i, mut j) = (0usize, 0usize);
        let (na, nb) = (old.len as usize, new.len as usize);
        let (mut r, mut a) = (0usize, 0usize);
        while i < na && j < nb {
            let (x, y) = (old.edges[i], new.edges[j]);
            let (kx, ky) = (x >> ROW_BITS, y >> ROW_BITS);
            if kx == ky {
                if x != y {
                    self.removed[r] = x & ROW_MASK;
                    r += 1;
                    self.added[a] = y & ROW_MASK;
                    a += 1;
                }
                i += 1;
                j += 1;
            } else if kx < ky {
                self.removed[r] = x & ROW_MASK;
                r += 1;
                i += 1;
            } else {
                self.added[a] = y & ROW_MASK;
                a += 1;
                j += 1;
            }
        }
        while i < na {
            self.removed[r] = old.edges[i] & ROW_MASK;
            r += 1;
            i += 1;
        }
        while j < nb {
            self.added[a] = new.edges[j] & ROW_MASK;
            a += 1;
            j += 1;
        }
        for k in 0..2 {
            if old.king[k] != new.king[k] {
                self.removed[r] = old.king[k];
                r += 1;
                self.added[a] = new.king[k];
                a += 1;
            }
        }
        self.n_removed = r;
        self.n_added = a;
    }
}

impl Default for ThreatDiff {
    fn default() -> Self {
        Self::new()
    }
}

/// How many pieces `color` has in hand and in the enemy camp, weighted: the small ones in
/// hand 1, bishops and rooks in hand 5; the pieces in the camp 1, bishops and rooks (and
/// their promotions) 5. The count bucket is a quarter of that, at most 7.
fn king_count_bucket(pos: &Position, color: Color) -> u32 {
    let hand = pos.hand(color);
    let mut count = 0u32;
    for kind in [
        PieceKind::Pawn,
        PieceKind::Lance,
        PieceKind::Knight,
        PieceKind::Silver,
        PieceKind::Gold,
    ] {
        count += hand.count(kind) as u32;
    }
    count += 5 * (hand.count(PieceKind::Bishop) as u32 + hand.count(PieceKind::Rook) as u32);
    let board = pos.board();
    for sq in (board.occupied() & promotion_zone(color)).squares() {
        let Some(piece) = board.get(sq) else { continue };
        if piece.color != color || piece.kind == PieceKind::King {
            continue;
        }
        count += if matches!(piece.kind, PieceKind::Bishop | PieceKind::Rook) {
            5
        } else {
            1
        };
    }
    (count >> 2).min(7)
}

/// Both perspectives' threat lists for `pos`, Black's first.
///
/// # Panics
/// Panics if `pos` is missing either king.
pub fn build_lists(pos: &Position) -> [ThreatList; 2] {
    let t = tables();
    let board = pos.board();
    let occ = board.occupied();
    let black_king = pos
        .king_square(Color::Black)
        .unwrap_or_else(|| panic!("position has no Black king"));
    let white_king = pos
        .king_square(Color::White)
        .unwrap_or_else(|| panic!("position has no White king"));
    let targets =
        occ.and_not(Bitboard::from_square(black_king) | Bitboard::from_square(white_king));

    // A perspective mirrors left to right when its own king stands on files 6 to 9 as it
    // sees the board: file index 5 or more for Black, 3 or less for White.
    let mirrored = [
        usize::from(black_king.file() >= 5),
        usize::from(white_king.file() <= 3),
    ];
    let separate = mirrored[0] != mirrored[1];
    let frame = [&t.frame[0][mirrored[0]], &t.frame[1][mirrored[1]]];

    let mut lists = [ThreatList::EMPTY; 2];
    let mut n = [0usize; 2];

    'attackers: for from in occ.squares() {
        let Some(attacker) = board.get(from) else {
            continue;
        };
        if attacker.kind == PieceKind::King {
            continue;
        }
        let a_color = attacker.color.index();
        let ac = class_of(attacker);
        let from_ix = from.index() as usize;
        for to in (attacks_of(attacker, from, occ) & targets).squares() {
            let target = board
                .get(to)
                .expect("a target square holds a piece by construction");
            let t_color = target.color.index();
            let dc = class_of(target);
            let to_ix = to.index() as usize;
            let key = ((from_ix * 81 + to_ix) as u32) << ROW_BITS;

            // Both lists share one pass when they mirror alike: an edge is then taken only
            // while both have room, and once either is full nothing more is taken.
            if !separate && (n[0] >= MAX_EDGES || n[1] >= MAX_EDGES) {
                break 'attackers;
            }
            for p in 0..2 {
                let a_side = usize::from(a_color != p);
                let d_side = usize::from(t_color != p);
                // The other side's pieces attacking each other are not features.
                if a_side == 1 && d_side == 1 {
                    continue;
                }
                if n[p] >= MAX_EDGES {
                    continue;
                }
                let row = edge_row(t, frame[p], a_side, ac, d_side, dc, from_ix, to_ix);
                lists[p].edges[n[p]] = key | row;
                n[p] += 1;
            }
            if separate && n[0] >= MAX_EDGES && n[1] >= MAX_EDGES {
                break 'attackers;
            }
        }
    }

    [lists[0].king, lists[1].king] = king_rows(pos, frame, black_king, white_king);
    lists[0].len = n[0] as u32;
    lists[1].len = n[1] as u32;
    lists
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn edge_row(
    t: &Tables,
    frame: &[u8; 81],
    a_side: usize,
    ac: usize,
    d_side: usize,
    dc: usize,
    from_ix: usize,
    to_ix: usize,
) -> u32 {
    let row = t.block_start[a_side][ac][d_side][dc]
        + t.pattern_row
            [((ac + 9 * a_side) * 81 + frame[from_ix] as usize) * 81 + frame[to_ix] as usize]
            as u32;
    debug_assert!((row as usize) < EDGE_ROWS);
    row
}

#[inline]
fn king_rows(
    pos: &Position,
    frame: [&[u8; 81]; 2],
    black_king: Square,
    white_king: Square,
) -> [[u32; 2]; 2] {
    let bucket = [
        king_count_bucket(pos, Color::Black),
        king_count_bucket(pos, Color::White),
    ];
    let (bk, wk) = (black_king.index() as usize, white_king.index() as usize);
    [
        [
            KING_ROWS_OWN + bucket[0] * 81 + frame[0][bk] as u32,
            KING_ROWS_OTHER + bucket[1] * 81 + frame[0][wk] as u32,
        ],
        [
            KING_ROWS_OWN + bucket[1] * 81 + frame[1][wk] as u32,
            KING_ROWS_OTHER + bucket[0] * 81 + frame[1][bk] as u32,
        ],
    ]
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BoardChange {
    pub changed: Bitboard,
    /// `changed` squares that went from empty to occupied or back.
    pub flipped: Bitboard,
}

/// `build_lists(pos)` from the parent's lists, for a move that moved neither king.
/// Returns `false` at the `MAX_EDGES` cap; the caller then uses `build_lists`.
pub fn update_lists(
    pos: &Position,
    old: &[ThreatList; 2],
    change: BoardChange,
    new: &mut [ThreatList; 2],
    diff: &mut [ThreatDiff; 2],
) -> bool {
    if old[0].len as usize >= MAX_EDGES || old[1].len as usize >= MAX_EDGES {
        return false;
    }
    let t = tables();
    let board = pos.board();
    let occ = board.occupied();
    let occ_before = occ ^ change.flipped;
    let black_king = pos
        .king_square(Color::Black)
        .unwrap_or_else(|| panic!("position has no Black king"));
    let white_king = pos
        .king_square(Color::White)
        .unwrap_or_else(|| panic!("position has no White king"));
    let targets =
        occ.and_not(Bitboard::from_square(black_king) | Bitboard::from_square(white_king));

    let mut near = Bitboard::EMPTY;
    for sq in change.changed.squares() {
        near |= rook_attacks(sq, occ)
            | bishop_attacks(sq, occ)
            | rook_attacks(sq, occ_before)
            | bishop_attacks(sq, occ_before)
            | knight_attacks(Color::Black, sq)
            | knight_attacks(Color::White, sq);
    }
    let mut touched = change.changed;
    for sq in (near & targets).and_not(change.changed).squares() {
        let piece = board
            .get(sq)
            .expect("an occupied square holds a piece by construction");
        let reach = attacks_of(piece, sq, occ) | attacks_of(piece, sq, occ_before);
        if !(reach & change.changed).is_empty() {
            touched |= Bitboard::from_square(sq);
        }
    }
    let mut touched_ix = 0u128;
    for sq in touched.squares() {
        touched_ix |= 1u128 << sq.index();
    }

    let mirrored = [
        usize::from(black_king.file() >= 5),
        usize::from(white_king.file() <= 3),
    ];
    let frame = [&t.frame[0][mirrored[0]], &t.frame[1][mirrored[1]]];

    let mut fresh = [ThreatList::EMPTY; 2];
    let mut nf = [0usize; 2];
    for from in touched.squares() {
        let Some(attacker) = board.get(from) else {
            continue;
        };
        if attacker.kind == PieceKind::King {
            continue;
        }
        let a_color = attacker.color.index();
        let ac = class_of(attacker);
        let from_ix = from.index() as usize;
        for to in (attacks_of(attacker, from, occ) & targets).squares() {
            let target = board
                .get(to)
                .expect("a target square holds a piece by construction");
            let t_color = target.color.index();
            let dc = class_of(target);
            let to_ix = to.index() as usize;
            let key = ((from_ix * 81 + to_ix) as u32) << ROW_BITS;
            for p in 0..2 {
                let a_side = usize::from(a_color != p);
                let d_side = usize::from(t_color != p);
                if a_side == 1 && d_side == 1 {
                    continue;
                }
                if nf[p] >= MAX_EDGES {
                    return false;
                }
                fresh[p].edges[nf[p]] =
                    key | edge_row(t, frame[p], a_side, ac, d_side, dc, from_ix, to_ix);
                nf[p] += 1;
            }
        }
    }

    let king = king_rows(pos, frame, black_king, white_king);
    for p in 0..2 {
        let o = &old[p];
        let f = &mut fresh[p];
        let out = &mut new[p];
        let mut stale = ThreatList::EMPTY;
        let (mut i, mut j, mut n, mut ns) = (0usize, 0usize, 0usize, 0usize);
        let (no, nfp) = (o.len as usize, nf[p]);
        while i < no {
            let e = o.edges[i];
            i += 1;
            if (touched_ix >> ((e >> ROW_BITS) / 81)) & 1 != 0 {
                stale.edges[ns] = e;
                ns += 1;
                continue;
            }
            while j < nfp && f.edges[j] < e {
                if n + 1 >= MAX_EDGES {
                    return false;
                }
                out.edges[n] = f.edges[j];
                n += 1;
                j += 1;
            }
            if n + 1 >= MAX_EDGES {
                return false;
            }
            out.edges[n] = e;
            n += 1;
        }
        while j < nfp {
            if n + 1 >= MAX_EDGES {
                return false;
            }
            out.edges[n] = f.edges[j];
            n += 1;
            j += 1;
        }
        out.len = n as u32;
        out.king = king[p];
        stale.len = ns as u32;
        stale.king = o.king;
        f.len = nfp as u32;
        f.king = king[p];
        diff[p].between(&stale, f);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use attic_state::parse_sfen;

    const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

    #[test]
    fn every_class_makes_the_documented_number_of_moves() {
        // `Tables::build` asserts this for all 18 patterns; building them is the test.
        let t = Tables::build();
        assert_eq!(
            t.block_start[1][8][1][8] as usize + ATTACKS_PER_CLASS[8],
            EDGE_ROWS
        );
        assert_eq!(ATTACKS_PER_CLASS.iter().sum::<usize>() * 36, EDGE_ROWS);
    }

    #[test]
    fn pattern_rows_are_a_bijection_onto_the_class_block() {
        let t = tables();
        for pattern in 0..18 {
            let class = pattern % 9;
            let mut seen = vec![false; ATTACKS_PER_CLASS[class]];
            for from in 0..81 {
                for to in 0..81 {
                    let v = t.pattern_row[(pattern * 81 + from) * 81 + to];
                    if v != u16::MAX {
                        assert!(!std::mem::replace(&mut seen[v as usize], true));
                    }
                }
            }
            assert!(seen.iter().all(|&s| s), "pattern {pattern}");
        }
    }

    #[test]
    fn frames_turn_for_white_and_mirror_on_request() {
        let t = tables();
        assert_eq!(t.frame[0][0][10], 10);
        assert_eq!(t.frame[1][0][10], 70);
        assert_eq!(t.frame[0][1][10], mirror(10) as u8);
        assert_eq!(t.frame[1][1][10], mirror(70) as u8);
        assert_eq!(mirror(0), 72);
        assert_eq!(mirror(80), 8);
    }

    #[test]
    fn the_empty_board_move_sets_are_the_shogi_ones() {
        // A lance on the corner square, toward rank one: nothing; toward rank nine: the file.
        assert!(empty_board_targets(1, false, 0).is_empty());
        assert_eq!(empty_board_targets(1, true, 0), (1..9).collect::<Vec<_>>());
        // A knight on the middle file and rank five jumps two ranks, one file either side.
        assert_eq!(
            empty_board_targets(2, false, 4 * 9 + 4),
            vec![3 * 9 + 2, 5 * 9 + 2]
        );
        // A dragon on the centre reaches 16 + 4 squares.
        assert_eq!(empty_board_targets(8, false, 4 * 9 + 4).len(), 20);
    }

    #[test]
    fn startpos_edges_are_found_in_ascending_key_order() {
        let pos = parse_sfen(STARTPOS).unwrap();
        let lists = build_lists(&pos);
        for list in &lists {
            let keys: Vec<u32> = list.edges[..list.len as usize]
                .iter()
                .map(|e| e >> ROW_BITS)
                .collect();
            assert!(keys.windows(2).all(|w| w[0] < w[1]), "{keys:?}");
            assert!(list.edge_count() > 0);
            // No row is repeated.
            let mut rows = list.rows();
            rows.sort_unstable();
            assert!(rows.windows(2).all(|w| w[0] != w[1]));
        }
    }

    #[test]
    fn startpos_is_the_same_for_both_sides_by_symmetry() {
        // The start position is symmetric under the turn that maps Black to White, so each
        // perspective sees the same set of rows.
        let pos = parse_sfen(STARTPOS).unwrap();
        let lists = build_lists(&pos);
        let mut a = lists[0].rows();
        let mut b = lists[1].rows();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }

    #[test]
    fn opponent_pieces_attacking_each_other_are_not_features() {
        // Two White pawns in a file, the front one attacked by the rear? No: a white pawn
        // attacks toward rank nine, so put them so that one attacks the other. Black has
        // nothing but its king. Black's list has no edge; White's has the one.
        let pos = parse_sfen("4k4/9/9/9/4p4/4p4/9/9/4K4 b - 1").unwrap();
        let lists = build_lists(&pos);
        assert_eq!(
            lists[0].edge_count(),
            0,
            "Black does not see White's pawn on White's pawn"
        );
        assert_eq!(lists[1].edge_count(), 1, "White sees its own attack");
    }

    #[test]
    fn a_king_is_neither_attacker_nor_target() {
        let pos = parse_sfen("4k4/4P4/9/9/9/9/9/4p4/4K4 b - 1").unwrap();
        let lists = build_lists(&pos);
        // The pawn on 5b attacks the king on 5a, and the pawn on 5h attacks the king on 5i;
        // neither is an edge.
        assert_eq!(lists[0].edge_count(), 0);
        assert_eq!(lists[1].edge_count(), 0);
    }

    #[test]
    fn a_slider_stops_at_the_first_piece_and_attacks_it() {
        // A Black rook on the file with a White pawn then a White gold behind it.
        let pos = parse_sfen("4k4/9/9/4g4/4p4/9/9/4R4/4K4 b - 1").unwrap();
        let lists = build_lists(&pos);
        // Rook (5h) attacks the pawn (5e) and not the gold (5d) behind it.
        // The pawn attacks toward rank nine: the square 5f; empty. The gold attacks 5e: the
        // pawn, its own side's piece, which Black does not see (both White).
        // Black sees: rook -> pawn. White sees: rook -> pawn, gold -> pawn.
        assert_eq!(lists[0].edge_count(), 1);
        assert_eq!(lists[1].edge_count(), 2);
    }

    fn diff_applies(old: &ThreatList, new: &ThreatList) {
        let mut d = ThreatDiff::new();
        d.between(old, new);
        let mut rows: Vec<u32> = old.rows();
        for r in d.removed() {
            let at = rows
                .iter()
                .position(|x| x == r)
                .expect("removed row was present");
            rows.swap_remove(at);
        }
        rows.extend_from_slice(d.added());
        rows.sort_unstable();
        let mut want = new.rows();
        want.sort_unstable();
        assert_eq!(rows, want);
    }

    #[test]
    fn a_diff_turns_one_list_into_the_other() {
        let sfens = [
            STARTPOS,
            "4k4/9/9/4g4/4p4/9/9/4R4/4K4 b - 1",
            "4k4/9/9/9/4p4/4p4/9/9/4K4 b - 1",
            "lnsgkgsnl/1r5b1/pppp1pppp/4p4/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 1",
        ];
        let lists: Vec<[ThreatList; 2]> = sfens
            .iter()
            .map(|s| build_lists(&parse_sfen(s).unwrap()))
            .collect();
        for a in &lists {
            for b in &lists {
                for p in 0..2 {
                    diff_applies(&a[p], &b[p]);
                }
            }
        }
    }
}
