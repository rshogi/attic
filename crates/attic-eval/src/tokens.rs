//! Manaka's own token space: what the network is told about a position when
//! nothing is crossed with anything.
//!
//! A position becomes a multiset of *tokens*, in each of two frames, and a token
//! is a row of the embedding table. A legal position has forty of them, kings
//! included, and every one of the forty has a distinct row in a given frame.
//!
//! * A **board token** is a piece standing on a square: kind (fourteen, the four
//!   minor promotions and the two major ones kept apart), whose it is relative
//!   to the frame, and the square.
//! * A **hand token** is the *n*-th copy of a kind in someone's hand. Hands are
//!   unary: three pawns held are the tokens for copies 0, 1 and 2, so a capture
//!   adds exactly one token and never moves a count from one bucket to another.
//!
//! The *frame* is the perspective. `f(Black)` reads the board as it is; `f(White)`
//! turns it by 180° and flips ownership, so a position and its mirror produce the
//! same rows. Which frame the head reads first is the only place the side to move
//! enters the network; the token multisets themselves never change when the turn
//! passes.
//!
//! This is the counterpart of [`crate::features`] for the shapes that do not
//! borrow YaneuraOu's numbering. It is a module of its own, and not a third
//! branch of that one, for two reasons. The ids are not planes of 81 squares the
//! way BonaPiece's are (a hand slot sits after the whole board, and its copy
//! number is zero-based), so none of that file's plane tables apply. And that
//! file's incremental machinery exists to cope with a king that moves under a
//! feature set crossed on it, which this space never does: a king move is the
//! same two removed-and-added tokens as any other move.
//!
//! Ported from `manaka-core`'s `token.rs` (`TokenRef::id`, `tokens`); the
//! numbering below is that file's, and `tests.rs` checks it against the
//! definition written out from scratch rather than against this module's tables.
//!
//! ## What a move changes
//!
//! At most two tokens leave and two arrive, in each frame: the lifted piece, the
//! captured one, the piece as it lands (possibly promoted) and the capture
//! entering the mover's hand — or, for a drop, the hand slot emptying and the
//! piece appearing. [`MoveDelta::from_move`] reads that off the move and the
//! pre-move position without scanning the board, and the accumulator applies it.
//! A delta applied to the token list of the position before the move gives the
//! token list of the position after it, which `tests.rs` checks over played-out
//! games with every kind of move in them.

use attic_state::{Color, Move, Piece, PieceKind, Position, Square};

/// A token's row in the embedding table.
pub type FeatureIndex = u32;

/// Tokens a legal position presents: forty pieces, wherever they are, kings
/// included.
pub const MAX_ACTIVE_FEATURES: usize = 40;

pub(crate) const SQUARES: usize = Square::COUNT;

/// Piece kinds in token order: the eight [`PieceKind`]s, then the promoted
/// pawn, lance, knight, silver, bishop and rook. A promoted gold or king does
/// not exist and keeps its own number, so the table needs no special row.
const BOARD_KIND: [[u8; 2]; PieceKind::COUNT] = {
    let mut table = [[0u8; 2]; PieceKind::COUNT];
    let mut kind = 0;
    while kind < PieceKind::COUNT {
        table[kind] = [kind as u8, kind as u8];
        kind += 1;
    }
    table[PieceKind::Pawn.index()][1] = 8;
    table[PieceKind::Lance.index()][1] = 9;
    table[PieceKind::Knight.index()][1] = 10;
    table[PieceKind::Silver.index()][1] = 11;
    table[PieceKind::Bishop.index()][1] = 12;
    table[PieceKind::Rook.index()][1] = 13;
    table
};

/// Board kinds, promoted ones included.
const BOARD_KINDS: usize = 14;

/// Rows taken by the board: kind x ours-or-theirs x square.
pub(crate) const BOARD_ROWS: usize = BOARD_KINDS * 2 * SQUARES;

/// The kinds that can be held, in token order, which is [`PieceKind`]'s order
/// with the king left off.
pub(crate) const HAND_KINDS: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

/// The most copies of each hand kind that can exist.
const HAND_MAX: [usize; 7] = [18, 4, 4, 4, 4, 2, 2];

/// Hand rows for one owner, which is what [`HAND_MAX`] sums to.
pub(crate) const HAND_SIDE: usize = 38;

/// Where each hand kind's block of copies starts within one owner's hand rows.
const HAND_OFFSET: [usize; 7] = {
    let mut table = [0usize; 7];
    let mut kind = 1;
    while kind < 7 {
        table[kind] = table[kind - 1] + HAND_MAX[kind - 1];
        kind += 1;
    }
    table
};

const _: () = {
    assert!(HAND_OFFSET[6] + HAND_MAX[6] == HAND_SIDE);
    assert!(BOARD_ROWS + 2 * HAND_SIDE == crate::types::NUM_FEATURES);
};

/// A square as the frame reads it, in Manaka's numbering.
///
/// Manaka numbers a square `(9 - file) * 9 + (rank - 1)`, with the first square
/// of an SFEN's first rank (file 9, rank `a`) as 0, and YaneuraOu — which is
/// what [`Square::index`] is — `(file - 1) * 9 + (rank - 1)`. The two differ by
/// the direction the files run and nothing else, so one maps to the other by
/// `(8 - sq / 9) * 9 + sq % 9`, and the mapping is its own inverse. The white
/// frame turns the board by 180° first, which in Manaka's numbering is
/// `80 - sq` and here is the same thing done to the YaneuraOu index before it
/// is converted; the two orders agree.
const FRAME_SQUARE: [[u8; SQUARES]; Color::COUNT] = {
    let mut table = [[0u8; SQUARES]; Color::COUNT];
    let mut sq = 0;
    while sq < SQUARES {
        let black = sq;
        let white = SQUARES - 1 - sq;
        table[Color::Black.index()][sq] = ((8 - black / 9) * 9 + black % 9) as u8;
        table[Color::White.index()][sq] = ((8 - white / 9) * 9 + white % 9) as u8;
        sq += 1;
    }
    table
};

/// The row of a piece standing on `sq`, read from `frame`.
#[inline]
pub(crate) fn board_row(piece: Piece, sq: Square, frame: Color) -> FeatureIndex {
    let theirs = usize::from(piece.color != frame);
    let kind = BOARD_KIND[piece.kind.index()][usize::from(piece.promoted)] as usize;
    let square = FRAME_SQUARE[frame.index()][sq.index() as usize] as usize;
    ((kind * 2 + theirs) * SQUARES + square) as FeatureIndex
}

/// The row of the `copy`-th (zero-based) piece of `kind` in `owner`'s hand, read
/// from `frame`.
///
/// # Panics
/// Panics, in a debug build, if `kind` cannot be held or `copy` is past the most
/// of it there can be.
#[inline]
pub(crate) fn hand_row(owner: Color, kind: PieceKind, copy: usize, frame: Color) -> FeatureIndex {
    debug_assert!(kind != PieceKind::King, "a king is never held in hand");
    debug_assert!(copy < HAND_MAX[kind.index()], "more copies than exist");
    let theirs = usize::from(owner != frame);
    (BOARD_ROWS + theirs * HAND_SIDE + HAND_OFFSET[kind.index()] + copy) as FeatureIndex
}

/// The row the same piece has in the other frame.
///
/// The frames differ in two things, and both are in the row: whose a piece is,
/// relative to the frame, and which way up the board is read. A board row keeps
/// its kind, swaps ours and theirs, and turns its square by 180°, which in
/// Manaka's numbering is `80 − sq` (see [`FRAME_SQUARE`]). A hand row swaps ours
/// and theirs and keeps its kind and copy. So `other_frame(board_row(p, sq,
/// Black)) == board_row(p, sq, White)`, the same for hands, and the mapping is its
/// own inverse.
///
/// Because [`active_features_into`] lists the pieces in the same order for either
/// frame, and a [`MoveDelta`] names the same slots in the same order for both,
/// the white frame's list is the black frame's read through this, token for
/// token. A table over pairs of tokens is read for both frames from the black
/// frame's list on the strength of that (see `manaka/pair.rs`).
#[inline]
pub(crate) const fn other_frame(row: FeatureIndex) -> FeatureIndex {
    let row = row as usize;
    let other = if row < BOARD_ROWS {
        let (owned_kind, square) = (row / SQUARES, row % SQUARES);
        // `owned_kind` is `kind · 2 + theirs`: flipping the low bit swaps ours and theirs.
        (owned_kind ^ 1) * SQUARES + (SQUARES - 1 - square)
    } else {
        let (theirs, slot) = (
            (row - BOARD_ROWS) / HAND_SIDE,
            (row - BOARD_ROWS) % HAND_SIDE,
        );
        BOARD_ROWS + (1 - theirs) * HAND_SIDE + slot
    };
    other as FeatureIndex
}

/// Every token of `pos` as `frame` reads it: one per board piece in square
/// order, then one per held piece. A legal position gives exactly
/// [`MAX_ACTIVE_FEATURES`] of them.
///
/// Where the other spaces pad a sparse position up to forty with a zero feature,
/// this does not: the multiset is the position, and a test position with six
/// pieces has six tokens. The reference sums over the tokens that are there.
pub fn active_features(pos: &Position, frame: Color) -> Vec<FeatureIndex> {
    let mut list = Vec::with_capacity(MAX_ACTIVE_FEATURES);
    active_features_into(pos, frame, &mut list);
    list
}

/// [`active_features`] writing into a caller-owned buffer, so that a refresh
/// does not allocate.
pub(crate) fn active_features_into(pos: &Position, frame: Color, list: &mut Vec<FeatureIndex>) {
    list.clear();
    list.reserve(MAX_ACTIVE_FEATURES);
    for index in 0..SQUARES as u8 {
        let sq = Square::from_index(index).expect("index < Square::COUNT is valid");
        if let Some(piece) = pos.board().get(sq) {
            list.push(board_row(piece, sq, frame));
        }
    }
    for owner in [Color::Black, Color::White] {
        let hand = pos.hand(owner);
        for kind in HAND_KINDS {
            for copy in 0..usize::from(hand.count(kind)) {
                list.push(hand_row(owner, kind, copy, frame));
            }
        }
    }
}

/// Tokens for both frames, indexed by [`Color::index`].
pub fn active_features_both(pos: &Position) -> [Vec<FeatureIndex>; Color::COUNT] {
    [
        active_features(pos, Color::Black),
        active_features(pos, Color::White),
    ]
}

/// Whether `frame`'s accumulator half must be rebuilt after `mv` rather than
/// updated: never. No row here contains a king square, so a king move is two
/// tokens like any other, and the question only exists because the other spaces
/// share this surface with HalfKP, where the answer can be yes.
pub fn requires_full_refresh(_mv: Move, _frame: Color) -> bool {
    false
}

/// One board or hand slot a move touches, before a frame has read it.
#[derive(Clone, Copy)]
enum Slot {
    Board {
        sq: Square,
        piece: Piece,
    },
    /// The `copy`-th (zero-based) piece of `kind` in `owner`'s hand.
    Hand {
        owner: Color,
        kind: PieceKind,
        copy: usize,
    },
}

impl Slot {
    #[inline]
    fn row(self, frame: Color) -> FeatureIndex {
        match self {
            Slot::Board { sq, piece } => board_row(piece, sq, frame),
            Slot::Hand { owner, kind, copy } => hand_row(owner, kind, copy, frame),
        }
    }
}

/// The tokens one frame loses and gains across a move. At most two of each, so
/// the lists are fixed two-slot arrays.
///
/// Plain data written field by field, for the reason the NNUE delta is: a value
/// built zero-initialised and moved by value is copied with vector moves out of
/// a stack slot that was just written with scalar stores, and a load spanning
/// several recent stores cannot be forwarded.
#[derive(Clone, Copy)]
pub struct PerspectiveDelta {
    removed: [FeatureIndex; 2],
    added: [FeatureIndex; 2],
    n_removed: u8,
    n_added: u8,
}

impl PerspectiveDelta {
    /// Tokens to take out of the pre-move multiset, in the order to do it.
    #[inline]
    pub fn removed(&self) -> &[FeatureIndex] {
        &self.removed[..usize::from(self.n_removed)]
    }

    /// Tokens to put into it, in the order to do it.
    #[inline]
    pub fn added(&self) -> &[FeatureIndex] {
        &self.added[..usize::from(self.n_added)]
    }
}

/// The per-frame token delta a move induces, computed from the pre-move
/// position without scanning the board.
pub struct MoveDelta {
    halves: [PerspectiveDelta; Color::COUNT],
}

impl MoveDelta {
    /// This move's delta for `frame`. Always `Some`: the signature is the one
    /// the shapes that can need a rebuild have, so the search is written once.
    #[inline]
    pub fn half(&self, frame: Color) -> Option<&PerspectiveDelta> {
        Some(&self.halves[frame.index()])
    }

    /// Compute the delta for `mv` from the **pre-move** `pos`, which is not
    /// mutated.
    ///
    /// The removed tokens come before the added ones in the order they are
    /// listed in, and a capture's victim leaves before its captor lands. That
    /// order is the one a table that depends on the other tokens present (a
    /// pair table) has to follow: take each token out of the set before asking
    /// what it paired with, and put each one in after.
    ///
    /// # Panics
    /// Panics if a non-drop move has no piece on its origin square.
    #[inline]
    pub fn from_move(pos: &Position, mv: Move) -> MoveDelta {
        let mover = pos.side_to_move();
        let to = mv.to_sq();

        if mv.is_drop() {
            // The last copy held leaves the hand and an unpromoted piece
            // appears at `to`.
            let kind = mv.dropped_piece_kind();
            let held = usize::from(pos.hand(mover).count(kind));
            debug_assert!(held > 0, "dropping a piece that is not held");
            return Self::assemble(
                [Slot::Hand {
                    owner: mover,
                    kind,
                    copy: held - 1,
                }],
                [Slot::Board {
                    sq: to,
                    piece: Piece::new(kind, mover),
                }],
            );
        }

        let from = mv.from_sq();
        // The `Move` encoding already carries the moving piece, so the board is
        // not read for it. `before` undoes any promotion.
        let after = mv.moved_piece_after();
        let before = Piece {
            promoted: after.promoted && !mv.is_promote(),
            ..after
        };
        debug_assert_eq!(
            pos.board().get(from),
            Some(before),
            "Move encoding disagrees with the board on the mover at {from:?}",
        );
        let lifted = Slot::Board {
            sq: from,
            piece: before,
        };
        let placed = Slot::Board {
            sq: to,
            piece: after,
        };

        match pos.board().get(to) {
            None => Self::assemble([lifted], [placed]),
            // The victim leaves the board and enters the mover's hand as its
            // unpromoted kind, behind the copies already there.
            Some(captured) => {
                let held = usize::from(pos.hand(mover).count(captured.kind));
                Self::assemble(
                    [
                        Slot::Board {
                            sq: to,
                            piece: captured,
                        },
                        lifted,
                    ],
                    [
                        Slot::Hand {
                            owner: mover,
                            kind: captured.kind,
                            copy: held,
                        },
                        placed,
                    ],
                )
            }
        }
    }

    #[inline(always)]
    fn assemble<const R: usize, const A: usize>(removed: [Slot; R], added: [Slot; A]) -> MoveDelta {
        MoveDelta {
            halves: [
                Self::half_for(Color::Black, &removed, &added),
                Self::half_for(Color::White, &removed, &added),
            ],
        }
    }

    #[inline(always)]
    fn half_for<const R: usize, const A: usize>(
        frame: Color,
        removed: &[Slot; R],
        added: &[Slot; A],
    ) -> PerspectiveDelta {
        let mut half = PerspectiveDelta {
            removed: [0; 2],
            added: [0; 2],
            n_removed: R as u8,
            n_added: A as u8,
        };
        for (slot, row) in removed.iter().zip(&mut half.removed) {
            *row = slot.row(frame);
        }
        for (slot, row) in added.iter().zip(&mut half.added) {
            *row = slot.row(frame);
        }
        half
    }
}

#[cfg(test)]
mod tests;
