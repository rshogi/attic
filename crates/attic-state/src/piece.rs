use crate::color::Color;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PieceKind {
    Pawn = 0,
    Lance = 1,
    Knight = 2,
    Silver = 3,
    Gold = 4,
    Bishop = 5,
    Rook = 6,
    King = 7,
}

impl PieceKind {
    pub const COUNT: usize = 8;

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn can_promote(self) -> bool {
        matches!(
            self,
            PieceKind::Pawn
                | PieceKind::Lance
                | PieceKind::Knight
                | PieceKind::Silver
                | PieceKind::Bishop
                | PieceKind::Rook
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    pub kind: PieceKind,
    pub color: Color,
    pub promoted: bool,
}

impl PieceKind {
    /// The kind whose [`Self::index`] is `index` (`0..PieceKind::COUNT`); the
    /// inverse of [`Self::index`], for the `const` table builders.
    pub(crate) const fn from_index(index: usize) -> PieceKind {
        match index {
            0 => PieceKind::Pawn,
            1 => PieceKind::Lance,
            2 => PieceKind::Knight,
            3 => PieceKind::Silver,
            4 => PieceKind::Gold,
            5 => PieceKind::Bishop,
            6 => PieceKind::Rook,
            _ => PieceKind::King,
        }
    }
}

impl Piece {
    /// The number of distinct [`Self::code`] values: `promoted(2) x color(2) x
    /// kind(8)`. A promoted gold or king is never realised on a board but keeps
    /// a slot, so that the code stays a branch-free index computation.
    pub(crate) const CODES: usize = 2 * Color::COUNT * PieceKind::COUNT;

    /// A dense index `0..CODES` over every `(kind, color, promoted)`, for the
    /// per-piece lookup tables.
    #[inline(always)]
    pub(crate) const fn code(self) -> usize {
        (self.promoted as usize * Color::COUNT + self.color as usize) * PieceKind::COUNT
            + self.kind as usize
    }

    /// The inverse of [`Self::code`].
    pub(crate) const fn from_code(code: usize) -> Piece {
        Piece {
            kind: PieceKind::from_index(code % PieceKind::COUNT),
            color: if (code / PieceKind::COUNT).is_multiple_of(Color::COUNT) {
                Color::Black
            } else {
                Color::White
            },
            promoted: code / (PieceKind::COUNT * Color::COUNT) != 0,
        }
    }

    pub const fn new(kind: PieceKind, color: Color) -> Self {
        Self {
            kind,
            color,
            promoted: false,
        }
    }

    pub const fn promoted(kind: PieceKind, color: Color) -> Option<Self> {
        if kind.can_promote() {
            Some(Self {
                kind,
                color,
                promoted: true,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_unpromoted() {
        let p = Piece::new(PieceKind::Pawn, Color::Black);
        assert_eq!(p.kind, PieceKind::Pawn);
        assert_eq!(p.color, Color::Black);
        assert!(!p.promoted);
    }

    #[test]
    fn code_round_trips_over_every_slot() {
        for code in 0..Piece::CODES {
            assert_eq!(Piece::from_code(code).code(), code);
        }
    }

    #[test]
    fn promoted_rejects_gold_and_king() {
        assert!(Piece::promoted(PieceKind::Gold, Color::Black).is_none());
        assert!(Piece::promoted(PieceKind::King, Color::White).is_none());
    }

    #[test]
    fn promoted_accepts_promotable_kinds() {
        for kind in [
            PieceKind::Pawn,
            PieceKind::Lance,
            PieceKind::Knight,
            PieceKind::Silver,
            PieceKind::Bishop,
            PieceKind::Rook,
        ] {
            let p = Piece::promoted(kind, Color::Black).unwrap();
            assert!(p.promoted);
            assert_eq!(p.kind, kind);
        }
    }
}
