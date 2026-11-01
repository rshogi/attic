use crate::piece::PieceKind;

/// The most copies of each kind that can be held, indexed by
/// [`PieceKind::index`]. A king is never held, so its cap is zero.
const MAX_BY_KIND: [u8; PieceKind::COUNT] = [18, 4, 4, 4, 4, 2, 2, 0];

/// The pieces a side holds, one count per kind.
///
/// The counts are the bytes of one `u64`, kind `k` in bits `8k..8k + 8`, so that
/// copying a hand is a register move and changing one count is a single
/// read-modify-write of the whole word. (Byte fields would make every copy of
/// the hand right after a capture or drop a store-forwarding stall.) The king
/// byte is always zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Hand(u64);

impl Hand {
    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn count(&self, kind: PieceKind) -> u8 {
        (self.0 >> (8 * kind.index())) as u8
    }

    pub fn increment(&mut self, kind: PieceKind) {
        if self.count(kind) < MAX_BY_KIND[kind.index()] {
            self.0 += 1 << (8 * kind.index());
        }
    }

    pub fn decrement(&mut self, kind: PieceKind) {
        if self.count(kind) > 0 {
            self.0 -= 1 << (8 * kind.index());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_has_zero_counts() {
        let h = Hand::empty();
        for kind in [
            PieceKind::Pawn,
            PieceKind::Lance,
            PieceKind::Knight,
            PieceKind::Silver,
            PieceKind::Gold,
            PieceKind::Bishop,
            PieceKind::Rook,
        ] {
            assert_eq!(h.count(kind), 0);
        }
    }

    #[test]
    fn increment_clamps_at_pawn_cap() {
        let mut h = Hand::empty();
        for _ in 0..30 {
            h.increment(PieceKind::Pawn);
        }
        assert_eq!(h.count(PieceKind::Pawn), 18);
    }

    #[test]
    fn increment_clamps_at_rook_cap() {
        let mut h = Hand::empty();
        for _ in 0..10 {
            h.increment(PieceKind::Rook);
        }
        assert_eq!(h.count(PieceKind::Rook), 2);
    }

    #[test]
    fn decrement_saturates_at_zero() {
        let mut h = Hand::empty();
        h.decrement(PieceKind::Lance);
        assert_eq!(h.count(PieceKind::Lance), 0);
    }

    #[test]
    fn king_is_not_held_in_hand() {
        let mut h = Hand::empty();
        h.increment(PieceKind::King);
        assert_eq!(h.count(PieceKind::King), 0);
    }
}
