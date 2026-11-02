//! Static Exchange Evaluation "greater-or-equal" test, ported from
//! `Position::see_ge` (the `#if !STOCKFISH` shogi path).
//!
//! `see_ge(m, threshold)` returns `true` iff the material swing of the optimal
//! capture/recapture sequence on `m`'s destination square is at least
//! `threshold`.
//!
//! Two reference behaviours are not what the names suggest:
//!
//! * The moving piece's own promotion is **not** credited: the value taken is
//!   that of the piece as it stands on `from`, which a promotion move leaves
//!   unpromoted until it reaches `to`. The reference ignores the final
//!   promotion gain so that its early cutoff stays valid.
//! * The least-valuable-attacker try order is PAWN, LANCE, KNIGHT, SILVER,
//!   GOLDS, BISHOP, ROOK, HORSE, DRAGON, KING — ROOK before HORSE, even though
//!   `HorseValue < RookValue`. Node-count parity depends on it.

use crate::bitboard::{
    Bitboard, between, bishop_step_effect, dir_toward, lance_attacks, lance_step_effect,
    rook_step_effect, slide_ray,
};
use crate::board::pat;
use crate::color::Color;
use crate::move_::Move;
use crate::movegen::{attackers_to_both, try_find_king};
use crate::piece::Piece;
use crate::position::Position;
use crate::square::Square;

// The Apery material values, verbatim from the reference's `Eval::` enum (the
// `USE_PIECE_VALUE` block).
const PAWN_VALUE: i32 = 90;
const LANCE_VALUE: i32 = 315;
const KNIGHT_VALUE: i32 = 405;
const SILVER_VALUE: i32 = 495;
const GOLD_VALUE: i32 = 540;
const BISHOP_VALUE: i32 = 855;
const ROOK_VALUE: i32 = 990;
const PRO_PAWN_VALUE: i32 = 540;
const PRO_LANCE_VALUE: i32 = 540;
const PRO_KNIGHT_VALUE: i32 = 540;
const PRO_SILVER_VALUE: i32 = 540;
const HORSE_VALUE: i32 = 945;
const DRAGON_VALUE: i32 = 1395;
const KING_VALUE: i32 = 15000;

/// The match form of [`piece_value`], used once to build [`PIECE_VALUE_OF_CODE`].
const fn piece_value_match(p: Piece) -> i32 {
    use crate::piece::PieceKind;
    match (p.kind, p.promoted) {
        (PieceKind::Pawn, false) => PAWN_VALUE,
        (PieceKind::Pawn, true) => PRO_PAWN_VALUE,
        (PieceKind::Lance, false) => LANCE_VALUE,
        (PieceKind::Lance, true) => PRO_LANCE_VALUE,
        (PieceKind::Knight, false) => KNIGHT_VALUE,
        (PieceKind::Knight, true) => PRO_KNIGHT_VALUE,
        (PieceKind::Silver, false) => SILVER_VALUE,
        (PieceKind::Silver, true) => PRO_SILVER_VALUE,
        (PieceKind::Gold, _) => GOLD_VALUE,
        (PieceKind::Bishop, false) => BISHOP_VALUE,
        (PieceKind::Bishop, true) => HORSE_VALUE,
        (PieceKind::Rook, false) => ROOK_VALUE,
        (PieceKind::Rook, true) => DRAGON_VALUE,
        (PieceKind::King, _) => KING_VALUE,
    }
}

/// [`piece_value`] for every [`Piece::code`], so the lookup is one load rather
/// than a two-level `match`.
static PIECE_VALUE_OF_CODE: [i32; Piece::CODES] = {
    let mut t = [0i32; Piece::CODES];
    let mut code = 0;
    while code < Piece::CODES {
        t[code] = piece_value_match(Piece::from_code(code));
        code += 1;
    }
    t
};

/// `Eval::PieceValue[piece]` — the value of a concrete board piece, with
/// promoted pieces returning their promoted value.
#[inline(always)]
pub fn piece_value(p: Piece) -> i32 {
    PIECE_VALUE_OF_CODE[p.code()]
}

/// The least-valuable attacker among `stm_attackers`, as the reference's
/// `pieces(<TYPE>)` else-if chain tries them: PAWN, LANCE, KNIGHT, SILVER,
/// GOLDS (plain gold and any promoted `{Pawn, Lance, Knight, Silver}`), BISHOP,
/// ROOK, HORSE, DRAGON. Returns the attackers of that bucket, the constant the
/// swap loop subtracts for it, and whether it is a knight (which jumps, so
/// consuming it uncovers nothing). `None` means only a KING remains — the
/// reference's terminal `else` branch. `stm_attackers` is non-empty by the
/// caller's guarantee.
///
/// Each bucket is one `AND` and one `ptest`; the constants are immediates, so
/// no bucket table is built or indexed.
#[inline(always)]
fn least_valuable_attacker(
    board: &crate::board::Board,
    stm_attackers: Bitboard,
    stm: Color,
) -> Option<(Bitboard, i32, bool)> {
    let bb = stm_attackers & board.pieces_pattern(stm, pat::PAWN);
    if !bb.is_empty() {
        return Some((bb, PAWN_VALUE, false));
    }
    let bb = stm_attackers & board.pieces_pattern(stm, pat::LANCE);
    if !bb.is_empty() {
        return Some((bb, LANCE_VALUE, false));
    }
    let bb = stm_attackers & board.pieces_pattern(stm, pat::KNIGHT);
    if !bb.is_empty() {
        return Some((bb, KNIGHT_VALUE, true));
    }
    let bb = stm_attackers & board.pieces_pattern(stm, pat::SILVER);
    if !bb.is_empty() {
        return Some((bb, SILVER_VALUE, false));
    }
    let bb = stm_attackers & board.pieces_pattern(stm, pat::GOLD);
    if !bb.is_empty() {
        return Some((bb, GOLD_VALUE, false));
    }
    let bb = stm_attackers & board.pieces_pattern(stm, pat::BISHOP);
    if !bb.is_empty() {
        return Some((bb, BISHOP_VALUE, false));
    }
    let bb = stm_attackers & board.pieces_pattern(stm, pat::ROOK);
    if !bb.is_empty() {
        return Some((bb, ROOK_VALUE, false));
    }
    let bb = stm_attackers & board.pieces_pattern(stm, pat::HORSE);
    if !bb.is_empty() {
        return Some((bb, HORSE_VALUE, false));
    }
    let bb = stm_attackers & board.pieces_pattern(stm, pat::DRAGON);
    if !bb.is_empty() {
        return Some((bb, DRAGON_VALUE, false));
    }
    None
}

/// Reveal the x-ray sliders uncovered when the piece on `removed` (already
/// cleared from `occ`) is consumed. `removed` is always aligned with `to` on
/// one of the eight ray directions, because the caller skips knights.
///
/// Only the one ray from `to` through `removed` can open, so only that ray is
/// walked — the reference's `rayEffect<DIR>(to, occupied)` switch — and the
/// pieces that attack along it are the ones OR-ed in: bishops and horses on a
/// diagonal, rooks and dragons on a rank, and on a file the rooks and dragons
/// plus the lance that faces `to` (a Black lance attacks toward rank 0, so it
/// looks at `to` from the rank-8 side, and the other way round for White).
#[inline(always)]
fn reveal_sliders(
    board: &crate::board::Board,
    to: Square,
    removed: Square,
    occ: Bitboard,
) -> Bitboard {
    let rook_dragon = board.pieces_pattern(Color::Black, pat::ROOK)
        | board.pieces_pattern(Color::Black, pat::DRAGON)
        | board.pieces_pattern(Color::White, pat::ROOK)
        | board.pieces_pattern(Color::White, pat::DRAGON);
    let dir = dir_toward(to, removed);
    debug_assert!(dir < 8, "non-knight SEE attacker is aligned with `to`");
    match dir {
        // Toward rank 0: the revealed lance is a White one (it attacks toward
        // rank 8).
        0 => {
            lance_attacks(Color::Black, to, occ)
                & (rook_dragon | board.pieces_pattern(Color::White, pat::LANCE))
        }
        // Toward rank 8: a Black lance.
        1 => {
            lance_attacks(Color::White, to, occ)
                & (rook_dragon | board.pieces_pattern(Color::Black, pat::LANCE))
        }
        2 | 3 => slide_ray(dir, to, occ) & rook_dragon,
        4..=7 => {
            let bishop_horse = board.pieces_pattern(Color::Black, pat::BISHOP)
                | board.pieces_pattern(Color::Black, pat::HORSE)
                | board.pieces_pattern(Color::White, pat::BISHOP)
                | board.pieces_pattern(Color::White, pat::HORSE);
            slide_ray(dir, to, occ) & bishop_horse
        }
        _ => unreachable!("non-knight SEE attacker is aligned with `to`"),
    }
}

/// Port of `Position::update_slider_blockers(c)`: returns
/// `(blockersForKing[c], pinners[~c])` as square sets. `blockersForKing[c]` is
/// the set of single blockers between `c`'s king and an enemy sniper;
/// `pinners[~c]` is the set of enemy snipers whose single blocker is one of
/// `c`'s own pieces. Both are computed over the full pre-move board.
pub(crate) fn slider_blockers(board: &crate::board::Board, c: Color) -> (Bitboard, Bitboard) {
    slider_blockers_at(board, c, try_find_king(board, c))
}

/// [`slider_blockers`] with `c`'s king square (`None` if the board has no `c`
/// king) already known, so a caller that located both kings does not search
/// the piece sets again.
#[inline]
pub(crate) fn slider_blockers_at(
    board: &crate::board::Board,
    c: Color,
    king: Option<Square>,
) -> (Bitboard, Bitboard) {
    let ksq = match king {
        Some(s) => s,
        None => return (Bitboard::EMPTY, Bitboard::EMPTY),
    };
    debug_assert_eq!(king, try_find_king(board, c));
    let enemy = c.flip();

    // The occupancy-independent full rays to the edge — the reference's
    // `rookStepEffect` / `bishopStepEffect` / `lanceStepEffect`, each a single
    // table load (equal to the occupancy-limited query under an empty board).
    let rook_line = rook_step_effect(ksq);
    let bishop_line = bishop_step_effect(ksq);
    let lance_line = lance_step_effect(c, ksq);
    let rook_dragon =
        board.pieces_pattern(enemy, pat::ROOK) | board.pieces_pattern(enemy, pat::DRAGON);
    let bishop_horse =
        board.pieces_pattern(enemy, pat::BISHOP) | board.pieces_pattern(enemy, pat::HORSE);
    let lance = board.pieces_pattern(enemy, pat::LANCE);
    let snipers = (rook_line & rook_dragon) | (bishop_line & bishop_horse) | (lance_line & lance);

    // Snipers are removed from the occupancy, so a slider standing in front of
    // another sniper is not itself counted as a blocker.
    let occupancy = board.occupied() ^ snipers;
    let own = board.pieces_color(c);

    let mut blockers = Bitboard::EMPTY;
    let mut pinners = Bitboard::EMPTY;
    for sniper_sq in snipers.squares() {
        let b = between(ksq, sniper_sq) & occupancy;
        if b.popcount() == 1 {
            blockers |= b;
            if !(b & own).is_empty() {
                pinners |= Bitboard::from_square(sniper_sq);
            }
        }
    }

    (blockers, pinners)
}

impl Position {
    /// Static Exchange Evaluation "greater-or-equal" test.
    ///
    /// Returns `true` iff the SEE value of move `m` is `>= threshold`.
    ///
    /// `m` is interpreted against the current side to move and need not be a
    /// capture, but SEE is only meaningful for captures: a quiet move onto an
    /// empty square has victim value `0`.
    pub fn see_ge(&self, m: Move, threshold: i32) -> bool {
        let board = self.board();
        let drop = m.is_drop();
        let to = m.to_sq();

        let victim_value = match board.get(to) {
            Some(p) => piece_value(p),
            None => 0,
        };
        let mut swap = victim_value - threshold;
        if swap < 0 {
            return false;
        }

        let mover = self.side_to_move();
        let from_value = if drop {
            piece_value(Piece::new(m.dropped_piece_kind(), mover))
        } else {
            piece_value(
                board
                    .get(m.from_sq())
                    .expect("see_ge: board move has no piece on `from`"),
            )
        };
        swap = from_value - swap;
        if swap <= 0 {
            return true;
        }

        // Clearing `from` reveals the x-ray attackers behind the moving piece.
        // Clearing `to` is immaterial — it never blocks attacks to itself — but
        // the reference xors it, so this does too.
        let mut removed = Bitboard::from_square(to);
        if !drop {
            removed |= Bitboard::from_square(m.from_sq());
        }
        let mut occupied = board.occupied().and_not(removed);

        // Collected once, then maintained incrementally: `attackers &= occupied`
        // drops consumed pieces and [`reveal_sliders`] adds the ones uncovered
        // behind them.
        let mut attackers = attackers_to_both(board, to, occupied);

        // Read in place rather than copied out: most calls end in the first
        // round or two and never look at the pins.
        let check_info = self.check_info();

        let mut stm = mover;
        let mut res: i32 = 1;

        loop {
            stm = stm.flip();
            attackers &= occupied;
            let mut stm_attackers = attackers & board.pieces_color(stm);

            if stm_attackers.is_empty() {
                break;
            }

            // Don't allow pinned pieces to attack while a pinner is still on
            // the board. The `~stm` snipers pinning `stm`'s pieces to `stm`'s
            // king are `pinners(stm)` flipped: [`slider_blockers`] returns
            // `pinners[~c]` alongside `blockersForKing[c]`.
            if !(check_info.pinners(stm.flip()) & occupied).is_empty() {
                stm_attackers = stm_attackers.and_not(check_info.blockers(stm));
                if stm_attackers.is_empty() {
                    break;
                }
            }

            res ^= 1;

            let Some((bb, value, is_knight)) = least_valuable_attacker(board, stm_attackers, stm)
            else {
                // Only a KING remains, and capturing with it loses the king
                // if the opponent still attacks `to`.
                let opp_attackers = attackers & board.pieces_color(stm.flip());
                let final_res = if !opp_attackers.is_empty() {
                    res ^ 1
                } else {
                    res
                };
                return final_res != 0;
            };

            swap = value - swap;
            if swap < res {
                break;
            }

            // The reference's `bb.pop()`: the lowest square of the bucket. The
            // attacker is in `occupied` (it came out of `attackers &= occupied`),
            // so toggling its bit clears it.
            let lva_sq = bb.lowest_square();
            occupied ^= Bitboard::from_square(lva_sq);
            // Knights jump, so they uncover nothing.
            if !is_knight {
                attackers |= reveal_sliders(board, to, lva_sq, occupied);
            }
        }

        res != 0
    }
}

#[cfg(test)]
mod oracle;

#[cfg(test)]
mod equivalence {
    use super::*;
    use crate::move_::Move;
    use crate::piece::PieceKind;
    use crate::sfen::parse_sfen;

    const FIXTURE_SFENS: &[&str] = &[
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",
        "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
        "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
        "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
        "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
    ];

    const THRESHOLDS: &[i32] = &[-2000, -990, -500, -90, -1, 0, 1, 90, 500, 990, 2000];

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn pick(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// The victim on `to` minus the mover's value, added to the swept
    /// thresholds so that each move's break-even boundary is probed directly.
    fn naive_exchange_value(pos: &Position, m: Move) -> i32 {
        let board = pos.board();
        let victim = board.get(m.to_sq()).map_or(0, piece_value);
        let mover = if m.is_drop() {
            piece_value(Piece::new(m.dropped_piece_kind(), pos.side_to_move()))
        } else {
            board.get(m.from_sq()).map_or(0, piece_value)
        };
        victim - mover
    }

    fn check_move(pos: &Position, m: Move) {
        let mut thresholds: Vec<i32> = THRESHOLDS.to_vec();
        let naive = naive_exchange_value(pos, m);
        thresholds.push(naive);
        thresholds.push(naive + 1);
        thresholds.push(naive - 1);
        for &th in &thresholds {
            let got = pos.see_ge(m, th);
            assert_eq!(
                got,
                pos.see_ge_reference(m, th),
                "see_ge disagrees with full-rescan reference on move {m:?} at threshold {th}",
            );
            assert_eq!(
                got,
                pos.see_ge_incremental(m, th),
                "see_ge disagrees with the incremental oracle on move {m:?} at threshold {th}",
            );
        }
    }

    fn check_slider_blockers(pos: &Position) {
        let board = pos.board();
        for c in [Color::Black, Color::White] {
            assert_eq!(
                slider_blockers(board, c),
                oracle::slider_blockers_scalar_bb(board, c),
                "slider_blockers disagrees with scalar oracle for {c:?}",
            );
        }
    }

    /// [`attackers_to_both`] must equal the per-colour OR on every square, both
    /// under the full occupancy and under the reduced occupancies SEE walks
    /// through as attackers are consumed.
    fn check_attackers_to_both(pos: &Position) {
        use crate::bitboard::Bitboard;
        use crate::movegen::attackers_bb_occ;

        let board = pos.board();
        let full = board.occupied();

        for idx in 0..Square::COUNT as u8 {
            let sq = Square::from_index(idx).unwrap();

            let mut occs: Vec<Bitboard> = vec![full];
            for removed in full.squares() {
                occs.push(full & !Bitboard::from_square(removed));
            }

            for &occ in &occs {
                let fused = attackers_to_both(board, sq, occ);
                let per_color = attackers_bb_occ(board, sq, Color::Black, occ)
                    | attackers_bb_occ(board, sq, Color::White, occ);
                assert_eq!(
                    fused, per_color,
                    "attackers_to_both disagrees with per-colour OR at {sq:?} under occ {occ:?}",
                );
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn optimized_see_ge_matches_reference_on_fixture_playouts() {
        const MIN_PLIES: usize = 60;

        for (fi, sfen) in FIXTURE_SFENS.iter().enumerate() {
            let mut pos = parse_sfen(sfen).expect("fixture sfen parses");
            let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D ^ (fi as u64).wrapping_add(1));
            let mut legal: Vec<Move> = Vec::new();

            let mut plies = 0usize;
            while plies < MIN_PLIES {
                check_slider_blockers(&pos);
                check_attackers_to_both(&pos);

                legal.clear();
                pos.generate_legal_all(&mut legal);
                if legal.is_empty() {
                    break;
                }

                for &m in &legal {
                    check_move(&pos, m);
                }

                let m = legal[rng.pick(legal.len())];
                pos.do_move(m);
                plies += 1;
            }
        }
    }

    /// A piece already promoted on the board contributes its promoted value: a
    /// Tokin is worth a Gold, a Horse `HorseValue`, a Dragon `DragonValue`.
    #[test]
    fn promoted_piece_values() {
        assert_eq!(piece_value(Piece::new(PieceKind::Pawn, Color::Black)), 90);
        assert_eq!(
            piece_value(Piece::promoted(PieceKind::Pawn, Color::Black).unwrap()),
            540
        );
        assert_eq!(
            piece_value(Piece::promoted(PieceKind::Bishop, Color::Black).unwrap()),
            945
        );
        assert_eq!(
            piece_value(Piece::promoted(PieceKind::Rook, Color::Black).unwrap()),
            1395
        );
    }
}
