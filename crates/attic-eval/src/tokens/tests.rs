//! The numbering against the definition, and the delta against the scan.
//!
//! Nothing here is checked against the tables in `super`. The oracle is the
//! numbering written out a second way from the definition — a position as
//! SFEN text, a square as `(9 - file) * 9 + (rank - 1)`, a hand copy as an
//! offset into its owner's block — so that a wrong table and a wrong reading of
//! the table cannot agree with each other.

use super::*;
use attic_state::{format_sfen, parse_sfen};

const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
/// Sparse and hand-heavy: six pieces on the board and six held ones.
const DROP_HEAVY: &str = "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1";
/// A middlegame with promoted pieces of both colours and hands on both sides.
const MIXED: &str = "ln2b4/5skgl/p4pn1r/2+P1p2pp/1sP3p2/2GSKP1GP/P1NPS2PN/1+pRpP2+bL/L4G3 w Pp 94";

/// One of every promoted kind for each colour, which no opening or random game
/// of this length puts on the board together: a mapping that numbered the
/// promoted rook as the promoted bishop would pass every other test here.
const EVERY_PROMOTION: &str = "+p+l+n+s+b+r1k1/9/9/9/9/9/9/9/+P+L+N+S+B+R1K1 b - 1";
/// Both hands full: the most of every kind there can be, which is the only
/// place the last copy of a pawn (the eighteenth) and the tail of each block
/// are numbered.
const FULL_HANDS: &str = "4k4/9/9/9/9/9/9/9/4K4 b 18P4L4N4S4G2B2R18p4l4n4s4g2b2r 1";

/// The kind number of an SFEN letter, promoted or not: Pawn 0, Lance 1, Knight
/// 2, Silver 3, Gold 4, Bishop 5, Rook 6, King 7, then ProPawn 8 through
/// ProRook 13.
fn spec_kind(letter: char, promoted: bool) -> usize {
    let base = "plnsgbrk"
        .find(letter.to_ascii_lowercase())
        .expect("a piece letter");
    match (promoted, base) {
        (false, _) => base,
        (true, 0..=3) => 8 + base,
        (true, 5 | 6) => 7 + base,
        (true, _) => panic!("{letter} does not promote"),
    }
}

/// Every token of an SFEN as `frame` reads it, straight from the definition.
fn spec_tokens(sfen: &str, frame: Color) -> Vec<usize> {
    let mut fields = sfen.split_whitespace();
    let (board, _side, hands) = (
        fields.next().unwrap(),
        fields.next().unwrap(),
        fields.next().unwrap(),
    );
    let mut out = Vec::new();

    for (rank_index, row) in board.split('/').enumerate() {
        let rank = rank_index + 1;
        let mut file = 9;
        let mut promoted = false;
        for c in row.chars() {
            if let Some(skip) = c.to_digit(10) {
                file -= skip as usize;
            } else if c == '+' {
                promoted = true;
            } else {
                let owner = if c.is_ascii_uppercase() {
                    Color::Black
                } else {
                    Color::White
                };
                let theirs = usize::from(owner != frame);
                let sq = (9 - file) * 9 + (rank - 1);
                let sq = if frame == Color::White { 80 - sq } else { sq };
                out.push((spec_kind(c, promoted) * 2 + theirs) * 81 + sq);
                promoted = false;
                file -= 1;
            }
        }
        assert_eq!(file, 0, "rank {rank} of {sfen:?} is not nine files wide");
    }

    if hands != "-" {
        let mut count = 0usize;
        for c in hands.chars() {
            if let Some(digit) = c.to_digit(10) {
                count = count * 10 + digit as usize;
                continue;
            }
            let owner = if c.is_ascii_uppercase() {
                Color::Black
            } else {
                Color::White
            };
            let kind = "plnsgbr"
                .find(c.to_ascii_lowercase())
                .expect("a hand letter");
            let max = [18, 4, 4, 4, 4, 2, 2];
            let start = 2268 + usize::from(owner != frame) * 38 + max[..kind].iter().sum::<usize>();
            for copy in 0..count.max(1) {
                out.push(start + copy);
            }
            count = 0;
        }
    }
    out
}

fn sorted(mut list: Vec<FeatureIndex>) -> Vec<usize> {
    list.sort_unstable();
    list.into_iter().map(|t| t as usize).collect()
}

/// A small deterministic generator (xorshift64*), so that a failure is the same
/// failure on every run and every machine.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as usize % n
    }
}

/// Played-out games from the opening, each ply with the position it was played
/// from. Random rather than scripted, because the cases that matter — a capture
/// of a promoted piece, a drop of the last copy, a drop of a pawn with seventeen
/// others held — are ones a fixed line does not reach.
fn games(count: usize, plies: usize) -> Vec<(Position, Move)> {
    let mut line = Vec::new();
    for seed in 1..=count as u64 {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
        let mut pos = Position::startpos();
        let mut legal = Vec::new();
        for _ in 0..plies {
            legal.clear();
            pos.generate_legal_all(&mut legal);
            if legal.is_empty() {
                break;
            }
            let mv = legal[rng.below(legal.len())];
            line.push((pos.clone(), mv));
            let _ = pos.do_move(mv);
        }
    }
    line
}

/// The numbering is the definition's, in both frames, on a position with hands
/// and promoted pieces as well as the opening.
#[test]
fn the_rows_are_numbered_as_the_definition_says() {
    for sfen in [STARTPOS, DROP_HEAVY, MIXED, EVERY_PROMOTION, FULL_HANDS] {
        let pos = parse_sfen(sfen).expect("a valid sfen");
        for frame in [Color::Black, Color::White] {
            let mut want = spec_tokens(sfen, frame);
            want.sort_unstable();
            assert_eq!(
                sorted(active_features(&pos, frame)),
                want,
                "{sfen} {frame:?}"
            );
        }
    }
    // And on what a game reaches, which is where a drop of a later copy and a
    // piece standing on a far corner come from.
    for (pos, _) in games(2, 200) {
        let sfen = format_sfen(&pos);
        for frame in [Color::Black, Color::White] {
            let mut want = spec_tokens(&sfen, frame);
            want.sort_unstable();
            assert_eq!(
                sorted(active_features(&pos, frame)),
                want,
                "{sfen} {frame:?}"
            );
        }
    }
}

/// A legal position has forty tokens, all different, all inside the table; a
/// sparse one has only the pieces it has, with no padding.
#[test]
fn a_legal_position_has_forty_distinct_rows_and_a_sparse_one_has_no_padding() {
    for (pos, _) in games(2, 150) {
        for frame in [Color::Black, Color::White] {
            let list = sorted(active_features(&pos, frame));
            assert_eq!(list.len(), MAX_ACTIVE_FEATURES);
            assert!(list.windows(2).all(|w| w[0] < w[1]), "a repeated row");
            assert!(list.iter().all(|&t| t < crate::NUM_FEATURES));
        }
    }
    let sparse = parse_sfen(DROP_HEAVY).expect("a valid sfen");
    assert_eq!(active_features(&sparse, Color::Black).len(), 12);
}

/// The position turned half way round, with the colours swapped.
fn turned(pos: &Position) -> Position {
    let mut out = Position::empty();
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).unwrap();
        if let Some(piece) = pos.board().get(sq) {
            let to = Square::from_index(80 - index).unwrap();
            let piece = Piece {
                color: piece.color.flip(),
                ..piece
            };
            out.board_mut().set(to, Some(piece));
        }
    }
    for color in [Color::Black, Color::White] {
        for kind in HAND_KINDS {
            for _ in 0..pos.hand(color).count(kind) {
                out.hand_mut(color.flip()).increment(kind);
            }
        }
    }
    out.set_side_to_move(pos.side_to_move().flip());
    out
}

/// "A position and its mirror produce the same ids": the white frame of a
/// position is the black frame of the same position turned half way round.
/// This is the sentence the whole space rests on, since it is what lets one
/// table serve both colours.
#[test]
fn the_white_frame_is_the_black_frame_of_the_turned_position() {
    for (pos, _) in games(2, 120) {
        let turned = turned(&pos);
        assert_eq!(
            sorted(active_features(&pos, Color::White)),
            sorted(active_features(&turned, Color::Black)),
        );
        assert_eq!(
            sorted(active_features(&pos, Color::Black)),
            sorted(active_features(&turned, Color::White)),
        );
    }
}

/// What a game reached, so that the delta test cannot pass by never meeting a
/// kind of move.
#[derive(Default, Debug)]
struct Seen {
    quiet: usize,
    captures: usize,
    captures_of_promoted: usize,
    promotions: usize,
    drops: usize,
    drops_of_a_later_copy: usize,
    king_moves: usize,
}

/// The delta takes the token multiset of the position before a move to that of
/// the position after it, in both frames, for every kind of move.
///
/// The scan is the oracle and is not trusted to be right about the delta's
/// order, only about its content: the removed tokens are all present before the
/// move, and applying them and the added ones leaves exactly the tokens a fresh
/// scan of the new position lists.
#[test]
fn the_delta_takes_the_tokens_before_a_move_to_the_tokens_after_it() {
    let mut seen = Seen::default();
    for (pos, mv) in games(12, 300) {
        let delta = MoveDelta::from_move(&pos, mv);
        let mut after = pos.clone();
        let _ = after.do_move(mv);

        if mv.is_drop() {
            seen.drops += 1;
            if pos.hand(pos.side_to_move()).count(mv.dropped_piece_kind()) > 1 {
                seen.drops_of_a_later_copy += 1;
            }
        } else {
            seen.promotions += usize::from(mv.is_promote());
            seen.king_moves += usize::from(mv.moved_piece_after().kind == PieceKind::King);
            match pos.board().get(mv.to_sq()) {
                None => seen.quiet += 1,
                Some(victim) => {
                    seen.captures += 1;
                    seen.captures_of_promoted += usize::from(victim.promoted);
                }
            }
        }

        for frame in [Color::Black, Color::White] {
            let half = delta.half(frame).expect("no half is ever rebuilt");
            assert_eq!(half.removed().len(), half.added().len());
            assert!(half.removed().len() <= 2);

            let mut tokens = active_features(&pos, frame);
            for token in half.removed() {
                let at = tokens
                    .iter()
                    .position(|t| t == token)
                    .unwrap_or_else(|| panic!("{token} is removed but was not there"));
                tokens.swap_remove(at);
            }
            tokens.extend_from_slice(half.added());
            assert_eq!(
                sorted(tokens),
                sorted(active_features(&after, frame)),
                "{} {frame:?} {mv:?}",
                format_sfen(&pos),
            );
        }
    }
    for (what, n) in [
        ("quiet moves", seen.quiet),
        ("captures", seen.captures),
        ("captures of a promoted piece", seen.captures_of_promoted),
        ("promotions", seen.promotions),
        ("drops", seen.drops),
        ("drops of a later copy", seen.drops_of_a_later_copy),
        ("king moves", seen.king_moves),
    ] {
        assert!(n > 0, "the games met no {what}: {seen:?}");
    }
}

/// A king move is the same one token out, one in as any other. HalfKP is the
/// space that cannot say this, and this one has to.
#[test]
fn a_king_move_is_an_ordinary_delta() {
    let pos = parse_sfen(STARTPOS).expect("a valid sfen");
    // 5i-5h is not legal at the opening, so walk a pawn out of the way first.
    let mut pos = pos;
    for usi in ["7g7f", "3c3d", "6i7h", "3d3e"] {
        let mv = attic_state::parse_usi_move(usi, &pos).expect("a legal move");
        let _ = pos.do_move(mv);
    }
    let mv = attic_state::parse_usi_move("5i6h", &pos).expect("a legal king move");
    assert_eq!(pos.board().get(mv.from_sq()).unwrap().kind, PieceKind::King);
    assert!(!requires_full_refresh(mv, Color::Black));
    assert!(!requires_full_refresh(mv, Color::White));
    let delta = MoveDelta::from_move(&pos, mv);
    for frame in [Color::Black, Color::White] {
        let half = delta.half(frame).unwrap();
        assert_eq!((half.removed().len(), half.added().len()), (1, 1));
    }
}

/// The white frame's tokens are the black frame's read through `other_frame`,
/// in the same order, and so is each half of every move's delta. A pair table
/// read for both frames from the black frame's list relies on the order as well
/// as on the rows. Over random games and the positions above, and the mapping is
/// its own inverse on every row.
#[test]
fn the_white_frame_is_the_black_frame_read_through_other_frame_in_order() {
    for row in 0..crate::types::NUM_FEATURES as FeatureIndex {
        assert_eq!(other_frame(other_frame(row)), row, "row {row}");
    }
    let fixed = [STARTPOS, DROP_HEAVY, MIXED, EVERY_PROMOTION, FULL_HANDS]
        .map(|sfen| parse_sfen(sfen).expect("a valid sfen"));
    for pos in &fixed {
        let black: Vec<FeatureIndex> = active_features(pos, Color::Black)
            .into_iter()
            .map(other_frame)
            .collect();
        assert_eq!(
            black,
            active_features(pos, Color::White),
            "{}",
            format_sfen(pos)
        );
    }
    for (pos, mv) in games(6, 160) {
        let black: Vec<FeatureIndex> = active_features(&pos, Color::Black)
            .into_iter()
            .map(other_frame)
            .collect();
        assert_eq!(
            black,
            active_features(&pos, Color::White),
            "{}",
            format_sfen(&pos)
        );
        let delta = MoveDelta::from_move(&pos, mv);
        let (b, w) = (
            delta.half(Color::Black).expect("a token delta"),
            delta.half(Color::White).expect("a token delta"),
        );
        let read =
            |list: &[FeatureIndex]| list.iter().copied().map(other_frame).collect::<Vec<_>>();
        assert_eq!(read(b.removed()), w.removed(), "{}", format_sfen(&pos));
        assert_eq!(read(b.added()), w.added(), "{}", format_sfen(&pos));
    }
}
