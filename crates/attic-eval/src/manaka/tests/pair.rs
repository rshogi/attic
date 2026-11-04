//! The token-pair helpers against the definition: which row a pair is, and
//! which group a token's number puts it in.

use attic_state::{Color, Position, parse_sfen};

use super::super::pair::{pair_row, theirs};
use super::super::{HAS_PAIR, PAIR_ROWS, TOKENS};
use super::fixtures::owned_tokens;

/// The same three positions `tokens/tests.rs` uses for the numbering, for the
/// reason it does: between them they put every promoted kind of both colours on
/// the board, fill both hands to the most there can be of every kind, and
/// include an ordinary middlegame.
const POSITIONS: [&str; 3] = [
    "+p+l+n+s+b+r1k1/9/9/9/9/9/9/9/+P+L+N+S+B+R1K1 b - 1",
    "4k4/9/9/9/9/9/9/9/4K4 b 18P4L4N4S4G2B2R18p4l4n4s4g2b2r 1",
    "ln2b4/5skgl/p4pn1r/2+P1p2pp/1sP3p2/2GSKP1GP/P1NPS2PN/1+pRpP2+bL/L4G3 w Pp 94",
];

/// The pair rows are the upper triangle read row by row, so walking it in that
/// order must count 0, 1, 2, … without a gap or a repeat, to exactly `C(2344, 2)`,
/// whichever way round the two tokens are given.
#[test]
fn pair_rows_number_the_upper_triangle_in_order() {
    let mut next = 0;
    for lo in 0..TOKENS as u32 {
        for hi in lo + 1..TOKENS as u32 {
            assert_eq!(pair_row(lo, hi), next, "({lo}, {hi})");
            assert_eq!(pair_row(hi, lo), next, "({hi}, {lo})");
            next += 1;
        }
    }
    assert_eq!(next, 2_745_996);
    // The table's own size, in a shape that has one; the others have no rows.
    assert_eq!(PAIR_ROWS, if HAS_PAIR { next } else { 0 });
}

/// A token's group is whose it is, read from its number: the frame's own side is
/// 0 and the opponent's is 1, for board pieces of every kind and for hand copies.
#[test]
fn a_token_number_says_whose_it_is() {
    for sfen in POSITIONS {
        let pos = parse_sfen(sfen).expect("a valid sfen");
        for frame in [Color::Black, Color::White] {
            for (token, opponents) in owned_tokens(&pos, frame) {
                assert_eq!(theirs(token), usize::from(opponents), "{sfen}: {token}");
            }
        }
    }
    let pos = Position::startpos();
    let own: usize = owned_tokens(&pos, Color::Black)
        .iter()
        .filter(|(token, _)| theirs(*token) == 0)
        .count();
    assert_eq!(own, 20, "twenty pieces are the frame's own at the start");
}
