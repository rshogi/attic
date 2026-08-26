//! PackedSfen — the 256-bit (32-byte) Huffman position encoding, ported
//! bit-for-bit from `SfenPacker::pack`.
//!
//! The `.ybb` opening-book index keys positions by this exact encoding, so any
//! single-bit divergence from the reference writer makes the index binary
//! search miss.
//!
//! Bits are packed least-significant-first within each byte, bytes ascending:
//! stream bit 0 is bit 0 (LSB) of `data[0]`. The trailing "piece box" pass —
//! every piece neither on the board nor in a hand — pads the stream so that any
//! position within the standard piece complement lands on exactly 256 bits.

use crate::color::Color;
use crate::piece::{Piece, PieceKind};
use crate::position::Position;
use crate::sfen::SfenError;
use crate::square::Square;

/// Length in bytes of a [`PackedSfen`].
pub const PACKED_SFEN_LEN: usize = 32;

/// A shogi position packed into 32 bytes, identical to YaneuraOu's `PackedSfen`.
///
/// The encoding covers side-to-move, both king squares, the board, and both
/// hands — but **not** the game ply. Two positions that differ only in ply pack
/// to identical bytes (the `.ybb` index carries ply separately).
pub type PackedSfen = [u8; PACKED_SFEN_LEN];

/// Piece kinds in the reference's "Apery" enumeration order, used for both the
/// hand and the piece-box passes (`to_apery_pieces[]`).
const HAND_ORDER: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

/// Starting piece-box counts, indexed by `PieceKind::index()` for
/// `Pawn..=Rook`. King is never boxed.
const PIECE_BOX_START: [i32; 7] = [18, 4, 4, 4, 4, 2, 2];

/// LSB-first bit writer over the fixed 32-byte buffer.
struct BitWriter {
    data: PackedSfen,
    cursor: usize,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            data: [0; PACKED_SFEN_LEN],
            cursor: 0,
        }
    }

    #[inline]
    fn write_one_bit(&mut self, b: bool) {
        let byte = self.cursor / 8;
        // An over-populated (illegal) position runs past 256 bits; dropping the
        // overflow keeps the encoder total, and its output is only meaningful
        // for legal inputs anyway.
        if b && byte < PACKED_SFEN_LEN {
            self.data[byte] |= 1 << (self.cursor & 7);
        }
        self.cursor += 1;
    }

    #[inline]
    fn write_n_bit(&mut self, d: u32, n: u32) {
        for i in 0..n {
            self.write_one_bit((d & (1 << i)) != 0);
        }
    }
}

/// LSB-first bit reader over the fixed 32-byte buffer, the mirror of
/// [`BitWriter`].
struct BitReader<'a> {
    data: &'a PackedSfen,
    cursor: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a PackedSfen) -> Self {
        Self { data, cursor: 0 }
    }

    #[inline]
    fn read_one_bit(&mut self) -> Result<bool, SfenError> {
        let byte = self.cursor / 8;
        if byte >= PACKED_SFEN_LEN {
            return Err(SfenError::PackedTruncated);
        }
        let bit = (self.data[byte] >> (self.cursor & 7)) & 1;
        self.cursor += 1;
        Ok(bit != 0)
    }

    #[inline]
    fn read_n_bit(&mut self, n: u32) -> Result<u32, SfenError> {
        let mut value = 0;
        for i in 0..n {
            if self.read_one_bit()? {
                value |= 1 << i;
            }
        }
        Ok(value)
    }
}

/// `(code, bits)` for the on-board / hand Huffman code of a raw piece kind
/// (`huffman_table[]`).
fn huffman_board(kind: PieceKind) -> (u32, u32) {
    match kind {
        PieceKind::Pawn => (0x01, 2),
        PieceKind::Lance => (0x03, 4),
        PieceKind::Knight => (0x0b, 4),
        PieceKind::Silver => (0x07, 4),
        PieceKind::Gold => (0x0f, 5),
        PieceKind::Bishop => (0x1f, 6),
        PieceKind::Rook => (0x3f, 6),
        PieceKind::King => unreachable!("king is not Huffman-coded as a board/hand piece"),
    }
}

/// `(code, bits)` for the piece-box Huffman code of a raw piece kind
/// (`huffman_table_piecebox[]`).
fn huffman_piece_box(kind: PieceKind) -> (u32, u32) {
    match kind {
        PieceKind::Pawn => (0x02, 2),
        PieceKind::Lance => (0x09, 4),
        PieceKind::Knight => (0x0d, 4),
        PieceKind::Silver => (0x0b, 4),
        PieceKind::Gold => (0x1b, 5),
        PieceKind::Bishop => (0x2f, 6),
        PieceKind::Rook => (0x3f, 6),
        PieceKind::King => unreachable!("king is never in the piece box"),
    }
}

/// A board piece, Huffman-coded (`write_board_piece_to_stream`): code, then a
/// promote bit (except Gold), then a color bit.
fn write_board_piece(w: &mut BitWriter, piece: Piece) {
    let (code, bits) = huffman_board(piece.kind);
    w.write_n_bit(code, bits);
    if piece.kind != PieceKind::Gold {
        w.write_one_bit(piece.promoted);
    }
    w.write_one_bit(piece.color == Color::White);
}

/// A hand piece, Huffman-coded (`write_hand_piece_to_stream`): the board code
/// with its low bit dropped, then a forced-unpromoted bit (except Gold), then a
/// color bit.
fn write_hand_piece(w: &mut BitWriter, kind: PieceKind, color: Color) {
    let (code, bits) = huffman_board(kind);
    w.write_n_bit(code >> 1, bits - 1);
    if kind != PieceKind::Gold {
        w.write_one_bit(false);
    }
    w.write_one_bit(color == Color::White);
}

/// A piece-box piece, Huffman-coded (`write_piecebox_piece_to_stream`): the
/// piece-box code, then a zero color bit (except Gold, which encodes its color
/// implicitly).
fn write_piece_box_piece(w: &mut BitWriter, kind: PieceKind) {
    let (code, bits) = huffman_piece_box(kind);
    w.write_n_bit(code, bits);
    if kind != PieceKind::Gold {
        w.write_one_bit(false);
    }
}

/// Locate a color's king by scanning the board in index order; `SQ_NB` (81) if
/// the king is absent (mirrors `pos.square<KING>(c)` returning `SQ_NB`).
fn king_square(pos: &Position, color: Color) -> u32 {
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).expect("index < COUNT");
        if let Some(p) = pos.board().get(sq)
            && p.kind == PieceKind::King
            && p.color == color
        {
            return sq.index() as u32;
        }
    }
    Square::COUNT as u32
}

/// Pack a [`Position`] into its 32-byte [`PackedSfen`], bit-identical to the
/// reference `Position::sfen_pack`.
pub fn sfen_pack(pos: &Position) -> PackedSfen {
    let mut w = BitWriter::new();

    w.write_one_bit(pos.side_to_move() == Color::White);

    w.write_n_bit(king_square(pos, Color::Black), 7);
    w.write_n_bit(king_square(pos, Color::White), 7);

    // Kings carry their square above, so the board pass skips them.
    let mut box_count = PIECE_BOX_START;
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).expect("index < COUNT");
        match pos.board().get(sq) {
            Some(p) if p.kind == PieceKind::King => {},
            Some(p) => {
                write_board_piece(&mut w, p);
                box_count[p.kind.index()] -= 1;
            },
            None => w.write_one_bit(false),
        }
    }

    for color in [Color::Black, Color::White] {
        for kind in HAND_ORDER {
            let n = pos.hand(color).count(kind);
            for _ in 0..n {
                write_hand_piece(&mut w, kind, color);
            }
            box_count[kind.index()] -= i32::from(n);
        }
    }

    for kind in HAND_ORDER {
        let leftover = box_count[kind.index()].max(0);
        for _ in 0..leftover {
            write_piece_box_piece(&mut w, kind);
        }
    }

    debug_assert_eq!(
        w.cursor, 256,
        "packed sfen must be exactly 256 bits for a legal position"
    );
    w.data
}

/// Read one board square's Huffman code, `None` for an empty square.
///
/// The board codes form a prefix-free set that this walks one bit at a time
/// rather than matching against [`huffman_board`]: `0` empty, `10` Pawn,
/// `1100` Lance, `1101` Knight, `1110` Silver, `11110` Gold, `111110` Bishop,
/// `111111` Rook.
fn read_board_kind(r: &mut BitReader<'_>) -> Result<Option<PieceKind>, SfenError> {
    if !r.read_one_bit()? {
        return Ok(None);
    }
    let kind = if !r.read_one_bit()? {
        PieceKind::Pawn
    } else if !r.read_one_bit()? {
        if r.read_one_bit()? {
            PieceKind::Knight
        } else {
            PieceKind::Lance
        }
    } else if !r.read_one_bit()? {
        PieceKind::Silver
    } else if !r.read_one_bit()? {
        PieceKind::Gold
    } else if !r.read_one_bit()? {
        PieceKind::Bishop
    } else {
        PieceKind::Rook
    };
    Ok(Some(kind))
}

/// A board piece: the Huffman code, a promote bit (except Gold), a color bit.
fn read_board_piece(r: &mut BitReader<'_>) -> Result<Option<Piece>, SfenError> {
    let Some(kind) = read_board_kind(r)? else {
        return Ok(None);
    };
    let promoted = if kind == PieceKind::Gold {
        false
    } else {
        r.read_one_bit()?
    };
    let color = if r.read_one_bit()? {
        Color::White
    } else {
        Color::Black
    };
    let piece = if promoted {
        // Only promotable kinds carry a set promote bit, and `huffman_board`
        // never yields King, so this is unreachable for a stream our own writer
        // produced.
        Piece::promoted(kind, color).ok_or(SfenError::PackedPieceOverflow)?
    } else {
        Piece::new(kind, color)
    };
    Ok(Some(piece))
}

/// Where an off-board piece was found: a hand, or the piece box.
enum OffBoard {
    Hand(PieceKind, Color),
    Box(PieceKind),
}

/// Read one hand-or-piece-box entry.
///
/// The two passes share a bit stream with no length prefix between them, and
/// the reference's two tables are chosen so the union stays prefix-free once
/// the trailing promote/color bits are counted. Reading them together is what
/// makes the boundary unnecessary — see the table in `sfen_packer.cpp`.
fn read_off_board(r: &mut BitReader<'_>) -> Result<OffBoard, SfenError> {
    let color = |bit: bool| if bit { Color::White } else { Color::Black };

    if !r.read_one_bit()? {
        // 0 0 c — hand Pawn; 0 1 0 — boxed Pawn.
        return if r.read_one_bit()? {
            r.read_one_bit()?;
            Ok(OffBoard::Box(PieceKind::Pawn))
        } else {
            Ok(OffBoard::Hand(PieceKind::Pawn, color(r.read_one_bit()?)))
        };
    }
    if !r.read_one_bit()? {
        // 1 0 x — Lance or Knight, then a hand/box discriminator.
        let kind = if r.read_one_bit()? {
            PieceKind::Knight
        } else {
            PieceKind::Lance
        };
        return if r.read_one_bit()? {
            r.read_one_bit()?;
            Ok(OffBoard::Box(kind))
        } else {
            Ok(OffBoard::Hand(kind, color(r.read_one_bit()?)))
        };
    }
    if !r.read_one_bit()? {
        // 1 1 0 0 c — hand Silver; 1 1 0 1 0 — boxed Silver; 1 1 0 1 1 — boxed
        // Gold, which encodes no color bit of its own.
        return if !r.read_one_bit()? {
            Ok(OffBoard::Hand(PieceKind::Silver, color(r.read_one_bit()?)))
        } else if r.read_one_bit()? {
            Ok(OffBoard::Box(PieceKind::Gold))
        } else {
            Ok(OffBoard::Box(PieceKind::Silver))
        };
    }
    if !r.read_one_bit()? {
        // 1 1 1 0 c — hand Gold.
        return Ok(OffBoard::Hand(PieceKind::Gold, color(r.read_one_bit()?)));
    }
    // 1 1 1 1 x — Bishop or Rook, then a hand/box discriminator.
    let kind = if r.read_one_bit()? {
        PieceKind::Rook
    } else {
        PieceKind::Bishop
    };
    if r.read_one_bit()? {
        r.read_one_bit()?;
        Ok(OffBoard::Box(kind))
    } else {
        Ok(OffBoard::Hand(kind, color(r.read_one_bit()?)))
    }
}

/// Unpack a 32-byte [`PackedSfen`] into a [`Position`], the inverse of
/// [`sfen_pack`].
///
/// **The ply is not restored, because the encoding does not carry one.** The
/// returned position has the ply [`Position::empty`] starts with; a caller that
/// knows the move number sets it with
/// [`set_ply`](Position::set_ply).
///
/// The hand and piece-box passes have no length between them, so they are read
/// as one run that ends when every piece of the standard set is accounted for.
/// A stream claiming more pieces of a kind than exist yields
/// [`SfenError::PackedPieceOverflow`] rather than reading forever.
pub fn sfen_unpack(packed: &PackedSfen) -> Result<Position, SfenError> {
    let mut r = BitReader::new(packed);
    let mut pos = Position::empty();

    // 1. side to move
    let side = if r.read_one_bit()? {
        Color::White
    } else {
        Color::Black
    };

    // 2. king squares, Black then White. `Square::COUNT` means absent.
    let mut kings = [None; 2];
    for color in [Color::Black, Color::White] {
        let raw = r.read_n_bit(7)?;
        if raw == Square::COUNT as u32 {
            continue;
        }
        let index = u8::try_from(raw).map_err(|_| SfenError::PackedInvalidKingSquare(raw))?;
        let sq = Square::from_index(index).ok_or(SfenError::PackedInvalidKingSquare(raw))?;
        kings[color.index()] = Some(sq);
        pos.board_mut()
            .set(sq, Some(Piece::new(PieceKind::King, color)));
    }

    // 3. board pieces, kings skipped because step 2 already placed them
    let mut box_count = PIECE_BOX_START;
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).expect("index < COUNT");
        if kings.iter().any(|k| *k == Some(sq)) {
            continue;
        }
        if let Some(piece) = read_board_piece(&mut r)? {
            let slot = &mut box_count[piece.kind.index()];
            if *slot == 0 {
                return Err(SfenError::PackedPieceOverflow);
            }
            *slot -= 1;
            pos.board_mut().set(sq, Some(piece));
        }
    }

    // 4+5. hands and the piece box, as one run: whatever the board did not
    // claim is still out there, and the stream ends when it is all placed.
    while box_count.iter().sum::<i32>() > 0 {
        let entry = read_off_board(&mut r)?;
        let kind = match entry {
            OffBoard::Hand(kind, color) => {
                pos.hand_mut(color).increment(kind);
                kind
            },
            OffBoard::Box(kind) => kind,
        };
        let slot = &mut box_count[kind.index()];
        if *slot == 0 {
            return Err(SfenError::PackedPieceOverflow);
        }
        *slot -= 1;
    }

    pos.set_side_to_move(side);
    // The setters above bypass incremental key maintenance, exactly as
    // `parse_sfen` does; seed the keys once here.
    pos.refresh_keys();
    Ok(pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sfen::{format_sfen, parse_sfen};

    /// Parse an SFEN that may omit the trailing ply field (the reference's own
    /// PackedSfen test vectors are 3-field). Ply does not affect packing.
    fn parse(sfen: &str) -> Position {
        let with_ply = if sfen.split(' ').count() == 3 {
            format!("{sfen} 1")
        } else {
            sfen.to_string()
        };
        parse_sfen(&with_ply).expect("valid sfen")
    }

    /// Vectors transcribed from the reference's PackedSfen unit test, whose
    /// bytes were produced by cshogi (`board.to_psfen`) — an implementation
    /// independent of both.
    #[test]
    fn matches_reference_cshogi_vectors() {
        let cases: [(&str, [u8; 32]); 4] = [
            (
                "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w -",
                [
                    89, 164, 81, 34, 12, 171, 68, 252, 44, 167, 68, 56, 94, 137, 240, 72, 132, 87,
                    34, 60, 167, 68, 56, 86, 137, 248, 88, 70, 137, 48, 188, 126,
                ],
            ),
            (
                "lns1kgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w -",
                [
                    89, 164, 81, 34, 12, 171, 68, 252, 44, 167, 68, 56, 94, 137, 240, 72, 4, 18,
                    225, 57, 37, 194, 177, 74, 196, 199, 50, 74, 132, 97, 191, 126,
                ],
            ),
            (
                "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGK4 w -",
                [
                    89, 164, 81, 34, 88, 37, 226, 199, 41, 17, 188, 18, 129, 68, 120, 37, 194, 115,
                    74, 132, 99, 149, 136, 143, 101, 148, 8, 67, 106, 107, 191, 126,
                ],
            ),
            (
                "lnsgk4/9/ppppppppp/9/9/9/PPPPPPPPP/9/LNSGK4 w GBRgbr",
                [
                    89, 36, 18, 1, 137, 128, 68, 64, 34, 144, 8, 175, 68, 120, 78, 137, 112, 172,
                    18, 97, 25, 37, 194, 112, 30, 159, 251, 252, 166, 212, 218, 90,
                ],
            ),
        ];

        for (sfen, expected) in cases {
            let pos = parse(sfen);
            let packed = sfen_pack(&pos);
            assert_eq!(packed, expected, "packed sfen mismatch for {sfen}");
        }
    }

    #[test]
    fn ply_does_not_affect_packing() {
        let a =
            parse_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1").unwrap();
        let b =
            parse_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 99").unwrap();
        assert_eq!(sfen_pack(&a), sfen_pack(&b));
    }

    #[test]
    fn side_to_move_flips_bit0() {
        let black =
            parse_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1").unwrap();
        let white =
            parse_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1").unwrap();
        assert_eq!(sfen_pack(&black)[0] & 1, 0);
        assert_eq!(sfen_pack(&white)[0] & 1, 1);
    }

    /// Every sfen these tests use, in one place: the cshogi vectors plus the
    /// parity fixtures (promoted board pieces, both-color hands, sparse boards,
    /// a board missing its back rank).
    const ROUND_TRIP_SFENS: [&str; 10] = [
        "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "lns1kgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGK4 w - 1",
        "lnsgk4/9/ppppppppp/9/9/9/PPPPPPPPP/9/LNSGK4 w GBRgbr 1",
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",
        "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
        "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
        "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
        "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
    ];

    #[test]
    fn unpacking_a_packed_position_returns_it() {
        for sfen in ROUND_TRIP_SFENS {
            let pos = parse_sfen(sfen).expect("valid sfen");
            // Compare against what `format_sfen` makes of the input, not the
            // input itself: hands are written in a fixed order, so a vector
            // spelling them differently ("GBRgbr" for "RBGrbg") would fail here
            // over the spelling rather than over the round trip.
            let canonical = format_sfen(&pos);
            let restored = sfen_unpack(&sfen_pack(&pos)).expect("unpacks");
            assert_eq!(
                format_sfen(&restored),
                canonical,
                "round trip changed {sfen}"
            );
        }
    }

    #[test]
    fn unpacking_repacks_to_the_same_bytes() {
        // The stronger statement: whatever the decoder built packs back to the
        // bytes it was handed, so no information was dropped on the way in.
        for sfen in ROUND_TRIP_SFENS {
            let packed = sfen_pack(&parse_sfen(sfen).expect("valid sfen"));
            let restored = sfen_unpack(&packed).expect("unpacks");
            assert_eq!(sfen_pack(&restored), packed, "repack differs for {sfen}");
        }
    }

    #[test]
    fn the_ply_is_not_restored_because_it_was_never_stored() {
        let pos = parse_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 99")
            .expect("valid sfen");
        let restored = sfen_unpack(&sfen_pack(&pos)).expect("unpacks");
        assert_eq!(
            restored.ply(),
            Position::empty().ply(),
            "unpack must not invent a ply"
        );
    }

    #[test]
    fn an_all_ones_stream_is_rejected_rather_than_looping() {
        // Every bit set decodes as Rook forever, which overruns the piece box
        // before the buffer ends.
        let err = sfen_unpack(&[0xff; PACKED_SFEN_LEN]).expect_err("must not succeed");
        assert!(
            matches!(
                err,
                SfenError::PackedPieceOverflow | SfenError::PackedInvalidKingSquare(_)
            ),
            "unexpected error {err:?}"
        );
    }

    #[test]
    fn fixture_sfens_pack_to_full_width() {
        // Promoted board pieces, both-color hands and sparse boards must pack
        // without tripping the 256-bit debug assert; the cshogi vectors above
        // do not reach them.
        for sfen in [
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",
            "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
            "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
            "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
            "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
        ] {
            let pos = parse_sfen(sfen).unwrap();
            let packed = sfen_pack(&pos);
            assert_eq!(packed.len(), 32, "sfen {sfen}");
        }
    }
}
