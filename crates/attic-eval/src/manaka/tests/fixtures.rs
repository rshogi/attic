//! What the tests share: a network no trainer made, and the positions to
//! evaluate it on. The exporter's own strings are in `exported.rs` and the file
//! the network is written to is `encode.rs`'s.

use std::sync::OnceLock;

use attic_state::{Move, Position};

use super::super::{
    DIM, FEATURE_LEN, FIELD, FIELD_LEN, HEAD, HIDDEN, HIDDEN2, MEAN, MOMENT, MOMENT_LEN, ORDER,
    PAIR, PAIR_ROWS, SCALE_LEN, SymNetwork, TOKENS,
};
use super::encode::map_channels;

/// A network no trainer made, with weights small enough to stay inside the
/// range the legacy encoding holds (`i8` at 64 reaches ±1.98), so that every
/// encoding can write it without a clamp.
pub(super) fn synthetic() -> &'static SymNetwork {
    static NET: OnceLock<SymNetwork> = OnceLock::new();
    NET.get_or_init(|| {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        SymNetwork {
            embed: rng.fill(TOKENS * DIM, 0.6),
            gamma: rng.fill(ORDER * DIM, 0.4).iter().map(|g| 1.0 + g).collect(),
            w1: rng.fill(HIDDEN * FEATURE_LEN, 0.3),
            b1: rng.fill(HIDDEN, 0.3),
            w2: rng.fill(HIDDEN2 * HIDDEN, 0.3),
            b2: rng.fill(HIDDEN2, 0.3),
            value_w: rng.fill(HEAD, 0.3),
            value_b: 0.1,
            pair_gamma: rng.fill(PAIR, 0.4).iter().map(|g| 1.0 + g).collect(),
            pair: pair_table(&mut rng),
            scale_gamma: rng.fill(SCALE_LEN, 0.4),
            // Drawn last, so that the networks of the shapes that have none of
            // these are the ones they were before there were any.
            moment_gamma: gains(&mut rng, MOMENT_LEN),
            moment_embed: rng.fill(TOKENS * MOMENT, 0.6),
            field_gamma: gains(&mut rng, FIELD),
            field: rng.fill(TOKENS * FIELD_LEN, 0.3),
        }
    })
}

/// `0..n`, for an `n` that is a shape's constant and is zero in the shapes that
/// lack the block. Written as `0..FIELD` that is a range clippy sees is empty in
/// those shapes, and refuses; through a function argument it is a number it does
/// not know.
pub(super) fn upto(n: usize) -> std::ops::Range<usize> {
    0..n
}

/// `len` gains around one, as a trained norm's are.
fn gains(rng: &mut Rng, len: usize) -> Vec<f32> {
    rng.fill(len, 0.4).iter().map(|g| 1.0 + g).collect()
}

/// A pair table: `PAIR_ROWS` rows of ±0.3 in the mean channels and, in the last
/// `LSE` channels, `exp θ` for a `θ` spread over ±1.5 — positive, as a trained
/// table's are, and wide enough that a sum of 780 of them is a few hundred and
/// its logarithm is not flat. Empty without a pair table.
fn pair_table(rng: &mut Rng) -> Vec<f32> {
    let mut table = rng.fill(PAIR_ROWS * PAIR, 0.3);
    map_channels(&mut table, PAIR, MEAN, |v| (v * 5.0).exp());
    table
}

pub(super) fn assert_close(what: &str, got: &[f32], want: &[f32], tolerance: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    assert!(worst <= tolerance, "{what}: off by {worst}");
}

/// A sequence of legal moves from the opening, the same every run, each with
/// the position it was played from.
pub(super) fn game(plies: usize) -> Vec<(Position, Move)> {
    let mut pos = Position::startpos();
    let mut line = Vec::new();
    let mut legal = Vec::new();
    for ply in 0..plies {
        legal.clear();
        pos.generate_legal_all(&mut legal);
        if legal.is_empty() {
            break;
        }
        let mv = legal[(ply * 7 + 3) % legal.len()];
        line.push((pos.clone(), mv));
        let _ = pos.do_move(mv);
    }
    line
}

/// Legal moves chosen at random from the opening, each with the position it was
/// played from. `game` above is the same line every run and stays out of most
/// captures and all drops in its first forty plies; this one reaches them, and
/// is still the same line for a given `seed`.
pub(super) fn random_game(seed: u64, plies: usize) -> Vec<(Position, Move)> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
    let mut pos = Position::startpos();
    let mut line = Vec::new();
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
    line
}

/// A small deterministic generator (xorshift64*), so a failure is the same
/// failure on every run and every machine.
struct Rng(u64);

impl Rng {
    fn advance(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `0..n`.
    fn below(&mut self, n: usize) -> usize {
        (self.advance() >> 33) as usize % n
    }

    /// Uniform in `[-amplitude, amplitude)`.
    fn next(&mut self, amplitude: f32) -> f32 {
        let bits = self.advance() >> 40;
        (bits as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amplitude
    }

    fn fill(&mut self, len: usize, amplitude: f32) -> Vec<f32> {
        (0..len).map(|_| self.next(amplitude)).collect()
    }
}

/// Every token of `pos` as `frame` reads it, each with whether the opponent of
/// the frame owns it.
///
/// The owner comes from the position (the piece's colour, the hand's owner) and
/// not from the token's number, so that what reads the number back out can be
/// checked against it. The shapes whose numbering says nothing about ownership
/// have no `split`, and report every token as the frame's own.
#[cfg(feature = "sym_token")]
pub(super) fn owned_tokens(pos: &Position, frame: attic_state::Color) -> Vec<(u32, bool)> {
    use crate::tokens::{HAND_KINDS, board_row, hand_row};
    use attic_state::{Color, Square};

    let mut tokens = Vec::new();
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).expect("a square");
        if let Some(piece) = pos.board().get(sq) {
            tokens.push((board_row(piece, sq, frame), piece.color != frame));
        }
    }
    for owner in [Color::Black, Color::White] {
        for kind in HAND_KINDS {
            for copy in 0..usize::from(pos.hand(owner).count(kind)) {
                tokens.push((hand_row(owner, kind, copy, frame), owner != frame));
            }
        }
    }
    tokens
}

/// The KP and HalfKP shapes have no ownership in their numbering and no `split`:
/// every token is the frame's own as far as a group is concerned.
#[cfg(not(feature = "sym_token"))]
pub(super) fn owned_tokens(pos: &Position, frame: attic_state::Color) -> Vec<(u32, bool)> {
    crate::active_features(pos, frame)
        .into_iter()
        .map(|token| (token, false))
        .collect()
}

/// [`assert_close`] with a tolerance that grows with the size of the numbers: a
/// sum of `exp θ` over 780 pairs is in the hundreds, and the rounding of an
/// `f32` there is a fraction of its own size, which an absolute bound meant for
/// sums of order one cannot be right for.
pub(super) fn assert_close_relative(what: &str, got: &[f32], want: &[f32], relative: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let allowed = relative * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= allowed,
            "{what}[{i}]: {g} against {w}, allowed {allowed}"
        );
    }
}
