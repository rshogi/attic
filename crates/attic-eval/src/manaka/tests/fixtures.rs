//! What the tests share: a network no trainer made, and the positions to
//! evaluate it on. The exporter's own strings are in `exported.rs` and the file
//! the network is written to is `encode.rs`'s.

use std::sync::OnceLock;

use attic_state::{Move, Position};

use super::super::{
    DIM, FEATURE_LEN, FIELD, FIELD_LEN, FirstLayer, HEAD, HIDDEN, HIDDEN2, MOMENT, MOMENT_LEN,
    ORDER, PAIR, PAIR_ROWS, PairTable, SymNetwork, TOKENS,
};

/// The synthetic pair table from values drawn in `f32`: rounded to halves the way
/// the exporter rounds them, which is how a loaded network keeps its table.
fn pair_table(values: &[f32]) -> PairTable {
    #[cfg(feature = "sym_token")]
    {
        use super::encode::to_half;
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| to_half(v).to_le_bytes())
            .collect();
        PairTable::from_le_bytes(&bytes)
    }
    #[cfg(not(feature = "sym_token"))]
    {
        assert!(
            values.is_empty(),
            "a pair table exists only in the token space"
        );
        PairTable::empty()
    }
}

/// A network no trainer made.
///
/// A block is only multiplied by its gain and clipped, and the sums here are in
/// the units, so the gains and the first dense layer are drawn small: at gains
/// around one every block would sit on a rail, and a clip that is always on a rail
/// is a clip no test can see. The factors are in [`Gain`], found by looking at
/// where the sums land (see
/// `a_test_net_exercises_the_clip_in_every_block_and_layer`).
pub(super) fn synthetic() -> &'static SymNetwork {
    static NET: OnceLock<SymNetwork> = OnceLock::new();
    NET.get_or_init(|| {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        SymNetwork {
            embed: rng.fill(TOKENS * DIM, 0.6),
            gamma: rng
                .fill(ORDER * DIM, 0.4)
                .iter()
                .enumerate()
                .map(|(i, g)| (1.0 + g) * Gain::order(i / DIM))
                .collect(),
            w1: rng.fill(HIDDEN * FEATURE_LEN, 0.3 * Gain::W1),
            b1: rng.fill(HIDDEN, 0.3),
            w2: rng.fill(HIDDEN2 * HIDDEN, 0.3 * Gain::W2),
            b2: rng.fill(HIDDEN2, 0.3),
            value_w: rng.fill(HEAD, 0.3),
            value_b: 0.1,
            pair_gamma: gains(&mut rng, PAIR, Gain::PAIR),
            pair: pair_table(&rng.fill(PAIR_ROWS * PAIR, 0.3)),
            // Drawn last, so that the networks of the shapes that have none of
            // these are the ones they were before there were any.
            moment_gamma: gains(&mut rng, MOMENT_LEN, Gain::MOMENT),
            moment_embed: rng.fill(TOKENS * MOMENT, 0.6),
            field_gamma: gains(&mut rng, FIELD, Gain::FIELD),
            field: rng.fill(TOKENS * FIELD_LEN, 0.3),
            first: FirstLayer::F32,
        }
    })
}

/// The factor the test net takes off a unit gain, or puts on a layer's weights,
/// for each kind of block.
struct Gain;

impl Gain {
    const W1: f32 = 0.3;
    /// The second layer reads at most `HIDDEN` activations in `[0, 1]`, where the
    /// first reads hundreds, so it wants more weight to reach a rail.
    const W2: f32 = 2.0;
    const PAIR: f32 = 0.1;
    const MOMENT: f32 = 0.2;
    const FIELD: f32 = 0.6;

    /// The power-sum block of order `k + 1`: the sums grow with the order.
    fn order(k: usize) -> f32 {
        [0.2, 0.1, 0.1, 0.05][k.min(3)]
    }
}

/// `0..n`, for an `n` that is a shape's constant and is zero in the shapes that
/// lack the block. Written as `0..FIELD` that is a range clippy sees is empty in
/// those shapes, and refuses; through a function argument it is a number it does
/// not know.
pub(super) fn upto(n: usize) -> std::ops::Range<usize> {
    0..n
}

/// The smaller of two shape constants. Written as `HIDDEN.min(HIDDEN2)` it is a
/// comparison clippy sees is decided at compile time, and refuses; through a
/// function argument it is a number it does not know.
pub(super) fn least(a: usize, b: usize) -> usize {
    a.min(b)
}

/// `len` gains around one, as a trained norm's are, times `scale`.
fn gains(rng: &mut Rng, len: usize, scale: f32) -> Vec<f32> {
    rng.fill(len, 0.4)
        .iter()
        .map(|g| (1.0 + g) * scale)
        .collect()
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

/// Every token of `pos` as `frame` reads it.
///
/// Derived from the position (the piece on each square, the copies in each hand)
/// and not from `active_features`, so that what the accumulator reads can be
/// checked against a second route to the same set.
#[cfg(feature = "sym_token")]
pub(super) fn tokens_present(pos: &Position, frame: attic_state::Color) -> Vec<u32> {
    use crate::tokens::{HAND_KINDS, board_row, hand_row};
    use attic_state::{Color, Square};

    let mut tokens = Vec::new();
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).expect("a square");
        if let Some(piece) = pos.board().get(sq) {
            tokens.push(board_row(piece, sq, frame));
        }
    }
    for owner in [Color::Black, Color::White] {
        for kind in HAND_KINDS {
            for copy in 0..usize::from(pos.hand(owner).count(kind)) {
                tokens.push(hand_row(owner, kind, copy, frame));
            }
        }
    }
    tokens
}

/// The KP and HalfKP shapes number a position's features in the one place there
/// is for them, `active_features`.
#[cfg(not(feature = "sym_token"))]
pub(super) fn tokens_present(pos: &Position, frame: attic_state::Color) -> Vec<u32> {
    crate::active_features(pos, frame)
}

/// [`assert_close`] with a tolerance that grows with the size of the numbers: a
/// sum of 780 pair rows is large, and the rounding of an `f32` there is a
/// fraction of its own size, which an absolute bound meant for sums of order one
/// cannot be right for.
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
