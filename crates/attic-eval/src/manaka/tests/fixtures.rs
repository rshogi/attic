//! What the tests share: the exporter's own strings, a network no trainer made,
//! and that network written out as a file.

use std::sync::OnceLock;

use attic_state::{Move, Position};

use super::super::forward::LSE_FLOOR;
use super::super::loader::{
    EMBED_SCALE, Encoding, VERSION, WEIGHT_SCALE, arch_hash, expected_arch,
};
use super::super::{
    DIM, FEATURE_LEN, HAS_PAIR, HEAD, HIDDEN, HIDDEN2, KS, MEAN, ORDER, PAIR, PAIR_ROWS, SCALE_LEN,
    SECOND_LAYER, SymNetwork, TOKENS,
};

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
        }
    })
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

/// The column an element of a row-major array of `width` columns is in. The
/// width is an argument for the reason [`map_channels`]'s is: `i % PAIR` written
/// out is a remainder by zero in a shape without a pair table, which rustc
/// refuses to compile even in a branch that shape never takes.
pub(super) fn column_of(width: usize) -> impl Fn(usize) -> usize {
    move |i| i % width
}

/// `f` applied to the channels from `first` on of every `width`-wide row of
/// `table`. The width is an argument and not the constant it always is, because
/// a shape without a pair table is compiled with this in it and clippy refuses a
/// constant chunk size of zero.
pub(super) fn map_channels(table: &mut [f32], width: usize, first: usize, f: impl Fn(f32) -> f32) {
    // A table that is off has no rows, and `chunks_exact` refuses a width of zero.
    if width == 0 {
        return;
    }
    for row in table.chunks_exact_mut(width) {
        for v in &mut row[first..] {
            *v = f(*v);
        }
    }
}

/// The encoded file of the synthetic net, once per encoding. A pair table makes
/// each of these tens to hundreds of megabytes, and nearly every test wants one.
pub(super) fn encoded(encoding: Encoding) -> &'static [u8] {
    static FILES: [OnceLock<Vec<u8>>; 3] = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
    let slot = Encoding::ALL
        .iter()
        .position(|e| *e == encoding)
        .expect("a known encoding");
    FILES[slot].get_or_init(|| encode(synthetic(), encoding))
}

/// The scale `export-nnue` gives one group of a `SYMscaled` array: the largest
/// `f32` for which the group's largest element still fits `i16`, so that
/// element lands on ±32767 and nothing clamps. An all-zero group has scale 1.
pub(super) fn group_scale<'a>(group: impl Iterator<Item = &'a f32>) -> f32 {
    let max = group.fold(0.0f32, |m, v| m.max(v.abs()));
    if max == 0.0 {
        return 1.0;
    }
    let mut scale = 32767.0 / max;
    while max * scale > 32767.0 {
        scale = f32::from_bits(scale.to_bits() - 1);
    }
    scale
}

/// The scales of a `rows × cols` array grouped by column (`i mod cols`).
///
/// One pass over the rows rather than a strided pass per column: the pair table
/// is `PAIR` columns of 2.7 million, and walking it `PAIR` times is the slowest
/// thing a test here would do.
pub(super) fn column_scales(values: &[f32], cols: usize) -> Vec<f32> {
    let mut largest = vec![0.0f32; cols];
    for row in values.chunks_exact(cols) {
        for (max, v) in largest.iter_mut().zip(row) {
            *max = max.max(v.abs());
        }
    }
    largest
        .into_iter()
        .map(|max| group_scale(std::iter::once(&max)))
        .collect()
}

/// The scales of a `rows × cols` array grouped by row (`i div cols`).
pub(super) fn row_scales(values: &[f32], cols: usize) -> Vec<f32> {
    values
        .chunks_exact(cols)
        .map(|r| group_scale(r.iter()))
        .collect()
}

fn put_f32s(out: &mut Vec<u8>, values: &[f32]) {
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// A `SYMscaled` array: its scales, then `round(x · scale)` as `i16`, half away
/// from zero. Nothing at all for an empty array.
fn put_scaled(out: &mut Vec<u8>, values: &[f32], scales: &[f32], group: impl Fn(usize) -> usize) {
    if values.is_empty() {
        return;
    }
    put_f32s(out, scales);
    for (i, v) in values.iter().enumerate() {
        let q = (v * scales[group(i)]).round();
        assert!(
            q.abs() <= 32767.0,
            "a scale that clamps is not the exporter's"
        );
        out.extend_from_slice(&(q as i16).to_le_bytes());
    }
}

/// `net` written the way `export-nnue` writes it in `encoding`.
pub(super) fn encode(net: &SymNetwork, encoding: Encoding) -> Vec<u8> {
    let arch = expected_arch(encoding);
    let hash = arch_hash(&arch);
    let mut out = Vec::new();
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&hash.to_le_bytes());
    out.extend_from_slice(&(arch.len() as u32).to_le_bytes());
    out.extend_from_slice(arch.as_bytes());

    out.extend_from_slice(&hash.wrapping_add(1).to_le_bytes());
    match encoding {
        Encoding::Scaled => {
            put_scaled(&mut out, &net.embed, &column_scales(&net.embed, DIM), |i| {
                i % DIM
            })
        }
        Encoding::F32 => put_f32s(&mut out, &net.embed),
        Encoding::Legacy => {
            for v in &net.embed {
                out.extend_from_slice(&((v * EMBED_SCALE).round() as i16).to_le_bytes());
            }
        }
    }

    out.extend_from_slice(&hash.wrapping_add(2).to_le_bytes());
    put_f32s(&mut out, &net.gamma);
    match encoding {
        Encoding::Scaled => {
            put_scaled(&mut out, &net.w1, &row_scales(&net.w1, FEATURE_LEN), |i| {
                i / FEATURE_LEN
            });
            put_f32s(&mut out, &net.b1);
            if SECOND_LAYER {
                put_scaled(&mut out, &net.w2, &row_scales(&net.w2, HIDDEN), |i| {
                    i / HIDDEN
                });
            }
            put_f32s(&mut out, &net.b2);
            put_f32s(&mut out, &net.value_w);
            put_f32s(&mut out, &[net.value_b]);
            if HAS_PAIR {
                out.extend_from_slice(&hash.wrapping_add(3).to_le_bytes());
                put_f32s(&mut out, &net.pair_gamma);
                // `θ`, not `exp θ`: the reader exponentiates the last channels.
                let mut theta = net.pair.clone();
                map_channels(&mut theta, PAIR, MEAN, |v| v.max(LSE_FLOOR).ln());
                let scales = column_scales(&theta, PAIR);
                put_scaled(&mut out, &theta, &scales, column_of(PAIR));
            }
        }
        Encoding::F32 => {
            put_f32s(&mut out, &net.w1);
            put_f32s(&mut out, &net.b1);
            put_f32s(&mut out, &net.w2);
            put_f32s(&mut out, &net.b2);
            put_f32s(&mut out, &net.value_w);
            put_f32s(&mut out, &[net.value_b]);
            if HAS_PAIR {
                out.extend_from_slice(&hash.wrapping_add(3).to_le_bytes());
                put_f32s(&mut out, &net.pair_gamma);
                put_f32s(&mut out, &net.pair);
            }
        }
        Encoding::Legacy => {
            let i8s = |out: &mut Vec<u8>, values: &[f32]| {
                for v in values {
                    // `as` saturates, which is the exporter's clamp.
                    out.push((v * WEIGHT_SCALE).round() as i8 as u8);
                }
            };
            let i32s = |out: &mut Vec<u8>, values: &[f32]| {
                for v in values {
                    out.extend_from_slice(&((v * WEIGHT_SCALE).round() as i32).to_le_bytes());
                }
            };
            i8s(&mut out, &net.w1);
            i32s(&mut out, &net.b1);
            i8s(&mut out, &net.w2);
            i32s(&mut out, &net.b2);
            i8s(&mut out, &net.value_w);
            i32s(&mut out, &[net.value_b]);
        }
    }
    if KS && encoding != Encoding::Legacy {
        out.extend_from_slice(&hash.wrapping_add(6).to_le_bytes());
        put_f32s(&mut out, &net.scale_gamma);
    }
    out
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
