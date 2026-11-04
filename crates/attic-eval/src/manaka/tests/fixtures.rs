//! What the tests share: the exporter's own strings, a network no trainer made,
//! and that network written out as a file.

use std::sync::OnceLock;

use attic_state::{Move, Position};

use super::super::loader::{
    EMBED_SCALE, Encoding, VERSION, WEIGHT_SCALE, arch_hash, expected_arch,
};
use super::super::{
    DIM, FEATURE_LEN, HEAD, HIDDEN, HIDDEN2, ORDER, SECOND_LAYER, SymNetwork, TOKENS,
};

/// What Manaka's exporter (`sym_arch_string`) wrote for this shape: the string
/// without its `ModelType` word, and the file hash it stored for each of the
/// three encodings. Copied from real exports, not from `expected_arch`:
/// asserting a function against itself would pass for any string.
///
/// The d64 / d256 hashes are from the 34-shape export of `atmos-34`; the d128
/// ones are from files the same exporter wrote for a randomly initialised net,
/// since `atmos-34` has no d128 KP or HalfKP shape.
pub(super) struct Exported {
    /// Everything after `ModelType=<word>;Features=`.
    pub(super) rest: &'static str,
    pub(super) scaled: u32,
    pub(super) f32: u32,
    pub(super) legacy: u32,
}

#[cfg(feature = "sym_kp_64x2_512")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "K+P[1710->64x2],Network=Dense[1<-512](ReLU[512](Dense[512<-384](RmsNormScaled[192](Newton[3](PowerSum[3](InputSlice[192(0:192)])))))",
    scaled: 0xf6b9_acc7,
    f32: 0x6e54_0c82,
    legacy: 0x0ff7_e8cd,
};
#[cfg(feature = "sym_halfkp_64x2_512")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "HalfKP(Friend)[125389->64x2],Network=Dense[1<-512](ReLU[512](Dense[512<-384](RmsNormScaled[192](Newton[3](PowerSum[3](InputSlice[192(0:192)])))))",
    scaled: 0xcf53_3409,
    f32: 0x6854_422a,
    legacy: 0x6f7e_587b,
};
#[cfg(feature = "sym_kp_128x2_512")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "K+P[1710->128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-768](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)])))))",
    scaled: 0xbaff_2ef9,
    f32: 0xe700_9396,
    legacy: 0x4c53_92cf,
};
#[cfg(feature = "sym_halfkp_128x2_512")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "HalfKP(Friend)[125389->128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-768](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)])))))",
    scaled: 0x3798_f7d3,
    f32: 0x1832_185e,
    legacy: 0x70a3_ccf5,
};
#[cfg(feature = "sym_kp_256x2_512")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "K+P[1710->256x2],Network=Dense[1<-512](ReLU[512](Dense[512<-1536](RmsNormScaled[768](Newton[3](PowerSum[3](InputSlice[768(0:768)])))))",
    scaled: 0x4691_8805,
    f32: 0xac38_e330,
    legacy: 0x7cfa_4607,
};
#[cfg(feature = "sym_halfkp_256x2_512")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "HalfKP(Friend)[125389->256x2],Network=Dense[1<-512](ReLU[512](Dense[512<-1536](RmsNormScaled[768](Newton[3](PowerSum[3](InputSlice[768(0:768)])))))",
    scaled: 0xfe57_24cb,
    f32: 0xc30f_72e8,
    legacy: 0x779a_cf31,
};
#[cfg(feature = "sym_kp_256x2_o1_32_32")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "K+P[1710->256x2],Network=Dense[1<-32](ReLU[32](Dense[32<-32](ReLU[32](Dense[32<-512](RmsNormScaled[256](Newton[1](PowerSum[1](InputSlice[256(0:256)])))))))",
    scaled: 0x5eb6_71aa,
    f32: 0x2df6_cba1,
    legacy: 0x0f12_1c44,
};
#[cfg(feature = "sym_halfkp_256x2_o1_32_32")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "HalfKP(Friend)[125389->256x2],Network=Dense[1<-32](ReLU[32](Dense[32<-32](ReLU[32](Dense[32<-512](RmsNormScaled[256](Newton[1](PowerSum[1](InputSlice[256(0:256)])))))))",
    scaled: 0x079b_c6a8,
    f32: 0xf554_4019,
    legacy: 0xc1e0_8026,
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
        }
    })
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
pub(super) fn column_scales(values: &[f32], cols: usize) -> Vec<f32> {
    (0..cols)
        .map(|c| group_scale(values.iter().skip(c).step_by(cols)))
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
        }
        Encoding::F32 => {
            put_f32s(&mut out, &net.w1);
            put_f32s(&mut out, &net.b1);
            put_f32s(&mut out, &net.w2);
            put_f32s(&mut out, &net.b2);
            put_f32s(&mut out, &net.value_w);
            put_f32s(&mut out, &[net.value_b]);
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

/// A small deterministic generator (xorshift64*), so a failure is the same
/// failure on every run and every machine.
struct Rng(u64);

impl Rng {
    /// Uniform in `[-amplitude, amplitude)`.
    fn next(&mut self, amplitude: f32) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
        (bits as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amplitude
    }

    fn fill(&mut self, len: usize, amplitude: f32) -> Vec<f32> {
        (0..len).map(|_| self.next(amplitude)).collect()
    }
}
