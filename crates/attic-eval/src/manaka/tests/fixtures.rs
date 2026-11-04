//! What the tests share: the exporter's own strings, a network no trainer made,
//! and that network written out as a file.

use std::sync::OnceLock;

use attic_state::{Move, Position};

use super::super::loader::{EMBED_SCALE, VERSION, WEIGHT_SCALE, arch_hash, expected_arch};
use super::super::{DIM, FEATURE_LEN, HEAD, HIDDEN, HIDDEN2, ORDER, SymNetwork, TOKENS};

/// The architecture string Manaka's exporter (`sym_arch_string`) wrote for this
/// shape, and the file hash it wrote with it. Copied from real exports, not
/// from `expected_arch`: asserting a function against itself would pass for
/// any string.
#[cfg(feature = "sym_kp_64x2_512")]
pub(super) const EXPORTED: (&str, u32) = (
    "ModelType=SYM;Features=K+P[1710->64x2],Network=Dense[1<-512](ReLU[512](Dense[512<-384](RmsNormScaled[192](Newton[3](PowerSum[3](InputSlice[192(0:192)])))))",
    0x0ff7_e8cd,
);
#[cfg(feature = "sym_halfkp_64x2_512")]
pub(super) const EXPORTED: (&str, u32) = (
    "ModelType=SYM;Features=HalfKP(Friend)[125389->64x2],Network=Dense[1<-512](ReLU[512](Dense[512<-384](RmsNormScaled[192](Newton[3](PowerSum[3](InputSlice[192(0:192)])))))",
    0x6f7e_587b,
);
#[cfg(feature = "sym_kp_128x2_512")]
pub(super) const EXPORTED: (&str, u32) = (
    "ModelType=SYM;Features=K+P[1710->128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-768](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)])))))",
    0x4c53_92cf,
);
#[cfg(feature = "sym_halfkp_128x2_512")]
pub(super) const EXPORTED: (&str, u32) = (
    "ModelType=SYM;Features=HalfKP(Friend)[125389->128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-768](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)])))))",
    0x70a3_ccf5,
);
#[cfg(feature = "sym_kp_256x2_512")]
pub(super) const EXPORTED: (&str, u32) = (
    "ModelType=SYM;Features=K+P[1710->256x2],Network=Dense[1<-512](ReLU[512](Dense[512<-1536](RmsNormScaled[768](Newton[3](PowerSum[3](InputSlice[768(0:768)])))))",
    0x7cfa_4607,
);
#[cfg(feature = "sym_halfkp_256x2_512")]
pub(super) const EXPORTED: (&str, u32) = (
    "ModelType=SYM;Features=HalfKP(Friend)[125389->256x2],Network=Dense[1<-512](ReLU[512](Dense[512<-1536](RmsNormScaled[768](Newton[3](PowerSum[3](InputSlice[768(0:768)])))))",
    0x779a_cf31,
);
#[cfg(feature = "sym_kp_256x2_o1_32_32")]
pub(super) const EXPORTED: (&str, u32) = (
    "ModelType=SYM;Features=K+P[1710->256x2],Network=Dense[1<-32](ReLU[32](Dense[32<-32](ReLU[32](Dense[32<-512](RmsNormScaled[256](Newton[1](PowerSum[1](InputSlice[256(0:256)])))))))",
    0x0f12_1c44,
);
#[cfg(feature = "sym_halfkp_256x2_o1_32_32")]
pub(super) const EXPORTED: (&str, u32) = (
    "ModelType=SYM;Features=HalfKP(Friend)[125389->256x2],Network=Dense[1<-32](ReLU[32](Dense[32<-32](ReLU[32](Dense[32<-512](RmsNormScaled[256](Newton[1](PowerSum[1](InputSlice[256(0:256)])))))))",
    0xc1e0_8026,
);

/// A network no trainer made, with weights small enough to stay inside the
/// ranges a quantised file holds (`i8` at 64 reaches ±1.98).
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

/// `net` written the way `write_sym_network` writes it.
pub(super) fn encode(net: &SymNetwork, quantised: bool) -> Vec<u8> {
    let arch = expected_arch(quantised);
    let hash = arch_hash(&arch);
    let mut out = Vec::new();
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&hash.to_le_bytes());
    out.extend_from_slice(&(arch.len() as u32).to_le_bytes());
    out.extend_from_slice(arch.as_bytes());

    let f32s = |out: &mut Vec<u8>, values: &[f32]| {
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
    };
    let i8s = |out: &mut Vec<u8>, values: &[f32]| {
        for v in values {
            out.push((v * WEIGHT_SCALE).round() as i8 as u8);
        }
    };
    let i32s = |out: &mut Vec<u8>, values: &[f32]| {
        for v in values {
            out.extend_from_slice(&((v * WEIGHT_SCALE).round() as i32).to_le_bytes());
        }
    };

    out.extend_from_slice(&hash.wrapping_add(1).to_le_bytes());
    match quantised {
        true => {
            for v in &net.embed {
                out.extend_from_slice(&((v * EMBED_SCALE).round() as i16).to_le_bytes());
            }
        }
        false => f32s(&mut out, &net.embed),
    }
    out.extend_from_slice(&hash.wrapping_add(2).to_le_bytes());
    f32s(&mut out, &net.gamma);
    match quantised {
        true => {
            i8s(&mut out, &net.w1);
            i32s(&mut out, &net.b1);
            i8s(&mut out, &net.w2);
            i32s(&mut out, &net.b2);
            i8s(&mut out, &net.value_w);
            i32s(&mut out, &[net.value_b]);
        }
        false => {
            f32s(&mut out, &net.w1);
            f32s(&mut out, &net.b1);
            f32s(&mut out, &net.w2);
            f32s(&mut out, &net.b2);
            f32s(&mut out, &net.value_w);
            f32s(&mut out, &[net.value_b]);
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
