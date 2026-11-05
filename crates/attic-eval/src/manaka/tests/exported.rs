//! What Manaka's exporter wrote for each shape's architecture string.
//!
//! Kept apart from the fixtures because it grows by a constant for every shape,
//! and because it is a table of facts and not code: each is copied from a real
//! export, never produced by this crate.

/// What Manaka's exporter (`sym_arch_string`) wrote for this shape: the string
/// without its `ModelType` word, and the file hash it stored for each of the
/// encodings it writes. Copied from real exports, not from `expected_arch`:
/// asserting a function against itself would pass for any string.
///
/// The d64 / d256 hashes are from the 34-shape export of `atmos-34`; the d128
/// ones are from files the same exporter wrote for a randomly initialised net,
/// since `atmos-34` has no d128 KP or HalfKP shape. The token shapes' `SYMscaled`
/// and `SYMf32` hashes are `atmos-34`'s too (all of them have weights there,
/// d128 included); the plain ones' `SYM` hashes are from the first exporter, the
/// one that wrote `ModelType=SYM` and nothing else, run over the same `.mnw`.
/// The shapes with a pair table, a second moment, a square field or kept
/// magnitudes have no `SYM` file at all: that encoding has no section for them,
/// so the exporter refuses to write one.
pub(super) struct Exported {
    /// Everything after `ModelType=<word>;Features=`.
    pub(super) rest: &'static str,
    pub(super) scaled: u32,
    pub(super) f32: u32,
    pub(super) legacy: Option<u32>,
}

#[cfg(feature = "sym_kp_64x2_512")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "K+P[1710->64x2],Network=Dense[1<-512](ReLU[512](Dense[512<-384](RmsNormScaled[192](Newton[3](PowerSum[3](InputSlice[192(0:192)])))))",
    scaled: 0xf6b9_acc7,
    f32: 0x6e54_0c82,
    legacy: Some(0x0ff7_e8cd),
};

#[cfg(feature = "sym_halfkp_64x2_512")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "HalfKP(Friend)[125389->64x2],Network=Dense[1<-512](ReLU[512](Dense[512<-384](RmsNormScaled[192](Newton[3](PowerSum[3](InputSlice[192(0:192)])))))",
    scaled: 0xcf53_3409,
    f32: 0x6854_422a,
    legacy: Some(0x6f7e_587b),
};

#[cfg(feature = "sym_kp_128x2_512")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "K+P[1710->128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-768](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)])))))",
    scaled: 0xbaff_2ef9,
    f32: 0xe700_9396,
    legacy: Some(0x4c53_92cf),
};

#[cfg(feature = "sym_halfkp_128x2_512")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "HalfKP(Friend)[125389->128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-768](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)])))))",
    scaled: 0x3798_f7d3,
    f32: 0x1832_185e,
    legacy: Some(0x70a3_ccf5),
};

#[cfg(feature = "sym_kp_256x2_512")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "K+P[1710->256x2],Network=Dense[1<-512](ReLU[512](Dense[512<-1536](RmsNormScaled[768](Newton[3](PowerSum[3](InputSlice[768(0:768)])))))",
    scaled: 0x4691_8805,
    f32: 0xac38_e330,
    legacy: Some(0x7cfa_4607),
};

#[cfg(feature = "sym_halfkp_256x2_512")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "HalfKP(Friend)[125389->256x2],Network=Dense[1<-512](ReLU[512](Dense[512<-1536](RmsNormScaled[768](Newton[3](PowerSum[3](InputSlice[768(0:768)])))))",
    scaled: 0xfe57_24cb,
    f32: 0xc30f_72e8,
    legacy: Some(0x779a_cf31),
};

#[cfg(feature = "sym_kp_256x2_o1_32_32")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "K+P[1710->256x2],Network=Dense[1<-32](ReLU[32](Dense[32<-32](ReLU[32](Dense[32<-512](RmsNormScaled[256](Newton[1](PowerSum[1](InputSlice[256(0:256)])))))))",
    scaled: 0x5eb6_71aa,
    f32: 0x2df6_cba1,
    legacy: Some(0x0f12_1c44),
};

#[cfg(feature = "sym_halfkp_256x2_o1_32_32")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "HalfKP(Friend)[125389->256x2],Network=Dense[1<-32](ReLU[32](Dense[32<-32](ReLU[32](Dense[32<-512](RmsNormScaled[256](Newton[1](PowerSum[1](InputSlice[256(0:256)])))))))",
    scaled: 0x079b_c6a8,
    f32: 0xf554_4019,
    legacy: Some(0xc1e0_8026),
};

#[cfg(feature = "sym_token_64x2_512")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[64x2],Network=Dense[1<-512](ReLU[512](Dense[512<-384](RmsNormScaled[192](Newton[3](PowerSum[3](InputSlice[192(0:192)])))))",
    scaled: 0xdef0_e7d3,
    f32: 0x47bb_126c,
    legacy: Some(0x04b9_13ad),
};

#[cfg(feature = "sym_token_128x2_512")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-768](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)])))))",
    scaled: 0xa7a2_7b4d,
    f32: 0x7171_d448,
    legacy: Some(0xaf66_abaf),
};

#[cfg(feature = "sym_token_256x2_512")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[256x2],Network=Dense[1<-512](ReLU[512](Dense[512<-1536](RmsNormScaled[768](Newton[3](PowerSum[3](InputSlice[768(0:768)])))))",
    scaled: 0x1a1a_b599,
    f32: 0x4fe7_ad9e,
    legacy: Some(0x8927_1ae7),
};

#[cfg(feature = "sym_token_256x2_o1_32_32")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[256x2],Network=Dense[1<-32](ReLU[32](Dense[32<-32](ReLU[32](Dense[32<-512](RmsNormScaled[256](Newton[1](PowerSum[1](InputSlice[256(0:256)])))))))",
    scaled: 0x3433_856e,
    f32: 0x9620_ac1f,
    legacy: Some(0xa1f8_aca4),
};

#[cfg(feature = "sym_token_128x2_512_pair16")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-800](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),RmsNormScaled[16](PairSum[16](InputSlice[16(384:400)]))))",
    scaled: 0x2774_67d8,
    f32: 0x6956_14dd,
    legacy: None,
};

#[cfg(feature = "sym_token_128x2_512_pair32")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-832](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),RmsNormScaled[32](PairSum[32](InputSlice[32(384:416)]))))",
    scaled: 0x42fb_1258,
    f32: 0x3f80_905d,
    legacy: None,
};

#[cfg(feature = "sym_token_128x2_512_pair64")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-896](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),RmsNormScaled[64](PairSum[64](InputSlice[64(384:448)]))))",
    scaled: 0x0125_4002,
    f32: 0x7d4b_14b7,
    legacy: None,
};

#[cfg(feature = "sym_token_64x2_512_pair16")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[64x2],Network=Dense[1<-512](ReLU[512](Dense[512<-416](RmsNormScaled[192](Newton[3](PowerSum[3](InputSlice[192(0:192)]))),RmsNormScaled[16](PairSum[16](InputSlice[16(192:208)]))))",
    scaled: 0xf392_acb8,
    f32: 0x30f5_e81f,
    legacy: None,
};

#[cfg(feature = "sym_token_256x2_512_pair16")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[256x2],Network=Dense[1<-512](ReLU[512](Dense[512<-1568](RmsNormScaled[768](Newton[3](PowerSum[3](InputSlice[768(0:768)]))),RmsNormScaled[16](PairSum[16](InputSlice[16(768:784)]))))",
    scaled: 0x1143_41bf,
    f32: 0xed85_90ec,
    legacy: None,
};

#[cfg(feature = "sym_token_128x2_o2_512_pair16")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-544](RmsNormScaled[256](Newton[2](PowerSum[2](InputSlice[256(0:256)]))),RmsNormScaled[16](PairSum[16](InputSlice[16(256:272)]))))",
    scaled: 0x0c1a_1b34,
    f32: 0xbfb8_6e99,
    legacy: None,
};

#[cfg(feature = "sym_token_128x2_o4_512_pair16")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-1056](RmsNormScaled[512](Newton[4](PowerSum[4](InputSlice[512(0:512)]))),RmsNormScaled[16](PairSum[16](InputSlice[16(512:528)]))))",
    scaled: 0x5825_6133,
    f32: 0xc22b_fb60,
    legacy: None,
};

#[cfg(feature = "sym_token_128x2_1024_pair16")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-1024](ReLU[1024](Dense[1024<-800](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),RmsNormScaled[16](PairSum[16](InputSlice[16(384:400)]))))",
    scaled: 0xf1f9_ad1f,
    f32: 0xb99c_9a94,
    legacy: None,
};

#[cfg(feature = "sym_token_128x2_512_256_pair16")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-256](ReLU[256](Dense[256<-512](ReLU[512](Dense[512<-800](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),RmsNormScaled[16](PairSum[16](InputSlice[16(384:400)]))))))",
    scaled: 0x58bb_8b49,
    f32: 0x7611_2ea0,
    legacy: None,
};

#[cfg(feature = "sym_token_128x2_512_512_pair16")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-512](ReLU[512](Dense[512<-800](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),RmsNormScaled[16](PairSum[16](InputSlice[16(384:400)]))))))",
    scaled: 0xaf4e_6722,
    f32: 0x1a2b_2bfb,
    legacy: None,
};

#[cfg(feature = "sym_token_128x2_512_pair16_lse4")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-800](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),RmsNormScaled[12](PairSum[12](InputSlice[12(384:396)])),LogSumExp[4](PairSum[4](InputSlice[4(396:400)]))))",
    scaled: 0xf9a9_621c,
    f32: 0xffbc_323b,
    legacy: None,
};

#[cfg(feature = "sym_token_128x2_512_pair16_lse8")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-800](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),RmsNormScaled[8](PairSum[8](InputSlice[8(384:392)])),LogSumExp[8](PairSum[8](InputSlice[8(392:400)]))))",
    scaled: 0x2d4e_9387,
    f32: 0x10aa_8d82,
    legacy: None,
};

#[cfg(feature = "sym_token_128x2_512_pair16_lse16")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-800](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),LogSumExp[16](PairSum[16](InputSlice[16(384:400)]))))",
    scaled: 0x28c2_3ba2,
    f32: 0xad7c_9737,
    legacy: None,
};

#[cfg(feature = "sym_token_128x2_512_pair16_lse4_ks")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-808](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),RmsNormScaled[12](PairSum[12](InputSlice[12(384:396)])),LogSumExp[4](PairSum[4](InputSlice[4(396:400)])),KeepScale[4]))",
    scaled: 0x1644_b48d,
    f32: 0x09dd_05d8,
    legacy: None,
};

#[cfg(feature = "sym_token_128x2_512_pair16_ks")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-808](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),RmsNormScaled[16](PairSum[16](InputSlice[16(384:400)])),KeepScale[4]))",
    scaled: 0x8634_ca59,
    f32: 0x2c43_4636,
    legacy: None,
};

#[cfg(feature = "sym_token_128x2_512_pair16_split")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-1568](OwnerSplit[2](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)])))),RmsNormScaled[16](PairSum[16](InputSlice[16(768:784)]))))",
    scaled: 0xa79f_cab5,
    f32: 0x0631_3a80,
    legacy: None,
};

#[cfg(feature = "sym_token_128x2_512_pair16_m16")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-1072](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),RmsNormScaled[16](PairSum[16](InputSlice[16(384:400)])),RmsNormScaled[136](SecondMoment[16](InputSlice[136(400:536)]))))",
    scaled: 0xc958_e2b0,
    f32: 0xa581_dc89,
    legacy: None,
};

#[cfg(feature = "sym_token_128x2_512_pair16_m32")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-1856](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),RmsNormScaled[16](PairSum[16](InputSlice[16(384:400)])),RmsNormScaled[528](SecondMoment[32](InputSlice[528(400:928)]))))",
    scaled: 0xa7a5_bb99,
    f32: 0x7bc8_1be0,
    legacy: None,
};

#[cfg(feature = "sym_token_128x2_512_pair16_f2")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-1124](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),RmsNormScaled[16](PairSum[16](InputSlice[16(384:400)])),RmsNormScaled[162](SquareField[2](InputSlice[162(400:562)]))))",
    scaled: 0x999c_e9a5,
    f32: 0x95c4_dfa4,
    legacy: None,
};

#[cfg(feature = "sym_token_128x2_512_pair16_f4")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-1448](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),RmsNormScaled[16](PairSum[16](InputSlice[16(384:400)])),RmsNormScaled[324](SquareField[4](InputSlice[324(400:724)]))))",
    scaled: 0xbefd_e46c,
    f32: 0x6d34_6b05,
    legacy: None,
};

#[cfg(feature = "sym_token_128x2_512_pair16_f8")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-512](ReLU[512](Dense[512<-2096](RmsNormScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),RmsNormScaled[16](PairSum[16](InputSlice[16(384:400)])),RmsNormScaled[648](SquareField[8](InputSlice[648(400:1048)]))))",
    scaled: 0x0e09_b526,
    f32: 0x769d_f041,
    legacy: None,
};
