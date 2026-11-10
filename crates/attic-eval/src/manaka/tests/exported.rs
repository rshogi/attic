//! What Manaka's exporter wrote for each shape's architecture string.
//!
//! Kept apart from the fixtures because it grows by a constant for every shape,
//! and because it is a table of facts and not code: each is copied from a real
//! export, never produced by this crate.

/// What Manaka's exporter (`sym_arch_string`) wrote for this shape: the string
/// without its `ModelType` word, and the file hash it stored for each of the
/// encodings it writes. Copied from the header of the files `manaka-train
/// export-nnue` wrote for a net of each shape, `--f32` and not: asserting a
/// function against its own output passes for any string, and these are what the
/// exporter said.
///
/// A new shape is a new row, from a new export.
pub(super) struct Exported {
    /// Everything after `ModelType=<word>;Features=`.
    pub(super) rest: &'static str,
    pub(super) scaled: u32,
    pub(super) f32: u32,
}

#[cfg(feature = "synn_token_128x2_32_32")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-768](ClipScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)])))))))",
    scaled: 0xa08f8938,
    f32: 0xec1b12f1,
};

#[cfg(feature = "synn_kp_128x2_32_32")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "K+P[1710->128x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-768](ClipScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)])))))))",
    scaled: 0xf45f2e0a,
    f32: 0xd25c1931,
};

#[cfg(feature = "synn_halfkp_128x2_32_32")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "HalfKP(Friend)[125389->128x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-768](ClipScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)])))))))",
    scaled: 0xd85c6dba,
    f32: 0xe039709b,
};

#[cfg(feature = "synn_token_128x2_32_32_pair16")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-800](ClipScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),ClipScaled[16](PairSum[16](InputSlice[16(384:400)]))))))",
    scaled: 0x47b4b26b,
    f32: 0x57d79e9c,
};

#[cfg(feature = "synn_token_128x2_32_32_pair16_m8_f2")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-1196](ClipScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),ClipScaled[16](PairSum[16](InputSlice[16(384:400)])),ClipScaled[36](SecondMoment[8](InputSlice[36(400:436)])),ClipScaled[162](SquareField[2](InputSlice[162(436:598)]))))))",
    scaled: 0x4e58be8f,
    f32: 0xe4270674,
};
