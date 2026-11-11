//! What Manaka's exporter wrote for each shape's architecture string.
//!
//! Kept apart from the fixtures because it grows by a constant for every shape,
//! and because it is a table of facts and not code: each is copied from a real
//! export, never produced by this crate.

/// What Manaka's exporter (`sym_arch_string`) wrote for this shape: the string
/// after its leading `Features=`, and the file hash it stored. Copied from the
/// header of the file `manaka-train export-nnue` wrote for a net of each shape (at
/// tideborn's commit 91421804): asserting a function against its own output passes
/// for any string, and these are what the exporter said.
///
/// A new shape is a new row, from a new export.
pub(super) struct Exported {
    /// Everything after `Features=`.
    pub(super) rest: &'static str,
    pub(super) hash: u32,
}

#[cfg(feature = "synn_token_128x2_32_32")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-768](ClipScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)])))))))",
    hash: 0x263525dd,
};

#[cfg(feature = "synn_kp_128x2_32_32")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "K+P[1710->128x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-768](ClipScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)])))))))",
    hash: 0xe7c9e6dd,
};

/// From the file `export-nnue` wrote at tideborn's develop e6141b7d for a net
/// trained at this shape (2026-11-11), not at 91421804 like the rest: the shape
/// is newer than that commit. The SYNN string is assembled from the shape there,
/// so the exporter needed no change to write it.
#[cfg(feature = "synn_kp_256x2_32_32")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "K+P[1710->256x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-1536](ClipScaled[768](Newton[3](PowerSum[3](InputSlice[768(0:768)])))))))",
    hash: 0x7b28b1e1,
};

#[cfg(feature = "synn_halfkp_128x2_32_32")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "HalfKP(Friend)[125389->128x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-768](ClipScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)])))))))",
    hash: 0xe1e6babf,
};

#[cfg(feature = "synn_token_128x2_32_32_pair16")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-800](ClipScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),ClipScaled[16](PairSum[16](InputSlice[16(384:400)]))))))",
    hash: 0x25faf650,
};

#[cfg(feature = "synn_token_128x2_32_32_pair16_m8_f2")]
pub(super) const EXPORTED: Exported = Exported {
    rest: "Tokens[128x2],Network=Dense[1<-32](ClippedReLU[32](Dense[32<-32](ClippedReLU[32](Dense[32<-1196](ClipScaled[384](Newton[3](PowerSum[3](InputSlice[384(0:384)]))),ClipScaled[16](PairSum[16](InputSlice[16(384:400)])),ClipScaled[36](SecondMoment[8](InputSlice[36(400:436)])),ClipScaled[162](SquareField[2](InputSlice[162(436:598)]))))))",
    hash: 0x00191180,
};
