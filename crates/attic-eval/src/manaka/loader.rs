//! Reading the `nn.bin` container, in each of the three encodings of its numbers.
//!
//! One layout, three encodings, and `ModelType` in the architecture string says
//! which. Each encoding has a reader of its own that takes the body from the
//! first section to the last, so a new encoding is one more function and one
//! more arm of the `match` in [`parse_network`]; nothing else here knows how a
//! number is stored. Whatever the encoding, the result is the same `f32` tables,
//! which is what keeps the forward pass free of any of them.

use std::path::Path;

use crate::types::NnueError;

use super::reader::Reader;
use super::{DIM, FEATURE_LEN, HEAD, HIDDEN, HIDDEN2, ORDER, SymNetwork, TOKENS};

/// The container's version word, shared with every `nn.bin` YaneuraOu writes.
///
/// These files are not YaneuraOu's — it would refuse one at the architecture
/// string — but they are the same *container*, which is what lets one reader
/// open both and decide from the header which it is holding.
pub(super) const VERSION: u32 = 0x7AF3_2F16;

/// What one embedding channel was multiplied by on its way to `i16`, in the
/// legacy encoding. `SYMscaled` carries a scale per group instead.
pub(super) const EMBED_SCALE: f32 = 512.0;

/// What a dense weight or bias was multiplied by in the legacy encoding: six
/// bits of fraction, the same `1 << kWeightScaleBits` NNUE uses.
pub(super) const WEIGHT_SCALE: f32 = 64.0;

/// How the numbers of a file are stored. The three differ in the `ModelType`
/// word of the architecture string and nowhere else.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Encoding {
    /// `i16` arrays, each with a scale per group that the file carries; biases,
    /// gains and the value row `f32`. What `export-nnue` writes by default.
    Scaled,
    /// Every array `f32`, exactly as the trainer holds it.
    F32,
    /// `i16` embedding at one scale and `i8`/`i32` dense arrays at another. It
    /// clamps any trained weight past 1.98, so it is read for the files already
    /// written and no longer produced.
    Legacy,
}

impl Encoding {
    const ALL: [Encoding; 3] = [Encoding::Scaled, Encoding::F32, Encoding::Legacy];

    /// The `ModelType` word.
    pub(super) fn model_type(self) -> &'static str {
        match self {
            Encoding::Scaled => "SYMscaled",
            Encoding::F32 => "SYMf32",
            Encoding::Legacy => "SYM",
        }
    }
}

/// The architecture string a file for this build must carry.
///
/// Assembled rather than stored so that it cannot drift from the constants it
/// describes: change `DIM` and the expected string changes with it, and a file
/// written at the old width stops matching instead of being read at the wrong
/// stride. Manaka's exporter builds the same string from its own `Arch`
/// (`sym_arch_string`), parentheses included: it closes one fewer than it
/// opens, and a reader that balanced them would refuse every file.
///
/// The second hidden layer is spelled out when there is one. It is a layer in
/// the file, not a detail of it: a name that left it out would describe a net
/// whose value row is `HIDDEN` wide when it is `HIDDEN2` wide.
pub(super) fn expected_arch(encoding: Encoding) -> String {
    let features = match cfg!(feature = "sym_kp") {
        true => format!("K+P[{TOKENS}->"),
        false => format!("HalfKP(Friend)[{TOKENS}->"),
    };
    let model = encoding.model_type();
    let block = ORDER * DIM;
    let inner = format!(
        "Dense[{HIDDEN}<-{input}](\
         RmsNormScaled[{block}](Newton[{ORDER}](PowerSum[{ORDER}](InputSlice[{block}(0:{block})])))))",
        input = 2 * block,
    );
    let network = match HIDDEN2 {
        0 => format!("Dense[1<-{HIDDEN}](ReLU[{HIDDEN}]({inner}"),
        _ => format!(
            "Dense[1<-{HIDDEN2}](ReLU[{HIDDEN2}](Dense[{HIDDEN2}<-{HIDDEN}](ReLU[{HIDDEN}]({inner}))"
        ),
    };
    format!("ModelType={model};Features={features}{DIM}x2],Network={network}")
}

/// FNV-1a over the architecture string, the same digest Manaka's exporter
/// writes. Neither side keeps a table of magic numbers that could drift out of
/// step with the structure they are supposed to describe.
pub(super) fn arch_hash(arch: &str) -> u32 {
    let mut hash: u32 = 0x811C_9DC5;
    for byte in arch.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

/// Read a network from the `nn.bin` container, checking the architecture
/// string before a single weight.
///
/// The header is the whole of the identification: `ModelType=SYM…` says this is
/// not one of YaneuraOu's, and the rest of the string spells the structure. A
/// file for a different shape — another width, another feature space, another
/// order — produces a different string and is refused here rather than read at
/// the wrong stride.
pub fn load_network(path: &Path) -> Result<SymNetwork, NnueError> {
    let bytes = std::fs::read(path).map_err(|source| NnueError::Io {
        path: path.display().to_string(),
        source,
    })?;
    parse_network(&bytes)
}

/// [`load_network`] from the bytes of the file, so that a test can read a
/// network it just wrote without a file for it to live in.
pub(super) fn parse_network(bytes: &[u8]) -> Result<SymNetwork, NnueError> {
    let mut r = Reader { bytes, pos: 0 };

    let version = r.u32_le()?;
    if version != VERSION {
        return Err(NnueError::InvalidFormat {
            reason: format!("container version {version:#010x}, this build reads {VERSION:#010x}"),
        });
    }

    // Every spelling of this shape is accepted, and which one the file carries
    // decides how the body is read. Keeping one reader for all is the point: a
    // rating run that wants to price the quantisation has to be able to put the
    // same binary behind each file.
    let hash = r.u32_le()?;
    let len = r.u32_le()? as usize;
    let arch = std::str::from_utf8(r.take(len)?).map_err(|_| NnueError::InvalidFormat {
        reason: "architecture string is not UTF-8".to_string(),
    })?;
    let Some(encoding) = Encoding::ALL
        .into_iter()
        .find(|e| arch == expected_arch(*e))
    else {
        return Err(NnueError::InvalidFormat {
            reason: format!(
                "file says {arch:?}, this build reads {:?}",
                expected_arch(Encoding::Scaled)
            ),
        });
    };
    let want_hash = arch_hash(arch);
    // Redundant once the string matched, and kept for the same reason the
    // section hashes are: four bytes that catch a file whose header was
    // written by one version of the exporter and whose body was not.
    if hash != want_hash {
        return Err(NnueError::InvalidFormat {
            reason: format!(
                "file hash is {hash:#010x} for an architecture that digests to {want_hash:#010x}"
            ),
        });
    }

    let net = match encoding {
        Encoding::Scaled => read_scaled(&mut r, want_hash)?,
        Encoding::F32 => read_f32(&mut r, want_hash)?,
        Encoding::Legacy => read_legacy(&mut r, want_hash)?,
    };

    if r.pos != bytes.len() {
        return Err(NnueError::SizeMismatch {
            expected: r.pos,
            got: bytes.len(),
        });
    }
    Ok(net)
}

/// `SYMscaled`: section 1 is the embedding and section 2 is everything dense,
/// with a scale per group on the arrays that have one. `gamma`, the biases and
/// the value row are `f32`: the value row because its entries multiply hidden
/// units that are not bounded, so a step that is invisible in `W1` is amplified
/// there, and the rest because they are a few thousand numbers.
///
/// The shapes served here have none of the optional sections (pair, moment,
/// field, kept scale), so the file ends where section 2 does.
fn read_scaled(r: &mut Reader<'_>, hash: u32) -> Result<SymNetwork, NnueError> {
    r.expect_hash(hash.wrapping_add(1), "feature transformer")?;
    let embed = r.scaled_per_column(TOKENS, DIM, "embed")?;

    r.expect_hash(hash.wrapping_add(2), "dense stack")?;
    let gamma = r.f32s(ORDER * DIM)?;
    let w1 = r.scaled_per_row(HIDDEN, FEATURE_LEN, "w1")?;
    let b1 = r.f32s(HIDDEN)?;
    let w2 = r.scaled_per_row(HIDDEN2, HIDDEN, "w2")?;
    let b2 = r.f32s(HIDDEN2)?;
    let value_w = r.f32s(HEAD)?;
    let value_b = r.f32s(1)?[0];
    Ok(SymNetwork {
        embed,
        gamma,
        w1,
        b1,
        w2,
        b2,
        value_w,
        value_b,
    })
}

/// `SYMf32`: the trainer's own numbers, verbatim.
fn read_f32(r: &mut Reader<'_>, hash: u32) -> Result<SymNetwork, NnueError> {
    r.expect_hash(hash.wrapping_add(1), "feature transformer")?;
    let embed = r.f32s(TOKENS * DIM)?;

    r.expect_hash(hash.wrapping_add(2), "dense stack")?;
    let gamma = r.f32s(ORDER * DIM)?;
    // The second layer's arrays sit between the first's and the value row, and
    // are zero long for a shape with one layer, so they read nothing there.
    let w1 = r.f32s(HIDDEN * FEATURE_LEN)?;
    let b1 = r.f32s(HIDDEN)?;
    let w2 = r.f32s(HIDDEN2 * HIDDEN)?;
    let b2 = r.f32s(HIDDEN2)?;
    let value_w = r.f32s(HEAD)?;
    let value_b = r.f32s(1)?[0];
    Ok(SymNetwork {
        embed,
        gamma,
        w1,
        b1,
        w2,
        b2,
        value_w,
        value_b,
    })
}

/// `SYM`, the first quantised encoding: one scale per kind of array, as NNUE
/// does. An `i8` at 64 holds ±1.98 and trained weights reach 28, so the writer
/// clamped them; this reads what is in the file and cannot know what was lost.
///
/// `embed` is stored at [`EMBED_SCALE`], so `p_k` arrives at `S^k` — a
/// different scale per order, which would normally have to be tracked and
/// undone one at a time. It does not, because the next thing each block meets
/// is a division by its own root-mean-square, and a constant factor divides
/// out of that exactly. Dividing each array by the one constant it was
/// multiplied by on the way in is therefore all there is to do.
fn read_legacy(r: &mut Reader<'_>, hash: u32) -> Result<SymNetwork, NnueError> {
    r.expect_hash(hash.wrapping_add(1), "feature transformer")?;
    let embed = r.i16s_scaled(TOKENS * DIM, EMBED_SCALE)?;

    r.expect_hash(hash.wrapping_add(2), "dense stack")?;
    // `gamma` was never quantised in any encoding: it applies immediately
    // after the norm, the one place in the forward where a rounding error is
    // not divided back out again.
    let gamma = r.f32s(ORDER * DIM)?;
    let w1 = r.i8s_scaled(HIDDEN * FEATURE_LEN, WEIGHT_SCALE)?;
    let b1 = r.i32s_scaled(HIDDEN, WEIGHT_SCALE)?;
    let w2 = r.i8s_scaled(HIDDEN2 * HIDDEN, WEIGHT_SCALE)?;
    let b2 = r.i32s_scaled(HIDDEN2, WEIGHT_SCALE)?;
    let value_w = r.i8s_scaled(HEAD, WEIGHT_SCALE)?;
    let value_b = r.i32s_scaled(1, WEIGHT_SCALE)?[0];
    Ok(SymNetwork {
        embed,
        gamma,
        w1,
        b1,
        w2,
        b2,
        value_w,
        value_b,
    })
}

/// Read a network, with the warning list the NNUE loader returns.
///
/// Always empty here. A mismatch in these files is refused outright — the
/// architecture string either matches or it does not — where NNUE tolerates a
/// stale hash with a warning because the reference does.
pub fn load_network_with_warnings(path: &Path) -> Result<(SymNetwork, Vec<String>), NnueError> {
    load_network(path).map(|net| (net, Vec::new()))
}
