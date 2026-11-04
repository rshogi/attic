//! Reading the `nn.bin` container, quantised or not.

use std::path::Path;

use crate::types::NnueError;

use super::{DIM, FEATURE_LEN, HEAD, HIDDEN, HIDDEN2, ORDER, SymNetwork, TOKENS};

/// The container's version word, shared with every `nn.bin` YaneuraOu writes.
///
/// These files are not YaneuraOu's — it would refuse one at the architecture
/// string — but they are the same *container*, which is what lets one reader
/// open both and decide from the header which it is holding.
pub(super) const VERSION: u32 = 0x7AF3_2F16;

/// What one embedding channel was multiplied by on its way to `i16`.
pub(super) const EMBED_SCALE: f32 = 512.0;

/// What a dense weight or bias was multiplied by: six bits of fraction, the
/// same `1 << kWeightScaleBits` NNUE uses.
pub(super) const WEIGHT_SCALE: f32 = 64.0;

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], NnueError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| NnueError::InvalidFormat {
                reason: "read offset overflows usize".to_string(),
            })?;
        if end > self.bytes.len() {
            return Err(NnueError::SizeMismatch {
                expected: end,
                got: self.bytes.len(),
            });
        }
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32_le(&mut self) -> Result<u32, NnueError> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, NnueError> {
        let s = self.take(n * 4)?;
        let (words, _) = s.as_chunks::<4>();
        Ok(words.iter().map(|c| f32::from_le_bytes(*c)).collect())
    }

    /// Read a quantised block back onto the scale it was trained at.
    ///
    /// One division by the one constant that array was multiplied by — see
    /// the module docs on why the power sums need no per-order correction.
    fn i16s_scaled(&mut self, n: usize, scale: f32) -> Result<Vec<f32>, NnueError> {
        let s = self.take(n * 2)?;
        let (pairs, _) = s.as_chunks::<2>();
        Ok(pairs
            .iter()
            .map(|c| f32::from(i16::from_le_bytes(*c)) / scale)
            .collect())
    }

    fn i8s_scaled(&mut self, n: usize, scale: f32) -> Result<Vec<f32>, NnueError> {
        let s = self.take(n)?;
        Ok(s.iter().map(|&b| f32::from(b as i8) / scale).collect())
    }

    fn i32s_scaled(&mut self, n: usize, scale: f32) -> Result<Vec<f32>, NnueError> {
        let s = self.take(n * 4)?;
        let (words, _) = s.as_chunks::<4>();
        Ok(words
            .iter()
            .map(|c| i32::from_le_bytes(*c) as f32 / scale)
            .collect())
    }

    fn expect_hash(&mut self, want: u32, section: &str) -> Result<(), NnueError> {
        let got = self.u32_le()?;
        if got == want {
            return Ok(());
        }
        Err(NnueError::InvalidFormat {
            reason: format!("{section} hash is {got:#010x}, this build expects {want:#010x}"),
        })
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
pub(super) fn expected_arch(quantised: bool) -> String {
    let features = match cfg!(feature = "sym_kp") {
        true => format!("K+P[{TOKENS}->"),
        false => format!("HalfKP(Friend)[{TOKENS}->"),
    };
    let model = match quantised {
        true => "SYM",
        false => "SYMf32",
    };
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
/// The header is the whole of the identification: `ModelType=SYM` says this is
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

    // Both spellings of this shape are accepted, and which one the file
    // carries decides how the body is read. Keeping one reader for the two is
    // the point: a rating run that wants to price the quantisation has to be
    // able to put the same binary behind both files.
    let quantised_arch = expected_arch(true);
    let float_arch = expected_arch(false);

    let hash = r.u32_le()?;
    let len = r.u32_le()? as usize;
    let arch = std::str::from_utf8(r.take(len)?).map_err(|_| NnueError::InvalidFormat {
        reason: "architecture string is not UTF-8".to_string(),
    })?;
    let quantised = if arch == quantised_arch {
        true
    } else if arch == float_arch {
        false
    } else {
        return Err(NnueError::InvalidFormat {
            reason: format!("file says {arch:?}, this build reads {quantised_arch:?}"),
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

    r.expect_hash(want_hash.wrapping_add(1), "feature transformer")?;
    let embed = match quantised {
        true => r.i16s_scaled(TOKENS * DIM, EMBED_SCALE)?,
        false => r.f32s(TOKENS * DIM)?,
    };

    r.expect_hash(want_hash.wrapping_add(2), "dense stack")?;
    // `gamma` was never quantised in either encoding: it applies immediately
    // after the norm, the one place in the forward where a rounding error is
    // not divided back out again.
    let gamma = r.f32s(ORDER * DIM)?;
    // The second layer's arrays sit between the first's and the value row, and
    // are zero long for a shape with one layer, so they read nothing there.
    let (w1, b1, w2, b2, value_w, value_b) = match quantised {
        true => (
            r.i8s_scaled(HIDDEN * FEATURE_LEN, WEIGHT_SCALE)?,
            r.i32s_scaled(HIDDEN, WEIGHT_SCALE)?,
            r.i8s_scaled(HIDDEN2 * HIDDEN, WEIGHT_SCALE)?,
            r.i32s_scaled(HIDDEN2, WEIGHT_SCALE)?,
            r.i8s_scaled(HEAD, WEIGHT_SCALE)?,
            r.i32s_scaled(1, WEIGHT_SCALE)?[0],
        ),
        false => (
            r.f32s(HIDDEN * FEATURE_LEN)?,
            r.f32s(HIDDEN)?,
            r.f32s(HIDDEN2 * HIDDEN)?,
            r.f32s(HIDDEN2)?,
            r.f32s(HEAD)?,
            r.f32s(1)?[0],
        ),
    };

    if r.pos != bytes.len() {
        return Err(NnueError::SizeMismatch {
            expected: r.pos,
            got: bytes.len(),
        });
    }

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
