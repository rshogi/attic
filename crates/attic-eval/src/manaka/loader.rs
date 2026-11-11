//! Reading the `nn.bin` container: the header and the architecture string, and
//! then the body.
//!
//! One layout and one way to store a number, so the string says nothing about
//! how the numbers are stored: it leads with `Features=` as YaneuraOu's does and
//! spells the structure. `body.rs` reads the body from the first section to the
//! last, and nothing else knows how a number is stored. The result is `f32`
//! tables, which is what keeps the forward pass free of how they were written.

use std::path::Path;

use crate::types::NnueError;

use super::body::read_body;
use super::reader::Reader;
use super::{
    DIM, FEATURE_LEN, FIELD, FIELD_AT, FIELD_LEN, HAS_FIELD, HAS_MOMENT, HAS_PAIR, HIDDEN, HIDDEN2,
    MOMENT, MOMENT_AT, MOMENT_LEN, ORDER, PAIR, PAIR_AT, SymNetwork, TOKENS, W1Precision,
};

/// The container's version word, shared with every `nn.bin` YaneuraOu writes.
///
/// These files are not YaneuraOu's — it would refuse one at the architecture
/// string — but they are the same *container*.
pub(super) const VERSION: u32 = 0x7AF3_2F16;

/// The architecture string a file for this build must carry.
///
/// Assembled rather than stored so that it cannot drift from the constants it
/// describes: change `DIM` and the expected string changes with it, and a file
/// written at the old width stops matching instead of being read at the wrong
/// stride. Manaka's exporter builds the same string from its own `Arch`
/// (`sym_arch_string`), parentheses included: it closes one fewer than it
/// opens, and a reader that balanced them would refuse every file.
///
/// The words are the names of what the arithmetic is, as `PowerSum` and `Newton`
/// are: `ClipScaled` for each block, which is multiplied by its gain and clipped
/// to `[0, 1]`, and `ClippedReLU` after each hidden layer. They are also all that
/// tells this net from a SYM one, whose words are `RmsNormScaled` and `ReLU`, so a
/// SYM file is refused here by its string. The second hidden layer is a layer in
/// the file, not a detail of it: a name that left it out would describe a net whose
/// value row is `HIDDEN` wide when it is `HIDDEN2` wide.
pub(super) fn expected_arch() -> String {
    // Manaka's own space is the one that does not name its row count: the
    // exporter writes `Tokens[<d>x2]`, and the 2,344 is implied by the word.
    let features = if cfg!(feature = "sym_token") {
        "Tokens[".to_string()
    } else if cfg!(feature = "sym_kp") {
        format!("K+P[{TOKENS}->")
    } else {
        format!("HalfKP(Friend)[{TOKENS}->")
    };
    let block = ORDER * DIM;

    // The head's input, block by block, as the exporter spells them. The power
    // block comes first and the others follow in the order `shape.rs` lays them
    // out, each only if the shape has it.
    let mut input = format!(
        "ClipScaled[{block}](Newton[{ORDER}](PowerSum[{ORDER}](InputSlice[{block}(0:{block})])))"
    );
    if HAS_PAIR {
        let end = PAIR_AT + PAIR;
        input +=
            &format!(",ClipScaled[{PAIR}](PairSum[{PAIR}](InputSlice[{PAIR}({PAIR_AT}:{end})]))");
    }
    if HAS_MOMENT {
        let end = MOMENT_AT + MOMENT_LEN;
        input += &format!(
            ",ClipScaled[{MOMENT_LEN}](SecondMoment[{MOMENT}](InputSlice[{MOMENT_LEN}({MOMENT_AT}:{end})]))"
        );
    }
    if HAS_FIELD {
        let end = FIELD_AT + FIELD_LEN;
        input += &format!(
            ",ClipScaled[{FIELD_LEN}](SquareField[{FIELD}](InputSlice[{FIELD_LEN}({FIELD_AT}:{end})]))"
        );
    }

    // One `)` short, as the exporter's is: the outermost `Dense` is never
    // closed. What is closed here is the first layer and its activation, and the
    // second layer and its activation too.
    let network = format!(
        "Dense[1<-{HIDDEN2}](ClippedReLU[{HIDDEN2}](Dense[{HIDDEN2}<-{HIDDEN}]\
         (ClippedReLU[{HIDDEN}](Dense[{HIDDEN}<-{FEATURE_LEN}]({input}))))"
    );
    format!("Features={features}{DIM}x2],Network={network}")
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
/// The header is the whole of the identification: the string spells the structure,
/// and is compared with the one this build makes, byte for byte. A file for a
/// different shape — another width, another feature space, another order, or a SYM
/// net rather than a SYNN one — produces a different string and is refused here
/// rather than read at the wrong stride.
pub fn load_network(path: &Path) -> Result<SymNetwork, NnueError> {
    let bytes = std::fs::read(path).map_err(|source| NnueError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut net = parse_network(&bytes)?;
    // The first layer runs on `i16` weights (see `SymNetwork::use_first_layer`),
    // which is about 1.2 times the nodes per second for under a centipawn.
    // `ATTIC_SYNN_W1=f32` puts it back on the `f32` layer that defines it, to
    // measure one against the other, and `ATTIC_SYNN_W1=i8` puts it on `i8`
    // weights, faster again for about ten centipawns; anything else, or nothing,
    // is `i16`. Read here and not in `parse_network` so that the tests, which read
    // bytes they made, keep the `f32` layer.
    let name = std::env::var("ATTIC_SYNN_W1").ok();
    net.use_first_layer(W1Precision::from_name(name.as_deref()));
    Ok(net)
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

    let hash = r.u32_le()?;
    let len = r.u32_le()? as usize;
    let arch = std::str::from_utf8(r.take(len)?).map_err(|_| NnueError::InvalidFormat {
        reason: "architecture string is not UTF-8".to_string(),
    })?;
    // Built, not parsed: the string is this shape's or it is refused.
    let want = expected_arch();
    if arch != want {
        return Err(NnueError::InvalidFormat {
            reason: format!("file says {arch:?}, this build reads {want:?}"),
        });
    }
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

    let net = read_body(&mut r, want_hash)?;

    if r.pos != bytes.len() {
        return Err(NnueError::SizeMismatch {
            expected: r.pos,
            got: bytes.len(),
        });
    }
    Ok(net)
}

/// Read a network, with the warning list the NNUE loader returns.
///
/// Always empty here. A mismatch in these files is refused outright — the
/// architecture string either matches or it does not — where NNUE tolerates a
/// stale hash with a warning because the reference does.
pub fn load_network_with_warnings(path: &Path) -> Result<(SymNetwork, Vec<String>), NnueError> {
    load_network(path).map(|net| (net, Vec::new()))
}
