//! Manaka's own evaluator, in the shared `nn.bin` container.
//!
//! The file is the same container every network here arrives in; what differs
//! is the architecture string, which leads with `ModelType=SYM` and spells a
//! structure YaneuraOu has no reader for. That is the whole identification —
//! one format, and the header says which shape is inside.
//!
//! Nothing below it is NNUE. The shapes served here — `sym_kp_128x2_512` and
//! `sym_halfkp_128x2_512` — borrow YaneuraOu's *feature numbering* (see
//! [`crate::features`], shared with the standard shapes) and nothing else: the
//! accumulator carries power sums to the third order rather than a single sum,
//! and each block is divided by its own root-mean-square where NNUE clips.
//!
//! ## Why the quantisation needs no per-order correction
//!
//! `embed` is stored at [`EMBED_SCALE`], so `p_k` arrives at `S^k` — a
//! different scale per order, which would normally have to be tracked and
//! undone one at a time. It does not, because the next thing each block meets
//! is a division by its own root-mean-square, and a constant factor divides
//! out of that exactly: `e_k / rms(e_k)` is the same number whether `e_k` was
//! built from `v` or from `v·S`. Every array is therefore divisible by the one
//! constant it was multiplied by, which is what the reader does on the way in.
//!
//! ## The forward pass
//!
//! Per perspective, over the forty active features:
//!
//! 1. Power sums `p_k[c] = Σ_i embed[token_i][c]^k` for `k = 1..=order`.
//! 2. Newton's identities turn those into elementary symmetric polynomials:
//!    `k·e_k = Σ_{j=1..k} (−1)^{j−1} · e_{k−j} · p_j`, with `e_0 = 1`.
//! 3. Each `e_k` block is RMS-normalised and scaled by its own `gamma`.
//!
//! The two perspectives' blocks are concatenated — side to move first — into
//! `2 · order · dim` values, which one hidden layer and a `tanh` value row
//! reduce to a scalar in `[-1, 1]`.
//!
//! Ported from `manaka-core`: `params.rs` (the file), `accum/newton.rs` (step
//! 2) and `eval.rs` (steps 1 and 3, and the head).

use std::path::Path;

use attic_state::{Color, Position};

use crate::features::{
    DiffScratch, FeatureIndex, MAX_ACTIVE_FEATURES, active_features, active_features_into,
    changed_indices_into,
};
use crate::types::NnueError;

/// The container's version word, shared with every `nn.bin` YaneuraOu writes.
///
/// These files are not YaneuraOu's — it would refuse one at the architecture
/// string — but they are the same *container*, which is what lets one reader
/// open both and decide from the header which it is holding.
const VERSION: u32 = 0x7AF3_2F16;

/// What one embedding channel was multiplied by on its way to `i16`.
const EMBED_SCALE: f32 = 512.0;

/// What a dense weight or bias was multiplied by: six bits of fraction, the
/// same `1 << kWeightScaleBits` NNUE uses.
const WEIGHT_SCALE: f32 = 64.0;

/// Guard on the root-mean-square, matching `manaka_core::eval::RMS_EPS`. Both
/// forwards floor at the same number or they are not the same function.
const RMS_EPS: f32 = 1e-6;

/// The width of one perspective's embedding.
#[cfg(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512"))]
pub const DIM: usize = 128;

/// Highest power the accumulator carries.
#[cfg(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512"))]
pub const ORDER: usize = 3;

/// The single hidden layer's width.
#[cfg(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512"))]
pub const HIDDEN: usize = 512;

/// Rows of the embedding table.
///
/// KP is `fe_end + 2 · 81`. HalfKP is `81 · fe_end + 1` — the `+ 1` is the
/// padding row both kings point at, which YaneuraOu has no equivalent of;
/// see [`crate::features::king_feature`].
#[cfg(feature = "sym_kp_128x2_512")]
pub const TOKENS: usize = 1_710;
#[cfg(feature = "sym_halfkp_128x2_512")]
pub const TOKENS: usize = 125_389;

/// What the hidden layer reads: both perspectives, every order.
pub const FEATURE_LEN: usize = 2 * ORDER * DIM;

/// A loaded `MNW1` network, with the tables this shape actually reads.
///
/// The pair, moment and field tables are absent by construction — Manaka's
/// shape grammar refuses them outside its own token space, because each wants
/// a row per token and these spaces have up to 125,389 of them — so the file
/// carries them at length zero and this does not name them at all.
#[derive(Debug)]
pub struct SymNetwork {
    /// `[TOKENS][DIM]`.
    pub embed: Vec<f32>,
    /// `[ORDER][DIM]` — the per-channel scale each normalised block is given.
    pub gamma: Vec<f32>,
    /// `[HIDDEN][FEATURE_LEN]`.
    pub w1: Vec<f32>,
    /// `[HIDDEN]`.
    pub b1: Vec<f32>,
    /// `[HIDDEN]`.
    pub value_w: Vec<f32>,
    pub value_b: f32,
}

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
/// stride. Manaka's exporter builds the same string from its own `Arch`.
fn expected_arch(quantised: bool) -> String {
    let features = match cfg!(feature = "sym_kp_128x2_512") {
        true => format!("K+P[{TOKENS}->"),
        false => format!("HalfKP(Friend)[{TOKENS}->"),
    };
    let model = match quantised {
        true => "SYM",
        false => "SYMf32",
    };
    let block = ORDER * DIM;
    format!(
        "ModelType={model};Features={features}{DIM}x2],\
         Network=Dense[1<-{HIDDEN}](ReLU[{HIDDEN}](Dense[{HIDDEN}<-{input}](\
         RmsNormScaled[{block}](Newton[{ORDER}](PowerSum[{ORDER}](InputSlice[{block}(0:{block})])))))",
        input = 2 * block,
    )
}

/// FNV-1a over the architecture string, the same digest Manaka's exporter
/// writes. Neither side keeps a table of magic numbers that could drift out of
/// step with the structure they are supposed to describe.
fn arch_hash(arch: &str) -> u32 {
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
    let mut r = Reader {
        bytes: &bytes,
        pos: 0,
    };

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
    let (w1, b1, value_w, value_b) = match quantised {
        true => (
            r.i8s_scaled(HIDDEN * FEATURE_LEN, WEIGHT_SCALE)?,
            r.i32s_scaled(HIDDEN, WEIGHT_SCALE)?,
            r.i8s_scaled(HIDDEN, WEIGHT_SCALE)?,
            r.i32s_scaled(1, WEIGHT_SCALE)?[0],
        ),
        false => (
            r.f32s(HIDDEN * FEATURE_LEN)?,
            r.f32s(HIDDEN)?,
            r.f32s(HIDDEN)?,
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
        value_w,
        value_b,
    })
}

/// Newton's identities, channel by channel: `k·e_k = Σ_{j=1..k} (−1)^{j−1} ·
/// e_{k−j} · p_j`, with `e_0 = 1`.
///
/// Not incrementally updatable, which is the reason the accumulator keeps `p`
/// and this runs once per evaluation rather than once per token change.
fn elementary_from_power(p: &[f32]) -> [f32; ORDER * DIM] {
    debug_assert_eq!(p.len(), ORDER * DIM);
    let mut out = [0.0f32; ORDER * DIM];
    for channel in 0..DIM {
        let mut e = [1.0f32; ORDER + 1];
        for k in 1..=ORDER {
            let mut acc = 0.0f32;
            let mut sign = 1.0f32;
            for j in 1..=k {
                acc += sign * e[k - j] * p[(j - 1) * DIM + channel];
                sign = -sign;
            }
            e[k] = acc / k as f32;
            out[(k - 1) * DIM + channel] = e[k];
        }
    }
    out
}

/// One perspective's contribution to the head's input: each order block
/// RMS-normalised, then scaled by its own gamma.
fn blocks_from_sums(net: &SymNetwork, sums: &[f32], out: &mut [f32]) {
    debug_assert_eq!(out.len(), ORDER * DIM);
    let e = elementary_from_power(sums);
    for k in 0..ORDER {
        let block = &e[k * DIM..(k + 1) * DIM];
        let mean_square: f32 = block.iter().map(|v| v * v).sum::<f32>() / DIM as f32;
        let scale = 1.0 / (mean_square + RMS_EPS).sqrt();
        let gamma = &net.gamma[k * DIM..(k + 1) * DIM];
        let out = &mut out[k * DIM..(k + 1) * DIM];
        for channel in 0..DIM {
            // Left to right, the same grouping Manaka's own forward uses. f32
            // addition is not associative and neither is this product, so
            // folding `scale * gamma` first would be a different number.
            out[channel] = block[channel] * scale * gamma[channel];
        }
    }
}

/// The head, from an accumulator: polynomials, norm, one hidden layer, `tanh`.
fn value_from_sums(net: &SymNetwork, acc: &Accumulator, stm: Color) -> f32 {
    let mut features = [0.0f32; FEATURE_LEN];
    // Side to move first: the head was trained reading its own frame in the
    // first half, and swapping them is a different network.
    for (half, frame) in [stm, stm.flip()].into_iter().enumerate() {
        blocks_from_sums(
            net,
            acc.perspective(frame),
            &mut features[half * ORDER * DIM..(half + 1) * ORDER * DIM],
        );
    }

    let mut hidden = [0.0f32; HIDDEN];
    for (row, slot) in hidden.iter_mut().enumerate() {
        let weights = &net.w1[row * FEATURE_LEN..(row + 1) * FEATURE_LEN];
        let sum: f32 = weights
            .iter()
            .zip(features.iter())
            .map(|(w, f)| w * f)
            .sum();
        *slot = (net.b1[row] + sum).max(0.0);
    }

    let sum: f32 = net
        .value_w
        .iter()
        .zip(hidden.iter())
        .map(|(w, h)| w * h)
        .sum();
    (net.value_b + sum).tanh()
}

/// Static evaluation of `pos` from the side to move's perspective, in
/// `[-1, 1]`, rebuilding the accumulator from scratch.
///
/// # Panics
/// Panics if `pos` is missing either king (via feature extraction).
pub fn evaluate_value(net: &SymNetwork, pos: &Position) -> f32 {
    let mut acc = Accumulator::new();
    acc.refresh(net, pos);
    value_from_sums(net, &acc, pos.side_to_move())
}

/// What a unit value is worth before [`crate::fv_scale`] divides it.
///
/// `127 · 64`, which is what NNUE's output layer carries: its `i32` is a
/// unit-scaled score times the activation scale times the weight scale, and
/// `ComputeScore` divides that by `FV_SCALE`. Matching it here is what puts
/// the two evaluators on one scale — at the default 16, both hand the search
/// about 508 per unit — so a search tuned against one is not retuned for the
/// other, and two networks of different kinds can be compared on a ladder.
pub const RAW_SCALE: f32 = 127.0 * 64.0;

/// Static evaluation of `pos`, positive when the side to move is better.
///
/// `tanh` bounds the value at one, so what the search sees is bounded too:
/// `±RAW_SCALE / fv_scale()`. That is the whole of the difference from NNUE,
/// whose output is unbounded and whose scale therefore has more work to do.
///
/// # Panics
/// Panics if `pos` is missing either king.
pub fn evaluate(net: &SymNetwork, pos: &Position) -> i32 {
    let raw = evaluate_value(net, pos) * RAW_SCALE;
    (raw / crate::fv_scale() as f32) as i32
}

impl SymNetwork {
    /// A second copy for another NUMA node.
    ///
    /// A plain clone: these tables are ordinary `Vec`s rather than views into
    /// a shared arena, so there is no backing allocation to duplicate by hand.
    pub fn replicate(&self) -> Self {
        SymNetwork {
            embed: self.embed.clone(),
            gamma: self.gamma.clone(),
            w1: self.w1.clone(),
            b1: self.b1.clone(),
            value_w: self.value_w.clone(),
            value_b: self.value_b,
        }
    }
}

/// Read a network, with the warning list the NNUE loader returns.
///
/// Always empty here. A mismatch in these files is refused outright — the
/// architecture string either matches or it does not — where NNUE tolerates a
/// stale hash with a warning because the reference does.
pub fn load_network_with_warnings(path: &Path) -> Result<(SymNetwork, Vec<String>), NnueError> {
    load_network(path).map(|net| (net, Vec::new()))
}

/// The accumulator the search threads through the tree.
///
/// Holds each perspective's power sums, and the active-feature list they were
/// built from so that the next position can be reached by difference.
///
/// ## Why the sums update incrementally and the polynomials do not
///
/// `p_k[c] = Σ_i v_i[c]^k` is a sum over tokens, so a token leaving subtracts
/// its own `k`-th powers and a token arriving adds them — the other
/// thirty-eight are untouched. The elementary symmetric polynomials that
/// [`elementary_from_power`] derives are *not* sums over tokens and cannot be
/// updated this way, which is exactly why the accumulator stores `p` and the
/// evaluation converts. That conversion is `ORDER · DIM` multiply-adds once
/// per evaluation rather than per token, and it is the whole reason Newton's
/// identities are in the forward at all.
///
/// ## Why there is no king-move special case
///
/// HalfKP renumbers all forty features when the perspective's own king moves,
/// which for NNUE means a rebuild. Here it needs no special handling: the diff
/// against the previous list reports thirty-eight removals and thirty-eight
/// additions, and applying those is arithmetically a rebuild. It costs what a
/// rebuild costs and produces what a rebuild produces, so the correctness does
/// not depend on the caller noticing.
#[derive(Debug, Clone)]
pub struct Accumulator {
    /// `[perspective][ORDER * DIM]`, indexed by [`Color::index`].
    sums: [Vec<f32>; Color::COUNT],
    /// What each half was built from, kept so a derive can diff against it.
    active: [Vec<FeatureIndex>; Color::COUNT],
}

impl Default for Accumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Accumulator {
    pub fn new() -> Self {
        Accumulator {
            sums: [vec![0.0; ORDER * DIM], vec![0.0; ORDER * DIM]],
            active: [
                Vec::with_capacity(MAX_ACTIVE_FEATURES),
                Vec::with_capacity(MAX_ACTIVE_FEATURES),
            ],
        }
    }

    /// Rebuild both perspectives from scratch.
    ///
    /// # Panics
    /// Panics if `pos` is missing either king.
    pub fn refresh(&mut self, net: &SymNetwork, pos: &Position) {
        for color in [Color::Black, Color::White] {
            let i = color.index();
            self.active[i] = active_features(pos, color);
            self.sums[i].fill(0.0);
            for &feature in &self.active[i] {
                accumulate(&mut self.sums[i], net, feature, 1.0);
            }
        }
    }

    /// The power sums for one perspective.
    pub fn perspective(&self, color: Color) -> &[f32] {
        &self.sums[color.index()]
    }

    /// Derive the accumulator for the position after `mv` from `self`, the one
    /// before it.
    ///
    /// `pos` is the pre-move position and is left byte-for-byte unchanged: the
    /// do/undo happens inside. Mirrors the NNUE accumulator's contract so that
    /// the search treats the two the same way.
    ///
    /// # Panics
    /// Panics if `pos` is missing either king, before or after `mv`.
    pub fn update_after_move(
        &self,
        net: &SymNetwork,
        pos: &mut Position,
        mv: attic_state::Move,
    ) -> Accumulator {
        let undo = pos.do_move(mv);
        let mut next = self.clone();
        // The one call site that has no cache to borrow from — the root, once
        // per iteration. The tree goes through `derive_into_cached`.
        let mut scratch = Scratch::default();
        next.rebase(net, pos, &mut scratch);
        pos.undo_move(mv, undo);
        next
    }

    /// Move `dst` to the position `post_pos` describes, starting from `src`.
    ///
    /// The `delta` and `cache` arguments are NNUE's and go unread: the diff
    /// this needs is over feature indices, which the two active lists already
    /// give, and there is no king bucket to cache a rebuild for.
    pub fn derive_into_cached(
        src: &Accumulator,
        dst: &mut Accumulator,
        net: &SymNetwork,
        post_pos: &Position,
        _delta: &crate::features::MoveDelta,
        cache: &mut FinnyCache,
    ) {
        // `clone_from` reuses `dst`'s allocations rather than replacing them,
        // which is what makes a node free of the heap once the stack is warm.
        dst.clone_from(src);
        dst.rebase(net, post_pos, &mut cache.scratch);
    }

    /// Bring `self` — currently holding some earlier position — to `pos`, by
    /// difference against the feature list it was built from.
    ///
    /// Every buffer it needs comes from `scratch`, so a node costs no
    /// allocation: the new feature list is written over the previous one's
    /// storage and then swapped in, and the diff reuses its own four vectors.
    fn rebase(&mut self, net: &SymNetwork, pos: &Position, scratch: &mut Scratch) {
        for color in [Color::Black, Color::White] {
            let i = color.index();
            active_features_into(pos, color, &mut scratch.after[i]);
            changed_indices_into(&self.active[i], &scratch.after[i], &mut scratch.diff);
            for &feature in &scratch.diff.removed {
                accumulate(&mut self.sums[i], net, feature, -1.0);
            }
            for &feature in &scratch.diff.added {
                accumulate(&mut self.sums[i], net, feature, 1.0);
            }
            // Swap rather than assign: `self.active[i]`'s allocation becomes
            // next ply's scratch instead of being dropped and re-made.
            std::mem::swap(&mut self.active[i], &mut scratch.after[i]);
        }
    }
}

/// Add or subtract one feature's powers, in place.
///
/// `sign` is `±1`, and the subtraction is exact in the sense that matters:
/// `add` then `sub` of the same row restores the sum bit-for-bit only up to
/// f32 rounding, which is why [`Accumulator::rebase`] always diffs against a
/// list it holds rather than composing many small deltas blindly.
fn accumulate(sums: &mut [f32], net: &SymNetwork, feature: FeatureIndex, sign: f32) {
    let row = &net.embed[feature as usize * DIM..(feature as usize + 1) * DIM];
    for (channel, &value) in row.iter().enumerate() {
        let mut power = value;
        for k in 0..ORDER {
            sums[k * DIM + channel] += sign * power;
            power *= value;
        }
    }
}

/// The buffers a derive needs, owned across the whole search.
///
/// Two feature lists and the diff's four vectors. Held here rather than in
/// [`Accumulator`] because the accumulator is copied down the tree and this is
/// not — one worker needs one set of these, not one per ply.
#[derive(Debug, Default)]
struct Scratch {
    after: [Vec<FeatureIndex>; Color::COUNT],
    diff: DiffScratch,
}

/// Where NNUE keeps its king-bucket refresh cache. Nothing here buckets by
/// king, so what it carries instead is the derive's scratch — the name is
/// NNUE's because the search allocates one per worker and hands it to every
/// derive, which is exactly what this needs too.
#[derive(Debug, Default)]
pub struct FinnyCache {
    scratch: Scratch,
}

impl FinnyCache {
    pub fn new() -> Box<Self> {
        Box::new(FinnyCache::default())
    }
}

/// Static evaluation from an already-built accumulator.
///
/// The entry point for search: no feature extraction and no power sums, just
/// the polynomials, the norm and the head.
pub fn evaluate_with(net: &SymNetwork, acc: &Accumulator, pos: &Position) -> i32 {
    let raw = value_from_sums(net, acc, pos.side_to_move()) * RAW_SCALE;
    (raw / crate::fv_scale() as f32) as i32
}
