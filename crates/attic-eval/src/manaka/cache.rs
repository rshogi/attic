//! The buffers a derive needs, owned across the whole search.

#[cfg(not(feature = "sym_token"))]
use attic_state::Color;

#[cfg(not(feature = "sym_token"))]
use crate::features::{DiffScratch, FeatureIndex};

/// Two feature lists and the diff's four vectors. Held here rather than in
/// [`Accumulator`](super::Accumulator) because the accumulator is copied down the
/// tree and this is not — one worker needs one set of these, not one per ply.
#[cfg(not(feature = "sym_token"))]
#[derive(Debug, Default)]
pub(super) struct Scratch {
    pub(super) after: [Vec<FeatureIndex>; Color::COUNT],
    pub(super) diff: DiffScratch,
}

/// Where NNUE keeps its king-bucket refresh cache. Nothing here buckets by
/// king, so what it carries instead is the derive's scratch — the name is
/// NNUE's because the search allocates one per worker and hands it to every
/// derive, which is exactly what this needs too.
#[cfg(not(feature = "sym_token"))]
#[derive(Debug, Default)]
pub struct FinnyCache {
    pub(super) scratch: Scratch,
}

/// Where NNUE keeps its king-bucket refresh cache. The token shapes have
/// neither a bucket nor a scan to buffer, so this holds nothing; it exists
/// because the search allocates one per worker and hands it to every derive.
#[cfg(feature = "sym_token")]
#[derive(Debug, Default)]
pub struct FinnyCache;

impl FinnyCache {
    pub fn new() -> Box<Self> {
        // `Box::default`, not `Box::new(FinnyCache::default())`: the token
        // shapes' cache is a unit struct, and clippy rejects calling `default`
        // on one. This spelling is the same for both.
        Box::default()
    }
}
