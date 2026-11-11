//! The buffers a derive needs, owned across the whole search.

#[cfg(not(feature = "sym_token"))]
use crate::features::{FeatureIndex, MAX_ACTIVE_FEATURES};

/// Where NNUE keeps its king-bucket refresh cache. Nothing here caches a
/// bucket; what the KP and HalfKP shapes carry instead is the feature list a
/// half is rebuilt through when its own king moves (HalfKP), so that the rebuild
/// does not allocate. It is held here rather than in
/// [`Accumulator`](super::Accumulator) because the accumulator is copied down the
/// tree and this is not — one worker needs one list, not one per ply.
#[cfg(not(feature = "sym_token"))]
#[derive(Debug)]
pub struct FinnyCache {
    pub(super) features: Vec<FeatureIndex>,
}

#[cfg(not(feature = "sym_token"))]
impl Default for FinnyCache {
    fn default() -> Self {
        FinnyCache {
            features: Vec::with_capacity(MAX_ACTIVE_FEATURES),
        }
    }
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
