//! Evaluation-layer NNUE support for the SFNN-1536 architecture: loading and
//! validating the `nn.bin` network file, feature extraction, the accumulator,
//! the forward pass and the SIMD kernels.
//!
//! It also carries two evaluators that are not NNUE, selected by shape like the
//! rest: `manaka` for the `synn_*` shapes and `hce` for the hand-crafted ones.
//! Neither loads an `nn.bin`.
//!
//! It may import State and Storage, never Protocol or Search.

mod aligned;
// The hand-crafted shapes count pieces and attacks straight off the `Position`
// and number nothing in BonaPiece's space, so there is no feature index for
// this module to hand them. Manaka's own token space is numbered in a space
// that is not BonaPiece's either, and takes `tokens` in its place: the two
// modules offer the search the same names.
#[cfg(not(any(feature = "hce", feature = "sym_token")))]
mod features;
mod scale;
#[cfg(feature = "sym_token")]
mod tokens;
mod types;

#[cfg(not(any(feature = "sym", feature = "hce")))]
mod finny;
#[cfg(not(any(feature = "sym", feature = "hce")))]
mod loader;
#[cfg(not(any(feature = "sym", feature = "hce")))]
mod network;
#[cfg(not(any(feature = "sym", feature = "hce")))]
pub mod progress;
#[cfg(not(any(feature = "sym", feature = "hce")))]
mod simd;
#[cfg(not(any(feature = "sym", feature = "hce")))]
mod transformer;

#[cfg(feature = "sym")]
pub mod manaka;

#[cfg(feature = "hce")]
pub mod hce;

// Under a Manaka shape these are what the search binds to. The names are
// NNUE's because the search is written against them and the two evaluators
// are interchangeable at this surface — which is the point: a shape swap is
// a feature flag, not an edit to `qsearch`.
#[cfg(feature = "sym")]
pub use manaka::{
    Accumulator, FinnyCache, SymNetwork as NnueNetwork, evaluate, evaluate_with, load_network,
    load_network_with_warnings,
};

// And the same names again for the hand-crafted shapes. `Accumulator`,
// `FinnyCache` and the deltas carry nothing there, which the search never has
// to know.
#[cfg(feature = "hce")]
pub use hce::{
    Accumulator, FinnyCache, HceNetwork as NnueNetwork, MoveDelta, PerspectiveDelta, evaluate,
    evaluate_with, load_network, load_network_with_warnings,
};

pub use aligned::Aligned64;
#[cfg(not(any(feature = "hce", feature = "sym_token")))]
pub use features::{
    FeatureIndex, MAX_ACTIVE_FEATURES, active_features, active_features_both, requires_full_refresh,
};
#[cfg(feature = "sym_token")]
pub use tokens::{
    FeatureIndex, MAX_ACTIVE_FEATURES, active_features, active_features_both, requires_full_refresh,
};
// The scale is what the network evaluators divide their fixed-point output by,
// and the USI layer drives it from the `FV_SCALE` option whichever shape is
// loaded. The hand-crafted shapes were fitted in internal units and never
// consult it, but the option stays accepted so that one driver serves every
// shape.
pub use scale::{FV_SCALE_DEFAULT, fv_scale, set_fv_scale};
pub use types::{NUM_FEATURES, NnueError};

// `FEATURE_DIMENSION` is HalfKA_hm2's own product and means nothing to the
// shapes that do not cross a piece on a mirrored king.
#[cfg(feature = "sfnn")]
pub use features::FEATURE_DIMENSION;

#[cfg(not(any(feature = "hce", feature = "sym_token")))]
pub use features::{MoveDelta, PerspectiveDelta};
#[cfg(not(any(feature = "sym", feature = "hce")))]
pub use finny::FinnyCache;
#[cfg(not(any(feature = "sym", feature = "hce")))]
pub use loader::{load_network, load_network_with_dims, load_network_with_warnings};
#[cfg(not(any(feature = "sym", feature = "hce")))]
pub use network::{evaluate, evaluate_with, layer_stack_index};
#[cfg(not(any(feature = "sym", feature = "hce")))]
pub use simd::{Backend, active_backend};
#[cfg(feature = "sym_token")]
pub use tokens::{MoveDelta, PerspectiveDelta};
#[cfg(not(any(feature = "sym", feature = "hce")))]
pub use transformer::{Accumulator, FT_OUTPUT_DIMS};
#[cfg(not(any(feature = "sym", feature = "hce")))]
pub use types::{
    FC_0_INPUT_DIMS, FC_0_OUTPUT_DIMS, FC_0_PADDED_INPUT_DIMS, FC_1_INPUT_DIMS, FC_1_OUTPUT_DIMS,
    FC_1_PADDED_INPUT_DIMS, FC_2_INPUT_DIMS, FC_2_OUTPUT_DIMS, FC_2_PADDED_INPUT_DIMS, HIDDEN_SIZE,
    HIDDEN1_DIMS, HIDDEN2_DIMS, LAYER_STACKS, NetDims, NetHeader, NetworkStack, NnueNetwork,
    NnueNetworkBuilder,
};
