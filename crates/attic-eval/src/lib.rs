//! Evaluation-layer NNUE support for the SFNN-1536 architecture: loading and
//! validating the `nn.bin` network file, feature extraction, the accumulator,
//! the forward pass and the SIMD kernels.
//!
//! It may import State and Storage, never Protocol or Search.

mod aligned;
mod features;
mod scale;
mod types;

#[cfg(not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512")))]
mod finny;
#[cfg(not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512")))]
mod loader;
#[cfg(not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512")))]
mod network;
#[cfg(not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512")))]
pub mod progress;
#[cfg(not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512")))]
mod simd;
#[cfg(not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512")))]
mod transformer;

#[cfg(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512"))]
pub mod manaka;

// Under a Manaka shape these are what the search binds to. The names are
// NNUE's because the search is written against them and the two evaluators
// are interchangeable at this surface — which is the point: a shape swap is
// a feature flag, not an edit to `qsearch`.
#[cfg(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512"))]
pub use manaka::{
    Accumulator, FinnyCache, SymNetwork as NnueNetwork, evaluate, evaluate_with, load_network,
    load_network_with_warnings,
};

pub use aligned::Aligned64;
pub use features::{
    FeatureIndex, MAX_ACTIVE_FEATURES, active_features, active_features_both,
    requires_full_refresh,
};
// The scale is shared: both evaluators divide their output by it, and the
// USI layer drives it from the `FV_SCALE` option whichever shape is loaded.
pub use scale::{FV_SCALE_DEFAULT, fv_scale, set_fv_scale};
pub use types::{NUM_FEATURES, NnueError};

// `FEATURE_DIMENSION` is HalfKA_hm2's own product and means nothing to the
// shapes that do not cross a piece on a mirrored king.
#[cfg(any(
    feature = "sfnn_halfkahm2_1536_15_32_k3k3",
    feature = "sfnn_halfkahm2_2304_15_64_progress8ek"
))]
pub use features::FEATURE_DIMENSION;

pub use features::{MoveDelta, PerspectiveDelta};
#[cfg(not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512")))]
pub use finny::FinnyCache;
#[cfg(not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512")))]
pub use loader::{load_network, load_network_with_dims, load_network_with_warnings};
#[cfg(not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512")))]
pub use network::{evaluate, evaluate_with, layer_stack_index};
#[cfg(not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512")))]
pub use simd::{Backend, active_backend};
#[cfg(not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512")))]
pub use transformer::{Accumulator, FT_OUTPUT_DIMS};
#[cfg(not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512")))]
pub use types::{
    FC_0_INPUT_DIMS, FC_0_OUTPUT_DIMS, FC_0_PADDED_INPUT_DIMS, FC_1_INPUT_DIMS, FC_1_OUTPUT_DIMS,
    FC_1_PADDED_INPUT_DIMS, FC_2_INPUT_DIMS, FC_2_OUTPUT_DIMS, FC_2_PADDED_INPUT_DIMS, HIDDEN_SIZE,
    HIDDEN1_DIMS, HIDDEN2_DIMS, LAYER_STACKS, NetDims, NetHeader, NetworkStack, NnueNetwork,
    NnueNetworkBuilder,
};
