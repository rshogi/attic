//! Scalar/SIMD kernel selection for the NNUE forward pass.
//!
//! The backend is chosen **at compile time** from the CPU features the build
//! itself enables. This repo builds with `-C target-cpu=native` and runs each
//! binary on the machine that produced it, so "what the build enables" is the
//! host CPU's feature set, and a run-time probe could never pick anything the
//! build had not already fixed. Two feature sets matter: `avx512f` + `avx512bw`
//! for the feature-transformer and element-wise kernels, and those plus
//! `avx512vnni` for the fused fc chain.
//!
//! Both backends stay **compiled unconditionally**, only the call sites being
//! `cfg`-gated, so that the SIMD-equals-scalar tests in each backend module keep
//! exercising both. Those tests, and only those, probe the CPU at run time, so
//! that they skip gracefully on a host without the features. Whichever backend
//! a build selects, the evaluation output is bit-identical.
//!
//! ## Safety invariants
//!
//! Every AVX-512 entry point is an `unsafe fn` carrying a
//! `#[target_feature(enable = ...)]` attribute, so calling it is sound only when
//! the named features are present on the running CPU. The wrappers in this
//! module are the sole callers in non-test code, and each `unsafe` call sits
//! behind a `cfg` gate naming exactly the features its callee enables.

// Compiling both backends unconditionally leaves whichever one this build did
// not select without a caller outside `cfg(test)`.
#[allow(dead_code)]
pub mod scalar;
#[allow(dead_code)]
pub mod scalar_post_ft;

// The feature-transformer kernels for a CPU with AVX2 but no AVX-512 (Zen2,
// which is what the 2304 shape is generated on). Compiled unconditionally like
// the others so its scalar-parity tests always build.
#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
pub mod avx2_post_ft;

// Excluded under `sfnn_halfkahm2_2304_15_64_progress8ek`: the post-FT chain there ends in `affine_padded32`
// helpers that take a fixed `[u8; 32]`, and fc_1's output is 64 in that shape.
// Upstream C++ takes a separate `kUsePackedTail` path for exactly this case
// (`AVX512 && kHidden2Dims == 64`), which has no port here yet. The target for
// the 2304 shape is Zen2, which has no AVX-512 at all, so nothing is lost by
// not compiling it — write the packed-tail kernels before running 2304 on an
// AVX-512 host.
#[allow(dead_code)]
#[cfg(all(target_arch = "x86_64", feature = "sfnn_halfkahm2_1536_15_32_k3k3"))]
pub mod avx512;
#[allow(dead_code)]
#[cfg(all(target_arch = "x86_64", feature = "sfnn_halfkahm2_1536_15_32_k3k3"))]
pub mod avx512_post_ft;

/// Which kernel backend this build compiled into the forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Portable scalar baseline (always correct, always available).
    Scalar,
    /// AVX-512 F + BW + VNNI kernels (the full SIMD forward pass).
    Avx512Vnni,
}

/// The kernel backend baked into this build.
///
/// [`Backend::Scalar`] covers the partial case of an `avx512f`+`avx512bw` build
/// without VNNI, where the feature transformer is SIMD but the layer stack is
/// not. The eval-parity test reads this to catch a build that silently compiled
/// scalar on a VNNI-capable host.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni"
))]
pub const fn active_backend() -> Backend {
    Backend::Avx512Vnni
}

/// The kernel backend baked into this build, non-VNNI arm.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni"
)))]
pub const fn active_backend() -> Backend {
    Backend::Scalar
}

/// Feature-transformer accumulate/update kernels, selected at compile time.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
pub mod transformer_kernel {
    use super::avx512;
    use crate::features::FeatureIndex;

    /// Add each active feature's FT weight column into `out`.
    #[inline]
    pub fn add_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
        // SAFETY: this module is compiled only into a build enabling avx512f +
        // avx512bw, exactly the features the callee names, and such a build only
        // ever runs on a host providing them.
        unsafe { avx512::add_features(out, weights, indices) }
    }

    /// Subtract each feature's FT weight column from `out`.
    #[inline]
    pub fn sub_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
        // SAFETY: see `add_features`.
        unsafe { avx512::sub_features(out, weights, indices) }
    }

    /// Fused single-add / single-sub delta.
    #[inline]
    pub fn add_sub_features(
        out: &mut [i16],
        weights: &[i16],
        added: &[FeatureIndex],
        removed: &[FeatureIndex],
    ) {
        // SAFETY: see `add_features`.
        unsafe { avx512::add_sub_features(out, weights, added, removed) }
    }

    /// Fused single-add / double-sub delta (capture-style updates).
    #[inline]
    pub fn add_sub_sub_features(
        out: &mut [i16],
        weights: &[i16],
        added: &[FeatureIndex],
        removed_a: &[FeatureIndex],
        removed_b: &[FeatureIndex],
    ) {
        // SAFETY: see `add_features`.
        unsafe { avx512::add_sub_sub_features(out, weights, added, removed_a, removed_b) }
    }
}

/// Feature-transformer accumulate/update kernels — scalar arm (the AVX-512 arm
/// above carries the selection rule).
/// The AVX2 arm: a host with AVX2 but not AVX-512 (Zen2). Conditions are spelled
/// out rather than leaning on `not(avx512)`, so adding a third arm cannot
/// silently change what the others match.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512bw"))
))]
pub mod transformer_kernel {
    use crate::features::FeatureIndex;
    use super::avx2;

    /// Add each active feature's FT weight column into `out`.
    #[inline]
    pub fn add_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
        // SAFETY: this module is compiled only into a build that enables avx2 —
        // exactly what the callee's `#[target_feature]` names — and such a build
        // only runs on a host providing it (`-C target-cpu=native`; build and
        // run happen on the same machine).
        unsafe { avx2::add_features(out, weights, indices) }
    }

    /// Subtract each feature's FT weight column from `out`.
    #[inline]
    pub fn sub_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
        // SAFETY: see `add_features`.
        unsafe { avx2::sub_features(out, weights, indices) }
    }

    /// Fused single-add / single-sub delta.
    #[inline]
    pub fn add_sub_features(
        out: &mut [i16],
        weights: &[i16],
        added: &[FeatureIndex],
        removed: &[FeatureIndex],
    ) {
        // SAFETY: see `add_features`.
        unsafe { avx2::add_sub_features(out, weights, added, removed) }
    }

    /// Fused single-add / double-sub delta (a capture).
    #[inline]
    pub fn add_sub_sub_features(
        out: &mut [i16],
        weights: &[i16],
        added: &[FeatureIndex],
        removed_a: &[FeatureIndex],
        removed_b: &[FeatureIndex],
    ) {
        // SAFETY: see `add_features`.
        unsafe { avx2::add_sub_sub_features(out, weights, added, removed_a, removed_b) }
    }
}

/// The portable arm: neither AVX-512 nor AVX2.
#[cfg(not(all(
    target_arch = "x86_64",
    any(
        all(target_feature = "avx512f", target_feature = "avx512bw"),
        target_feature = "avx2"
    )
)))]
pub mod transformer_kernel {
    pub use super::scalar::{add_features, add_sub_features, add_sub_sub_features, sub_features};
}

/// Output-transform and layer element-wise kernels, selected at compile time on
/// the same condition as [`transformer_kernel`].
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
pub mod post_ft_kernel {
    use super::{avx512_post_ft, scalar_post_ft};

    /// Pairwise element-wise multiply for one perspective half.
    #[inline]
    pub fn ewm_one_perspective(half: &[i16], out: &mut [u8]) {
        // SAFETY: this module is compiled only into a build enabling avx512f +
        // avx512bw, exactly the features the callee names, and such a build only
        // ever runs on a host providing them.
        unsafe { avx512_post_ft::ewm_one_perspective(half, out) }
    }

    /// Clipped ReLU.
    #[inline]
    pub fn clipped_relu(input: &[i32], output: &mut [u8]) {
        // SAFETY: see `ewm_one_perspective`.
        unsafe { avx512_post_ft::clipped_relu(input, output) }
    }

    /// Squared clipped ReLU.
    #[inline]
    pub fn sqr_clipped_relu(input: &[i32], output: &mut [u8]) {
        // SAFETY: see `ewm_one_perspective`.
        unsafe { avx512_post_ft::sqr_clipped_relu(input, output) }
    }

    /// Integer affine transform. The AVX-512 form needs VNNI and is only worth
    /// its setup at the wide fc_0 shape, which the fused chain already covers,
    /// so this per-layer entry point always uses the scalar kernel.
    #[inline]
    pub fn affine(
        output: &mut [i32],
        biases: &[i32],
        weights: &[i8],
        input: &[u8],
        in_dims: usize,
        padded_in: usize,
    ) {
        scalar_post_ft::affine(output, biases, weights, input, in_dims, padded_in);
    }

    /// [`affine`] for inputs of at most 127, which the KP/HalfKP chain
    /// guarantees: the AVX2 kernel, since `avx512bw` implies `avx2`.
    #[inline]
    pub fn affine_u7(
        output: &mut [i32],
        biases: &[i32],
        weights: &[i8],
        input: &[u8],
        in_dims: usize,
        padded_in: usize,
    ) {
        // SAFETY: a build enabling avx512bw enables avx2 with it, and only runs
        // on a host providing both.
        unsafe {
            super::avx2_post_ft::affine_u7(output, biases, weights, input, in_dims, padded_in)
        }
    }
}

/// Output-transform and layer element-wise kernels — scalar arm (the AVX-512 arm
/// above carries the selection rule).
/// The AVX2 arm. Partial on purpose: `ewm_one_perspective` and `clipped_relu`
/// are ported, `sqr_clipped_relu` is not.
///
/// That one squares into `i64` and needs `srai_epi64` / `min_epi64`, neither of
/// which AVX2 has, plus an `i64 -> u8` narrowing with no direct instruction. It
/// is the least frequent of the three, so it stays scalar until the two cheap
/// ones are shown to pay.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512bw"))
))]
pub mod post_ft_kernel {
    pub use super::scalar_post_ft::{affine, sqr_clipped_relu};

    /// [`affine`] for inputs of at most 127 (`maddubs` saturates above that).
    /// Every layer of the KP/HalfKP chain is fed `clamp(0, 127)` bytes; the SFNN
    /// chains are not and keep the general kernel.
    #[inline]
    pub fn affine_u7(
        output: &mut [i32],
        biases: &[i32],
        weights: &[i8],
        input: &[u8],
        in_dims: usize,
        padded_in: usize,
    ) {
        // SAFETY: this module is compiled only into a build that enables avx2 —
        // exactly what the callee's `#[target_feature]` names — and such a build
        // only runs on a host providing it. The caller guarantees the 7-bit
        // input.
        unsafe {
            super::avx2_post_ft::affine_u7(output, biases, weights, input, in_dims, padded_in)
        }
    }

    /// Pairwise element-wise multiply for one perspective half.
    #[inline]
    pub fn ewm_one_perspective(half: &[i16], out: &mut [u8]) {
        // SAFETY: this module is compiled only into a build that enables avx2 —
        // exactly what the callee's `#[target_feature]` names — and such a build
        // only runs on a host providing it.
        unsafe { super::avx2_post_ft::ewm_one_perspective(half, out) }
    }

    /// Clipped ReLU.
    #[inline]
    pub fn clipped_relu(input: &[i32], output: &mut [u8]) {
        // SAFETY: see `ewm_one_perspective`.
        unsafe { super::avx2_post_ft::clipped_relu(input, output) }
    }
}

/// The portable arm: neither AVX-512 nor AVX2.
#[cfg(not(all(
    target_arch = "x86_64",
    any(
        all(target_feature = "avx512f", target_feature = "avx512bw"),
        target_feature = "avx2"
    )
)))]
pub mod post_ft_kernel {
    pub use super::scalar_post_ft::{
        affine, affine_u7, clipped_relu, ewm_one_perspective, sqr_clipped_relu,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_backend_is_avx512_when_cpu_supports_vnni() {
        // Backend selection is compile-time, so this checks the *build* was
        // configured for its host: under `-C target-cpu=native` a VNNI-capable
        // CPU must yield the SIMD backend, and a scalar one here means the build
        // did not enable the host's features.
        #[cfg(target_arch = "x86_64")]
        {
            let has_vnni = std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vnni");
            if has_vnni {
                assert_eq!(
                    active_backend(),
                    Backend::Avx512Vnni,
                    "CPU reports AVX-512 VNNI but this build compiled the scalar \
                     backend — check that `-C target-cpu=native` is in effect",
                );
            } else {
                assert_eq!(active_backend(), Backend::Scalar);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        assert_eq!(active_backend(), Backend::Scalar);
    }
}
