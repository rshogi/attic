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
//! A third feature set is the WebAssembly one: a build for a wasm target that enables `simd128`
//! gets `wasm_simd128` and `wasm_simd128_post_ft`, a complete set of kernels (the feature
//! transformer, the output transform and every dense layer) on 128-bit vectors. The optional
//! relaxed integer dot product is restricted to 7-bit activations, for which its result is exact.
//! Each kernel equals the scalar kernel bit for bit. A wasm build without `simd128` runs the scalar kernels. Their modules
//! compile only for a wasm target with the feature, being made of its intrinsics, and are tested
//! under a wasm runtime.
//!
//! The x86 backends stay **compiled unconditionally**, only the call sites being
//! `cfg`-gated, so that the SIMD-equals-scalar tests in each backend module keep
//! exercising both. Those tests probe the CPU at run time, so that they skip
//! gracefully on a host without the features. The only other run-time probe is
//! a test compiled solely into a `-C target-cpu=native` build, the one build
//! shape in which the host's features are the build's. Whichever backend a
//! build selects, the evaluation output is bit-identical.
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

// The feature transformer and element-wise kernels support whole 512-bit
// registers. The dense chains use avx512_post_ft for the 1536 shape and
// avx512_packed_tail for the 64-wide ones: the progress-based 1024 and 2304
// shapes and Cyfamate's threat shape.
#[allow(dead_code)]
#[cfg(all(target_arch = "x86_64", feature = "sfnn"))]
pub mod avx512;
#[allow(dead_code)]
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "sfnn_progress", feature = "sfnn_threat")
))]
pub mod avx512_packed_tail;
#[allow(dead_code)]
#[cfg(all(target_arch = "x86_64", feature = "sfnn"))]
pub mod avx512_post_ft;
// Index arithmetic only, so on every target: the network builds its group-major
// copies from it, and its tests run where AVX-512 does not.
#[cfg(any(feature = "sfnn_progress", feature = "sfnn_threat"))]
pub mod group_layout;

// The kernels for a WebAssembly build with `simd128` on (`-C target-feature=+simd128`), the
// same shapes of work as the AVX2 ones above on 128-bit vectors. Selected by the target
// features alone, like every other backend, and absent from every other build; a wasm build
// without `simd128` runs the scalar kernels.
#[allow(dead_code)]
#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
pub mod wasm_simd128;
#[allow(dead_code)]
#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
pub mod wasm_simd128_post_ft;

/// Which kernel backend this build compiled into the forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Portable scalar baseline (always correct, always available).
    Scalar,
    /// AVX-512 F + BW + VNNI kernels (the full SIMD forward pass).
    Avx512Vnni,
    /// WebAssembly `simd128` kernels (the full forward pass, on 128-bit vectors).
    WasmSimd128,
}

/// The kernel backend baked into this build.
///
/// [`Backend::Scalar`] covers the partial case of an `avx512f`+`avx512bw` build
/// without VNNI, where the feature transformer is SIMD but the layer stack is
/// not. The eval-parity test reads this to catch a build that silently compiled
/// scalar on a VNNI-capable host.
///
/// Every SFNN shape reports [`Backend::Avx512Vnni`] on such a build. The
/// ordinary NNUE shapes use AVX2 when it is enabled.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni",
    feature = "sfnn"
))]
pub const fn active_backend() -> Backend {
    Backend::Avx512Vnni
}

/// The kernel backend baked into this build, WebAssembly arm: every kernel of the forward
/// pass has a `simd128` form, so a build that enables the feature runs them all.
#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
pub const fn active_backend() -> Backend {
    Backend::WasmSimd128
}

/// The kernel backend baked into this build, non-VNNI arm.
#[cfg(not(any(
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vnni",
        feature = "sfnn"
    ),
    all(target_family = "wasm", target_feature = "simd128")
)))]
pub const fn active_backend() -> Backend {
    Backend::Scalar
}

/// Feature-transformer accumulate/update kernels, selected at compile time.
///
/// The AVX-512 arm is the SFNN shapes'. A standard shape on an AVX-512 target
/// falls to the AVX2 arm below, which every shape has.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    feature = "sfnn"
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

    /// `dst = src + sum(added) - sum(removed)` in one pass over the accumulator
    /// half: the child-from-parent derive, without the copy.
    #[inline]
    pub fn derive_features(
        dst: &mut [i16],
        src: &[i16],
        weights: &[i16],
        added: &[FeatureIndex],
        removed: &[FeatureIndex],
    ) {
        // SAFETY: see `add_features`.
        unsafe { avx512::derive_features(dst, src, weights, added, removed) }
    }

    /// Take the `removed` threat rows off `out` and put the `added` ones on (the threat
    /// shape's int8 rows, widened and added unscaled).
    #[cfg(feature = "sfnn_threat")]
    #[inline]
    pub fn apply_threat_rows(out: &mut [i16], weights: &[i8], removed: &[u32], added: &[u32]) {
        // SAFETY: see `add_features`.
        unsafe { avx512::apply_threat_rows(out, weights, removed, added) }
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
    not(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        feature = "sfnn"
    ))
))]
pub mod transformer_kernel {
    use super::avx2;
    use crate::features::FeatureIndex;

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

    /// `dst = src + sum(added) - sum(removed)` in one pass over the accumulator
    /// half: the child-from-parent derive, without the copy.
    #[inline]
    pub fn derive_features(
        dst: &mut [i16],
        src: &[i16],
        weights: &[i16],
        added: &[FeatureIndex],
        removed: &[FeatureIndex],
    ) {
        // SAFETY: see `add_features`.
        unsafe { avx2::derive_features(dst, src, weights, added, removed) }
    }

    /// Take the `removed` threat rows off `out` and put the `added` ones on (the threat
    /// shape's int8 rows, widened and added unscaled).
    #[cfg(feature = "sfnn_threat")]
    #[inline]
    pub fn apply_threat_rows(out: &mut [i16], weights: &[i8], removed: &[u32], added: &[u32]) {
        // SAFETY: see `add_features`.
        unsafe { avx2::apply_threat_rows(out, weights, removed, added) }
    }
}

/// The WebAssembly arm: a wasm build with `simd128`. The kernels are safe functions with the
/// scalar ones' signatures, so this arm re-exports them as the portable one does.
#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
pub mod transformer_kernel {
    #[cfg(feature = "sfnn_threat")]
    pub use super::wasm_simd128::apply_threat_rows;
    pub use super::wasm_simd128::{
        add_features, add_sub_features, add_sub_sub_features, derive_features, sub_features,
    };
}

/// The portable arm: neither AVX-512, AVX2 nor wasm `simd128`.
#[cfg(not(any(
    all(
        target_arch = "x86_64",
        any(
            all(target_feature = "avx512f", target_feature = "avx512bw"),
            target_feature = "avx2"
        )
    ),
    all(target_family = "wasm", target_feature = "simd128")
)))]
pub mod transformer_kernel {
    #[cfg(feature = "sfnn_threat")]
    pub use super::scalar::apply_threat_rows;
    pub use super::scalar::{
        add_features, add_sub_features, add_sub_sub_features, derive_features, sub_features,
    };
}

/// Output-transform and layer element-wise kernels, selected at compile time on
/// the same condition as [`transformer_kernel`].
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    feature = "sfnn"
))]
pub mod post_ft_kernel {
    use super::{avx512_post_ft, scalar_post_ft};

    /// Pairwise element-wise multiply for one perspective half. The threat shape
    /// uses [`ewm_one_perspective_file_scale`] instead.
    #[inline]
    #[cfg_attr(feature = "sfnn_threat", allow(dead_code))]
    pub fn ewm_one_perspective(half: &[i16], out: &mut [u8]) {
        // SAFETY: this module is compiled only into a build enabling avx512f +
        // avx512bw, exactly the features the callee names, and such a build only
        // ever runs on a host providing them.
        unsafe { avx512_post_ft::ewm_one_perspective(half, out) }
    }

    /// Pairwise element-wise multiply for one perspective of the threat shape, whose
    /// accumulator is at the scale the file holds: clamp to 127, multiply, shift by 7.
    #[cfg(feature = "sfnn_threat")]
    #[inline]
    pub fn ewm_one_perspective_file_scale(half: &[i16], out: &mut [u8]) {
        // SAFETY: see `ewm_one_perspective`.
        unsafe { avx512_post_ft::ewm_one_perspective_file_scale(half, out) }
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
    #[allow(dead_code)]
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

    /// [`affine_u7`] with `i16` accumulation across eight chunks, for weights
    /// that passed `scalar_post_ft::pair_sums_fit_i16`.
    #[inline]
    #[allow(dead_code)]
    pub fn affine_u7_wide(
        output: &mut [i32],
        biases: &[i32],
        weights: &[i8],
        input: &[u8],
        in_dims: usize,
        padded_in: usize,
    ) {
        // SAFETY: see `affine_u7`.
        unsafe {
            super::avx2_post_ft::affine_u7_wide(output, biases, weights, input, in_dims, padded_in)
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
    not(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        feature = "sfnn"
    ))
))]
pub mod post_ft_kernel {
    // Used by the SFNN chains only.
    #[allow(unused_imports)]
    pub use super::scalar_post_ft::{affine, sqr_clipped_relu};

    /// [`affine`] for inputs of at most 127 (`maddubs` saturates above that).
    /// Every layer of the KP/HalfKP chain is fed `clamp(0, 127)` bytes; the SFNN
    /// chains are not and keep the general kernel.
    #[inline]
    #[allow(dead_code)]
    pub fn affine_u7(
        output: &mut [i32],
        biases: &[i32],
        weights: &[i8],
        input: &[u8],
        in_dims: usize,
        padded_in: usize,
    ) {
        // With AVX-VNNI in the build, `vpdpbusd` does `maddubs` + `madd` + `add`
        // in one instruction and has no saturation to bound the input for.
        // SAFETY: the build enables avxvnni (and avx2, which this module needs),
        // exactly the features the callee names, and such a build only runs on
        // a host providing them.
        #[cfg(target_feature = "avxvnni")]
        unsafe {
            super::avx2_post_ft::affine_u7_vnni(output, biases, weights, input, in_dims, padded_in)
        }
        // SAFETY: this module is compiled only into a build that enables avx2 —
        // exactly what the callee's `#[target_feature]` names — and such a build
        // only runs on a host providing it. The caller guarantees the 7-bit
        // input.
        #[cfg(not(target_feature = "avxvnni"))]
        unsafe {
            super::avx2_post_ft::affine_u7(output, biases, weights, input, in_dims, padded_in)
        }
    }

    /// [`affine_u7`] with `i16` accumulation across eight chunks, for weights
    /// that passed `scalar_post_ft::pair_sums_fit_i16`.
    #[inline]
    #[allow(dead_code)]
    pub fn affine_u7_wide(
        output: &mut [i32],
        biases: &[i32],
        weights: &[i8],
        input: &[u8],
        in_dims: usize,
        padded_in: usize,
    ) {
        // With AVX-VNNI there is no `i16` sum to keep in range, so the wide form
        // is just the VNNI kernel; the weights' `pair_sums_fit_i16` is moot.
        // SAFETY: see `affine_u7`.
        #[cfg(target_feature = "avxvnni")]
        unsafe {
            super::avx2_post_ft::affine_u7_vnni(output, biases, weights, input, in_dims, padded_in)
        }
        // SAFETY: see `affine_u7`.
        #[cfg(not(target_feature = "avxvnni"))]
        unsafe {
            super::avx2_post_ft::affine_u7_wide(output, biases, weights, input, in_dims, padded_in)
        }
    }

    /// Pairwise element-wise multiply for one perspective half. Used by the SFNN
    /// output transform only.
    #[inline]
    #[allow(dead_code)]
    pub fn ewm_one_perspective(half: &[i16], out: &mut [u8]) {
        // SAFETY: this module is compiled only into a build that enables avx2 —
        // exactly what the callee's `#[target_feature]` names — and such a build
        // only runs on a host providing it.
        unsafe { super::avx2_post_ft::ewm_one_perspective(half, out) }
    }

    /// Pairwise element-wise multiply for one perspective of the threat shape, whose
    /// accumulator is at the scale the file holds: clamp to 127, multiply, shift by 7.
    #[cfg(feature = "sfnn_threat")]
    #[inline]
    pub fn ewm_one_perspective_file_scale(half: &[i16], out: &mut [u8]) {
        // SAFETY: see `ewm_one_perspective`.
        unsafe { super::avx2_post_ft::ewm_one_perspective_file_scale(half, out) }
    }

    /// Clipped ReLU.
    #[inline]
    pub fn clipped_relu(input: &[i32], output: &mut [u8]) {
        // SAFETY: see `ewm_one_perspective`.
        unsafe { super::avx2_post_ft::clipped_relu(input, output) }
    }
}

/// The WebAssembly arm: a wasm build with `simd128`. Complete, unlike the AVX2 arm: the
/// squared clipped ReLU has a vector form here, and so does the general `affine`, which the
/// `dot_i16x8` kernel serves for every input byte.
#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
pub mod post_ft_kernel {
    #[cfg(feature = "sfnn_threat")]
    pub use super::wasm_simd128_post_ft::ewm_one_perspective_file_scale;
    // The standard shapes alone call the `_wide` form and `clip_to_u7`, the SFNN chains alone the
    // squared ReLU, and the unfused reference flow alone the general `affine`.
    #[allow(unused_imports)]
    pub use super::wasm_simd128_post_ft::{
        affine, affine_u7, affine_u7_wide, clip_to_u7, clipped_relu, ewm_one_perspective,
        sqr_clipped_relu,
    };
}

/// The portable arm: neither AVX-512, AVX2 nor wasm `simd128`.
#[cfg(not(any(
    all(
        target_arch = "x86_64",
        any(
            all(target_feature = "avx512f", target_feature = "avx512bw"),
            target_feature = "avx2"
        )
    ),
    all(target_family = "wasm", target_feature = "simd128")
)))]
pub mod post_ft_kernel {
    #[cfg(feature = "sfnn_threat")]
    pub use super::scalar_post_ft::ewm_one_perspective_file_scale;
    // Each shape calls a subset of these.
    #[allow(unused_imports)]
    pub use super::scalar_post_ft::{
        affine, affine_u7, affine_u7_wide, clipped_relu, ewm_one_perspective, sqr_clipped_relu,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_backend_follows_compile_time_target_features() {
        // The SFNN shapes with VNNI kernels report the SIMD backend.
        let simd_build = cfg!(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vnni",
            feature = "sfnn"
        ));
        let expected = if simd_build {
            Backend::Avx512Vnni
        } else if cfg!(all(target_family = "wasm", target_feature = "simd128")) {
            Backend::WasmSimd128
        } else {
            Backend::Scalar
        };
        assert_eq!(active_backend(), expected);
    }

    // Only a `-C target-cpu=native` build promises the host's features; under a
    // fixed baseline (CI pins `x86-64-v3`) the scalar backend is correct on any
    // CPU, a runner that happens to have AVX-512 VNNI included. The shapes with
    // the AVX-512 kernels must pick them up.
    #[cfg(all(target_cpu_native, target_arch = "x86_64", feature = "sfnn"))]
    #[test]
    fn native_build_on_a_vnni_host_is_the_simd_build() {
        let has_vnni = std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni");
        if has_vnni {
            assert_eq!(
                active_backend(),
                Backend::Avx512Vnni,
                "a `-C target-cpu=native` build on an AVX-512 VNNI host compiled \
                 the scalar backend",
            );
        }
    }
}
