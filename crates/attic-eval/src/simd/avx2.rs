//! AVX2 feature-transformer accumulate/update kernels.
//!
//! The same four operations as [`crate::simd::avx512`], on 256-bit registers.
//! This is the backend Zen2 gets: EPYC 7742 has AVX2 but no AVX-512 at all, so
//! without these the accumulator update — the most frequent work in the whole
//! evaluation — runs scalar while the reference build it is compared against
//! runs `64AVX2`.
//!
//! The port is mechanical: these kernels are pure element-wise `i16` add/sub
//! over contiguous columns. There are no mask registers to emulate, no
//! narrowing converts, and no lane-crossing shuffles — the three things that
//! make the post-FT kernels hard. Halving the lane count is the whole change.
//!
//! Both shapes divide evenly: `1536 / 16 = 96` and `2304 / 16 = 144`, so no
//! tail handling is needed.
//!
//! `wrapping_add`/`wrapping_sub` semantics carry over: `_mm256_add_epi16` wraps
//! on overflow exactly as the scalar baseline does, which the parity tests pin.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::features::FeatureIndex;
use crate::types::HIDDEN_SIZE;

const LANES: usize = 16;
const NUM_CHUNKS: usize = HIDDEN_SIZE / LANES;

/// # Safety
/// The running CPU must support `avx2`. `out` and every referenced weight
/// column must be `HIDDEN_SIZE` long.
#[target_feature(enable = "avx2")]
pub unsafe fn add_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
    debug_assert_eq!(out.len(), HIDDEN_SIZE);
    let out_ptr = out.as_mut_ptr();
    for &idx in indices {
        let base = idx as usize * HIDDEN_SIZE;
        let col_ptr = weights[base..base + HIDDEN_SIZE].as_ptr();
        for chunk in 0..NUM_CHUNKS {
            let offset = chunk * LANES;
            // SAFETY: chunk*LANES+LANES <= HIDDEN_SIZE = out.len() = col.len();
            // the unaligned 256-bit ops impose no alignment requirement.
            unsafe {
                let o = _mm256_loadu_si256(out_ptr.add(offset).cast::<__m256i>());
                let w = _mm256_loadu_si256(col_ptr.add(offset).cast::<__m256i>());
                _mm256_storeu_si256(
                    out_ptr.add(offset).cast::<__m256i>(),
                    _mm256_add_epi16(o, w),
                );
            }
        }
    }
}

/// # Safety
/// See [`add_features`].
#[target_feature(enable = "avx2")]
pub unsafe fn sub_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
    debug_assert_eq!(out.len(), HIDDEN_SIZE);
    let out_ptr = out.as_mut_ptr();
    for &idx in indices {
        let base = idx as usize * HIDDEN_SIZE;
        let col_ptr = weights[base..base + HIDDEN_SIZE].as_ptr();
        for chunk in 0..NUM_CHUNKS {
            let offset = chunk * LANES;
            // SAFETY: see `add_features`.
            unsafe {
                let o = _mm256_loadu_si256(out_ptr.add(offset).cast::<__m256i>());
                let w = _mm256_loadu_si256(col_ptr.add(offset).cast::<__m256i>());
                _mm256_storeu_si256(
                    out_ptr.add(offset).cast::<__m256i>(),
                    _mm256_sub_epi16(o, w),
                );
            }
        }
    }
}

/// # Safety
/// See [`add_features`].
#[target_feature(enable = "avx2")]
pub unsafe fn add_sub_features(
    out: &mut [i16],
    weights: &[i16],
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    debug_assert_eq!(out.len(), HIDDEN_SIZE);
    let out_ptr = out.as_mut_ptr();
    // Chunk-major so the accumulator slot stays in a register across every
    // column, as the AVX-512 version does — one load and one store per chunk
    // however many features moved.
    for chunk in 0..NUM_CHUNKS {
        let offset = chunk * LANES;
        // SAFETY: `chunk * LANES + LANES <= HIDDEN_SIZE = out.len()`.
        let mut acc = unsafe { _mm256_loadu_si256(out_ptr.add(offset).cast::<__m256i>()) };
        for &idx in added {
            let base = idx as usize * HIDDEN_SIZE;
            let col_ptr = weights[base..base + HIDDEN_SIZE].as_ptr();
            // SAFETY: `col` is `HIDDEN_SIZE`-long, same offset bound.
            let w = unsafe { _mm256_loadu_si256(col_ptr.add(offset).cast::<__m256i>()) };
            acc = _mm256_add_epi16(acc, w);
        }
        for &idx in removed {
            let base = idx as usize * HIDDEN_SIZE;
            let col_ptr = weights[base..base + HIDDEN_SIZE].as_ptr();
            // SAFETY: see above.
            let w = unsafe { _mm256_loadu_si256(col_ptr.add(offset).cast::<__m256i>()) };
            acc = _mm256_sub_epi16(acc, w);
        }
        // SAFETY: same offset bound on `out`.
        unsafe { _mm256_storeu_si256(out_ptr.add(offset).cast::<__m256i>(), acc) };
    }
}

/// # Safety
/// See [`add_features`].
#[target_feature(enable = "avx2")]
pub unsafe fn add_sub_sub_features(
    out: &mut [i16],
    weights: &[i16],
    added: &[FeatureIndex],
    removed_a: &[FeatureIndex],
    removed_b: &[FeatureIndex],
) {
    debug_assert_eq!(out.len(), HIDDEN_SIZE);
    let out_ptr = out.as_mut_ptr();
    for chunk in 0..NUM_CHUNKS {
        let offset = chunk * LANES;
        // SAFETY: `chunk * LANES + LANES <= HIDDEN_SIZE = out.len()`.
        let mut acc = unsafe { _mm256_loadu_si256(out_ptr.add(offset).cast::<__m256i>()) };
        for &idx in added {
            let base = idx as usize * HIDDEN_SIZE;
            let col_ptr = weights[base..base + HIDDEN_SIZE].as_ptr();
            // SAFETY: `col` is `HIDDEN_SIZE`-long, same offset bound.
            let w = unsafe { _mm256_loadu_si256(col_ptr.add(offset).cast::<__m256i>()) };
            acc = _mm256_add_epi16(acc, w);
        }
        for &idx in removed_a {
            let base = idx as usize * HIDDEN_SIZE;
            let col_ptr = weights[base..base + HIDDEN_SIZE].as_ptr();
            // SAFETY: see above.
            let w = unsafe { _mm256_loadu_si256(col_ptr.add(offset).cast::<__m256i>()) };
            acc = _mm256_sub_epi16(acc, w);
        }
        for &idx in removed_b {
            let base = idx as usize * HIDDEN_SIZE;
            let col_ptr = weights[base..base + HIDDEN_SIZE].as_ptr();
            // SAFETY: see above.
            let w = unsafe { _mm256_loadu_si256(col_ptr.add(offset).cast::<__m256i>()) };
            acc = _mm256_sub_epi16(acc, w);
        }
        // SAFETY: same offset bound on `out`.
        unsafe { _mm256_storeu_si256(out_ptr.add(offset).cast::<__m256i>(), acc) };
    }
}

/// One-pass derive: `dst = src + sum(added) - sum(removed)`, per lane, wrapping.
///
/// The search derives a child accumulator from its parent on every `do_move`.
/// Copying the parent and then applying the delta in place reads and writes the
/// 512-byte half twice; here each 32-byte chunk is loaded from `src` once, has
/// every column applied in a register, and is stored to `dst` once. The `i16`
/// add/sub wrap and commute, so the order of the columns does not matter and
/// the result equals the copy-then-apply form bit for bit.
///
/// A [`PerspectiveDelta`](crate::features::PerspectiveDelta) carries at most
/// two columns per side, so every combination up to 2 x 2 is its own
/// monomorphization with the column pointers resolved (and bounds-checked)
/// once, outside the chunk loop. Longer lists fall back to copy-then-apply.
///
/// # Safety
/// The running CPU must support `avx2`. `dst` and `src` must both be
/// `HIDDEN_SIZE` long (asserted).
#[target_feature(enable = "avx2")]
pub unsafe fn derive_features(
    dst: &mut [i16],
    src: &[i16],
    weights: &[i16],
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    assert_eq!(dst.len(), HIDDEN_SIZE);
    assert_eq!(src.len(), HIDDEN_SIZE);
    // SAFETY: the callees need avx2, which this function's attribute grants.
    unsafe {
        match (added, removed) {
            ([], []) => derive_fixed::<0, 0>(dst, src, weights, [], []),
            ([a], []) => derive_fixed::<1, 0>(dst, src, weights, [*a], []),
            ([], [r]) => derive_fixed::<0, 1>(dst, src, weights, [], [*r]),
            ([a], [r]) => derive_fixed::<1, 1>(dst, src, weights, [*a], [*r]),
            ([a0, a1], [r]) => derive_fixed::<2, 1>(dst, src, weights, [*a0, *a1], [*r]),
            ([a], [r0, r1]) => derive_fixed::<1, 2>(dst, src, weights, [*a], [*r0, *r1]),
            ([a0, a1], [r0, r1]) => derive_fixed::<2, 2>(dst, src, weights, [*a0, *a1], [*r0, *r1]),
            _ => {
                dst.copy_from_slice(src);
                sub_features(dst, weights, removed);
                add_features(dst, weights, added);
            }
        }
    }
}

/// [`derive_features`] for exactly `A` added and `R` removed columns.
///
/// # Safety
/// See [`derive_features`]; `dst` and `src` are `HIDDEN_SIZE` long.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn derive_fixed<const A: usize, const R: usize>(
    dst: &mut [i16],
    src: &[i16],
    weights: &[i16],
    added: [FeatureIndex; A],
    removed: [FeatureIndex; R],
) {
    // Slicing checks each column against `weights` once, here, not per chunk.
    let a_cols: [*const i16; A] =
        std::array::from_fn(|k| weights[added[k] as usize * HIDDEN_SIZE..][..HIDDEN_SIZE].as_ptr());
    let r_cols: [*const i16; R] = std::array::from_fn(|k| {
        weights[removed[k] as usize * HIDDEN_SIZE..][..HIDDEN_SIZE].as_ptr()
    });
    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();
    for chunk in 0..NUM_CHUNKS {
        let offset = chunk * LANES;
        // SAFETY: `offset + LANES <= HIDDEN_SIZE`, the length of `src`, `dst`
        // and every column.
        unsafe {
            let mut acc = _mm256_loadu_si256(src_ptr.add(offset).cast::<__m256i>());
            for col in a_cols {
                acc = _mm256_add_epi16(acc, _mm256_loadu_si256(col.add(offset).cast::<__m256i>()));
            }
            for col in r_cols {
                acc = _mm256_sub_epi16(acc, _mm256_loadu_si256(col.add(offset).cast::<__m256i>()));
            }
            _mm256_storeu_si256(dst_ptr.add(offset).cast::<__m256i>(), acc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::scalar;

    /// Skip unless the CPU has AVX2, so the parity checks run where they can
    /// and are skipped — not failed — elsewhere.
    macro_rules! require_avx2 {
        () => {
            if !std::arch::is_x86_feature_detected!("avx2") {
                eprintln!("skipping AVX2 parity test: avx2 unavailable");
                return;
            }
        };
    }

    const FEATURES: usize = 6;

    fn fill_weights(weights: &mut [i16], seed: i16) {
        for (i, slot) in weights.iter_mut().enumerate() {
            // Spread across the i16 range so wrapping is actually exercised.
            *slot = (i as i16)
                .wrapping_mul(31)
                .wrapping_add(seed)
                .wrapping_mul(7);
        }
    }

    fn seeded_out(seed: i16) -> Vec<i16> {
        (0..HIDDEN_SIZE)
            .map(|i| (i as i16).wrapping_mul(17).wrapping_add(seed))
            .collect()
    }

    fn corpus() -> (Vec<i16>, Vec<FeatureIndex>) {
        let mut weights = vec![0i16; FEATURES * HIDDEN_SIZE];
        fill_weights(&mut weights, 13);
        (weights, (0..FEATURES as FeatureIndex).collect())
    }

    #[test]
    fn add_features_matches_scalar() {
        require_avx2!();
        let (weights, indices) = corpus();
        let mut want = seeded_out(3);
        let mut got = want.clone();
        scalar::add_features(&mut want, &weights, &indices);
        // SAFETY: guarded by `require_avx2!`.
        unsafe { add_features(&mut got, &weights, &indices) };
        assert_eq!(want, got);
    }

    #[test]
    fn sub_features_matches_scalar() {
        require_avx2!();
        let (weights, indices) = corpus();
        let mut want = seeded_out(5);
        let mut got = want.clone();
        scalar::sub_features(&mut want, &weights, &indices);
        // SAFETY: guarded by `require_avx2!`.
        unsafe { sub_features(&mut got, &weights, &indices) };
        assert_eq!(want, got);
    }

    #[test]
    fn add_sub_features_matches_scalar() {
        require_avx2!();
        let (weights, indices) = corpus();
        let (added, removed) = indices.split_at(2);
        let mut want = seeded_out(7);
        let mut got = want.clone();
        scalar::add_sub_features(&mut want, &weights, added, removed);
        // SAFETY: guarded by `require_avx2!`.
        unsafe { add_sub_features(&mut got, &weights, added, removed) };
        assert_eq!(want, got);
    }

    #[test]
    fn add_sub_sub_features_matches_scalar() {
        require_avx2!();
        let (weights, indices) = corpus();
        let (added, rest) = indices.split_at(2);
        let (removed_a, removed_b) = rest.split_at(2);
        let mut want = seeded_out(11);
        let mut got = want.clone();
        scalar::add_sub_sub_features(&mut want, &weights, added, removed_a, removed_b);
        // SAFETY: guarded by `require_avx2!`.
        unsafe { add_sub_sub_features(&mut got, &weights, added, removed_a, removed_b) };
        assert_eq!(want, got);
    }
    #[test]
    fn derive_features_matches_copy_then_apply_for_every_arity() {
        require_avx2!();
        let (weights, indices) = corpus();
        let src = seeded_out(19);
        // Every (added, removed) length up to 3 each, so the fixed-size
        // monomorphizations and the long-list fallback are all exercised.
        for na in 0..=3usize {
            for nr in 0..=3usize {
                let added = &indices[..na];
                let removed = &indices[3..3 + nr];
                let mut want = src.clone();
                scalar::sub_features(&mut want, &weights, removed);
                scalar::add_features(&mut want, &weights, added);
                let mut got = vec![0x5555i16; HIDDEN_SIZE];
                // SAFETY: guarded by `require_avx2!`.
                unsafe { derive_features(&mut got, &src, &weights, added, removed) };
                assert_eq!(want, got, "added {na} removed {nr}");
            }
        }
    }

    /// The same column on both sides must cancel exactly, wrapping included.
    #[test]
    fn derive_features_cancels_a_column_added_and_removed() {
        require_avx2!();
        let (weights, _) = corpus();
        let src = seeded_out(23);
        let mut got = vec![0i16; HIDDEN_SIZE];
        // SAFETY: guarded by `require_avx2!`.
        unsafe { derive_features(&mut got, &src, &weights, &[4], &[4]) };
        assert_eq!(got, src);
    }
}
