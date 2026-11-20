//! AVX-512 (F + BW) feature-transformer accumulate/update kernels, each
//! bit-identical to its [`crate::simd::scalar`] counterpart.
//!
//! Unlike the reference, which cfg-gates the whole module on the target
//! features, this one is *compiled* on every `x86_64` target so that the parity
//! tests below can hold it against the scalar baseline on any AVX-512-capable
//! host. The `#[target_feature]` attribute on each entry point imposes no
//! whole-binary requirement; it only makes the compiler emit AVX-512 code for
//! that one function, which the linker drops when nothing calls it.

use std::arch::x86_64::{
    __m512i, _mm512_add_epi16, _mm512_loadu_si512, _mm512_storeu_si512, _mm512_sub_epi16,
};

#[cfg(feature = "sfnn_threat")]
use std::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm512_cvtepi8_epi16, _mm512_setzero_si512};

use crate::features::FeatureIndex;
use crate::types::HIDDEN_SIZE;

const NUM_CHUNKS: usize = HIDDEN_SIZE / 32;
const LANES: usize = 32;

/// # Safety
/// The running CPU must support `avx512f` and `avx512bw`. `out` and every
/// referenced weight column must be `HIDDEN_SIZE` long.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn add_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
    debug_assert_eq!(out.len(), HIDDEN_SIZE);
    let out_ptr = out.as_mut_ptr();
    for &idx in indices {
        let base = idx as usize * HIDDEN_SIZE;
        let col = &weights[base..base + HIDDEN_SIZE];
        let col_ptr = col.as_ptr();
        for chunk in 0..NUM_CHUNKS {
            let offset = chunk * LANES;
            // SAFETY: `chunk*LANES+LANES <= HIDDEN_SIZE = out.len() = col.len()`,
            // and the unaligned 512-bit ops impose no alignment requirement.
            unsafe {
                let o = _mm512_loadu_si512(out_ptr.add(offset).cast::<__m512i>());
                let w = _mm512_loadu_si512(col_ptr.add(offset).cast::<__m512i>());
                _mm512_storeu_si512(
                    out_ptr.add(offset).cast::<__m512i>(),
                    _mm512_add_epi16(o, w),
                );
            }
        }
    }
}

/// # Safety
/// See [`add_features`].
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn sub_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
    debug_assert_eq!(out.len(), HIDDEN_SIZE);
    let out_ptr = out.as_mut_ptr();
    for &idx in indices {
        let base = idx as usize * HIDDEN_SIZE;
        let col = &weights[base..base + HIDDEN_SIZE];
        let col_ptr = col.as_ptr();
        for chunk in 0..NUM_CHUNKS {
            let offset = chunk * LANES;
            // SAFETY: see `add_features`.
            unsafe {
                let o = _mm512_loadu_si512(out_ptr.add(offset).cast::<__m512i>());
                let w = _mm512_loadu_si512(col_ptr.add(offset).cast::<__m512i>());
                _mm512_storeu_si512(
                    out_ptr.add(offset).cast::<__m512i>(),
                    _mm512_sub_epi16(o, w),
                );
            }
        }
    }
}

/// # Safety
/// See [`add_features`].
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn add_sub_features(
    out: &mut [i16],
    weights: &[i16],
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    debug_assert_eq!(out.len(), HIDDEN_SIZE);
    let out_ptr = out.as_mut_ptr();
    for chunk in 0..NUM_CHUNKS {
        let offset = chunk * LANES;
        // SAFETY: `chunk * LANES + LANES <= HIDDEN_SIZE = out.len()`.
        let mut acc = unsafe { _mm512_loadu_si512(out_ptr.add(offset).cast::<__m512i>()) };
        for &idx in added {
            let base = idx as usize * HIDDEN_SIZE;
            let col_ptr = weights[base..base + HIDDEN_SIZE].as_ptr();
            // SAFETY: `col` is `HIDDEN_SIZE`-long, same offset bound.
            let w = unsafe { _mm512_loadu_si512(col_ptr.add(offset).cast::<__m512i>()) };
            acc = _mm512_add_epi16(acc, w);
        }
        for &idx in removed {
            let base = idx as usize * HIDDEN_SIZE;
            let col_ptr = weights[base..base + HIDDEN_SIZE].as_ptr();
            // SAFETY: see above.
            let w = unsafe { _mm512_loadu_si512(col_ptr.add(offset).cast::<__m512i>()) };
            acc = _mm512_sub_epi16(acc, w);
        }
        // SAFETY: same offset bound on `out`.
        unsafe { _mm512_storeu_si512(out_ptr.add(offset).cast::<__m512i>(), acc) };
    }
}

/// # Safety
/// See [`add_features`].
#[target_feature(enable = "avx512f,avx512bw")]
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
        let mut acc = unsafe { _mm512_loadu_si512(out_ptr.add(offset).cast::<__m512i>()) };
        for &idx in added {
            let base = idx as usize * HIDDEN_SIZE;
            let col_ptr = weights[base..base + HIDDEN_SIZE].as_ptr();
            // SAFETY: `col` is `HIDDEN_SIZE`-long, same offset bound.
            let w = unsafe { _mm512_loadu_si512(col_ptr.add(offset).cast::<__m512i>()) };
            acc = _mm512_add_epi16(acc, w);
        }
        for &idx in removed_a {
            let base = idx as usize * HIDDEN_SIZE;
            let col_ptr = weights[base..base + HIDDEN_SIZE].as_ptr();
            // SAFETY: see above.
            let w = unsafe { _mm512_loadu_si512(col_ptr.add(offset).cast::<__m512i>()) };
            acc = _mm512_sub_epi16(acc, w);
        }
        for &idx in removed_b {
            let base = idx as usize * HIDDEN_SIZE;
            let col_ptr = weights[base..base + HIDDEN_SIZE].as_ptr();
            // SAFETY: see above.
            let w = unsafe { _mm512_loadu_si512(col_ptr.add(offset).cast::<__m512i>()) };
            acc = _mm512_sub_epi16(acc, w);
        }
        // SAFETY: same offset bound on `out`.
        unsafe { _mm512_storeu_si512(out_ptr.add(offset).cast::<__m512i>(), acc) };
    }
}

/// `dst = src + sum(added) - sum(removed)`, one pass over the accumulator
/// half: [`crate::simd::avx2::derive_features`] on 512-bit registers.
///
/// Each 64-byte chunk is loaded from `src` once, has every column applied in a
/// register and is stored to `dst` once, instead of copying the half and then
/// reading and writing it again for each list. The `i16` adds wrap and commute,
/// so the result is the copy-then-apply form's bit for bit. Up to two columns
/// a side get their own monomorphization; longer lists copy, then apply.
///
/// # Safety
/// The running CPU must support `avx512f` and `avx512bw`. `dst` and `src` must
/// both be `HIDDEN_SIZE` long (asserted).
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn derive_features(
    dst: &mut [i16],
    src: &[i16],
    weights: &[i16],
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    assert_eq!(dst.len(), HIDDEN_SIZE);
    assert_eq!(src.len(), HIDDEN_SIZE);
    // SAFETY: the callees need avx512f + avx512bw, which this function's
    // attribute grants.
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
#[target_feature(enable = "avx512f,avx512bw")]
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
            let mut acc = _mm512_loadu_si512(src_ptr.add(offset).cast::<__m512i>());
            for col in a_cols {
                acc = _mm512_add_epi16(acc, _mm512_loadu_si512(col.add(offset).cast::<__m512i>()));
            }
            for col in r_cols {
                acc = _mm512_sub_epi16(acc, _mm512_loadu_si512(col.add(offset).cast::<__m512i>()));
            }
            _mm512_storeu_si512(dst_ptr.add(offset).cast::<__m512i>(), acc);
        }
    }
}

/// Take the `removed` threat rows off `out` and put the `added` ones on, in place.
///
/// The AVX-512 form of [`crate::simd::avx2::apply_threat_rows`]: a row's signed bytes are
/// widened to `i16` (`vpmovsxbw`) and added unscaled, wrapping like the feature columns.
/// Eight registers (256 lanes) are loaded from `out`, every row of the change goes over
/// them while they stay in registers, and they are stored back.
///
/// # Safety
/// The running CPU must support `avx512f` and `avx512bw`. `out` must be `HIDDEN_SIZE`
/// long, and every row in `removed` and `added` must be below
/// `weights.len() / HIDDEN_SIZE` (checked).
#[cfg(feature = "sfnn_threat")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn apply_threat_rows(out: &mut [i16], weights: &[i8], removed: &[u32], added: &[u32]) {
    const TILE_REGS: usize = 8;
    const TILE: usize = TILE_REGS * LANES;
    const _: () = assert!(HIDDEN_SIZE.is_multiple_of(TILE));

    assert_eq!(out.len(), HIDDEN_SIZE);
    let rows = weights.len() / HIDDEN_SIZE;
    // Checked once here so that the loops below load with plain pointers.
    assert!(
        removed
            .iter()
            .chain(added)
            .all(|&row| (row as usize) < rows),
        "a threat row is out of range"
    );
    let out_ptr = out.as_mut_ptr();
    let w_ptr = weights.as_ptr();
    for tile in 0..HIDDEN_SIZE / TILE {
        let off = tile * TILE;
        // SAFETY: `off + TILE <= HIDDEN_SIZE = out.len()`, and each row slice starts at
        // `row * HIDDEN_SIZE + off` with `row < rows`, so its `TILE` bytes (32 per register)
        // lie inside `weights`.
        unsafe {
            let mut acc = [_mm512_setzero_si512(); TILE_REGS];
            for (k, slot) in acc.iter_mut().enumerate() {
                *slot = _mm512_loadu_si512(out_ptr.add(off + k * LANES).cast::<__m512i>());
            }
            for &row in removed {
                let base = w_ptr.add(row as usize * HIDDEN_SIZE + off);
                for (k, slot) in acc.iter_mut().enumerate() {
                    let w = _mm512_cvtepi8_epi16(_mm256_loadu_si256(
                        base.add(k * LANES).cast::<__m256i>(),
                    ));
                    *slot = _mm512_sub_epi16(*slot, w);
                }
            }
            for &row in added {
                let base = w_ptr.add(row as usize * HIDDEN_SIZE + off);
                for (k, slot) in acc.iter_mut().enumerate() {
                    let w = _mm512_cvtepi8_epi16(_mm256_loadu_si256(
                        base.add(k * LANES).cast::<__m256i>(),
                    ));
                    *slot = _mm512_add_epi16(*slot, w);
                }
            }
            for (k, slot) in acc.iter().enumerate() {
                _mm512_storeu_si512(out_ptr.add(off + k * LANES).cast::<__m512i>(), *slot);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::scalar;

    /// Skip the body unless the CPU has the features these kernels need, so
    /// that the parity checks are skipped rather than failed elsewhere.
    macro_rules! require_avx512bw {
        () => {
            if !(std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw"))
            {
                eprintln!("skipping AVX-512 parity test: avx512f/avx512bw unavailable");
                return;
            }
        };
    }

    fn fill_weights(weights: &mut [i16], seed: i16) {
        for (i, slot) in weights.iter_mut().enumerate() {
            *slot = ((i as i32 * 7 + seed as i32) % 29 - 14) as i16;
        }
    }

    fn seeded_initial(seed: i16) -> [i16; HIDDEN_SIZE] {
        let mut out = [0i16; HIDDEN_SIZE];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = ((i as i32 * 13 + seed as i32) % 41 - 20) as i16;
        }
        out
    }

    #[test]
    fn add_features_matches_scalar_on_random_corpus() {
        require_avx512bw!();
        let mut weights = vec![0i16; HIDDEN_SIZE * 8].into_boxed_slice();
        fill_weights(&mut weights, 17);
        let indices: [FeatureIndex; 4] = [0, 2, 5, 7];
        let initial = seeded_initial(3);

        let mut avx = initial;
        // SAFETY: guarded by `require_avx512bw!`.
        unsafe { add_features(&mut avx, &weights, &indices) };

        let mut sca = initial;
        scalar::add_features(&mut sca, &weights, &indices);

        assert_eq!(avx, sca);
    }

    #[test]
    fn sub_features_matches_scalar_on_random_corpus() {
        require_avx512bw!();
        let mut weights = vec![0i16; HIDDEN_SIZE * 8].into_boxed_slice();
        fill_weights(&mut weights, 29);
        let indices: [FeatureIndex; 3] = [1, 3, 6];
        let initial = seeded_initial(13);

        let mut avx = initial;
        // SAFETY: guarded by `require_avx512bw!`.
        unsafe { sub_features(&mut avx, &weights, &indices) };

        let mut sca = initial;
        scalar::sub_features(&mut sca, &weights, &indices);

        assert_eq!(avx, sca);
    }

    #[test]
    fn add_sub_features_matches_scalar_on_random_corpus() {
        require_avx512bw!();
        let mut weights = vec![0i16; HIDDEN_SIZE * 8].into_boxed_slice();
        fill_weights(&mut weights, 23);
        let added: [FeatureIndex; 1] = [3];
        let removed: [FeatureIndex; 1] = [6];
        let initial = seeded_initial(7);

        let mut avx = initial;
        // SAFETY: guarded by `require_avx512bw!`.
        unsafe { add_sub_features(&mut avx, &weights, &added, &removed) };

        let mut sca = initial;
        scalar::add_sub_features(&mut sca, &weights, &added, &removed);

        assert_eq!(avx, sca);
    }

    #[test]
    fn add_sub_sub_features_matches_scalar_on_random_corpus() {
        require_avx512bw!();
        let mut weights = vec![0i16; HIDDEN_SIZE * 8].into_boxed_slice();
        fill_weights(&mut weights, 31);
        let added: [FeatureIndex; 1] = [2];
        let removed_a: [FeatureIndex; 1] = [5];
        let removed_b: [FeatureIndex; 1] = [7];
        let initial = seeded_initial(19);

        let mut avx = initial;
        // SAFETY: guarded by `require_avx512bw!`.
        unsafe { add_sub_sub_features(&mut avx, &weights, &added, &removed_a, &removed_b) };

        let mut sca = initial;
        scalar::add_sub_sub_features(&mut sca, &weights, &added, &removed_a, &removed_b);

        assert_eq!(avx, sca);
    }

    #[test]
    fn derive_features_matches_copy_then_apply_for_every_arity() {
        require_avx512bw!();
        let mut weights = vec![0i16; HIDDEN_SIZE * 8].into_boxed_slice();
        fill_weights(&mut weights, 37);
        let indices: [FeatureIndex; 6] = [0, 2, 5, 1, 4, 7];
        let src = seeded_initial(29);
        // Every (added, removed) length up to 3 each, so the fixed-size
        // monomorphizations and the long-list fallback are all exercised.
        for na in 0..=3usize {
            for nr in 0..=3usize {
                let added = &indices[..na];
                let removed = &indices[3..3 + nr];
                let mut want = src;
                scalar::sub_features(&mut want, &weights, removed);
                scalar::add_features(&mut want, &weights, added);
                let mut got = [0x5555i16; HIDDEN_SIZE];
                // SAFETY: guarded by `require_avx512bw!`.
                unsafe { derive_features(&mut got, &src, &weights, added, removed) };
                assert_eq!(want, got, "added {na} removed {nr}");
            }
        }
    }

    /// The same column on both sides must cancel exactly, wrapping included.
    #[test]
    fn derive_features_cancels_a_column_added_and_removed() {
        require_avx512bw!();
        let mut weights = vec![0i16; HIDDEN_SIZE * 8].into_boxed_slice();
        fill_weights(&mut weights, 41);
        let src = seeded_initial(31);
        let mut got = [0i16; HIDDEN_SIZE];
        // SAFETY: guarded by `require_avx512bw!`.
        unsafe { derive_features(&mut got, &src, &weights, &[4], &[4]) };
        assert_eq!(got, src);
    }

    /// The AVX-512 threat-row kernel against the scalar one, over rows with every byte
    /// value and an accumulator near the `i16` limits, so that the widening and the wrap
    /// both matter.
    #[cfg(feature = "sfnn_threat")]
    #[test]
    fn apply_threat_rows_matches_scalar() {
        require_avx512bw!();
        const ROWS: usize = 12;
        let weights: Vec<i8> = (0..ROWS * HIDDEN_SIZE)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 7) as i8)
            .collect();
        assert!(weights.contains(&i8::MIN) && weights.contains(&i8::MAX));
        let init: Vec<i16> = (0..HIDDEN_SIZE)
            .map(|i| match i % 4 {
                0 => i16::MAX - (i % 7) as i16,
                1 => i16::MIN + (i % 5) as i16,
                _ => (i as i16).wrapping_mul(97),
            })
            .collect();
        let shapes: [(&[u32], &[u32]); 5] = [
            (&[], &[]),
            (&[], &[0, 3, 11]),
            (&[2, 5], &[]),
            (&[1, 4, 9], &[0, 4, 10, 11]),
            (&[7, 7], &[7, 7, 7]),
        ];
        for (removed, added) in shapes {
            let mut want = init.clone();
            scalar::apply_threat_rows(&mut want, &weights, removed, added);
            let mut got = init.clone();
            // SAFETY: guarded by `require_avx512bw!`.
            unsafe { apply_threat_rows(&mut got, &weights, removed, added) };
            assert_eq!(want, got, "removed {removed:?} added {added:?}");
        }
    }

    /// A row put on and taken off again leaves the accumulator as it was.
    #[cfg(feature = "sfnn_threat")]
    #[test]
    fn apply_threat_rows_undoes_itself() {
        require_avx512bw!();
        let weights: Vec<i8> = (0..3 * HIDDEN_SIZE).map(|i| (i * 31) as i8).collect();
        let init = seeded_initial(29);
        let mut got = init;
        // SAFETY: guarded by `require_avx512bw!`.
        unsafe {
            apply_threat_rows(&mut got, &weights, &[], &[0, 1, 2]);
            apply_threat_rows(&mut got, &weights, &[2, 0, 1], &[]);
        }
        assert_eq!(got, init);
    }

    #[cfg(feature = "sfnn_threat")]
    #[test]
    #[should_panic(expected = "out of range")]
    fn apply_threat_rows_refuses_a_row_past_the_block() {
        require_avx512bw!();
        let weights = vec![0i8; 2 * HIDDEN_SIZE];
        let mut out = seeded_initial(1);
        // SAFETY: guarded by `require_avx512bw!`; the call panics before touching memory.
        unsafe { apply_threat_rows(&mut out, &weights, &[], &[2]) };
    }
}
