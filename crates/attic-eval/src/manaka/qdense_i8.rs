//! The first dense layer with `i8` weights against a `u8` input: NNUE's types,
//! run on `vpmaddubsw` or `vpdpbusd`.
//!
//! The layer is `HIDDEN × FEATURE_LEN` multiply-adds against an input that is
//! already in `[0, 1]`, as [`QuantW1`](super::QuantW1)'s is. The input is held as
//! `round(x · 127)`, a `u8` between 0 and 127, and each row of the weights as
//! `round(w / max · 127)`, an `i8` between −127 and 127, where `max` is the row's
//! largest magnitude. One instruction takes thirty-two products where the `i16`
//! layer's takes sixteen, and the weights are half the bytes: at width 128 the
//! whole matrix is 24.6 KB, which sits in a core's L1 where the `i16` one does
//! not.
//!
//! The answer carries the same two roundings as the `i16` layer's, and they are
//! far coarser: the input to 1/127 and the weights to 127 levels a row. On trained
//! SYNN nets (KP and token at width 128, KP at 256, a thousand positions, 290 cp
//! to the unit) that moves the value by 10 to 18 cp on average, which is 0.4 to
//! 0.7 points of win rate (`σ(cp / 290)`, the trainer's), and 2 to 3.5 points at
//! the 99th percentile. It costs the validation loss 0.00006 to 0.00014. Nothing
//! here is trained for it; the layer is a switch, `ATTIC_SYNN_W1=i8`.
//!
//! ## Why nothing saturates or overflows
//!
//! `vpmaddubsw` multiplies a `u8` by an `i8` and adds neighbouring pairs into an
//! `i16` with saturation. With the input at most 127, a pair is at most
//! `2 · 127 · 127 = 32,258`, inside an `i16`, so it never saturates and the sum is
//! the one `vpdpbusd` and the scalar path compute. That is why the input stops at
//! 127 and not the 255 a `u8` could hold. A row's total is at most
//! `cols · 127 · 127`, about 25 million at the widest shape here, far inside an
//! `i32`.

/// What 1.0 is in the quantised input, and the most levels a weight has either
/// side of zero: `vpmaddubsw` cannot saturate on a pair below it (see above).
pub(super) const LEVELS: f32 = 127.0;

/// The bytes of an AVX2 register, which is how far a row is padded.
const LANES: usize = 32;

/// How many rows one pass handles, as in the `i16` layer: eight accumulators and
/// the input register fit in the sixteen vector registers.
const ROWS: usize = 8;

/// The first layer's input width, padded to whole vectors: what the quantised
/// input takes and what a row of [`QuantW1I8`] is stored as.
const INPUT_STRIDE: usize = super::FEATURE_LEN.next_multiple_of(LANES);

/// The first layer's weights as `i8`, one scale per row.
#[derive(Clone, Debug)]
pub struct QuantW1I8 {
    /// `[rows][stride]`, each row padded with zeros to a whole number of vectors so
    /// that no kernel needs a tail.
    w: Vec<i8>,
    /// What an `i32` row total is multiplied by to be the `f32` layer's sum:
    /// `max / 127 / 127`, the weight's step times the input's.
    scale: Vec<f32>,
    rows: usize,
    cols: usize,
    stride: usize,
}

impl QuantW1I8 {
    /// Quantise `w`, `rows` by `cols`, row-major.
    pub fn new(w: &[f32], rows: usize, cols: usize) -> QuantW1I8 {
        assert_eq!(w.len(), rows * cols);
        let stride = cols.next_multiple_of(LANES);
        let mut out = vec![0i8; rows * stride];
        let mut scale = vec![0.0f32; rows];
        for (row, src) in w.chunks_exact(cols).enumerate() {
            let max = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            if max == 0.0 || !max.is_finite() {
                continue;
            }
            for (dst, &v) in out[row * stride..].iter_mut().zip(src) {
                *dst = (v / max * LEVELS).round_ties_even() as i8;
            }
            scale[row] = max / LEVELS / LEVELS;
        }
        QuantW1I8 {
            w: out,
            scale,
            rows,
            cols,
            stride,
        }
    }

    /// `out[row] = clamp(bias[row] + weights[row] · input, 0, 1)`, with the product
    /// taken in integers.
    ///
    /// `input` has to be in `[0, 1]`, which every block the head reads is after its
    /// clip. This is the head's first layer, `HIDDEN` rows over `FEATURE_LEN`
    /// columns.
    pub fn layer_clip(&self, input: &[f32], bias: &[f32], out: &mut [f32]) {
        // On the stack and exactly as wide as the layer, as the `i16` layer's is.
        let mut x = [0u8; INPUT_STRIDE];
        self.layer_clip_with(input, bias, out, &mut x);
    }

    /// [`QuantW1I8::layer_clip`] with the buffer for the quantised input supplied,
    /// which has to be one row's padded width (`stride`). Split out so that the
    /// tests can run a layer of any width.
    fn layer_clip_with(&self, input: &[f32], bias: &[f32], out: &mut [f32], x: &mut [u8]) {
        debug_assert_eq!(input.len(), self.cols);
        debug_assert_eq!(out.len(), self.rows);
        debug_assert_eq!(bias.len(), self.rows);
        assert_eq!(x.len(), self.stride, "the buffer is one padded row wide");
        quantize_input(input, x);
        let x = &*x;
        let row = |at: usize| &self.w[at * self.stride..(at + 1) * self.stride];
        let mut at = 0;
        while at + ROWS <= self.rows {
            let sums = dots::<ROWS>(std::array::from_fn(|r| row(at + r)), x);
            for (r, sum) in sums.into_iter().enumerate() {
                out[at + r] = (bias[at + r] + sum as f32 * self.scale[at + r]).clamp(0.0, 1.0);
            }
            at += ROWS;
        }
        while at < self.rows {
            let sum = dots::<1>([row(at)], x)[0];
            out[at] = (bias[at] + sum as f32 * self.scale[at]).clamp(0.0, 1.0);
            at += 1;
        }
    }

    /// [`QuantW1I8::layer_clip`] for a layer of any width, for the tests.
    #[cfg(test)]
    pub(super) fn layer_clip_any_width(&self, input: &[f32], bias: &[f32], out: &mut [f32]) {
        let mut x = vec![0u8; self.stride];
        self.layer_clip_with(input, bias, out, &mut x);
    }

    #[cfg(test)]
    pub(super) fn weights(&self, row: usize) -> &[i8] {
        &self.w[row * self.stride..(row + 1) * self.stride]
    }
}

/// The input as `round(x · 127)`, with the padding zeroed.
///
/// The conversion rounds to nearest with ties to even, as `round_ties_even` does,
/// so the vector and scalar paths agree.
fn quantize_input(x: &[f32], out: &mut [u8]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: the `cfg` above is the build enabling AVX2 for the CPU it runs on.
        unsafe { quantize_input_avx2(x, out) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        quantize_input_scalar(x, out);
    }
}

/// [`quantize_input`] one number at a time.
#[allow(dead_code)]
fn quantize_input_scalar(x: &[f32], out: &mut [u8]) {
    debug_assert!(out.len() >= x.len());
    for (slot, &v) in out.iter_mut().zip(x) {
        *slot = (v * LEVELS).round_ties_even() as u8;
    }
    for slot in &mut out[x.len()..] {
        *slot = 0;
    }
}

/// [`quantize_input_scalar`] thirty-two at a time.
///
/// # Safety
/// The running CPU must support AVX2. The `cfg` that compiles this in names it.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
unsafe fn quantize_input_avx2(x: &[f32], out: &mut [u8]) {
    use std::arch::x86_64::*;

    assert!(out.len() >= x.len());
    let blocks = x.len() / LANES;
    let scale = _mm256_set1_ps(LEVELS);
    // `packs` and `packus` work inside each 128-bit half, so the four groups of
    // eight come out as their low and high quarters interleaved: dword `i` of the
    // result holds group `i % 4`'s quarter `i / 4`. The permute puts them back.
    let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for block in 0..blocks {
        let at = block * LANES;
        // SAFETY: `at + 32 <= blocks · 32 <= x.len() <= out.len()`; the accesses
        // are unaligned.
        unsafe {
            let load = |offset: usize| {
                _mm256_cvtps_epi32(_mm256_mul_ps(
                    _mm256_loadu_ps(x.as_ptr().add(at + offset)),
                    scale,
                ))
            };
            // Neither pack saturates: the input is in [0, 1], so every value is
            // between 0 and 127.
            let low = _mm256_packs_epi32(load(0), load(8));
            let high = _mm256_packs_epi32(load(16), load(24));
            let packed = _mm256_packus_epi16(low, high);
            let ordered = _mm256_permutevar8x32_epi32(packed, order);
            _mm256_storeu_si256(out.as_mut_ptr().add(at).cast::<__m256i>(), ordered);
        }
    }
    for at in blocks * LANES..x.len() {
        out[at] = (x[at] * LEVELS).round_ties_even() as u8;
    }
    for slot in &mut out[x.len()..] {
        *slot = 0;
    }
}

/// `Σ w[i] · x[i]` for `R` rows against one input, in `i32`.
#[inline]
fn dots<const R: usize>(rows: [&[i8]; R], x: &[u8]) -> [i32; R] {
    #[cfg(all(target_arch = "x86_64", target_feature = "avxvnni"))]
    {
        // SAFETY: the `cfg` above is the build enabling AVX-VNNI for the CPU it runs on.
        unsafe { dots_vnni(rows, x) }
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avxvnni")
    ))]
    {
        // SAFETY: the `cfg` above is the build enabling AVX2 for the CPU it runs on.
        unsafe { dots_maddubs(rows, x) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        dots_scalar(rows, x)
    }
}

/// [`dots`] one product at a time: the reference the vector paths are held to.
#[allow(dead_code)]
fn dots_scalar<const R: usize>(rows: [&[i8]; R], x: &[u8]) -> [i32; R] {
    std::array::from_fn(|r| {
        debug_assert_eq!(rows[r].len(), x.len());
        rows[r]
            .iter()
            .zip(x)
            .map(|(&w, &x)| i32::from(w) * i32::from(x))
            .sum()
    })
}

/// The total of an `i32` vector's eight lanes.
///
/// # Safety
/// AVX2 must be available.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
unsafe fn total(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let sum = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let sum = _mm_add_epi32(sum, _mm_shuffle_epi32::<0b01_00_11_10>(sum));
    let sum = _mm_add_epi32(sum, _mm_shuffle_epi32::<0b10_11_00_01>(sum));
    _mm_cvtsi128_si32(sum)
}

/// [`dots_scalar`] with `vpdpbusd`: thirty-two products added into eight `i32`
/// lanes by one instruction.
///
/// # Safety
/// The running CPU must support AVX2 and AVX-VNNI.
#[cfg(all(target_arch = "x86_64", target_feature = "avxvnni"))]
#[allow(dead_code)]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn dots_vnni<const R: usize>(rows: [&[i8]; R], x: &[u8]) -> [i32; R] {
    use std::arch::x86_64::*;

    let n = x.len();
    assert_eq!(n % LANES, 0, "a padded row is whole vectors");
    for row in rows {
        assert_eq!(row.len(), n, "a row is as long as the input");
    }
    let mut acc = [_mm256_setzero_si256(); R];
    for at in (0..n).step_by(LANES) {
        // SAFETY: `at + 32 <= n`, and the input and every row are `n` long
        // (asserted above); the loads are unaligned.
        unsafe {
            let input = _mm256_loadu_si256(x.as_ptr().add(at).cast::<__m256i>());
            for r in 0..R {
                let weights = _mm256_loadu_si256(rows[r].as_ptr().add(at).cast::<__m256i>());
                acc[r] = _mm256_dpbusd_avx_epi32(acc[r], input, weights);
            }
        }
    }
    // SAFETY: AVX2 is enabled on this function.
    std::array::from_fn(|r| unsafe { total(acc[r]) })
}

/// [`dots_vnni`] for a CPU with AVX2 and no AVX-VNNI: `vpmaddubsw` into `i16`
/// pairs, `vpmaddwd` against ones into `i32`, then `vpaddd`. The pairs cannot
/// saturate (see the module's note), so the total is the scalar one.
///
/// # Safety
/// The running CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
unsafe fn dots_maddubs<const R: usize>(rows: [&[i8]; R], x: &[u8]) -> [i32; R] {
    use std::arch::x86_64::*;

    let n = x.len();
    assert_eq!(n % LANES, 0, "a padded row is whole vectors");
    for row in rows {
        assert_eq!(row.len(), n, "a row is as long as the input");
    }
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_si256(); R];
    for at in (0..n).step_by(LANES) {
        // SAFETY: as in `dots_vnni`.
        unsafe {
            let input = _mm256_loadu_si256(x.as_ptr().add(at).cast::<__m256i>());
            for r in 0..R {
                let weights = _mm256_loadu_si256(rows[r].as_ptr().add(at).cast::<__m256i>());
                let pairs = _mm256_maddubs_epi16(input, weights);
                acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(pairs, ones));
            }
        }
    }
    // SAFETY: AVX2 is enabled on this function.
    std::array::from_fn(|r| unsafe { total(acc[r]) })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed stream of numbers in `[-1, 1)`, with a few exact zeros.
    fn stream(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                match (state >> 33) % 11 {
                    0 => 0.0,
                    _ => ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0,
                }
            })
            .collect()
    }

    /// An input as the head sees it: every entry in `[0, 1]`, a good share of them
    /// on the rails.
    fn clipped(count: usize, seed: u64) -> Vec<f32> {
        stream(count, seed)
            .into_iter()
            .map(|v| (v * 1.6 + 0.5).clamp(0.0, 1.0))
            .collect()
    }

    const WIDTHS: [usize; 9] = [32, 33, 64, 768, 800, 1196, 1200, 1536, 1];

    #[test]
    fn the_input_is_quantised_the_same_way_by_both_paths() {
        for n in WIDTHS {
            let x = clipped(n, 1);
            let padded = n.next_multiple_of(LANES);
            let mut want = vec![7u8; padded];
            quantize_input_scalar(&x, &mut want);
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            {
                let mut got = vec![7u8; padded];
                // SAFETY: the `cfg` is the build enabling AVX2.
                unsafe { quantize_input_avx2(&x, &mut got) };
                assert_eq!(got, want, "n={n}");
            }
            assert!(
                want[n..].iter().all(|&v| v == 0),
                "the padding is zero, n={n}"
            );
        }
        // The rails land exactly on 0 and 127, and every value keeps its place.
        let x: Vec<f32> = (0..64).map(|i| i as f32 / 127.0).collect();
        let mut out = vec![0u8; 64];
        quantize_input(&x, &mut out);
        assert_eq!(out, (0..64).collect::<Vec<u8>>());
        let mut rails = vec![9u8; 32];
        let mut x = vec![0.0f32; 32];
        x[1] = 1.0;
        quantize_input(&x, &mut rails);
        assert_eq!((rails[0], rails[1]), (0, 127));
    }

    #[test]
    fn the_vector_dot_products_are_the_scalar_one_exactly() {
        for n in [32usize, 64, 768, 800, 1216, 1536] {
            let x: Vec<u8> = stream(n, 2)
                .iter()
                .map(|v| (v.abs() * LEVELS) as u8)
                .collect();
            let data: Vec<i8> = stream(8 * n, 3)
                .iter()
                .map(|v| (v * LEVELS) as i8)
                .collect();
            let rows: [&[i8]; 8] = std::array::from_fn(|r| &data[r * n..(r + 1) * n]);
            let want = dots_scalar(rows, &x);
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: the build is for the host CPU, which has AVX2.
                let maddubs = unsafe { dots_maddubs(rows, &x) };
                assert_eq!(maddubs, want, "vpmaddubsw, n={n}");
            }
            #[cfg(all(target_arch = "x86_64", target_feature = "avxvnni"))]
            {
                // SAFETY: the `cfg` is the build enabling AVX-VNNI.
                let vnni = unsafe { dots_vnni(rows, &x) };
                assert_eq!(vnni, want, "vpdpbusd, n={n}");
            }
            assert_eq!(dots(rows, &x), want, "dispatched, n={n}");
        }
    }

    /// The worst a pair can be for `vpmaddubsw` — the input at its top and both
    /// weights at the same extreme — is still the scalar sum: nothing saturates.
    #[test]
    fn the_extremes_do_not_saturate() {
        let n = 1536;
        let x = vec![LEVELS as u8; n];
        for w in [127i8, -127] {
            let row = vec![w; n];
            let want = i32::from(w) * 127 * n as i32;
            assert_eq!(dots_scalar([row.as_slice()], &x)[0], want);
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: the build is for the host CPU, which has AVX2.
                assert_eq!(unsafe { dots_maddubs([row.as_slice()], &x) }[0], want);
            }
            assert_eq!(dots([row.as_slice()], &x)[0], want, "dispatched, {w}");
        }
    }

    /// Each row's largest magnitude lands on ±127 and the rest in proportion.
    #[test]
    fn a_row_is_127_levels_of_its_own_largest_magnitude() {
        let w = vec![0.5f32, -2.0, 1.0, 0.004, 0.0, 0.0, 0.0, 0.0];
        let q = QuantW1I8::new(&w, 2, 4);
        assert_eq!(&q.weights(0)[..4], &[32, -127, 64, 0]);
        assert!(
            q.weights(1).iter().all(|&v| v == 0),
            "a zero row stays zero"
        );
    }

    #[test]
    fn the_layer_is_the_f32_layer_to_within_the_two_roundings() {
        for (rows, cols) in [
            (32usize, 768usize),
            (32, 800),
            (32, 1196),
            (32, 1536),
            (5, 33),
        ] {
            let w: Vec<f32> = stream(rows * cols, 5).iter().map(|v| v * 0.08).collect();
            let bias: Vec<f32> = stream(rows, 6).iter().map(|v| v * 0.2).collect();
            let x = clipped(cols, 7);
            let q = QuantW1I8::new(&w, rows, cols);
            let mut got = vec![0.0f32; rows];
            q.layer_clip_any_width(&x, &bias, &mut got);
            for row in 0..rows {
                let src = &w[row * cols..(row + 1) * cols];
                let max = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let sum: f64 = (0..cols).map(|i| f64::from(src[i]) * f64::from(x[i])).sum();
                let want = (f64::from(bias[row]) + sum).clamp(0.0, 1.0);
                // Each product is off by at most half a weight step times one plus
                // half an input step times the weight: a bound on the sum of all of
                // them, which the real error sits well inside.
                let step = f64::from(max) / 127.0;
                let bound = cols as f64 * (step / 2.0 + f64::from(max) / 254.0);
                let error = (f64::from(got[row]) - want).abs();
                assert!(error <= bound, "{rows}x{cols} row {row}: {error} > {bound}");
                // And it is a rounding, not a different layer: a few hundredths at
                // most for weights of a few hundredths.
                assert!(error < 0.05, "{rows}x{cols} row {row}: error {error}");
            }
        }
    }

    #[test]
    fn a_zero_row_stays_zero() {
        let w = vec![0.0f32; 2 * 32];
        let q = QuantW1I8::new(&w, 2, 32);
        let mut out = [9.0f32; 2];
        q.layer_clip_any_width(&clipped(32, 1), &[0.25, -0.25], &mut out);
        assert_eq!(out, [0.25, 0.0]);
    }
}
