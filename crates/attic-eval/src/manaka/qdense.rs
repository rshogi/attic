//! The first dense layer with `i16` weights, run on an integer dot product.
//!
//! The layer is `HIDDEN × FEATURE_LEN` multiply-adds against an input that is
//! already in `[0, 1]`: every block is clipped before it gets here. That is what
//! makes a single `i16` plane enough. The input is held as `round(x · 4096)`, so
//! it is between 0 and 4096, and an `i16` weight times it, summed over the row, has
//! a bound that can be worked out from the row alone — see [`row_levels`].
//!
//! What a row costs is one `vpdpwssd` per sixteen products where the `f32` layer
//! spends one multiply and one add per eight. The answer is not the `f32` layer's:
//! it carries the two roundings (the weights to a row's levels, the input to
//! 1/4096) and nothing else, because the integer sum itself is exact. Measured on
//! trained NNUE weights, the cost is about 0.3 cp on average against 508 cp to the
//! unit. On trained SYNN nets (KP and token at width 128, KP at 256, a thousand
//! positions) it is 0.24 to 0.28 cp on average and about 1 cp at the 99th
//! percentile, against 290 cp to the unit. It is the layer a loaded network runs;
//! the `f32` layer stays the definition, and `ATTIC_SYNN_W1=f32` selects it.
//! `ATTIC_SYNN_W1=i8` selects the coarser and faster layer of `qdense_i8.rs`.
//!
//! ## Why the sum cannot overflow
//!
//! `vpdpwssd` and `vpaddd` add modulo 2³², so what has to fit is the *final* sum of
//! a row and not the partial ones — a total that is inside `i32` comes out right
//! however the partial sums wrapped on the way. With the input non-negative, the
//! total is at most `4096 · Σ w⁺` and at least `−4096 · Σ w⁻`, so a row is
//! quantised at the most levels for which `max(Σ w⁺, Σ w⁻) · 4096` fits. That is a
//! tighter bound than `Σ|w|`.
//!
//! It does cost levels. On the trained NNUE nets (512 columns, 32 rows) a row gets
//! 5,100 to 17,000 levels, twelve to fourteen bits, not the 32,767 an `i16` could
//! hold; that is the quantisation whose error is quoted above, so the number
//! already includes it. A row of 800 ordinary weights gets fewer still.

/// What 1.0 is in the quantised input. Chosen on trained NNUE weights: 4096 had the
/// smallest error. Below it the input rounding dominates (1024 was four times
/// worse); above it the weights have to give up levels so that the row still fits
/// in an `i32` (8192 was worse on HalfKP).
pub(super) const INPUT_LEVELS: f32 = 4096.0;

/// The most levels an `i16` weight may have: `-32768` is left out so that the range
/// is symmetric. A row rarely gets this many: see [`row_levels`].
const WEIGHT_LEVELS: f32 = 32_767.0;

/// The lanes of an AVX2 register of `i16`, which is how far a row is padded.
const LANES: usize = 16;

/// How many rows one pass handles. Eight accumulators and the input register fit in
/// the sixteen vector registers with room left over; sixteen of them do not.
const ROWS: usize = 8;

/// The levels a row is quantised at: as many as `i16` has, unless that would let
/// the row's total overflow an `i32` for some input in `[0, INPUT_LEVELS]`.
///
/// `positive` and `negative` are the sums of the row's positive weights and of the
/// magnitudes of its negative ones, in units of the row's largest magnitude `max`.
/// Rounding each weight to the nearest level adds at most half a level to each, so
/// `cols / 2` of headroom is taken off before the division.
fn row_levels(positive: f64, negative: f64, cols: usize) -> f32 {
    let room = f64::from(i32::MAX) / f64::from(INPUT_LEVELS) - cols as f64 / 2.0;
    let worst = positive.max(negative).max(1.0);
    (room / worst).floor().clamp(1.0, f64::from(WEIGHT_LEVELS)) as f32
}

/// The first layer's weights as `i16`, one scale per row.
#[derive(Clone, Debug)]
pub struct QuantW1 {
    /// `[rows][stride]`, each row padded with zeros to a whole number of vectors so
    /// that no kernel needs a tail.
    w: Vec<i16>,
    /// What an `i32` row total is multiplied by to be the `f32` layer's sum:
    /// `row scale / INPUT_LEVELS`.
    scale: Vec<f32>,
    rows: usize,
    cols: usize,
    stride: usize,
}

impl QuantW1 {
    /// Quantise `w`, `rows` by `cols`, row-major.
    pub fn new(w: &[f32], rows: usize, cols: usize) -> QuantW1 {
        assert_eq!(w.len(), rows * cols);
        let stride = cols.next_multiple_of(LANES);
        let mut out = vec![0i16; rows * stride];
        let mut scale = vec![0.0f32; rows];
        for (row, src) in w.chunks_exact(cols).enumerate() {
            let max = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            if max == 0.0 || !max.is_finite() {
                continue;
            }
            let (mut positive, mut negative) = (0.0f64, 0.0f64);
            for &v in src {
                let unit = f64::from(v / max);
                positive += unit.max(0.0);
                negative += (-unit).max(0.0);
            }
            let levels = row_levels(positive, negative, cols);
            for (dst, &v) in out[row * stride..].iter_mut().zip(src) {
                *dst = (v / max * levels).round_ties_even() as i16;
            }
            scale[row] = max / levels / INPUT_LEVELS;
        }
        QuantW1 {
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
    /// clip: the bound in [`row_levels`] is for that range. This is the head's first
    /// layer, `HIDDEN` rows over `FEATURE_LEN` columns.
    pub fn layer_clip(&self, input: &[f32], bias: &[f32], out: &mut [f32]) {
        // The one buffer a layer needs, on the stack and exactly as wide as the
        // layer: a `Vec` would be an allocation per node, and a buffer sized for the
        // widest shape would be cleared in full on every call.
        let mut x = [0i16; INPUT_STRIDE];
        self.layer_clip_with(input, bias, out, &mut x);
    }

    /// [`QuantW1::layer_clip`] with the buffer for the quantised input supplied, which
    /// has to be one row's padded width (`stride`). Split out so that the tests can
    /// run a layer of any width.
    fn layer_clip_with(&self, input: &[f32], bias: &[f32], out: &mut [f32], x: &mut [i16]) {
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

    /// [`QuantW1::layer_clip`] for a layer of any width, for the tests.
    #[cfg(test)]
    pub(super) fn layer_clip_any_width(&self, input: &[f32], bias: &[f32], out: &mut [f32]) {
        let mut x = vec![0i16; self.stride];
        self.layer_clip_with(input, bias, out, &mut x);
    }

    /// The `i32` total of one row against an already-quantised input, for the tests.
    #[cfg(test)]
    pub(super) fn row_total(&self, row: usize, x: &[i16]) -> i32 {
        dots::<1>([&self.w[row * self.stride..(row + 1) * self.stride]], x)[0]
    }

    #[cfg(test)]
    pub(super) fn stride(&self) -> usize {
        self.stride
    }

    #[cfg(test)]
    pub(super) fn weights(&self, row: usize) -> &[i16] {
        &self.w[row * self.stride..(row + 1) * self.stride]
    }

    #[cfg(test)]
    pub(super) fn scale(&self, row: usize) -> f32 {
        self.scale[row]
    }
}

/// The first layer's input width, padded to whole vectors: what the quantised input
/// takes and what a row of [`QuantW1`] is stored as. The layer is always the head's
/// `HIDDEN` rows over `FEATURE_LEN` columns.
const INPUT_STRIDE: usize = super::FEATURE_LEN.next_multiple_of(LANES);

/// The input as `round(x · INPUT_LEVELS)`, with the padding zeroed.
///
/// `x · 4096` is exact in `f32` (a power of two), so the rounding is the one the
/// conversion does, to nearest with ties to even, and the vector and scalar paths
/// agree.
fn quantize_input(x: &[f32], out: &mut [i16]) {
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
fn quantize_input_scalar(x: &[f32], out: &mut [i16]) {
    debug_assert!(out.len() >= x.len());
    for (slot, &v) in out.iter_mut().zip(x) {
        *slot = (v * INPUT_LEVELS).round_ties_even() as i16;
    }
    for slot in &mut out[x.len()..] {
        *slot = 0;
    }
}

/// [`quantize_input_scalar`] sixteen at a time.
///
/// # Safety
/// The running CPU must support AVX2. The `cfg` that compiles this in names it.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
unsafe fn quantize_input_avx2(x: &[f32], out: &mut [i16]) {
    use std::arch::x86_64::*;

    assert!(out.len() >= x.len());
    let blocks = x.len() / LANES;
    let scale = _mm256_set1_ps(INPUT_LEVELS);
    for block in 0..blocks {
        let at = block * LANES;
        // SAFETY: `at + 16 <= blocks · 16 <= x.len() <= out.len()`; the accesses are
        // unaligned.
        unsafe {
            let low = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(at)), scale));
            let high = _mm256_cvtps_epi32(_mm256_mul_ps(
                _mm256_loadu_ps(x.as_ptr().add(at + 8)),
                scale,
            ));
            // `packs` works inside each 128-bit half, so the halves come out
            // interleaved; the permute puts the sixteen back in order. Saturation
            // cannot happen here: the input is in [0, 1].
            let packed = _mm256_packs_epi32(low, high);
            let ordered = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);
            _mm256_storeu_si256(out.as_mut_ptr().add(at).cast::<__m256i>(), ordered);
        }
    }
    for at in blocks * LANES..x.len() {
        out[at] = (x[at] * INPUT_LEVELS).round_ties_even() as i16;
    }
    for slot in &mut out[x.len()..] {
        *slot = 0;
    }
}

/// `Σ w[i] · x[i]` for `R` rows against one input, in `i32` arithmetic that wraps.
#[inline]
fn dots<const R: usize>(rows: [&[i16]; R], x: &[i16]) -> [i32; R] {
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
        unsafe { dots_madd(rows, x) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        dots_scalar(rows, x)
    }
}

/// [`dots`] one product at a time: the reference the vector paths are held to.
#[allow(dead_code)]
fn dots_scalar<const R: usize>(rows: [&[i16]; R], x: &[i16]) -> [i32; R] {
    std::array::from_fn(|r| {
        debug_assert_eq!(rows[r].len(), x.len());
        rows[r].iter().zip(x).fold(0i32, |a, (&w, &x)| {
            a.wrapping_add(i32::from(w) * i32::from(x))
        })
    })
}

/// The total of an `i32` vector's eight lanes, wrapping.
///
/// # Safety
/// AVX2 must be available.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
unsafe fn total(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let mut lanes = [0i32; 8];
    // SAFETY: `lanes` is eight `i32`, which is what one register stores.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), v) };
    lanes.iter().fold(0i32, |a, &b| a.wrapping_add(b))
}

/// [`dots_scalar`] with `vpdpwssd`: sixteen products added into eight `i32` lanes
/// by one instruction.
///
/// # Safety
/// The running CPU must support AVX2 and AVX-VNNI.
#[cfg(all(target_arch = "x86_64", target_feature = "avxvnni"))]
#[allow(dead_code)]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn dots_vnni<const R: usize>(rows: [&[i16]; R], x: &[i16]) -> [i32; R] {
    use std::arch::x86_64::*;

    let n = x.len();
    assert_eq!(n % LANES, 0, "a padded row is whole vectors");
    for row in rows {
        assert_eq!(row.len(), n, "a row is as long as the input");
    }
    let mut acc = [_mm256_setzero_si256(); R];
    for at in (0..n).step_by(LANES) {
        // SAFETY: `at + 16 <= n`, and the input and every row are `n` long
        // (asserted above); the loads are unaligned.
        unsafe {
            let input = _mm256_loadu_si256(x.as_ptr().add(at).cast::<__m256i>());
            for r in 0..R {
                let weights = _mm256_loadu_si256(rows[r].as_ptr().add(at).cast::<__m256i>());
                acc[r] = _mm256_dpwssd_avx_epi32(acc[r], weights, input);
            }
        }
    }
    // SAFETY: AVX2 is enabled on this function.
    std::array::from_fn(|r| unsafe { total(acc[r]) })
}

/// [`dots_vnni`] for a CPU with AVX2 and no AVX-VNNI: `vpmaddwd` and then `vpaddd`.
///
/// # Safety
/// The running CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
unsafe fn dots_madd<const R: usize>(rows: [&[i16]; R], x: &[i16]) -> [i32; R] {
    use std::arch::x86_64::*;

    let n = x.len();
    assert_eq!(n % LANES, 0, "a padded row is whole vectors");
    for row in rows {
        assert_eq!(row.len(), n, "a row is as long as the input");
    }
    let mut acc = [_mm256_setzero_si256(); R];
    for at in (0..n).step_by(LANES) {
        // SAFETY: as in `dots_vnni`.
        unsafe {
            let input = _mm256_loadu_si256(x.as_ptr().add(at).cast::<__m256i>());
            for r in 0..R {
                let weights = _mm256_loadu_si256(rows[r].as_ptr().add(at).cast::<__m256i>());
                acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(weights, input));
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

    const WIDTHS: [usize; 8] = [16, 17, 33, 768, 800, 1196, 1200, 1];

    #[test]
    fn the_input_is_quantised_the_same_way_by_both_paths() {
        for n in WIDTHS {
            let x = clipped(n, 1);
            let padded = n.next_multiple_of(LANES);
            let mut want = vec![7i16; padded];
            quantize_input_scalar(&x, &mut want);
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            {
                let mut got = vec![7i16; padded];
                // SAFETY: the `cfg` is the build enabling AVX2.
                unsafe { quantize_input_avx2(&x, &mut got) };
                assert_eq!(got, want, "n={n}");
            }
            assert!(
                want[n..].iter().all(|&v| v == 0),
                "the padding is zero, n={n}"
            );
            // The rails land exactly on 0 and INPUT_LEVELS.
            let mut rails = vec![0i16; 16];
            let mut x = vec![0.0f32; 16];
            x[1] = 1.0;
            quantize_input(&x, &mut rails);
            assert_eq!((rails[0], rails[1]), (0, INPUT_LEVELS as i16));
        }
    }

    #[test]
    fn the_vector_dot_products_are_the_scalar_one_exactly() {
        for n in [16usize, 32, 768, 800, 1200] {
            let x: Vec<i16> = stream(n, 2).iter().map(|v| (v * 4096.0) as i16).collect();
            let data: Vec<i16> = stream(8 * n, 3)
                .iter()
                .map(|v| (v * 32767.0) as i16)
                .collect();
            let rows: [&[i16]; 8] = std::array::from_fn(|r| &data[r * n..(r + 1) * n]);
            let want = dots_scalar(rows, &x);
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: the build is for the host CPU, which has AVX2.
                let madd = unsafe { dots_madd(rows, &x) };
                assert_eq!(madd, want, "vpmaddwd, n={n}");
            }
            #[cfg(all(target_arch = "x86_64", target_feature = "avxvnni"))]
            {
                // SAFETY: the `cfg` is the build enabling AVX-VNNI.
                let vnni = unsafe { dots_vnni(rows, &x) };
                assert_eq!(vnni, want, "vpdpwssd, n={n}");
            }
            assert_eq!(dots(rows, &x), want, "dispatched, n={n}");
        }
    }

    /// The bound in `row_levels`, held to the worst input a row can be given: each
    /// positive weight against 4096 and each negative one against zero, and the
    /// reverse. If the row total is exact (equal to the same sum in `i64`) it did
    /// not wrap, whatever the partial sums did.
    #[test]
    fn a_row_total_cannot_overflow_an_i32() {
        // Rows built to be hard: big weights, all one sign, a few huge outliers.
        for (name, cols, make) in [
            (
                "all positive",
                1196usize,
                (|_: usize, v: f32| v.abs()) as fn(usize, f32) -> f32,
            ),
            ("all negative", 1196, |_, v| -v.abs()),
            ("mixed", 1196, |_, v| v),
            ("outlier", 800, |i, v| if i == 5 { 40.0 } else { v * 0.01 }),
            ("narrow", 33, |_, v| v.abs()),
        ] {
            let rows = 6;
            let w: Vec<f32> = stream(rows * cols, 9)
                .iter()
                .enumerate()
                .map(|(i, &v)| make(i % cols, v))
                .collect();
            let q = QuantW1::new(&w, rows, cols);
            for row in 0..rows {
                let weights = q.weights(row);
                for sign in [1i32, -1] {
                    // The input that maximises `sign · total`: full where `sign · w`
                    // is positive.
                    let x: Vec<i16> = (0..q.stride())
                        .map(|i| {
                            if i < cols && sign * i32::from(weights[i]) > 0 {
                                INPUT_LEVELS as i16
                            } else {
                                0
                            }
                        })
                        .collect();
                    let exact: i64 = weights
                        .iter()
                        .zip(&x)
                        .map(|(&w, &x)| i64::from(w) * i64::from(x))
                        .sum();
                    assert!(
                        exact.abs() <= i64::from(i32::MAX),
                        "{name} row {row}: the worst total {exact} does not fit"
                    );
                    assert_eq!(i64::from(q.row_total(row, &x)), exact, "{name} row {row}");
                }
            }
        }
    }

    /// The levels a row gets follow from its weights and nothing else: the most that
    /// keep the worst total inside an `i32`, never more than an `i16` holds.
    #[test]
    fn a_row_gets_the_most_levels_that_fit() {
        // Small weights, a few of them: fits every level an `i16` has.
        let few = vec![0.0f32, 1.0, -1.0, 0.0];
        let q = QuantW1::new(&few, 1, 4);
        assert_eq!(
            q.weights(0).iter().map(|v| v.unsigned_abs()).max(),
            Some(32_767)
        );

        // Rows that cannot: the top level shrinks as the weights' sum grows, and the
        // worst total for it stays inside `i32` while one level more would not.
        for cols in [512usize, 800, 1196] {
            let w: Vec<f32> = stream(cols, 4)
                .iter()
                .map(|v| v.abs() * 0.1 + 0.02)
                .collect();
            let q = QuantW1::new(&w, 1, cols);
            let weights = q.weights(0);
            let top = weights.iter().map(|v| v.unsigned_abs()).max().unwrap();
            assert!((1..32_767).contains(&top), "{cols}: {top}");
            let total: i64 =
                weights.iter().map(|&v| i64::from(v)).sum::<i64>() * INPUT_LEVELS as i64;
            assert!(
                total <= i64::from(i32::MAX),
                "{cols}: the worst total {total} overflows"
            );
            // Within a few percent of the bound, so a row is not being given fewer
            // levels than it must (rounding takes half a level per weight off).
            assert!(
                total as f64 > 0.9 * f64::from(i32::MAX),
                "{cols}: the bound is far from tight ({total})"
            );
        }
    }

    #[test]
    fn the_layer_is_the_f32_layer_to_within_the_two_roundings() {
        for (rows, cols) in [(32usize, 768usize), (32, 800), (32, 1196), (5, 33)] {
            let w: Vec<f32> = stream(rows * cols, 5).iter().map(|v| v * 0.08).collect();
            let bias: Vec<f32> = stream(rows, 6).iter().map(|v| v * 0.2).collect();
            let x = clipped(cols, 7);
            let q = QuantW1::new(&w, rows, cols);
            let mut got = vec![0.0f32; rows];
            q.layer_clip_any_width(&x, &bias, &mut got);
            let mut worst = 0.0f64;
            for row in 0..rows {
                let sum: f64 = (0..cols)
                    .map(|i| f64::from(w[row * cols + i]) * f64::from(x[i]))
                    .sum();
                let want = (f64::from(bias[row]) + sum).clamp(0.0, 1.0);
                worst = worst.max((f64::from(got[row]) - want).abs());
            }
            // Input rounding is at most half of 1/4096 per entry against weights of
            // a few hundredths, summed over `cols`; weight rounding is far smaller.
            assert!(worst < 2.0e-3, "{rows}x{cols}: worst error {worst}");
            println!("{rows}x{cols}: worst |error| {worst:.2e}");
            // The scale is what turns a total into the layer's sum.
            assert!(q.scale(0) > 0.0);
        }
    }

    #[test]
    fn a_zero_row_stays_zero() {
        let w = vec![0.0f32; 2 * 32];
        let q = QuantW1::new(&w, 2, 32);
        let mut out = [9.0f32; 2];
        q.layer_clip_any_width(&clipped(32, 1), &[0.25, -0.25], &mut out);
        assert_eq!(out, [0.25, 0.0]);
    }
}
