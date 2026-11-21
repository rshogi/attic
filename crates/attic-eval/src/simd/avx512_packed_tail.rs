//! The dense chain of the 64-wide SFNN shapes (the 1024s, NAGISA V4's 2304 and
//! Cyfamate's threat shape) on AVX-512 VNNI: the reference's `kUsePackedTail`
//! path, which it takes for exactly these shapes (`USE_AVX512 && kHidden2Dims == 64`).
//!
//! [`super::avx512_post_ft::fused_fc_chain`] cannot serve them: it takes fc_1's
//! and fc_2's inputs as one 32-byte register, and fc_2 reads 64 here. The
//! reference does not widen that kernel either. It turns every layer around,
//! reading the weights group-major ([`super::group_layout`]) so that one
//! 4-byte group of the input goes into all of a layer's outputs at once:
//!
//! - **fc_0** (2304 → 16, or 1024 → 16): one 512-bit register holds all
//!   sixteen outputs, and every nonzero group of the input is one broadcast and
//!   one `vpdpbusd`. The groups that are zero, which the element-wise multiply
//!   leaves many of, are listed out sixteen at a time and never loaded. Their
//!   product is zero, so skipping them changes nothing.
//! - **fc_1** (30 → 64): four registers, the eight input groups broadcast in
//!   turn, then the clipped ReLU on all 64 outputs as two `packs`, a shift and a
//!   `max`, left in a register.
//! - **fc_2** (64 → 1): one `vpdpbusd` against fc_2's weights, which
//!   `fc_2_packed` laid out in the order the `packs` leave fc_1's bytes in.
//!
//! Every step is bit for bit the scalar chain's (`network::per_layer_flow_unfused`),
//! which the tests below hold it to: `vpdpbusd` adds products without
//! saturating and wraps its `i32` lanes as the scalar `wrapping_add` does, and
//! the clipped ReLU's two saturating narrows land on the same `[0, 127]` the
//! scalar clamp does (see [`clipped_relu_64`]).

use std::arch::x86_64::{
    __m512i, _mm512_add_epi32, _mm512_cmpneq_epi32_mask, _mm512_dpbusd_epi32, _mm512_loadu_si512,
    _mm512_maskz_compress_epi32, _mm512_max_epi8, _mm512_packs_epi16, _mm512_packs_epi32,
    _mm512_reduce_add_epi32, _mm512_set1_epi32, _mm512_setr_epi32, _mm512_setzero_si512,
    _mm512_srai_epi16, _mm512_storeu_si512,
};

use crate::simd::group_layout::GROUP;
use crate::simd::scalar_post_ft;
use crate::types::{
    FC_0_INPUT_DIMS, FC_0_OUTPUT_DIMS, FC_0_PADDED_INPUT_DIMS, FC_1_OUTPUT_DIMS,
    FC_1_PADDED_INPUT_DIMS, FC_2_OUTPUT_DIMS, FC_2_PADDED_INPUT_DIMS, HIDDEN1_DIMS,
};

const WEIGHT_SCALE_BITS: u32 = 6;
/// `i32` lanes in a 512-bit register, and so outputs a register holds.
const I32_LANES: usize = 16;
/// Input groups fc_0 tests for zero with one compare.
const GROUPS_PER_MASK: usize = I32_LANES;

// What the chain below hard-codes: fc_0 and its sixteen outputs are one
// register, fc_1 reads eight groups into four registers, fc_2 is one output
// over one register of input.
const _: () = {
    assert!(FC_0_OUTPUT_DIMS == I32_LANES);
    assert!(FC_0_INPUT_DIMS == FC_0_PADDED_INPUT_DIMS);
    assert!(FC_0_INPUT_DIMS.is_multiple_of(GROUP * GROUPS_PER_MASK));
    assert!(FC_1_OUTPUT_DIMS == 4 * I32_LANES);
    assert!(FC_1_PADDED_INPUT_DIMS == 32);
    assert!(2 * HIDDEN1_DIMS <= FC_1_PADDED_INPUT_DIMS);
    assert!(FC_2_OUTPUT_DIMS == 1);
    assert!(FC_2_PADDED_INPUT_DIMS == 64);
};

/// The group-major weights and the biases of one layer stack, as
/// `NetworkStack` holds them.
pub struct Stack<'a> {
    pub fc_0_biases: &'a [i32],
    /// fc_0's weights, [`by_group`](super::group_layout::by_group).
    pub fc_0_grouped: &'a [i8],
    pub fc_1_biases: &'a [i32],
    /// fc_1's weights, [`by_group`](super::group_layout::by_group).
    pub fc_1_grouped: &'a [i8],
    pub fc_2_biases: &'a [i32],
    /// fc_2's weights, [`fc_2_packed`](super::group_layout::fc_2_packed).
    pub fc_2_packed: &'a [i8],
}

/// The whole `fc_0 → SqrClippedReLU/ClippedReLU → fc_1 → ClippedReLU → fc_2 →
/// shortcut` chain, returning the network's raw output before `FV_SCALE`.
///
/// # Panics
/// Panics unless every slice of `stack` is the length this build's shape gives
/// it. The loads below go through raw pointers and rely on that.
///
/// # Safety
/// The running CPU must support `avx512f`, `avx512bw` and `avx512vnni`.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn fc_chain(transformed: &[u8; FC_0_INPUT_DIMS], stack: &Stack<'_>) -> i32 {
    assert!(
        stack.fc_0_biases.len() == FC_0_OUTPUT_DIMS
            && stack.fc_0_grouped.len() == FC_0_OUTPUT_DIMS * FC_0_PADDED_INPUT_DIMS
            && stack.fc_1_biases.len() == FC_1_OUTPUT_DIMS
            && stack.fc_1_grouped.len() == FC_1_OUTPUT_DIMS * FC_1_PADDED_INPUT_DIMS
            && stack.fc_2_biases.len() == FC_2_OUTPUT_DIMS
            && stack.fc_2_packed.len() == FC_2_OUTPUT_DIMS * FC_2_PADDED_INPUT_DIMS,
        "the stack's dense layers are not the sizes this build's shape needs"
    );

    // SAFETY: the target_feature grant covers the callees; the lengths are
    // asserted above.
    let fc_0_out = unsafe { fc_0(transformed, stack.fc_0_biases, stack.fc_0_grouped) };

    // fc_1's input: [SqrClippedReLU(15) | ClippedReLU(15) | 0, 0], the 16th
    // fc_0 output being the shortcut. Fifteen lanes are below the width the
    // SIMD activations take, so these are the scalar ones, as everywhere else.
    let mut fc_1_in = [0u8; FC_1_PADDED_INPUT_DIMS];
    scalar_post_ft::sqr_clipped_relu(&fc_0_out[..HIDDEN1_DIMS], &mut fc_1_in[..HIDDEN1_DIMS]);
    scalar_post_ft::clipped_relu(
        &fc_0_out[..HIDDEN1_DIMS],
        &mut fc_1_in[HIDDEN1_DIMS..2 * HIDDEN1_DIMS],
    );

    // SAFETY: as above.
    let fc_2_out = unsafe {
        let ac_1 = fc_1_clipped(&fc_1_in, stack.fc_1_biases, stack.fc_1_grouped);
        fc_2(ac_1, stack.fc_2_biases[0], stack.fc_2_packed)
    };

    // The shortcut output is raw, pre-ReLU, and can be negative.
    fc_2_out.wrapping_add(fc_0_out[HIDDEN1_DIMS])
}

/// fc_0's input groups.
const FC_0_GROUPS: usize = FC_0_INPUT_DIMS / GROUP;

/// fc_0 over the nonzero input groups only.
///
/// Two passes. The first lists the nonzero groups: sixteen are compared with
/// zero at a time and the indices of the nonzero ones compressed to the front
/// of a register and stored, with no branch per group. The second adds each
/// listed group into one of four accumulators in turn, so that consecutive
/// `vpdpbusd`s do not wait on each other's result; the four are summed at the
/// end. `i32` addition wraps and commutes, so the split changes no bit.
///
/// # Safety
/// The CPU must support `avx512f`, `avx512bw` and `avx512vnni`; `biases` must
/// be 16 long and `grouped` `16 * FC_0_PADDED_INPUT_DIMS`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn fc_0(
    input: &[u8; FC_0_INPUT_DIMS],
    biases: &[i32],
    grouped: &[i8],
) -> [i32; FC_0_OUTPUT_DIMS] {
    let in_ptr = input.as_ptr();
    let w_ptr = grouped.as_ptr();

    // Room for one whole register past the last group: every store writes
    // sixteen indices, however few of them count.
    let mut nonzero = [0u32; FC_0_GROUPS + GROUPS_PER_MASK];
    let mut count = 0usize;
    let zero = _mm512_setzero_si512();
    let lanes = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    for block in 0..FC_0_GROUPS / GROUPS_PER_MASK {
        // SAFETY: `(block + 1) * 64 <= FC_0_INPUT_DIMS`, the input's length; the
        // store's sixteen slots start at `count <= block * 16`, inside the list.
        unsafe {
            let words = _mm512_loadu_si512(
                in_ptr
                    .add(block * GROUPS_PER_MASK * GROUP)
                    .cast::<__m512i>(),
            );
            let mask = _mm512_cmpneq_epi32_mask(words, zero);
            let ids = _mm512_add_epi32(lanes, _mm512_set1_epi32((block * GROUPS_PER_MASK) as i32));
            _mm512_storeu_si512(
                nonzero.as_mut_ptr().add(count).cast::<__m512i>(),
                _mm512_maskz_compress_epi32(mask, ids),
            );
            count += mask.count_ones() as usize;
        }
    }

    // SAFETY: `biases` is 16 `i32`, one register.
    let mut acc = [
        unsafe { _mm512_loadu_si512(biases.as_ptr().cast::<__m512i>()) },
        zero,
        zero,
        zero,
    ];
    let (quads, rest) = nonzero[..count].as_chunks::<4>();
    for q in quads {
        for (a, &g) in acc.iter_mut().zip(q) {
            // SAFETY: see `add_group`; `g` came from the list above.
            *a = unsafe { add_group(*a, in_ptr, w_ptr, g as usize) };
        }
    }
    for &g in rest {
        // SAFETY: as above.
        acc[0] = unsafe { add_group(acc[0], in_ptr, w_ptr, g as usize) };
    }
    let sum = _mm512_add_epi32(
        _mm512_add_epi32(acc[0], acc[1]),
        _mm512_add_epi32(acc[2], acc[3]),
    );
    let mut out = [0i32; FC_0_OUTPUT_DIMS];
    // SAFETY: `out` is 16 `i32`, one register.
    unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast::<__m512i>(), sum) };
    out
}

/// `acc` plus input group `g` times its sixteen outputs' weights.
///
/// # Safety
/// The CPU must support `avx512f`, `avx512bw` and `avx512vnni`. `g` must be a
/// group of fc_0's input: its 4 bytes inside `input`, and its 64 weight bytes
/// (16 outputs * 4) inside fc_0's group-major weights at `weights`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn add_group(acc: __m512i, input: *const u8, weights: *const i8, g: usize) -> __m512i {
    debug_assert!(g < FC_0_GROUPS);
    // SAFETY: the caller's.
    unsafe {
        let word = input.add(g * GROUP).cast::<i32>().read_unaligned();
        let col = _mm512_loadu_si512(weights.add(g * FC_0_OUTPUT_DIMS * GROUP).cast::<__m512i>());
        _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(word), col)
    }
}

/// fc_1 and the clipped ReLU after it, returned as the 64 bytes fc_2 reads, in
/// the order [`clipped_relu_64`] leaves them.
///
/// # Safety
/// The CPU must support `avx512f`, `avx512bw` and `avx512vnni`; `biases` must
/// be 64 long and `grouped` `64 * 32`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn fc_1_clipped(
    input: &[u8; FC_1_PADDED_INPUT_DIMS],
    biases: &[i32],
    grouped: &[i8],
) -> __m512i {
    let b_ptr = biases.as_ptr();
    let w_ptr = grouped.as_ptr();
    let mut acc = [_mm512_setzero_si512(); 4];
    for (k, a) in acc.iter_mut().enumerate() {
        // SAFETY: `biases` is 64 `i32`, four registers.
        *a = unsafe { _mm512_loadu_si512(b_ptr.add(k * I32_LANES).cast::<__m512i>()) };
    }
    for g in 0..FC_1_PADDED_INPUT_DIMS / GROUP {
        let word = i32::from_le_bytes(
            input[g * GROUP..][..GROUP]
                .try_into()
                .expect("a group is four bytes"),
        );
        if word == 0 {
            continue;
        }
        let x = _mm512_set1_epi32(word);
        for (k, a) in acc.iter_mut().enumerate() {
            // SAFETY: group `g`'s 256 weight bytes are `g * 256 ..`, and
            // `g < 8` keeps them inside the 2048 of `grouped`.
            let col = unsafe {
                _mm512_loadu_si512(
                    w_ptr
                        .add((g * FC_1_OUTPUT_DIMS + k * I32_LANES) * GROUP)
                        .cast::<__m512i>(),
                )
            };
            *a = _mm512_dpbusd_epi32(*a, x, col);
        }
    }
    // SAFETY: this function's features include the avx512f + avx512bw the callee names.
    unsafe { clipped_relu_64(acc) }
}

/// `clamp(x >> 6, 0, 127)` on the 64 `i32` of `acc`, narrowed to bytes.
///
/// `packs_epi32` saturates to `i16`, the shift is arithmetic, and `packs_epi16`
/// then saturates to `i8`, after which `max` with zero takes the floor. A lane
/// in `i16` range shifts exactly as it would as an `i32`; one above it
/// saturates to 32767, shifts to 511 and saturates to 127, as the scalar clamp
/// gives anything whose shift is at least 512; one below saturates, shifts
/// negative and comes out 0. So every `i32` lands where the scalar clamp puts
/// it.
///
/// `packs` works within each 128-bit lane, so the bytes come out with natural
/// group `4q + L` at group `4L + q`: see
/// [`FC_2_PACKED_ORDER`](super::group_layout::FC_2_PACKED_ORDER).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn clipped_relu_64(acc: [__m512i; 4]) -> __m512i {
    let words0 = _mm512_srai_epi16::<WEIGHT_SCALE_BITS>(_mm512_packs_epi32(acc[0], acc[1]));
    let words1 = _mm512_srai_epi16::<WEIGHT_SCALE_BITS>(_mm512_packs_epi32(acc[2], acc[3]));
    _mm512_max_epi8(_mm512_packs_epi16(words0, words1), _mm512_setzero_si512())
}

/// fc_2 over fc_1's clipped bytes.
///
/// # Safety
/// The CPU must support `avx512f`, `avx512bw` and `avx512vnni`; `packed` must
/// be 64 long.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn fc_2(ac_1: __m512i, bias: i32, packed: &[i8]) -> i32 {
    // SAFETY: `packed` is 64 bytes, one register.
    let w = unsafe { _mm512_loadu_si512(packed.as_ptr().cast::<__m512i>()) };
    let sum = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ac_1, w);
    bias.wrapping_add(_mm512_reduce_add_epi32(sum))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::group_layout::{by_group, fc_2_packed};

    macro_rules! require_vnni {
        () => {
            if !(std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vnni"))
            {
                eprintln!("skipping AVX-512 VNNI packed-tail test: features unavailable");
                return;
            }
        };
    }

    /// A stack's row-major arrays, as the file holds them.
    struct RowMajor {
        fc_0_biases: Vec<i32>,
        fc_0_weights: Vec<i8>,
        fc_1_biases: Vec<i32>,
        fc_1_weights: Vec<i8>,
        fc_2_biases: Vec<i32>,
        fc_2_weights: Vec<i8>,
    }

    /// The same arrays group-major, as the chain reads them.
    struct Grouped {
        fc_0: Vec<i8>,
        fc_1: Vec<i8>,
        fc_2: Vec<i8>,
    }

    impl RowMajor {
        fn grouped(&self) -> Grouped {
            let mut fc_0 = vec![0i8; self.fc_0_weights.len()];
            by_group(
                &self.fc_0_weights,
                FC_0_OUTPUT_DIMS,
                FC_0_PADDED_INPUT_DIMS,
                &mut fc_0,
            );
            let mut fc_1 = vec![0i8; self.fc_1_weights.len()];
            by_group(
                &self.fc_1_weights,
                FC_1_OUTPUT_DIMS,
                FC_1_PADDED_INPUT_DIMS,
                &mut fc_1,
            );
            let mut fc_2 = vec![0i8; self.fc_2_weights.len()];
            fc_2_packed(&self.fc_2_weights, &mut fc_2);
            Grouped { fc_0, fc_1, fc_2 }
        }

        fn stack<'a>(&'a self, g: &'a Grouped) -> Stack<'a> {
            Stack {
                fc_0_biases: &self.fc_0_biases,
                fc_0_grouped: &g.fc_0,
                fc_1_biases: &self.fc_1_biases,
                fc_1_grouped: &g.fc_1,
                fc_2_biases: &self.fc_2_biases,
                fc_2_packed: &g.fc_2,
            }
        }

        /// The scalar chain over the row-major arrays: the reference this
        /// module has to equal.
        fn scalar(&self, transformed: &[u8; FC_0_INPUT_DIMS]) -> i32 {
            use crate::types::{FC_1_INPUT_DIMS, FC_2_INPUT_DIMS};
            let mut fc_0_out = [0i32; FC_0_OUTPUT_DIMS];
            scalar_post_ft::affine(
                &mut fc_0_out,
                &self.fc_0_biases,
                &self.fc_0_weights,
                transformed,
                FC_0_INPUT_DIMS,
                FC_0_PADDED_INPUT_DIMS,
            );
            let mut fc_1_in = [0u8; FC_1_PADDED_INPUT_DIMS];
            scalar_post_ft::sqr_clipped_relu(
                &fc_0_out[..HIDDEN1_DIMS],
                &mut fc_1_in[..HIDDEN1_DIMS],
            );
            scalar_post_ft::clipped_relu(
                &fc_0_out[..HIDDEN1_DIMS],
                &mut fc_1_in[HIDDEN1_DIMS..2 * HIDDEN1_DIMS],
            );
            let mut fc_1_out = [0i32; FC_1_OUTPUT_DIMS];
            scalar_post_ft::affine(
                &mut fc_1_out,
                &self.fc_1_biases,
                &self.fc_1_weights,
                &fc_1_in,
                FC_1_INPUT_DIMS,
                FC_1_PADDED_INPUT_DIMS,
            );
            let mut ac_1 = [0u8; FC_1_OUTPUT_DIMS];
            scalar_post_ft::clipped_relu(&fc_1_out, &mut ac_1);
            let mut fc_2_out = [0i32; FC_2_OUTPUT_DIMS];
            scalar_post_ft::affine(
                &mut fc_2_out,
                &self.fc_2_biases,
                &self.fc_2_weights,
                &ac_1,
                FC_2_INPUT_DIMS,
                FC_2_PADDED_INPUT_DIMS,
            );
            fc_2_out[0].wrapping_add(fc_0_out[HIDDEN1_DIMS])
        }
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Weights drawn from `pick`, biases spread wide enough that every
    /// activation sees its floor, its ceiling and the range between.
    fn random_stack(rng: &mut Rng, pick: impl Fn(&mut Rng) -> i8, bias: i64) -> RowMajor {
        let i8s = |n: usize, rng: &mut Rng| -> Vec<i8> { (0..n).map(|_| pick(rng)).collect() };
        let fc_0_weights = i8s(FC_0_OUTPUT_DIMS * FC_0_PADDED_INPUT_DIMS, rng);
        let fc_1_weights = i8s(FC_1_OUTPUT_DIMS * FC_1_PADDED_INPUT_DIMS, rng);
        let fc_2_weights = i8s(FC_2_OUTPUT_DIMS * FC_2_PADDED_INPUT_DIMS, rng);
        let mut i32s = |n: usize| -> Vec<i32> {
            (0..n)
                .map(|_| (rng.next() % (2 * bias as u64 + 1)) as i64 - bias)
                .map(|v| v as i32)
                .collect()
        };
        RowMajor {
            fc_0_biases: i32s(FC_0_OUTPUT_DIMS),
            fc_0_weights,
            fc_1_biases: i32s(FC_1_OUTPUT_DIMS),
            fc_1_weights,
            fc_2_biases: i32s(FC_2_OUTPUT_DIMS),
            fc_2_weights,
        }
    }

    /// Inputs as the output transform makes them, at most 126, with a share of
    /// zeros: the zero groups are the ones fc_0 skips.
    fn random_input(rng: &mut Rng, zero_permille: u64) -> [u8; FC_0_INPUT_DIMS] {
        let mut t = [0u8; FC_0_INPUT_DIMS];
        for x in t.iter_mut() {
            *x = if rng.next() % 1000 < zero_permille {
                0
            } else {
                (rng.next() % 127) as u8
            };
        }
        t
    }

    #[test]
    fn the_chain_matches_the_scalar_one_on_random_stacks_and_inputs() {
        require_vnni!();
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for round in 0..64 {
            let weights: fn(&mut Rng) -> i8 = match round % 4 {
                0 => |r| r.next() as i8,
                1 => |_| i8::MIN,
                2 => |_| i8::MAX,
                _ => |r| if r.next() & 1 == 0 { i8::MIN } else { i8::MAX },
            };
            let bias = [100_000i64, 10_000, 1_000_000, i32::MAX as i64][round % 4];
            let row = random_stack(&mut rng, weights, bias);
            let grouped = row.grouped();
            let stack = row.stack(&grouped);
            for zeros in [0u64, 500, 900, 1000] {
                let t = random_input(&mut rng, zeros);
                // SAFETY: guarded by `require_vnni!`.
                let simd = unsafe { fc_chain(&t, &stack) };
                assert_eq!(
                    simd,
                    row.scalar(&t),
                    "round {round}, {zeros} per mille zeros"
                );
            }
        }
    }

    /// Every `i32` fc_1 can put in a lane, through the clipped ReLU, against the
    /// scalar clamp: the boundaries of each saturation and of the shift.
    #[test]
    fn clipped_relu_64_matches_the_scalar_clamp_at_every_boundary() {
        require_vnni!();
        let edges = [
            i32::MIN,
            i32::MIN + 1,
            -2_097_152,
            -32_769,
            -32_768,
            -32_767,
            -65,
            -64,
            -63,
            -1,
            0,
            1,
            63,
            64,
            65,
            8_127,
            8_128,
            8_191,
            8_192,
            32_766,
            32_767,
            32_768,
            1 << 20,
            i32::MAX - 1,
            i32::MAX,
        ];
        let mut values = [0i32; FC_1_OUTPUT_DIMS];
        for start in 0..edges.len() {
            for (k, v) in values.iter_mut().enumerate() {
                *v = edges[(start + k) % edges.len()];
            }
            // SAFETY: guarded by `require_vnni!`; `values` is 64 `i32`.
            let bytes: [u8; 64] = unsafe {
                let acc: [__m512i; 4] = std::array::from_fn(|k| {
                    _mm512_loadu_si512(values.as_ptr().add(k * I32_LANES).cast::<__m512i>())
                });
                std::mem::transmute(clipped_relu_64(acc))
            };
            let mut want = [0u8; FC_1_OUTPUT_DIMS];
            scalar_post_ft::clipped_relu(&values, &mut want);
            for (n, &to) in crate::simd::group_layout::FC_2_PACKED_ORDER
                .iter()
                .enumerate()
            {
                assert_eq!(
                    bytes[to * GROUP..][..GROUP],
                    want[n * GROUP..][..GROUP],
                    "start {start}, natural group {n}"
                );
            }
        }
    }

    #[test]
    fn a_stack_of_other_sizes_is_refused() {
        // Gated like the others: the function is compiled for AVX-512, so even
        // the check at its head is not safe to reach on a host without it.
        require_vnni!();
        let short = [0i32; 1];
        let bytes = [0i8; 1];
        let stack = Stack {
            fc_0_biases: &short,
            fc_0_grouped: &bytes,
            fc_1_biases: &short,
            fc_1_grouped: &bytes,
            fc_2_biases: &short,
            fc_2_packed: &bytes,
        };
        // SAFETY: guarded by `require_vnni!`.
        let refused =
            std::panic::catch_unwind(|| unsafe { fc_chain(&[0u8; FC_0_INPUT_DIMS], &stack) });
        assert!(refused.is_err());
    }
}
