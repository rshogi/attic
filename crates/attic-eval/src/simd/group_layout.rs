//! The group-major weight layouts the AVX-512 chain of the 64-wide SFNN
//! shapes reads.
//!
//! The file stores every dense layer row-major, `[output][padded input]`, and
//! the scalar, AVX2 and 1536 AVX-512 kernels read it that way: one output at a
//! time, a dot product along its row. [`super::avx512_packed_tail`] goes the
//! other way, as the reference's `AffineTransformSparseInputExplicit` and
//! `PropagateClippedReLUPackedToOutput` do: one 4-byte group of the input at a
//! time, into every output at once. That needs the four weights each output
//! gives a group side by side, so these functions copy the row-major arrays
//! into that order once, when the network is built.
//!
//! Pure index arithmetic, compiled on every target, so the tests below run on
//! every host and not only on one with AVX-512.

/// Input bytes one `vpdpbusd` lane multiplies and adds.
pub const GROUP: usize = 4;

/// Copy row-major `weights` (`rows` outputs, `padded_in` inputs each) into
/// `out` grouped by 4-byte input group:
/// `out[(i / 4) * rows * 4 + o * 4 + i % 4] = weights[o * padded_in + i]`.
///
/// Group `g` is then `rows * 4` contiguous bytes, output `o`'s four weights for
/// inputs `4g..4g + 4` at `o * 4`: for 16 outputs, one 512-bit register whose
/// `i32` lane `o` is output `o`'s share of the group. The reference's
/// `get_weight_index_scrambled` is the same map.
///
/// # Panics
/// Panics unless `padded_in` is a multiple of 4 and both slices are
/// `rows * padded_in` long.
pub fn by_group(weights: &[i8], rows: usize, padded_in: usize, out: &mut [i8]) {
    by_block(weights, rows, padded_in, GROUP, out);
}

/// Input bytes in one 512-bit register: the block the dense fc_0 of
/// Cyfamate's threat shape reads its input and its weights in.
pub const CHUNK: usize = 64;

/// Copy row-major `weights` into `out` chunk by chunk of [`CHUNK`] inputs:
/// `out[(i / 64) * rows * 64 + o * 64 + i % 64] = weights[o * padded_in + i]`.
///
/// Chunk `c` is then `rows` registers in a row, register `o` holding output
/// `o`'s weights for inputs `64c..64c + 64`: what one `vpdpbusd` multiplies
/// chunk `c` of the input by for that output.
///
/// # Panics
/// Panics unless `padded_in` is a multiple of 64 and both slices are
/// `rows * padded_in` long.
pub fn by_chunk(weights: &[i8], rows: usize, padded_in: usize, out: &mut [i8]) {
    by_block(weights, rows, padded_in, CHUNK, out);
}

/// fc_0's weights in the order the AVX-512 chain reads them: [`by_chunk`] for
/// Cyfamate's threat shape, whose fc_0 runs dense, [`by_group`] for the shapes
/// whose fc_0 skips the zero groups of its input.
///
/// # Panics
/// As the layout it picks.
pub fn fc_0_for_chain(weights: &[i8], rows: usize, padded_in: usize, out: &mut [i8]) {
    if cfg!(feature = "sfnn_threat") {
        by_chunk(weights, rows, padded_in, out);
    } else {
        by_group(weights, rows, padded_in, out);
    }
}

/// `out[(i / width) * rows * width + o * width + i % width] = weights[o * padded_in + i]`.
fn by_block(weights: &[i8], rows: usize, padded_in: usize, width: usize, out: &mut [i8]) {
    assert!(
        padded_in.is_multiple_of(width),
        "padded_in must be whole blocks of {width}"
    );
    assert_eq!(weights.len(), rows * padded_in);
    assert_eq!(out.len(), rows * padded_in);
    for o in 0..rows {
        for i in 0..padded_in {
            out[(i / width) * rows * width + o * width + i % width] = weights[o * padded_in + i];
        }
    }
}

/// Where each 4-byte group of fc_2's 64 inputs lands once fc_1's 64 outputs
/// have been narrowed to bytes by two `packs` in the AVX-512 tail.
///
/// `packs` works within each 128-bit lane, so narrowing the four 16-output
/// registers `acc[0..4]` leaves lane `L` holding outputs `4L..4L+4` of each in
/// turn: group `4L + q` of the result is natural group `4q + L`. Entry `n` is
/// where natural group `n` ends up, which is the reference's `kPerm`.
pub const FC_2_PACKED_ORDER: [usize; 16] = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];

/// fc_2's 64 weights in the order the AVX-512 tail hands it fc_1's clipped
/// output: natural group `n` moved to group [`FC_2_PACKED_ORDER`]`[n]`.
///
/// # Panics
/// Panics unless both slices are 64 long.
pub fn fc_2_packed(weights: &[i8], out: &mut [i8]) {
    assert_eq!(weights.len(), FC_2_PACKED_ORDER.len() * GROUP);
    assert_eq!(out.len(), weights.len());
    for (n, &to) in FC_2_PACKED_ORDER.iter().enumerate() {
        out[to * GROUP..][..GROUP].copy_from_slice(&weights[n * GROUP..][..GROUP]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn by_group_puts_each_output_s_four_weights_of_a_group_side_by_side() {
        let (rows, padded_in) = (16, 32);
        let weights: Vec<i8> = (0..rows * padded_in).map(|k| (k % 251) as i8).collect();
        let mut out = vec![0i8; weights.len()];
        by_group(&weights, rows, padded_in, &mut out);
        for g in 0..padded_in / GROUP {
            let block = &out[g * rows * GROUP..][..rows * GROUP];
            for o in 0..rows {
                assert_eq!(
                    &block[o * GROUP..][..GROUP],
                    &weights[o * padded_in + g * GROUP..][..GROUP],
                    "group {g}, output {o}"
                );
            }
        }
    }

    #[test]
    fn by_group_is_a_permutation() {
        let (rows, padded_in) = (64, 32);
        let weights: Vec<i8> = (0..rows * padded_in)
            .map(|k| (k as i32 - 1000) as i8)
            .collect();
        let mut out = vec![0i8; weights.len()];
        by_group(&weights, rows, padded_in, &mut out);
        let mut a: Vec<i8> = weights.clone();
        let mut b = out.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }

    #[test]
    fn by_chunk_puts_each_output_s_sixty_four_weights_of_a_chunk_in_one_row() {
        let (rows, padded_in) = (16, 256);
        let weights: Vec<i8> = (0..rows * padded_in).map(|k| (k % 251) as i8).collect();
        let mut out = vec![0i8; weights.len()];
        by_chunk(&weights, rows, padded_in, &mut out);
        for c in 0..padded_in / CHUNK {
            let block = &out[c * rows * CHUNK..][..rows * CHUNK];
            for o in 0..rows {
                assert_eq!(
                    &block[o * CHUNK..][..CHUNK],
                    &weights[o * padded_in + c * CHUNK..][..CHUNK],
                    "chunk {c}, output {o}"
                );
            }
        }
    }

    /// The order is the one two rounds of `packs` produce: lane `L` of
    /// `packs(packs(r0, r1), packs(r2, r3))` holds `r0[4L..]`, `r1[4L..]`,
    /// `r2[4L..]`, `r3[4L..]`, four lanes of four outputs each.
    #[test]
    fn fc_2_packed_order_is_what_two_packs_leave() {
        let mut packed_group_of_natural = [usize::MAX; 16];
        for lane in 0..4 {
            for q in 0..4 {
                // Natural group `4q + lane` is outputs `16q + 4 lane ..`, which
                // the packs put at position `q` of 128-bit lane `lane`.
                packed_group_of_natural[4 * q + lane] = 4 * lane + q;
            }
        }
        assert_eq!(packed_group_of_natural, FC_2_PACKED_ORDER);
    }

    #[test]
    fn fc_2_packed_moves_whole_groups() {
        let weights: Vec<i8> = (0..64).map(|k| k as i8).collect();
        let mut out = vec![0i8; 64];
        fc_2_packed(&weights, &mut out);
        for (n, &to) in FC_2_PACKED_ORDER.iter().enumerate() {
            assert_eq!(&out[to * GROUP..][..GROUP], &weights[n * GROUP..][..GROUP]);
        }
    }
}
