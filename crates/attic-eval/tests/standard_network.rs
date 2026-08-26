//! Loading the two standard YaneuraOu shapes from a real `nn.bin`.
//!
//! Gated on an environment variable naming the file, the same way `loader.rs`'s
//! `ATTIC_HALFKA2304_NN` unit test is: these networks are large and none of
//! them live in the repository. Point it at the file that matches whichever of
//! `kp256` / `halfkp256` the crate was built with.

#![cfg(any(feature = "nnue_kp_256x2_32_32", feature = "nnue_halfkp_256x2_32_32"))]

use std::path::PathBuf;

use attic_eval::{HIDDEN_SIZE, LAYER_STACKS, NUM_FEATURES, load_network_with_warnings};

fn network_path() -> Option<PathBuf> {
    std::env::var_os("ATTIC_STANDARD_NN").map(PathBuf::from)
}

/// The whole point of the exercise: a file YaneuraOu wrote is read back with
/// nothing to complain about.
///
/// Warnings are what a wrong hash produces — the loader does not stop for one —
/// so asserting they are absent is the only way this test can fail usefully.
#[test]
fn a_real_network_loads_without_warnings() {
    let Some(path) = network_path() else {
        eprintln!("set ATTIC_STANDARD_NN to a KP256/HalfKP256 nn.bin to run this");
        return;
    };
    let (net, warnings) = load_network_with_warnings(&path).expect("the file loads");
    assert!(warnings.is_empty(), "hash mismatches: {warnings:?}");
    assert_eq!(net.stacks.len(), LAYER_STACKS, "one stack, no bucketing");
    assert_eq!(net.ft_biases.len(), HIDDEN_SIZE);
    assert_eq!(net.ft_weights.len(), HIDDEN_SIZE * NUM_FEATURES);
}

/// Every parameter array is 64-byte aligned, because the SIMD kernels load
/// from them without checking.
#[test]
fn every_array_is_aligned() {
    let Some(path) = network_path() else {
        return;
    };
    let (net, _) = load_network_with_warnings(&path).expect("the file loads");
    let aligned = |ptr: *const u8| ptr as usize % 64 == 0;
    assert!(aligned(net.ft_biases.as_ptr() as *const u8));
    assert!(aligned(net.ft_weights.as_ptr() as *const u8));
    for stack in &net.stacks {
        assert!(aligned(stack.fc_0_biases.as_ptr() as *const u8));
        assert!(aligned(stack.fc_0_weights.as_ptr() as *const u8));
        assert!(aligned(stack.fc_1_biases.as_ptr() as *const u8));
        assert!(aligned(stack.fc_1_weights.as_ptr() as *const u8));
        assert!(aligned(stack.fc_2_biases.as_ptr() as *const u8));
        assert!(aligned(stack.fc_2_weights.as_ptr() as *const u8));
    }
}
