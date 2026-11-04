//! The file format: the architecture string, the lengths, and the two encodings.

use super::super::loader::{EMBED_SCALE, WEIGHT_SCALE, arch_hash, expected_arch, parse_network};
use super::super::{DIM, FEATURE_LEN, HEAD, HIDDEN, HIDDEN2, ORDER, SECOND_LAYER, TOKENS};
use super::fixtures::{EXPORTED, assert_close, encode, synthetic};

/// The string a build expects is the string Manaka's exporter writes for the
/// same shape, byte for byte, and hashes to what the exporter stored.
///
/// The exporter closes one parenthesis fewer than it opens. A reader that
/// "corrected" that would refuse every file, so this is what keeps the two
/// spellings from drifting apart.
#[test]
fn the_expected_architecture_is_what_the_exporter_writes() {
    let (text, hash) = EXPORTED;
    assert_eq!(expected_arch(true), text);
    assert_eq!(expected_arch(false), text.replacen("SYM;", "SYMf32;", 1));
    assert_eq!(arch_hash(text), hash);
}

/// The lengths follow from four numbers, and the second layer is absent from a
/// shape that has one layer rather than present and empty by accident.
#[test]
fn the_derived_lengths_agree_with_each_other() {
    assert_eq!(FEATURE_LEN, 2 * ORDER * DIM);
    assert_eq!(HEAD, if SECOND_LAYER { HIDDEN2 } else { HIDDEN });
    assert_eq!(TOKENS, crate::NUM_FEATURES);
    let net = synthetic();
    assert_eq!(net.w2.len(), HIDDEN2 * HIDDEN);
    assert_eq!(net.b2.len(), HIDDEN2);
    assert_eq!(net.value_w.len(), HEAD);
}

/// An f32 file is read back as the network that was written, exactly.
#[test]
fn an_f32_file_reads_back_exactly() {
    let net = synthetic();
    let read = parse_network(&encode(net, false)).expect("the file loads");
    assert_eq!(read.embed, net.embed);
    assert_eq!(read.gamma, net.gamma);
    assert_eq!(read.w1, net.w1);
    assert_eq!(read.b1, net.b1);
    assert_eq!(read.w2, net.w2);
    assert_eq!(read.b2, net.b2);
    assert_eq!(read.value_w, net.value_w);
    assert_eq!(read.value_b, net.value_b);
}

/// A quantised file is read back to within half a step of each array's own
/// scale, and `gamma`, which is never quantised, exactly.
#[test]
fn a_quantised_file_reads_back_within_half_a_step() {
    let net = synthetic();
    let read = parse_network(&encode(net, true)).expect("the file loads");
    let half_embed = 0.5 / EMBED_SCALE + 1e-6;
    let half_dense = 0.5 / WEIGHT_SCALE + 1e-6;
    assert_close("embed", &read.embed, &net.embed, half_embed);
    assert_eq!(read.gamma, net.gamma);
    assert_close("w1", &read.w1, &net.w1, half_dense);
    assert_close("b1", &read.b1, &net.b1, half_dense);
    assert_close("w2", &read.w2, &net.w2, half_dense);
    assert_close("b2", &read.b2, &net.b2, half_dense);
    assert_close("value_w", &read.value_w, &net.value_w, half_dense);
    assert!((read.value_b - net.value_b).abs() <= half_dense);
}

/// A file that is the wrong shape is refused at its header, and one that is the
/// wrong length is refused at its end: a reader that took either would be
/// reading at the wrong stride.
#[test]
fn a_file_of_the_wrong_shape_or_length_is_refused() {
    let bytes = encode(synthetic(), false);

    // One character of the architecture string, past the 12-byte header.
    let mut other = bytes.clone();
    other[12 + 14] ^= 0x01;
    let error = parse_network(&other).expect_err("another architecture string");
    assert!(error.to_string().contains("file says"), "{error}");

    assert!(parse_network(&bytes[..bytes.len() - 1]).is_err(), "short");
    let mut long = bytes;
    long.push(0);
    assert!(parse_network(&long).is_err(), "trailing byte");
}
