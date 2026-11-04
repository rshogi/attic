//! The file format: the architecture string, the lengths, and the three encodings.

use super::super::loader::{Encoding, arch_hash, expected_arch, parse_network};
use super::super::{
    DIM, FEATURE_LEN, HEAD, HIDDEN, HIDDEN2, ORDER, SECOND_LAYER, TOKENS, evaluate_value,
};
use super::fixtures::{EXPORTED, column_scales, encode, game, row_scales, synthetic};

/// The string a build expects is the string Manaka's exporter writes for the
/// same shape in each encoding, byte for byte, and hashes to what the exporter
/// stored.
///
/// The exporter closes one parenthesis fewer than it opens, and says so in its
/// own spec. A reader that "corrected" that would refuse every file, so this is
/// what keeps the two spellings from drifting apart.
#[test]
fn the_expected_architecture_is_what_the_exporter_writes() {
    let exported = EXPORTED;
    for (encoding, hash) in [
        (Encoding::Scaled, exported.scaled),
        (Encoding::F32, exported.f32),
        (Encoding::Legacy, exported.legacy),
    ] {
        let text = format!(
            "ModelType={};Features={}",
            encoding.model_type(),
            exported.rest
        );
        assert_eq!(expected_arch(encoding), text, "{encoding:?}");
        assert_eq!(arch_hash(&text), hash, "{encoding:?}");
    }
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
    let read = parse_network(&encode(net, Encoding::F32)).expect("the file loads");
    assert_eq!(read.embed, net.embed);
    assert_eq!(read.gamma, net.gamma);
    assert_eq!(read.w1, net.w1);
    assert_eq!(read.b1, net.b1);
    assert_eq!(read.w2, net.w2);
    assert_eq!(read.b2, net.b2);
    assert_eq!(read.value_w, net.value_w);
    assert_eq!(read.value_b, net.value_b);
}

/// Every element of a `SYMscaled` array is within half a step of its own group's
/// scale, and the arrays that stay `f32` are exact. The groups are the point:
/// the embedding's is a channel (`i mod d`), a dense weight's is its row
/// (`i div cols`), and a reader that swapped them would divide by a neighbour's
/// scale and be off by the ratio of the two.
#[test]
fn a_scaled_file_reads_back_within_half_a_step_of_its_own_group() {
    let net = synthetic();
    let read = parse_network(&encode(net, Encoding::Scaled)).expect("the file loads");

    let within = |what: &str, got: &[f32], want: &[f32], scale_of: &dyn Fn(usize) -> f32| {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let allowed = 0.5 / scale_of(i) + w.abs() * 2e-7;
            assert!(
                (g - w).abs() <= allowed,
                "{what}[{i}]: read {g} for {w}, half a step is {allowed}"
            );
        }
    };
    let embed_scales = column_scales(&net.embed, DIM);
    within("embed", &read.embed, &net.embed, &|i| embed_scales[i % DIM]);
    let w1_scales = row_scales(&net.w1, FEATURE_LEN);
    within("w1", &read.w1, &net.w1, &|i| w1_scales[i / FEATURE_LEN]);
    if SECOND_LAYER {
        let w2_scales = row_scales(&net.w2, HIDDEN);
        within("w2", &read.w2, &net.w2, &|i| w2_scales[i / HIDDEN]);
    }
    assert_eq!(read.gamma, net.gamma);
    assert_eq!(read.b1, net.b1);
    assert_eq!(read.b2, net.b2);
    assert_eq!(read.value_w, net.value_w);
    assert_eq!(read.value_b, net.value_b);
}

/// What `SYMscaled` is for. A trained net has weights far outside the legacy
/// encoding's ±1.98 (`W1` reaches 28, `value_w` 17 in the nets measured), and the
/// legacy writer clamps them. A scaled file holds the same weight to a step
/// of `1 / scale`, however large the group it sits in.
#[test]
fn a_scaled_file_holds_a_weight_the_legacy_encoding_clamps() {
    let mut net = synthetic().replicate();
    net.embed[3] = 7.5;
    net.w1[0] = 20.0;
    net.value_w[0] = 15.0;

    let scaled = parse_network(&encode(&net, Encoding::Scaled)).expect("the file loads");
    assert!(
        (scaled.w1[0] - 20.0).abs() < 1e-3,
        "w1[0] = {}",
        scaled.w1[0]
    );
    assert!(
        (scaled.embed[3] - 7.5).abs() < 1e-3,
        "embed[3] = {}",
        scaled.embed[3]
    );
    assert_eq!(scaled.value_w[0], 15.0);

    let legacy = parse_network(&encode(&net, Encoding::Legacy)).expect("the file loads");
    assert_eq!(
        legacy.w1[0],
        127.0 / 64.0,
        "the legacy writer clamps at 127 / 64"
    );
}

/// The three encodings of one net are one network: the f32 file evaluates to the
/// net exactly, and the scaled one to within what its steps cost.
#[test]
fn the_encodings_of_one_net_evaluate_alike() {
    let net = synthetic();
    let f32_net = parse_network(&encode(net, Encoding::F32)).expect("the file loads");
    let scaled = parse_network(&encode(net, Encoding::Scaled)).expect("the file loads");
    let mut worst = 0.0f32;
    for (pos, _) in game(30) {
        let want = evaluate_value(net, &pos);
        assert_eq!(evaluate_value(&f32_net, &pos), want);
        worst = worst.max((evaluate_value(&scaled, &pos) - want).abs());
    }
    assert!(
        worst < 1e-3,
        "scaled is {worst} from the net it was made from"
    );
}

/// A file that is the wrong shape is refused at its header, one with a damaged
/// section word at that section, one that is the wrong length at its end, and a
/// scale that is not a positive number before it divides anything: a reader
/// that took any of them would be reading at the wrong stride or handing the
/// search an infinity.
#[test]
fn a_damaged_file_is_refused_in_every_encoding() {
    for encoding in [Encoding::Scaled, Encoding::F32, Encoding::Legacy] {
        let bytes = encode(synthetic(), encoding);
        let arch_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let section_one = 12 + arch_len;

        // One character of the architecture string, past the 12-byte header.
        let mut other = bytes.clone();
        other[12 + 14] ^= 0x01;
        let error = parse_network(&other).expect_err("another architecture string");
        assert!(
            error.to_string().contains("file says"),
            "{encoding:?}: {error}"
        );

        let mut word = bytes.clone();
        word[section_one] ^= 0x01;
        let error = parse_network(&word).expect_err("a damaged section word");
        assert!(error.to_string().contains("hash"), "{encoding:?}: {error}");

        assert!(
            parse_network(&bytes[..bytes.len() - 1]).is_err(),
            "{encoding:?}: short"
        );
        let mut long = bytes.clone();
        long.push(0);
        assert!(parse_network(&long).is_err(), "{encoding:?}: trailing byte");

        if encoding == Encoding::Scaled {
            for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
                // The first scale of the embedding follows the section word.
                let mut scaled = bytes.clone();
                scaled[section_one + 4..section_one + 8].copy_from_slice(&bad.to_le_bytes());
                let error = parse_network(&scaled).expect_err("a scale that cannot divide");
                assert!(error.to_string().contains("scale"), "{bad}: {error}");
            }
        }
    }
}
