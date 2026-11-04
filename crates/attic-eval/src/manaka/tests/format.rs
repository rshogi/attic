//! The file format: the architecture string, the lengths, and the three encodings.

use super::super::loader::{Encoding, arch_hash, expected_arch, parse_network};
use super::super::{
    BLOCK_LEN, DIM, FEATURE_LEN, GROUPS, HAS_PAIR, HEAD, HIDDEN, HIDDEN2, KS, LSE, MEAN, ORDER,
    PAIR, PAIR_ROWS, POWER_LEN, SCALE_LEN, SECOND_LAYER, TOKENS, evaluate_value,
};
use super::exported::EXPORTED;
use super::fixtures::{
    column_of, column_scales, encode, encoded, game, map_channels, row_scales, synthetic,
};

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
        (Encoding::Scaled, Some(exported.scaled)),
        (Encoding::F32, Some(exported.f32)),
        (Encoding::Legacy, exported.legacy),
    ] {
        // The exporter writes no such file, so there is no string to match; the
        // reader has to agree that there is none.
        let Some(hash) = hash else {
            assert!(!encoding.exists(), "{encoding:?} exists with no export");
            continue;
        };
        assert!(encoding.exists(), "{encoding:?} has an export");
        let text = format!(
            "ModelType={};Features={}",
            encoding.model_type(),
            exported.rest
        );
        assert_eq!(expected_arch(encoding), text, "{encoding:?}");
        assert_eq!(arch_hash(&text), hash, "{encoding:?}");
    }
}

/// The lengths follow from the shape's numbers, and a layer or table the shape
/// lacks is absent rather than present and empty by accident.
#[test]
fn the_derived_lengths_agree_with_each_other() {
    // The head reads both perspectives, each the power blocks, the pair channels
    // and the kept magnitudes, in that order and with nothing between.
    assert_eq!(POWER_LEN, GROUPS * ORDER * DIM);
    assert_eq!(BLOCK_LEN, POWER_LEN + PAIR + SCALE_LEN);
    assert_eq!(FEATURE_LEN, 2 * BLOCK_LEN);
    assert_eq!(PAIR, MEAN + LSE);
    assert_eq!(HEAD, if SECOND_LAYER { HIDDEN2 } else { HIDDEN });
    assert_eq!(TOKENS, crate::NUM_FEATURES);
    // One magnitude per normalised block: every order of every group, and the
    // pair's mean channels when there are any.
    let norms = GROUPS * ORDER + usize::from(MEAN != 0);
    assert_eq!(SCALE_LEN, if KS { norms } else { 0 });
    assert_eq!(PAIR_ROWS, if HAS_PAIR { 2_745_996 } else { 0 });
    let net = synthetic();
    assert_eq!(net.embed.len(), TOKENS * DIM);
    assert_eq!(net.w1.len(), HIDDEN * FEATURE_LEN);
    assert_eq!(net.w2.len(), HIDDEN2 * HIDDEN);
    assert_eq!(net.b2.len(), HIDDEN2);
    assert_eq!(net.value_w.len(), HEAD);
    assert_eq!(net.pair_gamma.len(), PAIR);
    assert_eq!(net.pair.len(), PAIR_ROWS * PAIR);
    assert_eq!(net.scale_gamma.len(), SCALE_LEN);
}

/// An f32 file is read back as the network that was written, exactly.
#[test]
fn an_f32_file_reads_back_exactly() {
    let net = synthetic();
    let read = parse_network(encoded(Encoding::F32)).expect("the file loads");
    assert_eq!(read.embed, net.embed);
    assert_eq!(read.gamma, net.gamma);
    assert_eq!(read.w1, net.w1);
    assert_eq!(read.b1, net.b1);
    assert_eq!(read.w2, net.w2);
    assert_eq!(read.b2, net.b2);
    assert_eq!(read.value_w, net.value_w);
    assert_eq!(read.value_b, net.value_b);
    assert_eq!(read.pair_gamma, net.pair_gamma);
    assert_eq!(read.pair, net.pair);
    assert_eq!(read.scale_gamma, net.scale_gamma);
}

/// Every element of a `SYMscaled` array is within half a step of its own group's
/// scale, and the arrays that stay `f32` are exact. The groups are the point:
/// the embedding's is a channel (`i mod d`), a dense weight's is its row
/// (`i div cols`), and a reader that swapped them would divide by a neighbour's
/// scale and be off by the ratio of the two.
#[test]
fn a_scaled_file_reads_back_within_half_a_step_of_its_own_group() {
    let net = synthetic();
    let read = parse_network(encoded(Encoding::Scaled)).expect("the file loads");

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
    if HAS_PAIR {
        // The pair table's groups are its channels, and its log-sum-exp channels
        // are stored as `θ` and exponentiated on the way in. A step of `1/scale`
        // in `θ` is a factor of `e^(1/scale)` in the table, so those are held to
        // a relative half-step of their own.
        let mut theta = net.pair.clone();
        map_channels(&mut theta, PAIR, MEAN, f32::ln);
        let pair_scales = column_scales(&theta, PAIR);
        for (i, (g, w)) in read.pair.iter().zip(&net.pair).enumerate() {
            let channel = column_of(PAIR)(i);
            let half_step = 0.5 / pair_scales[channel];
            // The mean channels are compared as they are and the log-sum-exp
            // ones by their logarithm; `LSE` of them end the row.
            let ok = if PAIR - channel > LSE {
                (g - w).abs() <= half_step + w.abs() * 2e-7
            } else {
                (g.ln() - w.ln()).abs() <= half_step + 1e-6
            };
            assert!(
                ok,
                "pair[{i}]: read {g} for {w}, half a step is {half_step}"
            );
        }
    }
    assert_eq!(read.gamma, net.gamma);
    assert_eq!(read.b1, net.b1);
    assert_eq!(read.b2, net.b2);
    assert_eq!(read.value_w, net.value_w);
    assert_eq!(read.value_b, net.value_b);
    assert_eq!(read.pair_gamma, net.pair_gamma);
    assert_eq!(read.scale_gamma, net.scale_gamma);
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

    // A shape with a pair table or kept magnitudes has no legacy file to clamp.
    if Encoding::Legacy.exists() {
        let legacy = parse_network(&encode(&net, Encoding::Legacy)).expect("the file loads");
        assert_eq!(
            legacy.w1[0],
            127.0 / 64.0,
            "the legacy writer clamps at 127 / 64"
        );
    }
}

/// The three encodings of one net are one network: the f32 file evaluates to the
/// net exactly, and the scaled one to within what its steps cost.
#[test]
fn the_encodings_of_one_net_evaluate_alike() {
    let net = synthetic();
    let f32_net = parse_network(encoded(Encoding::F32)).expect("the file loads");
    let scaled = parse_network(encoded(Encoding::Scaled)).expect("the file loads");
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
    for encoding in Encoding::ALL.into_iter().filter(|e| e.exists()) {
        // One copy, damaged and put right again: a pair table makes the file
        // large enough that a copy per case would be most of the test's time.
        let mut bytes = encoded(encoding).to_vec();
        let arch_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let section_one = 12 + arch_len;

        // One character of the architecture string, past the 12-byte header.
        bytes[12 + 14] ^= 0x01;
        let error = parse_network(&bytes).expect_err("another architecture string");
        bytes[12 + 14] ^= 0x01;
        assert!(
            error.to_string().contains("file says"),
            "{encoding:?}: {error}"
        );

        // The word that opens each section: the first, and each optional one a
        // shape has. Each is found by its value, so that this does not repeat
        // the layout it is checking.
        let mut words = vec![(section_one, "feature transformer")];
        if HAS_PAIR {
            words.push((pair_section(encoding, section_one), "pair table"));
        }
        if KS {
            words.push((bytes.len() - 4 - 4 * SCALE_LEN, "kept magnitudes"));
        }
        for (at, name) in words {
            bytes[at] ^= 0x01;
            let error = parse_network(&bytes).expect_err("a damaged section word");
            bytes[at] ^= 0x01;
            assert!(
                error.to_string().contains("hash"),
                "{encoding:?}, {name}: {error}"
            );
        }

        assert!(
            parse_network(&bytes[..bytes.len() - 1]).is_err(),
            "{encoding:?}: short"
        );
        bytes.push(0);
        assert!(
            parse_network(&bytes).is_err(),
            "{encoding:?}: trailing byte"
        );
        bytes.pop();

        if encoding == Encoding::Scaled {
            for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
                // The first scale of the embedding follows the section word.
                let at = section_one + 4;
                let good: [u8; 4] = bytes[at..at + 4].try_into().unwrap();
                bytes[at..at + 4].copy_from_slice(&bad.to_le_bytes());
                let error = parse_network(&bytes).expect_err("a scale that cannot divide");
                bytes[at..at + 4].copy_from_slice(&good);
                assert!(error.to_string().contains("scale"), "{bad}: {error}");
            }
        }
        assert!(parse_network(&bytes).is_ok(), "{encoding:?}: put right");
    }
}

/// Where the pair table's section word is, from the sizes of the two sections
/// before it, written out from the format's description (`docs/network.md`,
/// "Sections") and not from the reader. A scaled array is its scales and then
/// two bytes an element; an `f32` one is four bytes an element; gains, biases and
/// the value row are `f32` in both.
fn pair_section(encoding: Encoding, section_one: usize) -> usize {
    let (embed, w1, w2) = match encoding {
        Encoding::Scaled => (
            4 * DIM + 2 * TOKENS * DIM,
            4 * HIDDEN + 2 * HIDDEN * FEATURE_LEN,
            if SECOND_LAYER {
                4 * HIDDEN2 + 2 * HIDDEN2 * HIDDEN
            } else {
                0
            },
        ),
        _ => (
            4 * TOKENS * DIM,
            4 * HIDDEN * FEATURE_LEN,
            4 * HIDDEN2 * HIDDEN,
        ),
    };
    let dense = 4 * ORDER * DIM + w1 + 4 * HIDDEN + w2 + 4 * HIDDEN2 + 4 * HEAD + 4;
    section_one + 4 + embed + 4 + dense
}
