//! The file format: the architecture string, the lengths, and how the numbers are
//! stored.

use super::super::loader::{arch_hash, expected_arch, parse_network};
use super::super::shape::SQUARES;
use super::super::{
    BLOCK_LEN, DIM, FEATURE_LEN, FIELD, FIELD_AT, FIELD_LEN, HAS_FIELD, HAS_MOMENT, HAS_PAIR, HEAD,
    HIDDEN, HIDDEN2, MOMENT, MOMENT_AT, MOMENT_LEN, ORDER, PAIR, PAIR_AT, PAIR_ROWS, POWER_LEN,
    TOKENS, evaluate_value,
};
use super::encode::{encode, encoded};
use super::exported::EXPORTED;
use super::fixtures::{game, synthetic};

/// The string a build expects is the string Manaka's exporter writes for the
/// same shape, byte for byte, and hashes to what the exporter stored.
///
/// The string leads with `Features=`: there is no word in front of it saying how
/// the numbers are stored, because they are stored one way. The exporter closes
/// one parenthesis fewer than it opens, and says so in its own spec. A reader that
/// "corrected" that would refuse every file, so this is what keeps the two
/// spellings from drifting apart.
#[test]
fn the_expected_architecture_is_what_the_exporter_writes() {
    let exported = EXPORTED;
    let text = format!("Features={}", exported.rest);
    assert_eq!(expected_arch(), text);
    assert_eq!(arch_hash(&text), exported.hash);
}

/// The file the synthetic net is written to, with its architecture string
/// replaced by `arch` and the length and hash that go with it, so that the only
/// thing wrong with it is the string.
fn with_architecture(arch: &str) -> Vec<u8> {
    let bytes = encoded();
    let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let mut out = Vec::new();
    out.extend_from_slice(&bytes[..4]);
    out.extend_from_slice(&arch_hash(arch).to_le_bytes());
    out.extend_from_slice(&(arch.len() as u32).to_le_bytes());
    out.extend_from_slice(arch.as_bytes());
    out.extend_from_slice(&bytes[12 + len..]);
    out
}

/// A string that says anything this build does not is refused at the header,
/// whatever follows it: a reader that guessed would read the body at the stride the
/// guess implied. The string before this format carried a `ModelType` word in front
/// of `Features=`, and those files are refused too, since their body is another
/// layout.
#[test]
fn a_string_that_is_not_this_shapes_is_refused() {
    let arch = expected_arch();
    assert!(parse_network(&with_architecture(&arch)).is_ok());
    assert!(arch.starts_with("Features="), "{arch}");
    for other in [
        format!("ModelType=SYNNscaled;{arch}"),
        format!("ModelType=SYNNf32;{arch}"),
        format!("ModelType=SYNN;{arch}"),
        arch.replacen("Features=", "features=", 1),
    ] {
        let error = parse_network(&with_architecture(&other)).expect_err(&other);
        assert!(error.to_string().contains("file says"), "{other}: {error}");
    }
}

/// The two words that make a string SYNN's are what the arithmetic is, so a file
/// whose blocks or hidden layers are named otherwise is refused, though every
/// section after the string is the one it would have read. The SYM spelling,
/// `RmsNormScaled` and `ReLU`, is the case that matters: it is the other family's
/// file, and its arithmetic is not this one's.
#[test]
fn a_file_naming_other_arithmetic_is_refused() {
    let arch = expected_arch();
    for (from, to) in [
        ("ClipScaled", "NormScaled"),
        ("ClippedReLU", "Activation"),
        ("ClipScaled", "RmsNormScaled"),
        ("ClippedReLU", "ReLU"),
    ] {
        let other = arch.replace(from, to);
        assert_ne!(other, arch);
        let error = parse_network(&with_architecture(&other)).expect_err(to);
        assert!(error.to_string().contains("file says"), "{to}: {error}");
    }
    // Both at once is a whole SYM string.
    let sym = arch
        .replace("ClipScaled", "RmsNormScaled")
        .replace("ClippedReLU", "ReLU");
    let error = parse_network(&with_architecture(&sym)).expect_err("a SYM string");
    assert!(error.to_string().contains("file says"), "{error}");
}

/// The lengths follow from the shape's numbers, and a layer or table the shape
/// lacks is absent rather than present and empty by accident.
#[test]
fn the_derived_lengths_agree_with_each_other() {
    // The head reads both perspectives, each the power blocks, the pair channels,
    // the second moment and the square field, in that order and with nothing
    // between.
    assert_eq!(POWER_LEN, ORDER * DIM);
    assert_eq!(BLOCK_LEN, POWER_LEN + PAIR + MOMENT_LEN + FIELD_LEN);
    assert_eq!(PAIR_AT, POWER_LEN);
    assert_eq!(MOMENT_AT, PAIR_AT + PAIR);
    assert_eq!(FIELD_AT, MOMENT_AT + MOMENT_LEN);
    assert_eq!(FEATURE_LEN, 2 * BLOCK_LEN);
    // The triangle of a symmetric `m × m` matrix, diagonal once, and 81 squares of
    // `f` channels.
    assert_eq!(
        MOMENT_LEN,
        if HAS_MOMENT {
            MOMENT * (MOMENT + 1) / 2
        } else {
            0
        }
    );
    assert_eq!(FIELD_LEN, SQUARES * FIELD);
    assert_eq!(SQUARES, 81);
    assert_eq!(HEAD, HIDDEN2);
    assert_eq!(TOKENS, crate::NUM_FEATURES);
    assert_eq!(PAIR_ROWS, if HAS_PAIR { 2_745_996 } else { 0 });
    let net = synthetic();
    assert_eq!(net.embed.len(), TOKENS * DIM);
    assert_eq!(net.w1.len(), HIDDEN * FEATURE_LEN);
    assert_eq!(net.w2.len(), HIDDEN2 * HIDDEN);
    assert_eq!(net.b2.len(), HIDDEN2);
    assert_eq!(net.value_w.len(), HEAD);
    assert_eq!(net.pair_gamma.len(), PAIR);
    assert_eq!(net.pair.len(), PAIR_ROWS * PAIR);
    assert_eq!(net.moment_gamma.len(), MOMENT_LEN);
    assert_eq!(net.moment_embed.len(), TOKENS * MOMENT);
    assert_eq!(net.field_gamma.len(), FIELD);
    assert_eq!(net.field.len(), TOKENS * FIELD_LEN);
}

/// What a half costs a number: the nearest half is at most half a step away, which
/// is `|x| · 2^−11` for a normal half and `2^−25` below `2^−14`, where every half is
/// a multiple of `2^−24`.
fn within_a_half_step(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let allowed = (w.abs() * 2f32.powi(-11)).max(2f32.powi(-25));
        assert!(
            (g - w).abs() <= allowed,
            "{what}[{i}]: read {g} for {w}, half a step is {allowed}"
        );
    }
}

/// Every array a half holds reads back within half a step of the number written,
/// and every array that is `f32` reads back exactly. The synthetic net is drawn in
/// `f32`, so this is what separates a reader that widens correctly from one that
/// is merely close.
#[test]
fn a_file_reads_back_within_half_a_step() {
    let net = synthetic();
    let read = parse_network(encoded()).expect("the file loads");

    within_a_half_step("embed", &read.embed, &net.embed);
    within_a_half_step("w1", &read.w1, &net.w1);
    within_a_half_step("w2", &read.w2, &net.w2);
    within_a_half_step("pair", &read.pair, &net.pair);
    within_a_half_step("moment_embed", &read.moment_embed, &net.moment_embed);
    within_a_half_step("field", &read.field, &net.field);

    assert_eq!(read.gamma, net.gamma);
    assert_eq!(read.b1, net.b1);
    assert_eq!(read.b2, net.b2);
    assert_eq!(read.value_w, net.value_w);
    assert_eq!(read.value_b, net.value_b);
    assert_eq!(read.pair_gamma, net.pair_gamma);
    assert_eq!(read.moment_gamma, net.moment_gamma);
    assert_eq!(read.field_gamma, net.field_gamma);
}

/// A net whose halved arrays are already halves is read back exactly, and written
/// again to the same bytes. The first half of that is the reader being exact (a
/// half is an `f32`), the second is the writer being the identity on what it has
/// already rounded: the two are one round trip with nothing left to round.
#[test]
fn a_net_of_halves_is_read_back_exactly() {
    let first = parse_network(encoded()).expect("the file loads");
    let bytes = encode(&first);
    assert_eq!(bytes, encoded(), "writing what was read changed the file");
    let second = parse_network(&bytes).expect("the file loads");
    assert_eq!(second.embed, first.embed);
    assert_eq!(second.w1, first.w1);
    assert_eq!(second.w2, first.w2);
    assert_eq!(*second.pair, *first.pair);
    assert_eq!(second.moment_embed, first.moment_embed);
    assert_eq!(second.field, first.field);
}

/// A trained net has weights far outside what a fixed scale holds (`W1` reaches 28,
/// `value_w` 17 in the nets measured), and a half holds up to 65,504 with a relative
/// step of `2^−11` wherever the weight sits. The test weights are a large one, a
/// small one that is subnormal as a half, and the largest a half holds.
#[test]
fn a_half_holds_a_large_weight_and_a_small_one() {
    let mut net = synthetic().replicate();
    net.embed[3] = 7.5;
    net.w1[0] = 1234.5678;
    net.w1[1] = 3e-6;
    net.w1[2] = 65504.0;
    net.w1[3] = -65504.0;
    net.value_w[0] = 15.0;

    let read = parse_network(&encode(&net)).expect("the file loads");
    assert_eq!(read.embed[3], 7.5);
    assert_eq!(read.w1[2], 65504.0);
    assert_eq!(read.w1[3], -65504.0);
    within_a_half_step("w1", &read.w1[..4], &net.w1[..4]);
    // `3e-6` is under `2^−14` and so a subnormal: a multiple of `2^−24`.
    assert!(read.w1[1] > 0.0 && (read.w1[1] / 2f32.powi(-24)).fract() == 0.0);
    assert_eq!(read.value_w[0], 15.0);
}

/// The file evaluates to what the net it was made from does, to within what the
/// halves cost. The bound is not a guess: a half is off by at most `2^−11` of the
/// weight, the first layer sums `FEATURE_LEN` of them against inputs in `[0, 1]`,
/// and the value is read after two clipped layers, so the sum of rounding errors
/// has to be well inside a thousandth of the unit.
#[test]
fn the_file_evaluates_like_the_net_it_was_made_from() {
    let net = synthetic();
    let read = parse_network(encoded()).expect("the file loads");
    let mut worst = 0.0f32;
    for (pos, _) in game(30) {
        let want = evaluate_value(net, &pos);
        worst = worst.max((evaluate_value(&read, &pos) - want).abs());
    }
    assert!(
        worst < 1e-3,
        "the file is {worst} from the net it was made from"
    );
}

/// A file that is the wrong shape is refused at its header, one with a damaged
/// section word at that section, one that is the wrong length at its end, and a
/// half that is an infinity or a NaN where it is read: a reader that took any of
/// them would be reading at the wrong stride or handing the search an infinity.
#[test]
fn a_damaged_file_is_refused() {
    // One copy, damaged and put right again: a pair table makes the file large
    // enough that a copy per case would be most of the test's time.
    let mut bytes = encoded().to_vec();
    let arch_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let section_one = 12 + arch_len;

    // One character of the architecture string, past the 12-byte header.
    bytes[12 + 14] ^= 0x01;
    let error = parse_network(&bytes).expect_err("another architecture string");
    bytes[12 + 14] ^= 0x01;
    assert!(error.to_string().contains("file says"), "{error}");

    // The word that opens each section: the first, and each optional one a shape
    // has. Each is found by its value, so that this does not repeat the layout it
    // is checking.
    let (optional, end) = optional_sections(section_one);
    assert_eq!(end, bytes.len(), "the sizes add up to the file");
    let mut words = vec![(section_one, "feature transformer")];
    words.extend(optional);
    for (at, name) in words {
        bytes[at] ^= 0x01;
        let error = parse_network(&bytes).expect_err("a damaged section word");
        bytes[at] ^= 0x01;
        assert!(error.to_string().contains("hash"), "{name}: {error}");
    }

    assert!(parse_network(&bytes[..bytes.len() - 1]).is_err(), "short");
    bytes.push(0);
    assert!(parse_network(&bytes).is_err(), "trailing byte");
    bytes.pop();

    // The first half of the embedding follows the section word. An exponent of 31 is
    // infinity (fraction 0) or a NaN (any other), and neither is a number a file
    // holds.
    for bad in [0x7c00u16, 0xfc00, 0x7e00, 0x7c01, 0xffff] {
        let at = section_one + 4;
        let good: [u8; 2] = bytes[at..at + 2].try_into().unwrap();
        bytes[at..at + 2].copy_from_slice(&bad.to_le_bytes());
        let error = parse_network(&bytes).expect_err("an infinity or a NaN");
        bytes[at..at + 2].copy_from_slice(&good);
        let said = error.to_string();
        assert!(
            said.contains("embed[0]") && said.contains("infinity or a NaN"),
            "{bad:#06x}: {said}"
        );
    }
    assert!(parse_network(&bytes).is_ok(), "put right");
}

/// Where the word of each optional section is, and where the file ends, from the
/// sizes of the sections before it, written out from the format's description
/// (`docs/network.md`, "Sections") and not from the reader. A half is two bytes and
/// nothing in front of the array; gains, biases and the value row are `f32`.
fn optional_sections(section_one: usize) -> (Vec<(usize, &'static str)>, usize) {
    let embed = 2 * TOKENS * DIM;
    let dense = 4 * ORDER * DIM
        + 2 * HIDDEN * FEATURE_LEN
        + 4 * HIDDEN
        + 2 * HIDDEN2 * HIDDEN
        + 4 * HIDDEN2
        + 4 * HEAD
        + 4;

    // Each section is its word, its gains and its table.
    let mut at = section_one + 4 + embed + 4 + dense;
    let mut words = Vec::new();
    let mut section = |present: bool, name: &'static str, size: usize| {
        if present {
            words.push((at, name));
            at += 4 + size;
        }
    };
    section(HAS_PAIR, "pair table", 4 * PAIR + 2 * PAIR_ROWS * PAIR);
    section(
        HAS_MOMENT,
        "second moment",
        4 * MOMENT_LEN + 2 * TOKENS * MOMENT,
    );
    section(
        HAS_FIELD,
        "square field",
        4 * FIELD + 2 * TOKENS * FIELD_LEN,
    );
    (words, at)
}
