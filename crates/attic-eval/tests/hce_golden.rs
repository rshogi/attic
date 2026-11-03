//! The hand-crafted feature vectors, against the C++ engine's own numbers.
//!
//! The 164 Mobility counts mean nothing except in shielune's registration
//! order: a vector in a different order is still 164 plausible small integers,
//! and multiplied by weights fitted for the right order it produces an
//! evaluation that is wrong everywhere and looks wrong nowhere. That is what
//! this test is for, and why it compares every element of every position
//! rather than the dot product they feed.
//!
//! `tests/data/hce-dump-v2.txt` carries the engine's side of it; its header
//! says where it came from.

#![cfg(feature = "hce")]

use attic_eval::hce::{NUM_MATERIAL, NUM_MOBILITY, extract_material, extract_mobility};
use attic_state::parse_sfen;

struct Case {
    sfen: String,
    mobility: Vec<i32>,
    material: Vec<i32>,
}

fn cases() -> Vec<Case> {
    let text = include_str!("data/hce-dump-v2.txt");
    let mut cases = Vec::new();
    let mut sfen: Option<String> = None;
    let mut mobility: Option<Vec<i32>> = None;

    for line in text.lines() {
        let numbers = |rest: &str| -> Vec<i32> {
            rest.split_whitespace()
                .map(|v| v.parse().expect("the dump holds integers"))
                .collect()
        };
        if let Some(rest) = line.strip_prefix("sfen ") {
            sfen = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("want ") {
            mobility = Some(numbers(rest));
        } else if let Some(rest) = line.strip_prefix("mat ") {
            cases.push(Case {
                sfen: sfen.take().expect("a `mat` line follows its `sfen`"),
                mobility: mobility.take().expect("a `mat` line follows its `want`"),
                material: numbers(rest),
            });
        }
    }
    cases
}

/// The dump is only worth reading if it is the whole one.
#[test]
fn the_fixture_is_three_hundred_positions() {
    let cases = cases();
    assert_eq!(cases.len(), 300);
    for case in &cases {
        assert_eq!(case.mobility.len(), NUM_MOBILITY, "{}", case.sfen);
        assert_eq!(case.material.len(), NUM_MATERIAL, "{}", case.sfen);
    }
}

/// Every one of the 164 counts, in every one of the 300 positions.
///
/// The mismatch is reported by feature index and SFEN rather than as "the
/// vectors differ", because which index is wrong is what says whether the
/// fault is a piece's attack pattern, the drop scan, or the order itself.
#[test]
fn mobility_matches_the_engine() {
    for case in cases() {
        let pos = parse_sfen(&case.sfen).expect("the dump holds parseable SFENs");
        let got = extract_mobility(&pos);
        for (index, want) in case.mobility.iter().enumerate() {
            assert_eq!(
                got[index] as i32, *want,
                "mobility feature {index} in {}",
                case.sfen
            );
        }
    }
}

/// The same for Material's 17.
#[test]
fn material_matches_the_engine() {
    for case in cases() {
        let pos = parse_sfen(&case.sfen).expect("the dump holds parseable SFENs");
        let got = extract_material(&pos);
        for (index, want) in case.material.iter().enumerate() {
            assert_eq!(
                got[index] as i32, *want,
                "material slot {index} in {}",
                case.sfen
            );
        }
    }
}
