//! `--help`, `--version`, and the option table they print.

use super::*;

fn parse_of(args: &[&str]) -> Result<Invocation, String> {
    parse(args.iter().map(|arg| arg.to_string()))
}

/// Both answers have to work on their own. Requiring `--eval-dir` and
/// `--out` first would mean the only way to read the option list is to
/// already know two of the options.
#[test]
fn help_and_version_need_no_other_flags() {
    assert!(matches!(parse_of(&["--help"]), Ok(Invocation::Help)));
    assert!(matches!(parse_of(&["-h"]), Ok(Invocation::Help)));
    assert!(matches!(parse_of(&["--version"]), Ok(Invocation::Version)));
    assert!(matches!(parse_of(&["-V"]), Ok(Invocation::Version)));
    // And they win over a command line that is otherwise unusable, so a
    // half-remembered invocation can be fixed by appending `--help`.
    assert!(matches!(
        parse_of(&["--depth", "9", "--help"]),
        Ok(Invocation::Help)
    ));
}

/// Every flag the help names has to be a flag the parser takes. The two
/// drift apart in the direction that matters most — the help is what a
/// caller reads, so a flag misspelled here is a command line that fails
/// for a reason the reader cannot see.
#[test]
fn every_flag_named_in_the_help_is_accepted() {
    let flags: Vec<&str> = Options::HELP
        .split_whitespace()
        .filter(|word| word.starts_with("--") && word.len() > 2)
        // The notes are prose about the same flags, and a sentence can end
        // in one; strip the punctuation rather than skipping the section.
        .map(|word| word.trim_end_matches(['.', ',', ';', ':']))
        // These two are answers, not settings: they parse fine and take no
        // value, so the "needs a value" check below does not apply.
        .filter(|word| !matches!(*word, "--help" | "--version"))
        .collect();
    assert!(flags.len() > 15, "the help lists {} flags", flags.len());

    for flag in flags {
        let error = parse_of(&[flag])
            .err()
            .unwrap_or_else(|| panic!("{flag} parsed with no value"));
        assert!(
            !error.starts_with("unknown flag"),
            "the help names {flag}, which the parser rejects: {error}"
        );
    }
}

/// The build stamp is what tells two binaries apart, so an empty one is a
/// silently useless `--version`.
#[test]
fn the_version_line_carries_a_build() {
    let line = version_line();
    assert!(line.starts_with("attic-gensfen "), "{line}");
    assert!(!BUILD.is_empty(), "the build stamp is empty");
    assert!(line.contains(BUILD), "{line}");
}

/// `--entering-king` has to round-trip: the log prints what the caller
/// would have typed, not the variant's Rust name.
#[test]
fn every_rule_name_parses_back_to_its_rule() {
    for spelling in attic_search::EnteringKingRule::STRINGS {
        let rule = parse_entering_king(spelling).expect("a declared spelling parses");
        assert_eq!(entering_king_name(rule), spelling);
    }
}
