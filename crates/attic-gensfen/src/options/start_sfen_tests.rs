//! What the parser makes of the flags that choose where a game starts and how
//! it is judged.

use super::*;

/// A file that deletes itself, so a failing assert does not leave the temp
/// directory holding fixtures.
struct TempFile(PathBuf);

impl TempFile {
    fn new(tag: &str, body: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("attic-gensfen-{}-{tag}.sfen", std::process::id()));
        std::fs::write(&path, body).expect("write fixture");
        Self(path)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn parse_with(extra: &[&str]) -> Result<Options, String> {
    let mut args = vec!["--eval-dir", "eval", "--out", "out"];
    args.extend_from_slice(extra);
    Options::parse(args.into_iter().map(String::from))
}

#[test]
fn the_repetition_rule_is_the_builds_unless_told_otherwise() {
    use attic_state::RepetitionRule;

    assert_eq!(
        parse_with(&[]).expect("parses").repetition_rule,
        RepetitionRule::DEFAULT,
        "a run that never says gets what the binary was built to do",
    );
    for (spelling, rule) in [
        ("on", RepetitionRule::QuickDraw),
        ("off", RepetitionRule::Precomputed),
    ] {
        let options = parse_with(&["--quick-draw", spelling]).expect("parses");
        assert_eq!(options.repetition_rule, rule, "--quick-draw {spelling}");
    }
}

#[test]
fn an_unknown_quick_draw_setting_is_refused_rather_than_defaulted() {
    for spelling in ["", "true", "ON", "0", "no-quick-draw"] {
        let err = parse_with(&["--quick-draw", spelling])
            .err()
            .unwrap_or_else(|| panic!("--quick-draw {spelling:?} was accepted"));
        assert!(err.contains("--quick-draw"), "{err}");
    }
}

#[test]
fn declaration_is_on_by_default_and_takes_the_engines_spellings() {
    use attic_search::EnteringKingRule;

    assert_eq!(
        parse_with(&[]).expect("parses").entering_king,
        EnteringKingRule::Point27,
        "a generator that cannot declare writes entered-king games out as \
         draws, so the rule has to be on unless it is turned off",
    );
    for (spelling, rule) in [
        ("CSARule24", EnteringKingRule::Point24),
        ("CSARule27", EnteringKingRule::Point27),
        ("TryRule", EnteringKingRule::Try),
        ("NoEnteringKing", EnteringKingRule::None),
    ] {
        let options = parse_with(&["--entering-king", spelling]).expect("parses");
        assert_eq!(options.entering_king, rule, "--entering-king {spelling}");
    }
}

#[test]
fn an_unknown_entering_king_rule_is_refused_rather_than_disabled() {
    // `EnteringKingRule::from_option` answers an unrecognised string with
    // `None`, which is safe behind a validated USI combo and silently turns
    // declaration off on a command line.
    let error = match parse_with(&["--entering-king", "27"]) {
        Err(error) => error,
        Ok(_) => panic!("`27` is not one of the engine's spellings"),
    };
    assert!(error.contains("CSARule27"), "the error lists the spellings");
}

#[test]
fn without_the_flag_there_are_no_start_positions() {
    let options = parse_with(&[]).expect("parses");
    assert!(
        options.start_sfens.is_empty(),
        "the default must stay the initial position"
    );
}

#[test]
fn the_sfen_prefix_and_crlf_are_stripped() {
    // Both are how the published opening sets are written, so a file copied
    // from there has to load unchanged.
    let file = TempFile::new(
        "prefixed",
        "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1\r\n\
         sfen 4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1\r\n",
    );
    let options = parse_with(&["--start-sfens", file.0.to_str().unwrap()]).expect("parses");
    assert_eq!(options.start_sfens.len(), 2);
    assert!(
        options.start_sfens.iter().all(|s| !s.starts_with("sfen ")),
        "stored lines must be sfen bodies"
    );
    assert!(
        options.start_sfens.iter().all(|s| !s.contains('\r')),
        "a stray CR would reach the parser"
    );
}

#[test]
fn blank_lines_are_skipped_rather_than_failing() {
    let file = TempFile::new(
        "blanks",
        "\nlnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1\n\n",
    );
    let options = parse_with(&["--start-sfens", file.0.to_str().unwrap()]).expect("parses");
    assert_eq!(options.start_sfens.len(), 1);
}

#[test]
fn a_file_with_no_positions_is_rejected() {
    let file = TempFile::new("empty", "\n\n");
    let err = parse_with(&["--start-sfens", file.0.to_str().unwrap()])
        .err()
        .expect("an empty list would silently play from the initial position");
    assert!(err.contains("no positions"), "unexpected error {err:?}");
}

#[test]
fn a_bad_line_names_its_line_number() {
    let file = TempFile::new(
        "bad",
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1\nnonsense\n",
    );
    let err = parse_with(&["--start-sfens", file.0.to_str().unwrap()])
        .err()
        .expect("must not parse");
    assert!(err.contains("line 2"), "unexpected error {err:?}");
}

#[test]
fn a_missing_file_is_reported_rather_than_ignored() {
    let err = parse_with(&["--start-sfens", "/nonexistent/attic/openings.sfen"])
        .err()
        .expect("must not parse");
    assert!(err.contains("cannot read"), "unexpected error {err:?}");
}

#[test]
fn the_dedup_filter_and_the_tree_are_off_unless_told_otherwise() {
    let options = parse_with(&[]).expect("parses");
    assert_eq!(
        options.dedup_mb, 0,
        "a run that never says plays what it always played"
    );
    assert_eq!(options.dedup_max_ply, 16);
    assert_eq!(options.move_margin, 0);
    assert_eq!(options.book_ply, 0, "a run that never says plays games");
    assert_eq!(options.book_width, 0);
    assert_eq!(options.book_band, 0);
    assert_eq!(
        options.random_move_count, 5,
        "the filter does not replace the random plies, and the help says so"
    );
}

#[test]
fn the_dedup_and_book_flags_take_their_values() {
    let options = parse_with(&[
        "--dedup-mb",
        "1",
        "--dedup-max-ply",
        "8",
        "--move-margin",
        "50",
        "--book-ply",
        "32",
        "--book-width",
        "500",
        "--book-band",
        "60",
    ])
    .expect("parses");
    assert_eq!(options.dedup_mb, 1);
    assert_eq!(options.dedup_max_ply, 8);
    assert_eq!(options.move_margin, 50);
    assert_eq!(options.book_ply, 32);
    assert_eq!(options.book_width, 500);
    assert_eq!(options.book_band, 60);
}

#[test]
fn every_dedup_and_book_flag_is_in_the_help() {
    // The test above the help's own (`every_flag_named_in_the_help_is_accepted`)
    // runs the other direction, so a flag the parser takes and the table
    // forgot would pass it.
    for flag in [
        "--dedup-mb",
        "--dedup-max-ply",
        "--move-margin",
        "--book-ply",
        "--book-width",
        "--book-band",
    ] {
        assert!(Options::HELP.contains(flag), "{flag} is not in the help");
    }
}

#[test]
fn a_recovery_run_refuses_the_dedup_filter() {
    // Left to run, a repeated board would play a different candidate into a
    // game whose kept prefix already carries the original ending -- the hazard
    // the --random-move-count refusal names, reached through another flag.
    let err = parse_with(&[
        "--recover-from",
        "/nonexistent/attic/shards",
        "--format",
        "pack",
        "--random-move-count",
        "0",
        "--dedup-mb",
        "64",
    ])
    .err()
    .expect("a non-zero --dedup-mb must not survive into a recovery run");
    assert!(
        err.contains("--dedup-mb 0"),
        "the error has to name the way out: {err:?}"
    );
}

#[test]
fn a_recovery_run_parses_with_the_filter_off_as_it_is_by_default() {
    let options = parse_with(&[
        "--recover-from",
        "/nonexistent/attic/shards",
        "--format",
        "pack",
        "--random-move-count",
        "0",
    ])
    .expect("parses");
    assert_eq!(options.dedup_mb, 0);
}
