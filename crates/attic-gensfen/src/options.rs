//! Command-line options.

use std::path::PathBuf;

/// Output encoding. The two consumers want different things: one wants the
/// candidate distribution a policy target is built from, the other wants the
/// fixed-width record upstream's own generator writes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// One game per line: every position with its MultiPV candidates, scores,
    /// root score, and played move.
    Jsonl,
    /// `PackedSfenValue`: 32-byte packed position, score, move, ply, result,
    /// pad — 40 bytes per position, no candidates.
    Psv,
    /// `PackedSfenValue` followed by the MultiPV candidates, fixed width.
    ///
    /// The first 40 bytes are byte-identical to [`Psv`](Format::Psv), so a
    /// reader that only wants the position and its score can ignore the rest.
    /// JSONL carries the same information but stores the position as text,
    /// which neither packs nor compresses as well.
    Pack,
}

impl Format {
    /// The spelling `--format` takes, for the startup log.
    pub fn name(self) -> &'static str {
        match self {
            Format::Jsonl => "jsonl",
            Format::Psv => "psv",
            Format::Pack => "pack",
        }
    }
}

impl std::str::FromStr for Format {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "jsonl" => Ok(Format::Jsonl),
            "psv" => Ok(Format::Psv),
            "pack" => Ok(Format::Pack),
            other => Err(format!("unknown format {other:?} (jsonl, psv, pack)")),
        }
    }
}

pub struct Options {
    pub eval_dir: PathBuf,
    pub out: PathBuf,
    pub depth: i32,
    /// Hard node ceiling per search (`go nodes N`). `0` leaves the search
    /// bounded by `depth` alone.
    ///
    /// Both may be set: whichever is reached first ends the search, matching
    /// `go depth D nodes N`.
    pub nodes: u64,
    pub multipv: usize,
    pub threads: usize,
    /// `0` runs until killed — what the cluster launcher wants.
    pub games: u64,
    /// Stop once this many *positions* have been recorded, rather than this
    /// many games. `0` disables it.
    ///
    /// A corpus is sized in positions, and games vary from a 30-ply resign to
    /// the 320-ply cap, so `--games` overshoots or undershoots by whatever the
    /// average game length turns out to be.
    pub positions: u64,
    pub max_ply: u16,
    /// Which entering-king (nyugyoku) law ends a game by declaration, spelled
    /// the way the engine's own `EnteringKingRule` option is
    /// (`CSARule27`, `CSARule24`, `TryRule`, `NoEnteringKing`, …).
    ///
    /// This is a *rule of the games being generated*, not a search setting: a
    /// king that walks into the enemy camp is never mated and never resigns, so
    /// without it those games run to `max_ply` and are written out as draws.
    /// Defaults to `CSARule27`, which is the engine's default and the law
    /// ordinary tournament play uses.
    pub entering_king: attic_search::EnteringKingRule,
    /// Resign once one side has been worse than this for `resign_count` plies
    /// in a row. Without it a lost game plays on to `max_ply` or to mate, and
    /// those positions teach nothing that the resignation itself does not.
    ///
    /// `0` or less turns resignation off, which is what generating endgame
    /// positions wants: the game then runs to mate and the mate scores are
    /// recorded, provided `eval_limit` is off as well.
    pub resign_cp: i32,
    /// Plies in a row the same side must be behind before it resigns. Counted
    /// over the side that is behind, not the side to move — see `Resignation`
    /// in [`crate::game`] for why that distinction is load-bearing.
    pub resign_count: u32,
    /// Cores to leave idle for whoever else is on the machine.
    ///
    /// `--threads` is a ceiling; concurrency is adjusted toward this headroom
    /// while running. See [`crate::governor`].
    pub headroom: usize,
    /// Seconds between concurrency adjustments. `0` fixes concurrency at
    /// `--threads`, which is what a dedicated machine or a benchmark wants.
    pub governor_secs: u64,
    /// Skip positions whose root score exceeds this, in centipawns. `0`
    /// disables the filter.
    ///
    /// A won position teaches little that the result does not already say, and
    /// a mate score is not on the same scale as an evaluation at all — the
    /// engine reports `VALUE_MATE` (32000), not a centipawn value. Upstream's
    /// generator calls the same knob `eval_limit`.
    pub eval_limit: i32,
    pub fv_scale: i32,
    pub format: Format,
    /// Plies per game replaced by a uniformly-drawn legal move. `0` plays the
    /// search's best move throughout.
    ///
    /// Without it a fixed start position yields one game however many workers
    /// run, and the corpus is one line of play seen from many angles. Drawing
    /// from *all* legal moves rather than from the MultiPV candidates is
    /// deliberate: the candidates are what the search already likes, so a
    /// position reached through them is one the engine would have considered
    /// anyway. Upstream calls this `random_move_count`.
    pub random_move_count: u32,
    /// First ply eligible to be randomised, counted from the start of the game.
    pub random_move_minply: u16,
    /// Last ply eligible to be randomised. Randomising late costs more than it
    /// buys — the position is already decided — so the window closes while the
    /// opening is still open.
    pub random_move_maxply: u16,
    /// Positions to start games from, one drawn per game. Empty plays from the
    /// initial position, which is the historical behaviour.
    ///
    /// `--random-move-count` only reaches plies inside its window, so with a
    /// fixed start every game replays the same opening up to
    /// `--random-move-minply` and those boards are recorded once per game. Over
    /// 6.5e8 positions that was 18% of the corpus, nearly all of it below ply
    /// 16. Starting past the window spends the search on positions that are
    /// distinct to begin with.
    ///
    /// Stored as the sfen body — no `sfen ` prefix — and parsed per game rather
    /// than kept as `Position`, so each game gets a position with no history
    /// behind it. One parse against a depth-9 search is nothing.
    pub start_sfens: Vec<String>,
    /// A directory of existing `.bin` shards to resume games out of, instead of
    /// starting new ones.
    ///
    /// Each game in there is replayed from the last position it still has a
    /// record for; see [`crate::recover`]. Set, this replaces the opening draw
    /// entirely — `--start-sfens` and `--random-move-count` have nothing to act
    /// on, and both are refused alongside it.
    pub recover_from: Option<PathBuf>,
    pub seed: u64,
    /// Hash size per worker, in MB. Small on purpose: hundreds of workers each
    /// hold one, and a fixed-depth search does not benefit from a large table
    /// the way a fixed-time one does.
    pub hash_mb: usize,
}

/// What the command line asked for.
///
/// `--help` and `--version` are answers rather than settings, and they are not
/// errors either: they belong on stdout with an exit status of 0, which is what
/// `command --help | less` and every packaging check expect.
pub enum Invocation {
    Run(Box<Options>),
    Help,
    Version,
}

/// The build this binary came from, stamped in by `build.rs`.
///
/// The workspace version tracks upstream and does not move for a fix, so it
/// cannot tell two builds apart on its own. Every `--version` and every run's
/// first log line carries this, so a corpus can be traced to the code that
/// wrote it.
pub const BUILD: &str = env!("ATTIC_GENSFEN_BUILD");

/// `attic-gensfen 9.70.0 (1fc9e08)`.
pub fn version_line() -> String {
    format!("attic-gensfen {} ({BUILD})", env!("CARGO_PKG_VERSION"))
}

/// Read the command line, answering `--help` and `--version` before anything
/// else so they work without `--eval-dir` and `--out`.
pub fn parse(args: impl Iterator<Item = String>) -> Result<Invocation, String> {
    let args: Vec<String> = args.collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Ok(Invocation::Help);
    }
    if args.iter().any(|arg| arg == "--version" || arg == "-V") {
        return Ok(Invocation::Version);
    }
    Options::parse(args.into_iter()).map(|options| Invocation::Run(Box::new(options)))
}

impl Options {
    /// The first line of `--help`, and what a bad flag is answered with.
    pub const USAGE: &'static str = "\
usage: attic-gensfen --eval-dir DIR --out FILE [options]
       attic-gensfen --help | --version";

    /// The full option table.
    ///
    /// Every flag is listed with the value it takes and the default it has,
    /// because the defaults are what bite: `--random-move-count` is 5, not 0,
    /// and a corpus meant to be all best moves comes out with five random
    /// plies a game if the flag is left off. Nothing here is a substitute for
    /// the notes at the end, which are the things that are not visible in the
    /// output when they go wrong.
    pub const HELP: &'static str = "\
Generate NNUE teacher data by self-play, driving the search in-process.

required:
  --eval-dir DIR         directory holding nn.bin
  --out FILE             where to write; opened for APPEND, never truncated

search:
  --depth N              search depth                              [9]
  --nodes N              node ceiling per search, 0 for none       [0]
  --multipv N            candidates recorded per position          [5]
  --fv-scale N           evaluation scale; NAGISA V4 wants 28      [28]
  --hash MB              transposition table PER WORKER            [16]

how much:
  --games N              stop after N games, 0 to run until killed [0]
  --positions N          stop after N positions instead, 0 for off [0]
  --threads N            worker threads, 0 for every system thread [0]
  --headroom N           threads left to the rest of the machine   [8]
  --governor-secs N      how often to re-read the load, 0 to pin   [30]

how a game ends:
  --max-ply N            hard cap on the length of a game          [320]
  --entering-king RULE   NoEnteringKing, CSARule24, CSARule24H,
                         CSARule27, CSARule27H, TryRule            [CSARule27]
  --resign-cp N          resign at this score, 0 to play on        [3000]
  --resign-count N       plies at that score before resigning      [1]

what gets written:
  --format F             jsonl, psv, or pack                       [jsonl]
  --eval-limit N         drop records past this score, 0 for none  [0]

where a game starts:
  --start-sfens FILE     draw openings from this file, one per line
  --random-move-count N  random plies played per game, 0 for none  [5]
  --random-move-minply N first ply that may be random              [16]
  --random-move-maxply N last ply that may be random               [32]
  --recover-from DIR     resume the games in this corpus instead of
                         drawing openings; refuses --start-sfens and
                         a non-zero --random-move-count
  --seed N               base for the random number generator      [1]

  -h, --help             this text
  -V, --version          version and the commit it was built from

notes:
  --random-move-count defaults to 5. A corpus of nothing but search best
  moves needs --random-move-count 0 stated; leaving the flag off does not
  give it. The ply window counts plies played in THIS game, so with an
  opening at move 37 the random plies land at moves 53 to 69, not 16 to 32.

  --out is opened for append. Re-running with the same --out and the same
  --seed writes the same games again, into the same file.

  --eval-limit drops records without shortening the game, so a corpus
  written with it has games whose tails are missing rather than games that
  ended early. --recover-from is what plays those tails.

  Repetition and the entering-king law end games; a game that reaches
  --max-ply is recorded as a draw.

  --hash is per worker, not per process and not per NUMA node. The workers
  on a NUMA node share one table sized --hash times their number, so 256
  threads over 8 NUMA nodes at the default hold 512 MB a node, 4 GB in all.";

    pub fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut eval_dir = None;
        let mut out = None;
        let mut depth = 9;
        let mut nodes = 0u64;
        let mut multipv = 5;
        let mut threads = 0usize;
        let mut games = 0u64;
        let mut positions = 0u64;
        let mut max_ply = 320u16;
        let mut entering_king = attic_search::EnteringKingRule::Point27;
        let mut resign_cp = 3000;
        let mut resign_count = 1;
        let mut headroom = 8usize;
        let mut governor_secs = 30u64;
        let mut eval_limit = 0i32;
        let mut fv_scale = 28;
        let mut format = Format::Jsonl;
        let mut random_move_count = 5u32;
        let mut random_move_minply = 16u16;
        let mut random_move_maxply = 32u16;
        let mut start_sfens: Option<PathBuf> = None;
        let mut recover_from: Option<PathBuf> = None;
        let mut seed = 1u64;
        let mut hash_mb = 16usize;

        let mut args = args.peekable();
        while let Some(flag) = args.next() {
            let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--eval-dir" => eval_dir = Some(PathBuf::from(value()?)),
                "--out" => out = Some(PathBuf::from(value()?)),
                "--depth" => depth = parse_field(&value()?, "--depth")?,
                "--nodes" => nodes = parse_field(&value()?, "--nodes")?,
                "--multipv" => multipv = parse_field(&value()?, "--multipv")?,
                "--threads" => threads = parse_field(&value()?, "--threads")?,
                "--games" => games = parse_field(&value()?, "--games")?,
                "--positions" => positions = parse_field(&value()?, "--positions")?,
                "--max-ply" => max_ply = parse_field(&value()?, "--max-ply")?,
                "--entering-king" => entering_king = parse_entering_king(&value()?)?,
                "--resign-cp" => resign_cp = parse_field(&value()?, "--resign-cp")?,
                "--resign-count" => resign_count = parse_field(&value()?, "--resign-count")?,
                "--headroom" => headroom = parse_field(&value()?, "--headroom")?,
                "--governor-secs" => governor_secs = parse_field(&value()?, "--governor-secs")?,
                "--eval-limit" => eval_limit = parse_field(&value()?, "--eval-limit")?,
                "--fv-scale" => fv_scale = parse_field(&value()?, "--fv-scale")?,
                "--format" => format = parse_field(&value()?, "--format")?,
                "--random-move-count" => {
                    random_move_count = parse_field(&value()?, "--random-move-count")?;
                },
                "--random-move-minply" => {
                    random_move_minply = parse_field(&value()?, "--random-move-minply")?;
                },
                "--random-move-maxply" => {
                    random_move_maxply = parse_field(&value()?, "--random-move-maxply")?;
                },
                "--start-sfens" => start_sfens = Some(PathBuf::from(value()?)),
                "--recover-from" => recover_from = Some(PathBuf::from(value()?)),
                "--seed" => seed = parse_field(&value()?, "--seed")?,
                "--hash" => hash_mb = parse_field(&value()?, "--hash")?,
                other => return Err(format!("unknown flag {other}")),
            }
        }

        // Defaulting to every core is wrong on a shared machine; the launcher
        // knows how many are actually free and passes that.
        if threads == 0 {
            threads = attic_numa::system_threads().max(1);
        }
        if multipv == 0 {
            return Err("--multipv must be at least 1".to_string());
        }
        if depth <= 0 && nodes == 0 {
            return Err("--depth or --nodes must bound the search".to_string());
        }
        if random_move_count > 0 && random_move_minply > random_move_maxply {
            return Err(format!(
                "--random-move-minply {random_move_minply} is past \
                 --random-move-maxply {random_move_maxply}"
            ));
        }
        if let Some(dir) = &recover_from {
            if start_sfens.is_some() {
                return Err(
                    "--recover-from replaces the opening draw; --start-sfens has nothing to act on"
                        .to_string(),
                );
            }
            if random_move_count > 0 {
                return Err(
                    "--recover-from resumes positions that are already decided; a random move \
                     there rewrites a result the prefix was recorded with. Pass \
                     --random-move-count 0"
                        .to_string(),
                );
            }
            if matches!(format, Format::Jsonl) {
                return Err(
                    "--recover-from carries the kept records through as bytes, which only \
                     --format psv or pack can hold"
                        .to_string(),
                );
            }
            // The output is opened for append, so pointing it at the input
            // directory grows the very shards being read — and the reader would
            // then walk into records this run wrote.
            let out_dir = out
                .as_ref()
                .and_then(|path| path.parent())
                .map(|parent| parent.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."));
            if let (Ok(input), Ok(output)) = (dir.canonicalize(), out_dir.canonicalize())
                && input == output
            {
                return Err(format!(
                    "--recover-from and --out share the directory {}; the output is opened for \
                     append and would grow the shards being read",
                    input.display()
                ));
            }
        }

        let start_sfens = match start_sfens {
            Some(path) => load_start_sfens(&path)?,
            None => Vec::new(),
        };

        Ok(Options {
            eval_dir: eval_dir.ok_or("--eval-dir is required")?,
            out: out.ok_or("--out is required")?,
            depth,
            nodes,
            multipv,
            threads,
            games,
            positions,
            max_ply,
            entering_king,
            resign_cp,
            resign_count,
            headroom,
            governor_secs,
            eval_limit,
            fv_scale,
            format,
            random_move_count,
            random_move_minply,
            random_move_maxply,
            start_sfens,
            recover_from,
            seed,
            hash_mb,
        })
    }
}

/// The spelling `--entering-king` would have taken for this rule.
///
/// The inverse of [`parse_entering_king`], for the startup log. `Debug` names
/// the variant (`Point27`), which is not the name the caller typed
/// (`CSARule27`) and would send a reader looking for a flag value that does not
/// exist.
pub fn entering_king_name(rule: attic_search::EnteringKingRule) -> &'static str {
    use attic_search::EnteringKingRule;

    match rule {
        EnteringKingRule::None => "NoEnteringKing",
        EnteringKingRule::Point24 => "CSARule24",
        EnteringKingRule::Point24H => "CSARule24H",
        EnteringKingRule::Point27 => "CSARule27",
        EnteringKingRule::Point27H => "CSARule27H",
        EnteringKingRule::Try => "TryRule",
    }
}

/// `--entering-king`, taking the engine's own option spellings.
///
/// Deliberately not [`EnteringKingRule::from_option`], which answers an
/// unrecognised string with [`EnteringKingRule::None`] because the USI option
/// layer has already validated it against the declared `var` list. Nothing has
/// validated a command line, and there the same fallback would silently turn
/// declaration off for the whole run and write the resulting games out as
/// draws.
fn parse_entering_king(raw: &str) -> Result<attic_search::EnteringKingRule, String> {
    use attic_search::EnteringKingRule;

    EnteringKingRule::STRINGS
        .contains(&raw)
        .then(|| EnteringKingRule::from_option(raw))
        .ok_or_else(|| {
            format!(
                "unknown --entering-king {raw:?} ({})",
                EnteringKingRule::STRINGS.join(", ")
            )
        })
}

fn parse_field<T: std::str::FromStr>(raw: &str, flag: &str) -> Result<T, String> {
    raw.parse()
        .map_err(|_| format!("{flag}: cannot parse {raw:?}"))
}

#[cfg(test)]
mod help_tests {
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
}

#[cfg(test)]
mod start_sfen_tests {
    use super::*;

    /// A file that deletes itself, so a failing assert does not leave the temp
    /// directory holding fixtures.
    struct TempFile(PathBuf);

    impl TempFile {
        fn new(tag: &str, body: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("attic-gensfen-{}-{tag}.sfen", std::process::id()));
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
}

/// Read a starting-position file: one sfen per line, blank lines skipped.
///
/// The published opening sets carry a `sfen ` prefix and CRLF line endings, so
/// both are accepted and stripped here — the stored strings are sfen bodies,
/// which is what `parse_sfen` wants. Every line is parsed now rather than at
/// first use: a typo in a file of millions should stop the run before a
/// hundred workers have been started, not one game into whichever shard drew it.
fn load_start_sfens(path: &std::path::Path) -> Result<Vec<String>, String> {
    let display = path.display();
    let text = std::fs::read_to_string(path)
        .map_err(|err| format!("--start-sfens: cannot read {display}: {err}"))?;

    let mut out = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let body = line.strip_prefix("sfen ").unwrap_or(line);
        attic_state::parse_sfen(body)
            .map_err(|err| format!("--start-sfens: {display} line {}: {err}", index + 1))?;
        out.push(body.to_string());
    }

    if out.is_empty() {
        return Err(format!("--start-sfens: no positions in {display}"));
    }
    Ok(out)
}
