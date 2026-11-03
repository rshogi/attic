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
    /// Which repetition rule the search plays by: the quick-draw one the
    /// tournament build of the reference uses (`--quick-draw on`), or the one it
    /// has without it (`--quick-draw off`).
    ///
    /// The two search different trees, so this decides what the teacher data is
    /// a record *of*: with it on, a position repeated once is already a draw
    /// and the search never looks past it. It is a property of the search and
    /// does not change how a *game* is scored, which always waits for the fourth
    /// occurrence. Defaults to what this binary was built to do
    /// ([`RepetitionRule::DEFAULT`], from the `quick-draw` feature), so a run
    /// that never says is the run it always was.
    pub repetition_rule: attic_state::RepetitionRule,
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
    ///
    /// A randomised ply records a `played` that is not the search's move and
    /// so sits outside the MultiPV list, and `manaka-teacher` refuses such a
    /// row: 47,101,578 of the V4 corpus's 1,419,944,833 rows, 3.32%, searched
    /// and paid for and then thrown away. `--dedup-mb` is the other source of
    /// divergence and does not have that cost, so a run that leaves the
    /// divergence to it states `--random-move-count 0`: the default here is 5
    /// and the filter does not replace it.
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
    /// Size, in MB per NUMA node, of the table of boards already searched. `0`,
    /// the default, turns the filter off and searches every position the games
    /// reach, which is what a run without this flag has always done: a corpus
    /// written before the flag existed and one written after it, with the flag
    /// left off, are the same bytes. 64 is the size the filter was measured at.
    ///
    /// A board that has been searched once teaches nothing the second time:
    /// `manaka-pack` folds records sharing a position into one row, so a
    /// retraced line is paid for and then averaged away. The board is still
    /// searched, but its record is dropped and the game plays one of the
    /// search's other MultiPV candidates: the game that got there first played
    /// the best move, so this one parts company and the plies after it are new.
    ///
    /// The divergence is not a uniformly-drawn legal move, which would be
    /// faster and much worse. From a book capped at ±300 centipawns, drawing
    /// uniformly left 29.7% of the following plies at |eval| ≥ 1000: most legal
    /// moves in the middlegame hang something, so the draw decided the game.
    ///
    /// The table is lossy on purpose; see [`crate::seen::Seen`].
    pub dedup_mb: usize,
    /// The filter only reads the table for this many plies of a game.
    ///
    /// Counted from the start of the game like `--max-ply` and
    /// `--random-move-minply/maxply`, not from the board's own ply, so the
    /// window means the same thing whichever book is loaded.
    ///
    /// Two games from one book position share a prefix and part company once,
    /// so the plies worth testing are the early ones. Past the bound the table
    /// would only write entries nothing comes back for, and those evict the
    /// early entries that do get hit.
    ///
    /// The default of 16 is measured: over 519 games from 64 ply-24 book
    /// positions, 99.7% of the 891 repeats fired inside the first sixteen plies
    /// of a game and 87.4% inside the first four.
    pub dedup_max_ply: u16,
    /// Play a candidate drawn uniformly from those the search scored within
    /// this many centipawns of its best, rather than the best move.
    ///
    /// This is what builds an opening book rather than a corpus. Every ply is
    /// a move the search itself called near-best, so a position the walk
    /// reaches is one best play could have reached too — unlike
    /// `--random-move-count`, which reaches a position by hanging something.
    /// The games branch without a random move anywhere in them, which is also
    /// why a ply drawn this way takes the place of a randomised one.
    ///
    /// It also bounds the divergence a repeat draws: without a margin that
    /// draw is over the whole candidate list, so a large `--multipv` lets it
    /// pick the twentieth-best move.
    ///
    /// `0` leaves every ply greedy, which is what generating a corpus wants:
    /// the recorded result is then best play's, not a walk's. In the tree it
    /// admits every candidate as a child.
    pub move_margin: i32,
    /// Build a book as a tree instead of playing games, expanding until the
    /// boards carry this move number.
    ///
    /// **A move number, not a game length.** `--max-ply`, `--dedup-max-ply`
    /// and the random-move window all count plies from the start of a game;
    /// this one is absolute, because a book is named by the move number its
    /// boards carry and `--start-sfens` roots already carry their own. A run
    /// with roots at move 24 and `--book-ply 32` expands eight times.
    ///
    /// `0` plays games, which is everything else in this file.
    ///
    /// The tree ignores what only a game has: `--games`, `--positions`,
    /// `--max-ply`, `--random-move-*`, `--dedup-*`, resignation and
    /// adjudication. It writes one `<out>.plyNN.txt` per level, `sfen `
    /// prefixed, so a level feeds straight back in as `--start-sfens`, and a
    /// `<out>.plyNN.cp.txt` beside it with each board's score and parent.
    pub book_ply: u16,
    /// Boards kept per level of the tree. `0` keeps all of them.
    ///
    /// A level is sampled by shuffling it whole and cutting, which is why the
    /// frontier waits for a level to be complete before handing any of it out.
    /// Six children a board means an uncapped tree multiplies by six a ply;
    /// this is the only thing that bounds it.
    pub book_width: usize,
    /// Drop a board of the tree this many centipawns or more from the score
    /// the book was rooted at, and do not expand it. `0` keeps every board.
    ///
    /// From the root's score, which is one number for the whole run, and not
    /// from the level the board is in. `--move-margin` is the one measured per
    /// node, and that is exactly why it cannot do this job: a node the search
    /// reads at -50 admits moves down to -100, and those admit -150. Nothing
    /// pulls back towards even, so a margin of 50 over eight plies spreads the
    /// ninth over 200 centipawns. Measured, without a band: median +80, but
    /// from -136 to +607.
    ///
    /// The anchor is the median of the root level's own scores, taken when
    /// that level closes — the median so that one wild starting position
    /// cannot carry the book with it. Scores are flipped into the root's
    /// frame before being compared, one flip per level, since a level is one
    /// side to move. The search reads the initial position at +94 for black,
    /// so a band read from zero rather than from the root would keep the
    /// boards that are abnormally level and drop the ordinary ones.
    ///
    /// How tight a band is meaningful is a question about `--threads`. A
    /// node's workers share one transposition table, so two sixteen-thread
    /// runs of the same tree agreed on 4 of 500 boards and differed by 32 on
    /// average — below about 64 there, the band cuts on which worker won a
    /// race rather than on the position. Single-threaded there is nobody to
    /// race: two runs to move 5 wrote all 340 boards identically, so any band
    /// is meaningful, at roughly an order of magnitude in wall clock.
    ///
    /// It prunes rather than filters: the subtree under a rejected board is
    /// never searched at all. That is what keeps a banded book equal the whole
    /// way down instead of only at its last ply, and it is the difference
    /// between generating a book and mining one out of a corpus after the
    /// fact.
    pub book_band: i32,
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
        let mut repetition_rule = attic_state::RepetitionRule::DEFAULT;
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
        let mut dedup_mb = 0usize;
        let mut dedup_max_ply = 16u16;
        let mut move_margin = 0i32;
        let mut book_ply = 0u16;
        let mut book_width = 0usize;
        let mut book_band = 0i32;

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
                "--quick-draw" => repetition_rule = parse_quick_draw(&value()?)?,
                "--resign-cp" => resign_cp = parse_field(&value()?, "--resign-cp")?,
                "--resign-count" => resign_count = parse_field(&value()?, "--resign-count")?,
                "--headroom" => headroom = parse_field(&value()?, "--headroom")?,
                "--governor-secs" => governor_secs = parse_field(&value()?, "--governor-secs")?,
                "--eval-limit" => eval_limit = parse_field(&value()?, "--eval-limit")?,
                "--fv-scale" => fv_scale = parse_field(&value()?, "--fv-scale")?,
                "--format" => format = parse_field(&value()?, "--format")?,
                "--random-move-count" => {
                    random_move_count = parse_field(&value()?, "--random-move-count")?;
                }
                "--random-move-minply" => {
                    random_move_minply = parse_field(&value()?, "--random-move-minply")?;
                }
                "--random-move-maxply" => {
                    random_move_maxply = parse_field(&value()?, "--random-move-maxply")?;
                }
                "--start-sfens" => start_sfens = Some(PathBuf::from(value()?)),
                "--recover-from" => recover_from = Some(PathBuf::from(value()?)),
                "--seed" => seed = parse_field(&value()?, "--seed")?,
                "--hash" => hash_mb = parse_field(&value()?, "--hash")?,
                "--dedup-mb" => dedup_mb = parse_field(&value()?, "--dedup-mb")?,
                "--dedup-max-ply" => {
                    dedup_max_ply = parse_field(&value()?, "--dedup-max-ply")?;
                }
                "--move-margin" => move_margin = parse_field(&value()?, "--move-margin")?,
                "--book-ply" => book_ply = parse_field(&value()?, "--book-ply")?,
                "--book-width" => book_width = parse_field(&value()?, "--book-width")?,
                "--book-band" => book_band = parse_field(&value()?, "--book-band")?,
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
            // Same hazard, reached the other way: a repeated board makes the
            // game play a different candidate, and here that move lands in a
            // game whose earlier records already carry the original ending.
            if dedup_mb > 0 {
                return Err(
                    "--recover-from resumes positions that are already decided; the dedup \
                     filter diverges a game with a different move, which rewrites a result the \
                     prefix was recorded with. Leave --dedup-mb off, or pass --dedup-mb 0"
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
                    "--recover-from and --out share the directory {}; the output is opened \
                     for append and would grow the shards being read",
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
            repetition_rule,
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
            dedup_mb,
            dedup_max_ply,
            move_margin,
            book_ply,
            book_width,
            book_band,
        })
    }

    /// A complete `Options` for tests to amend.
    ///
    /// Shared rather than written out in each test module so that a new flag
    /// is added in one place: every literal that named every field had to be
    /// touched to add one, and a module that was not touched failed to
    /// compile for a reason unrelated to what it tests.
    #[cfg(test)]
    pub(crate) fn for_tests() -> Self {
        Options {
            eval_dir: PathBuf::new(),
            out: PathBuf::new(),
            depth: 9,
            nodes: 0,
            multipv: 5,
            threads: 1,
            games: 0,
            positions: 0,
            max_ply: 320,
            entering_king: attic_search::EnteringKingRule::Point27,
            repetition_rule: attic_state::RepetitionRule::DEFAULT,
            resign_cp: 3000,
            resign_count: 4,
            headroom: 0,
            governor_secs: 0,
            eval_limit: 0,
            fv_scale: 28,
            format: Format::Pack,
            dedup_mb: 0,
            dedup_max_ply: 48,
            move_margin: 0,
            random_move_count: 0,
            random_move_minply: 16,
            random_move_maxply: 32,
            start_sfens: Vec::new(),
            recover_from: None,
            seed: 1,
            hash_mb: 16,
            book_ply: 0,
            book_width: 0,
            book_band: 0,
        }
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

/// `on` or `off`, and nothing else: a typo that fell back to a default would
/// generate a whole run under the rule the caller was trying to avoid.
fn parse_quick_draw(raw: &str) -> Result<attic_state::RepetitionRule, String> {
    use attic_state::RepetitionRule;

    match raw {
        "on" => Ok(RepetitionRule::QuickDraw),
        "off" => Ok(RepetitionRule::Precomputed),
        _ => Err(format!("unknown --quick-draw {raw:?} (on, off)")),
    }
}

fn parse_field<T: std::str::FromStr>(raw: &str, flag: &str) -> Result<T, String> {
    raw.parse()
        .map_err(|_| format!("{flag}: cannot parse {raw:?}"))
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

mod help;

#[cfg(test)]
mod help_tests;
#[cfg(test)]
mod start_sfen_tests;
