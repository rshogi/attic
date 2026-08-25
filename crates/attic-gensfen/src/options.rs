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
    /// Resign once the score is worse than this for `resign_count` plies in a
    /// row. Without it a lost game plays on to `max_ply`, and those positions
    /// teach nothing that the resignation itself does not.
    pub resign_cp: i32,
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
    pub seed: u64,
    /// Hash size per worker, in MB. Small on purpose: hundreds of workers each
    /// hold one, and a fixed-depth search does not benefit from a large table
    /// the way a fixed-time one does.
    pub hash_mb: usize,
}

impl Options {
    pub const USAGE: &'static str = "\
usage: attic-gensfen --eval-dir DIR --out FILE
                     [--depth 9] [--nodes N] [--multipv 5] [--threads N]
                     [--games N] [--positions N]
                     [--max-ply 320] [--resign-cp 3000] [--resign-count 4]
                     [--fv-scale 28] [--format jsonl|psv|pack]
                     [--headroom 8] [--governor-secs 30] [--eval-limit N]
                     [--random-move-count 5] [--random-move-minply 16]
                     [--random-move-maxply 32]
                     [--seed 1] [--hash 16]";

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
        let mut resign_cp = 3000;
        let mut resign_count = 4;
        let mut headroom = 8usize;
        let mut governor_secs = 30u64;
        let mut eval_limit = 0i32;
        let mut fv_scale = 28;
        let mut format = Format::Jsonl;
        let mut random_move_count = 5u32;
        let mut random_move_minply = 16u16;
        let mut random_move_maxply = 32u16;
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
            seed,
            hash_mb,
        })
    }
}

fn parse_field<T: std::str::FromStr>(raw: &str, flag: &str) -> Result<T, String> {
    raw.parse()
        .map_err(|_| format!("{flag}: cannot parse {raw:?}"))
}
