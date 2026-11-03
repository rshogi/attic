//! Playing one game and recording what each position taught.

use std::sync::{Arc, Mutex};

use attic_search::{EnteringKingConfig, PvInfo, PvSink, QSearch, RootKind};
use attic_state::{
    Move, PackedSfen, Position, RepetitionState, format_sfen, format_usi_move, parse_sfen,
    sfen_pack,
};

use crate::options::Options;

/// One position's contribution to the corpus.
pub struct Record {
    pub ply: u16,
    /// The 32-byte packed position, for the fixed-width output. Kept alongside
    /// the text form rather than derived later: packing needs the `Position`,
    /// which is gone by the time a game is serialised.
    pub packed: PackedSfen,
    pub sfen: String,
    /// `0` when Black is to move, `1` when White is — the value target is kept
    /// from Black's point of view, so the consumer needs to know whose turn it
    /// was to flip it.
    pub side: u8,
    /// Root score in centipawns, from the side to move.
    pub cp: i32,
    pub depth: i32,
    /// MultiPV candidates as `(move, centipawns)`, best first. Kept as `Move`
    /// rather than USI text so both encodings can render it without parsing.
    pub candidates: Vec<(Move, i32)>,
    /// The move actually played.
    pub played: Move,
}

pub struct Game {
    pub records: Vec<Record>,
    /// From Black's point of view: `1` Black won, `-1` White won, `0` draw.
    pub result: i32,
    pub reason: &'static str,
}

/// Collects PV lines, keeping only the deepest iteration reported.
///
/// The search emits every iterative-deepening iteration; only the last is the
/// answer. Keyed on depth rather than "the last N lines" because a fail-high
/// re-search can emit an extra line for a depth already reported.
#[derive(Default)]
pub struct DeepestPvs {
    depth: i32,
    lines: Vec<PvInfo>,
}

impl DeepestPvs {
    fn reset(&mut self) {
        self.depth = 0;
        self.lines.clear();
    }
}

pub struct Collector(pub Arc<Mutex<DeepestPvs>>);

impl PvSink for Collector {
    fn emit(&mut self, info: &PvInfo) {
        let mut slot = self.0.lock().expect("pv collector mutex poisoned");
        if info.depth > slot.depth {
            slot.depth = info.depth;
            slot.lines.clear();
        }
        if info.depth == slot.depth {
            slot.lines.push(info.clone());
        }
    }
}

/// A small xorshift, so the corpus does not need a rand dependency.
///
/// The quality bar here is "two workers must not replay each other's games",
/// which a seeded xorshift clears; nothing downstream is a statistical test of
/// the stream itself.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // A zero state is a fixed point for xorshift; nudge it off.
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
}

/// Which plies of the coming game get a uniformly-drawn legal move instead of
/// the search's answer, indexed by ply from the start of the game.
///
/// Upstream's generator draws the plies up front rather than rolling a die at
/// each one: a per-ply probability gives a game with none and a game with
/// twelve, and the point is that *every* game leaves the main line a fixed
/// number of times. `--random-move-minply` and `--random-move-maxply` are
/// 1-based, the way a move number is written, so ply 1 is the first move.
fn pick_random_plies(options: &Options, rng: &mut Rng) -> Vec<bool> {
    let mut flags = vec![false; options.max_ply as usize];
    if options.random_move_count == 0 {
        return flags;
    }
    let lo = usize::from(options.random_move_minply.max(1)) - 1;
    let hi = usize::from(options.random_move_maxply).min(flags.len());
    if lo >= hi {
        return flags;
    }

    // Partial Fisher-Yates over the window: pick without replacement, so the
    // count is exact rather than an average.
    let mut window: Vec<usize> = (lo..hi).collect();
    let take = (options.random_move_count as usize).min(window.len());
    for i in 0..take {
        let j = i + rng.below(window.len() - i);
        window.swap(i, j);
        flags[window[i]] = true;
    }
    flags
}

/// How long one side has been lost enough to resign.
///
/// The score is from the side to move and the side to move alternates, so the
/// losing side reads a losing score only on its own turns — on the other side's
/// turn the same position reads as won. Counting "the side to move is behind"
/// therefore breaks the run every other ply and never gets past one. What is
/// counted here is the *side that is behind*, which does not alternate.
#[derive(Default)]
struct Resignation {
    /// The side that is behind, `true` for Black, `None` when neither is.
    losing: Option<bool>,
    streak: u32,
}

impl Resignation {
    /// The side that should resign, once it has been behind for `resign_count`
    /// plies running. A `resign_cp` of zero or less turns the judgement off.
    fn update(&mut self, root_cp: i32, black_to_move: bool, options: &Options) -> Option<bool> {
        let behind = match options.resign_cp > 0 {
            false => None,
            true if root_cp <= -options.resign_cp => Some(black_to_move),
            true if root_cp >= options.resign_cp => Some(!black_to_move),
            _ => None,
        };
        match behind {
            Some(side) if self.losing == Some(side) => self.streak += 1,
            Some(side) => {
                self.losing = Some(side);
                self.streak = 1;
            }
            None => {
                self.losing = None;
                self.streak = 0;
            }
        }
        // `max(1)` so a count of zero resigns on the first losing ply rather
        // than on the position before it was ever judged.
        behind.filter(|_| self.streak >= options.resign_count.max(1))
    }
}

/// How a terminal position scores, as `(result, reason)`.
///
/// `result` is from Black throughout, matching [`Game::result`], which is why
/// every one of these needs to know whose turn it was.
type Ending = (i32, &'static str);

/// Score a win for the side to move.
fn stm_wins(black_to_move: bool) -> i32 {
    if black_to_move { 1 } else { -1 }
}

/// The result a repetition carries, or `None` when the position is not one.
///
/// The viewpoint is [`RepetitionState`]'s: `Lose` means the **side to move** is
/// the one that has been perpetually checking, so it is the side to move that
/// loses. Getting this backwards is silent — perpetual check is rare enough
/// that an inverted sign would not show up in an aggregate win rate.
fn repetition_ending(state: RepetitionState, black_to_move: bool) -> Option<Ending> {
    match state {
        RepetitionState::Draw => Some((0, "repetition")),
        RepetitionState::Lose => Some((-stm_wins(black_to_move), "perpetual_check")),
        RepetitionState::Win => Some((stm_wins(black_to_move), "perpetual_check")),
        // `Superior` / `Inferior` describe a repetition with *unequal* hands,
        // which is no repetition under the rules at all — they are search
        // heuristics and end no game. `game_repetition` never returns them, and
        // treating them as a draw here would end games that are still running.
        RepetitionState::None | RepetitionState::Superior | RepetitionState::Inferior => None,
    }
}

/// The result a root search that never searched carries, or `None` for an
/// ordinary one.
///
/// Both shortcuts return from `run_root` before a node is visited, so they emit
/// no `info` line and reach the PV collector as an empty list. Read the kind
/// rather than the list: taken as "no candidates", a declaration win would score
/// as a **mate against the declaring side** — not a missing rule but an
/// inverted result.
fn root_ending(kind: RootKind, black_to_move: bool) -> Option<Ending> {
    match kind {
        RootKind::DeclarationWin => Some((stm_wins(black_to_move), "declaration")),
        RootKind::Resign => Some((-stm_wins(black_to_move), "mated")),
        RootKind::Normal => None,
    }
}

/// Where a game begins when it does not begin from an opening.
///
/// The two fields travel together because they are one fact: the game is being
/// picked up part-way through, and both where it stopped and how far it had
/// already got are needed to continue it. Passing only the position is what let
/// a recovered game play a second full `--max-ply` budget.
pub struct Start {
    /// The position to play on from, its move number already restored.
    pub position: Position,
    /// Plies the original game had already spent against `--max-ply`.
    pub played: u16,
}

/// Plays one game with an already-built searcher.
///
/// `qs` is deliberately a caller-owned parameter rather than built here.
/// `QSearch::new` allocates `WorkerHistories`, whose continuation table alone is
/// `(2*2*32*81) * (32*81)` `i16` — 51MB, zero-filled. Building one per move cost
/// ~5GB of allocation per game and made the search 28x slower than the reference
/// engine; the searcher is built once per worker and reused across games.
/// `start` overrides where the game begins, which is what recovery plays from:
/// a position restored out of an existing corpus, its move number already set,
/// and the plies it took to get there. `None` draws an opening the usual way.
pub fn play_game(
    qs: &mut QSearch<'_>,
    collected: &Arc<Mutex<DeepestPvs>>,
    options: &Options,
    rng: &mut Rng,
    start: Option<Start>,
) -> Game {
    // Drawn here rather than in the worker because `Rng::below` is private to
    // this module, and the worker's `rng` is already threaded through.
    let (mut pos, already_played) = match start {
        Some(start) => (start.position, start.played),
        None => {
            let pos = match options.start_sfens.len() {
                0 => Position::startpos(),
                count => {
                    let sfen = &options.start_sfens[rng.below(count)];
                    // Every line was parsed when the file was read, so a failure
                    // here would mean the list was corrupted after startup.
                    parse_sfen(sfen).expect("start positions are validated at startup")
                }
            };
            (pos, 0)
        }
    };
    // Before the first move, which is the only time the rule can be chosen. A
    // position reaches here with no moves on it either way -- an opening drawn
    // from the list, the start position, or a record unpacked for recovery --
    // and the searches copy it, so the rule travels into every one of them.
    pos.set_repetition_rule(options.repetition_rule);
    // The entering-king thresholds are a snapshot of the root's material, so
    // they are taken once per game — and, for the `_H` rules, they have to be
    // taken from *this* position rather than left at the default.
    let entering_king = EnteringKingConfig::new(options.entering_king, &pos);
    qs.set_entering_king(entering_king);
    let mut records = Vec::new();
    // Plies played in this game, which is not `records.len()` — the eval-limit
    // filter drops records without shortening the game — and not `pos.ply()`
    // either, which counts from the opening's own move number.
    //
    // A recovered game starts this at what the original had already spent:
    // `--max-ply` bounds the game, not the continuation, and starting from zero
    // handed every recovered game a second full budget.
    let mut ply_count = already_played;
    let mut resignation = Resignation::default();
    let mut undos = Vec::new();
    let random_plies = pick_random_plies(options, rng);
    let mut legal: Vec<Move> = Vec::with_capacity(128);

    let (result, reason) = loop {
        if ply_count >= options.max_ply {
            break (0, "max_ply");
        }

        collected
            .lock()
            .expect("pv collector mutex poisoned")
            .reset();

        let outcome = qs.run_root(&pos, options.depth);
        let side_to_move_is_black = pos.side_to_move() == attic_state::Color::Black;

        if let Some(ending) = root_ending(outcome.kind, side_to_move_is_black) {
            break ending;
        }

        let mut lines = {
            let mut slot = collected.lock().expect("pv collector mutex poisoned");
            let mut lines = std::mem::take(&mut slot.lines);
            lines.sort_by_key(|info| info.multipv);
            lines.dedup_by_key(|info| info.multipv);
            lines
        };
        lines.retain(|info| !info.pv.is_empty());

        if lines.is_empty() {
            // No legal move: the side to move is mated.
            break (if side_to_move_is_black { -1 } else { 1 }, "mated");
        }

        let candidates: Vec<(Move, i32)> =
            lines.iter().map(|info| (info.pv[0], info.score)).collect();

        let root_cp = outcome.score;

        if let Some(black_resigns) = resignation.update(root_cp, side_to_move_is_black, options) {
            break (if black_resigns { -1 } else { 1 }, "resign");
        }

        // Spread the opening: always playing the best move from one start
        // position yields one game however many workers run. Every other ply
        // stays greedy so the result still reflects best play.
        //
        // The draw is over *all* legal moves, not the MultiPV candidates: a
        // position reached through the candidates is one the search already
        // liked, so the corpus stays inside the same handful of openings. The
        // record is kept either way — the candidates and the root score describe
        // the position, and they are true whatever gets played next — but
        // `played` is then not the search's answer, so a consumer training on
        // `played` alone wants the candidate list instead.
        let played = if random_plies.get(usize::from(ply_count)).copied() == Some(true) {
            legal.clear();
            pos.generate_legal_all(&mut legal);
            match legal.len() {
                0 => outcome.best_move,
                n => legal[rng.below(n)],
            }
        } else {
            outcome.best_move
        };

        // Past the limit the position is already decided, and a mate score is
        // `VALUE_MATE` rather than a centipawn value at all — neither belongs in
        // a corpus of evaluations. The game plays on so the result stays real;
        // only the record is dropped, which is why `ply` counts moves rather
        // than records.
        let over_limit = options.eval_limit > 0 && root_cp.abs() > options.eval_limit;

        if !over_limit {
            records.push(Record {
                // The position's own move number, so an opening that starts at
                // move 24 records 24 rather than 0. `sfen_pack` drops the ply,
                // making this the only place it survives.
                ply: pos.ply(),
                packed: sfen_pack(&pos),
                sfen: format_sfen(&pos),
                side: u8::from(!side_to_move_is_black),
                cp: root_cp,
                depth: outcome.depth,
                candidates,
                played,
            });
        }

        ply_count += 1;
        undos.push((played, pos.do_move(played)));

        // Judged on the position just reached, and with the *game* rule rather
        // than the search's: `Position::is_repetition` adjudicates a two-fold
        // under the default `quick-draw` build and looks back only sixteen
        // plies, both of which are right inside a tree and wrong for a result.
        let black_to_move = pos.side_to_move() == attic_state::Color::Black;
        if let Some(ending) = repetition_ending(pos.game_repetition(), black_to_move) {
            break ending;
        }
    };

    // Unwind so the position is reusable; `do_move` borrows structures the next
    // game would otherwise inherit mid-mutation.
    while let Some((mv, undo)) = undos.pop() {
        pos.undo_move(mv, undo);
    }

    Game {
        records,
        result,
        reason,
    }
}

#[cfg(test)]
mod random_ply_tests {
    use super::*;
    use crate::options::Format;

    pub(super) fn options(count: u32, minply: u16, maxply: u16) -> Options {
        Options {
            eval_dir: std::path::PathBuf::new(),
            out: std::path::PathBuf::new(),
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
            random_move_count: count,
            random_move_minply: minply,
            random_move_maxply: maxply,
            start_sfens: Vec::new(),
            recover_from: None,
            seed: 1,
            hash_mb: 16,
        }
    }

    #[test]
    fn the_window_is_the_move_number_the_flags_say() {
        // 1-based in, 0-based out: `--random-move-minply 16` means the 16th
        // move of the game, which is `ply_count` 15.
        let mut rng = Rng::new(3);
        let flags = pick_random_plies(&options(999, 16, 32), &mut rng);
        assert!(
            flags[15..32].iter().all(|&f| f),
            "the whole window is drawn"
        );
        assert!(!flags[14], "the ply before the window stays greedy");
        assert!(!flags[32], "the ply after the window stays greedy");
    }

    #[test]
    fn every_game_gets_exactly_the_count_asked_for() {
        // Drawing without replacement is the point: a per-ply coin would give
        // a game with none and a game with twelve.
        for seed in 0..32 {
            let mut rng = Rng::new(seed);
            let flags = pick_random_plies(&options(5, 16, 32), &mut rng);
            assert_eq!(flags.iter().filter(|&&f| f).count(), 5, "seed {seed}");
        }
    }

    #[test]
    fn zero_count_plays_the_search_throughout() {
        let mut rng = Rng::new(1);
        let flags = pick_random_plies(&options(0, 16, 32), &mut rng);
        assert!(flags.iter().all(|&f| !f));
    }

    #[test]
    fn a_window_past_max_ply_is_clamped_rather_than_panicking() {
        let mut rng = Rng::new(1);
        let mut opts = options(5, 16, 32);
        opts.max_ply = 20;
        let flags = pick_random_plies(&opts, &mut rng);
        assert_eq!(flags.len(), 20);
        assert_eq!(flags.iter().filter(|&&f| f).count(), 5);
    }
}

/// Bytes per `PackedSfenValue`: 32 packed + score + move + ply + result + pad.
pub const PSV_LEN: usize = 40;

/// Widest score the fixed-width encodings store, matching the engine's
/// `VALUE_MATE`.
///
/// Scores reach this as `VALUE_MATE - ply`, so a mate at ply 1 is 31999 and
/// nothing observed exceeds it; `VALUE_INFINITE` (32001) and `VALUE_NONE`
/// (32002) are search-internal sentinels that never leave the root. Clamping
/// rather than casting means a sentinel that did escape would land on the mate
/// score instead of wrapping into a small negative number.
const CP_LIMIT: i32 = 32000;

/// A search score as it is stored: clamped, then narrowed.
fn to_stored_cp(cp: i32) -> i16 {
    cp.clamp(-CP_LIMIT, CP_LIMIT) as i16
}

impl Game {
    /// The game as `PackedSfenValue` records, one per position.
    ///
    /// Field order and widths follow upstream's `PackedSfenValue`, little-endian
    /// throughout. The candidate list has no place in this layout and is
    /// dropped; a consumer that needs a policy target wants the JSONL form.
    ///
    /// `game_result` is stored from the side to move, which is what upstream's
    /// learner expects, so the game-level result is flipped on White's turns.
    pub fn to_psv(&self, out: &mut Vec<u8>) {
        out.reserve(self.records.len() * PSV_LEN);
        for record in &self.records {
            self.write_psv_prefix(record, out);
        }
    }

    /// Like [`to_psv`](Game::to_psv), with `candidates` MultiPV entries appended
    /// to each record as `(move16, centipawns)` pairs.
    ///
    /// Records stay fixed width: a position with fewer candidates than asked for
    /// is padded with zero moves, so a reader can seek by index. `candidates`
    /// comes from `--multipv` and is not stored in the file — every record in a
    /// run has the same width, and mixing runs means knowing which is which.
    pub fn to_pack(&self, out: &mut Vec<u8>, candidates: usize) {
        out.reserve(self.records.len() * (PSV_LEN + 4 * candidates));
        for record in &self.records {
            self.write_psv_prefix(record, out);
            for slot in 0..candidates {
                let (mv, cp) = match record.candidates.get(slot) {
                    Some(&(mv, cp)) => (mv.move16(), to_stored_cp(cp)),
                    None => (0, 0),
                };
                out.extend_from_slice(&mv.to_le_bytes());
                out.extend_from_slice(&cp.to_le_bytes());
            }
        }
    }

    /// The 40 bytes both fixed-width encodings begin with.
    ///
    /// `game_result` is stored from the side to move, which is what upstream's
    /// learner expects, so the game-level result is flipped on White's turns.
    fn write_psv_prefix(&self, record: &Record, out: &mut Vec<u8>) {
        let result = if record.side == 0 {
            self.result
        } else {
            -self.result
        };
        out.extend_from_slice(&record.packed);
        out.extend_from_slice(&to_stored_cp(record.cp).to_le_bytes());
        out.extend_from_slice(&record.played.move16().to_le_bytes());
        out.extend_from_slice(&record.ply.to_le_bytes());
        out.push(result as u8);
        out.push(0);
    }

    /// One JSONL line, matching the shape the existing corpus converter reads.
    pub fn to_json(&self, engine: &str, depth: i32, multipv: usize) -> String {
        let mut out = String::with_capacity(self.records.len() * 160);
        out.push_str("{\"engine\":\"");
        out.push_str(engine);
        out.push_str("\",\"depth\":");
        out.push_str(&depth.to_string());
        out.push_str(",\"multipv\":");
        out.push_str(&multipv.to_string());
        out.push_str(",\"result\":");
        out.push_str(&self.result.to_string());
        out.push_str(",\"reason\":\"");
        out.push_str(self.reason);
        out.push_str("\",\"ply_count\":");
        out.push_str(&self.records.len().to_string());
        out.push_str(",\"moves\":[");
        for (i, record) in self.records.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str("{\"ply\":");
            out.push_str(&record.ply.to_string());
            out.push_str(",\"sfen\":\"");
            out.push_str(&record.sfen);
            out.push_str("\",\"side\":");
            out.push_str(&record.side.to_string());
            out.push_str(",\"cp\":");
            out.push_str(&record.cp.to_string());
            out.push_str(",\"depth\":");
            out.push_str(&record.depth.to_string());
            out.push_str(",\"cand\":[");
            for (j, &(mv, cp)) in record.candidates.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                out.push_str("[\"");
                out.push_str(&format_usi_move(mv));
                out.push_str("\",");
                out.push_str(&cp.to_string());
                out.push(']');
            }
            out.push_str("],\"m\":\"");
            out.push_str(&format_usi_move(record.played));
            out.push_str("\"}");
        }
        out.push_str("]}");
        out
    }
}

#[cfg(test)]
mod resignation_tests {
    use super::*;

    /// `true` is Black to move, matching `Resignation::update`'s second argument
    /// and `side_to_move_is_black` at the call site.
    const BLACK: bool = true;
    const WHITE: bool = false;

    fn opts(resign_cp: i32, resign_count: u32) -> Options {
        Options {
            resign_cp,
            resign_count,
            ..super::random_ply_tests::options(0, 16, 32)
        }
    }

    #[test]
    fn a_streak_survives_the_side_to_move_alternating() {
        // The regression this guards: the score is from the side to move, so a
        // lost position reads `-3100` on the losing side's turn and `+3100` on
        // the winning side's. Counting "the side to move is behind" broke the
        // run every other ply and never got past one, which made the default
        // `resign_count` unreachable and left games playing on to mate.
        let options = opts(3000, 4);
        let mut resignation = Resignation::default();
        assert_eq!(resignation.update(-3100, BLACK, &options), None);
        assert_eq!(resignation.update(3100, WHITE, &options), None);
        assert_eq!(resignation.update(-3200, BLACK, &options), None);
        assert_eq!(
            resignation.update(3300, WHITE, &options),
            Some(BLACK),
            "four plies of Black being lost is four plies"
        );
    }

    #[test]
    fn coming_back_inside_the_bound_clears_the_count() {
        let options = opts(3000, 4);
        let mut resignation = Resignation::default();
        resignation.update(-3100, BLACK, &options);
        resignation.update(3100, WHITE, &options);
        assert_eq!(
            resignation.update(-100, BLACK, &options),
            None,
            "level again"
        );

        // Counting starts over, so the next three plies are not enough.
        assert_eq!(resignation.update(3100, WHITE, &options), None);
        assert_eq!(resignation.update(-3100, BLACK, &options), None);
        assert_eq!(resignation.update(3100, WHITE, &options), None);
        assert_eq!(resignation.update(-3100, BLACK, &options), Some(BLACK));
    }

    #[test]
    fn the_count_restarts_when_the_losing_side_changes() {
        let options = opts(3000, 2);
        let mut resignation = Resignation::default();
        assert_eq!(resignation.update(-3100, BLACK, &options), None);
        assert_eq!(
            resignation.update(-3100, WHITE, &options),
            None,
            "White is behind now; Black's count does not carry over"
        );
        assert_eq!(resignation.update(3100, BLACK, &options), Some(WHITE));
    }

    #[test]
    fn a_count_of_one_resigns_on_the_first_lost_ply() {
        let options = opts(3000, 1);
        assert_eq!(
            Resignation::default().update(-3000, BLACK, &options),
            Some(BLACK),
            "the bound itself is lost enough"
        );
    }

    #[test]
    fn a_score_inside_the_bound_never_resigns() {
        let options = opts(3000, 1);
        let mut resignation = Resignation::default();
        for ply in 0..64 {
            let score = if ply % 2 == 0 { -2999 } else { 2999 };
            assert_eq!(resignation.update(score, ply % 2 == 0, &options), None);
        }
    }

    #[test]
    fn a_resign_cp_of_zero_never_resigns() {
        // What generating endgame positions runs with: the game has to reach
        // mate for the mate scores to be recorded.
        let options = opts(0, 1);
        let mut resignation = Resignation::default();
        for ply in 0..64 {
            let mated = 31_999 - ply;
            assert_eq!(resignation.update(-mated, BLACK, &options), None);
        }
    }
}

// These tests play real games on a network of zeros, and the zero network is
// built with `NnueNetworkBuilder`, which the Manaka shapes (`sym_*`) do not have:
// their weights are f32 and arrive as `MNW1`, so there is no builder to reach for.
#[cfg(all(
    test,
    not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512"))
))]
mod resumed_budget_tests {
    use std::time::{Duration, Instant};

    use attic_eval::{NetHeader, NnueNetwork, NnueNetworkBuilder};
    use attic_search::PvOutputConfig;
    use attic_storage::TranspositionTable;

    use super::*;

    /// The budget most games here are given. Ten because a fourfold cannot
    /// happen inside it: returning to a position takes four plies, so a fourth
    /// occurrence needs twelve. That leaves `--max-ply` as the only way any of
    /// these games can end, which is what makes the ply counts assertable.
    const BUDGET: u16 = 10;

    /// A network of zeros. Every position evaluates to the same 0, so the games
    /// below are not good shogi — which is the point: what is under test is the
    /// ply budget, and a search with nothing to prefer still plays legal moves
    /// and still counts them.
    fn flat_net() -> NnueNetwork {
        let header = NetHeader {
            version: 0,
            hash: 0,
            arch_id: "synthetic".to_string(),
        };
        NnueNetworkBuilder::new(header, [0u8; 32]).build()
    }

    /// Resignation off, one ply of search, one PV line. Small on every axis:
    /// these tests play real games and only the *length* of one is the answer.
    pub(super) fn options(max_ply: u16) -> Options {
        let mut options = super::random_ply_tests::options(0, 1, 1);
        options.depth = 1;
        options.max_ply = max_ply;
        options.multipv = 1;
        options.resign_cp = 0;
        options
    }

    /// Play one game from `start`, returning the finished [`Game`].
    pub(super) fn play(start: Option<Start>, options: &Options) -> Game {
        let net = flat_net();
        let mut tt = TranspositionTable::new();
        tt.resize(1);
        let collected = Arc::new(Mutex::new(DeepestPvs::default()));
        let mut qs = QSearch::new(&net, &tt);
        qs.set_multi_pv(options.multipv);
        qs.set_pv_output(
            PvOutputConfig {
                multi_pv: options.multipv,
                pv_interval: Duration::ZERO,
                consideration_mode: false,
                output_fail_lh_pv: false,
                start_time: Instant::now(),
            },
            Box::new(Collector(Arc::clone(&collected))),
        );
        let mut rng = Rng::new(7);
        play_game(&mut qs, &collected, options, &mut rng, start)
    }

    /// Plies this game played. With `--eval-limit` off nothing is dropped, so
    /// the record count is the move count.
    fn plies(game: &Game) -> u16 {
        u16::try_from(game.records.len()).expect("these games are short")
    }

    #[test]
    fn a_game_resumed_at_its_limit_plays_no_further_ply() {
        // The sharp end of it: `ply_count` used to restart at zero on the
        // recovery pass, so a game that had already spent its whole `--max-ply`
        // was handed the budget again and played a second game on top of the
        // first. Here the budget is spent exactly, and the right number of
        // further plies is none at all.
        let options = options(BUDGET);
        let game = play(
            Some(Start {
                position: Position::startpos(),
                played: BUDGET,
            }),
            &options,
        );
        assert_eq!(plies(&game), 0, "the budget was already spent");
        assert_eq!(game.reason, "max_ply");
    }

    #[test]
    fn a_resumed_game_plays_only_the_plies_its_budget_has_left() {
        // Six spent of ten leaves four, and four is what it must play — not
        // ten.
        let options = options(BUDGET);
        let game = play(
            Some(Start {
                position: Position::startpos(),
                played: 6,
            }),
            &options,
        );
        assert_eq!(plies(&game), BUDGET - 6);
        assert_eq!(game.reason, "max_ply");
    }

    #[test]
    fn a_game_that_was_not_resumed_gets_the_whole_budget() {
        // The other side of the same arithmetic: nothing has been spent, so
        // nothing is deducted. This is what every recovered game used to get.
        let options = options(BUDGET);
        let game = play(None, &options);
        assert_eq!(plies(&game), BUDGET, "a fresh game plays its whole budget");
        assert_eq!(game.reason, "max_ply");
    }

    #[test]
    fn the_records_of_a_resumed_game_carry_the_resumed_move_numbers() {
        // `played` counts against the budget; `ply` is the position's own move
        // number, restored from the record the game resumed at. They are
        // different numbers and a corpus needs both — the budget so the game
        // stops, the move number so its records still sort into a game.
        let options = options(BUDGET);
        let mut position = Position::startpos();
        position.set_ply(137);
        let game = play(
            Some(Start {
                position,
                played: 7,
            }),
            &options,
        );

        let plies: Vec<u16> = game.records.iter().map(|record| record.ply).collect();
        assert_eq!(plies, vec![137, 138, 139], "7 spent of 10 leaves 3");
    }

    #[test]
    fn a_game_resumed_at_move_500_stops_at_move_512() {
        // The full-size shape of it, at the numbers a real corpus carries: a
        // game 500 plies deep, resumed under a 512-ply cap, must stop at 512.
        // Not at 1012, which zero-based counting gave it.
        //
        // The last position *recorded* is move 511 — that record is the one
        // whose move produces move 512, and the game ends there rather than
        // searching a position it has no budget to move from.
        let options = options(512);
        let mut position = Position::startpos();
        position.set_ply(500);
        let game = play(
            Some(Start {
                position,
                played: 500,
            }),
            &options,
        );

        assert_eq!(game.reason, "max_ply", "nothing else should end this game");
        assert_eq!(plies(&game), 12, "500 spent of 512 leaves 12");
        let last = game.records.last().expect("the game played something");
        assert_eq!(last.ply, 511, "the last record moves into 512");
        assert_eq!(
            game.records.first().expect("the game played something").ply,
            500,
            "and the first is the position it resumed at",
        );
    }
}

/// Playing games to the endings the rules give them, rather than checking the
/// judgements in isolation the way [`ending_tests`] does.
///
/// These share the zero-eval network of [`resumed_budget_tests`]: a search that
/// prefers nothing still has to be *stopped* by something, and what stops it is
/// exactly what is under test.
#[cfg(all(
    test,
    not(any(feature = "sym_kp_128x2_512", feature = "sym_halfkp_128x2_512"))
))]
mod played_ending_tests {
    use super::resumed_budget_tests::{options, play};
    use super::*;

    fn position(sfen: &str) -> Position {
        parse_sfen(sfen).expect("the sfen is valid")
    }

    /// Two bare kings. Neither side can mate, neither can declare — the kings
    /// are alone, so the eleven-piece condition can never hold — and with a
    /// flat eval the search has no reason to go anywhere in particular. What is
    /// left is shuffling, and shuffling is a fourfold.
    const BARE_KINGS: &str = "4k4/9/9/9/9/9/9/9/4K4 b - 1";

    /// Black to move, 30 points: eleven pieces in the enemy field (king, four
    /// golds, four silvers, two pawns) for 10 after the king is excluded, plus
    /// two rooks and two bishops in hand at five each.
    ///
    /// Thirty is the whole point of the position. It clears `CSARule27`'s 28
    /// and falls short of `CSARule24`'s 31, so the two rules disagree about it
    /// — which is the only way to tell that the flag selects a rule rather than
    /// just switching declaration on.
    const THIRTY_POINTS: &str = "KGGGG4/SSSS5/PP7/9/9/9/9/9/4k4 b 2R2B 1";

    /// The same position with one pawn more in hand: 31, which both rules
    /// allow.
    ///
    /// A pawn rather than the gold that first suggested itself. Black already
    /// holds all four golds on the board, and `sfen_pack` writes leftover
    /// pieces into a piece box whose size is fixed — a fifth gold makes the
    /// packed position 261 bits instead of 256. That is a `debug_assert`, so a
    /// release run would not panic on it; it would write the corrupt record.
    const THIRTY_ONE_POINTS: &str = "KGGGG4/SSSS5/PP7/9/9/9/9/9/4k4 b 2R2BP 1";

    #[test]
    fn every_fixture_here_packs_to_a_whole_record() {
        // Checked first because the failure mode is quiet: `sfen_pack`'s width
        // check is a `debug_assert`, so a fixture holding more pieces than a
        // set contains panics in a test build and silently writes a short
        // record in a release one.
        for sfen in [BARE_KINGS, THIRTY_POINTS, THIRTY_ONE_POINTS] {
            let position = position(sfen);
            assert_eq!(sfen_pack(&position).len(), 32, "{sfen}");
        }
    }

    #[test]
    fn two_bare_kings_shuffle_into_a_fourfold_draw() {
        // The rule that was missing entirely. Before it, this game ran to
        // `--max-ply` and was written as a draw for the wrong reason — and
        // every real game that repeated did the same, playing on past the
        // point the result was settled.
        let mut options = options(200);
        options.entering_king = attic_search::EnteringKingRule::Point27;
        let game = play(
            Some(Start {
                position: position(BARE_KINGS),
                played: 0,
            }),
            &options,
        );
        assert_eq!(game.reason, "repetition");
        assert_eq!(game.result, 0, "a fourfold is a draw for both sides");
        assert!(
            game.records.len() >= 12,
            "a fourth occurrence needs twelve plies, got {}",
            game.records.len(),
        );
        assert!(
            game.records.len() < 200,
            "and it must end well short of the cap, or the cap is what ended it",
        );
    }

    #[test]
    fn thirty_points_declares_under_the_27_rule() {
        let mut options = options(8);
        options.entering_king = attic_search::EnteringKingRule::Point27;
        let game = play(
            Some(Start {
                position: position(THIRTY_POINTS),
                played: 0,
            }),
            &options,
        );
        assert_eq!(game.reason, "declaration");
        assert_eq!(game.result, 1, "Black declared, so Black won");
        assert!(
            game.records.is_empty(),
            "the declaration is found before any move is searched",
        );
    }

    #[test]
    fn thirty_points_is_not_enough_under_the_24_rule() {
        // Same position, same side, one flag apart. `CSARule24` wants 31 from
        // both sides, so this one plays on — and a run that asked for the
        // 24-point law but silently got the 27-point one would end games here
        // that the tournament it is imitating would not.
        let mut options = options(8);
        options.entering_king = attic_search::EnteringKingRule::Point24;
        let game = play(
            Some(Start {
                position: position(THIRTY_POINTS),
                played: 0,
            }),
            &options,
        );
        assert_ne!(game.reason, "declaration", "30 is short of 31");
    }

    #[test]
    fn thirty_one_points_declares_under_both_rules() {
        // The other half of the discrimination: `CSARule24` is stricter, not
        // broken. One more point and it declares too, which is what separates
        // "the 24-rule is implemented" from "the 24-rule never fires".
        for rule in [
            attic_search::EnteringKingRule::Point24,
            attic_search::EnteringKingRule::Point27,
        ] {
            let mut options = options(8);
            options.entering_king = rule;
            let game = play(
                Some(Start {
                    position: position(THIRTY_ONE_POINTS),
                    played: 0,
                }),
                &options,
            );
            assert_eq!(game.reason, "declaration", "{rule:?}");
            assert_eq!(game.result, 1, "{rule:?}");
        }
    }

    #[test]
    fn the_none_rule_never_declares() {
        // `--entering-king NoEnteringKing` has to actually turn it off, since
        // that is the flag a run would use to reproduce the old corpus.
        let mut options = options(8);
        options.entering_king = attic_search::EnteringKingRule::None;
        let game = play(
            Some(Start {
                position: position(THIRTY_ONE_POINTS),
                played: 0,
            }),
            &options,
        );
        assert_ne!(game.reason, "declaration");
    }
}

#[cfg(test)]
mod ending_tests {
    use super::*;

    const BLACK: bool = true;
    const WHITE: bool = false;

    #[test]
    fn a_fourfold_ends_the_game_as_a_draw() {
        assert_eq!(
            repetition_ending(RepetitionState::Draw, BLACK),
            Some((0, "repetition")),
        );
        assert_eq!(
            repetition_ending(RepetitionState::Draw, WHITE),
            Some((0, "repetition")),
        );
    }

    #[test]
    fn perpetual_check_loses_for_the_side_that_was_checking() {
        // `RepetitionState::Lose` is stated from the side to move, so LOSE with
        // Black to move is a win for White: -1. Reading the state from the
        // *checked* side instead inverts every one of these, and perpetual
        // check is rare enough that the inversion would not move an aggregate
        // win rate off 50%.
        assert_eq!(
            repetition_ending(RepetitionState::Lose, BLACK),
            Some((-1, "perpetual_check")),
        );
        assert_eq!(
            repetition_ending(RepetitionState::Lose, WHITE),
            Some((1, "perpetual_check")),
        );
        assert_eq!(
            repetition_ending(RepetitionState::Win, BLACK),
            Some((1, "perpetual_check")),
        );
        assert_eq!(
            repetition_ending(RepetitionState::Win, WHITE),
            Some((-1, "perpetual_check")),
        );
    }

    #[test]
    fn an_unequal_hand_repetition_does_not_end_the_game() {
        // Superior/inferior are search heuristics for a repetition with unequal
        // hands, which is not a repetition under the rules. Ending a game on one
        // would cut games short and score them as draws.
        for state in [
            RepetitionState::None,
            RepetitionState::Superior,
            RepetitionState::Inferior,
        ] {
            assert_eq!(repetition_ending(state, BLACK), None, "{state:?}");
            assert_eq!(repetition_ending(state, WHITE), None, "{state:?}");
        }
    }

    #[test]
    fn a_declaration_wins_for_the_side_that_declared() {
        // The bug this replaced: `run_root` returns before searching, so the PV
        // collector is empty and the old loop read that as "no legal move" —
        // scoring an entering-king *win* as a mate against the declaring side.
        assert_eq!(
            root_ending(RootKind::DeclarationWin, BLACK),
            Some((1, "declaration")),
        );
        assert_eq!(
            root_ending(RootKind::DeclarationWin, WHITE),
            Some((-1, "declaration")),
        );
    }

    #[test]
    fn no_legal_move_loses_for_the_side_to_move() {
        assert_eq!(root_ending(RootKind::Resign, BLACK), Some((-1, "mated")));
        assert_eq!(root_ending(RootKind::Resign, WHITE), Some((1, "mated")));
    }

    #[test]
    fn an_ordinary_search_ends_nothing() {
        assert_eq!(root_ending(RootKind::Normal, BLACK), None);
        assert_eq!(root_ending(RootKind::Normal, WHITE), None);
    }
}
