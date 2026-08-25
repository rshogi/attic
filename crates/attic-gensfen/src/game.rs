//! Playing one game and recording what each position taught.

use std::sync::{Arc, Mutex};

use attic_search::{PvInfo, PvOutputConfig, PvSink, QSearch};
use attic_state::{Move, PackedSfen, Position, format_sfen, format_usi_move, sfen_pack};
use attic_storage::TranspositionTable;

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

/// Plays one game with an already-built searcher.
///
/// `qs` is deliberately a caller-owned parameter rather than built here.
/// `QSearch::new` allocates `WorkerHistories`, whose continuation table alone is
/// `(2*2*32*81) * (32*81)` `i16` — 51MB, zero-filled. Building one per move cost
/// ~5GB of allocation per game and made the search 28x slower than the reference
/// engine; the searcher is built once per worker and reused across games.
pub fn play_game(
    qs: &mut QSearch<'_>,
    collected: &Arc<Mutex<DeepestPvs>>,
    options: &Options,
    rng: &mut Rng,
) -> Game {
    let mut pos = Position::startpos();
    let mut records = Vec::new();
    // Plies played in this game, which is not `records.len()` — the eval-limit
    // filter drops records without shortening the game — and not `pos.ply()`
    // either, which counts from the opening's own move number.
    let mut ply_count = 0u16;
    let mut resign_streak = 0u32;
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

        let candidates: Vec<(Move, i32)> = lines
            .iter()
            .map(|info| (info.pv[0], i32::from(info.score)))
            .collect();

        let root_cp = outcome.score;

        // Resignation is judged from the side to move, so a streak only counts
        // when the same side keeps seeing a lost position.
        if root_cp <= -options.resign_cp {
            resign_streak += 1;
            if resign_streak >= options.resign_count {
                break (if side_to_move_is_black { -1 } else { 1 }, "resign");
            }
        } else {
            resign_streak = 0;
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

    fn options(count: u32, minply: u16, maxply: u16) -> Options {
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
        assert!(flags[15..32].iter().all(|&f| f), "the whole window is drawn");
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
