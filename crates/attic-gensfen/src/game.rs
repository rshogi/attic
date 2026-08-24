//! Playing one game and recording what each position taught.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
    /// `move16` of the played move, for the fixed-width output.
    pub played16: u16,
    pub sfen: String,
    /// `0` when Black is to move, `1` when White is — the value target is kept
    /// from Black's point of view, so the consumer needs to know whose turn it
    /// was to flip it.
    pub side: u8,
    /// Root score in centipawns, from the side to move.
    pub cp: i32,
    pub depth: i32,
    /// MultiPV candidates as `(usi move, centipawns)`, best first.
    pub candidates: Vec<(String, i32)>,
    /// The move actually played.
    pub played: String,
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
    let mut resign_streak = 0u32;
    let mut undos = Vec::new();

    let (result, reason) = loop {
        if records.len() as u16 >= options.max_ply {
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

        let candidates: Vec<(String, i32)> = lines
            .iter()
            .map(|info| (format_usi_move(info.pv[0]), i32::from(info.score)))
            .collect();

        let root_cp = i32::from(outcome.score);

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
        // position yields one game however many workers run. Later plies stay
        // greedy so the result still reflects best play.
        let played = if (records.len() as u16) < 30 && lines.len() > 1 {
            lines[rng.below(lines.len())].pv[0]
        } else {
            outcome.best_move
        };

        records.push(Record {
            ply: records.len() as u16,
            packed: sfen_pack(&pos),
            played16: played.move16(),
            sfen: format_sfen(&pos),
            side: u8::from(!side_to_move_is_black),
            cp: root_cp,
            depth: outcome.depth,
            candidates,
            played: format_usi_move(played),
        });

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

/// Bytes per `PackedSfenValue`: 32 packed + score + move + ply + result + pad.
pub const PSV_LEN: usize = 40;

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
            let result = if record.side == 0 {
                self.result
            } else {
                -self.result
            };
            out.extend_from_slice(&record.packed);
            out.extend_from_slice(&(record.cp as i16).to_le_bytes());
            out.extend_from_slice(&record.played16.to_le_bytes());
            out.extend_from_slice(&record.ply.to_le_bytes());
            out.push(result as u8);
            out.push(0);
        }
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
            for (j, (mv, cp)) in record.candidates.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                out.push_str("[\"");
                out.push_str(mv);
                out.push_str("\",");
                out.push_str(&cp.to_string());
                out.push(']');
            }
            out.push_str("],\"m\":\"");
            out.push_str(&record.played);
            out.push_str("\"}");
        }
        out.push_str("]}");
        out
    }
}
