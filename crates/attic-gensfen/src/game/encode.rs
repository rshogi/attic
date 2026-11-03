//! The encodings a finished [`Game`] is written in.

use attic_state::format_usi_move;

use super::{Game, Record};

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
