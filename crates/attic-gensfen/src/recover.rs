//! Reading an existing corpus back to find where each game's records stop.
//!
//! A run with `--eval-limit N` writes only the positions whose root score stays
//! inside ±N and plays the rest of the game anyway, so a shard holds games with
//! their tails missing and, where a score crossed the bound and came back,
//! holes in the middle. Recovery takes the last position each game still has a
//! record for and plays on from there.
//!
//! The continuation cannot be the original one — workers share a transposition
//! table and never clear it between games, so the same seed does not reproduce
//! the same moves. What is kept is the prefix of records up to the break; what
//! follows is a fresh game from that position.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use attic_state::{PackedSfen, Position, sfen_pack, sfen_unpack};

/// The fixed prefix every record starts with, candidates aside.
const PSV_LEN: usize = 40;
const PACKED_LEN: usize = 32;
const PLY_OFF: usize = 36;
const RESULT_OFF: usize = 38;

/// Bytes one record occupies. `--multipv` is not stored in the file, so it has
/// to be the same value the run was written with; a wrong one reads
/// plausible-looking records at a shifted offset and nothing detects it.
pub fn record_len(multipv: usize) -> usize {
    PSV_LEN + 4 * multipv
}

/// Side to move, as the packed position's first bit holds it: Black `0`,
/// White `1`. Bits are written least-significant-first, so it is bit 0 of
/// byte 0.
///
/// Read here rather than unpacked because rewriting a prefix's results needs
/// the side of every record, and unpacking each one to ask costs far more than
/// the answer is worth.
fn packed_is_white(record: &[u8]) -> bool {
    record[0] & 1 == 1
}

fn record_ply(record: &[u8]) -> u16 {
    u16::from_le_bytes([record[PLY_OFF], record[PLY_OFF + 1]])
}

/// One game to play on from.
pub struct Resume {
    /// The records kept from the original game, byte for byte. Their result
    /// bytes are stale until [`rewrite_results`] is applied.
    pub prefix: Vec<u8>,
    /// The position the last kept record holds, with its move number restored.
    pub position: Position,
    /// Plies the original game had already played when it reached `position`.
    ///
    /// `--max-ply` bounds the game, not the continuation, so the new run has to
    /// start its count here rather than at zero. Records up to the cut are one
    /// per ply — that is what `cut_at` looks for — so the prefix's length is
    /// the count.
    pub played: u16,
}

/// Stamp `result` into every record of a prefix, from each one's side to move.
///
/// The continuation is a new game and may end the other way round, so the
/// results the prefix was written with cannot be kept. `result` is from Black,
/// matching `Game::result`, and is negated on White's turns the same way
/// `Game::write_psv_prefix` does it.
pub fn rewrite_results(prefix: &mut [u8], result: i32, record_len: usize) {
    for record in prefix.chunks_exact_mut(record_len) {
        let stored = match packed_is_white(record) {
            true => -result,
            false => result,
        };
        record[RESULT_OFF] = stored as u8;
    }
}

/// Restore the position a record holds.
///
/// The 32 packed bytes carry no move number — two positions differing only in
/// ply pack identically — so the record's own `ply` field is the only place it
/// survives, and `set_ply` is how it goes back. That needs no `refresh_keys`:
/// the ply is deliberately not part of the Zobrist key.
fn position_from(record: &[u8]) -> Result<Position, String> {
    let packed: PackedSfen = record[..PACKED_LEN]
        .try_into()
        .expect("a record is longer than a packed position");
    let mut position = sfen_unpack(&packed).map_err(|err| format!("{err:?}"))?;

    // Re-packing and comparing is what separates a real record from a stream
    // read at the wrong offset: misaligned bytes decode to a position that is
    // merely wrong rather than to an error. `xtask`'s `unpack-starts` checks
    // the same way for the same reason.
    if sfen_pack(&position) != packed {
        return Err("the position does not re-pack to its own bytes; \
                    the file is probably misaligned"
            .to_string());
    }
    position.set_ply(record_ply(record));
    Ok(position)
}

/// Where a game's usable records end.
///
/// The first ply that jumps by more than one is where the eval limit started
/// dropping records; everything from there is discarded and replayed. A game
/// with no hole keeps all of its records and is resumed from its last one,
/// which is the same operation — the tail is missing there too, it just has no
/// gap to mark it.
fn cut_at(records: &[u8], record_len: usize) -> usize {
    let mut previous = None;
    for (index, record) in records.chunks_exact(record_len).enumerate() {
        let ply = record_ply(record);
        if let Some(previous) = previous {
            if ply > previous + 1 {
                return index;
            }
        }
        previous = Some(ply);
    }
    records.len() / record_len
}

/// Split one game's records into a resume point.
///
/// `None` when the game holds nothing usable, which a zero-length run is.
pub fn resume_from(records: &[u8], record_len: usize) -> Option<Result<Resume, String>> {
    let cut = cut_at(records, record_len);
    if cut == 0 {
        return None;
    }
    // The last kept record is where play restarts, and `play_game` records that
    // position itself as its first row — with a fresh search behind it and
    // whatever move the new game picks. Keeping the original alongside it would
    // store the same position twice, and the pair reads as a game boundary
    // because the ply does not advance between them. The new one is the one
    // that matches the game it belongs to, so the original is dropped.
    let last = &records[(cut - 1) * record_len..cut * record_len];
    let prefix = records[..(cut - 1) * record_len].to_vec();
    // `cut - 1` plies, saturating only against a corpus whose games are longer
    // than a `u16` — which the ply field itself could not have held.
    let played = u16::try_from(cut - 1).unwrap_or(u16::MAX);
    Some(position_from(last).map(|position| Resume {
        prefix,
        position,
        played,
    }))
}

/// One shard, read a game at a time.
///
/// Games are not delimited in the file. A boundary is a ply that did not
/// increase, which is what `fleet.py`'s corpus scan counts games by, and it
/// holds because each shard is one worker's output in the order it played.
pub struct Shard {
    reader: BufReader<File>,
    record_len: usize,
    /// Read but not yet claimed: the first record of the next game.
    pending: Option<Vec<u8>>,
}

impl Shard {
    pub fn open(path: &Path, multipv: usize) -> Result<Shard, String> {
        let file = File::open(path).map_err(|err| format!("{}: {err}", path.display()))?;
        let length = file
            .metadata()
            .map_err(|err| format!("{}: {err}", path.display()))?
            .len();
        let record_len = record_len(multipv);
        if length % record_len as u64 != 0 {
            return Err(format!(
                "{}: {length} bytes is not a whole number of {record_len}-byte records; \
                 --multipv {multipv} is probably not what wrote it",
                path.display()
            ));
        }
        Ok(Shard {
            reader: BufReader::with_capacity(1 << 20, file),
            record_len,
            pending: None,
        })
    }

    fn read_record(&mut self) -> Result<Option<Vec<u8>>, String> {
        let mut record = vec![0u8; self.record_len];
        let mut filled = 0;
        while filled < self.record_len {
            match self.reader.read(&mut record[filled..]) {
                Ok(0) if filled == 0 => return Ok(None),
                Ok(0) => return Err("the shard ends mid-record".to_string()),
                Ok(n) => filled += n,
                Err(err) => return Err(err.to_string()),
            }
        }
        Ok(Some(record))
    }

    /// The next game's records, concatenated. `None` at the end of the shard.
    pub fn next_game(&mut self) -> Option<Result<Vec<u8>, String>> {
        let first = match self.pending.take() {
            Some(record) => record,
            None => match self.read_record() {
                Ok(Some(record)) => record,
                Ok(None) => return None,
                Err(err) => return Some(Err(err)),
            },
        };

        let mut previous = record_ply(&first);
        let mut game = first;
        loop {
            match self.read_record() {
                Ok(None) => return Some(Ok(game)),
                Err(err) => return Some(Err(err)),
                Ok(Some(record)) => {
                    let ply = record_ply(&record);
                    if ply <= previous {
                        self.pending = Some(record);
                        return Some(Ok(game));
                    }
                    previous = ply;
                    game.extend_from_slice(&record);
                },
            }
        }
    }
}

/// The shards left to recover, handed out one at a time.
///
/// A shared cursor rather than a shard per worker index: the governor parks
/// workers by index, so a fixed assignment would leave the high-numbered
/// workers' shards untouched for as long as the fleet stays shrunk.
pub struct Source {
    shards: Vec<PathBuf>,
    cursor: std::sync::atomic::AtomicUsize,
    pub multipv: usize,
}

impl Source {
    pub fn new(dir: &Path, multipv: usize) -> Result<Source, String> {
        Ok(Source {
            shards: shards(dir)?,
            cursor: std::sync::atomic::AtomicUsize::new(0),
            multipv,
        })
    }

    pub fn len(&self) -> usize {
        self.shards.len()
    }

    /// The next unclaimed shard, or `None` once every one has been handed out.
    pub fn claim(&self) -> Option<&Path> {
        let index = self
            .cursor
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.shards.get(index).map(PathBuf::as_path)
    }

    /// Whether anything is left to hand out. Read by a parked worker deciding
    /// whether it is waiting for work that will never come.
    pub fn exhausted(&self) -> bool {
        self.cursor.load(std::sync::atomic::Ordering::Relaxed) >= self.shards.len()
    }
}

/// Every shard under `dir`, in a stable order.
///
/// `.bin` only: a directory of a `--format jsonl` run holds `.jsonl`, which is
/// a different shape and would be read as garbage at 60-byte strides.
pub fn shards(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let entries = std::fs::read_dir(dir).map_err(|err| format!("{}: {err}", dir.display()))?;
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == "bin"))
        .collect();
    if paths.is_empty() {
        return Err(format!("{}: no .bin shards", dir.display()));
    }
    paths.sort();
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use attic_state::parse_sfen;

    use super::*;

    const MULTIPV: usize = 5;

    /// One record holding `position`, stamped with `ply` and `result`.
    fn record(position: &Position, ply: u16, result: i8) -> Vec<u8> {
        let mut out = vec![0u8; record_len(MULTIPV)];
        out[..PACKED_LEN].copy_from_slice(&sfen_pack(position));
        out[PLY_OFF..PLY_OFF + 2].copy_from_slice(&ply.to_le_bytes());
        out[RESULT_OFF] = result as u8;
        out
    }

    fn startpos() -> Position {
        Position::startpos()
    }

    /// A position with White to move, for the side-to-move bit.
    fn white_to_move() -> Position {
        parse_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 2")
            .expect("the sfen is valid")
    }

    #[test]
    fn the_side_to_move_comes_off_the_first_packed_bit() {
        // What `rewrite_results` reads instead of unpacking every record.
        assert!(!packed_is_white(&record(&startpos(), 1, 0)));
        assert!(packed_is_white(&record(&white_to_move(), 2, 0)));
    }

    #[test]
    fn a_run_of_consecutive_plies_has_no_cut() {
        let mut records = Vec::new();
        for ply in 1..=8 {
            records.extend_from_slice(&record(&startpos(), ply, 1));
        }
        assert_eq!(cut_at(&records, record_len(MULTIPV)), 8);
    }

    #[test]
    fn the_cut_lands_on_the_first_ply_that_jumps() {
        // 1 2 3 7 8 — the eval limit dropped 4, 5 and 6.
        let mut records = Vec::new();
        for ply in [1, 2, 3, 7, 8] {
            records.extend_from_slice(&record(&startpos(), ply, 1));
        }
        assert_eq!(
            cut_at(&records, record_len(MULTIPV)),
            3,
            "the three records before the hole are kept"
        );
    }

    #[test]
    fn a_restored_position_reports_the_records_ply() {
        // The packed form carries no move number, so this is the only place it
        // survives a round trip.
        let restored = position_from(&record(&startpos(), 137, 1)).expect("the record decodes");
        assert_eq!(restored.ply(), 137);
        assert_eq!(sfen_pack(&restored), sfen_pack(&startpos()));
    }

    #[test]
    fn a_misaligned_record_is_refused_rather_than_decoded() {
        // Reading at the wrong offset yields bytes that may still unpack; the
        // re-pack check is what separates those from real records.
        let mut shifted = record(&startpos(), 1, 1);
        shifted[..PACKED_LEN].rotate_left(3);
        match position_from(&shifted) {
            Err(message) => assert!(message.contains("misaligned") || !message.is_empty()),
            Ok(position) => panic!("rotated bytes decoded to {}", position.ply()),
        }
    }

    #[test]
    fn results_are_rewritten_from_each_records_own_side() {
        let mut prefix = record(&startpos(), 1, 0);
        prefix.extend_from_slice(&record(&white_to_move(), 2, 0));
        rewrite_results(&mut prefix, 1, record_len(MULTIPV));

        let len = record_len(MULTIPV);
        assert_eq!(prefix[RESULT_OFF] as i8, 1, "Black to move keeps the result");
        assert_eq!(
            prefix[len + RESULT_OFF] as i8,
            -1,
            "White to move stores it flipped"
        );
    }

    #[test]
    fn a_resume_keeps_the_records_before_the_hole() {
        let mut records = Vec::new();
        for ply in [1, 2, 3, 9] {
            records.extend_from_slice(&record(&startpos(), ply, 1));
        }
        let resume = resume_from(&records, record_len(MULTIPV))
            .expect("the game has records")
            .expect("they decode");
        assert_eq!(resume.position.ply(), 3, "resumed from the last kept record");
        assert_eq!(
            resume.prefix.len(),
            2 * record_len(MULTIPV),
            "ply 3 is dropped: the replay records that position itself"
        );
    }

    #[test]
    fn a_resume_carries_the_plies_the_original_game_already_spent() {
        // `--max-ply` bounds the game, not the continuation. Before this was
        // carried through, a recovered game restarted its count at zero and got
        // a second full budget — a 320-ply cap that let games reach 640.
        let mut records = Vec::new();
        for ply in 1..=40 {
            records.extend_from_slice(&record(&startpos(), ply, 1));
        }
        let resume = resume_from(&records, record_len(MULTIPV))
            .expect("the game has records")
            .expect("they decode");
        assert_eq!(
            resume.played, 39,
            "the last record is replayed, so 39 plies are behind it",
        );
        assert_eq!(
            usize::from(resume.played) * record_len(MULTIPV),
            resume.prefix.len(),
            "up to the cut there is one record per ply",
        );
    }

    #[test]
    fn a_single_record_game_keeps_no_prefix() {
        // The whole game is replayed from its only position, so there is
        // nothing in front of it to carry through.
        let records = record(&startpos(), 5, 1);
        let resume = resume_from(&records, record_len(MULTIPV))
            .expect("the game has a record")
            .expect("it decodes");
        assert!(resume.prefix.is_empty());
        assert_eq!(resume.position.ply(), 5);
    }
}
