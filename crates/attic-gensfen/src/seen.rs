//! Remembering which boards this run has already searched.
//!
//! `manaka-pack` folds records that share a position into one row, so a board
//! searched twice is searched once for nothing: the second depth-9 MultiPV-5
//! search is paid for, written, and then averaged away. Measured on the V4
//! corpus, the fold removed 10.28% of the records and **99.12% of what it
//! removed sat below ply 32** — games out of the same opening replay the same
//! greedy line until they hit a randomised ply.
//!
//! So the generator asks, before searching, whether it has seen the board
//! before. A board it has seen is still searched, because the move it plays
//! instead is drawn from that search's own MultiPV list, but the ply is not
//! recorded and the game leaves the line that reached it first. See
//! `play_game` in [`crate::game`] for why the search is not skipped.
//!
//! **The table is allowed to be wrong.** Correctness lives in the fold, which
//! sees every record; this only decides where to spend a search. A collision
//! that reports a hit costs one position the corpus would have held; a hit it
//! fails to report costs a duplicate search the fold removes exactly as it does
//! today. Neither is a wrong number in the output, which is why the table can
//! be small and lock-free.

use std::sync::atomic::{AtomicU64, Ordering};

/// A fixed table of boards already searched, shared across the workers of one
/// NUMA node.
///
/// One slot per key, no chaining and no tags: a slot holds the whole key, and a
/// new key overwrites whatever was there. Every access is `Relaxed` for the
/// same reason [`attic_storage::TranspositionTable`]'s is — nothing is
/// published *through* these words, and a read that misses a concurrent write
/// merely searches a position twice.
pub struct Seen {
    slots: Box<[AtomicU64]>,
    /// How far to shift a key to reach its slot. Kept instead of a mask
    /// because the index comes from the high bits.
    shift: u32,
}

impl Seen {
    /// A table of about `mb` megabytes, or `None` when the filter is off.
    ///
    /// Rounded **down** to a power of two, so `--dedup-mb 100` gives 64 MB
    /// rather than the 128 MB rounding up would have quietly taken.
    pub fn new(mb: usize) -> Option<Self> {
        let wanted = mb.checked_mul(1024 * 1024)? / size_of::<u64>();
        if wanted == 0 {
            return None;
        }
        let bits = wanted.ilog2();
        let slots = 1usize << bits;
        Some(Self {
            slots: (0..slots).map(|_| AtomicU64::new(0)).collect(),
            shift: 64 - bits,
        })
    }

    /// Records `key` and reports whether it was already there.
    ///
    /// The index comes from the key's high bits and the slot holds the key
    /// itself, so a single `swap` both answers and updates. A key of exactly
    /// `0` is indistinguishable from an empty slot and always reports a miss;
    /// it is one board out of 2^64 and it costs one repeated search.
    pub fn mark(&self, key: u64) -> bool {
        let index = (key >> self.shift) as usize;
        self.slots[index].swap(key, Ordering::Relaxed) == key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_misses_the_first_time_and_hits_the_second() {
        let seen = Seen::new(1).expect("1 MB is a table");
        assert!(!seen.mark(0x1234_5678_9abc_def0));
        assert!(seen.mark(0x1234_5678_9abc_def0));
    }

    #[test]
    fn a_key_that_lands_on_a_taken_slot_evicts_it_rather_than_matching() {
        let seen = Seen::new(1).expect("1 MB is a table");
        // Same high bits, so the same slot; different low bits, so not the
        // same key.
        let first = 0xdead_0000_0000_0001;
        let second = 0xdead_0000_0000_0002;
        assert!(!seen.mark(first));
        assert!(!seen.mark(second), "a different key must not read as a hit");
        assert!(
            !seen.mark(first),
            "and the first one is gone, not remembered"
        );
    }

    #[test]
    fn the_table_is_off_at_zero_megabytes() {
        assert!(Seen::new(0).is_none());
    }

    #[test]
    fn a_size_between_two_powers_rounds_down() {
        let seen = Seen::new(3).expect("3 MB is a table");
        // 3 MB / 8 B = 393,216 slots wanted; the table takes 2^18 = 262,144.
        assert_eq!(seen.slots.len(), 1 << 18);
        assert_eq!(seen.shift, 64 - 18);
    }
}
