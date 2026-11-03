//! Building a book by expanding a tree, rather than by playing games.
//!
//! A game is a walk: it reaches one new board per search only at its deepest
//! ply, and the next game from the same root walks the plies before it again.
//! Measured on a nine-ply book, 43% of the searches landed on boards the run
//! had already written — 9,333 searches for 5,323 boards. The waste is worst
//! where the tree is narrowest, which is exactly where a book starts.
//!
//! Expanding a tree pays for each board once. The search's own MultiPV list
//! names the children, so branching costs nothing beyond the search the board
//! needed anyway: `--move-margin` decides how many of that list are admitted,
//! and each admitted move is a child.
//!
//! The frontier is fleet-wide rather than per NUMA node. Two nodes drawing
//! from separate frontiers would expand the same boards, which is the cost
//! this module exists to remove — so unlike the transposition table and the
//! weights, this one is shared across the fabric. It is consulted once per
//! search, not once per node of one, so the traffic is nothing.

mod filters;
mod output;
mod worker;

#[cfg(test)]
mod tests;

pub use worker::run;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};

use attic_state::parse_sfen;

use crate::game::Rng;
use crate::options::Options;

/// One board handed to a worker.
pub struct Task {
    pub sfen: String,
    /// The board this one was expanded from, empty for a root. Carried so a
    /// finished book can be audited move by move: without it a level is a bag
    /// of positions with no way back to how each was reached.
    pub parent: String,
    /// False on the last level, where children would be built and then thrown
    /// away. The searches are the cost, but a wide level still makes six
    /// `do_move`s and six SFEN strings per board for nothing.
    pub expand: bool,
}

struct State {
    /// The move number the current level's boards carry, for naming its file.
    ply: u16,
    /// Boards at the current level no worker has taken yet.
    take: Vec<(String, String)>,
    /// Boards at the current level a search kept, with the root score the
    /// search gave them. Written when the level closes, not as they arrive: a
    /// board is only known to be in the book after it has been searched.
    kept: Vec<(i32, String, String)>,
    /// Children found so far. They become the next level.
    next: Vec<(String, String)>,
    /// The score of the position the book was rooted at, in that position's
    /// own frame, and the move number it carried. Every band is measured from
    /// this and nothing else. `None` until the root level closes.
    anchor: Option<i32>,
    anchor_ply: u16,
    /// The level below the current one, kept back so its scores can be
    /// replaced by the values backed up from these children. See `back_up`.
    held: Vec<(i32, String, String)>,
    held_ply: u16,
    /// Position keys ever queued. A transposition — the same board reached by
    /// a different move order, which the shogi opening is full of — is
    /// expanded once. This is the exact set the lossy `Seen` table
    /// approximates for games.
    seen: HashSet<u64>,
    /// Workers holding a board out of `take`. The level is not closed until
    /// this reaches zero, or a straggler's children would arrive after the
    /// level they belong to had been sampled.
    busy: usize,
    rng: Rng,
    done: bool,
}

pub struct Frontier {
    state: Mutex<State>,
    ready: Condvar,
    /// Expand until the boards carry this move number. **A move number, not a
    /// game length** — unlike `--max-ply` and `--dedup-max-ply`, which count
    /// plies from the start of a game. A book is named by the move number its
    /// boards carry, and `--start-sfens` roots already carry their own.
    stop: u16,
    /// Boards kept per level, `0` for all of them.
    width: usize,
    /// Drop a board whose root score is worse than this, and do not expand it.
    /// `0` for no band.
    band: i32,
    /// `--move-margin`, applied a second time against the children's own
    /// scores. `0` for no check.
    margin: i32,
    out: PathBuf,
    searched: AtomicU64,
    dropped: AtomicU64,
    refuted: AtomicU64,
}

impl Frontier {
    /// `roots` are SFEN text, with or without the leading `sfen `.
    ///
    /// The level's move number is taken from the first root and then counted
    /// up, so a book built from roots with mixed move numbers is named by the
    /// first of them.
    pub fn new(roots: Vec<String>, options: &Options) -> Self {
        let ply = roots
            .first()
            .and_then(|sfen| parse_sfen(sfen).ok())
            .map_or(1, |pos| pos.ply());
        Frontier {
            state: Mutex::new(State {
                ply,
                take: roots
                    .into_iter()
                    .map(|sfen| (sfen, String::new()))
                    .collect(),
                kept: Vec::new(),
                next: Vec::new(),
                held: Vec::new(),
                held_ply: 0,
                anchor: None,
                anchor_ply: 0,
                seen: HashSet::new(),
                busy: 0,
                rng: Rng::new(options.seed),
                done: false,
            }),
            ready: Condvar::new(),
            stop: options.book_ply,
            width: options.book_width,
            band: options.book_band,
            margin: options.move_margin,
            out: options.out.clone(),
            searched: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            refuted: AtomicU64::new(0),
        }
    }

    /// The next board to search, or `None` once the last level is written.
    ///
    /// Blocks while the level is draining: a worker that finds `take` empty
    /// cannot conclude the run is over, because the boards still out with
    /// other workers are about to supply the next level.
    pub fn claim(&self) -> Option<Task> {
        let mut state = self.state.lock().expect("frontier mutex poisoned");
        loop {
            if state.done {
                return None;
            }
            if let Some((sfen, parent)) = state.take.pop() {
                state.busy += 1;
                return Some(Task {
                    sfen,
                    parent,
                    expand: state.ply < self.stop,
                });
            }
            if state.busy > 0 {
                state = self
                    .ready
                    .wait(state)
                    .expect("frontier mutex poisoned while waiting");
                continue;
            }
            self.advance(&mut state);
            self.ready.notify_all();
        }
    }

    /// Hand back a searched board: `kept` as `(root score, SFEN, parent)`
    /// unless the band rejected it, and the children to queue as
    /// `(position key, SFEN, parent)`.
    pub fn finish(
        &self,
        kept: Option<(i32, String, String)>,
        children: Vec<(u64, String, String)>,
    ) {
        let mut state = self.state.lock().expect("frontier mutex poisoned");
        self.searched.fetch_add(1, Ordering::Relaxed);
        match kept {
            Some(scored) => state.kept.push(scored),
            None => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
        for (key, sfen, parent) in children {
            if state.seen.insert(key) {
                state.next.push((sfen, parent));
            }
        }
        state.busy -= 1;
        if state.busy == 0 {
            self.ready.notify_all();
        }
    }

    /// Close the level, write it, and sample the next one.
    ///
    /// The cut is why a level has to be complete before any of it is popped:
    /// shuffling the whole level and truncating draws uniformly from it, while
    /// taking the first `width` boards to arrive would take the children of
    /// whichever parents happened to be expanded first — at six children a
    /// board, a level of `width` boards would descend from `width / 6` of the
    /// level above it, and the book would narrow with every ply.
    fn advance(&self, state: &mut State) {
        let mut kept = std::mem::take(&mut state.kept);
        let mut next = std::mem::take(&mut state.next);
        let found = next.len();
        let refuted = self.apply_margin(&mut kept, &mut next);
        let cut = self.apply_band(state, &mut kept, &mut next);
        // On disk the moment it closes, before anything is backed up into it.
        // A book to move 32 takes long enough that it will be stopped part-way,
        // and a run stopped at move 20 has to leave every level up to 19 there
        // and usable — a level is named by its move number precisely so that a
        // shallower book can be taken out of a deeper run.
        if let Err(err) = self.write_level(state.ply, &kept) {
            eprintln!("gensfen: cannot write the ply {} book: {err}", state.ply);
        }
        // The level above is then written a second time, now that its children
        // are in to be backed up from. Only the scores change: the SFEN file a
        // deeper run reads back was final on the first write.
        self.back_up(&mut state.held, &kept);
        if !state.held.is_empty()
            && let Err(err) = self.write_level(state.held_ply, &state.held)
        {
            eprintln!(
                "gensfen: cannot write the ply {} book: {err}",
                state.held_ply
            );
        }
        eprintln!(
            "gensfen: ply {}: {} boards kept, {} children{}{}",
            state.ply,
            kept.len(),
            found,
            match refuted {
                0 => String::new(),
                n => format!(", {n} refuted"),
            },
            match cut {
                0 => String::new(),
                n => format!(", {n} outside the band"),
            }
        );

        if state.ply >= self.stop || next.is_empty() {
            // Already written, and it stays as its own search scored it: the
            // deepest level has no searched children to be backed up from. A
            // book whose last level needs backed-up scores has to be built one
            // move deeper than it is used, which is what stopping short costs
            // as well.
            state.done = true;
            return;
        }
        state.held_ply = state.ply;
        state.held = kept;
        state.ply += 1;
        if self.width > 0 && next.len() > self.width {
            state.rng.shuffle(&mut next);
            next.truncate(self.width);
        }
        state.take = next;
    }

    pub fn counts(&self) -> (u64, u64, u64) {
        (
            self.searched.load(Ordering::Relaxed),
            self.dropped.load(Ordering::Relaxed),
            self.refuted.load(Ordering::Relaxed),
        )
    }
}
