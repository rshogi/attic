//! The three judgements a closing level goes through: backing scores up from
//! the level below, and the two cuts that keep a book even.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;

use super::{Frontier, State};

impl Frontier {
    /// Replace a level's scores with the values backed up from the level
    /// below it.
    ///
    /// A board's own search reports what it saw at the root of that search; a
    /// parent's search saw the same child one ply shallower. Taking the score
    /// from the child's own search instead makes the two agree, and makes the
    /// number one ply deeper into the bargain.
    ///
    /// A child's score is from the opponent's side to move, so the parent's
    /// best child is the **lowest** of them and the parent's value is its
    /// negation.
    ///
    /// This can do nothing for the deepest level, which has no searched
    /// children: a book whose last level needs backed-up scores has to be
    /// built one move deeper than it is used.
    pub(super) fn back_up(
        &self,
        held: &mut [(i32, String, String)],
        children: &[(i32, String, String)],
    ) {
        if held.is_empty() {
            return;
        }
        let mut best: HashMap<&str, i32> = HashMap::new();
        for (cp, _, parent) in children {
            let slot = best.entry(parent.as_str()).or_insert(*cp);
            *slot = (*slot).min(*cp);
        }
        for (cp, sfen, _) in held.iter_mut() {
            if let Some(&best) = best.get(sfen.as_str()) {
                *cp = -best;
            }
        }
    }

    /// Drop a board whose own search refutes the move that reached it, and its
    /// children with it.
    ///
    /// `--move-margin` is applied inside a search, against the scores that
    /// search gave the moves at depth `D - 1`. The board is then searched
    /// itself, at depth `D`, and it is that deeper score the book records. The
    /// two disagree often enough to matter: over 9,915 edges only 42% of the
    /// children stood inside the margin their parent admitted them under, the
    /// worst by 474 centipawns. Searching a ply deeper does not close it —
    /// depth 10 measured worse, at 34%.
    ///
    /// Since the tree searches every child, the margin can simply be judged a
    /// second time against the children's own scores. Siblings share a side to
    /// move, so they compare directly, and the best of them is the lowest.
    ///
    /// `--book-width` truncates the level after this, so a parent whose better
    /// siblings were cut keeps children it would otherwise have lost. That errs
    /// towards keeping a board, which is the safe direction.
    pub(super) fn apply_margin(
        &self,
        kept: &mut Vec<(i32, String, String)>,
        next: &mut Vec<(String, String)>,
    ) -> usize {
        if self.margin <= 0 || kept.is_empty() {
            return 0;
        }
        let mut best: HashMap<&str, i32> = HashMap::new();
        for (cp, _, parent) in kept.iter() {
            let slot = best.entry(parent.as_str()).or_insert(*cp);
            *slot = (*slot).min(*cp);
        }
        // Roots have no parent and nothing to be compared against.
        let cut: HashMap<String, i32> = best
            .into_iter()
            .filter(|(parent, _)| !parent.is_empty())
            .map(|(parent, best)| (parent.to_string(), best + self.margin))
            .collect();

        let before = kept.len();
        let mut dropped = HashSet::new();
        kept.retain(|(cp, sfen, parent)| {
            if cut.get(parent).is_none_or(|limit| cp <= limit) {
                return true;
            }
            dropped.insert(sfen.clone());
            false
        });
        next.retain(|(_, parent)| !dropped.contains(parent));
        let n = before - kept.len();
        self.refuted.fetch_add(n as u64, Ordering::Relaxed);
        n
    }

    /// Drop the boards that have drifted too far from the root, and their
    /// children with them.
    ///
    /// The distance is measured from the score of the position the book was
    /// rooted at — a fixed number for the whole run — and not from the level
    /// the board is in. That is the whole point of it. `--move-margin` is
    /// measured from the best move at each node, so its reference point walks
    /// with the position: a node the search reads at -50 admits moves down to
    /// -100, and from there down to -150. Nothing pulls back towards even, so
    /// eight plies of a margin of 50 spread the ninth over 200 centipawns.
    /// Measured, without this: median +80, but from -136 to +607.
    ///
    /// Anchoring removes the walk rather than absorbing it. The band holds for
    /// every level at once, so a book is as even at move nine as at move one,
    /// and how deep it goes stops mattering to how far it can drift.
    ///
    /// Scores arrive from each board's own side to move, so they are flipped
    /// into the root's frame before being compared. A level is one side to
    /// move, so this is one flip per level, decided by parity.
    ///
    /// The children go with the parent because a board nobody may enter is not
    /// worth expanding — which is what makes this cheaper than filtering the
    /// last level, and what keeps a book even all the way down rather than
    /// only at its end.
    ///
    /// A child is queued under the first parent that reached it, so one that
    /// also transposes from a parent inside the band goes down with a banded
    /// one. That loses a reachable board rather than admitting an unreachable
    /// one.
    pub(super) fn apply_band(
        &self,
        state: &mut State,
        kept: &mut Vec<(i32, String, String)>,
        next: &mut Vec<(String, String)>,
    ) -> usize {
        if kept.is_empty() {
            return 0;
        }
        // The root level sets the anchor and is never banded against itself.
        // Several roots take their median, so one wild starting position does
        // not carry the whole book with it.
        let Some(anchor) = state.anchor else {
            let mut scores: Vec<i32> = kept.iter().map(|&(cp, _, _)| cp).collect();
            scores.sort_unstable();
            state.anchor = Some(scores[scores.len() / 2]);
            state.anchor_ply = state.ply;
            eprintln!(
                "gensfen: anchored at {} centipawns",
                scores[scores.len() / 2]
            );
            return 0;
        };
        if self.band <= 0 {
            return 0;
        }
        // Into the root's frame: a level is one side to move, so the flip is
        // the same for every board in it.
        let flip = (state.ply - state.anchor_ply) % 2 == 1;

        let before = kept.len();
        let mut dropped = HashSet::new();
        kept.retain(|(cp, sfen, _)| {
            let rooted = if flip { -cp } else { *cp };
            if (rooted - anchor).abs() <= self.band {
                return true;
            }
            dropped.insert(sfen.clone());
            false
        });
        next.retain(|(_, parent)| !dropped.contains(parent));
        let cut = before - kept.len();
        self.dropped.fetch_add(cut as u64, Ordering::Relaxed);
        cut
    }
}
