//! The accumulator the search threads through the tree, and what a derive needs.

use attic_state::{Color, Position};

use crate::features::{
    DiffScratch, FeatureIndex, MAX_ACTIVE_FEATURES, active_features, active_features_into,
    changed_indices_into,
};

use super::{DIM, ORDER, SymNetwork};

/// The accumulator the search threads through the tree.
///
/// Holds each perspective's power sums, and the active-feature list they were
/// built from so that the next position can be reached by difference.
///
/// ## Why the sums update incrementally and the polynomials do not
///
/// `p_k[c] = Σ_i v_i[c]^k` is a sum over tokens, so a token leaving subtracts
/// its own `k`-th powers and a token arriving adds them — the other
/// thirty-eight are untouched. The elementary symmetric polynomials that
/// [`elementary_from_power`] derives are *not* sums over tokens and cannot be
/// updated this way, which is exactly why the accumulator stores `p` and the
/// evaluation converts. That conversion is `ORDER · DIM` multiply-adds once
/// per evaluation rather than per token, and it is the whole reason Newton's
/// identities are in the forward at all.
///
/// ## Why there is no king-move special case
///
/// HalfKP renumbers all forty features when the perspective's own king moves,
/// which for NNUE means a rebuild. Here it needs no special handling: the diff
/// against the previous list reports thirty-eight removals and thirty-eight
/// additions, and applying those is arithmetically a rebuild. It costs what a
/// rebuild costs and produces what a rebuild produces, so the correctness does
/// not depend on the caller noticing.
#[derive(Debug, Clone)]
pub struct Accumulator {
    /// `[perspective][ORDER * DIM]`, indexed by [`Color::index`].
    sums: [Vec<f32>; Color::COUNT],
    /// What each half was built from, kept so a derive can diff against it.
    active: [Vec<FeatureIndex>; Color::COUNT],
}

impl Default for Accumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Accumulator {
    pub fn new() -> Self {
        Accumulator {
            sums: [vec![0.0; ORDER * DIM], vec![0.0; ORDER * DIM]],
            active: [
                Vec::with_capacity(MAX_ACTIVE_FEATURES),
                Vec::with_capacity(MAX_ACTIVE_FEATURES),
            ],
        }
    }

    /// Rebuild both perspectives from scratch.
    ///
    /// # Panics
    /// Panics if `pos` is missing either king.
    pub fn refresh(&mut self, net: &SymNetwork, pos: &Position) {
        for color in [Color::Black, Color::White] {
            let i = color.index();
            self.active[i] = active_features(pos, color);
            self.sums[i].fill(0.0);
            for &feature in &self.active[i] {
                accumulate(&mut self.sums[i], net, feature, 1.0);
            }
        }
    }

    /// The power sums for one perspective.
    pub fn perspective(&self, color: Color) -> &[f32] {
        &self.sums[color.index()]
    }

    /// Derive the accumulator for the position after `mv` from `self`, the one
    /// before it.
    ///
    /// `pos` is the pre-move position and is left byte-for-byte unchanged: the
    /// do/undo happens inside. Mirrors the NNUE accumulator's contract so that
    /// the search treats the two the same way.
    ///
    /// # Panics
    /// Panics if `pos` is missing either king, before or after `mv`.
    pub fn update_after_move(
        &self,
        net: &SymNetwork,
        pos: &mut Position,
        mv: attic_state::Move,
    ) -> Accumulator {
        let undo = pos.do_move(mv);
        let mut next = self.clone();
        // The one call site that has no cache to borrow from — the root, once
        // per iteration. The tree goes through `derive_into_cached`.
        let mut scratch = Scratch::default();
        next.rebase(net, pos, &mut scratch);
        pos.undo_move(mv, undo);
        next
    }

    /// Move `dst` to the position `post_pos` describes, starting from `src`.
    ///
    /// The `delta` and `cache` arguments are NNUE's and go unread: the diff
    /// this needs is over feature indices, which the two active lists already
    /// give, and there is no king bucket to cache a rebuild for.
    pub fn derive_into_cached(
        src: &Accumulator,
        dst: &mut Accumulator,
        net: &SymNetwork,
        post_pos: &Position,
        _delta: &crate::features::MoveDelta,
        cache: &mut FinnyCache,
    ) {
        // `clone_from` reuses `dst`'s allocations rather than replacing them,
        // which is what makes a node free of the heap once the stack is warm.
        dst.clone_from(src);
        dst.rebase(net, post_pos, &mut cache.scratch);
    }

    /// Bring `self` — currently holding some earlier position — to `pos`, by
    /// difference against the feature list it was built from.
    ///
    /// Every buffer it needs comes from `scratch`, so a node costs no
    /// allocation: the new feature list is written over the previous one's
    /// storage and then swapped in, and the diff reuses its own four vectors.
    fn rebase(&mut self, net: &SymNetwork, pos: &Position, scratch: &mut Scratch) {
        for color in [Color::Black, Color::White] {
            let i = color.index();
            active_features_into(pos, color, &mut scratch.after[i]);
            changed_indices_into(&self.active[i], &scratch.after[i], &mut scratch.diff);
            for &feature in &scratch.diff.removed {
                accumulate(&mut self.sums[i], net, feature, -1.0);
            }
            for &feature in &scratch.diff.added {
                accumulate(&mut self.sums[i], net, feature, 1.0);
            }
            // Swap rather than assign: `self.active[i]`'s allocation becomes
            // next ply's scratch instead of being dropped and re-made.
            std::mem::swap(&mut self.active[i], &mut scratch.after[i]);
        }
    }
}

/// Add or subtract one feature's powers, in place.
///
/// `sign` is `±1`, and the subtraction is exact in the sense that matters:
/// `add` then `sub` of the same row restores the sum bit-for-bit only up to
/// f32 rounding, which is why [`Accumulator::rebase`] always diffs against a
/// list it holds rather than composing many small deltas blindly.
fn accumulate(sums: &mut [f32], net: &SymNetwork, feature: FeatureIndex, sign: f32) {
    let row = &net.embed[feature as usize * DIM..(feature as usize + 1) * DIM];
    for (channel, &value) in row.iter().enumerate() {
        let mut power = value;
        for k in 0..ORDER {
            sums[k * DIM + channel] += sign * power;
            power *= value;
        }
    }
}

/// The buffers a derive needs, owned across the whole search.
///
/// Two feature lists and the diff's four vectors. Held here rather than in
/// [`Accumulator`] because the accumulator is copied down the tree and this is
/// not — one worker needs one set of these, not one per ply.
#[derive(Debug, Default)]
struct Scratch {
    after: [Vec<FeatureIndex>; Color::COUNT],
    diff: DiffScratch,
}

/// Where NNUE keeps its king-bucket refresh cache. Nothing here buckets by
/// king, so what it carries instead is the derive's scratch — the name is
/// NNUE's because the search allocates one per worker and hands it to every
/// derive, which is exactly what this needs too.
#[derive(Debug, Default)]
pub struct FinnyCache {
    scratch: Scratch,
}

impl FinnyCache {
    pub fn new() -> Box<Self> {
        Box::new(FinnyCache::default())
    }
}
