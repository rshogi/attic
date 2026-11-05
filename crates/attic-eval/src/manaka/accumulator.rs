//! The accumulator the search threads through the tree, and what a derive needs.

use attic_state::{Color, Position};

#[cfg(not(feature = "sym_token"))]
use crate::features::{
    FeatureIndex, MAX_ACTIVE_FEATURES, active_features, active_features_into, changed_indices_into,
};
#[cfg(feature = "sym_token")]
use crate::tokens::{FeatureIndex, MAX_ACTIVE_FEATURES, MoveDelta, active_features};

use super::cache::FinnyCache;
#[cfg(not(feature = "sym_token"))]
use super::cache::Scratch;
#[cfg(feature = "sym_token")]
use super::field::touch_field;
#[cfg(feature = "sym_token")]
use super::moment::touch_moment;
#[cfg(feature = "sym_token")]
use super::pair::{theirs, touch_pairs};
use super::{DIM, FIELD_LEN, MOMENT_LEN, ORDER, PAIR, POWER_LEN, SPLIT, SymNetwork};

/// What the pair sums are accumulated in. `f64`, which is not what the tables
/// are stored in or the forward pass computes in (both `f32`): see the note on
/// [`Accumulator`].
pub(super) type PairSum = f64;

/// What the second moment's sums are accumulated in.
pub(super) type MomentSum = f64;

/// What the square field's sums are accumulated in.
pub(super) type FieldSum = f64;

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
///
/// ## How the token shapes get from one position to the next
///
/// The KP and HalfKP shapes scan the new position and diff the list against the
/// one they hold, which is robust to being handed any earlier accumulator and
/// costs a scan and two sorts per node. The token shapes do not: their search
/// hands every derive the accumulator of the position's parent and the
/// [`MoveDelta`] of the move between them, and that names the (at most) two
/// tokens that leave and the two that arrive without looking at the board.
/// `active` is kept up to date by the same delta, so that anything that needs
/// the other tokens present — a pair table pairs each changed token with them —
/// has the set it needs at the point it needs it: [`Accumulator::apply`] takes
/// each removed token out of `active` *before* it subtracts the token's row, and
/// puts each added one in *after* it adds the row.
///
/// ## The pair table
///
/// A pair table is the one sum here that is over *pairs* of tokens, so a token
/// that arrives adds one row for every token already present and one that
/// leaves subtracts the same rows. That is why `active` is kept, and why the
/// order above matters: a capture takes the victim out, then the lifted piece,
/// and the lifted piece's rows against the victim are gone with the victim's.
///
/// Each of those is a floating-point sum taken out of a larger one, which is
/// where `f32` drifts, and the log-sum-exp channels are the worst case: they
/// sum `exp θ`, up to about 1e4, and a logarithm of what is left over after a
/// large term has come and gone reads the rounding of that large term. The
/// search seeds the root with [`Accumulator::refresh`] every iteration, which
/// is what keeps the drift to a few plies' worth; see `tests/sym_network.rs` for
/// how much that is on a trained net.
#[derive(Debug, Clone)]
pub struct Accumulator {
    /// `[perspective][POWER_LEN]`, indexed by [`Color::index`]; within one
    /// perspective `[group][order][channel]`, a split shape having two groups.
    sums: [Vec<f32>; Color::COUNT],
    /// `[perspective][PAIR]`: the sum over the pair table of every pair of
    /// tokens present. Empty for a shape without one.
    pair: [Vec<PairSum>; Color::COUNT],
    /// `[perspective][MOMENT_LEN]`: the upper triangle of `Σ v vᵀ` over the
    /// tokens present. Empty for a shape without a second moment.
    moment: [Vec<MomentSum>; Color::COUNT],
    /// `[perspective][FIELD_LEN]`: the sum of the present tokens' rows of the
    /// field, `[square][channel]`. Empty for a shape without a field.
    field: [Vec<FieldSum>; Color::COUNT],
    /// What each half was built from, kept so a derive can diff against it (KP,
    /// HalfKP) or so that the tokens present are known (the token shapes).
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
            sums: [vec![0.0; POWER_LEN], vec![0.0; POWER_LEN]],
            pair: [vec![0.0; PAIR], vec![0.0; PAIR]],
            moment: [vec![0.0; MOMENT_LEN], vec![0.0; MOMENT_LEN]],
            field: [vec![0.0; FIELD_LEN], vec![0.0; FIELD_LEN]],
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
            // Every pair once: each token against the ones before it.
            self.pair[i].fill(0.0);
            self.moment[i].fill(0.0);
            self.field[i].fill(0.0);
            #[cfg(feature = "sym_token")]
            for (at, &feature) in self.active[i].iter().enumerate() {
                touch_pairs(&mut self.pair[i], net, feature, &self.active[i][..at], 1.0);
                touch_moment(&mut self.moment[i], net, feature, 1.0);
                touch_field(&mut self.field[i], net, feature, 1.0);
            }
        }
    }

    /// The power sums for one perspective, `[group][order][channel]`.
    pub fn perspective(&self, color: Color) -> &[f32] {
        &self.sums[color.index()]
    }

    /// The pair table's sum for one perspective, `[channel]`; empty without one.
    /// The last `LSE` channels hold `Σ exp θ`, not yet a logarithm.
    ///
    /// Accumulated in `f64` and handed over rounded to `f32`, which is what the
    /// forward pass computes in; see [`Accumulator`].
    pub fn pair(&self, color: Color) -> [f32; PAIR] {
        std::array::from_fn(|channel| self.pair[color.index()][channel] as f32)
    }

    /// The second moment's upper triangle for one perspective, `[entry]`; empty
    /// without one. Rounded to `f32` from the `f64` it is accumulated in.
    pub fn moment(&self, color: Color) -> [f32; MOMENT_LEN] {
        std::array::from_fn(|entry| self.moment[color.index()][entry] as f32)
    }

    /// The square field's sum for one perspective, `[square][channel]` flattened;
    /// empty without one. Rounded to `f32` from the `f64` it is accumulated in.
    pub fn field(&self, color: Color) -> [f32; FIELD_LEN] {
        std::array::from_fn(|at| self.field[color.index()][at] as f32)
    }

    /// The tokens present in one perspective, in no particular order.
    #[cfg(all(test, feature = "sym_token"))]
    pub(super) fn tokens(&self, color: Color) -> &[FeatureIndex] {
        &self.active[color.index()]
    }
}

/// KP and HalfKP: by difference against the list the accumulator was built from.
#[cfg(not(feature = "sym_token"))]
impl Accumulator {
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

/// The token shapes: by the move's delta, from the parent's accumulator.
#[cfg(feature = "sym_token")]
impl Accumulator {
    /// Derive the accumulator for the position after `mv` from `self`, the one
    /// before it.
    ///
    /// `pos` is the pre-move position and is not changed; it is `&mut` for the
    /// signature the other shapes have, where the move is done and undone
    /// inside.
    pub fn update_after_move(
        &self,
        net: &SymNetwork,
        pos: &mut Position,
        mv: attic_state::Move,
    ) -> Accumulator {
        let delta = MoveDelta::from_move(pos, mv);
        let mut next = self.clone();
        next.apply(net, &delta);
        next
    }

    /// Move `dst` to the position after the move `delta` describes, starting from
    /// `src`, which has to be the accumulator of the position the move was made
    /// from.
    ///
    /// The position and the cache go unread: a token's row does not depend on
    /// any other piece, so the delta is everything, and there is no bucket whose
    /// rebuild a cache could hold.
    pub fn derive_into_cached(
        src: &Accumulator,
        dst: &mut Accumulator,
        net: &SymNetwork,
        _post_pos: &Position,
        delta: &MoveDelta,
        _cache: &mut FinnyCache,
    ) {
        // `clone_from` reuses `dst`'s allocations rather than replacing them,
        // which is what makes a node free of the heap once the stack is warm.
        dst.clone_from(src);
        dst.apply(net, delta);
    }

    /// Take the tokens `delta` removes out of both halves and put the ones it
    /// adds in.
    ///
    /// Each removal leaves `active` before its row is subtracted and each
    /// addition enters it after its row is added, so at the moment a token's row
    /// is touched `active` is the set of *other* tokens present — what a table
    /// over pairs of tokens must read to subtract or add the right rows.
    ///
    /// # Panics
    /// Panics if a removed token is not present, which means `self` was not the
    /// accumulator of the position the move was made from.
    fn apply(&mut self, net: &SymNetwork, delta: &MoveDelta) {
        for color in [Color::Black, Color::White] {
            let i = color.index();
            let half = delta.half(color).expect("a token delta is never a rebuild");
            for &token in half.removed() {
                let at = self.active[i]
                    .iter()
                    .position(|&present| present == token)
                    .expect("a removed token was not in the accumulator it is removed from");
                self.active[i].swap_remove(at);
                accumulate(&mut self.sums[i], net, token, -1.0);
                touch_pairs(&mut self.pair[i], net, token, &self.active[i], -1.0);
                touch_moment(&mut self.moment[i], net, token, -1.0);
                touch_field(&mut self.field[i], net, token, -1.0);
            }
            for &token in half.added() {
                accumulate(&mut self.sums[i], net, token, 1.0);
                touch_pairs(&mut self.pair[i], net, token, &self.active[i], 1.0);
                touch_moment(&mut self.moment[i], net, token, 1.0);
                touch_field(&mut self.field[i], net, token, 1.0);
                self.active[i].push(token);
            }
        }
    }
}

/// Add or subtract one feature's powers, in place.
///
/// `sign` is `±1`, and the subtraction is exact in the sense that matters:
/// `add` then `sub` of the same row restores the sum bit-for-bit only up to
/// f32 rounding, which is why the KP and HalfKP shapes always diff against a
/// list they hold rather than composing many small deltas blindly. The token
/// shapes do compose them, from the parent's accumulator at every node, and
/// the search seeds the root with a fresh [`Accumulator::refresh`] at the
/// start of every search.
fn accumulate(sums: &mut [f32], net: &SymNetwork, feature: FeatureIndex, sign: f32) {
    let row = &net.embed[feature as usize * DIM..(feature as usize + 1) * DIM];
    let base = group(feature) * ORDER * DIM;
    for (channel, &value) in row.iter().enumerate() {
        let mut power = value;
        for k in 0..ORDER {
            sums[base + k * DIM + channel] += sign * power;
            power *= value;
        }
    }
}

/// Which group of power sums a feature goes into: always the first, unless the
/// shape is split by owner, in which case the second takes the opponent's.
#[cfg(feature = "sym_token")]
#[inline]
fn group(feature: FeatureIndex) -> usize {
    if SPLIT { theirs(feature) } else { 0 }
}

/// The KP and HalfKP spaces have no `split`: their numbering says nothing about
/// whose a feature is.
#[cfg(not(feature = "sym_token"))]
#[inline]
fn group(_feature: FeatureIndex) -> usize {
    const { assert!(!SPLIT) };
    0
}
