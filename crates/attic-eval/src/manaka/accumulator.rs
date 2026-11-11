//! The accumulator the search threads through the tree, and what a derive needs.

use attic_state::{Color, Position};

#[cfg(not(feature = "sym_token"))]
use crate::features::{FeatureIndex, MoveDelta, active_features, active_features_into};
#[cfg(feature = "sym_token")]
use crate::tokens::{FeatureIndex, MAX_ACTIVE_FEATURES, MoveDelta, active_features};

use super::cache::FinnyCache;
#[cfg(feature = "sym_token")]
use super::field::touch_field;
#[cfg(feature = "sym_token")]
use super::moment::touch_moment;
#[cfg(feature = "sym_token")]
use super::pair::touch_pairs;
use super::{DIM, FIELD_LEN, MOMENT_LEN, ORDER, PAIR, POWER_LEN, SymNetwork};

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
/// Holds each perspective's power sums, and in the token shapes the tokens they
/// were built from.
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
/// ## How a derive gets from one position to the next
///
/// The search hands every derive the accumulator of the position's parent and
/// the `MoveDelta` of the move between them, and that names the (at most) two
/// features that leave and the two that arrive without looking at the board.
/// Every shape applies it to the parent's sums. The KP and HalfKP shapes used to
/// scan the new position and diff its list against one they held instead, which
/// found the same four or so features and cost a scan and two sorts per node,
/// more than the rest of the derive put together.
///
/// The one move a delta cannot describe is HalfKP's own-king move: every
/// feature of that half is crossed with the king's square, so all forty change.
/// That half is rebuilt from the position after the move, which is forty rows
/// where a diff of the two lists would have been seventy-six.
///
/// ## The token shapes' set of tokens
///
/// In the token shapes `active` is kept up to date by the same delta, so that
/// anything that needs the other tokens present — a pair table pairs each
/// changed token with them — has the set it needs at the point it needs it: a
/// derive takes each removed token out of `active` *before* it subtracts the
/// token's row, and puts each added one in *after* it adds the row.
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
/// where `f32` drifts, so the pair sum is accumulated in `f64`. A token's rows,
/// at most thirty-nine, are added up in `f32` first and go into it as one total
/// (`touch_pairs`), so the `f64` sum takes one rounding per token touched. The search seeds
/// the root with [`Accumulator::refresh`] every iteration, which is what keeps
/// the drift to a few plies' worth; see `tests/sym_network.rs` for how much that
/// is on a trained net.
#[derive(Debug, Clone)]
pub struct Accumulator {
    /// `[perspective][POWER_LEN]`, indexed by [`Color::index`]; within one
    /// perspective `[order][channel]`.
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
    /// The tokens present in each half, which a table over pairs of tokens is
    /// read against. The KP and HalfKP shapes have no such table and keep no
    /// list: the delta names what changes, and a half that is rebuilt is
    /// rebuilt from the position.
    #[cfg(feature = "sym_token")]
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
            #[cfg(feature = "sym_token")]
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
            let features = active_features(pos, color);
            accumulate(&ZERO_SUMS, &mut self.sums[i], &net.embed, &[], &features);
            self.pair[i].fill(0.0);
            self.moment[i].fill(0.0);
            self.field[i].fill(0.0);
            // Every pair once: each token against the ones before it.
            #[cfg(feature = "sym_token")]
            {
                for (at, &feature) in features.iter().enumerate() {
                    touch_pairs(&mut self.pair[i], net, feature, &features[..at], 1.0);
                    touch_moment(&mut self.moment[i], net, feature, 1.0);
                    touch_field(&mut self.field[i], net, feature, 1.0);
                }
                self.active[i] = features;
            }
        }
    }

    /// The power sums for one perspective, `[order][channel]`.
    pub fn perspective(&self, color: Color) -> &[f32] {
        &self.sums[color.index()]
    }

    /// The pair table's sum for one perspective, `[channel]`; empty without one.
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

/// KP and HalfKP: by the move's delta, from the parent's accumulator, with a
/// half rebuilt where the delta has none.
#[cfg(not(feature = "sym_token"))]
impl Accumulator {
    /// Derive the accumulator for the position after `mv` from `self`, the one
    /// before it.
    ///
    /// `pos` is the pre-move position and is left byte-for-byte unchanged: the
    /// do/undo happens inside, because a half that is rebuilt is rebuilt from
    /// the position after the move. Mirrors the NNUE accumulator's contract so
    /// that the search treats the two the same way.
    ///
    /// # Panics
    /// Panics if `pos` is missing either king, before or after `mv`.
    pub fn update_after_move(
        &self,
        net: &SymNetwork,
        pos: &mut Position,
        mv: attic_state::Move,
    ) -> Accumulator {
        let delta = MoveDelta::from_move(pos, mv);
        let undo = pos.do_move(mv);
        let mut next = Accumulator::new();
        // The one call site that has no cache to borrow from — the root, once
        // per iteration. The tree goes through `derive_into_cached`.
        let mut cache = FinnyCache::default();
        Self::derive_into_cached(self, &mut next, net, pos, &delta, &mut cache);
        pos.undo_move(mv, undo);
        next
    }

    /// Write into `dst` the accumulator of the position after the move `delta`
    /// describes, from `src`, which has to be the accumulator of the position the
    /// move was made from.
    ///
    /// Each half of `dst` is written from `src` in one pass, the parent's sums
    /// read, the move's rows taken out and put in, and the result stored: there is
    /// no copy of the parent first. Whatever `dst` held is overwritten.
    ///
    /// The delta is NNUE's, and the features it names are the ones the scan of a
    /// position would list: both read `features.rs`, which numbers K+P and HalfKP
    /// for every shape that uses them (its tests hold `MoveDelta::from_move` to
    /// the scan on every legal move of a spread of positions). A KP half always
    /// has one, two removed and two added at most. A HalfKP half has none when
    /// its own king moved, because every feature in it is crossed with that
    /// king's square: that half is summed again from the forty features of
    /// `post_pos`, through `cache`'s list so that the rebuild does not allocate.
    /// That is the only time `post_pos` and `cache` are read.
    pub fn derive_into_cached(
        src: &Accumulator,
        dst: &mut Accumulator,
        net: &SymNetwork,
        post_pos: &Position,
        delta: &MoveDelta,
        cache: &mut FinnyCache,
    ) {
        for color in [Color::Black, Color::White] {
            let i = color.index();
            match delta.half(color) {
                Some(half) => accumulate(
                    &src.sums[i],
                    &mut dst.sums[i],
                    &net.embed,
                    half.removed(),
                    half.added(),
                ),
                None => {
                    active_features_into(post_pos, color, &mut cache.features);
                    accumulate(
                        &ZERO_SUMS,
                        &mut dst.sums[i],
                        &net.embed,
                        &[],
                        &cache.features,
                    );
                }
            }
        }
    }
}

// The KP and HalfKP spaces have no pair table, second moment or square field
// (Manaka's shape grammar refuses them there), so the power sums are all a
// derive writes: the other three are empty in both accumulators.
#[cfg(not(feature = "sym_token"))]
const _: () = assert!(
    PAIR == 0 && MOMENT_LEN == 0 && FIELD_LEN == 0,
    "a KP or HalfKP shape carries nothing but its power sums"
);

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
        let mut next = Accumulator::new();
        Self::derive_into_cached(self, &mut next, net, pos, &delta, &mut FinnyCache);
        next
    }

    /// Write into `dst` the accumulator of the position after the move `delta`
    /// describes, from `src`, which has to be the accumulator of the position the
    /// move was made from. Whatever `dst` held is overwritten.
    ///
    /// The position and the cache go unread: a token's row does not depend on
    /// any other piece, so the delta is everything, and there is no bucket whose
    /// rebuild a cache could hold.
    ///
    /// The power sums are written from `src` in one pass, with no copy of the
    /// parent's first. The pair, moment and field sums and the set of tokens are
    /// copied and then changed, token by token: each removal leaves `active`
    /// before its row is subtracted and each addition enters it after its row is
    /// added, so at the moment a token's row is touched `active` is the set of
    /// *other* tokens present — what a table over pairs of tokens must read to
    /// subtract or add the right rows.
    ///
    /// # Panics
    /// Panics if a removed token is not present, which means `src` was not the
    /// accumulator of the position the move was made from.
    pub fn derive_into_cached(
        src: &Accumulator,
        dst: &mut Accumulator,
        net: &SymNetwork,
        _post_pos: &Position,
        delta: &MoveDelta,
        _cache: &mut FinnyCache,
    ) {
        for color in [Color::Black, Color::White] {
            let i = color.index();
            let half = delta.half(color).expect("a token delta is never a rebuild");
            // The power sums depend on nothing but the rows being taken out and put
            // in, and in what order, so they go in one pass ahead of the tables that
            // read `active`.
            accumulate(
                &src.sums[i],
                &mut dst.sums[i],
                &net.embed,
                half.removed(),
                half.added(),
            );
            dst.pair[i].copy_from_slice(&src.pair[i]);
            dst.moment[i].copy_from_slice(&src.moment[i]);
            dst.field[i].copy_from_slice(&src.field[i]);
            dst.active[i].clone_from(&src.active[i]);
            dst.touch_tokens(net, i, half.removed(), half.added());
        }
    }

    /// The pair, moment and field sums and the set of tokens of half `i`, taken
    /// from what they hold to the position after `removed` leave and `added`
    /// arrive. The power sums are not touched.
    fn touch_tokens(
        &mut self,
        net: &SymNetwork,
        i: usize,
        removed: &[FeatureIndex],
        added: &[FeatureIndex],
    ) {
        for &token in removed {
            let at = self.active[i]
                .iter()
                .position(|&present| present == token)
                .expect("a removed token was not in the accumulator it is removed from");
            self.active[i].swap_remove(at);
            touch_pairs(&mut self.pair[i], net, token, &self.active[i], -1.0);
            touch_moment(&mut self.moment[i], net, token, -1.0);
            touch_field(&mut self.field[i], net, token, -1.0);
        }
        for &token in added {
            touch_pairs(&mut self.pair[i], net, token, &self.active[i], 1.0);
            touch_moment(&mut self.moment[i], net, token, 1.0);
            touch_field(&mut self.field[i], net, token, 1.0);
            self.active[i].push(token);
        }
    }
}

/// What a half's power sums start from when they are built rather than derived.
static ZERO_SUMS: [f32; POWER_LEN] = [0.0; POWER_LEN];

/// Write into `dst` the power sums `src` holds with the powers of the `removed`
/// features' rows taken out and the `added` ones put in, in that order.
///
/// `src` and `dst` are separate, so a derive reads its parent's sums and writes
/// its own in the same pass and never copies the parent's first; a rebuild
/// passes [`ZERO_SUMS`] as `src`.
///
/// The subtraction is not exact: `add` then `sub` of the same row restores the
/// sum only up to f32 rounding. Every shape composes these deltas from the
/// parent's accumulator at every node (a HalfKP half whose king moved starts
/// again from zero), and the search seeds the root with a fresh
/// [`Accumulator::refresh`] at the start of every search, which is what bounds
/// the drift to a search's depth.
///
/// Eight channels at a time, each group's sums read once from `src`, taken
/// through every feature in turn and written once to `dst`, where doing a
/// feature at a time reads and writes all of the sums for each. A channel sees
/// what it always saw — for each feature in order, `sums_k += sign · v^k` and
/// then the power raised by one more multiplication by `v`, `k` from one to
/// `ORDER` — so the sums are the same bits as copying `src` and updating it in
/// place.
fn accumulate(
    src: &[f32],
    dst: &mut [f32],
    embed: &[f32],
    removed: &[FeatureIndex],
    added: &[FeatureIndex],
) {
    const LANES: usize = 8;
    const _: () = assert!(
        DIM.is_multiple_of(LANES),
        "channels come in whole groups of eight"
    );
    assert_eq!(src.len(), ORDER * DIM);
    assert_eq!(dst.len(), ORDER * DIM);
    let row = |feature: FeatureIndex, at: usize| -> [f32; LANES] {
        let base = feature as usize * DIM + at;
        embed[base..base + LANES]
            .try_into()
            .expect("a group of eight")
    };
    for at in (0..DIM).step_by(LANES) {
        let mut group = [[0.0f32; LANES]; ORDER];
        for (k, sum) in group.iter_mut().enumerate() {
            sum.copy_from_slice(&src[k * DIM + at..k * DIM + at + LANES]);
        }
        for (features, sign) in [(removed, -1.0f32), (added, 1.0f32)] {
            for &feature in features {
                let value = row(feature, at);
                let mut power = value;
                for sum in &mut group {
                    for lane in 0..LANES {
                        sum[lane] += sign * power[lane];
                    }
                    for lane in 0..LANES {
                        power[lane] *= value[lane];
                    }
                }
            }
        }
        for (k, sum) in group.iter().enumerate() {
            dst[k * DIM + at..k * DIM + at + LANES].copy_from_slice(sum);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One feature at a time over the whole row, with the order as the inner loop:
    /// the form the grouped pass replaced, kept to hold it to the same bits.
    fn one_at_a_time(sums: &mut [f32], embed: &[f32], feature: FeatureIndex, sign: f32) {
        let row = &embed[feature as usize * DIM..(feature as usize + 1) * DIM];
        for (channel, &value) in row.iter().enumerate() {
            let mut power = value;
            for k in 0..ORDER {
                sums[k * DIM + channel] += sign * power;
                power *= value;
            }
        }
    }

    #[test]
    fn a_batch_of_features_is_the_features_one_at_a_time_to_the_last_bit() {
        // A table of values of the size the embeddings have, spread either side of
        // zero; a feature is a row of it.
        let tokens = 64;
        let mut state = 11u64;
        let embed: Vec<f32> = (0..tokens * DIM)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 1.2
            })
            .collect();
        let mut pick = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as usize % tokens) as FeatureIndex
        };
        for round in 0..200 {
            let removed: Vec<FeatureIndex> = (0..round % 3).map(|_| pick()).collect();
            let added: Vec<FeatureIndex> = (0..round % 5).map(|_| pick()).collect();
            let start = vec![0.25f32; ORDER * DIM];
            let mut want = start.clone();
            // Whatever the destination held is overwritten, not added to.
            let mut got = vec![f32::NAN; ORDER * DIM];
            for &feature in &removed {
                one_at_a_time(&mut want, &embed, feature, -1.0);
            }
            for &feature in &added {
                one_at_a_time(&mut want, &embed, feature, 1.0);
            }
            accumulate(&start, &mut got, &embed, &removed, &added);
            for (at, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "round {round} entry {at}");
            }
        }
    }
}
