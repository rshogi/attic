//! The forward pass written a second way, in `f64`, from the definition.

use attic_state::Position;

use super::super::forward::{LSE_FLOOR, RMS_EPS};
use super::super::shape::SQUARES;
use super::super::{
    DIM, FEATURE_LEN, FIELD, GROUPS, HAS_FIELD, HAS_MOMENT, HAS_PAIR, KS, MEAN, MOMENT, ORDER,
    PAIR, SECOND_LAYER, SPLIT, SymNetwork, TOKENS,
};
use super::encode::column_of;
use super::fixtures::{owned_tokens, upto};

/// The row of the pair `{a, b}` counted out: the pairs with a lower token below
/// `lo` come first, `TOKENS − 1 − i` of them for each `i`, and `hi` is
/// `hi − lo − 1` into its own. The accumulator's formula is the closed form of
/// this sum; the two should never need to agree by construction.
fn counted_pair_row(a: u32, b: u32) -> usize {
    let (lo, hi) = (a.min(b) as usize, a.max(b) as usize);
    (0..lo).map(|i| TOKENS - 1 - i).sum::<usize>() + (hi - lo - 1)
}

/// The forward pass written a second way, in `f64`, from the definition.
///
/// The elementary symmetric polynomial `e_k` is the coefficient of `x^k` in
/// `∏ (1 + v_i x)`, so this builds it by multiplying out the product rather
/// than by Newton's identities from power sums — a different route to the same
/// number, which is what makes agreement mean something. The pair sum is taken
/// over every pair of tokens present, one at a time, at a row number counted
/// out rather than computed; the group of a split shape is read off the
/// position, not the token's number. The second moment is built entry by entry
/// from its definition, `Σ_i v_i[a] · v_i[b]` for each `a ≤ b`, and the field slot
/// by slot, `Σ_i field[i][s·f + c]`, where the accumulator adds a token at a time.
/// The norm, the hidden layers and the head are then the plain textbook forms.
pub(super) fn reference_value(net: &SymNetwork, pos: &Position) -> f64 {
    let mut features = Vec::with_capacity(FEATURE_LEN);
    for color in [pos.side_to_move(), pos.side_to_move().flip()] {
        let tokens = owned_tokens(pos, color);
        // The `r` of every norm, in the order the blocks come out, for `ks`.
        let mut kept = Vec::new();
        let rms = |block: &[f64]| {
            let mean_square = block.iter().map(|v| v * v).sum::<f64>() / block.len() as f64;
            (mean_square + f64::from(RMS_EPS)).sqrt()
        };

        for group in 0..GROUPS {
            let mut elementary = vec![[0.0f64; ORDER + 1]; DIM];
            for (channel, e) in elementary.iter_mut().enumerate() {
                e[0] = 1.0;
                for &(token, theirs) in &tokens {
                    // A split shape sums the frame's own pieces apart from the
                    // opponent's; every other shape has the one group.
                    if SPLIT && usize::from(theirs) != group {
                        continue;
                    }
                    let v = f64::from(net.embed[token as usize * DIM + channel]);
                    for k in (1..=ORDER).rev() {
                        e[k] += e[k - 1] * v;
                    }
                }
            }
            for k in 1..=ORDER {
                let block: Vec<f64> = elementary.iter().map(|e| e[k]).collect();
                let r = rms(&block);
                kept.push(r);
                for (channel, v) in block.iter().enumerate() {
                    let gamma = f64::from(net.gamma[(k - 1) * DIM + channel]);
                    features.push(v * (1.0 / r) * gamma);
                }
            }
        }

        if HAS_PAIR {
            let mut sum = [0.0f64; PAIR];
            for (i, &(a, _)) in tokens.iter().enumerate() {
                for &(b, _) in &tokens[i + 1..] {
                    let row = counted_pair_row(a, b);
                    for (channel, s) in sum.iter_mut().enumerate() {
                        *s += f64::from(net.pair[row * PAIR + channel]);
                    }
                }
            }
            let gamma = |channel: usize| f64::from(net.pair_gamma[channel]);
            if MEAN != 0 {
                let r = rms(&sum[..MEAN]);
                kept.push(r);
                features.extend((0..MEAN).map(|c| sum[c] * (1.0 / r) * gamma(c)));
            }
            features.extend((MEAN..PAIR).map(|c| sum[c].max(f64::from(LSE_FLOOR)).ln() * gamma(c)));
        }

        if HAS_MOMENT {
            // Entry by entry, the upper triangle row by row: (0,0) (0,1) …
            // (0,m−1) (1,1) … (m−1,m−1). The sum over tokens is taken for each
            // entry on its own.
            let mut triangle = Vec::new();
            for a in upto(MOMENT) {
                for b in a..MOMENT {
                    let entry: f64 = tokens
                        .iter()
                        .map(|&(token, _)| {
                            let row = &net.moment_embed[token as usize * MOMENT..][..MOMENT];
                            f64::from(row[a]) * f64::from(row[b])
                        })
                        .sum();
                    triangle.push(entry);
                }
            }
            let r = rms(&triangle);
            kept.push(r);
            // A gain for each entry, not each row or channel.
            features.extend(
                triangle
                    .iter()
                    .zip(&net.moment_gamma)
                    .map(|(m, g)| m * (1.0 / r) * f64::from(*g)),
            );
        }

        if HAS_FIELD {
            // Slot by slot, square-major: the 81 squares in turn, each of `f`
            // channels.
            let mut slots = Vec::new();
            for square in upto(SQUARES) {
                for channel in upto(FIELD) {
                    let sum: f64 = tokens
                        .iter()
                        .map(|&(token, _)| {
                            let row = &net.field[token as usize * SQUARES * FIELD..];
                            f64::from(row[square * FIELD + channel])
                        })
                        .sum();
                    slots.push(sum);
                }
            }
            let r = rms(&slots);
            kept.push(r);
            // The gain is one per channel and every square shares it.
            let channel_of = column_of(FIELD);
            features.extend(
                slots
                    .iter()
                    .enumerate()
                    .map(|(i, a)| a * (1.0 / r) * f64::from(net.field_gamma[channel_of(i)])),
            );
        }

        if KS {
            features.extend(
                kept.iter()
                    .zip(&net.scale_gamma)
                    .map(|(r, g)| r.ln() * f64::from(*g)),
            );
        }
    }

    let dense = |w: &[f32], b: &[f32], x: &[f64]| -> Vec<f64> {
        b.iter()
            .enumerate()
            .map(|(row, b)| {
                let dot: f64 = w[row * x.len()..(row + 1) * x.len()]
                    .iter()
                    .zip(x)
                    .map(|(w, x)| f64::from(*w) * x)
                    .sum();
                (f64::from(*b) + dot).max(0.0)
            })
            .collect()
    };
    let mut head = dense(&net.w1, &net.b1, &features);
    if SECOND_LAYER {
        head = dense(&net.w2, &net.b2, &head);
    }
    let dot: f64 = net
        .value_w
        .iter()
        .zip(&head)
        .map(|(w, h)| f64::from(*w) * h)
        .sum();
    (f64::from(net.value_b) + dot).tanh()
}
