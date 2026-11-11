//! The forward pass written a second way, in `f64`, from the definition.

use attic_state::Position;

use super::super::shape::SQUARES;
use super::super::{
    DIM, FEATURE_LEN, FIELD, HAS_FIELD, HAS_MOMENT, HAS_PAIR, MOMENT, ORDER, PAIR, SymNetwork,
    TOKENS,
};
use super::encode::column_of;
use super::fixtures::{tokens_present, upto};

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
/// out rather than computed. The second moment is built entry by entry from its
/// definition, `Σ_i v_i[a] · v_i[b]` for each `a ≤ b`, and the field slot by slot,
/// `Σ_i field[i][s·f + c]`, where the accumulator adds a token at a time.
///
/// The head is then the plain textbook form of the definition: each block is
/// `clamp(x · gamma, 0, 1)`, each hidden layer is `clamp(b + w · x, 0, 1)` and the
/// value is the row's output as it is.
pub(super) fn reference_value(net: &SymNetwork, pos: &Position) -> f64 {
    let mut features = Vec::with_capacity(FEATURE_LEN);
    for color in [pos.side_to_move(), pos.side_to_move().flip()] {
        let tokens = tokens_present(pos, color);
        // What a block becomes on its way to the first dense layer, with the gain
        // of its `i`-th entry given by `gamma`: multiplied by it and clipped.
        let put = |features: &mut Vec<f64>, block: &[f64], gamma: &dyn Fn(usize) -> f64| {
            features.extend(
                block
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (v * gamma(i)).clamp(0.0, 1.0)),
            );
        };

        let mut elementary = vec![[0.0f64; ORDER + 1]; DIM];
        for (channel, e) in elementary.iter_mut().enumerate() {
            e[0] = 1.0;
            for &token in &tokens {
                let v = f64::from(net.embed[token as usize * DIM + channel]);
                for k in (1..=ORDER).rev() {
                    e[k] += e[k - 1] * v;
                }
            }
        }
        for k in 1..=ORDER {
            let block: Vec<f64> = elementary.iter().map(|e| e[k]).collect();
            put(&mut features, &block, &|channel| {
                f64::from(net.gamma[(k - 1) * DIM + channel])
            });
        }

        if HAS_PAIR {
            let mut sum = [0.0f64; PAIR];
            for (i, &a) in tokens.iter().enumerate() {
                for &b in &tokens[i + 1..] {
                    let row = counted_pair_row(a, b);
                    for (channel, s) in sum.iter_mut().enumerate() {
                        *s += f64::from(net.pair.value(row, channel));
                    }
                }
            }
            put(&mut features, &sum, &|channel| {
                f64::from(net.pair_gamma[channel])
            });
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
                        .map(|&token| {
                            let row = &net.moment_embed[token as usize * MOMENT..][..MOMENT];
                            f64::from(row[a]) * f64::from(row[b])
                        })
                        .sum();
                    triangle.push(entry);
                }
            }
            // A gain for each entry, not each row or channel.
            put(&mut features, &triangle, &|entry| {
                f64::from(net.moment_gamma[entry])
            });
        }

        if HAS_FIELD {
            // Slot by slot, square-major: the 81 squares in turn, each of `f`
            // channels.
            let mut slots = Vec::new();
            for square in upto(SQUARES) {
                for channel in upto(FIELD) {
                    let sum: f64 = tokens
                        .iter()
                        .map(|&token| {
                            let row = &net.field[token as usize * SQUARES * FIELD..];
                            f64::from(row[square * FIELD + channel])
                        })
                        .sum();
                    slots.push(sum);
                }
            }
            // The gain is one per channel and every square shares it.
            let channel_of = column_of(FIELD);
            put(&mut features, &slots, &|slot| {
                f64::from(net.field_gamma[channel_of(slot)])
            });
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
                (f64::from(*b) + dot).clamp(0.0, 1.0)
            })
            .collect()
    };
    let head = dense(&net.w2, &net.b2, &dense(&net.w1, &net.b1, &features));
    let dot: f64 = net
        .value_w
        .iter()
        .zip(&head)
        .map(|(w, h)| f64::from(*w) * h)
        .sum();
    f64::from(net.value_b) + dot
}
