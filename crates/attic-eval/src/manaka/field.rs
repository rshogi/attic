//! What the square field needs from a token: its row of `field`, which the
//! accumulator adds to a sum of 81 slots of `FIELD` channels.
//!
//! A sum over tokens and nothing else, as the moment is: no other token's
//! presence changes what a token adds.

use crate::tokens::FeatureIndex;

use super::accumulator::FieldSum;
use super::{FIELD, FIELD_LEN, SymNetwork};

/// Add (`sign = 1`) or subtract (`sign = −1`) `token`'s row of the field into
/// `sums`.
///
/// The row is `[square][channel]` flattened, the entry for slot `s` and channel
/// `c` at `s · FIELD + c`, and `sums` is laid out the same way, so the two line up
/// element for element and nothing here needs to know about either index.
#[inline]
pub(super) fn touch_field(
    sums: &mut [FieldSum],
    net: &SymNetwork,
    token: FeatureIndex,
    sign: FieldSum,
) {
    if FIELD == 0 {
        return;
    }
    let sums: &mut [FieldSum; FIELD_LEN] = sums.try_into().expect("a field is FIELD_LEN long");
    let at = token as usize * FIELD_LEN;
    let row: &[f32; FIELD_LEN] = net.field[at..at + FIELD_LEN]
        .try_into()
        .expect("a field row is FIELD_LEN long");
    for (sum, &value) in sums.iter_mut().zip(row) {
        *sum += sign * FieldSum::from(value);
    }
}
