//! The dimensions of the shape this build was compiled for.
//!
//! Nothing else in `manaka` names a width: everything is derived from the
//! constants here, so that a shape of other dimensions is a row in [`DIMS`] and
//! no edit anywhere else.

/// What the shapes differ in. Everything else this module sizes follows from
/// these four numbers and the feature space.
struct Dims {
    /// The width of one perspective's embedding.
    dim: usize,
    /// Highest power the accumulator carries.
    order: usize,
    /// The first hidden layer's width.
    hidden: usize,
    /// The second hidden layer's width, or 0 for a shape that has one layer.
    hidden2: usize,
}

/// One row per set of dimensions, shared by the KP and HalfKP shape that carry
/// them. The feature space is not here: it is `sym_kp` or `sym_halfkp` and
/// decides the token count, not these four numbers.
///
/// A chain of `cfg!` rather than a `#[cfg]` per row, so that selecting two
/// shapes at once is not a pile of duplicate-definition errors but the one
/// message `types.rs` asserts. The last arm is for `sym` enabled without a
/// shape, which that assertion reports as well.
const DIMS: Dims = if cfg!(any(
    feature = "sym_kp_64x2_512",
    feature = "sym_halfkp_64x2_512"
)) {
    Dims {
        dim: 64,
        order: 3,
        hidden: 512,
        hidden2: 0,
    }
} else if cfg!(any(
    feature = "sym_kp_128x2_512",
    feature = "sym_halfkp_128x2_512"
)) {
    Dims {
        dim: 128,
        order: 3,
        hidden: 512,
        hidden2: 0,
    }
} else if cfg!(any(
    feature = "sym_kp_256x2_512",
    feature = "sym_halfkp_256x2_512"
)) {
    Dims {
        dim: 256,
        order: 3,
        hidden: 512,
        hidden2: 0,
    }
} else if cfg!(any(
    feature = "sym_kp_256x2_o1_32_32",
    feature = "sym_halfkp_256x2_o1_32_32"
)) {
    Dims {
        dim: 256,
        order: 1,
        hidden: 32,
        hidden2: 32,
    }
} else {
    panic!("a Manaka shape has to be selected: see the features of attic-eval")
};

/// The width of one perspective's embedding.
pub const DIM: usize = DIMS.dim;

/// Highest power the accumulator carries.
pub const ORDER: usize = DIMS.order;

/// The first hidden layer's width.
pub const HIDDEN: usize = DIMS.hidden;

/// The second hidden layer's width, or 0 when the shape has only one.
pub const HIDDEN2: usize = DIMS.hidden2;

/// Whether the shape has a second hidden layer.
///
/// A constant, not a field of the network, so that a one-layer shape tests a
/// number the compiler knows: the second layer's loop, its stack array (zero
/// length) and its tables vanish from those builds, and their forward pass is
/// the one they had before the layer existed. Spelled `!= 0` and not `> 0`
/// because clippy rejects the latter on the shapes where it is always false,
/// which is the point of it being a constant.
pub const SECOND_LAYER: bool = HIDDEN2 != 0;

/// What the value row reads: the last hidden layer there is.
pub const HEAD: usize = if SECOND_LAYER { HIDDEN2 } else { HIDDEN };

/// Rows of the embedding table.
///
/// KP is `fe_end + 2 · 81`. HalfKP is `81 · fe_end + 1` — the `+ 1` is the
/// padding row both kings point at, which YaneuraOu has no equivalent of;
/// see [`crate::features::king_feature`].
pub const TOKENS: usize = crate::types::NUM_FEATURES;

/// What the first hidden layer reads: both perspectives, every order.
pub const FEATURE_LEN: usize = 2 * ORDER * DIM;
