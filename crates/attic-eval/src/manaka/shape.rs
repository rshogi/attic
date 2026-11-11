//! The dimensions of the shape this build was compiled for.
//!
//! Nothing else in `manaka` names a width: everything is derived from the
//! constants here, so that a shape of other dimensions is a row in [`DIMS`] and
//! no edit anywhere else.

/// What the shapes differ in. Everything else this module sizes follows from
/// these numbers and the feature space.
struct Dims {
    /// The width of one perspective's embedding.
    dim: usize,
    /// Highest power the accumulator carries.
    order: usize,
    /// The first hidden layer's width.
    hidden: usize,
    /// The second hidden layer's width. NNUE's back half has two layers, and so
    /// does every shape here.
    hidden2: usize,
    /// Channels of the pair table, or 0 for a shape without one. It is a table
    /// of one vector per unordered pair of tokens, so it exists only in the
    /// token space, whose 2,344 rows make 2,745,996 pairs.
    pair: usize,
    /// Channels of the second moment's embedding, or 0 for a shape without one.
    /// The block it feeds is the upper triangle of `Σ v vᵀ`, `m(m+1)/2` long.
    moment: usize,
    /// Channels per square of the square field, or 0 for a shape without one.
    /// The block it feeds is 81 squares of that many channels.
    field: usize,
}

impl Dims {
    /// A shape with only the power sums.
    const fn new(dim: usize, order: usize, hidden: usize, hidden2: usize) -> Dims {
        Dims {
            dim,
            order,
            hidden,
            hidden2,
            pair: 0,
            moment: 0,
            field: 0,
        }
    }

    const fn pair(self, pair: usize) -> Dims {
        Dims { pair, ..self }
    }

    const fn moment(self, moment: usize) -> Dims {
        Dims { moment, ..self }
    }

    const fn field(self, field: usize) -> Dims {
        Dims { field, ..self }
    }
}

/// One row per shape, or per set of dimensions that the KP, HalfKP and plain
/// token shapes share. The feature space is not here: it is `sym_kp`,
/// `sym_halfkp` or `sym_token` and decides the token count, not these numbers.
/// The shapes with a pair table, a second moment or a square field have no KP or
/// HalfKP counterpart, so each has a row of its own, in the order Manaka's shape
/// code lists the knobs.
///
/// A chain of `cfg!` rather than a `#[cfg]` per row, so that selecting two
/// shapes at once is not a pile of duplicate-definition errors but the one
/// message `types.rs` asserts. The last arm is for `sym` enabled without a
/// shape, which that assertion reports as well.
const DIMS: Dims = if cfg!(any(
    feature = "synn_kp_128x2_32_32",
    feature = "synn_halfkp_128x2_32_32",
    feature = "synn_token_128x2_32_32"
)) {
    Dims::new(128, 3, 32, 32)
} else if cfg!(any(
    feature = "synn_kp_256x2_32_32",
    feature = "synn_token_256x2_32_32"
)) {
    Dims::new(256, 3, 32, 32)
} else if cfg!(feature = "synn_token_128x2_32_32_pair16") {
    Dims::new(128, 3, 32, 32).pair(16)
} else if cfg!(feature = "synn_token_128x2_32_32_pair16_m8_f2") {
    Dims::new(128, 3, 32, 32).pair(16).moment(8).field(2)
} else {
    panic!("a Manaka shape has to be selected: see the features of attic-eval")
};

/// The width of one perspective's embedding.
pub const DIM: usize = DIMS.dim;

/// Highest power the accumulator carries.
pub const ORDER: usize = DIMS.order;

/// The first hidden layer's width.
pub const HIDDEN: usize = DIMS.hidden;

/// The second hidden layer's width.
pub const HIDDEN2: usize = DIMS.hidden2;

/// What the value row reads: the last hidden layer, which is the second.
pub const HEAD: usize = HIDDEN2;

/// Rows of the embedding table.
///
/// KP is `fe_end + 2 · 81`. HalfKP is `81 · fe_end + 1` — the `+ 1` is the
/// padding row both kings point at, which YaneuraOu has no equivalent of;
/// see [`crate::features::king_feature`]. Manaka's own space is
/// `14 · 2 · 81 + 2 · 38`: see `tokens.rs`.
pub const TOKENS: usize = crate::types::NUM_FEATURES;

/// Channels of the pair table, or 0 when the shape has none.
pub const PAIR: usize = DIMS.pair;

/// Channels of the second moment's embedding, or 0 when the shape has none.
pub const MOMENT: usize = DIMS.moment;

/// Channels per square of the square field, or 0 when the shape has none.
pub const FIELD: usize = DIMS.field;

/// Whether the shape has a pair table.
///
/// A constant, not a field of the network, so that a shape without one tests a
/// number the compiler knows: the pair sum, its table and its block vanish from
/// that build. Spelled `!= 0` and not `> 0` because clippy rejects the latter on
/// the shapes where it is always false, which is the point of it being a
/// constant.
pub const HAS_PAIR: bool = PAIR != 0;

/// Rows of the pair table: one per unordered pair of distinct tokens, which is
/// `C(TOKENS, 2)`. Zero without a pair table, since in the KP and HalfKP spaces
/// the same formula would name 7.9e9 rows that nothing reads.
pub const PAIR_ROWS: usize = if HAS_PAIR {
    TOKENS * (TOKENS - 1) / 2
} else {
    0
};

/// Whether the shape has a second moment. Spelled `!= 0` for the reason
/// [`HAS_PAIR`] is.
pub const HAS_MOMENT: bool = MOMENT != 0;

/// Whether the shape has a square field.
pub const HAS_FIELD: bool = FIELD != 0;

/// The squares a field has a slice for: the board's. The reader never uses what
/// a slice means, only that there are 81 of them.
pub(super) const SQUARES: usize = attic_state::Square::COUNT;

/// Entries of the second moment: the upper triangle of an `MOMENT × MOMENT`
/// matrix, diagonal included, which is what the head reads. The rest is the same
/// numbers again, since `Σ v vᵀ` is symmetric.
pub const MOMENT_LEN: usize = MOMENT * (MOMENT + 1) / 2;

/// Entries of the square field: a slice of `FIELD` channels for each square.
pub const FIELD_LEN: usize = SQUARES * FIELD;

/// The power sums one perspective carries: every order.
pub const POWER_LEN: usize = ORDER * DIM;

/// One perspective's share of what the first hidden layer reads, laid out as
/// the power blocks, then the pair channels, the second moment and the square
/// field. A block that a shape lacks is zero long, so its offset is where the
/// next one starts.
pub const BLOCK_LEN: usize = POWER_LEN + PAIR + MOMENT_LEN + FIELD_LEN;

/// Where the pair channels start within a block.
pub const PAIR_AT: usize = POWER_LEN;

/// Where the second moment starts within a block.
pub const MOMENT_AT: usize = PAIR_AT + PAIR;

/// Where the square field starts within a block.
pub const FIELD_AT: usize = MOMENT_AT + MOMENT_LEN;

/// What the first hidden layer reads: both perspectives' blocks.
pub const FEATURE_LEN: usize = 2 * BLOCK_LEN;

// Exactly one shape, counted. `DIMS` above picks its row by testing the shapes in
// turn, so two at once would silently take the first. `sym` on its own, which no
// shape is, counts zero and fails here too, which is why the guard in `types.rs`
// can stand for every one of them with `sym`.
#[cfg(feature = "sym")]
const _: () = {
    macro_rules! selected {
        ($($shape:literal),+ $(,)?) => { 0 $(+ cfg!(feature = $shape) as usize)+ };
    }
    let selected = selected!(
        "synn_kp_128x2_32_32",
        "synn_kp_256x2_32_32",
        "synn_halfkp_128x2_32_32",
        "synn_token_128x2_32_32",
        "synn_token_256x2_32_32",
        "synn_token_128x2_32_32_pair16",
        "synn_token_128x2_32_32_pair16_m8_f2",
    );
    assert!(
        selected == 1,
        "select exactly one Manaka shape: they differ in their dimensions"
    );
};

const _: () = {
    // NNUE's back half has two dense layers, and the file layout, the architecture
    // string and the forward pass all assume both.
    assert!(
        HIDDEN2 != 0,
        "a Manaka shape has two hidden layers, as NNUE's back half does"
    );
    // Only the token space is small enough for a row per pair, and the loader
    // and the accumulator index the table by a token pair directly.
    assert!(
        !HAS_PAIR || cfg!(feature = "sym_token"),
        "a pair table exists only in the token space"
    );
    // The moment and the field want a row per token, as the embedding does, and
    // the other spaces have up to 125,389 of them: a field of eight channels per
    // square would be 81 million numbers in the HalfKP space. Manaka's shape
    // grammar refuses both outside the token space for that reason, and so does
    // this.
    assert!(
        !(HAS_MOMENT || HAS_FIELD) || cfg!(feature = "sym_token"),
        "a second moment or a square field exists only in the token space"
    );
};
