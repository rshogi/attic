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
    /// The second hidden layer's width, or 0 for a shape that has one layer.
    hidden2: usize,
    /// Channels of the pair table, or 0 for a shape without one. It is a table
    /// of one vector per unordered pair of tokens, so it exists only in the
    /// token space, whose 2,344 rows make 2,745,996 pairs.
    pair: usize,
    /// How many of the pair table's *last* channels are read as a log-sum-exp
    /// instead of a mean. Carved out of `pair`, not added to it.
    lse: usize,
    /// Channels of the second moment's embedding, or 0 for a shape without one.
    /// The block it feeds is the upper triangle of `Σ v vᵀ`, `m(m+1)/2` long.
    moment: usize,
    /// Channels per square of the square field, or 0 for a shape without one.
    /// The block it feeds is 81 squares of that many channels.
    field: usize,
    /// Whether each normalised block also reports the magnitude the norm
    /// divided away (`ks`, for kept scale).
    ks: bool,
    /// Whether the power sums are taken twice, once over the frame's own pieces
    /// and once over the opponent's (`split`).
    split: bool,
}

impl Dims {
    /// A shape with only the power sums, which is what all the older ones are.
    const fn new(dim: usize, order: usize, hidden: usize, hidden2: usize) -> Dims {
        Dims {
            dim,
            order,
            hidden,
            hidden2,
            pair: 0,
            lse: 0,
            moment: 0,
            field: 0,
            ks: false,
            split: false,
        }
    }

    const fn pair(self, pair: usize) -> Dims {
        Dims { pair, ..self }
    }

    const fn lse(self, lse: usize) -> Dims {
        Dims { lse, ..self }
    }

    const fn moment(self, moment: usize) -> Dims {
        Dims { moment, ..self }
    }

    const fn field(self, field: usize) -> Dims {
        Dims { field, ..self }
    }

    const fn ks(self) -> Dims {
        Dims { ks: true, ..self }
    }

    const fn split(self) -> Dims {
        Dims {
            split: true,
            ..self
        }
    }
}

/// One row per shape, or per set of dimensions that the KP, HalfKP and plain
/// token shapes share. The feature space is not here: it is `sym_kp`,
/// `sym_halfkp` or `sym_token` and decides the token count, not these numbers.
/// The shapes with a pair table have no KP or HalfKP counterpart, so each has a
/// row of its own, in the order Manaka's shape code lists the knobs.
///
/// A chain of `cfg!` rather than a `#[cfg]` per row, so that selecting two
/// shapes at once is not a pile of duplicate-definition errors but the one
/// message `types.rs` asserts. The last arm is for `sym` enabled without a
/// shape, which that assertion reports as well.
const DIMS: Dims = if cfg!(any(
    feature = "sym_kp_64x2_512",
    feature = "sym_halfkp_64x2_512",
    feature = "sym_token_64x2_512"
)) {
    Dims::new(64, 3, 512, 0)
} else if cfg!(any(
    feature = "sym_kp_128x2_512",
    feature = "sym_halfkp_128x2_512",
    feature = "sym_token_128x2_512"
)) {
    Dims::new(128, 3, 512, 0)
} else if cfg!(any(
    feature = "sym_kp_256x2_512",
    feature = "sym_halfkp_256x2_512",
    feature = "sym_token_256x2_512"
)) {
    Dims::new(256, 3, 512, 0)
} else if cfg!(any(
    feature = "sym_kp_256x2_o1_32_32",
    feature = "sym_halfkp_256x2_o1_32_32",
    feature = "sym_token_256x2_o1_32_32"
)) {
    Dims::new(256, 1, 32, 32)
} else if cfg!(feature = "sym_token_128x2_512_pair16") {
    Dims::new(128, 3, 512, 0).pair(16)
} else if cfg!(feature = "sym_token_128x2_512_pair32") {
    Dims::new(128, 3, 512, 0).pair(32)
} else if cfg!(feature = "sym_token_128x2_512_pair64") {
    Dims::new(128, 3, 512, 0).pair(64)
} else if cfg!(feature = "sym_token_64x2_512_pair16") {
    Dims::new(64, 3, 512, 0).pair(16)
} else if cfg!(feature = "sym_token_256x2_512_pair16") {
    Dims::new(256, 3, 512, 0).pair(16)
} else if cfg!(feature = "sym_token_128x2_o2_512_pair16") {
    Dims::new(128, 2, 512, 0).pair(16)
} else if cfg!(feature = "sym_token_128x2_o4_512_pair16") {
    Dims::new(128, 4, 512, 0).pair(16)
} else if cfg!(feature = "sym_token_128x2_1024_pair16") {
    Dims::new(128, 3, 1024, 0).pair(16)
} else if cfg!(feature = "sym_token_128x2_512_256_pair16") {
    Dims::new(128, 3, 512, 256).pair(16)
} else if cfg!(feature = "sym_token_128x2_512_512_pair16") {
    Dims::new(128, 3, 512, 512).pair(16)
} else if cfg!(feature = "sym_token_128x2_512_pair16_lse4") {
    Dims::new(128, 3, 512, 0).pair(16).lse(4)
} else if cfg!(feature = "sym_token_128x2_512_pair16_lse8") {
    Dims::new(128, 3, 512, 0).pair(16).lse(8)
} else if cfg!(feature = "sym_token_128x2_512_pair16_lse16") {
    Dims::new(128, 3, 512, 0).pair(16).lse(16)
} else if cfg!(feature = "sym_token_128x2_512_pair16_lse4_ks") {
    Dims::new(128, 3, 512, 0).pair(16).lse(4).ks()
} else if cfg!(feature = "sym_token_128x2_512_pair16_ks") {
    Dims::new(128, 3, 512, 0).pair(16).ks()
} else if cfg!(feature = "sym_token_128x2_512_pair16_split") {
    Dims::new(128, 3, 512, 0).pair(16).split()
} else if cfg!(feature = "sym_token_128x2_512_pair16_m16") {
    Dims::new(128, 3, 512, 0).pair(16).moment(16)
} else if cfg!(feature = "sym_token_128x2_512_pair16_m32") {
    Dims::new(128, 3, 512, 0).pair(16).moment(32)
} else if cfg!(feature = "sym_token_128x2_512_pair16_f2") {
    Dims::new(128, 3, 512, 0).pair(16).field(2)
} else if cfg!(feature = "sym_token_128x2_512_pair16_f4") {
    Dims::new(128, 3, 512, 0).pair(16).field(4)
} else if cfg!(feature = "sym_token_128x2_512_pair16_f8") {
    Dims::new(128, 3, 512, 0).pair(16).field(8)
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
/// see [`crate::features::king_feature`]. Manaka's own space is
/// `14 · 2 · 81 + 2 · 38`: see `tokens.rs`.
pub const TOKENS: usize = crate::types::NUM_FEATURES;

/// Channels of the pair table, or 0 when the shape has none.
pub const PAIR: usize = DIMS.pair;

/// How many of the pair table's last channels are read as a log-sum-exp.
pub const LSE: usize = DIMS.lse;

/// The pair channels read as a mean: normalised like a power block, and ahead of
/// the log-sum-exp ones in the table. A row that asked for more log-sum-exp
/// channels than there are channels would not subtract, and so does not compile.
pub const MEAN: usize = PAIR - LSE;

/// Channels of the second moment's embedding, or 0 when the shape has none.
pub const MOMENT: usize = DIMS.moment;

/// Channels per square of the square field, or 0 when the shape has none.
pub const FIELD: usize = DIMS.field;

/// Whether each normalised block also reports the magnitude the norm divided
/// away. A constant for the same reason [`SECOND_LAYER`] is.
pub const KS: bool = DIMS.ks;

/// Whether the power sums are taken separately over the frame's own pieces and
/// the opponent's.
pub const SPLIT: bool = DIMS.split;

/// Groups of power sums per perspective: two when split by owner, else one.
pub const GROUPS: usize = if SPLIT { 2 } else { 1 };

/// Whether the shape has a pair table. Spelled `!= 0` for the reason
/// [`SECOND_LAYER`] is.
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
/// [`SECOND_LAYER`] is.
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

/// The power sums one perspective carries: every order of every group.
pub const POWER_LEN: usize = GROUPS * ORDER * DIM;

/// Magnitudes the norms divided away that the head is told about, per
/// perspective: one for each normalised block, which is every order of every
/// group and the pair's mean channels, the moment and the field when there are
/// any. Zero without `ks`.
pub const SCALE_LEN: usize = if KS {
    GROUPS * ORDER + (MEAN != 0) as usize + HAS_MOMENT as usize + HAS_FIELD as usize
} else {
    0
};

/// One perspective's share of what the first hidden layer reads, laid out as
/// the power blocks, then the pair channels, the second moment, the square
/// field and the kept magnitudes. A block that a shape lacks is zero long, so its
/// offset is where the next one starts.
pub const BLOCK_LEN: usize = POWER_LEN + PAIR + MOMENT_LEN + FIELD_LEN + SCALE_LEN;

/// Where the pair channels start within a block.
pub const PAIR_AT: usize = POWER_LEN;

/// Where the second moment starts within a block.
pub const MOMENT_AT: usize = PAIR_AT + PAIR;

/// Where the square field starts within a block.
pub const FIELD_AT: usize = MOMENT_AT + MOMENT_LEN;

/// Where the kept magnitudes start within a block.
pub const SCALE_AT: usize = FIELD_AT + FIELD_LEN;

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
        "sym_kp_64x2_512",
        "sym_halfkp_64x2_512",
        "sym_kp_128x2_512",
        "sym_halfkp_128x2_512",
        "sym_kp_256x2_512",
        "sym_halfkp_256x2_512",
        "sym_kp_256x2_o1_32_32",
        "sym_halfkp_256x2_o1_32_32",
        "sym_token_64x2_512",
        "sym_token_128x2_512",
        "sym_token_256x2_512",
        "sym_token_256x2_o1_32_32",
        "sym_token_128x2_512_pair16",
        "sym_token_128x2_512_pair32",
        "sym_token_128x2_512_pair64",
        "sym_token_64x2_512_pair16",
        "sym_token_256x2_512_pair16",
        "sym_token_128x2_o2_512_pair16",
        "sym_token_128x2_o4_512_pair16",
        "sym_token_128x2_1024_pair16",
        "sym_token_128x2_512_256_pair16",
        "sym_token_128x2_512_512_pair16",
        "sym_token_128x2_512_pair16_lse4",
        "sym_token_128x2_512_pair16_lse8",
        "sym_token_128x2_512_pair16_lse16",
        "sym_token_128x2_512_pair16_lse4_ks",
        "sym_token_128x2_512_pair16_ks",
        "sym_token_128x2_512_pair16_split",
        "sym_token_128x2_512_pair16_m16",
        "sym_token_128x2_512_pair16_m32",
        "sym_token_128x2_512_pair16_f2",
        "sym_token_128x2_512_pair16_f4",
        "sym_token_128x2_512_pair16_f8",
    );
    assert!(
        selected == 1,
        "select exactly one Manaka shape: they differ in their dimensions"
    );
};

const _: () = {
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
