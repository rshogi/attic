//! Which of Sirius's eight layer stacks a position reads, and how a position near a
//! boundary between two of them is blended.
//!
//! Sirius cuts the game into eight phases by an integer game-phase sum `S` (the pieces in
//! the hands and the ones standing in the enemy camp, each kind weighted) compared against
//! a boundary that depends on how far the two kings have advanced and how well they are
//! defended (`code`). Within [`BLEND_WIDTH`] of a boundary the score is a linear mix of
//! the two stacks on either side of it.
//!
//! The boundaries are a table of 105 rows (`code` runs 0 to 104) of seven bounds. Sirius
//! derives it when it starts, by bisecting a floating-point bucket function for the first
//! `S` that lands in the next bucket; the table below is that result. It is data and not
//! recomputed here so that the bucket does not depend on the last bit of a `libm`; the test
//! `table_is_what_the_published_bucket_function_gives` recomputes it with this crate's own
//! `f64::exp` and `f64::ln` and compares, which is what ties the numbers to their source.

use attic_state::bitboard::{file_mask, promotion_zone, rank_mask};
use attic_state::{Bitboard, Color, PieceKind, Position};

/// Hand-piece weights in `S`, indexed by [`HAND_ORDER`]: pawn, lance, knight, silver, gold,
/// bishop, rook.
const HAND_WEIGHT: [i32; 7] = [
    2_759_090, 7_656_785, 5_700_977, 8_609_847, 10_791_395, 12_620_730, 21_694_945,
];
const HAND_ORDER: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

/// Weights of a piece standing in the enemy camp, by [`camp_slot`]: pawn, lance, knight,
/// silver, gold, bishop, rook, then the promoted pawn, lance, knight, silver, bishop, rook.
const CAMP_WEIGHT: [i32; 13] = [
    6_237_398, 5_260_384, 5_575_300, 5_797_877, 6_452_488, 4_764_672, 6_296_131, 2_517_497,
    5_300_307, 4_073_846, 4_556_510, 2_647_140, 6_513_183,
];

/// Half the width of the zone in which two stacks are mixed, in units of `S`
/// (`Sirius_GPB_W = 111`, fixed in Sirius's build).
const BLEND_WIDTH: i64 = 0x2485E8;
/// `ceil(2^64 / BLEND_WIDTH)`: the reciprocal Sirius multiplies by instead of dividing.
const BLEND_RECIPROCAL: u128 = 0x702_5F65_9ABD;
/// The blend weight's scale: `t` runs 0 to 512.
const BLEND_ONE: i32 = 512;

const STACKS: usize = 8;
const CODES: usize = 105;

/// `GP_KING_IB[code][i]` is the smallest `S` that leaves stack `i` for stack `i + 1`; the
/// bound that does not exist for stack seven is [`NO_BOUND`] throughout, and so is any
/// bound too high for `S` to reach.
#[rustfmt::skip]
static GP_KING_IB: [[i32; STACKS - 1]; CODES] = [
    [20898025, 36997031, 53517348, 70258782, 90348272, 121789394, 2147483647],
    [11576188, 27675193, 44195511, 60936945, 81026434, 112467556, 2147483647],
    [2254350, 18353356, 34873673, 51615107, 71704596, 103145719, 2147483647],
    [-7067488, 9031518, 25551835, 42293269, 62382759, 93823881, 2147483647],
    [-16389326, -290320, 16229997, 32971431, 53060921, 84502043, 2147483647],
    [-26441582, -11125219, 3775092, 17340159, 30323703, 41770785, 2147483647],
    [-36557882, -22033754, -8605188, 2596753, 11945269, 18750410, 2147483647],
    [-46645950, -32833497, -20605075, -11053752, -3735008, 1117920, 2147483647],
    [-56707722, -43538413, -32308810, -23978910, -17961086, -14187774, 2147483647],
    [-66744945, -54159942, -43775421, -36387278, -31275861, -28188554, 2147483647],
    [-76759194, -64707569, -55047874, -48408668, -43965516, -41352891, 2147483647],
    [-86751900, -75189248, -66158423, -60129393, -56199547, -53934971, 2147483647],
    [-96724363, -85611714, -77131896, -71609767, -68086670, -66088255, 93626133],
    [-106677769, -95980721, -87987815, -82893596, -79700808, -77912536, -30336700],
    [-116613203, -106301223, -98741809, -94013705, -91094511, -89476374, -54552385],
    [-126531658, -116577514, -109406581, -104995325, -102306482, -100828906, -72599874],
    [-136434050, -126813344, -119992599, -115858270, -113366045, -112006556, -88116363],
    [-146321221, -137012003, -130508590, -126618384, -124295956, -123037071, -102244699],
    [-156193948, -147176399, -140961908, -137288536, -135114226, -133942077, -115494530],
    [-166052952, -157309107, -151358803, -147879321, -145835355, -144738758, -128137904],
    [-175898899, -167412426, -161704637, -158399564, -156471189, -155440993, -140337281],
    [-185732410, -177488411, -172004042, -168856698, -167031532, -166060153, -152197486],
    [-195554060, -187538910, -182261044, -179257039, -177524589, -176605673, -163790110],
    [-205364387, -197565589, -192479171, -189605998, -187957296, -187085466, -175166227],
    [-215163893, -207569958, -202661526, -199908249, -198335569, -197506233, -186363551],
    [-224953046, -217553384, -212810856, -210167857, -208664494, -207873702, -197410728],
    [-234732287, -227517115, -222929601, -220388377, -218948476, -218192805, -208330027],
    [-244502027, -237462289, -233019942, -230572938, -229191356, -228467818, -219139104],
    [-254262653, -247389951, -243083834, -240724306, -239396502, -238702476, -229852191],
    [-264014527, -257301055, -253123038, -250844942, -249566885, -248900058, -240480933],
    [-273757993, -267196485, -263139144, -260937042, -259705138, -259063457, -251034973],
    [-283493373, -277077051, -273133592, -271002574, -269813605, -269195243, -261522392],
    [-293220971, -286943506, -283107694, -281043310, -279894383, -279297705, -271950025],
    [-302941075, -296796544, -293062644, -291060851, -289949356, -289372892, -282323709],
    [-312653956, -306636810, -302999534, -301056646, -299980221, -299422647, -292648463],
    [-322359872, -316464904, -312919367, -311032014, -309988513, -309448629, -302928637],
    [-332059067, -326281385, -322823064, -320988156, -319975625, -319452344, -313168027],
    [-341751771, -336086774, -332711471, -330926173, -329942825, -329435156, -323369963],
    [-351438205, -345881556, -342585373, -340847070, -339891271, -339398309, -333537383],
    [-360780738, -355230169, -351938404, -350202718, -349248450, -348756305, -342906660],
    [-370102576, -364552007, -361260242, -359524556, -358570288, -358078143, -352228497],
    [-379424414, -373873845, -370582080, -368846394, -367892126, -367399980, -361550335],
    [-388746251, -383195682, -379903917, -378168231, -377213964, -376721818, -370872173],
    [-398068089, -392517520, -389225755, -387490069, -386535801, -386043656, -380194010],
    [-407389927, -401839358, -398547593, -396811907, -395857639, -395365493, -389515848],
    [-416711765, -411161195, -407869431, -406133745, -405179477, -404687331, -398837686],
    [-426033602, -420483033, -417191268, -415455582, -414501314, -414009169, -408159523],
    [-435355440, -429804871, -426513106, -424777420, -423823152, -423331007, -417481361],
    [-444677278, -439126709, -435834944, -434099258, -433144990, -432652844, -426803199],
    [-453999115, -448448546, -445156781, -443421095, -442466828, -441974682, -436125037],
    [-463320953, -457770384, -454478619, -452742933, -451788665, -451296520, -445446874],
    [-472642791, -467092222, -463800457, -462064771, -461110503, -460618357, -454768712],
    [-481964629, -476414059, -473122294, -471386608, -470432341, -469940195, -464090550],
    [-491286466, -485735897, -482444132, -480708446, -479754178, -479262033, -473412387],
    [-500608304, -495057735, -491765970, -490030284, -489076016, -488583870, -482734225],
    [-509930142, -504379572, -501087808, -499352122, -498397854, -497905708, -492056063],
    [-519251979, -513701410, -510409645, -508673959, -507719692, -507227546, -501377901],
    [-528573817, -523023248, -519731483, -517995797, -517041529, -516549384, -510699738],
    [-537895655, -532345086, -529053321, -527317635, -526363367, -525871221, -520021576],
    [-547217493, -541666923, -538375158, -536639472, -535685205, -535193059, -529343414],
    [-556539330, -550988761, -547696996, -545961310, -545007042, -544514897, -538665251],
    [-565861168, -560310599, -557018834, -555283148, -554328880, -553836734, -547987089],
    [-575183006, -569632436, -566340672, -564604986, -563650718, -563158572, -557308927],
    [-584504843, -578954274, -575662509, -573926823, -572972556, -572480410, -566630765],
    [-593826681, -588276112, -584984347, -583248661, -582294393, -581802248, -575952602],
    [-603148519, -597597950, -594306185, -592570499, -591616231, -591124085, -585274440],
    [-612470357, -606919787, -603628022, -601892336, -600938069, -600445923, -594596278],
    [-621792194, -616241625, -612949860, -611214174, -610259906, -609767761, -603918115],
    [-631114032, -625563463, -622271698, -620536012, -619581744, -619089598, -613239953],
    [-640435870, -634885300, -631593536, -629857850, -628903582, -628411436, -622561791],
    [-649757707, -644207138, -640915373, -639179687, -638225419, -637733274, -631883629],
    [-659079545, -653528976, -650237211, -648501525, -647547257, -647055112, -641205466],
    [-668401383, -662850814, -659559049, -657823363, -656869095, -656376949, -650527304],
    [-677723220, -672172651, -668880886, -667145200, -666190933, -665698787, -659849142],
    [-687045058, -681494489, -678202724, -676467038, -675512770, -675020625, -669170979],
    [-696366896, -690816327, -687524562, -685788876, -684834608, -684342462, -678492817],
    [-705688734, -700138164, -696846400, -695110714, -694156446, -693664300, -687814655],
    [-715010571, -709460002, -706168237, -704432551, -703478283, -702986138, -697136493],
    [-724332409, -718781840, -715490075, -713754389, -712800121, -712307976, -706458330],
    [-733654247, -728103678, -724811913, -723076227, -722121959, -721629813, -715780168],
    [-742976084, -737425515, -734133750, -732398064, -731443797, -730951651, -725102006],
    [-752297922, -746747353, -743455588, -741719902, -740765634, -740273489, -734423843],
    [-761619760, -756069191, -752777426, -751041740, -750087472, -749595326, -743745681],
    [-770941598, -765391028, -762099264, -760363578, -759409310, -758917164, -753067519],
    [-780263435, -774712866, -771421101, -769685415, -768731147, -768239002, -762389357],
    [-789585273, -784034704, -780742939, -779007253, -778052985, -777560840, -771711194],
    [-798907111, -793356542, -790064777, -788329091, -787374823, -786882677, -781033032],
    [-808228948, -802678379, -799386614, -797650928, -796696661, -796204515, -790354870],
    [-817550786, -812000217, -808708452, -806972766, -806018498, -805526353, -799676707],
    [-826872624, -821322055, -818030290, -816294604, -815340336, -814848190, -808998545],
    [-836194462, -830643892, -827352127, -825616441, -824662174, -824170028, -818320383],
    [-845516299, -839965730, -836673965, -834938279, -833984011, -833491866, -827642220],
    [-854838137, -849287568, -845995803, -844260117, -843305849, -842813703, -836964058],
    [-864159975, -858609405, -855317641, -853581955, -852627687, -852135541, -846285896],
    [-873481812, -867931243, -864639478, -862903792, -861949525, -861457379, -855607734],
    [-882803650, -877253081, -873961316, -872225630, -871271362, -870779217, -864929571],
    [-892125488, -886574919, -883283154, -881547468, -880593200, -880101054, -874251409],
    [-901447326, -895896756, -892604991, -890869305, -889915038, -889422892, -883573247],
    [-910769163, -905218594, -901926829, -900191143, -899236875, -898744730, -892895084],
    [-920091001, -914540432, -911248667, -909512981, -908558713, -908066567, -902216922],
    [-929412839, -923862269, -920570505, -918834819, -917880551, -917388405, -911538760],
    [-938734676, -933184107, -929892342, -928156656, -927202389, -926710243, -920860598],
    [-948056514, -942505945, -939214180, -937478494, -936524226, -936032081, -930182435],
    [-957378352, -951827783, -948536018, -946800332, -945846064, -945353918, -939504273],
    [-966700190, -961149620, -957857855, -956122169, -955167902, -954675756, -948826111],
];

const NO_BOUND: i32 = i32::MAX;

/// What [`choose`] decided for a position: the stack it reads, and the neighbouring stack
/// to mix in with its weight, if it is near enough to a boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StackChoice {
    /// The layer stack the position belongs to.
    pub stack: usize,
    /// The other stack and the weight `t` (0 to 511) of `stack`'s score, out of 512, when
    /// the position is in a blend zone.
    pub blend: Option<(usize, i32)>,
}

/// Index into [`CAMP_WEIGHT`] of a board piece, or `None` for a king.
fn camp_slot(kind: PieceKind, promoted: bool) -> Option<usize> {
    Some(match (kind, promoted) {
        (PieceKind::Pawn, false) => 0,
        (PieceKind::Lance, false) => 1,
        (PieceKind::Knight, false) => 2,
        (PieceKind::Silver, false) => 3,
        (PieceKind::Gold, _) => 4,
        (PieceKind::Bishop, false) => 5,
        (PieceKind::Rook, false) => 6,
        (PieceKind::Pawn, true) => 7,
        (PieceKind::Lance, true) => 8,
        (PieceKind::Knight, true) => 9,
        (PieceKind::Silver, true) => 10,
        (PieceKind::Bishop, true) => 11,
        (PieceKind::Rook, true) => 12,
        (PieceKind::King, _) => return None,
    })
}

/// The game-phase sum `S`: for each side, its hand pieces and the pieces it has in the
/// enemy camp (ranks 1 to 3 for Black, 7 to 9 for White), by weight. Wraps like the
/// `i32` it is in Sirius.
pub fn phase_sum(pos: &Position) -> i32 {
    let board = pos.board();
    let mut sum = 0i32;
    for color in [Color::Black, Color::White] {
        let hand = pos.hand(color);
        for (kind, weight) in HAND_ORDER.iter().zip(HAND_WEIGHT) {
            sum = sum.wrapping_add((hand.count(*kind) as i32).wrapping_mul(weight));
        }
        for sq in (board.pieces_color(color) & promotion_zone(color)).squares() {
            let Some(piece) = board.get(sq) else { continue };
            if let Some(slot) = camp_slot(piece.kind, piece.promoted) {
                sum = sum.wrapping_add(CAMP_WEIGHT[slot]);
            }
        }
    }
    sum
}

/// The king code: for each side whose king has advanced (`adv >= 0`), `2 * adv + 1` times
/// one more than the number of its own pieces within two squares of the king, no further
/// back than one rank behind it, capped at three.
pub fn king_code(pos: &Position) -> usize {
    let board = pos.board();
    let mut code = 0usize;
    for color in [Color::Black, Color::White] {
        let king = pos
            .king_square(color)
            .unwrap_or_else(|| panic!("position has no {color:?} king"));
        let (file, rank) = (king.file() as i32, king.rank() as i32);
        let adv = match color {
            Color::Black => 6 - rank,
            Color::White => rank - 2,
        };
        if adv < 0 {
            continue;
        }
        // Two ranks ahead of the king to one behind it, on the files within two of it.
        let (low, high) = match color {
            Color::Black => (rank - 2, rank + 1),
            Color::White => (rank - 1, rank + 2),
        };
        let mut files = Bitboard::EMPTY;
        for f in (file - 2).max(0)..=(file + 2).min(8) {
            files |= file_mask(f as u8);
        }
        let mut ranks = Bitboard::EMPTY;
        for r in low.max(0)..=high.min(8) {
            ranks |= rank_mask(r as u8);
        }
        // The king stands in its own box and is not a defender.
        let support = (board.pieces_color(color) & files & ranks).popcount() as usize - 1;
        code += (2 * adv as usize + 1) * (support.min(3) + 1);
    }
    code
}

/// The stack `sum` falls in for `code`: how many bounds it has reached.
fn bucket(sum: i32, code: usize) -> usize {
    GP_KING_IB[code]
        .iter()
        .filter(|&&bound| sum >= bound)
        .count()
}

/// Choose the stack for `pos`, and the neighbour to blend it with.
///
/// # Panics
/// Panics if `pos` is missing either king.
pub fn choose(pos: &Position) -> StackChoice {
    choose_from(phase_sum(pos), king_code(pos))
}

/// [`choose`] from `S` and the king code, which a test may set freely.
pub fn choose_from(sum: i32, code: usize) -> StackChoice {
    let stack = bucket(sum, code);
    let bounds = &GP_KING_IB[code];

    // The distance to the nearer bound, and the stack across it. Above first, then below
    // only when strictly nearer, as Sirius orders them.
    let mut distance = BLEND_WIDTH;
    let mut neighbour = stack;
    if stack < STACKS - 1 && bounds[stack] != NO_BOUND {
        let up = bounds[stack] as i64 - sum as i64;
        if up < BLEND_WIDTH {
            distance = up;
            neighbour = stack + 1;
        }
    }
    if stack > 0 {
        let down = sum as i64 - bounds[stack - 1] as i64;
        if down < distance {
            distance = down;
            neighbour = stack - 1;
        }
    }

    // `distance` is compared as unsigned in Sirius; it is never negative here (`sum` is
    // below the bound above it and at or above the one below).
    if (distance as u64) < BLEND_WIDTH as u64 {
        let t = ((BLEND_RECIPROCAL * ((distance as u64) << 9) as u128) >> 64) as i32;
        if t != BLEND_ONE {
            return StackChoice {
                stack,
                blend: Some((neighbour, t)),
            };
        }
    }
    StackChoice { stack, blend: None }
}

/// The score of a position in a blend zone: `raw` moved toward `other` by the share of the
/// difference that `t` leaves. The difference is multiplied before it is divided, and the
/// division by 1024 rounds toward zero.
pub fn blend(raw: i32, other: i32, t: i32) -> i32 {
    let product = other.wrapping_sub(raw) as i64 * (BLEND_ONE - t) as i64;
    let rounded = if product < 0 {
        product + 0x3FF
    } else {
        product
    };
    ((rounded >> 10) as i32).wrapping_add(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use attic_state::{Square, parse_sfen};

    const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

    /// [`phase_sum`] square by square over the camp, whoever stands there.
    fn phase_sum_by_squares(pos: &Position) -> i32 {
        let board = pos.board();
        let mut sum = 0i32;
        for color in [Color::Black, Color::White] {
            let hand = pos.hand(color);
            for (kind, weight) in HAND_ORDER.iter().zip(HAND_WEIGHT) {
                sum = sum.wrapping_add((hand.count(*kind) as i32).wrapping_mul(weight));
            }
            for sq in (board.occupied() & promotion_zone(color)).squares() {
                let Some(piece) = board.get(sq) else { continue };
                if piece.color != color {
                    continue;
                }
                if let Some(slot) = camp_slot(piece.kind, piece.promoted) {
                    sum = sum.wrapping_add(CAMP_WEIGHT[slot]);
                }
            }
        }
        sum
    }

    /// [`king_code`] square by square over the box around each king.
    fn king_code_by_squares(pos: &Position) -> usize {
        let board = pos.board();
        let mut code = 0usize;
        for color in [Color::Black, Color::White] {
            let king = pos.king_square(color).unwrap();
            let (file, rank) = (king.file() as i32, king.rank() as i32);
            let adv = match color {
                Color::Black => 6 - rank,
                Color::White => rank - 2,
            };
            if adv < 0 {
                continue;
            }
            let mut support = 0usize;
            for f in (file - 2).max(0)..=(file + 2).min(8) {
                for r in (rank - 2).max(0)..=(rank + 2).min(8) {
                    let behind_ok = match color {
                        Color::Black => r <= rank + 1,
                        Color::White => r >= rank - 1,
                    };
                    if !behind_ok || (f == file && r == rank) {
                        continue;
                    }
                    let sq = Square::new(f as u8, r as u8).unwrap();
                    if board.get(sq).is_some_and(|p| p.color == color) {
                        support += 1;
                    }
                }
            }
            code += (2 * adv as usize + 1) * (support.min(3) + 1);
        }
        code
    }

    #[test]
    fn the_piece_set_forms_match_the_square_scans() {
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let (mut in_camp, mut advanced) = (0, 0);
        for _ in 0..40 {
            let mut pos = parse_sfen(STARTPOS).unwrap();
            for _ in 0..200 {
                let sum = phase_sum(&pos);
                assert_eq!(sum, phase_sum_by_squares(&pos), "{pos:?}");
                let code = king_code(&pos);
                assert_eq!(code, king_code_by_squares(&pos), "{pos:?}");
                in_camp += usize::from(sum != 0);
                advanced += usize::from(code != 0);
                let mut moves = Vec::new();
                pos.generate_legal_all(&mut moves);
                if moves.is_empty() {
                    break;
                }
                let mv = moves[(next() % moves.len() as u64) as usize];
                let _ = pos.do_move(mv);
            }
        }
        assert!(in_camp > 1000 && advanced > 1000, "{in_camp} {advanced}");
    }

    /// The bound function Sirius bisects, with the constants of the stack bounds. It is a
    /// transcription of `SPEC.md` section 8 and uses `f64::exp` / `f64::ln`.
    fn reference_bucket(x: f64, offset: f64, bonus: f64, s_bounds: &[f64; 8]) -> usize {
        const BOUNDS: [f64; 7] = [
            0.620586, 0.820163, 0.916409, 0.961542, 0.984851, 0.996475, 1.116108,
        ];
        let top = STACKS - 1;
        if bonus == 0.0 {
            let s = offset + x;
            if 12.853000000000002 <= s {
                let mut core = 1.0;
                if -10.0 <= s / -1.2853 {
                    core = 1.0 - (s / -1.2853).exp();
                }
                let v = (core + bonus).clamp(0.0, 1.5);
                return (0..top).find(|&i| v < BOUNDS[i]).unwrap_or(top);
            }
            return (0..top).find(|&i| s < s_bounds[i]).unwrap_or(top);
        }
        let t = (offset + x) / -1.2853;
        let mut core = 1.0;
        if -10.0 <= t {
            core = 1.0 - t.exp();
        }
        let v = (core + bonus).clamp(0.0, 1.5);
        (0..top).find(|&i| v < BOUNDS[i]).unwrap_or(top)
    }

    /// The order-preserving map from a double to a `u64`, and back.
    fn from_ordered(k: u64) -> f64 {
        let bits = if k >> 63 != 0 { k ^ (1 << 63) } else { !k };
        f64::from_bits(bits)
    }

    fn recompute_table() -> Vec<[i32; STACKS - 1]> {
        const BOUNDS: [f64; 7] = [
            0.620586, 0.820163, 0.916409, 0.961542, 0.984851, 0.996475, 1.116108,
        ];
        let mut s_bounds = [0.0f64; 8];
        for i in 0..8 {
            s_bounds[i] = if i >= STACKS - 1 || BOUNDS[i] >= 1.0 {
                1e300
            } else {
                (1.0 - BOUNDS[i]).ln() * -1.2853
            };
        }
        (0..CODES)
            .map(|code| {
                let king = code as f64 * 0.125;
                let offset = (king * 0.540983) / 0.121706;
                let bonus = if 0.51407 < king {
                    ((king + -0.51407) / 4.2430889999999994).min(1.0) * 0.5
                } else {
                    0.0
                };
                let mut row = [0i32; STACKS - 1];
                for (b, slot) in row.iter_mut().enumerate() {
                    let (mut low, mut high) = (0x000f_ffff_ffff_ffffu64, 0xfff0_0000_0000_0000u64);
                    while high - low > 1 {
                        let mid = low + ((high - low) >> 1);
                        if reference_bucket(from_ordered(mid), offset, bonus, &s_bounds) > b {
                            high = mid;
                        } else {
                            low = mid;
                        }
                    }
                    let r = (from_ordered(high) * 16_777_216.0).ceil();
                    *slot = if r >= 2_147_483_647.0 {
                        i32::MAX
                    } else {
                        r.max(-2_147_483_648.0) as i32
                    };
                }
                row
            })
            .collect()
    }

    #[test]
    fn table_is_what_the_published_bucket_function_gives() {
        let want = recompute_table();
        let mut differing = Vec::new();
        for (code, row) in want.iter().enumerate() {
            if *row != GP_KING_IB[code] {
                differing.push((code, *row, GP_KING_IB[code]));
            }
        }
        assert!(
            differing.is_empty(),
            "{} of {CODES} rows differ, first: {:?}",
            differing.len(),
            differing.first()
        );
    }

    #[test]
    fn bounds_rise_within_a_row_and_fall_with_the_code() {
        for row in &GP_KING_IB {
            assert!(row.windows(2).all(|w| w[0] <= w[1]), "{row:?}");
        }
        for pair in GP_KING_IB.windows(2) {
            for (safer, plainer) in pair[1].iter().zip(&pair[0]) {
                assert!(safer <= plainer, "a safer king lowers the bounds");
            }
        }
    }

    #[test]
    fn startpos_is_the_first_stack_with_no_blend() {
        let pos = parse_sfen(STARTPOS).unwrap();
        // Nothing in a hand or in the enemy camp, and neither king has advanced.
        assert_eq!(phase_sum(&pos), 0);
        assert_eq!(king_code(&pos), 0);
        assert_eq!(
            choose(&pos),
            StackChoice {
                stack: 0,
                blend: None
            }
        );
    }

    #[test]
    fn the_phase_sum_weighs_hands_and_the_enemy_camp() {
        // Black: a pawn and a rook in hand; a promoted pawn on rank 3 (in the camp) and a
        // pawn on rank 4 (not). White: a gold in hand; a knight on rank 7 (its camp), and
        // a Black piece standing in White's camp counts for nobody.
        let pos = parse_sfen("4k4/9/+P8/P8/9/9/1n7/4S4/4K4 b PRg 1").unwrap();
        let want = HAND_WEIGHT[0] + HAND_WEIGHT[6] + CAMP_WEIGHT[7] // Black
            + HAND_WEIGHT[4] + CAMP_WEIGHT[2]; // White
        assert_eq!(phase_sum(&pos), want);
    }

    #[test]
    fn the_king_code_counts_defenders_behind_an_advanced_king() {
        // Black king on rank 5 (adv = 1): the gold beside it and the silver behind it
        // count, the pawn two ranks behind it does not (rank > k + 1), and a piece of the
        // other side does not either. White's king on rank 1 has adv < 0.
        let pos = parse_sfen("4k4/9/9/9/4K4/3G5/4S4/4P4/4p4 b - 1").unwrap();
        // adv = 6 - 4 = 2 -> weight 5; support = G (rank 5 <= 5) and S (rank 6 > 5: no).
        // Only the gold is within "no further back than one rank behind the king".
        assert_eq!(king_code(&pos), 5 * (1 + 1));
        // Three defenders cap the multiplier at four.
        let pos = parse_sfen("4k4/9/9/9/4K4/2GGG4/9/9/9 b - 1").unwrap();
        assert_eq!(king_code(&pos), 5 * 4);
    }

    #[test]
    fn a_stack_is_the_number_of_bounds_reached() {
        let first = GP_KING_IB[0][0];
        assert_eq!(bucket(first - 1, 0), 0);
        assert_eq!(bucket(first, 0), 1);
        // Code 0 has no reachable bound for the last step, so even the largest sum stops at
        // the sixth stack; a code with all seven bounds in reach gets to the eighth.
        assert_eq!(GP_KING_IB[0][6], NO_BOUND);
        assert_eq!(bucket(i32::MAX - 1, 0), 6);
        assert_eq!(bucket(i32::MAX - 1, 104), STACKS - 1);
    }

    #[test]
    fn a_position_next_to_a_bound_blends_across_it() {
        let first = GP_KING_IB[0][0];
        // One below the bound: still stack 0, mixed with stack 1 at the very start of the
        // zone's inner half (t = 0).
        assert_eq!(
            choose_from(first - 1, 0),
            StackChoice {
                stack: 0,
                blend: Some((1, 0))
            }
        );
        // On the bound: stack 1, with stack 0 across it.
        assert_eq!(
            choose_from(first, 0),
            StackChoice {
                stack: 1,
                blend: Some((0, 0))
            }
        );
        // The last unit of the zone: weight 511 of 512 on the position's own stack.
        let edge = first - (BLEND_WIDTH as i32 - 1);
        assert_eq!(choose_from(edge, 0).blend, Some((1, 511)));
        // Out of the zone.
        assert_eq!(choose_from(first - BLEND_WIDTH as i32, 0).blend, None);
    }

    #[test]
    fn the_reciprocal_is_floor_of_512_d_over_w_for_every_d_in_the_zone() {
        for d in 0..BLEND_WIDTH {
            let t = ((BLEND_RECIPROCAL * ((d as u64) << 9) as u128) >> 64) as i64;
            assert_eq!(t, (512 * d) / BLEND_WIDTH, "d = {d}");
        }
    }

    #[test]
    fn the_blend_rounds_toward_zero() {
        // Half way (t = 0): (other - raw) * 512 / 1024.
        assert_eq!(blend(100, 200, 0), 150);
        assert_eq!(blend(100, 0, 0), 50);
        assert_eq!(blend(100, 1, 0), 100 - 49, "-99 / 2 is -49 toward zero");
        assert_eq!(blend(100, 201, 0), 100 + 50, "101 / 2 is 50");
        // No weight on the neighbour at the edge of the zone.
        assert_eq!(blend(100, 1_000_000, 512), 100);
        // The product is taken in 64 bits: 2e9 * 512 does not fit in 32.
        assert_eq!(blend(-1_000_000_000, 1_000_000_000, 0), 0);
    }
}
