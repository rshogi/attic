#![cfg(feature = "sfnn_threat")]

use attic_eval::MoveDelta;
use attic_eval::threat::{MAX_EDGES, ThreatDiff, ThreatList, build_lists, update_lists};
use attic_state::{Move, PieceKind, Position, format_sfen, format_usi_move, parse_sfen};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn pick(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const STARTS: [&str; 8] = [
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
    "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
    "4k4/9/4r4/9/9/9/4B4/9/4K4 b RG2gs2n3p 1",
    "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
    "l3k3l/9/9/9/9/9/9/9/L3K3L b 2R2B4G4S4N18P 1",
    "l4k2l/9/9/9/9/9/9/9/L2K4L w 2R2B4G4S4N18P 1",
    "4k4/9/9/9/9/9/9/9/4K4 b 2R2B4G4S4N4L18P 1",
    "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
];

/// Packed board whose lists start at the cap.
fn dense_start(rng: &mut Rng) -> String {
    let mut pieces: Vec<&str> = Vec::new();
    for (t, n) in [("+P", 18), ("+L", 4), ("+N", 4), ("+S", 4), ("G", 4)] {
        pieces.extend(std::iter::repeat_n(t, n));
    }
    let mut grid: Vec<Vec<String>> = vec![vec![String::new(); 9]; 9];
    let colour = |t: &str, rng: &mut Rng| {
        if rng.pick(2) == 0 {
            t.to_string()
        } else {
            t.to_lowercase()
        }
    };
    let mut files: Vec<usize> = (0..9).collect();
    for k in 0..9 {
        files.swap(k, k + rng.pick(9 - k));
    }
    for (k, t) in ["+R", "+R", "+B", "+B"].into_iter().enumerate() {
        grid[4][files[k]] = colour(t, rng);
    }
    for k in 0..pieces.len() {
        let j = k + rng.pick(pieces.len() - k);
        pieces.swap(k, j);
    }
    let mut it = pieces.into_iter();
    for row in &mut grid[3..6] {
        for cell in row.iter_mut().filter(|c| c.is_empty()) {
            *cell = colour(it.next().unwrap(), rng);
        }
    }
    let mut rest: Vec<(usize, usize)> = (0..9).flat_map(|f| [(2, f), (6, f)]).collect();
    for k in 0..rest.len() {
        let j = k + rng.pick(rest.len() - k);
        rest.swap(k, j);
    }
    for (t, (r, f)) in it.zip(rest) {
        grid[r][f] = colour(t, rng);
    }
    grid[0][0] = "k".into();
    grid[8][8] = "K".into();
    let ranks: Vec<String> = grid
        .iter()
        .map(|row| {
            let (mut out, mut gap) = (String::new(), 0);
            for c in row {
                if c.is_empty() {
                    gap += 1;
                } else {
                    if gap > 0 {
                        out += &gap.to_string();
                        gap = 0;
                    }
                    out += c;
                }
            }
            if gap > 0 {
                out += &gap.to_string();
            }
            out
        })
        .collect();
    format!(
        "{} {} - 1",
        ranks.join("/"),
        if rng.pick(2) == 0 { "b" } else { "w" }
    )
}

fn rows_sorted(d: &ThreatDiff) -> (Vec<u32>, Vec<u32>) {
    let (mut r, mut a) = (d.removed().to_vec(), d.added().to_vec());
    r.sort_unstable();
    a.sort_unstable();
    (r, a)
}

#[derive(Default, Debug)]
struct Count {
    followed: usize,
    refused_at_cap: usize,
    king_moves: usize,
    capped_positions: usize,
    captures: usize,
    drops: usize,
}

fn play(start: &str, seed: u64, plies: usize, n: &mut Count) {
    let mut pos: Position = parse_sfen(start).unwrap_or_else(|e| panic!("bad start: {e:?}"));
    let mut rng = Rng(seed | 1);
    let mut legal: Vec<Move> = Vec::with_capacity(256);
    let mut lists = build_lists(&pos);
    for ply in 0..plies {
        legal.clear();
        pos.generate_legal_all(&mut legal);
        if legal.is_empty() {
            break;
        }
        let is_capture = |m: &Move| !m.is_drop() && pos.board().get(m.to_sq()).is_some();
        let pool: Vec<Move> = match rng.pick(6) {
            0 | 1 => legal.iter().copied().filter(is_capture).collect(),
            2 | 3 => legal.iter().copied().filter(|m| m.is_drop()).collect(),
            4 => legal.iter().copied().filter(|m| m.is_promote()).collect(),
            _ => Vec::new(),
        };
        let mv = if pool.is_empty() {
            legal[rng.pick(legal.len())]
        } else {
            pool[rng.pick(pool.len())]
        };
        let ctx = format!(
            "`{}` ply {ply} `{}`",
            format_sfen(&pos),
            format_usi_move(mv)
        );
        let king_move = !mv.is_drop() && mv.moved_piece_after().kind == PieceKind::King;
        n.captures += usize::from(is_capture(&mv));
        n.drops += usize::from(mv.is_drop());
        let delta = MoveDelta::from_move(&pos, mv);
        let _undo = pos.do_move(mv);
        let built = build_lists(&pos);
        if built[0].edge_count() >= MAX_EDGES || built[1].edge_count() >= MAX_EDGES {
            n.capped_positions += 1;
        }
        if king_move {
            n.king_moves += 1;
        } else {
            let mut new = [ThreatList::EMPTY; 2];
            let mut diff = [ThreatDiff::new(), ThreatDiff::new()];
            if update_lists(&pos, &lists, delta.board_change(), &mut new, &mut diff) {
                n.followed += 1;
                for p in 0..2 {
                    assert_eq!(new[p], built[p], "{ctx}: perspective {p} list");
                    let mut want = ThreatDiff::new();
                    want.between(&lists[p], &built[p]);
                    assert_eq!(
                        rows_sorted(&diff[p]),
                        rows_sorted(&want),
                        "{ctx}: perspective {p} diff"
                    );
                }
            } else {
                let at_cap = |l: &[ThreatList; 2]| {
                    l[0].edge_count() >= MAX_EDGES || l[1].edge_count() >= MAX_EDGES
                };
                assert!(
                    at_cap(&lists) || at_cap(&built),
                    "{ctx}: refused below the cap ({} {} -> {} {})",
                    lists[0].edge_count(),
                    lists[1].edge_count(),
                    built[0].edge_count(),
                    built[1].edge_count()
                );
                n.refused_at_cap += 1;
            }
        }
        lists = built;
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn followed_lists_equal_built_lists_along_random_games() {
    let games: usize = std::env::var("ATTIC_FOLLOW_GAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let mut n = Count::default();
    for g in 0..games {
        let seed = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(g as u64 + 1);
        if g % 4 == 3 {
            let start = dense_start(&mut Rng(seed ^ 0xD1B5_4A32_D192_ED03 | 1));
            play(&start, seed, 120, &mut n);
        } else {
            play(STARTS[g % STARTS.len()], seed, 120, &mut n);
        }
    }
    eprintln!("threat follow: {n:?}");
    assert!(n.followed > 0);
    assert!(n.captures > 0 && n.drops > 0 && n.king_moves > 0);
    assert!(n.capped_positions > 0, "no list reached the cap");
    assert!(
        n.refused_at_cap > 0,
        "the cap was never met by a followed move"
    );
}
