//! Evaluate a file of positions with a Manaka network and print one row each,
//! to put the result beside Manaka's own `dump_eval`.
//!
//! ```text
//! cargo run --release -p attic-eval --features sym_kp_256x2_512 --example sym_dump -- \
//!     --net kp_pair0_d256_o3_h512.q.nn.bin --sfens positions.sfen [--limit N]
//! ```
//!
//! The file may be in any of the three encodings (`SYMscaled`, `SYMf32`, `SYM`);
//! the header says which.
//!
//! One tab-separated row per position: index, sfen, `value` (after the `tanh`,
//! side to move's view) and `pre` (before it). The values are printed to nine
//! significant digits, which is what an `f32` round-trips through.
//!
//! A tool and not a test, because the networks it reads are tens to hundreds of
//! megabytes and none of them live in the repository: `required-features`
//! keeps it out of every build that is not a Manaka shape.

use std::io::{BufRead, BufWriter, Write};
use std::path::PathBuf;

use attic_eval::manaka::{evaluate_pre, evaluate_value, load_network};
use attic_state::parse_sfen;

fn main() {
    let mut net = None;
    let mut sfens = None;
    let mut limit = usize::MAX;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--net" => net = args.next().map(PathBuf::from),
            "--sfens" => sfens = args.next().map(PathBuf::from),
            "--limit" => limit = args.next().and_then(|n| n.parse().ok()).unwrap_or(limit),
            other => {
                eprintln!("unknown argument {other:?}");
                std::process::exit(2);
            }
        }
    }
    let (Some(net), Some(sfens)) = (net, sfens) else {
        eprintln!("usage: sym_dump --net FILE --sfens FILE [--limit N]");
        std::process::exit(2);
    };

    let network = load_network(&net).unwrap_or_else(|e| {
        eprintln!("{}: {e}", net.display());
        std::process::exit(1);
    });
    let file = std::fs::File::open(&sfens).unwrap_or_else(|e| {
        eprintln!("{}: {e}", sfens.display());
        std::process::exit(1);
    });

    let mut out = BufWriter::new(std::io::stdout().lock());
    for (index, line) in std::io::BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .take(limit)
        .enumerate()
    {
        let sfen = line.trim();
        let pos = parse_sfen(sfen).unwrap_or_else(|e| {
            eprintln!("line {index}: {e:?}");
            std::process::exit(1);
        });
        let value = evaluate_value(&network, &pos);
        let pre = evaluate_pre(&network, &pos);
        writeln!(out, "{index}\t{sfen}\t{value:.9e}\t{pre:.9e}").expect("stdout");
    }
}
