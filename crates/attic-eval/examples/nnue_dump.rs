//! Evaluate a file of positions with a YaneuraOu-standard network and print one
//! row each, in the same columns as `sym_dump`, to put the result beside
//! Manaka's own `dump_eval` for an `nnue` shape.
//!
//! ```text
//! cargo run --release -p attic-eval --features nnue_kp_256x2_32_32 --example nnue_dump -- \
//!     --net nn.bin --sfens positions.sfen [--limit N]
//! ```
//!
//! One tab-separated row per position: index, sfen, then the score twice (the
//! columns `sym_dump` calls `value` and `pre`; an NNUE output has no `tanh` to
//! tell them apart). The score is the engine's integer evaluation divided by
//! the number of engine units one unit of Manaka's linear value row is worth:
//! `127 · 64 / FV_SCALE`, which is 508 at the default 16. That is the whole
//! of the conversion, so a difference here is the quantisation and the integer
//! truncation (one engine unit, about 0.002) and nothing else.
//!
//! Hash mismatches the loader tolerates are reported on stderr: they are
//! warnings to the engine and would be a wrong network to this.

use std::io::{BufRead, BufWriter, Write};
use std::path::PathBuf;

use attic_eval::{evaluate, fv_scale, load_network_with_warnings};
use attic_state::parse_sfen;

/// What one unit of a linear value row is worth once `FV_SCALE` has divided it.
/// The same product `manaka-train`'s exporter derives it from.
const ACTIVATION_SCALE: f32 = 127.0;
const WEIGHT_SCALE: f32 = 64.0;

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
        eprintln!("usage: nnue_dump --net FILE --sfens FILE [--limit N]");
        std::process::exit(2);
    };

    let (network, warnings) = load_network_with_warnings(&net).unwrap_or_else(|e| {
        eprintln!("{}: {e}", net.display());
        std::process::exit(1);
    });
    for warning in &warnings {
        eprintln!("warning: {warning}");
    }
    let file = std::fs::File::open(&sfens).unwrap_or_else(|e| {
        eprintln!("{}: {e}", sfens.display());
        std::process::exit(1);
    });

    let per_unit = ACTIVATION_SCALE * WEIGHT_SCALE / fv_scale() as f32;
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
        let value = evaluate(&network, &pos) as f32 / per_unit;
        writeln!(out, "{index}\t{sfen}\t{value:.9e}\t{value:.9e}").expect("stdout");
    }
}
