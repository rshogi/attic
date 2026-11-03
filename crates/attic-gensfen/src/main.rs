//! Teacher-data generation, driving the search in-process.
//!
//! The pipeline this replaces ran one `YaneuraOu` per game over USI, driven
//! from Python. That cost a copy of the 180MB network per game, paid USI text
//! encoding at every node, and funnelled every engine's `info` output through
//! one GIL. Holding the network once per NUMA node and playing many games on
//! native threads removes all three.
//!
//! What one position contributes to the corpus is unchanged: the MultiPV
//! candidates and their scores (the material for a policy distribution), the
//! root score (for the value target), and the move actually played.
//!
//! Usage:
//! ```text
//! attic-gensfen --eval-dir DIR --out FILE [--depth 9] [--multipv 5]
//!               [--threads N] [--games N] [--max-ply 320] [--resign-cp 3000]
//!               [--entering-king CSARule27]
//! ```

mod game;
mod governor;
mod options;
mod recover;
mod worker;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use attic_numa::{NumaAutoPolicy, NumaConfig};
use attic_search::Search;
use attic_storage::TranspositionTable;

use options::Options;

fn main() {
    let options = match Options::parse(std::env::args().skip(1)) {
        Ok(options) => options,
        Err(message) => {
            eprintln!("{message}");
            eprintln!("{}", Options::USAGE);
            std::process::exit(2);
        }
    };

    let network_path = options.eval_dir.join("nn.bin");
    let (base, warnings) = match Search::from_network_file_with_warnings(&network_path) {
        Ok(loaded) => loaded,
        Err(err) => {
            eprintln!("failed to load {}: {err}", network_path.display());
            std::process::exit(1);
        }
    };
    for warning in &warnings {
        eprintln!("warning: {warning}");
    }
    // NAGISA V4's scores were produced at 28. The loader's default is the stock
    // 16 and its `eval_options.txt` carries no FV_SCALE, so the caller is what
    // sets it; leaving it at 16 inflates every score by 28/16.
    attic_search::set_fv_scale(options.fv_scale);

    // `SystemNuma` rather than the engine's bundled-L3 default: the weights are
    // replicated per logical node, and on an 8-domain EPYC the L3 bundling
    // would make far more nodes than there are copies worth holding.
    let numa = NumaConfig::from_system(&NumaAutoPolicy::SystemNuma, true);
    // Never more nodes than workers: `--threads 4` on an 8-node machine must
    // put four workers on four nodes, not one on each of eight. Rounding the
    // per-node count up and then running it on every node is what made
    // `--threads 1` spawn eight.
    let nodes = numa.num_numa_nodes().max(1).min(options.threads.max(1));
    let per_node = options.threads / nodes;
    // The remainder rides on the low-numbered nodes rather than being dropped,
    // so the worker total is exactly what was asked for.
    let extra = options.threads % nodes;

    eprintln!(
        "gensfen: numa_nodes={nodes} threads={} depth={} multipv={} fv_scale={}",
        options.threads, options.depth, options.multipv, options.fv_scale
    );

    // `--threads` is the ceiling; the governor decides how much of it to use.
    let governor = governor::Governor::spawn(
        options.threads,
        options.headroom,
        std::time::Duration::from_secs(options.governor_secs),
    );

    // Recovery reads the corpus back at the same record width it was written
    // with, which `--format` and `--multipv` are what state — the files carry
    // neither.
    let source = match &options.recover_from {
        None => None,
        Some(dir) => {
            let stride = match options.format {
                options::Format::Psv => 0,
                _ => options.multipv,
            };
            match recover::Source::new(dir, stride) {
                Ok(source) => {
                    eprintln!(
                        "gensfen: recovering {} shards from {}",
                        source.len(),
                        dir.display()
                    );
                    Some(source)
                }
                Err(err) => {
                    eprintln!("{err}");
                    std::process::exit(1);
                }
            }
        }
    };
    let source = source.as_ref();

    let games_played = Arc::new(AtomicU64::new(0));
    let positions_written = Arc::new(AtomicU64::new(0));
    let numa = Arc::new(numa);

    std::thread::scope(|outer| {
        for node in 0..nodes {
            let numa = Arc::clone(&numa);
            let games_played = Arc::clone(&games_played);
            let positions_written = Arc::clone(&positions_written);
            let base = &base;
            let options = &options;

            outer.spawn(move || {
                // Bind before replicating: the copy is allocated by this thread,
                // so binding first is what puts the weights in this node's
                // memory rather than wherever the allocator last touched.
                numa.bind_current_thread_to_numa_node(node);
                let local = base.replicate();

                let here = per_node + usize::from(node < extra);
                let base_slot = node * per_node + node.min(extra);

                // One table per NUMA node, shared by that node's workers, at
                // the same total size as a table each. Node-scoped rather than
                // fleet-wide for the same reason the weights are: a single
                // table would put seven nodes out of eight on the far side of
                // the fabric for every probe.
                //
                // Games here are independent, so cross-worker hits are mostly
                // confined to the shared opening plies — this is a measurement,
                // not an assumption that sharing wins.
                let mut tt = TranspositionTable::new();
                tt.resize(options.hash_mb * here.max(1));
                let tt = &tt;

                std::thread::scope(|inner| {
                    for slot in 0..here {
                        let numa = Arc::clone(&numa);
                        let games_played = Arc::clone(&games_played);
                        let positions_written = Arc::clone(&positions_written);
                        let local = &local;

                        inner.spawn(move || {
                            numa.bind_current_thread_to_numa_node(node);
                            let index = base_slot + slot;
                            let seed = options
                                .seed
                                .wrapping_mul(1_000_003)
                                .wrapping_add(index as u64);
                            worker::run(
                                local.network(),
                                tt,
                                options,
                                governor,
                                seed,
                                index,
                                &games_played,
                                &positions_written,
                                source,
                            );
                        });
                    }
                });
            });
        }
    });

    eprintln!(
        "gensfen: finished, {} games / {} positions",
        games_played.load(Ordering::Relaxed),
        positions_written.load(Ordering::Relaxed)
    );
}
