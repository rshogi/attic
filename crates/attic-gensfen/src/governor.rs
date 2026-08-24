//! Letting concurrency follow whoever else is on the machine.
//!
//! `--threads` is a ceiling, not a setting. These machines are shared, and a
//! count fixed at launch is wrong within minutes: someone starts a job and we
//! should shrink, they finish and we should grow. So the process samples
//! `/proc/stat` periodically and moves a permit count between one and the
//! ceiling, aiming to leave `headroom` cores idle.
//!
//! Idle CPU is measured as a one-second delta, never load average — a load
//! figure is an exponential average minutes behind a job that just started,
//! which is exactly the case this exists to catch.
//!
//! Permits are taken between games, so shrinking never discards work in
//! flight; it stops threads from starting another game until there is room.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/// How many games may be in flight, adjusted in the background.
pub struct Governor {
    permits: AtomicUsize,
}

/// Total and idle jiffies since boot, from the aggregate `cpu` line.
fn sample() -> Option<(u64, u64)> {
    let stat = std::fs::read_to_string("/proc/stat").ok()?;
    let line = stat.lines().next()?;
    let fields: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|v| v.parse().ok())
        .collect();
    if fields.len() < 5 {
        return None;
    }
    // Columns 3 and 4 are idle and iowait.
    Some((fields.iter().sum(), fields[3] + fields[4]))
}

/// Idle cores across a one-second window, or `None` where `/proc/stat` is not
/// readable.
fn idle_cores(cores: usize) -> Option<f64> {
    let (total_before, idle_before) = sample()?;
    std::thread::sleep(Duration::from_secs(1));
    let (total_after, idle_after) = sample()?;
    let total = total_after.checked_sub(total_before)?;
    let idle = idle_after.saturating_sub(idle_before);
    if total == 0 {
        return None;
    }
    Some(idle as f64 / total as f64 * cores as f64)
}

impl Governor {
    /// A governor starting at `ceiling` permits. With `interval` zero it never
    /// adjusts — fixed concurrency, for benchmarks and for dedicated machines.
    pub fn spawn(ceiling: usize, headroom: usize, interval: Duration) -> &'static Governor {
        // Leaked on purpose: one per process, outliving every worker thread.
        let governor: &'static Governor = Box::leak(Box::new(Governor {
            permits: AtomicUsize::new(ceiling),
        }));
        if interval.is_zero() {
            return governor;
        }

        // Physical cores, not logical: past the physical count throughput
        // regresses, so hyperthreads are not room worth claiming.
        let cores = physical_cores().unwrap_or(ceiling);
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(interval);
                let Some(idle) = idle_cores(cores) else {
                    continue;
                };
                let current = governor.permits.load(Ordering::Relaxed);
                // Our own threads are part of what is busy, so the room we
                // could occupy is what is idle plus what we already hold.
                let room = idle + current as f64 - headroom as f64;
                // Move a quarter of the way each time: a full jump would
                // oscillate against a neighbour doing its own ramp-up.
                let target = current as f64 + (room - current as f64) * 0.25;
                let target = (target.round() as usize).clamp(1, ceiling);
                if target != current {
                    governor.permits.store(target, Ordering::Relaxed);
                    eprintln!(
                        "gensfen: concurrency={target} idle_cores={idle:.0} ceiling={ceiling}"
                    );
                }
            }
        });
        governor
    }

    /// Whether a worker holding slot `index` may start another game.
    ///
    /// Workers are ordered rather than counted: slot `index` runs while the
    /// permit count is above it, so shrinking always parks the highest slots
    /// and the same threads keep working. A counter would need every worker to
    /// claim and release, which is more coordination than a fixed order buys.
    pub fn admits(&self, index: usize) -> bool {
        index < self.permits.load(Ordering::Relaxed)
    }
}

/// Physical core count from `/proc/cpuinfo`, counting distinct
/// (physical id, core id) pairs.
fn physical_cores() -> Option<usize> {
    let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut seen = std::collections::HashSet::new();
    let (mut package, mut core) = (None, None);
    for line in info.lines() {
        let mut parts = line.splitn(2, ':');
        let key = parts.next()?.trim();
        let value = parts.next().map(str::trim);
        match key {
            "physical id" => package = value.and_then(|v| v.parse::<u32>().ok()),
            "core id" => core = value.and_then(|v| v.parse::<u32>().ok()),
            _ => {}
        }
        if let (Some(p), Some(c)) = (package, core) {
            seen.insert((p, c));
            package = None;
            core = None;
        }
    }
    (!seen.is_empty()).then_some(seen.len())
}
