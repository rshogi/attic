//! What the two files that read a trained network share.

use std::path::PathBuf;

/// The trained file the integration tests are pointed at, if any.
pub fn network_path() -> Option<PathBuf> {
    std::env::var_os("ATTIC_SYM_MNW").map(PathBuf::from)
}

/// The largest difference between two sums as a fraction of the size of the one
/// it is measured against, or of one for a sum smaller than that.
pub fn relative_drift(derived: &[f32], rebuilt: &[f32]) -> f32 {
    derived
        .iter()
        .zip(rebuilt)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
        .fold(0.0, f32::max)
}
