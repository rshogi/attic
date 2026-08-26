//! The live fixed-point scale every evaluator divides its output by.
//!
//! Its own module because it belongs to no single one of them. YaneuraOu keeps
//! `FV_SCALE` as a mutable global set from a USI option rather than as
//! something the network file carries, and the value that suits a network is a
//! property of how it was trained: the shipped default is 16, Hao wants 20,
//! NAGISA 28, AobaNNUE 40. A shape that lived in `network.rs` would vanish
//! along with it under the Manaka shapes, and those need the same knob for the
//! same reason.

use std::sync::atomic::{AtomicI32, Ordering};

/// The reference default (`Options.add("FV_SCALE", 16, ...)` in
/// `evaluate_nnue.cpp`, `int FV_SCALE = 16`). Also the USI option's default and
/// the condition the eval fixtures were captured under.
///
/// The SFNN shape NAGISA ships was trained at 28, so a caller loading that
/// network must pass `FV_SCALE 28` rather than rely on this.
pub const FV_SCALE_DEFAULT: i32 = 16;

static FV_SCALE: AtomicI32 = AtomicI32::new(FV_SCALE_DEFAULT);

/// The current fixed-point scale (the reference's live global `NNUE::FV_SCALE`).
pub fn fv_scale() -> i32 {
    FV_SCALE.load(Ordering::Relaxed)
}

/// Set the live fixed-point scale, mirroring the reference's `FV_SCALE` option
/// callback. The next evaluation divides by this value; the USI layer writes it
/// no later than the next `go`.
pub fn set_fv_scale(scale: i32) {
    FV_SCALE.store(scale, Ordering::Relaxed);
}
