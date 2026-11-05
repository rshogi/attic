//! Tests that need no file: a network is built in memory, written the way
//! Manaka's exporter writes one, and read back.
//!
//! They run under every `sym_*` shape, which is the point. CI has no trained
//! network for any of them, so these are what says a shape's dimensions, its
//! architecture string and its forward pass hang together. What a *trained*
//! file evaluates to is `tests/sym_network.rs`'s business, with a file.

mod encode;
mod exported;
mod fixtures;
mod format;
mod forward;
#[cfg(feature = "sym_token")]
mod moment_field;
#[cfg(feature = "sym_token")]
mod pair;
mod reference;
