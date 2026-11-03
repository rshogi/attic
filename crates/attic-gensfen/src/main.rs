//! The `attic-gensfen` binary: [`attic_gensfen::run`] on this process's arguments.

use std::process::ExitCode;

fn main() -> ExitCode {
    attic_gensfen::run(std::env::args().skip(1))
}
