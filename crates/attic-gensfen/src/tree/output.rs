//! Writing a level of the book to disk.

use std::io::{BufWriter, Write};
use std::path::Path;

use super::Frontier;

impl Frontier {
    /// One level as a `--start-sfens` file: `sfen ` prefixed, one per line, so
    /// a book this run produced feeds straight back in as the roots of a
    /// deeper one.
    ///
    /// Called as each level closes, and again for the level above once its
    /// children have been backed up into it.
    pub(super) fn write_level(
        &self,
        ply: u16,
        kept: &[(i32, String, String)],
    ) -> std::io::Result<()> {
        let path = self.out.with_extension(format!("ply{ply:02}.txt"));
        write_atomically(&path, |file| {
            for (_, sfen, _) in kept {
                writeln!(file, "sfen {sfen}")?;
            }
            Ok(())
        })?;

        // The score goes beside the level rather than into it: a line there
        // has to stay something `--start-sfens` parses, and a book is filtered
        // by score long after the run that built it. Centipawns from the side
        // to move, which for one level is one side — every board in it carries
        // the same move number. The third column is the board it came from, so
        // a level can be chained back to the root one move at a time.
        let path = self.out.with_extension(format!("ply{ply:02}.cp.txt"));
        write_atomically(&path, |file| {
            for (cp, sfen, parent) in kept {
                writeln!(file, "{cp}\tsfen {sfen}\tsfen {parent}")?;
            }
            Ok(())
        })
    }
}

/// Write through a temporary file and rename over the target.
///
/// A level is rewritten in place once its children back values up into it, and
/// a long book is stopped by hand rather than by finishing. Between those two,
/// a plain truncate-and-write leaves a window where the file on disk is half a
/// level — and half a level is indistinguishable from a small one.
fn write_atomically(
    path: &Path,
    body: impl FnOnce(&mut BufWriter<std::fs::File>) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let temp = path.with_extension("part");
    let mut file = BufWriter::new(std::fs::File::create(&temp)?);
    body(&mut file)?;
    file.flush()?;
    drop(file);
    std::fs::rename(&temp, path)
}
