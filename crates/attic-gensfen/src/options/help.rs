//! The text `--help` prints, and what a bad command line is answered with.

use super::Options;

impl Options {
    /// The first line of `--help`, and what a bad flag is answered with.
    pub const USAGE: &'static str = "\
usage: attic-gensfen --eval-dir DIR --out FILE [options]
       attic-gensfen --help | --version";

    /// The full option table.
    ///
    /// Every flag is listed with the value it takes and the default it has,
    /// because the defaults are what bite: `--random-move-count` is 5, not 0,
    /// and a corpus meant to be all best moves comes out with five random
    /// plies a game if the flag is left off. `--dedup-mb` is 0: the filter that
    /// leaves a repeated board out of the corpus is off unless it is stated.
    /// Nothing here is a substitute for the notes at the end, which are the
    /// things that are not visible in the output when they go wrong.
    pub const HELP: &'static str = "\
Generate NNUE teacher data by self-play, driving the search in-process.

required:
  --eval-dir DIR         directory holding nn.bin
  --out FILE             where to write; opened for APPEND, never truncated

search:
  --depth N              search depth                              [9]
  --nodes N              node ceiling per search, 0 for none       [0]
  --multipv N            candidates recorded per position          [5]
  --fv-scale N           evaluation scale; NAGISA V4 wants 28      [28]
  --hash MB              transposition table PER WORKER            [16]

how much:
  --games N              stop after N games, 0 to run until killed [0]
  --positions N          stop after N positions instead, 0 for off [0]
  --threads N            worker threads, 0 for every system thread [0]
  --headroom N           threads left to the rest of the machine   [8]
  --governor-secs N      how often to re-read the load, 0 to pin   [30]

how a game ends:
  --max-ply N            hard cap on the length of a game          [320]
  --entering-king RULE   NoEnteringKing, CSARule24, CSARule24H,
                         CSARule27, CSARule27H, TryRule            [CSARule27]
  --quick-draw on|off    repetition rule of the search: on adjudicates a
                         second occurrence, off waits for the fourth in
                         the stored chain. A game is scored on the fourth
                         either way                       [the build's default]
  --resign-cp N          resign at this score, 0 to play on        [3000]
  --resign-count N       plies at that score before resigning      [1]

what gets written:
  --format F             jsonl, psv, or pack                       [jsonl]
  --eval-limit N         drop records past this score, 0 for none  [0]

where a game starts:
  --start-sfens FILE     draw openings from this file, one per line
  --random-move-count N  random plies played per game, 0 for none  [5]
  --random-move-minply N first ply that may be random              [16]
  --random-move-maxply N last ply that may be random               [32]
  --recover-from DIR     resume the games in this corpus instead of
                         drawing openings; refuses --start-sfens, a
                         non-zero --random-move-count and a non-zero
                         --dedup-mb
  --seed N               base for the random number generator      [1]

repeated boards:
  --dedup-mb MB          table of boards already searched, PER NUMA
                         NODE; 0 turns the filter off              [0]
  --dedup-max-ply N      plies of a game the table is read for     [16]

books:
  --move-margin CP       play a candidate within CP of the best, not
                         the best; 0 plays the best only           [0]
  --book-ply N           grow a book as a tree up to this move number
                         instead of playing games, 0 to play games [0]
  --book-width N         boards kept per level, 0 for all          [0]
  --book-band CP         drop tree boards this far from the score the
                         book was rooted at, 0 for none            [0]

  -h, --help             this text
  -V, --version          version and the commit it was built from

notes:
  --random-move-count defaults to 5. A corpus of nothing but search best
  moves needs --random-move-count 0 stated; leaving the flag off does not
  give it. The ply window counts plies played in THIS game, so with an
  opening at move 37 the random plies land at moves 53 to 69, not 16 to 32.

  --out is opened for append. Re-running with the same --out and the same
  --seed writes the same games again, into the same file.

  --eval-limit drops records without shortening the game, so a corpus
  written with it has games whose tails are missing rather than games that
  ended early. --recover-from is what plays those tails.

  Repetition and the entering-king law end games; a game that reaches
  --max-ply is recorded as a draw.

  --hash is per worker, not per process and not per NUMA node. The workers
  on a NUMA node share one table sized --hash times their number, so 256
  threads over 8 NUMA nodes at the default hold 512 MB a node, 4 GB in all.

  --dedup-mb is off by default; 64 is the size it was measured at. With a
  table, a board it has already seen is still searched, because the move
  played in its place is drawn from that search's candidates, but the ply is
  not recorded and the game plays any candidate except the best. The table
  is lossy on purpose: it only decides where a search is spent, and a
  collision costs one position, never a wrong number. The ply window of
  --dedup-max-ply counts plies played in THIS game. A non-zero --dedup-mb
  is refused with --recover-from.

  --move-margin above 0 draws every ply from the candidates within that many
  centipawns of the best, and takes the place of the random plies: a ply is
  never both. In the tree, 0 admits every --multipv candidate as a child.

  --book-ply is a MOVE NUMBER, not a game length: roots from --start-sfens
  carry their own, and the initial position is move 1. The tree searches
  every board of a level, and the candidates --move-margin admits become the
  next level. --games, --positions, --max-ply, the random plies, --dedup-mb,
  resignation and adjudication do not apply to it. Each level is written
  as --out with its extension replaced, plyNN.txt, one sfen per line, so it
  feeds back in as --start-sfens; plyNN.cp.txt beside it holds each board's
  score, its sfen and its parent's. A level is on disk the moment it closes,
  so a run stopped by hand keeps every level finished.

  --book-band is measured from the median score of the root level, taken
  when that level closes, and not from the level a board is in. It prunes:
  the subtree under a dropped board is never searched. With several
  --threads the workers of a node share one transposition table, so a tight
  band cuts on which worker won a race rather than on the position; the
  bands that mean anything are wide, unless --threads is 1.";
}
