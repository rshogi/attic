# Attic

A Rust port of the shogi engine YaneuraOu, tracking upstream V9.70 (`76d58ef`).

## Motivation

To carry YaneuraOu from C++ into type-safe Rust without giving up compatibility
with it. The search is guaranteed to match the reference engine exactly.

## Networks

A network is selected at build time. There is no default; a build with no
network feature fails to compile.

| Network | Features | Files | FV_SCALE | Feature flag |
|---|---|---|---|---|
| `halfkahm2_1024_15_64` | HalfKA_hm2 | `nn.bin` + `progress.bin` | 28 | — |
| `halfkahm2_2304_15_64` | HalfKA_hm2 | `nn.bin` + `progress.bin` | 28 | `sfnn_halfkahm2_2304_15_64_progress8ek` |
| `halfkp_256x2-32-32` | HalfKP | `nn.bin` | 16 | `nnue_halfkp_256x2_32_32` |
| `halfkp_512x2-16-32` | HalfKP | `nn.bin` | 16 | — |
| `halfkp_768x2-16-64` | HalfKP | `nn.bin` | 16 | — |
| `halfkp_1024x2-8-32` | HalfKP | `nn.bin` | 16 | — |
| `halfkp_1024x2-8-64` | HalfKP | `nn.bin` | 16 | — |
| `halfkpe9_256x2-32-32` | HalfKPE9 | `nn.bin` | 16 | — |
| `halfkpvm_256x2-32-32` | HalfKPvm | `nn.bin` | 16 | — |
| `kp_256x2-32-32` | KP | `nn.bin` | 16 | `nnue_kp_256x2_32_32` |
| `sfnn-1536` | HalfKA_hm2 | `nn.bin` | 16 | `sfnn_halfkahm2_1536_15_32_k3k3` |

A dash under Feature flag means the shape is not implemented; feeding one fails
on the version check or on a size mismatch.

Manaka's own shapes are not in the table because they are not NNUE files:
`sym_kp_128x2_512` and `sym_halfkp_128x2_512` read `MNW1` containers, whose
weights are `f32` and whose blocks are RMS-normalised where NNUE clips. They
share this crate's feature extraction and nothing else.

`FV_SCALE` is the divisor the network was trained against. The engine defaults to
16 regardless, so any other value has to be supplied through `setoption` or
`eval_options.txt`.

Network files go in one directory, pointed at by `EvalDir`.

## Teacher generation — `attic-gensfen`

One copy of the network per NUMA node, shared by that node's workers, playing
games in parallel. Each position records its MultiPV candidates and scores, the
root score, and the move played.

```bash
attic-gensfen --eval-dir DIR --out FILE [options]
attic-gensfen --help        # every flag, with its default
attic-gensfen --version     # the version, and the commit it was built from
```

The full option table lives in `--help` rather than here, so that the binary
in front of you is the one that answers. What follows is the part of it that is
not visible from the flag names.

A run that wants nothing but search best moves, played to the end of the game:

```bash
attic-gensfen \
  --eval-dir ~/eval/halfka2304 --out /raid/corpus/shard.w0.bin \
  --format pack --depth 9 --multipv 5 --fv-scale 28 \
  --start-sfens ~/openings/start_sfens_ply37.txt \
  --random-move-count 0 \
  --max-ply 512 --entering-king CSARule27 \
  --eval-limit 0 --resign-cp 0
```

Three of those are stated because the defaults are the other way.
`--random-move-count` is 5, so leaving it off mixes five random plies into
every game. `--eval-limit 0` and `--resign-cp 0` turn off both of the
score-based cut-offs, which is what keeps the endgame — and the mate scores —
in the corpus. `--max-ply 512` is above the default 320 because an
entering-king game does not finish inside it.

| Format | Record | Contents |
|---|---|---|
| `jsonl` | one game per line | positions as SFEN text, candidates as USI text |
| `psv` | 40 bytes | `PackedSfenValue`: packed position, score, move, ply, result |
| `pack` | 40 + 4N bytes | `psv`, then N `(move16, centipawn)` candidate pairs |

`pack`'s first 40 bytes are byte-identical to `psv`, so a reader that wants only
the position and its score can ignore the rest. N comes from `--multipv` and is
not stored per record; shorter candidate lists are zero-padded so records stay
seekable by index.

`--depth` and `--nodes` combine; whichever is reached first ends the search.
`--positions` is checked before a game starts, so a run exceeds it by up to
`workers × game length`.

`--start-sfens` draws each game's opening from a file, one SFEN per line, and
without it every game starts from the initial position. A leading `sfen ` and
CRLF endings are accepted, so the published opening sets load unchanged. Lines
are all parsed at startup: a bad one stops the run before any worker starts.

It exists because `--random-move-count` only reaches plies inside its window.
With a fixed start, every game replays the same opening up to
`--random-move-minply`, and each of those boards is written once per game — over
one 6.5e8-position corpus that was 18% of the rows, nearly all below ply 16.

**The random-move window is measured from the start of the game, not from the
opening's move number.** Starting from ply-15 positions with the default window
randomises what are really plies 31–47. `--max-ply` is a game length too, so the
cap is unchanged by where the game began.

### How a game ends

Six ways. `jsonl` names the one that fired in each game's `reason`; `psv` and
`pack` have no field for it and carry only the result.

| Ending | `reason` | Result written | Controlled by |
|---|---|---|---|
| mate, or no legal move | `mated` | loss for the side to move | — |
| fourfold repetition | `repetition` | draw | — |
| perpetual check | `perpetual_check` | loss for the checking side | — |
| entering-king declaration | `declaration` | win for the declaring side | `--entering-king` |
| resignation | `resign` | loss for the side that is behind | `--resign-cp`, `--resign-count` |
| the ply cap | `max_ply` | draw | `--max-ply` |

Repetition is judged on the game's own rule — the fourth occurrence, over the
whole recorded history — and not on the search's, which decides on the second
under a `quick-draw` build and looks back sixteen plies at most. Both of the
search's shortcuts are worth Elo inside a tree and both are wrong for scoring a
game. Positions that repeat with unequal hands are not a repetition under the
rules, so those games play on.

`--entering-king` selects the law: `NoEnteringKing`, `CSARule24`, `CSARule24H`,
`CSARule27` (the default, and the engine's), `CSARule27H`, `TryRule`. A
spelling it does not recognise is refused rather than quietly becoming
`NoEnteringKing`.

`--resign-cp` measures the side that is behind, which does not alternate with
the turn, and `--resign-count` is how many plies it has to stay there — the
default 1 being the flag's own bound. `--resign-cp 0` turns the judgement off
and lets the game run to mate, which is what a corpus that wants mate scores
needs.

### Resuming games an eval limit cut short

`--eval-limit N` drops the records whose root score leaves ±N. It does not
shorten the game, so a shard written with it holds games with their tails
missing, and holes in the middle wherever a score crossed the bound and came
back.

`--recover-from DIR` reads those shards, restores the last position each game
still has a record for, and plays on from there:

```bash
attic-gensfen --eval-dir ~/eval/halfka2304 \
              --recover-from /raid/corpus/old --out /raid/corpus/new/shard.w0.bin \
              --format pack --depth 9 --multipv 5
```

The continuation is a different game from the one that was cut off — the
workers share a transposition table and never clear it between games, so the
same seed does not reproduce the same moves. What is kept is the prefix of
records up to the break, re-stamped with the new game's result; what followed
it is discarded. It refuses `--start-sfens` and a non-zero
`--random-move-count`, both of which have nothing to act on, and refuses an
`--out` inside the directory it is reading.

`--multipv` is not stored per record, so recovery has to be given the value the
shards were written with. A wrong one reads plausible-looking records at a
shifted offset and nothing detects it.

Measured on one shard: 0.5 s and 21.5 new positions per game, against 16 s to
play a whole one.

`--threads` is a ceiling rather than a setting. Concurrency follows the
machine's idle CPU while running, aiming to leave `--headroom` cores free, so a
run started on an empty machine gives way when someone else arrives and takes
the room back when they leave. `--governor-secs 0` fixes it instead, which is
what a dedicated machine or a benchmark wants.

`scripts/fleet.py` runs one process per machine over SSH; unrecognised
arguments are passed through.

```bash
scripts/fleet.py launch --nodes node1,node2 --depth 9 --format pack
scripts/fleet.py watch          # redraws, with games/sec per node
scripts/fleet.py status         # one frame
scripts/fleet.py stop
```

## Building

Requires `rustup` and `build-essential`. `.cargo/config.toml` applies
`-C target-cpu=native`, so build on the machine that runs it.

Exactly one network shape must be selected; there is no default, and a build
without one fails in `attic-eval` with a page of missing functions rather than
a message about features. The names are the ones in `attic-eval/Cargo.toml`.

```bash
cargo build --release -p attic --features sfnn_halfkahm2_1536_15_32_k3k3
cargo build --release -p attic --features sfnn_halfkahm2_2304_15_64_progress8ek
cargo build --release -p attic-gensfen --features sfnn_halfkahm2_2304_15_64_progress8ek
```

`attic-gensfen --version` prints the workspace version and the commit the
binary was built from, `-dirty` when the tree had uncommitted changes. The
version alone cannot tell two builds apart — it tracks upstream and does not
move for a fix — and a corpus written by the wrong binary is not detectably
wrong afterwards, so the commit is what identifies a build. It is also in the
first line every run logs. `--help` prints the full option table.

Linux x86_64 only. The SIMD backend is fixed at build time with no runtime probe:
AVX-512 (F+BW) if enabled, otherwise AVX2, otherwise scalar. All three are
bit-identical.

> ⚠ `sfnn_halfkahm2_2304_15_64_progress8ek` excludes the AVX-512 path: its tail
> is fixed at `[u8; 32]` and cannot take L2's 64 outputs (upstream's
> `kUsePackedTail` is not ported).

## License

GPL-3.0, inherited from upstream YaneuraOu and from attic-shogi. Copyright for
the ported code belongs to yaneurao, the YaneuraOu contributors, and the author
of attic-shogi.

- YaneuraOu: <https://github.com/yaneurao/YaneuraOu>
- attic-shogi: <https://github.com/ishidakei/attic-shogi>
