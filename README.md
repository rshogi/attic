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
| `halfkahm2_2304_15_64` | HalfKA_hm2 | `nn.bin` + `progress.bin` | 28 | `halfka2304` |
| `halfkp_256x2-32-32` | HalfKP | `nn.bin` | 16 | — |
| `halfkp_512x2-16-32` | HalfKP | `nn.bin` | 16 | — |
| `halfkp_768x2-16-64` | HalfKP | `nn.bin` | 16 | — |
| `halfkp_1024x2-8-32` | HalfKP | `nn.bin` | 16 | — |
| `halfkp_1024x2-8-64` | HalfKP | `nn.bin` | 16 | — |
| `halfkpe9_256x2-32-32` | HalfKPE9 | `nn.bin` | 16 | — |
| `halfkpvm_256x2-32-32` | HalfKPvm | `nn.bin` | 16 | — |
| `kp_256x2-32-32` | KP | `nn.bin` | 16 | — |
| `sfnn-1536` | HalfKA_hm2 | `nn.bin` | 16 | `sfnn1536` |

A dash under Feature flag means the shape is not implemented; feeding one fails
on the version check or on a size mismatch.

`FV_SCALE` is the divisor the network was trained against. The engine defaults to
16 regardless, so any other value has to be supplied through `setoption` or
`eval_options.txt`.

Network files go in one directory, pointed at by `EvalDir`.

## Teacher generation — `attic-gensfen`

One copy of the network per NUMA node, shared by that node's workers, playing
games in parallel. Each position records its MultiPV candidates and scores, the
root score, and the move played.

```bash
attic-gensfen --eval-dir DIR --out FILE \
              [--depth 9] [--nodes N] [--multipv 5] [--threads N] \
              [--games N] [--positions N] [--fv-scale 28] \
              [--format jsonl|psv|pack]
```

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

`--threads` is a ceiling rather than a setting. Concurrency follows the
machine's idle CPU while running, aiming to leave `--headroom` cores free, so a
run started on an empty machine gives way when someone else arrives and takes
the room back when they leave. `--governor-secs 0` fixes it instead, which is
what a dedicated machine or a benchmark wants.

`scripts/fleet.py` runs one process per machine over SSH; unrecognised
arguments are passed through.

```bash
scripts/fleet.py launch --nodes node1,node2 --depth 9 --format pack
scripts/fleet.py status
scripts/fleet.py stop
```

## Building

Requires `rustup` and `build-essential`. `.cargo/config.toml` applies
`-C target-cpu=native`, so build on the machine that runs it.

```bash
cargo build --release -p attic --features sfnn1536
cargo build --release -p attic --features halfka2304
cargo build --release -p attic-gensfen --features halfka2304
```

Linux x86_64 only. The SIMD backend is fixed at build time with no runtime probe:
AVX-512 (F+BW) if enabled, otherwise AVX2, otherwise scalar. All three are
bit-identical.

> ⚠ `halfka2304` excludes the AVX-512 path: its tail is fixed at `[u8; 32]` and
> cannot take L2's 64 outputs (upstream's `kUsePackedTail` is not ported).

## License

GPL-3.0, inherited from upstream YaneuraOu and from attic-shogi. Copyright for
the ported code belongs to yaneurao, the YaneuraOu contributors, and the author
of attic-shogi.

- YaneuraOu: <https://github.com/yaneurao/YaneuraOu>
- attic-shogi: <https://github.com/ishidakei/attic-shogi>
