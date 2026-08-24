# attic-shogi

Rust で書かれた、[YaneuraOu](https://github.com/yaneurao/YaneuraOu) の探索部の互換エンジンです。

基準は [V9.80](https://github.com/yaneurao/YaneuraOu/commit/d47c9bfb94210fb1f9b741d5ee16ce081b35cacd) です。以下で YaneuraOu と書いたものは、この版を指します。

## コマンド

USI プロトコルを標準入出力で話します。引数なしで `attic` を起動すると、コマンドの入力待ちになります。

| コマンド                                                                       | 説明                                                                                                                          |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `usi`                                                                          | エンジン情報とオプション一覧を出力し、`usiok` を返す                                                                          |
| `isready`                                                                      | 定跡と評価関数の `<EvalDir>/nn.bin` を読み込み、成功すれば `readyok` を返す                                                   |
| `setoption name <名前> value <値>`                                             | オプションを設定する。一覧は `usi` が出力する                                                                                 |
| `usinewgame`                                                                   | 新規対局の開始                                                                                                                |
| `position [startpos \| sfen <SFEN>] [moves <手>]`                              | 局面を設定する                                                                                                                |
| `go`                                                                           | 探索を開始し、`bestmove` を返す                                                                                               |
| `stop`                                                                         | 探索を停止する                                                                                                                |
| `ponderhit`                                                                    | 先読みが的中したことを通知する                                                                                                |
| `gameover`                                                                     | 対局の終了                                                                                                                    |
| `quit`                                                                         | 終了する                                                                                                                      |
| `bench [ttSizeMB] [threads] [limit] [default\|current\|<fenFile>] [limitType]` | 固定条件で NPS を計測する。引数はすべて省略でき、既定値は置換表 1024 MB、1 スレッド、`limit` 15000、`limitType` は `movetime` |

## YaneuraOu との互換性

同じ局面・同じ条件で探索したときに、各反復の深さ・評価値・ノード数・読み筋と `bestmove` が YaneuraOu と行単位で一致することを目指しています。確認できた範囲は次のとおりで、これ以外は確認していません。

| 評価関数                  | 確認した内容                                                |
| ------------------------- | ----------------------------------------------------------- |
| `nnue_kp_256x2_32_32`     | 1 局面 10 万ノード、置換表 16 MB で 16,384 局面の全行が一致 |
| `nnue_halfkp_256x2_32_32` | 18 局面 × 4 段階で一致                                      |
| `nnue_halfkp_128x2_32_32` | 1 局面 10 万ノード、置換表 16 MB で 1,184 局面の全行が一致  |
| `nnue_halfkp_768x2_16_64` | 1 局面 10 万ノード、置換表 16 MB で 1,184 局面の全行が一致  |
| 上記以外                  | 未確認                                                      |

## 探索速度

YaneuraOu を 1 とした比です。YaneuraOu 側も同じ評価関数で動かしています。1 スレッドで、1 局面 10 万ノード × 128 局面を交互に測定しました。

| 評価関数                  | `cargo xtask build-pgo` | `cargo build --release` |
| ------------------------- | ----------------------- | ----------------------- |
| `nnue_kp_256x2_32_32`     | 1.020                   | 0.976                   |
| `nnue_halfkp_128x2_32_32` | 1.029                   | 0.999                   |
| `nnue_halfkp_768x2_16_64` | 1.050                   | 1.042                   |

`cargo xtask build-pgo` は 3 つとも YaneuraOu 以上です。`cargo build --release` は `nnue_kp_256x2_32_32` と `nnue_halfkp_128x2_32_32` で 1.000 を下回ります。AVX-512 のカーネルと、表にない評価関数の速度は測っていません。

## 対応している評価関数

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
