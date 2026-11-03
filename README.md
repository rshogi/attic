# attic-shogi

Rust で書かれた、[YaneuraOu](https://github.com/yaneurao/YaneuraOu) の探索部の互換エンジンです。

- 基準は YaneuraOu V9.80（コミット `d47c9bfb94210fb1f9b741d5ee16ce081b35cacd`）です。
- バイナリ名は `attic`、USI の `id name` は `Attic 9.80git` です。
- YaneuraOu（GPLv3）の派生物のため、本リポジトリも GPLv3 です（[`LICENSE`](LICENSE)）。

## コマンド

USI プロトコルを標準入出力で話します。引数なしで `attic` を起動すると、コマンドの入力待ちになります。

| コマンド                                                                       | 説明                                                                                                                          |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `usi`                                                                          | エンジン情報とオプション一覧を出力し、`usiok` を返す                                                                          |
| `isready`                                                                      | 定跡と評価関数（`<EvalDir>/nn.bin`）を読み込み、成功すれば `readyok` を返す                                                   |
| `setoption name <名前> value <値>`                                             | オプションを設定する（オプションの一覧は `usi` が出力する）                                                                   |
| `usinewgame`                                                                   | 新規対局の開始                                                                                                                |
| `position [startpos \| sfen <SFEN>] [moves <手> …]`                            | 局面を設定する                                                                                                                |
| `go […]`                                                                       | 探索を開始し、`bestmove` を返す                                                                                               |
| `stop`                                                                         | 探索を停止する                                                                                                                |
| `ponderhit`                                                                    | 先読み（ponder）が的中したことを通知する                                                                                      |
| `gameover`                                                                     | 対局の終了                                                                                                                    |
| `quit`                                                                         | 終了する                                                                                                                      |
| `bench [ttSizeMB] [threads] [limit] [default\|current\|<fenFile>] [limitType]` | 固定条件で NPS を計測する。引数はすべて省略でき、既定値は置換表 1024 MB、1 スレッド、`limit` 15000、`limitType` は `movetime` |

コマンドラインのサブコマンドとして、指し手生成の数え上げ（perft）も使えます。

```bash
attic perft startpos <depth>
attic perft sfen <SFEN> <depth>
attic perft sfen <SFEN> moves <m1> [<m2> …] <depth>
```

YaneuraOu のデバッグ・補助コマンド（`d`、`eval`、`moves`、`flip`、`getoption`、`compiler`、`unittest`、`test` など）は実装していません。認識できない行は `info string unknown command: …` を出力して読み飛ばします。

## YaneuraOu V9.80 との互換性

同じ局面・同じ条件で探索したときに、各反復の深さ・評価値・ノード数・読み筋と `bestmove` が YaneuraOu V9.80 と行単位で一致することを目指しています。確認できた範囲は次のとおりで、これ以外は確認していません。

| 評価関数                  | 確認した内容                                                |
| ------------------------- | ----------------------------------------------------------- |
| `nnue_kp_256x2_32_32`     | 16,384 局面（1 局面 10 万ノード、置換表 16 MB）の全行が一致 |
| `nnue_halfkp_256x2_32_32` | 18 局面 × 4 段階で一致                                      |
| 上記以外                  | 未確認                                                      |

一致を確かめたのは、YaneuraOu の `make normal` に合わせた次のビルドです。

```bash
cargo build --release --no-default-features --features nnue_kp_256x2_32_32,ref_check_time
```

`--no-default-features` で `quick-draw`（千日手の早期打ち切り。YaneuraOu の `ENABLE_QUICK_DRAW` に当たる）を外し、`ref_check_time` で `go nodes N` の時間チェックを YaneuraOu と同じ位置に移します。

テストは [`crates/attic-protocol/tests/yaneuraou_parity.rs`](crates/attic-protocol/tests/yaneuraou_parity.rs) です（16,384 局面のうち 160 局面を使います）。評価関数ファイルは本リポジトリに含まれていないため、`ATTIC_KP256_NN` にパスを設定してください。

## 探索速度

YaneuraOu V9.80 を 1 とした比です（Intel Core i5-14400、`x86-64-v3`、1 スレッド、1 局面 10 万ノード × 128 局面を交互に測定）。

| ビルド                          | 速度比 |
| ------------------------------- | ------ |
| PGO（`cargo xtask build-pgo`）  | 1.020  |
| 通常（`cargo build --release`） | 0.976  |

PGO ビルドは YaneuraOu と同等以上ですが、通常ビルドは下回ります。この測定は 1 台の機械だけで行い、AVX-512 のカーネルと、SFNN・`sym_…` の速度は測っていません。

```bash
cargo xtask build-pgo --features <評価関数> --eval-dir <nn.bin のあるディレクトリ>
```

## 対応している評価関数

評価関数はビルド時に `attic` の feature で 1 つ選びます。実行時には切り替えられません。ファイルは `<EvalDir>/nn.bin`（既定は `eval/nn.bin`）として読み込みます。

| feature                                 | 備考                                                                      |
| --------------------------------------- | ------------------------------------------------------------------------- |
| `sfnn_halfkahm2_1536_15_32_k3k3`        | YaneuraOu の `YANEURAOU_ENGINE_SFNN1536` と同じ構成                       |
| `sfnn_halfkahm2_2304_15_64_progress8ek` | shielune 版 YaneuraOu の構成。`nn.bin` と同じ場所に `progress.bin` が必要 |
| `nnue_kp_256x2_32_32`                   | YaneuraOu の `YANEURAOU_ENGINE_NNUE_KP256`                                |
| `nnue_halfkp_256x2_32_32`               | YaneuraOu の `YANEURAOU_ENGINE_NNUE`                                      |
| `sym_kp_128x2_512`                      | Manaka 独自の構成（YaneuraOu にはない）                                   |
| `sym_halfkp_128x2_512`                  | Manaka 独自の構成（YaneuraOu にはない）                                   |
