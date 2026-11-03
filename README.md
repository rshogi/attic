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

| ビルド                    | コマンド                | 速度比 |
| ------------------------- | ----------------------- | ------ |
| `nnue_kp_256x2_32_32`     | `cargo xtask build-pgo` | 1.020  |
| `nnue_kp_256x2_32_32`     | `cargo build --release` | 0.976  |
| `nnue_halfkp_128x2_32_32` | `cargo xtask build-pgo` | 1.029  |
| `nnue_halfkp_128x2_32_32` | `cargo build --release` | 0.999  |
| `nnue_halfkp_768x2_16_64` | `cargo xtask build-pgo` | 1.050  |
| `nnue_halfkp_768x2_16_64` | `cargo build --release` | 1.042  |

AVX-512 のカーネルと、表にない評価関数の速度は測っていません。

## 対応している評価関数

| type | feature                            |
| ---- | ---------------------------------- |
| SFNN | `halfkahm2_1536_15_32_k3k3`        |
| SFNN | `halfkahm2_2304_15_64_progress8ek` |
| NNUE | `kp_256x2_32_32`                   |
| NNUE | `halfkp_256x2_32_32`               |
| NNUE | `halfkp_128x2_32_32`               |
| NNUE | `halfkp_768x2_16_64`               |
| SYM  | `kp_128x2_512`                     |
| SYM  | `halfkp_128x2_512`                 |
