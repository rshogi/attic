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

条件は、1 局面 10 万ノード、置換表 16 MB、1 スレッドで、16,384 局面です。比較したのは、`time`・`nps`・`hashfull` を除く全 `info` 行と、ponder 手を含む `bestmove` です。Attic は `--no-default-features --features <評価関数>,ref_check_time` でビルドしています。

| 評価関数                                | 比較先                             | 全行が一致した局面 |
| --------------------------------------- | ---------------------------------- | -----------------: |
| `nnue_kp_256x2_32_32`                   | YaneuraOu V9.80                    |    16,384 / 16,384 |
| `nnue_halfkp_256x2_32_32`               | YaneuraOu V9.80                    |    16,384 / 16,384 |
| `nnue_halfkp_128x2_32_32`               | YaneuraOu V9.80                    |    16,384 / 16,384 |
| `nnue_halfkp_768x2_16_64`               | YaneuraOu V9.80                    |    16,384 / 16,384 |
| `sfnn_halfkahm2_2304_15_64_progress8ek` | shielune 版 YaneuraOu（`8b55d5d`） |    16,330 / 16,384 |
| 上記以外                                | —                                  |             未確認 |

- `sfnn_halfkahm2_2304_15_64_progress8ek` の比較先は V9.80 ではありません。V9.80 には `progress8ek` の層スタックがないため、shielune 版を `LS_BUCKET_MODE progress8ek` で動かしています。
- 一致しなかった 54 局面は、原因を調べていません。同じ shielune 版を KP256 でビルドして同じ 16,384 局面を探索させると、V9.80 の KP256 と 26 局面で違います。この版の探索が V9.80 と同じではないことは分かっていますが、54 局面がすべてそれで説明できるかは確かめていません。
- `progress.bin` は使われています。外すと、確認した 40 局面のすべてで結果が変わります。
- `sfnn_halfkahm2_1536_15_32_k3k3` は評価関数のファイルがなく、`sym_kp_128x2_512` と `sym_halfkp_128x2_512` は YaneuraOu に対応する評価関数がないため、比較していません。

## 探索速度

YaneuraOu を 1 とした比です。YaneuraOu 側も同じ評価関数で動かしています。1 スレッドで、1 局面 10 万ノード × 128 局面を、YaneuraOu・`cargo build --release`・`cargo xtask build-pgo` の順を入れ替えながら 6 回ずつ測定しました。値は 6 回の比の平均 ± 95% 信頼区間です。他の仕事が 0.5 コアを超えて動いていた回は除いています。

| ビルド                                  | コマンド                | 速度比        |
| --------------------------------------- | ----------------------- | ------------- |
| `nnue_kp_256x2_32_32`                   | `cargo xtask build-pgo` | 1.053 ± 0.008 |
| `nnue_kp_256x2_32_32`                   | `cargo build --release` | 1.012 ± 0.006 |
| `nnue_halfkp_256x2_32_32`               | `cargo xtask build-pgo` | 1.022 ± 0.011 |
| `nnue_halfkp_256x2_32_32`               | `cargo build --release` | 0.980 ± 0.012 |
| `nnue_halfkp_128x2_32_32`               | `cargo xtask build-pgo` | 0.998 ± 0.002 |
| `nnue_halfkp_128x2_32_32`               | `cargo build --release` | 0.946 ± 0.008 |
| `nnue_halfkp_768x2_16_64`               | `cargo xtask build-pgo` | 1.008 ± 0.005 |
| `nnue_halfkp_768x2_16_64`               | `cargo build --release` | 0.971 ± 0.006 |
| `sfnn_halfkahm2_2304_15_64_progress8ek` | `cargo xtask build-pgo` | 0.416 ± 0.002 |
| `sfnn_halfkahm2_2304_15_64_progress8ek` | `cargo build --release` | 0.416 ± 0.002 |

- Core i5-14400 です。Attic は `-C target-cpu=native`（AVX2 と AVX-VNNI）、YaneuraOu は `TARGET_CPU=AVX2` でビルドしています。AVX-512 のカーネルは測っていません。
- 4 つの NNUE は、`cargo xtask build-pgo` で 0.998〜1.053、`cargo build --release` で 0.946〜1.012 です。どちらも、すべての評価関数で YaneuraOu 以上になるわけではありません。
- `sfnn_halfkahm2_2304_15_64_progress8ek` は YaneuraOu の 0.42 倍です。YaneuraOu 側は shielune 版（`8b55d5d`）で、PGO の有無で変わりません。原因は調べていません。
- `sfnn_halfkahm2_1536_15_32_k3k3`、`sym_kp_128x2_512`、`sym_halfkp_128x2_512` は測っていません。

## 対応している評価関数

| type | feature                               |
| ---- | ------------------------------------- |
| SFNN | `halfkahm2_1536_15_32_k3k3`           |
| SFNN | `halfkahm2_1024_15_64_progress8kpabs` |
| SFNN | `halfkahm2_2304_15_64_progress8ek`    |
| NNUE | `kp_256x2_32_32`                      |
| NNUE | `halfkp_256x2_32_32`                  |
| NNUE | `halfkp_128x2_32_32`                  |
| NNUE | `halfkp_768x2_16_64`                  |
| SYM  | `kp_128x2_512`                        |
| SYM  | `halfkp_128x2_512`                    |
