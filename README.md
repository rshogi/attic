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

| 評価関数                                   | 確認した内容                                                    |
| ------------------------------------------ | --------------------------------------------------------------- |
| `nnue_kp_256x2_32_32`                      | 1 局面 10 万ノード、置換表 16 MB で 16,384 局面の全行が一致     |
| `nnue_halfkp_256x2_32_32`                  | 1 局面 10 万ノード、置換表 16 MB で 16,384 局面の全行が一致     |
| `nnue_halfkp_128x2_32_32`                  | 1 局面 10 万ノード、置換表 16 MB で 16,384 局面の全行が一致     |
| `nnue_halfkp_768x2_16_64`                  | 1 局面 10 万ノード、置換表 16 MB で 16,384 局面の全行が一致     |
| `sfnn_halfkahm2_1024_15_64_progress8kpabs` | shielune 版 YaneuraOu と、16,384 局面中 16,353 局面の全行が一致 |
| `sfnn_halfkahm2_2304_15_64_progress8ek`    | shielune 版 YaneuraOu と、16,384 局面中 16,330 局面の全行が一致 |
| `sfnn_halfkahm2_1024_15_64_progress8ek`    | 固定深さ 10 の序盤 256 局面と相入玉 200 局面で全行が一致（下記） |
| 上記以外                                   | 未確認                                                          |

上の 2 つの SFNN は、YaneuraOu V9.80 に層スタックの選び方がないため、shielune 版と比べています。一致しなかった局面の原因は調べていません。同じ shielune 版を KP256 でビルドして同じ局面を探索しても、V9.80 の KP256 と 26 局面で違います。

### `sfnn_halfkahm2_1024_15_64_progress8ek` の確認

この構成は `sfnn_halfkahm2_1024_15_64_progress8kpabs`（NAGISA V3）と、ファイルの形も進行度の係数も同じで、層スタックの選び方だけが違います。相入玉の局面（先手玉が 1〜5 段、後手玉が 5〜9 段）を 9 番目の層スタックに割り当てます。規則は `sfnn_halfkahm2_2304_15_64_progress8ek` と同じです。どちらの規則で学習したネットかは、ファイルから分かりません。手元の、相入玉の bucket つきで学習したネット（tatara で学習して変換したもの）は、アーキテクチャ名もヘッダと FT のハッシュも V3 と同じでした。使うネットに合う構成を、ビルドで選んでください。

固定深さで YaneuraOu と突き合わせた結果です。参照は、shielune/YaneuraOu（`8b55d5d`）の `source/bitboard.h` の `least_significant_square_bb` の 1 行を V9.80 の形に戻したもの（`Bitboard(q0 & -q0, 0) : Bitboard(0, q1 & -q1)`）で、`LS_BUCKET_MODE=progress8ek` を設定しています。この 1 行を戻さない参照では、桂の SEE で桂が 2 枚同時に消え、探索が食い違う局面が出ます。上の 16,353 局面と 16,330 局面の食い違いの原因が、これかどうかは確かめていません。

| ネットワーク                                                         | 静的評価（520 局面）  | `go depth 10`、序盤 256 局面 | `go depth 10`、相入玉 200 局面（2,172 行） |
| -------------------------------------------------------------------- | --------------------- | ---------------------------- | ------------------------------------------ |
| NAGISA V3（`halfka1024`）                                            | 全局面で一致          | 全行が一致（2,816 行）       | 全行が一致                                 |
| 相入玉の bucket つきで学習した 1024 のネット（8 superbatch の試作） | 全局面で一致          | 全行が一致（2,817 行）       | 全行が一致                                 |

- 規則を取り違えると食い違います。上の 2 つのネットを `sfnn_halfkahm2_1024_15_64_progress8kpabs` のビルドで読むと、YaneuraOu（`progress8ek`）と合わない局面が出ます。静的評価では 520 局面中 509 局面が一致し（食い違う 11 局面は、520 局面のうちの相入玉の局面すべて）、相入玉 200 局面の探索で全行が一致したのは、試作のネットで 14 局面、V3 で 10 局面でした。逆に、V3 を YaneuraOu の `progress8kpabs` と `sfnn_halfkahm2_1024_15_64_progress8ek` で比べても、同じ数しか一致しません。
- 序盤 256 局面には相入玉の局面がないため、規則を取り違えても出力は変わりません。この 256 局面だけでは、2 つの構成を区別できません。
- 試作のネットは学習が短く、9 番目の層スタックが他とあまり違わないため、規則を取り違えたときの評価値の差は小さい（最大 16）です。V3 では 565 でした。
- 相入玉 200 局面は、開始局面から玉を前へ進める偏りをつけた乱数の指し手列で作り、先手玉が 1〜5 段、後手玉が 5〜9 段にいることを検査したものです。
- 既存の `sfnn_halfkahm2_1024_15_64_progress8kpabs` と `sfnn_halfkahm2_2304_15_64_progress8ek` は、この構成を足す前後で、静的評価 520 局面の値が同じで、`go depth 10` の出力（序盤 256 局面、相入玉 200 局面）も同じでした。
- 配布ディレクトリの `eval_options.txt` に `LS_BUCKET_MODE` の行があっても、attic にはこのオプションがないため、`Error : option name not found : LS_BUCKET_MODE` を出して無視します。規則はビルドで決まります。手元の V3 の配布物は `progress8kpabs`、試作のネットのディレクトリは `progress8ek` と書いていました。

## 探索速度

YaneuraOu を 1 とした比です。YaneuraOu 側も同じ評価関数で動かしています。1 スレッドで、1 局面 10 万ノード × 128 局面を交互に測定しました。

| ビルド                                     | コマンド                | 速度比 |
| ------------------------------------------ | ----------------------- | ------ |
| `nnue_kp_256x2_32_32`                      | `cargo xtask build-pgo` | 1.053  |
| `nnue_kp_256x2_32_32`                      | `cargo build --release` | 1.012  |
| `nnue_halfkp_256x2_32_32`                  | `cargo xtask build-pgo` | 1.022  |
| `nnue_halfkp_256x2_32_32`                  | `cargo build --release` | 0.980  |
| `nnue_halfkp_128x2_32_32`                  | `cargo xtask build-pgo` | 0.998  |
| `nnue_halfkp_128x2_32_32`                  | `cargo build --release` | 0.946  |
| `nnue_halfkp_768x2_16_64`                  | `cargo xtask build-pgo` | 1.008  |
| `nnue_halfkp_768x2_16_64`                  | `cargo build --release` | 0.971  |
| `sfnn_halfkahm2_1024_15_64_progress8kpabs` | `cargo xtask build-pgo` | 0.424  |
| `sfnn_halfkahm2_1024_15_64_progress8kpabs` | `cargo build --release` | 0.410  |
| `sfnn_halfkahm2_2304_15_64_progress8ek`    | `cargo xtask build-pgo` | 0.416  |
| `sfnn_halfkahm2_2304_15_64_progress8ek`    | `cargo build --release` | 0.416  |

SFNN の 2 つは、YaneuraOu 側が shielune 版です。遅い原因は調べていません。AVX-512 のカーネルと、表にない評価関数の速度は測っていません。

## 対応している評価関数

| type | feature                               |
| ---- | ------------------------------------- |
| SFNN | `halfkahm2_1536_15_32_k3k3`           |
| SFNN | `halfkahm2_1024_15_64_progress8kpabs` |
| SFNN | `halfkahm2_1024_15_64_progress8ek`    |
| SFNN | `halfkahm2_2304_15_64_progress8ek`    |
| NNUE | `kp_256x2_32_32`                      |
| NNUE | `halfkp_256x2_32_32`                  |
| NNUE | `halfkp_128x2_32_32`                  |
| NNUE | `halfkp_768x2_16_64`                  |
| SYM  | `kp_128x2_512`                        |
| SYM  | `halfkp_128x2_512`                    |
