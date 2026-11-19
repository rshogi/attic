# YaneuraOu との互換性

Attic の基準は [YaneuraOu V9.96](https://github.com/yaneurao/YaneuraOu/commit/b707119bbeeeb1972806c3371545d6056b30e061)（`b707119b`）です。

同じ局面・条件で探索し、各反復の深さ、評価値、ノード数、読み筋と `bestmove` を YaneuraOu の出力と行単位で照合しました。

照合には、YaneuraOu の `make normal` と同じく 4 回目の同一局面で千日手と判定するビルドを使います。`ref_check_time` は `go nodes N` の止まる位置を YaneuraOu に合わせます。`quick-draw` は YaneuraOu の `ENABLE_QUICK_DRAW`（`FOR_TOURNAMENT` ビルドで有効）に相当するため外します。

```sh
cargo build --release -p attic --no-default-features --features <shape>,ref_check_time
```

16,384 局面を各 10 万ノード、置換表 16 MB、1 スレッドで探索しました。`time`、`nps`、`hashfull` を除く出力行と `bestmove` がすべて一致した局面を数えています。探索ごとに `usinewgame` → `isready` → `position` → `go nodes 100000` を送ります。

## V9.96（`b707119b`）での検証

比較対象は YaneuraOu V9.96 の `b707119b`（`make normal TARGET_CPU=AVX2 COMPILER=g++`、`EXTRA_LDFLAGS=-static`）です。Attic は `ishidakei/main`（`e16990e`）を `develop`（`6e079d9`）にマージしたリビジョンで、musl 静的リンクおよび `-C target-cpu=x86-64-v3` でビルドしており、AVX2 カーネルを使います。

| 評価関数                  |   一致 |  総数 |   一致率 |
| ------------------------- | -----: | ----: | -------: |
| `nnue_kp_256x2_32_32`     |  16384 | 16384 | 100.000% |
| `nnue_halfkp_256x2_32_32` |  16384 | 16384 | 100.000% |
| 上記以外                  | 未確認 |     — |        — |

現行の 9 評価関数を使った 1 局面の比較は[パフォーマンス](performance.md#探索結果の照合)を参照してください。

## `sfnn_halfkahm2_1024_15_64_progress8ek`

この構成は `sfnn_halfkahm2_1024_15_64_progress8kpabs`（NAGISA V3）とファイル形式や進行度係数が共通で、層スタックの選択規則のみが異なります。相入玉局面（先手玉が 1〜5 段、後手玉が 5〜9 段）を 9 番目の層スタックに割り当てており、規則自体は `sfnn_halfkahm2_2304_15_64_progress8ek` と同等です。なお、どちらの規則で学習されたかはファイルから判別できません。手元の相入玉バケット付きモデル（tatara で学習後に変換）も、アーキテクチャ名やヘッダ・FT のハッシュ値は V3 と一致していました。使用するネットワークに適した構成をビルド時に選択してください。

- 配布ディレクトリの `eval_options.txt` に `LS_BUCKET_MODE` の行があっても、attic にはこのオプションがないため、`Error : option name not found : LS_BUCKET_MODE` を出して無視します。規則はビルドで決まります。手元の V3 の配布物は `progress8kpabs`、試作のネットのディレクトリは `progress8ek` と書いていました。

## `sfnn_halfkahm2_threat_1024_15_64_8gp`

Sirius の `nn.bin`（300,147,971 バイト。sha256 は `2d2f8e94…` で始まります）を、そのまま読む構成です。HalfKA_hm2 の特徴量に、盤上の駒どうしの利き（threat 特徴量）を足します。成り駒を含む玉以外の駒がすべて攻め手になり、飛び駒は最初に当たる駒で止まります。層スタックは 8 つで、手駒と敵陣の駒から整数で出す進行度と、玉の位置と守りの駒の数から選びます。境界の近くでは、隣の層スタックとの内分をとります。`FV_SCALE` の既定値は 26 で、評価値は ±31753 に収めます。`progress.bin` は使いません。

読み込めるのは、ヘッダハッシュが `0x684f7658`、アーキテクチャ文字列が空、threat プロファイルが 23 のファイルです。条件を満たさない場合は、理由を出力して読み込みを中止します。

静的評価は Sirius 本体の `e` コマンド（手番側から見た値）、探索は Sirius の評価関数を組み込んだ YaneuraOu と照合しました。Sirius 本体との探索一致は未検証です。

**静的評価**は、以下の 3 ビルドで 4 つの局面集合すべてが Sirius と一致しました。

- スカラー: `-C target-cpu=x86-64-v2`（AVX2 なし）
- AVX2: `-C target-cpu=x86-64-v3`
- ネイティブ: このマシン（Core Ultra 7 265KF）の AVX2 と AVX-VNNI。AVX-512 はありません

| 局面集合 | 内容                                  |   一致 |  総数 |
| -------- | ------------------------------------- | -----: | ----: |
| fixture  | 通常の局面                            |  16384 | 16384 |
| probe    | 入玉、駒の改変、駒の山                |   3000 |  3000 |
| packed   | 利きの一覧が 128 件の上限に達する局面 |    500 |   500 |
| moves    | 局面に指し手を進めた後                |   2000 |  2000 |

8 つの層スタックをすべて検証しました。局面数は順に 15,717、2,715、1,002、600、177、27、228、1,418 です。

**差分更新**は、手ごとに、全計算した累積値と一致します。利きの一覧も、全計算で作ったものと行の並びまで一致します。

- Sirius の局面集合の局面と、手で作った局面から始めた、ランダムな対局 477 局、計 40,049 手。取る手 12,021、打つ手 9,064、成る手 3,760、玉の手 11,551（うち、玉が左右反転の境界をまたぐ手が 2,026）を含みます。利きの一覧が上限に達した手が 77、2 つの視点で反転の有無が違う手が 19,940 です。3 つのビルドで同じ件数でした。
- 利きの一覧の（攻め手、対象）の組と並びは、別に書いた素朴な列挙（盤面の配列を走査）とも照合しています。行の番号は、上の Sirius との一致で確かめています。
- moves の 2,000 局面は、指し手を差分更新で進めた先の値も Sirius と一致しました（計 7,022 手）。

**探索の照合**では、上流 `b707119b` に Sirius の評価関数（読み込み、threat 特徴量、層スタックの選択と内分、FV_SCALE 26）を追加したビルド（`make normal`、`TARGET_CPU=AVX512VNNI`、静的リンク）と、Attic（`--no-default-features --features sfnn_halfkahm2_threat_1024_15_64_8gp,ref_check_time`）を、次の条件（16,384 局面、`go nodes 100000`、Threads 1、USI_Hash 16）で照合しました。16,384 局面すべてで全行が一致しています。実行環境はいずれも Xeon Silver 4210（AVX-512 対応）です。

**探索**の結果は、差分更新を行うビルドと毎手累積値を再生成するビルド（`threat_force_refresh`）で一致しました。Sirius の局面集合から抽出した 584 局面を `go depth 9`（Threads 1、USI_Hash 16）で探索し、`info` 行（`time`・`nps`・`hashfull` を除く）と `bestmove` の計 5,648 行を比較しました。同一バイナリを 2 回実行した場合も一致しました。[scripts/sirius_search_check.py](../scripts/sirius_search_check.py) で再現できます。

AVX-512 を持つ CPU（Xeon Silver 4210、`-C target-cpu=native`）でも、静的評価の 4 つの局面集合、差分更新の moves 2,000 局面、差分更新と作り直しの探索の比較（398 局面、`go depth 9`）が、すべて一致しました。この構成に AVX-512 のカーネルはなく、AVX-512 のマシンでも AVX2 のカーネルが動きます。

未検証の条件は以下のとおりです。

- 2 スレッド以上での探索。
- `FV_SCALE` が 26 以外のときの Sirius の値。Sirius と照合したのは 26 の値だけです。26 以外の割り算と ±31753 の丸めは、Attic 自身の出力（`FV_SCALE 1`）と比べています。fixture の最大値は 6,992 で、丸めに届く局面はありません。

Sirius の `nn.bin` と局面ファイルは配布物に含まれません。次のテストでは、環境変数 `ATTIC_SIRIUS_REF` に参照データのディレクトリを指定します。未設定の場合は通知を出してスキップします。指定先には `eval/nn.bin`、`fixture/`、`eval-re/` を置きます。カーネルはコンパイル時に決まるので、スカラーや AVX2 を試すには `RUSTFLAGS` で `target-cpu` を指定します。

```sh
ATTIC_SIRIUS_REF=<sirius-ref> RUSTFLAGS="-C target-cpu=x86-64-v2" \
  cargo test --release -p attic-eval --features sfnn_halfkahm2_threat_1024_15_64_8gp \
  --test sirius_parity --test sirius_incremental --test sirius_fv_scale -- --nocapture
```

## つようら王11 Micro / Mini

配布の評価ファイルを使い、`FV_SCALE=24` でやねうら王の `e` コマンドと照合した結果、Micro / Mini とも静的評価が 2,049 局面すべてで一致しました。1 局面の探索出力の比較は[パフォーマンス](performance.md#探索結果の照合)を参照してください。
