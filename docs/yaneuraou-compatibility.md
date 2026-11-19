# YaneuraOu との互換性

Attic の基準は [YaneuraOu V9.96](https://github.com/yaneurao/YaneuraOu/commit/b707119bbeeeb1972806c3371545d6056b30e061)（`b707119b`）です。

照合には、YaneuraOu の `make normal` と同じく 4 回目の同一局面で千日手と判定するビルドを使います。`ref_check_time` は `go nodes N` の止まる位置を YaneuraOu に合わせます。`quick-draw` は YaneuraOu の `ENABLE_QUICK_DRAW`（`FOR_TOURNAMENT` ビルドで有効）に相当するため外します。

```sh
cargo build --release -p attic --no-default-features --features <shape>,ref_check_time
```

## V9.96 系との AVX2 照合

YaneuraOu V9.96（`b707119b`）と Attic の検証ビルドを、各ネットワークで同じ21,884局面を使って照合しています。探索は AVX2 ビルド同士で `go nodes 100000`、1スレッド、置換表16 MiBです。

静的評価は `eval` の結果を比較し、全9ネットワークで21,884局面すべて一致しました。探索結果は `time`、`nps`、`hashfull` を除いた全 `info depth` 行と `bestmove` を比較します。探索の最終評価値は、最後の `info depth` 行の `score` です。

探索の照合は進行中です。表は2026年11月19日時点で一致を確認できた件数と、全21,884局面に対する割合です。実行中やエンジンが途中で終了した局面も分母に含みます。

| ネットワーク | 探索結果の一致 | 探索の最終評価値の一致 |
|---|---:|---:|
| `nnue_kp_256x2_32_32` | 19,865 (90.77%) | 20,113 (91.91%) |
| `nnue_halfkp_128x2_32_32` | 19,234 (87.89%) | 19,234 (87.89%) |
| `nnue_halfkp_256x2_32_32` | 19,234 (87.89%) | 19,234 (87.89%) |
| `nnue_halfkp_768x2_16_64` | 19,093 (87.25%) | 19,093 (87.25%) |
| `sfnn_halfkahm2_1024_15_64_progress8kpabs` | 18,875 (86.25%) | 18,875 (86.25%) |
| `sfnn_halfkahm2_2304_15_64_progress8` | 18,673 (85.33%) | 18,673 (85.33%) |
| `sfnn_halfkahm2_threat_1024_15_64_8gp` | 18,615 (85.06%) | 18,615 (85.06%) |
| `sfnn_ka2_2048_8_64_progress8x` | 18,703 (85.46%) | 18,703 (85.46%) |
| `sfnn_ka2_4096_8_64_progress8x` | 18,567 (84.84%) | 18,568 (84.85%) |

## `sfnn_halfkahm2_1024_15_64_progress8ek`

この構成は `sfnn_halfkahm2_1024_15_64_progress8kpabs` とファイル形式や進行度係数が共通で、層スタックの選択規則が異なります。先手玉が1〜5段、後手玉が5〜9段にいる相入玉局面は、`sfnn_halfkahm2_2304_15_64_progress8ek` と同じく9番目の層スタックを使います。

モデルファイルから学習時の選択規則は判別できません。Attic の選択規則はビルド時に決まり、`eval_options.txt` に `LS_BUCKET_MODE` がある場合は警告を出してその設定を無視します。

## `sfnn_halfkahm2_threat_1024_15_64_8gp`

HalfKA_hm2 の特徴量に、盤上の駒どうしの利き（threat 特徴量）を足します。成り駒を含む玉以外の駒がすべて攻め手になり、飛び駒は最初に当たる駒で止まります。層スタックは 8 つで、手駒と敵陣の駒から整数で出す進行度と、玉の位置と守りの駒の数から選びます。境界の近くでは、隣の層スタックとの内分をとります。`FV_SCALE` の既定値は 26 で、評価値は ±31753 に収めます。`progress.bin` は使いません。

読み込めるのは、ヘッダハッシュが `0x684f7658`、アーキテクチャ文字列が空、threat プロファイルが 23 のファイルです。条件を満たさない場合は、理由を出力して読み込みを中止します。
