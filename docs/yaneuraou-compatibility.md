# YaneuraOu との互換性

Attic の基準は [YaneuraOu V9.96](https://github.com/yaneurao/YaneuraOu/commit/b707119bbeeeb1972806c3371545d6056b30e061)（`b707119b`）です。

- YaneuraOu のビルドは `make normal`
- Attic の `ref_check_time` は有効、`quick-draw` は無効
- 千日手は同一局面の4回目で判定

```sh
cargo build --release -p attic --no-default-features --features <shape>,ref_check_time
```

## V9.96 系との AVX2 照合

無効局面への応答を揃えた Attic と YaneuraOu V9.96（`b707119b` ベース）の検証ビルドで、9ネットワークを各21,884局面で照合しました。条件は Linux AVX2、1スレッド、置換表16 MiB、`go nodes 10000` です。

各ネットワークの静的評価は全21,884局面で一致しました。探索した20,233局面では、`time`、`nps`、`hashfull` を除いた全 `info depth` 行と `bestmove` が一致しました。最後に報告された評価値（`score`）も全件一致しました。

残りの1,651局面は、開始時点で手番側が相手玉に王手をかけている無効局面です。両エンジンとも理由を `info string` で通知し、探索や詰みの評価を行わずに `bestmove resign` を返しました。

`sfnn_halfkahm2_2304_15_64_progress8` は進行度による8バケット（`progress8kpabs`）と `FV_SCALE=28`、`sfnn_halfkahm2_threat_1024_15_64_8gp` は `FV_SCALE=26` で照合しました。

| ネットワーク | 静的評価 | 探索結果 | 無効局面の応答 |
|---|---:|---:|---:|
| `nnue_kp_256x2_32_32` | 21,884/21,884 (100%) | 20,233/20,233 (100%) | 1,651/1,651 (100%) |
| `nnue_halfkp_128x2_32_32` | 21,884/21,884 (100%) | 20,233/20,233 (100%) | 1,651/1,651 (100%) |
| `nnue_halfkp_256x2_32_32` | 21,884/21,884 (100%) | 20,233/20,233 (100%) | 1,651/1,651 (100%) |
| `nnue_halfkp_768x2_16_64` | 21,884/21,884 (100%) | 20,233/20,233 (100%) | 1,651/1,651 (100%) |
| `sfnn_halfkahm2_1024_15_64_progress8kpabs` | 21,884/21,884 (100%) | 20,233/20,233 (100%) | 1,651/1,651 (100%) |
| `sfnn_halfkahm2_2304_15_64_progress8` | 21,884/21,884 (100%) | 20,233/20,233 (100%) | 1,651/1,651 (100%) |
| `sfnn_halfkahm2_threat_1024_15_64_8gp` | 21,884/21,884 (100%) | 20,233/20,233 (100%) | 1,651/1,651 (100%) |
| `sfnn_ka2_2048_8_64_progress8x` | 21,884/21,884 (100%) | 20,233/20,233 (100%) | 1,651/1,651 (100%) |
| `sfnn_ka2_4096_8_64_progress8x` | 21,884/21,884 (100%) | 20,233/20,233 (100%) | 1,651/1,651 (100%) |

## 評価関数別のパラメータ調整

- **NAGISA V4**（`sfnn_halfkahm2_2304_15_64_progress8`）の専用互換ブランチでは、探索係数、指し手の採点、千日手判定、置換表を NAGISA V4 の公開実行ファイルに合わせています。上の表は YaneuraOu V9.96 との照合結果です。 [NAGISA V4](https://github.com/keinoda/YaneuraOu/releases/tag/nagisa-v4) · [調整内容（PR #66）](https://github.com/rshogi/attic/pull/66)
- **Sirius**（`sfnn_halfkahm2_threat_1024_15_64_8gp`）では、層スタックの選択と境界での補間、`FV_SCALE` を Sirius 本体に合わせています。 [Sirius](https://yfamate.booth.pm/items/9158204) · [調整内容（PR #51）](https://github.com/rshogi/attic/pull/51)
