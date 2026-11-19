# YaneuraOu との互換性

Attic の基準は [YaneuraOu V9.96](https://github.com/yaneurao/YaneuraOu/commit/b707119bbeeeb1972806c3371545d6056b30e061)（`b707119b`）です。

- YaneuraOu のビルドは `make normal`
- Attic の `ref_check_time` は有効、`quick-draw` は無効
- 千日手は同一局面の4回目で判定

```sh
cargo build --release -p attic --no-default-features --features <shape>,ref_check_time
```

## V9.96 系との AVX2 照合

- 比較対象は Attic と YaneuraOu V9.96の検証ビルド
- Linux AVX2、1スレッド、置換表16 MiB、`go nodes 10000`
- 9ネットワーク × 21,884局面。各ネットワークの評価設定は両エンジンで統一

探索は `time`、`nps`、`hashfull` を除く全 `info depth` 行と `bestmove` を比較しました。

| ネットワーク | 静的評価 | 探索結果 |
|---|---:|---:|
| `nnue_kp_256x2_32_32` | 21,884/21,884 (100%) | 20,233/20,233 (100%) |
| `nnue_halfkp_128x2_32_32` | 21,884/21,884 (100%) | 20,233/20,233 (100%) |
| `nnue_halfkp_256x2_32_32` | 21,884/21,884 (100%) | 20,233/20,233 (100%) |
| `nnue_halfkp_768x2_16_64` | 21,884/21,884 (100%) | 20,233/20,233 (100%) |
| `sfnn_halfkahm2_1024_15_64_progress8kpabs` | 21,884/21,884 (100%) | 20,233/20,233 (100%) |
| `sfnn_halfkahm2_2304_15_64_progress8` | 21,884/21,884 (100%) | 20,233/20,233 (100%) |
| `sfnn_halfkahm2_threat_1024_15_64_8gp` | 21,884/21,884 (100%) | 20,233/20,233 (100%) |
| `sfnn_ka2_2048_8_64_progress8x` | 21,884/21,884 (100%) | 20,233/20,233 (100%) |
| `sfnn_ka2_4096_8_64_progress8x` | 21,884/21,884 (100%) | 20,233/20,233 (100%) |

## 評価関数別のチューニング

### `sfnn_halfkahm2_2304_15_64_progress8`

このネットワークでは、専用互換ブランチで NAGISA V4 が採用しているパフォーマンスチューニングを行っています。

- 探索係数と指し手の採点
- 千日手判定
- 置換表の構成

[NAGISA V4](https://github.com/keinoda/YaneuraOu/releases/tag/nagisa-v4) · [調整内容（PR #66）](https://github.com/rshogi/attic/pull/66)

### `sfnn_halfkahm2_threat_1024_15_64_8gp`

このネットワークでは、Sirius が採用している評価パラメータに合わせています。

- 進行度による層スタックの選択
- 層スタックの境界での補間
- 評価値のスケーリング（`FV_SCALE`）

[Sirius](https://yfamate.booth.pm/items/9158204) · [調整内容（PR #51）](https://github.com/rshogi/attic/pull/51)
