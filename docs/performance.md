# パフォーマンス

## Linux AVX2 と WASM SIMD128

Intel Core Ultra 7 265KF の P コア（論理 CPU 6）に固定し、1 スレッド、置換表 16 MiB、定跡無効で探索しました。
同じ局面（fixture 565）を各条件で 10 回探索し、毎回 `usinewgame` と `isready` を送ります。
`go nodes 1000000` から `bestmove` までの経過時間を測ります。評価ファイルは計測前に読み込みます。

![やねうら王を1倍とした Attic の速度比](benchmark/attic-vs-yaneuraou.svg)

### 探索速度比

YaneuraOu を 1.00 とし、YaneuraOu の平均秒数を Attic の平均秒数で割っています。

| ネットワーク | Linux AVX2 | WASM SIMD128 |
|---|---:|---:|
| `nnue_kp_256x2_32_32` | 1.11× | 1.14× |
| `nnue_halfkp_128x2_32_32` | 1.01× | 1.01× |
| `nnue_halfkp_256x2_32_32` | 1.04× | 1.07× |
| `nnue_halfkp_768x2_16_64` | 1.26× | 1.14× |
| `sfnn_halfkahm2_1024_15_64_progress8kpabs` | 0.98× | 1.06× |
| `sfnn_halfkahm2_2304_15_64_progress8ek` | 0.94× | 1.00× |
| `sfnn_ka2_2048_8_64_progress8x` | 1.91× | 1.89× |
| `sfnn_ka2_4096_8_64_progress8x` | 2.21× | 2.20× |
| `sfnn_halfkahm2_threat_1024_15_64_8gp` | 1.28× | 1.20× |

### NPS

![AVX2 と WASM の実測 NPS を Attic と YaneuraOu で比較](benchmark/nps.svg)

各条件の 10 回で探索した実ノード数の合計を、計測した経過時間の合計で割っています。単位はノード/秒です。

| ネットワーク | AVX2 Attic | AVX2 YaneuraOu | WASM Attic | WASM YaneuraOu |
|---|---:|---:|---:|---:|
| `nnue_kp_256x2_32_32` | 2,153,970 | 1,947,807 | 1,135,123 | 997,651 |
| `nnue_halfkp_128x2_32_32` | 2,243,614 | 2,220,612 | 1,287,181 | 1,276,309 |
| `nnue_halfkp_256x2_32_32` | 1,990,282 | 1,911,921 | 1,079,088 | 1,006,710 |
| `nnue_halfkp_768x2_16_64` | 1,636,779 | 1,301,233 | 846,631 | 745,023 |
| `sfnn_halfkahm2_1024_15_64_progress8kpabs` | 1,471,397 | 1,497,626 | 791,332 | 746,435 |
| `sfnn_halfkahm2_2304_15_64_progress8ek` | 987,860 | 1,054,924 | 521,166 | 522,542 |
| `sfnn_ka2_2048_8_64_progress8x` | 1,285,372 | 674,476 | 703,405 | 371,207 |
| `sfnn_ka2_4096_8_64_progress8x` | 843,976 | 381,686 | 464,690 | 211,410 |
| `sfnn_halfkahm2_threat_1024_15_64_8gp` | 756,322 | 591,093 | 449,975 | 373,509 |

平均秒数と標準偏差は [CSV](benchmark/results.csv) を参照してください。

## ビルド条件

計測に使った局面です。

```text
sfen +B4g1nl/2g2k3/ppnppsbpp/4r1p2/5p2P/3S2P2/PPPPPP1P1/2GS2S2/LN2KG1NL w Lrp 32
```

- Attic Linux は `e6a5dfe` と同じソースツリー（`7953ccf`）、Rust release、`target-cpu=x86-64-v3`、`ref_check_time`。
- Attic WASM は同じベースに実験版の WASI ホスト対応と SIMD128 評価カーネルを重ねた単一スレッドビルド。Node.js 26.5.0 で実行。
- やねうら王は V9.96 系。進行度付きネットワークには対応するビルド、`sfnn_halfkahm2_threat_1024_15_64_8gp` には評価関数を追加したパッチ版を使いました。WASM は Emscripten SIMD128 を同じ Node.js で実行。
- `FV_SCALE` は両エンジンで共通。`sfnn_ka2_*` は 24、`sfnn_halfkahm2_*_progress8*` は 28、`sfnn_halfkahm2_threat_1024_15_64_8gp` は 26 です。

## 探索結果の照合

時間、NPS、hashfull を除く `info depth` と `bestmove` を比較しました。出力が異なる行では、速度比に探索内容の違いも反映されます。

| 評価関数 | 環境 | 最終ノード数 | bestmove | 探索出力全体 |
|---|---|---:|---:|---:|
| `nnue_kp_256x2_32_32` | Linux AVX2 | 10/10 | 10/10 | 10/10 |
| `nnue_kp_256x2_32_32` | WASM SIMD128 | 10/10 | 10/10 | 10/10 |
| `nnue_halfkp_128x2_32_32` | Linux AVX2 | 10/10 | 10/10 | 10/10 |
| `nnue_halfkp_128x2_32_32` | WASM SIMD128 | 10/10 | 10/10 | 10/10 |
| `nnue_halfkp_256x2_32_32` | Linux AVX2 | 10/10 | 10/10 | 10/10 |
| `nnue_halfkp_256x2_32_32` | WASM SIMD128 | 10/10 | 10/10 | 10/10 |
| `nnue_halfkp_768x2_16_64` | Linux AVX2 | 10/10 | 10/10 | 10/10 |
| `nnue_halfkp_768x2_16_64` | WASM SIMD128 | 10/10 | 10/10 | 10/10 |
| `sfnn_halfkahm2_1024_15_64_progress8kpabs` | Linux AVX2 | 0/10 | 10/10 | 0/10 |
| `sfnn_halfkahm2_1024_15_64_progress8kpabs` | WASM SIMD128 | 0/10 | 8/10 | 0/10 |
| `sfnn_halfkahm2_2304_15_64_progress8ek` | Linux AVX2 | 0/10 | 10/10 | 0/10 |
| `sfnn_halfkahm2_2304_15_64_progress8ek` | WASM SIMD128 | 0/10 | 4/10 | 0/10 |
| `sfnn_ka2_2048_8_64_progress8x` | Linux AVX2 | 0/10 | 10/10 | 0/10 |
| `sfnn_ka2_2048_8_64_progress8x` | WASM SIMD128 | 0/10 | 9/10 | 0/10 |
| `sfnn_ka2_4096_8_64_progress8x` | Linux AVX2 | 0/10 | 0/10 | 0/10 |
| `sfnn_ka2_4096_8_64_progress8x` | WASM SIMD128 | 0/10 | 1/10 | 0/10 |
| `sfnn_halfkahm2_threat_1024_15_64_8gp` | Linux AVX2 | 10/10 | 10/10 | 10/10 |
| `sfnn_halfkahm2_threat_1024_15_64_8gp` | WASM SIMD128 | 10/10 | 10/10 | 10/10 |

## Attic 内の計測

`bench [ttSizeMB] [threads] [limit] [default|current|<fenFile>] [limitType]` は固定条件で NPS を計測します。引数はすべて省略でき、既定値は置換表 1024 MB、1 スレッド、`limit` 15000、`limitType` は `movetime` です。

### `sfnn_halfkahm2_threat_1024_15_64_8gp` の差分更新

[互換性の検証](yaneuraou-compatibility.md#sfnn_halfkahm2_threat_1024_15_64_8gp)で使った 584 局面（depth 9、Threads 1）を P コア 1 つに固定して測定しました。差分更新ビルドは 3 回の計測で 644,520〜693,304 NPS（約 674 万ノード）、全計算ビルドは 132,875 と 187,754 NPS でした。他のプロセスも動作しており、全計算ビルドの測定値には大きなばらつきがあります。
