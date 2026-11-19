# パフォーマンス

- CPU は Intel Core Ultra 7 265KF の P コア1つ（論理 CPU 7）
- スレッド数 1、置換表 16 MiB、定跡無効
- 同一局面を各ビルドで10回計測
- 評価ファイル読み込み後、`go nodes 1000000` から `bestmove` までを計測

計測局面

```text
sfen +B4g1nl/2g2k3/ppnppsbpp/4r1p2/5p2P/3S2P2/PPPPPP1P1/2GS2S2/LN2KG1NL w Lrp 32
```

## 探索速度比

Linux AVX2 と WASM SIMD128 の測定結果です。各ネットワークで **YaneuraOu AVX2 を 1.00×** とし、その平均探索時間を各ビルドの平均探索時間で割っています。

![YaneuraOu AVX2 を基準にした探索速度比](benchmark/attic-vs-yaneuraou.svg)

## NPS

各回の最後にエンジンが出力した `nps` の平均です。

![AVX2 と WASM の実測 NPS](benchmark/nps.svg)

## 互換性

[21,884局面の照合結果](compatibility.md#avx2)
