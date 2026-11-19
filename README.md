# attic-shogi

Rust 製の [YaneuraOu](https://github.com/yaneurao/YaneuraOu) 互換探索エンジンです。ベースは [V9.96](https://github.com/yaneurao/YaneuraOu/commit/b707119bbeeeb1972806c3371545d6056b30e061) で、`usi` の `id name` は `Attic 9.96git` です。

- [パフォーマンス](https://github.com/rshogi/attic/blob/master/docs/performance.md)
- [YaneuraOu との互換性](https://github.com/rshogi/attic/blob/master/docs/yaneuraou-compatibility.md)

## ビルドと実行

評価関数に合う feature を 1 つ選びます。KP256 の場合は次のとおりです。

```sh
cargo build --release -p attic --no-default-features --features nnue_kp_256x2_32_32
```

評価ファイルを `EvalDir` に配置し、USI 対応 GUI に `target/release/attic` を登録します。引数なしで起動すると、標準入出力で USI コマンドを受け付けます。

- `usi` はオプション一覧を返します。`setoption` で設定し、`isready` で評価関数や定跡を読み込みます。
- `position` で局面を設定し、`go` で探索します。`bench` で NPS を計測できます。

## 対応している評価関数

Cargo feature は、表の種類を小文字にして名前の前に付けます。区切りは `_` です。

| type | feature                               |
| ---- | ------------------------------------- |
| SFNN | `halfkahm2_1536_15_32_k3k3`           |
| SFNN | `halfkahm2_1024_15_64_progress8kpabs` |
| SFNN | `halfkahm2_1024_15_64_progress8ek`    |
| SFNN | `halfkahm2_2304_15_64_progress8ek`    |
| SFNN | `halfkahm2_threat_1024_15_64_8gp`     |
| SFNN | `ka2_2048_8_64_progress8x`             |
| SFNN | `ka2_4096_8_64_progress8x`             |
| NNUE | `kp_256x2_32_32`                      |
| NNUE | `kp_128x2_32_32`                      |
| NNUE | `halfkp_256x2_32_32`                  |
| NNUE | `halfkp_128x2_32_32`                  |
| NNUE | `halfkp_768x2_16_64`                  |
| SYNN | `token_128x2_32_32`                   |
| SYNN | `token_256x2_32_32`                   |
| SYNN | `kp_128x2_32_32`                      |
| SYNN | `kp_256x2_32_32`                      |
| SYNN | `halfkp_128x2_32_32`                  |
| SYNN | `token_128x2_32_32_pair16`            |
| SYNN | `token_128x2_32_32_pair16_o1`         |
| SYNN | `token_128x2_32_32_pair16_o1_hm`      |
| SYNN | `token_128x2_32_32_pair16_m8_f2`      |
| HCE  | `material_17`                         |
| HCE  | `mobility_107_57`                     |
