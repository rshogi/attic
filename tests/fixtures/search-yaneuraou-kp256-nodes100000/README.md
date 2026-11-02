# search-yaneuraou-kp256-nodes100000

YaneuraOu 9.80 (upstream `d47c9bfb`, `YANEURAOU_ENGINE_NNUE_KP256`, `make normal`) が
`go nodes 100000` で出力したものそのもの。Attic の探索が YaneuraOu と一致することを、
`crates/attic-protocol/tests/yaneuraou_parity.rs` が確かめる。

## 中身

- `fixture.jsonl` — 1 行 1 局面。`{"idx", "sfen", "info": [...], "bestmove": "..."}`。
  - `info` は `info depth` 行を出力順に全部。`time` / `nps` / `hashfull` は除いてある。
  - `bestmove` は ponder 手まで含む 1 行(`bestmove 5d8d ponder 3f3e`)。
- `manifest.json` — 何で作ったか。YaneuraOu のバイナリと評価ファイルの sha256、USI オプション、
  探索ごとの手順。
- この置き場所のファイルは 160 局面。全 16,384 局面を `idx` 順に並べて 128 局面おきに取った 128 局面と、
  ある修正を外すと YaneuraOu と食い違う局面 32 個(`manifest.json` の `sentinels`)。
  局面は `start_sfens_ply32_cp100.txt`(互角局面集)から、種 20261101 で重複なしに選んだもの。

## sentinel が守っているもの

16,384 局面で、修正を 1 つずつ外した Attic を同じ fixture と比べた結果。

| 外した修正 | 食い違う局面(16,384 中) | 入れた数 |
| :-- | --: | --: |
| `search_movegen.rs`: 王手回避の占有から王を除く(`movegen-occ`) | 8 | 8 |
| `tt.rs`: 偽のキー一致で、変換できない手の項を読み飛ばす(`tt-probe-skip`) | 68 | 12 |
| `qsearch.rs`: TT ヒット時に静的評価を引き直す(`tt-reeval`) | 306 | 12 |
| `search_movegen.rs`: `check_squares[KING]` を空にする | 0 | 0 |

等間隔の標本だけだと、`movegen-occ`(全体の 0.05%)はまず入らないので、食い違う局面を直接入れてある。
最後の `check_squares[KING]` は、YaneuraOu の定義に合わせた修正だが、この 16,384 局面では結果を変えない。
テストはこの修正を守っていない。

## 条件

Threads=1、`USI_Hash` 16 MB、MultiPV=1、`FV_SCALE` 16、`PvInterval` 0、定跡なし。
探索ごとに `usinewgame` → `isready`(`readyok` を待つ)→ `position sfen` → `go nodes 100000`。
評価関数は KP256(`manifest.json` の `eval_sha256` の `nn.bin`)。

## 走らせ方

一致するのは次のビルドだけ。

```sh
ATTIC_KP256_NN=/path/to/kp256/nn.bin \
cargo test --release -p attic-protocol --test yaneuraou_parity \
    --no-default-features --features nnue_kp_256x2_32_32,ref_check_time
```

- `--no-default-features` は `quick-draw`(千日手の早期打ち切り)を外す。参照の YaneuraOu は
  `ENABLE_QUICK_DRAW` なしの `make normal`。
- `ref_check_time` は `go nodes N` を YaneuraOu と同じ位置で止める。
- `nn.bin` が無いとテストは通知を出して通る。`ATTIC_REQUIRE_PARITY=1` を付けると失敗にする。

全 16,384 局面で走らせるときは、完全版の `fixture.jsonl` と `manifest.json` を別のディレクトリに
置いて `ATTIC_YO_FIXTURE_DIR` で指す。
