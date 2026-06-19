# qsearch candidate probe の順序比較拡張

- 日時: 2026-06-20 01:05 UTC
- ブランチ: `tooling/qsearch-candidate-order-probe`
- 目的: qsearch 直接生成実験で、候補集合だけでなく候補順序の差も検出できるようにする。

## 追加したもの

`qsearch_candidate_probe` に以下を追加した。

```text
--all-plies  startpos moves の途中局面を全て検査する
--ordered    候補手をソートせず、生成順序込みで比較する
```

前回の check-drop 直接生成案は探索木が変化した。集合が同じでも同点ソート時の元順序差が探索木に影響する可能性があるため、順序比較を追加した。

## 検証

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin qsearch_candidate_probe
```

結果: 通過。

## probe

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/qsearch_candidate_probe \
  --input taya36.sfen \
  --all-plies \
  --ordered \
  --max-positions 2000 \
  --show 3
```

```text
positions: 2000
mismatches: 0
ordered: true
all plies: true
reference candidates: 3774
current candidates: 3774
```

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/qsearch_candidate_probe \
  --input /tmp/shogi_value_drop/drop_windows.sfen \
  --ordered \
  --show 3
```

```text
positions: 193
mismatches: 0
ordered: true
all plies: false
reference candidates: 1252
current candidates: 1252
```

## 判断

qsearch 直接生成の次回実験では、まずこのprobeの `--ordered --all-plies` を通してから `search_profile` に進める。探索木が変わった場合に、集合差なのか順序差なのかを切り分けやすくなった。
