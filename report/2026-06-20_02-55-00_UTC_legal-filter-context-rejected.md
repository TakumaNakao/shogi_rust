# legal filter context rejected

- 作成日時: 2026-06-20 02:55:00 UTC
- 実験ブランチ: `perf/legal-filter-context`
- 実験コミット: `3b31917`
- 判断: 不採用

## 背景

GPT-5.5 xhigh の提案に従い、`Position::legal_moves()` の合法性フィルタで毎手計算している値を、ループ前に1回だけ計算する高速化を試した。

目的は探索木を一切変えず、合法手生成だけを速くすること。

## 実装

`shogi_lib/src/movegen.rs`:

- `LegalFilterContext` を追加。
- `side_to_move`, 自玉駒, 自玉位置, pinned bitboard, occupied bitboard を `legal_moves()` のフィルタ前にまとめて取得。
- `is_legal_with_context()` を追加。
- `swap_remove` ループ、生成順、候補集合は維持。

## 挙動確認

### tests

```text
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

結果: pass

### qsearch candidate order

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/qsearch_candidate_probe \
  --input taya36.sfen \
  --all-plies \
  --ordered \
  --max-positions 5000 \
  --show 3
```

```text
positions: 5000
mismatches: 0
ordered: true
all plies: true
reference candidates: 9089
current candidates: 9089
```

```text
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

### search_profile counters

実験ブランチ:

```text
samples: 72
total nodes: 19067196
quiescence nodes: 17270535
quiescence moves considered: 8841983
quiescence moves generated: 138833345
quiescence moves discarded: 129991362
quiescence moves searched: 3264998
quiescence see skips: 3096541
quiescence terminal mates: 424
check evasion extensions: 26959
aspiration fail lows: 0
aspiration fail highs: 0
aspiration researches: 0
elapsed ms: 68466.10
nodes/sec: 278491.03
```

探索カウンタは master と完全一致した。

## 速度比較

同一PC・同一コマンドで master を再測定した。

### legal_moves only

実験:

```text
elapsed ms: 1212.29
positions/sec: 296958.96
```

master:

```text
elapsed ms: 1201.79
positions/sec: 299553.00
```

### legal_moves + do/undo

実験:

```text
elapsed ms: 4319.25
positions/sec: 83347.77
```

master:

```text
elapsed ms: 4425.16
positions/sec: 81352.96
```

### quiescence movegen

実験:

```text
elapsed ms: 1477.52
positions/sec: 243651.59
```

master:

```text
elapsed ms: 1453.59
positions/sec: 247661.87
```

### search_profile

実験:

```text
elapsed ms: 68466.10
nodes/sec: 278491.03
```

master:

```text
elapsed ms: 62310.41
nodes/sec: 306003.39
```

## 判断

挙動は完全一致したが、速度改善が確認できなかった。

do/undo込みでは少し良い値が出たものの、legal_moves単体、qsearch候補生成、探索全体では master の方が速かった。採用基準の `search_profile` nodes/sec +1.5%以上にも届かない。

このため、`legal_moves()` 合法性フィルタのコンテキスト化は採用しない。

次は GPT-5.5 xhigh の判断どおり、探索ヒューリスティック追加を一旦止め、guard強めの value regression v2 へ進む。
