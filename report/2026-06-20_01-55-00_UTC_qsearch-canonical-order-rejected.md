# qsearch canonical order rejected

- 作成日時: 2026-06-20 01:55:00 UTC
- 実験ブランチ: `experiment/qsearch-canonical-order`
- 実験コミット: `c904373`
- 判断: 不採用

## 背景

qsearch直接生成高速化案は、候補集合が一致しても旧実装の `swap_remove` 副作用による候補順序まで再現できず、探索木が変わるため高速化単体として棄却した。

GPT-5.5 xhigh の提案に従い、次の段階として qsearch 内の同点順序を明示固定し、生成順依存を切る実験を行った。

## 実装

`src/ai.rs` の qsearch 候補ソートを以下に変更した。

- 既存の `score_move_without_counter()` のスコアを最優先。
- 同点時のみ qsearch用クラスで順序固定。
  - 成り捕獲
  - 捕獲
  - 成り王手
  - 王手
  - その他
  - 駒打ち
- 最後に通常手/駒打ちの数値キーで決定的に並べる。

## 検証

### tests

```text
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

結果: pass

### search_profile

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/search_profile \
  --weights policy_weights_v2.1.0.binary \
  --positions taya36.sfen \
  --samples 72 \
  --depth 5 \
  --seed 9501
```

```text
samples: 72
total nodes: 19066612
quiescence nodes: 17269951
quiescence moves considered: 8841038
quiescence moves generated: 138818579
quiescence moves discarded: 129977541
quiescence moves searched: 3264414
quiescence see skips: 3096543
quiescence terminal mates: 414
check evasion extensions: 26959
aspiration fail lows: 0
aspiration fail highs: 0
aspiration researches: 0
elapsed ms: 78510.30
nodes/sec: 242854.91
```

master基準と比較してノード数はほぼ同等だが、`quiescence terminal mates` などが変化しており、探索順序変更として扱う必要がある。

### same-engine 20局

new: `experiment/qsearch-canonical-order`  
baseline: `master`  
weights: `policy_weights_v2.1.0.binary`  
positions: `taya36.sfen`  
seed: `13001`  
depth: `5`  
time-limit-ms: `100`  
record-dir: `/tmp/shogi_bench_qsearch_canonical_20_seed13001`

```text
new wins: 9
baseline wins: 11
draws: 0
new decisive win rate: 45.00%
new total score rate: 45.00%
decisive win rate 95% CI: 25.82%..65.79%
total score rate 95% CI: 23.20%..66.80%
```

`record_analyze` 要約:

```text
end reasons:
  MaxPliesAdjudication: 1
  Resign: 19
paired starts:
  new sweeps: 0
  baseline sweeps: 1
  splits: 9
  draw/mixed pairs: 0
average final score for new: -230.5
average final score for NewWin: 313.8
average final score for BaselineWin: -675.9
terminal result mismatches: 0
non-terminal score/result sign mismatches: 0
```

## 判断

20局スモークで 9-11 と負け越し、paired starts でも new sweep がなく baseline sweep が1件出た。

このため、qsearch同点順序固定は採用しない。qsearch直接生成を続ける場合も、この同点順序固定を前提にする案はいったん見送る。

次候補は、より局所的な探索改善として「王手になる負SEE手だけ qsearch skip を緩める」実験を行う。
