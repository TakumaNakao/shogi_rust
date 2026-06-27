# qsearch 1候補 fast path 棄却

- 作成日時: 2026-06-27 17:40:42 UTC
- ブランチ: `perf/qsearch-single-candidate-fastpath`
- 結論: 採用しない。探索カウンタは完全一致したが、速度改善が安定しなかった。

## 目的

`quiescence_search` で qsearch 候補が1手だけの場合に、候補手の採点用 `Vec<(Move, i32)>` 作成と `sort_unstable_by_key` を省略する。

狙いは探索木を変えず、qsearch の小さな固定費だけを削ることだった。

## 変更内容

一時実験では、`src/ai.rs` の `legal_quiescence_moves_with_generated_count()` 後に以下の fast path を追加した。

- 候補が0手なら既存通り即return。
- 候補が1手なら `score_move_without_counter()` とsortを省き、その1手だけ既存と同じ SEE / do-undo / 千日手 / 再帰処理に通す。
- 候補が2手以上なら既存経路をそのまま使う。

## 検証

実験は `GPT-5.3-codex-spark` サブエージェントに委任した。

比較対象:

- base: `origin/training/strong-weight-learning-infra` (`11cf459`)
- new: base + qsearch 1候補 fast path

固定深さ profile で以下のカウンタがすべて一致した。

- `total nodes`
- `quiescence nodes`
- `quiescence moves considered/generated/discarded/searched`
- `quiescence see skips`
- `quiescence terminal mates`
- `check evasion extensions`
- `aspiration fail lows/highs/researches`

### 初回 profile

`taya36.sfen`, `samples=72`, `depth=5`, `seed=9501`, 各3回:

```text
elapsed ms median:
  new:  44261.59
  base: 44825.45
nodes/sec median:
  new:  430784.27
  base: 425365.42
nodes/sec improvement: +1.27%
```

`converted_records2016_10818.sfen`, `samples=32`, `depth=5`, `seed=9501`, 各1回:

```text
elapsed ms:
  new:  15694.37
  base: 15811.29
nodes/sec:
  new:  423303.68
  base: 420173.54
nodes/sec improvement: +0.74%
```

### 追加 profile

`taya36.sfen`, `samples=144`, `depth=5`, `seed=9501`, 各2回:

```text
elapsed ms median:
  new:  105752.13
  base: 100763.59
nodes/sec median:
  new:  403570.60
  base: 422354.12
nodes/sec improvement: -4.45%
```

`converted_records2016_10818.sfen`, `samples=64`, `depth=5`, `seed=9501`, 各2回:

```text
elapsed ms median:
  new:  24887.39
  base: 25125.46
nodes/sec median:
  new:  430861.21
  base: 426790.03
nodes/sec improvement: +0.95%
```

## 判断

探索カウンタ完全一致なので挙動不変の実装にはできていた。ただし速度改善が `taya36` の長め計測で逆方向に出た。

この程度の分岐追加は、局面集合やCPUノイズによって簡単に埋もれる。採用目安だった `nodes/sec +1.5%以上` も安定して満たしていない。

したがって、コード変更は戻し、結果のみ残す。

## 次の方針

- qsearch の固定費削減は、1候補fast pathのような微小分岐では不足。
- qsearch直接生成は過去に順序差で悪化しているため、再試行するなら「候補順序設計込みの探索変更」として扱う。
- 次の小改善候補は、GPT-5.5 xhigh が提案した `TT best move delayed scoring` を検討する。ただし TT 周辺は過去に失敗が多いため、固定深さカウンタ完全一致を最初のゲートにする。
