# KPP重み参照get_unchecked化の棄却

- 作成日時: 2026-06-19 22:37:45 UTC
- ブランチ: `perf/unchecked-kpp-weight-access`
- 結論: 採用しない。評価値と探索ノード数は一致したが、profileが大きく悪化した。

## 目的

`SparseModel::predict_from_position` のKPP重み参照で、Rustのインデックス境界チェックを避けるため、以下を試した。

```rust
debug_assert!(final_index < self.w.len());
prediction += unsafe { *self.w.get_unchecked(final_index) };
```

前回採用した `final_index < MAX_FEATURES` 分岐削除に続く、さらなるホットパス高速化を狙った。

## 検証

変更前、採用済みmaster:

```text
eval profile elapsed ms: 2894.85
search profile elapsed ms: 62230.49
search total nodes: 19067196
```

変更後:

```text
evals: 819200
score sum: 754071.6
elapsed ms: 6743.28
evals/sec: 121483.90
```

```text
samples: 72
total nodes: 19067196
quiescence nodes: 17270535
elapsed ms: 66951.49
nodes/sec: 284791.22
```

評価値合計と探索ノード数は一致したが、eval profileは大幅悪化、search profileも悪化した。

## 判断

採用しない。

推定理由:

- 通常の `self.w[final_index]` は、直前の分岐削除後にコンパイラが十分最適化できている可能性が高い。
- `get_unchecked` と `debug_assert` の形が、逆にベクトル化・境界チェック除去・インライン最適化を妨げた可能性がある。
- unsafe化は保守リスクも増えるため、明確なprofile改善がない限り採用しない。

## 次の方針

KPP重み参照のunsafe化は再試行しない。評価関数高速化は、安全な算術整理や局面走査回数削減を優先する。
