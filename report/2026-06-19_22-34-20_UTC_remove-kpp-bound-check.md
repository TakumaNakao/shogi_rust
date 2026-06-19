# KPP評価内の冗長境界チェック削除

- 作成日時: 2026-06-19 22:34:20 UTC
- ブランチ: `perf/remove-kpp-bound-check`
- 結論: 採用。探索ノード数は一致し、eval/search profileが改善した。

## 変更内容

`SparseModel::predict_from_position` のKPP pair index加算で、以下の境界チェックを削除した。

```rust
if final_index < MAX_FEATURES {
    prediction += self.w[final_index];
}
```

`final_index` は、正規化済み玉位置 `0..81` と、生成済みpiece idの組から作られるため、通常の合法局面では常に `MAX_FEATURES` 未満になる。

変更後:

```rust
prediction += self.w[final_index];
```

## profile

変更前 eval profile:

```text
evals: 819200
score sum: 754071.6
elapsed ms: 3273.68
evals/sec: 250238.49
```

変更後 eval profile:

```text
evals: 819200
score sum: 754071.6
elapsed ms: 2894.85
evals/sec: 282985.41
```

評価値合計は一致し、eval profileは約11.6%改善した。

変更前 search profile:

```text
samples: 72
total nodes: 19067196
quiescence nodes: 17270535
elapsed ms: 64146.45
nodes/sec: 297244.76
```

変更後 search profile:

```text
samples: 72
total nodes: 19067196
quiescence nodes: 17270535
elapsed ms: 62230.49
nodes/sec: 306396.37
```

探索ノード数は一致し、search profileは約3.0%改善した。

## 対局副作用確認

比較対象:

- new: 境界チェック削除版
- baseline: master
- weights: 両方 `policy_weights_v2.1.0.binary`
- positions: `taya36.sfen`
- depth 5 / time-limit-ms 100 / max-plies 200

20局:

```text
new wins: 9
baseline wins: 11
draws: 0
new total score rate: 45.00%
```

40局:

```text
new wins: 21
baseline wins: 19
draws: 0
new total score rate: 52.50%
```

20局はやや悪く出たが、40局ではほぼ互角だった。探索ノード数が一致している挙動不変の高速化であり、profile改善が明確なため採用する。

## 判断

採用する。

この変更単体で大幅な勝率向上は期待しないが、100ms条件では約3%の探索速度改善がそのまま持ち時間内の探索余裕になる。
