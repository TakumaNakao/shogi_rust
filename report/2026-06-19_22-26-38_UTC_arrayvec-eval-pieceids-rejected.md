# 評価関数piece id ArrayVec化の棄却

- 作成日時: 2026-06-19 22:26:38 UTC
- ブランチ: `perf/arrayvec-eval-pieceids`
- 結論: 採用しない。評価値は一致したが、eval/search profileが悪化した。

## 目的

`SparseModel::predict_from_position` は探索中に高頻度で呼ばれるため、piece id一時配列の `Vec::with_capacity(40)` を小型 `ArrayVec<usize, 40>` に置き換え、ヒープ確保を減らすことを狙った。

あわせて `extract_kpp_features_and_material` のpiece id一時配列も同じ形へ変更した。

## 検証

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
elapsed ms: 3333.54
evals/sec: 245744.45
```

score sumは一致し、評価挙動は同じだったが、速度は約1.8%悪化した。

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
elapsed ms: 69298.71
nodes/sec: 275145.05
```

ノード数は一致したが、探索時間は大きく悪化した。

## 判断

採用しない。

推定理由:

- `Vec::with_capacity(40)` のコストは想定より支配的ではない。
- `ArrayVec` 化により、関数フレームやコピー/初期化コストが増えた可能性がある。
- qsearch比率が高いため、評価関数内のわずかな悪化が全体に強く効いた。

## 次の方針

`ArrayVec` への単純置換は、探索手順ソート・評価関数の両方で悪化した。今後の高速化は、コンテナ置換ではなく、処理量そのものを減らすか、計算済み情報を局面側から安全に再利用する方向を検討する。
