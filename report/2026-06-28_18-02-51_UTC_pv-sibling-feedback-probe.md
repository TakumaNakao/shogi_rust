# PV sibling feedback probe

- 作成日時: 2026-06-28 18:02:51 UTC
- ブランチ: `training/strong-weight-learning-infra`
- 実験ディレクトリ: `data/mmto/runs/pv_sibling_feedback_probe_20260628_173100`

## 目的

root-only feedback は line-search の alpha=1 でも rerank mean が微小悪化して失敗した。

次の候補として、`mmto_tree_dump --emit-pv-sibling-nodes` でPV近傍局面を追加し、その中から hard feedback を作ることで、より密度の高い探索整合学習信号になるか確認した。

## dump

入力:

```text
data/mmto/positions/wdoor2023_2026_r4000_p16_120.sfen
```

主な条件:

```text
max_positions: 1000
teacher_depth: 5
student_depth: 4
teacher_score_top: 24
candidate_top: 24
score_all_legal_for_valid: true
emit_pv_sibling_nodes: true
pv_sibling_max_plies: 2
pv_sibling_sample_weight: 0.25
pv_sibling_total_weight_cap: 0.25
exclude_in_check: true
position_chunk_size: 128
jobs: 4
```

結果:

```text
total positions: 1000
train records: 3766
valid records: 435
root records: 932
pv sibling records: 3269
skipped positions: 68
```

`rank_stats`:

```text
samples: 4201
candidates mean: 22.27
selected_regret mean: 39.79
p90: 44.28
p95: 72.87
bad50: 0.0938
bad100: 0.0324
```

## feedback

strict:

```text
train: 409
guard: 136
guard candidate_regret mean: 70.57
```

loose:

```text
train: 242
guard: 81
guard candidate_regret mean: 100.47
```

今回の学習は strict 側を使用した。

## feedback-only 学習

条件:

```text
feedback_weight: 0.5
normal tree train: empty
best_metric: feedback-loss
best_guard_feedback_violation_increase: 0
learning_rate: 0.00005
max_weight_delta: 0.01
anchor_l2: 0.0003
epochs: 10
```

結果:

```text
best_epoch: 10
feedback loss: 101.230614 -> 101.223946
feedback violation: 0.6324 -> 0.5368
```

score gate:

```text
PASS
mean_abs_delta_cp: 0.03
p95_abs_delta_cp: 0.09
max_abs_delta_cp: 0.13
```

rerank gate:

```text
PASS
samples: 435
candidate mean: 239.03238
candidate p90: 29.945755
candidate p95: 43.253628
candidate bad50: 0.04367816
candidate bad100: 0.009195402
candidate match: 42.30%
fail_reasons: []
```

20局ベンチ:

```text
NewWin: 10
BaselineWin: 10
Draw: 0
paired starts:
  new sweeps: 1
  baseline sweeps: 0
  splits: 9
  draw/mixed: 0
```

12-8に届かないため候補重みは削除した。

## 判断

採用はしない。

ただし、root-only feedback と比較すると明確に良い兆候がある。

- held-out feedback violation が大きく改善した。
- score gate を余裕で通過した。
- rerank gate を通過した。
- 20局ベンチは中立だが、baseline sweep が0で、致命的な悪化は見えていない。

次はPV sibling feedbackを小さく捨てるのではなく、スケールアップして評価する価値がある。

## 次の実験

1. `max_positions=3000` または `5000` に増やす。
2. `position_chunk_size=128` のまま、長時間dumpを許容する。
3. feedback_weight は `0.5` を基準にする。
4. strict feedback を優先し、guard violation と rerank gate を必須にする。
5. 20局で12-8以上なら40局、さらに有望なら100局へ進む。

容量対策として、採用候補でない `.binary` は必ず削除する。
