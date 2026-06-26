# MMTO 100K Stream Learning Experiment Report

- Date: 2026-06-26 20:21:16 UTC
- Branch: `training/strong-weight-learning-infra`
- Baseline weights: `policy_weights_v2.1.0.binary`
- Main data: Wdoor high-rate 100K teacher dump, depth3 teacher / depth2 student / top16

## Summary

The 100K teacher dump can now be trained safely with streaming mode. Memory stayed around 4.6-4.8 GB RSS per trainer, so the previous OOM/crash problem is fixed for this scale.

However, the existing fixed-dump pairwise objective still did not produce an adoptable weight. Several variants improved training loss or a light offline/rerank metric, but none survived an actual weight-vs-baseline game check. The best diagnostic candidate passed a 400-position light rerank gate but lost a 20-game engine match 8-12, so it was rejected and the large candidate weight files were deleted.

## Experiments

### 1. 100K stream pairwise baseline

Run directory:

`data/mmto/runs/mmto_stream_100k_e2_lr3e5_20260626_193126`

Result:

- baseline valid: `loss=41.954914`, `selected_regret=109.46`, `p90=101.94`, `p95=143.37`, `bad50=0.2263`
- epoch 2 valid: `loss=41.837070`, `selected_regret=120.09`, `p90=101.90`, `p95=143.37`, `bad50=0.2268`
- `best_epoch=0`

Conclusion:

Loss improved, but the actual selection/regret metrics did not. Rejected.

### 2. Student-top bad50 focused pairwise

Run directory:

`data/mmto/runs/mmto_stream_100k_studenttop_bad50_20260626_193952`

Result:

- baseline valid `bad50=0.2263`
- epoch 3 valid `bad50=0.2267`
- `best_epoch=0`

Conclusion:

Focusing broad student-top bad candidates did not improve the validation target. Rejected.

### 3. Model-top capped selected regret

Run directory:

`data/mmto/runs/mmto_stream_100k_modeltop_capped_20260626_193952`

Result:

- `best_epoch=3`
- score gate passed: `p95_abs_delta_cp=0.07`, `max=0.13`
- light rerank depth3/teacher4, 200 positions:
  - baseline: `mean=12.86`, `bad50=0.0950`, `match=41.50%`
  - candidate: `mean=12.70`, `bad50=0.0900`, `match=41.00%`
  - failed because match rate worsened

Conclusion:

Some regret metrics improved, but root move agreement worsened. Rejected.

### 4. Current-selected hard-negative diagnostic

Run directory:

`data/mmto/runs/mmto_stream_100k_current_selected_diag_20260626_201000`

Result:

- pair count dropped to roughly 2K, isolating the current selected bad move instead of many unselected bad candidates
- `best_epoch=3`
- score gate passed: `p95_abs_delta_cp=0.08`, `max=0.14`
- light rerank depth3/teacher4, 400 positions:
  - baseline: `mean=12.76`, `p90=41.27`, `p95=67.99`, `bad50=0.0875`, `match=45.25%`
  - candidate: `mean=12.68`, `p90=38.22`, `p95=67.99`, `bad50=0.0850`, `match=45.25%`
  - rerank gate passed
- 20-game engine match against baseline weights:
  - new: 8
  - baseline: 12
  - draws: 0
  - score rate: 40.00%
  - paired starts: new sweeps 1, baseline sweeps 3, splits 6

Conclusion:

This supports the hypothesis that broad top-N bad candidate updates are harmful, but the current pairwise objective is still not strong enough for adoption. Rejected.

## Diagnosis

The main failure is not simply insufficient training time. The existing objective optimizes fixed-dump leaf pairwise loss, but the adoption gate evaluates root move choice after search. This mismatch explains why loss can improve while rerank or game performance does not.

The current-selected diagnostic is informative: narrowing updates to the currently selected mistake improved light rerank without hurting match rate, but the improvement was too small and did not survive game testing. This suggests the next step should be an objective change rather than more learning-rate/topK tuning.

## Next Plan

1. Implement `listwise-leaf` training for `mmto_tree_train`.
   - Teacher target: softmax over candidate `teacher_score`.
   - Student target: softmax over candidate model scores.
   - Loss: candidate-list cross entropy/KL.
   - Keep streaming train support.

2. After listwise works, run a small 9K smoke test and then a 100K stream test.

3. If listwise produces a candidate that passes light rerank, run a 20-game weight benchmark before any larger benchmark.

4. If listwise still fails, move to a DAgger-style loop:
   - train a tiny candidate,
   - re-dump current-student mistakes,
   - train on those mistakes rather than only baseline-student mistakes.

## Disk Cleanup

Rejected large candidate files were removed:

- `best.raw.binary`
- `candidate.raw.binary`

Run metadata, logs, and small JSON outputs were kept under `data/mmto/runs/` for analysis.
