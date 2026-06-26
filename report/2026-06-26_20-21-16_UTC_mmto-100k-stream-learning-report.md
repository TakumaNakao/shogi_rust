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

## Addendum: Listwise Leaf Experiments

After this report was first written, `mmto_tree_train` was extended with:

- `--loss-mode listwise-leaf`
- `--teacher-temperature-cp`
- `--model-temperature-cp`
- `--listwise-feature-source teacher-leaf|student-leaf|move`
- `--extra-valid-best-weight`

The feature-source comparison was important:

- `teacher-leaf` reduced listwise loss but worsened static validation regret.
- `move` improved some static metrics, but failed light rerank due p90/p95 regressions.
- `student-leaf` was the first clearly promising variant.

The best small result was:

`data/mmto/runs/mmto_listwise_smoke9k_studentleaf_lr001_20260626_202526`

- 9K smoke, `student-leaf`, `learning-rate=0.01`
- static valid improved: `selected_regret=147.28 -> 146.48`, `p90=114.48 -> 109.59`, `p95=157.20 -> 153.67`
- light rerank depth3/teacher4 400 positions passed:
  - baseline: `mean=515.99`, `p90=43.75`, `p95=71.92`, `bad50=0.0900`, `match=39.25%`
  - candidate: `mean=515.66`, `p90=42.98`, `p95=70.56`, `bad50=0.0850`, `match=40.25%`
- 20-game weight-only benchmark: new 12, baseline 8

This was not adopted because 20 games are not enough and the candidate weight was a small experimental artifact. The large weight files were deleted after logging the result.

Scaling the same idea to the full 100K dump did not work:

- `lr=0.01`, `l2=0.0001`: validation improved, but the max weight shrank from about `1.76` to about `1.00`, and rerank failed.
- `lr=0.002`, `l2=0`, `max-weight-delta=0.02`: validation improved strongly, score gate passed, but rerank failed on `p95`, `bad100`, and match rate.
- `lr=0.001`, `l2=0`, `max-weight-delta=0.005`: score gate passed, but rerank still failed on mean, p90, p95, bad50, and match rate.

The rerank failures show a consistent pattern: broad static improvements can introduce a small number of severe root move flips. Static tree validation does not reliably catch these failures.

Hard positions were extracted from failed rerank JSON reports:

`data/mmto/hard_valid/listwise_100k_studentleaf_rerank_hard_20260626.jsonl`

This produced 73 matched tree records. However, using it as `--extra-valid hard=... --extra-valid-best-weight 2.0` did not catch the rerank problem, because the static extra-valid metrics still improved while searched root moves could degrade.

Updated conclusion:

- `student-leaf listwise` is the most promising learning direction so far.
- Pure static listwise is insufficient for full 100K training.
- The next algorithmic improvement must explicitly handle current selected mistakes or rerank hard negatives, not just improve fixed static validation loss.

## Addendum: Current-Selected Hard Negative

`mmto_tree_train` was extended again with:

- `--listwise-hard-negative-weight`
- `--listwise-hard-negative-min-regret-cp`

This adds a small pairwise penalty to `listwise-leaf` when the current model's root static selection is at least the configured regret threshold worse than the teacher-best candidate. The hard-negative pair uses the same feature source as `--listwise-feature-source`.

Key results:

### 9K, weight 0.05, min regret 50

Run:

`data/mmto/runs/mmto_listwise_hardneg9k_studentleaf_w005_20260626_205457`

- static valid improved: `selected_regret=147.28 -> 145.45`, `bad50=0.2533 -> 0.2433`
- score gate passed: `p95_abs_delta_cp=4.00`, `max=6.87`
- light rerank depth3/teacher4 400 positions passed:
  - baseline: `mean=515.99`, `p90=43.75`, `p95=71.92`, `bad50=0.0900`, `match=39.25%`
  - candidate: `mean=515.32`, `p90=42.98`, `p95=70.47`, `bad50=0.0850`, `match=40.25%`
- 20-game benchmark: 10-10

This is safe-looking but not stronger than the earlier 9K student-leaf-only 12-8 smoke.

### 100K, weight 0.05, min regret 50

Run:

`data/mmto/runs/mmto_listwise_hardneg100k_studentleaf_w005_lr0002_delta002_20260626_205816`

- score gate passed: `p95_abs_delta_cp=3.47`, `max=5.57`
- light rerank improved mean, p90, bad50, and match, but failed on bad100:
  - baseline: `mean=12.76`, `p90=41.27`, `p95=67.99`, `bad50=0.0875`, `bad100=0.0150`, `match=45.25%`
  - candidate: `mean=12.48`, `p90=38.22`, `p95=67.99`, `bad50=0.0800`, `bad100=0.0200`, `match=46.25%`
- 20-game benchmark: 6-14

The 100K hard-negative candidate was rejected and its large weight files were deleted.

### Interpretation

The hard-negative term helps the static and light-rerank average case, but full 100K fixed-dump learning still creates severe tail mistakes that show up in games. Continuing to tune only `learning-rate`, `max-weight-delta`, and hard-negative weight is unlikely to solve the core issue.

Next priority should be a DAgger-style re-dump:

1. keep a failed candidate long enough to extract `hard_positions.sfen`;
2. rerun `mmto_tree_dump` with that candidate as the student and v2.1.0 as teacher;
3. ensure the actual student-selected bad move is force-included in the candidate set;
4. train on a mixture of normal Wdoor positions and these current-student mistakes.

## Addendum: Low-Memory DAgger Loop Trial

After the force-include change in `mmto_tree_dump`, a low-memory DAgger loop was tested.
The goal was not to adopt a short-run weight, but to check whether current-student hard
positions can be mined and mixed without crashing the 16 GB machine.

### Seed student-leaf candidate

Run:

`data/mmto/runs/mmto_dagger_seed_studentleaf_20260626_212044`

This regenerated the previously promising 9K `student-leaf` listwise candidate:

- static valid improved: `selected_regret=147.28 -> 146.45`, `p95=157.20 -> 153.67`
- holdout rerank depth3/teacher4, 1200 positions failed slightly:
  - baseline: `mean=263.29`, `p90=43.65`, `p95=67.99`, `bad50=0.0858`, `match=38.92%`
  - candidate: `mean=263.44`, `p90=43.74`, `p95=67.99`, `bad50=0.0875`, `match=38.58%`

The weight was rejected as an adoption candidate, but its rerank output produced 138
unique hard positions. Re-dumping those positions with the candidate as student produced:

- `data/mmto/runs/mmto_dagger_hard149_20260626_212525`
- `hard_train`: 110 records
- `hard_valid`: 28 records

Peak RSS for the dump was about 1.7 GB with `jobs=2` and `position-chunk-size=32`.

### 9K normal plus hard mix

Run:

`data/mmto/runs/mmto_dagger_mix9k_hard110x5_l2zero_20260626_213122`

The training set used 9K normal records plus the 110 hard records repeated 5 times.
Settings used `student-leaf`, `listwise-hard-negative-weight=0.02`, `min-regret=100`,
`l2=0`, and `max-weight-delta=0.01`.

Result:

- normal valid improved slightly: `selected_regret=147.28 -> 146.71`, `p95=157.20 -> 154.16`
- hard valid worsened: `selected_regret=72.03 -> 73.42`
- `best_epoch=0`

Rejected. Large candidate weights were deleted.

### 100K stream plus first hard set

Run:

`data/mmto/runs/mmto_stream100k_hard110x10_safe_20260626_213340`

The training file used the existing 100K stream dump plus the first hard set repeated 10
times. Training used `--stream-train`, `learning-rate=0.0008`, `max-weight-delta=0.005`,
and `listwise-hard-negative-weight=0.01`.

Static validation:

- valid `selected_regret=109.46 -> 109.01`
- valid `p95=143.37 -> 143.05`
- valid `bad50=0.2263 -> 0.2253`
- valid `bad100=0.1072 -> 0.1058`

Offline gates:

- score gate passed: `mean_abs_delta=0.37cp`, `p95=1.61cp`, `max=2.49cp`
- rerank depth3/teacher4, 1200 positions passed:
  - baseline: `mean=96.77`, `p90=43.82`, `p95=71.35`, `bad50=0.0925`, `bad100=0.0158`, `match=40.25%`
  - candidate: `mean=96.53`, `p90=43.69`, `p95=71.05`, `bad50=0.0883`, `bad100=0.0158`, `match=40.25%`

20-game weight-only benchmark:

- new: 10
- baseline: 10
- draws: 0
- paired starts: new sweeps 3, baseline sweeps 3, splits 4

Conclusion:

The candidate was safe enough to pass rerank, but not stronger in games. It was rejected
and the large weight file was deleted after using it to mine the next hard set.

### Second hard set and mixed retry

The accepted-for-mining candidate above produced 115 unique hard positions from rerank.
Re-dumping with it as student produced:

- `data/mmto/runs/mmto_dagger_hard116_r2_20260626_214339`
- `hard_train`: 92 records
- `hard_valid`: 23 records

A second 100K stream mix used both hard sets repeated 10 times:

`data/mmto/runs/mmto_stream100k_hard_r1r2_safe_20260626_214759`

Result:

- valid `selected_regret=109.46 -> 109.17`
- valid `bad50=0.2263 -> 0.2257`
- valid `p95=143.37 -> 143.47` worsened
- combined hard valid worsened: `selected_regret=86.09 -> 89.67`
- `best_epoch=0`

Rejected. Large candidate weights and large temporary mixed train files were deleted.

### Updated conclusion

The low-memory DAgger mechanics now work:

- hard positions can be mined from rerank JSON,
- `mmto_tree_dump` can re-dump current-student mistakes with bounded memory,
- 100K stream training can include small hard sets without OOM,
- strict score/rerank/game gates prevent bad weights from being adopted.

However, the current objective still does not produce a clearly stronger weight. The best
DAgger candidate improved static validation and rerank but only scored 10-10 in a 20-game
weight-only benchmark. The next learning change should focus on the objective, not another
simple hard-set repetition sweep. In particular, hard cases should probably be used as a
separate constrained penalty or replay buffer with per-position caps, rather than repeated
inside the same listwise stream where they can worsen hard-valid tails.

## Addendum: Separate Hard Replay Trial

`mmto_tree_train` was extended with a low-memory replay mechanism:

- `--replay-train PATH` can be specified multiple times.
- `--replay-weight` scales the learning rate for replay updates.
- `--replay-max-samples` caps the total replay samples per epoch.
- replay records are streamed from their own files after the normal train pass.

This avoids building another 900 MB mixed train file just to repeat a small hard set.

### Replay experiment

Run:

`data/mmto/runs/mmto_replay100k_hard_r1r2_20260626_215625`

Training used the existing 100K stream dump as normal training and the two hard DAgger
sets as replay:

- replay records: 202
- `replay-weight=0.05`
- `learning-rate=0.0008`
- `max-weight-delta=0.005`
- `listwise-hard-negative-weight=0.01`

Static validation:

- valid `selected_regret=109.46 -> 109.02`
- valid `p95=143.37 -> 143.05`
- valid `bad50=0.2263 -> 0.2255`
- valid `bad100=0.1072 -> 0.1060`
- hard valid worsened: `selected_regret=86.09 -> 90.34`

Offline gates:

- score gate passed: `mean_abs_delta=0.38cp`, `p95=1.64cp`, `max=2.53cp`
- rerank depth3/teacher4, 1200 positions passed:
  - baseline: `mean=13.85`, `p90=43.78`, `p95=71.61`, `bad50=0.0925`, `bad100=0.0250`, `match=39.67%`
  - candidate: `mean=13.83`, `p90=43.62`, `p95=71.61`, `bad50=0.0900`, `bad100=0.0250`, `match=39.67%`

Weight-only game tests:

- seed 9801 / 20 games: 11-8-1, total score 57.50%
- seed 9901 / 20 games: 11-7-2, total score 60.00%
- seed 10001 / 60 games: 23-36-1, total score 39.17%
- combined 100 games: 45-51-4, total score 47.00%

Conclusion:

The replay mechanism is useful infrastructure, but this replay objective did not produce a
stronger weight. The candidate was rejected and its large weight files were deleted. The
positive first 40 games were noise; the 60-game follow-up exposed the regression. Future
learning work should not adopt weights until at least 100 games or multiple seeds confirm
the result.
