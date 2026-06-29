# Phase-targeted MMTO probe results

- Date: 2026-06-28 13:09:43 UTC
- Branch: `training/strong-weight-learning-infra`
- Goal: decide whether the current MMTO/root-rerank training can be scaled by simply using more data/epochs.

## Summary

Phase-targeted extraction now produces balanced opening/middle/late input, but the current loss settings do not yet create a clearly better weight.

The two primary probes, listwise and pairwise/loss-top, both stopped at `best_epoch=0` when using the stricter default `BEST_METRIC=p95-regret`. This means the training loss can move slightly, but the selected-regret metric used for candidate selection does not improve enough to produce a candidate.

Follow-up probes with `BEST_METRIC=valid-loss` confirmed that candidates can be produced, but the resulting search-choice behavior is unchanged or marginally worse in rerank gate. This is not enough evidence to justify a long unattended run with the same objective.

## Probe runs

### Listwise, p95-regret selection

- Run dir: `data/mmto/runs/phase_target_listwise_probe_20260628_130048`
- Dataset: Wdoor 2023-2026, target records `opening=300 middle=300 late=150`
- Balanced records: `opening=299 middle=300 late=150`, total `749`
- Dump: `train=660 valid=74 skipped=15`
- Result: `best_epoch=0`, no score/rerank gate
- Generated `.binary`: none remaining

Judgement: do not scale this exact condition.

### Pairwise/loss-top, p95-regret selection

- Run dir: `data/mmto/runs/phase_target_pairwise_probe_20260628_130108`
- Dataset: same as listwise
- Dump: `train=660 valid=74 skipped=15`
- Result: `best_epoch=0`, no score/rerank gate
- Generated `.binary`: none remaining

Judgement: do not scale this exact condition.

### Pairwise/loss-top, valid-loss selection

- Run dir: `data/mmto/runs/phase_target_pairwise_validloss_20260628_130720`
- `best_epoch=3`
- Valid loss: `37.919697 -> 37.467899`
- Score gate: passed, `p95_abs_delta_cp=0.150408`
- Rerank gate: failed
- Rerank mean regret: baseline `1367.434`, candidate `1367.4371`
- Rerank p90/p95/match/bad50/bad100: unchanged
- Generated `.binary`: deleted after gate failure

Judgement: valid loss can select a candidate, but it does not improve root choice quality.

### Listwise, valid-loss selection

- Run dir: `data/mmto/runs/phase_target_listwise_validloss_20260628_130736`
- `best_epoch=3`
- Valid loss: `4.492890 -> 4.492877`
- Score gate: passed, `p95_abs_delta_cp=0.033185`
- Rerank gate: passed only because improvement requirements were zero
- Rerank mean/p90/p95/match/bad50/bad100: unchanged from baseline
- Generated `.binary`: deleted manually after confirming it was not useful

Judgement: not a scaling candidate; the improvement is too small and does not affect search choice.

## Interpretation

The current failure mode is not mainly insufficient training time. It is more likely an objective/gate mismatch:

- Offline loss improves, but root-choice metrics do not.
- `p95-regret` is useful as a conservative best metric, but on small validation sets it can be too discrete to create candidates.
- `valid-loss` creates candidates, but those candidates do not improve rerank behavior.
- Bench-failure feedback previously made ordinary validation worse when mixed directly into the loss.

Therefore, simply running this same setup longer is not justified. It may spend time optimizing a signal that does not transfer to actual search decisions.

## Next steps

1. Add phase-aware rerank reporting/gating so opening, middle, and late regressions are visible separately.
2. Keep hard bench-failure positions as replay/gate data first, not as ordinary training loss.
3. Improve the objective so it targets root/top-k search selection more directly, then require small probes to show rerank movement before any 24h run.

## Related change

`dataset_build` phase boundaries were aligned with `mmto_tree_dump` and `mmto_tree_train` diagnostics:

- opening: `ply <= 40`
- middle: `41 <= ply <= 90`
- late: `ply >= 91`

This prevents phase-balanced extraction and phase diagnostics from using different middle/late boundaries.
