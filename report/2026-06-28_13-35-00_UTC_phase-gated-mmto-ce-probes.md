# Phase-gated MMTO CE probe results

- Date: 2026-06-28 13:35:00 UTC
- Branch: `training/strong-weight-learning-infra`
- Purpose: check whether the current root/top-k MMTO objective should be scaled to a long run.

## Result

Do not run the current objective for 24h.

The phase-aware gate shows that both stronger pairwise training and listwise + teacher-top CE can lower offline valid loss while making rerank behavior worse. This is the failure mode we wanted the gate to catch: the model is optimizing a local training signal that does not transfer to actual root search selection.

## Runs

### Pairwise stronger probe

- Run dir: `data/mmto/runs/phase_gate_pairwise_stronger_20260628_131800`
- Source dump: `data/mmto/runs/phase_target_pairwise_probe_20260628_130108`
- Best metric: `valid-loss`
- `best_epoch=5`
- Valid loss: `37.919697 -> 37.155918`
- Score gate: PASS
- Rerank gate: FAIL

Rerank overall:

- mean: `1367.434 -> 1368.2021`
- p90: `58.2438 -> 58.5116`
- p95: `78.2442 -> 86.6213`
- match: `41.89% -> 41.89%`
- bad50: `13.51% -> 14.86%`
- bad100: `5.41% -> 5.41%`

Phase regression:

- opening: mean slightly worse, p90/p95/bad50 unchanged
- middle: mean, p90, p95, and bad50 worse
- late: effectively unchanged

Generated `.binary` files were deleted after gate failure.

### Listwise + teacher-top CE, small dump retry

- Run dir: `data/mmto/runs/phase_gate_listwise_ce_retry_20260628_132023`
- Source dump: `data/mmto/runs/phase_target_listwise_probe_20260628_130048`
- `best_epoch=5`
- Valid loss: `6.306502 -> 6.306201`
- Score gate: PASS
- Rerank gate: PASS

This pass was not enough to justify scaling because the rerank metrics were nearly identical and the sample was small. The only meaningful positive signal was opening mean/p95 improvement without phase regressions.

Generated `.binary` files were deleted after recording the result.

### Listwise + teacher-top CE, mid-size phase-targeted data

- Run dir: `data/mmto/runs/phase_gate_listwise_ce_mid_20260628_132336`
- Dataset targets: `opening=900 middle=900 late=450`
- Balanced records: `opening=874 middle=900 late=450`, total `2224`
- Dump: `train=1992 valid=222 skipped=10`
- `best_epoch=1`
- Valid loss: `6.346478 -> 6.346196`
- Score gate: PASS
- Rerank gate: FAIL

Rerank overall:

- mean: `21.35 -> 22.28`
- p90: `73.49 -> 80.06`
- p95: `116.29 -> 123.68`
- match: `40.54% -> 40.54%`
- bad50: `13.51% -> 14.41%`
- bad100: `7.66% -> 8.11%`

Phase regression:

- opening mean: `17.27 -> 18.84`
- opening p90: `33.49 -> 43.57`
- opening p95: `105.41 -> 107.25`
- opening bad50: `8.42% -> 9.47%`
- middle mean: `20.85 -> 21.50`
- middle bad50: `13.19% -> 14.29%`
- late: unchanged

The worst new regressions were large root-choice failures, for example:

- `delta=158.09`, baseline regret `0.43`, candidate regret `158.52`
- `delta=63.44`, baseline regret `0.00`, candidate regret `63.44`

Generated `.binary` files were deleted after gate failure. `hard_positions.sfen` contains 34 positions for follow-up analysis.

## Interpretation

The current objective is not merely undertrained.

When the learning rate and auxiliary CE/hard-negative weights are strong enough to move the weights, the offline loss can improve, but the rerank gate worsens. This means the objective is still not aligned with the final selection behavior of alpha-beta search.

The next step should be objective/gate improvement, not longer training:

1. Add an explicit baseline-choice preservation or regression penalty for positions where the current baseline already matches the teacher well.
2. Use hard positions as a protected validation/replay gate before mixing them into the loss.
3. Improve candidate generation/teacher depth only after the objective no longer damages phase rerank metrics on mid-size probes.

## Operational note

The phase gate CLI must pass negative defaults with `--option=value` syntax. The scripts were updated after one probe exposed the `unexpected argument '-1'` clap parsing failure.
