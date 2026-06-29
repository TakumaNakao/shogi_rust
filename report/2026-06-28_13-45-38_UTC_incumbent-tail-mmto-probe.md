# Incumbent/tail MMTO objective probe

- Date: 2026-06-28 13:45:38 UTC
- Branch: `training/strong-weight-learning-infra`
- Purpose: test whether incumbent protection and tail-regret penalty can stop the rerank regressions seen in phase-gated MMTO probes.

## Change Tested

`mmto_tree_train` now supports two optional auxiliary losses:

- `--incumbent-protection-weight`
  - Keeps the current baseline-selected move above the model top move when the baseline move is not much worse by teacher score.
- `--tail-regret-penalty-weight`
  - Penalizes the current model top move when its teacher regret exceeds a threshold, with a dynamic weight based on regret excess.

The run used the same mid-size phase-targeted dump as the previous failing CE probe, so the result isolates the objective change.

## Run

- Run dir: `data/mmto/runs/incumbent_tail_probe_20260628_134241`
- Source dump: `data/mmto/runs/phase_gate_listwise_ce_mid_20260628_132336`
- Data: `train=1992`, `valid=222`
- Main settings:
  - `LOSS_MODE=listwise-leaf`
  - `TEACHER_TOP_CE_WEIGHT=0.05`
  - `LISTWISE_HARD_NEGATIVE_WEIGHT=0.02`
  - `GAME_TEACHER_MARGIN_WEIGHT=0.02`
  - `CURRENT_TOP_MARGIN_WEIGHT=0.005`
  - `INCUMBENT_PROTECTION_WEIGHT=0.08`
  - `TAIL_REGRET_PENALTY_WEIGHT=0.04`

## Result

The run is not scalable.

Training selected `best_epoch=5` by valid loss:

- baseline valid loss: `16.829840`
- best valid loss: `16.603731`
- score gate: PASS
- rerank gate: FAIL

Rerank overall:

- mean: `21.3452 -> 22.2326`
- p90: `73.4937 -> 80.0572`
- p95: `116.2881 -> 116.2881`
- match: `40.54% -> 41.44%`
- bad50: `13.51% -> 14.41%`
- bad100: `7.66% -> 8.11%`

Phase regressions:

- opening mean: `17.2666 -> 17.5505`
- middle mean: `20.8539 -> 21.5529`
- middle bad50: `13.19% -> 14.29%`
- late mean: `33.3501 -> 36.3064`
- late bad50: `27.78% -> 30.56%`

Worst deltas still included large new root-choice failures:

- `delta=106.43`, baseline regret `0.00`, candidate regret `106.43`
- `delta=63.66`, baseline regret `0.00`, candidate regret `63.66`

Generated `.binary` files were deleted after gate failure.

## Interpretation

The added incumbent/tail terms improved the training objective and slightly improved match rate, but still worsened bad50 and mean regret. This confirms that match rate and offline loss are not sufficient adoption metrics.

The current objective still does not directly optimize the same decision surface as rerank gate. Longer training with these weights is not justified.

## Next Direction

Do not scale this objective. The next attempt should either:

1. use a best metric tied to `bad50-regret` or `p90-regret` instead of `valid-loss`, or
2. add a hard validation/replay set from rerank failures and select checkpoints by non-worsening tail metrics, or
3. reduce mate/outlier contamination in training samples before further objective work.

The immediate low-cost check is to rerun the same dump with the new objective but `BEST_METRIC=bad50-regret` or `p90-regret`; if `best_epoch=0`, that confirms there is no useful tail-improving gradient in this objective.
