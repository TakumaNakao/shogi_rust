# Tail metric checkpoint selection probes

- Date: 2026-06-28 13:48:52 UTC
- Branch: `training/strong-weight-learning-infra`
- Purpose: check whether the incumbent/tail objective contains any checkpoint that improves tail metrics, even if valid loss improves.

## Runs

Both runs used the same source dump:

- `data/mmto/runs/phase_gate_listwise_ce_mid_20260628_132336`
- train `1992`, valid `222`
- same objective as `incumbent_tail_probe_20260628_134241`

### `BEST_METRIC=bad50-regret`

- Run dir: `data/mmto/runs/incumbent_tail_bad50_metric_20260628_134638`
- `best_epoch=0`
- `best_value=0.427928`
- valid bad50: unchanged at `0.4279` for epochs 0-5
- score/rerank gates: not run because baseline remained best
- generated `.binary`: none remaining

### `BEST_METRIC=p90-regret`

- Run dir: `data/mmto/runs/incumbent_tail_p90_metric_20260628_134659`
- `best_epoch=0`
- `best_value=250.106567`
- valid p90: unchanged at `250.11` for epochs 0-5
- score/rerank gates: not run because baseline remained best
- generated `.binary`: none remaining

## Conclusion

The current incumbent/tail objective does not produce a tail-improving checkpoint on this validation split.

This is stronger evidence than the previous valid-loss failure:

- valid loss can improve,
- match rate can improve slightly,
- but bad50 and p90 do not improve,
- and rerank gate gets worse when valid-loss-selected checkpoints are evaluated.

Do not scale this objective. The next step should change checkpoint selection/gating structure or training data, not training duration.

## Next Step

Use rerank failures as hard validation/replay data for checkpoint selection before using them as ordinary loss. In particular:

- retain normal validation,
- add hard rerank feedback validation,
- select checkpoints only when hard feedback violation does not worsen,
- keep phase-aware rerank gate as the final offline gate.

This should be tested on the same mid-size dump before any longer run.
