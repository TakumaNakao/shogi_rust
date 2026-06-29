# Protected Feedback Phase-Fix Result

- Date: 2026-06-28 UTC
- Branch: `training/strong-weight-learning-infra`
- Baseline weights: `policy_weights_v2.1.0.binary`

## Summary

Strong-teacher filtered feedback was re-tested with a normal-position protection set. A bug in protection sampling was fixed first: the scripts looked for `endgame`, while the dump schema uses `late`. After the fix, the protection set became phase-balanced:

```text
opening: 400
middle:  400
late:    400
```

This produced a strong 20-game result, but the signal did not survive the 40-game gate. No weight was adopted.

## Code Changes

- Added `mmto_tree_train --loss-mode aux-only`.
  - This allows feedback to be the primary signal while normal positions contribute only auxiliary constraints such as incumbent protection.
- Added `tools/run_protected_feedback_from_existing.sh`.
  - Reuses existing strong-teacher dumps and feedback JSONs without repeating expensive depth-4 dump generation.
- Fixed protected sampling phase names.
  - `late` is now included in the phase-balanced protection set.

## Main Experiment

Run:

```text
data/mmto/runs/protected_feedback_phasefix_fb050_i05_20260628_231117
```

Settings:

```text
FEEDBACK_WEIGHT=0.50
INCUMBENT_PROTECTION_WEIGHT=0.05
LOSS_MODE=aux-only
PROTECTION_LINES=1200
```

Offline:

```text
best_epoch: 12
feedback loss:      98.585358 -> 98.583595
feedback violation: 0.508850  -> 0.486726
score gate: PASS
rerank gate: PASS
```

Rerank:

```text
baseline:  mean=270.11 p90=113.26 p95=145.35 match=33.72%
candidate: mean=268.42 p90=102.74 p95=142.94 match=34.87%

baseline bad50/bad100/bad200:  0.3095 / 0.1178 / 0.0208
candidate bad50/bad100/bad200: 0.2933 / 0.1085 / 0.0208
```

20-game smoke:

```text
new wins: 13
baseline wins: 6
draws: 1
new total score rate: 67.50%

end reasons:
  Resign: 19
  RepetitionDraw: 1

paired starts:
  new sweeps: 4
  baseline sweeps: 0
  splits: 5
  draw/mixed: 1
```

40-game gate:

```text
new wins: 19
baseline wins: 18
draws: 3
new total score rate: 51.25%
new decisive win rate: 51.35%

end reasons:
  Resign: 37
  RepetitionDraw: 3

paired starts:
  new sweeps: 2
  baseline sweeps: 2
  splits: 13
  draw/mixed: 3
```

The 40-game result is close to neutral and below the threshold for promotion to 100 games. The candidate weight was deleted.

## Secondary Experiment

Run:

```text
data/mmto/runs/protected_feedback_phasefix_fb050_i10_20260628_232044
```

This used `INCUMBENT_PROTECTION_WEIGHT=0.10`. It passed score/rerank gates but was not fully benched after the first candidate failed the 40-game gate. Its unvalidated binary was deleted.

## Interpretation

The corrected phase-balanced protection set improved the 20-game smoke result, which suggests the feedback signal is not useless. However, the 40-game result regressed to neutral. This is consistent with previous MMTO-lite attempts: offline feedback/rerank metrics improve, but the improvement does not reliably transfer to actual self-play.

The likely issue is not simply insufficient training time. Longer training on the same local objective would probably continue improving feedback loss while risking more distribution drift. The limiting factor is the objective and validation distribution:

- The hard-node feedback set is small and localized.
- Protection data still lacks `in_check` and low-legal-move positions.
- Validation is focused on a strong-teacher hard set, not enough on actual bench failure modes.
- 20-game smoke results remain noisy.

## Next Steps

1. Build a broader protection/guard set.
   - Include late positions, in-check positions, low-legal-move positions, and bench failure positions.
   - Current dumps contain no `in_check` or low-legal protection samples, so a new source or dump mode is needed.
2. Mine the 40-game failure records.
   - Convert repeated candidate mistakes into guard data first, not training data.
3. Only then retry protected feedback.
   - Require offline pass plus at least 40-game non-regression before keeping any binary.
4. Avoid simply extending epochs on the current objective.
   - Current evidence says objective quality, not training duration, is the bottleneck.

## Adoption

No model weight was adopted.
