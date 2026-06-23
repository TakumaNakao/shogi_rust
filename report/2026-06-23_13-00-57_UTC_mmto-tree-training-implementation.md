# MMTO tree training implementation

- Date: 2026-06-23 13:00:57 UTC
- Branch: `feature/mmto-tree-training`
- Goal: replace plain game-record CE and root-static MMTO-lite with a Bonanza/MMTO-style tree-regeneration training scaffold.

## Summary

Implemented the first end-to-end scaffold for tree-regenerated MMTO-KPP training.

New tools:

- `mmto_tree_dump`
  - Dumps `mmto_tree_v1` JSONL records.
  - Stores root SFEN, teacher/student root scores, teacher/student best moves, and candidate PV leaf SFENs.
  - Candidate set is the union of teacher top moves and student top moves, so student-selected high-regret moves are retained for hard pair generation.
- `mmto_tree_train`
  - Trains KPP weights from PV leaf feature differences.
  - Uses pairwise softplus loss: teacher-preferred leaf should score above student bad leaf by a margin.
  - Supports sparse AdaGrad, material freeze, weight anchor, max weight delta, best checkpoint, valid and extra-valid metrics.
- `mmto_score_gate`
  - Compares baseline/candidate weights on sampled positions.
  - Fails with exit code 2 when score-space delta exceeds configured p95/max limits.

Documentation:

- `docs/mmto_tree_training.md`

## Smoke Validation

Commands were run with tiny depths and 8 positions only to validate wiring, not strength.

Dump:

```text
target/release/mmto_tree_dump
  --student-weights policy_weights_v2.1.0.binary
  --input taya36.sfen
  --teacher-depth 2
  --student-depth 1
  --teacher-score-top 4
  --candidate-top 4
  --max-positions 8
```

Result:

```text
total positions: 8
train records: 6
valid records: 2
skipped positions: 0
```

The candidate union check confirmed that every record contained a `selected_by_student` candidate:

```text
train: 6 records, 6 selected_by_student candidates
valid: 2 records, 2 selected_by_student candidates
```

Train:

```text
baseline train: samples=6 pairs=42 loss=107.215134 selected_regret_mean=15.69
baseline valid: samples=2 pairs=16 loss=99.120804 selected_regret_mean=2.75
epoch 1 train: loss=107.173294 selected_regret=11.22
epoch 1 valid: loss=99.085533 selected_regret=2.72
```

Score gate:

```text
samples=100
mean_abs_delta_cp=0.04
p95=0.06
max=0.07
SCORE GATE PASSED
```

Build/test:

```text
env RUST_FONTCONFIG_DLOPEN=1 cargo fmt --check
env RUST_FONTCONFIG_DLOPEN=1 cargo check --bin mmto_tree_dump --bin mmto_tree_train --bin mmto_score_gate
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

All passed.

## Next Steps

1. Generate a small but meaningful depth4/depth5 tree dataset.
2. Train with conservative score-space limits and hard-valid extra validation.
3. Run 20-game smoke only as a breakage check.
4. Use 100+ games or multiple seeds before adopting any weight.

This commit does not adopt any new weight and does not justify a release tag.
