# MMTO rerank gates

- Date: 2026-06-23 14:36:37 UTC
- Branch: `feature/mmto-rerank-gates`
- Goal: prevent MMTO-tree training from accepting weights that improve cached loss but do not improve the move actually selected by search.

## Implemented

1. `mmto_tree_train` multi-threshold bad-regret logging
   - Added `--bad-regret-thresholds-cp`.
   - Default thresholds: `50,100,200,300`.
   - Stdout and CSV now include threshold-specific bad-regret ratios.
   - Existing `--bad-regret-cp` remains for `best-metric bad-regret` compatibility.

2. `mmto_rerank_gate`
   - Re-searches each validation position with both baseline and candidate weights.
   - Teacher-searches the selected moves and compares actual selected-move regret.
   - Fails when candidate mean/p90/p95 regret or bad-regret ratios worsen beyond allowed tolerances.
   - Outputs JSON with:
     - `baseline`
     - `candidate`
     - `worst_candidate`
     - `worst_delta`
     - `hard_positions`

3. Documentation update
   - Updated `docs/mmto_tree_training.md` with the stricter production-before-benchmark flow.
   - Clarified that `best_epoch=0` is not an adoption candidate.
   - Added rerank gate, hard-position extraction, DAgger redump, and blend steps.

## Smoke Validation

Small wiring smoke:

- `mmto_tree_dump`: 20 positions, depth2/depth1
- `mmto_tree_train`: 1 epoch, multi-threshold logging enabled
- `mmto_rerank_gate`: 4 valid positions, depth2/depth1

Observed:

```text
bad50 / bad100 / bad200 / bad300 appeared in train logs
mmto_rerank_gate produced hard_positions in JSON
RERANK GATE PASSED
```

Full test:

```text
env RUST_FONTCONFIG_DLOPEN=1 cargo fmt --check
env RUST_FONTCONFIG_DLOPEN=1 cargo check --bin mmto_tree_train --bin mmto_rerank_gate --bin mmto_tree_dump --bin mmto_score_gate --bin adjust_weights
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

All passed.

## Interpretation

This does not create or adopt a new weight. It adds the missing gate needed before spending longer compute on MMTO-tree learning. A candidate must now improve the move selected by re-search, not only the cached pairwise loss.
