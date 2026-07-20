# Refactoring handoff after v2.5.4

This document records the behavior that must survive the planned large
refactoring. It also separates known defects from cleanup opportunities so that
a structural rewrite does not silently turn into an algorithm change.

## Repository state

- The released engine baseline is `v2.5.4`.
- `master` also contains the live-progress fix for `usi_benchmark`.
- Released v2.5.x binaries use HalfKP-64 and the weights originally released
  with v2.5.0.
- The newly trained weights are an experiment. They have not replaced the
  released weights.
- Generated datasets, checkpoints, match records, and trained weights are local
  artifacts under `data/` and must not be committed.

The learning pipeline added after v2.5.4 consists of:

- `src/bin/dataset_build.rs`: deterministic game-level split and phase-balanced
  position extraction from CSA records.
- `src/bin/halfkp_search_teacher.rs`: teacher generation using this engine's own
  alpha-beta search.
- `src/halfkp_training.rs`: versioned packed teacher dataset reader/writer.
- `src/bin/halfkp_search_train.rs`: HalfKP-64 value and ranking trainer.
- `scripts/train_halfkp64_search.sh`: small development pipeline.
- `scripts/train_halfkp64_large.sh`: resumable large training pipeline.
- `scripts/bench_halfkp64_learned_vs_v250.sh`: paired learned-versus-v2.5.0
  benchmark.

## Non-negotiable invariants

### Search and USI

- Every legal, non-terminal `go` command must eventually emit exactly one legal
  `bestmove`, including timeout, external `stop`, worker panic, forced mate, and
  all-losing-root cases.
- Worker 0 owns the reported root result. Helper workers may populate the shared
  transposition table and request a shared stop, but may not publish a partially
  searched root move as the final answer.
- `Threads=0` means all logical CPUs available to the process, capped at 256.
  `Threads=1` retains the serial search path.
- Position, repetition history, HalfKP accumulator, history heuristic, killer
  moves, and node counters are worker-local. Only the transposition table and
  stop state are shared.
- A stop request made before search starts must not be cleared by search setup.
- Mate scores stored in the transposition table are ply-normalized and must
  round-trip correctly when probed at a different ply.
- A checked quiescence node has no stand-pat option and searches every legal
  evasion, including quiet moves. The maximum quiescence ply is still enforced.
- Search changes require fixed-position node/NPS checks and paired games. NPS
  alone is not a strength criterion.

### HalfKP evaluation

- Build the engine, trainer, and tools with the same `halfkp64` feature.
  HalfKP-32 and HalfKP-64 weights are intentionally incompatible.
- `HalfKpModel::MAGIC`, hidden width, input count, king-bucket count, piece-state
  count, tensor order, and little-endian `f32` representation form the deployed
  model format. Do not change them without a versioned migration.
- The incremental accumulator must be numerically equivalent to a full refresh
  after every legal move, capture, promotion, drop, and king-bucket change.
- The AVX2 and portable kernels must produce equivalent results. Windows builds
  must retain a portable fallback and must not assume AVX2 is available.
- Feature indices are perspective-dependent. King mirroring, piece ownership,
  hand slots, side-to-move selection, and material sign are evaluation
  semantics, not implementation details.
- The output combines the side-to-move accumulator, the opponent accumulator,
  and the side-to-move material term. Refactoring must preserve this ordering
  and score perspective.

### Teacher data and learning

- Packed teacher files use magic `HKST0002`, version 2, and contain the compiled
  HalfKP hidden/input dimensions. Readers must reject incompatible files.
- Candidate flags are a bit set:
  `1=search best`, `2=game move`, `4=random`, `8=tactical`.
- Candidate scores are root utilities. Search evaluates a child from the
  child's side to move, so teacher generation must negate the child score
  exactly once.
- Randomized descendants do not inherit the original game result.
- Train/validation/test separation is performed at game level with a fixed seed.
  The current large run uses 2023-2025 for training and 2026 for validation and
  test. Never split neighboring positions from one game across these sets.
- Phase quotas, rating filters, known-result filtering, and per-game caps prevent
  a few long games or one phase from dominating the dataset.
- Checkpoints contain optimizer state as well as weights. Resume must reproduce
  the next epoch rather than restart the optimizer.
- Validation selects the best checkpoint. The final test set must not influence
  hyperparameters, early stopping, kappa fitting, or checkpoint selection.

## Current experimental result

The completed large AdaGrad run stopped after epoch 8. Its final test metrics
were:

```text
records=33123
brier=0.05966610
logloss=0.678887
search_mae=75.313
top1=0.3125
pair=0.7355
regret_mean=81.29
regret_p95=150.0
game_top1=0.2141
```

At 1000 ms per move, the paired benchmark against the v2.5.0 weights produced
102 wins, 92 losses, and 6 draws in 200 games: a 52.50% score with a 95%
confidence interval of 45.68% to 59.32%. This is a promising candidate, but it
does not establish a strength improvement. The result directory name used by
that run contains `5s`, but its recorded `config.txt` correctly says
`time_limit_ms=1000`.

The learned file observed in that run had SHA-256:

```text
8d1375790e58ae072103208011d7973020657b0fdda3b58e5eecd58b3c52003d
```

Do not promote this file over the released v2.5.0 weights solely from that
result.

## Known correctness debt

These are not refactoring-only tasks. Preserve current behavior during the
mechanical phase, then fix each item with characterization and regression tests.

1. Repetition adjudication does not prove that every move in the repeating
   cycle was a check before declaring a perpetual-check loss.
2. The engine rebuilds a position from USI `position ... moves`, but the
   pre-search repetition detector does not receive the complete game history.
3. `usi_benchmark` relies on the same repetition implementation. A benchmark
   adjudication can therefore be wrong even when both engine processes behave
   correctly.
4. Teacher labels inherit the search implementation's current repetition and
   horizon behavior. Regenerate affected teachers after a search correctness
   fix; do not mix incompatible label semantics unnoticed.
5. Quiescence dominates the node count in current profiles. Its move generation
   and filtering are performance-critical, but changes can easily alter tactics.
6. Windows startup, thread shutdown, and CPU feature fallback are exercised by
   CI builds but do not yet have an end-to-end ShogiHome regression test.

## Structural and performance debt

- `src/ai.rs` combines iterative deepening, root control, alpha-beta,
  quiescence, shared-table coordination, statistics, and some USI-facing
  concerns. Split by responsibility only after characterization tests exist.
- HalfKP serialization and forward-pass logic are duplicated between the runtime
  evaluator and trainer. Move them behind one model-format API.
- Phase classification and dataset conventions are repeated across the dataset
  builder, teacher generator, trainer, and shell scripts.
- The teacher generator constructs a fresh search object for many candidates.
  Reusing immutable model state and bounded worker state should reduce overhead.
- The trainer materializes sparse per-record gradients and aggregates them with
  hash maps. This is memory- and allocation-heavy at large batch counts.
- Validation and test records are loaded fully into memory, while training data
  is counted once and then read again. Introduce streaming manifests and cached
  counts before increasing dataset size substantially.
- The schedule-free optimizer is custom and much less exercised than AdaGrad.
  Treat AdaGrad as the reference path until optimizer equivalence, resume, and
  convergence tests are added.
- Scripts hard-code dataset years, thresholds, and several paths. A checked,
  versioned experiment manifest should become the single source of truth.
- Match confidence intervals currently treat games as independent even though
  openings are paired and the small `taya36.sfen` set is reused. Report paired
  outcomes and diversify openings for release decisions.

## Recommended refactoring order

1. Freeze behavior with characterization tests for USI completion, root search,
   mate-score TT conversion, repetition, HalfKP refresh versus incremental
   evaluation, model I/O, and teacher-file round trips.
2. Extract versioned model and teacher-data format modules without changing
   bytes or floating-point operation order.
3. Separate search state, shared search coordination, and USI lifecycle. Keep
   the serial path as the reference implementation.
4. Correct repetition/perpetual-check handling and pass full USI game history
   through to search. Apply the same adjudicator to the benchmark.
5. Consolidate feature extraction and phase/result conventions used by dataset,
   teacher, and trainer code.
6. Replace trainer-wide record materialization and sparse hash aggregation with
   bounded streaming batches, measuring peak RSS and epoch time.
7. Add an experiment manifest containing input hashes, engine revision, search
   settings, seeds, optimizer settings, and output hashes.
8. Run Linux tests, Windows CI, fixed-position NPS, teacher/trainer smoke tests,
   and paired games after each behavioral change.

Do not combine steps 1-3 with search parameter tuning, evaluation changes, or
new training objectives. That separation is necessary to identify regressions.

## Verification commands

Use dynamic Fontconfig loading on Linux systems without `pkg-config`:

```bash
RUST_FONTCONFIG_DLOPEN=1 cargo check --release --features halfkp64 \
  --bin dataset_build --bin halfkp_search_teacher --bin halfkp_search_train \
  --bin usi_engine --bin usi_benchmark

RUST_FONTCONFIG_DLOPEN=1 cargo test --release --features halfkp64 --lib
RUST_FONTCONFIG_DLOPEN=1 cargo test --release --features halfkp64 \
  --bin halfkp_search_train

PROFILE=smoke STAGE=all \
  WORK_DIR=data/halfkp_search_learning_smoke \
  ./scripts/train_halfkp64_large.sh
```

For a full learned-versus-release gate:

```bash
NEW_WEIGHTS=/path/to/candidate.binary \
BASELINE_WEIGHTS=policy_weights_halfkp64_kpp_distilled_v2.5.0.binary \
TIME_LIMIT_MS=5000 GAMES=200 JOBS=1 THREADS=0 \
./scripts/bench_halfkp64_learned_vs_v250.sh
```

Keep `JOBS=1` when each engine uses `Threads=0`, otherwise simultaneous games
compete for the same CPU and invalidate the time-control comparison.

## Generated artifacts

Do not commit:

- `data/halfkp_search_learning_v1/`
- `data/halfkp_search_learning_large_v1/`
- `data/halfkp_search_learning_smoke/`
- `data/bench_halfkp64_learned_vs_v250_*/`
- `.hkst` teacher shards, checkpoints, match records, or generated weights

Before merging a refactoring branch, inspect `git status`, `git diff --check`,
the weight and dataset hashes, and the benchmark `config.txt`. Directory names
are descriptive only; recorded configuration and hashes are authoritative.
