# Parallel search

The engine supports Lazy SMP without changing the alpha-beta, PVS, quiescence,
extension, or evaluation rules.

## USI

The default remains one thread:

```text
setoption name Threads value 16
```

Worker 0 retains the normal root ordering. Helper workers rotate the root move
order at each completed iterative-deepening pass. Workers share the
transposition table and stop signal, while positions, HalfKP accumulators,
history, killer moves, repetition state, and counters remain thread-local.
Only a fully completed iteration can provide the final move.

## Profiling

```bash
target/release/search_profile \
  --halfkp-weights data/policy_weights_halfkp64_kpp_distilled.binary \
  --positions taya36.sfen \
  --samples 8 \
  --depth 32 \
  --time-limit-ms 1000 \
  --threads 16
```

For paired matches, keep `JOBS=1` so games do not compete for the same CPU:

```bash
HALFKP_THREADS=16 KPP_THREADS=1 JOBS=1 \
  ./scripts/bench_halfkp64_vs_kpp_5s.sh
```

`Threads=1` uses the original local `HashMap` transposition table and bypasses
all parallel synchronization.
