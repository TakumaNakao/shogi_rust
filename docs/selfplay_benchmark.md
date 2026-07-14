# Persistent selfplay benchmark

`usi_benchmark --persistent-engines` runs the candidate and baseline engines as
one long-lived USI process each. Games alternate the candidate's color and use
the same shuffled starting-position sequence as the regular benchmark. The
persistent mode is intentionally single-threaded, emits only the final
aggregate, and does not write game records.

Example (200 games, low-load guard):

```sh
scripts/safe_search_run.sh \
  --cpus 0-3 --nice 10 --timeout 1800 --cpu-sec 3600 \
  --max-rss-kb 3145728 --vm-kb 3145728 \
  --log report/selfplay_200.safe.log -- \
  target/release/usi_benchmark \
  --new-engine target/release/usi_engine \
  --baseline-engine /tmp/shogi_baseline/usi_engine \
  --new-weights policy_weights_v2.1.0.binary \
  --baseline-weights policy_weights_v2.1.0.binary \
  --positions taya36.sfen --games 200 --depth 4 \
  --time-limit-ms 100 --max-plies 200 --seed 20260714 \
  --jobs 1 --persistent-engines
```

Use the same engine binaries, weights, position file, depth, time limit, and
seed for baseline/candidate comparisons. Do not combine this mode with
`--record-dir`; the aggregate is the reproducible result artifact and the
wrapper log records resource limits and exit status.
