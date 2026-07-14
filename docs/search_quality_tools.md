# Search quality tools

Phase 0 tooling provides deterministic search reports and rule-only evidence suites. None of
the three miners loads an evaluation weight or another engine's output.
Position inputs may be plain SFEN/USI lines or `dataset_build` JSONL records containing a `sfen`
field. Frozen suites require JSONL records with `source_game_key` (the existing `game_key` field
is accepted and normalized to that name). Every non-empty input line is parsed strictly; malformed
lines and empty effective inputs are errors.

## Fixed-node search

The USI engine accepts `go nodes N`. A nodes-only command has no implicit wall-clock limit.
When `nodes` is combined with `movetime` or clock fields, the first reached limit stops search;
if both are observed at the same check, the node limit has priority.

`search_profile` accepts the same limit:

```console
cargo run --release --bin search_profile -- \
  --positions taya36.sfen --samples 36 --depth 8 --nodes 200000
```

### Quiescence check boundary

The production default is `DEFAULT_MAX_QUIESCENCE_CHECK_PLY=0`. At a non-check node this makes
quiescence capture-only from its root. A checked node ignores the boundary: it never uses
stand-pat and always searches every legal king move, capture, move interposition, and drop
interposition. Complete checked-node evasions are required for correctness but substantially
increase the tree, so non-capture checks are excluded from production quiescence to contain that
growth.

`search_profile --qcheck-ply N` is a diagnostic override for that profiling process only. It does
not configure `usi_engine` or change the production default. The Phase 1 checked-evasion change
must not be released on its own: on the fixed 200-record dev mate suite it reduced exact expected
first moves from 83 to 71 and mate-acceptable choices from 187 to 172 under the planned search
budget.

`search_failure_probe` writes one JSONL record for every position/depth/node combination:

```console
cargo run --release --bin search_failure_probe -- \
  --weights policy_weights_v2.1.0.binary \
  --positions data/dev.sfen --output data/search_quality/probe.jsonl \
  --depths 3,4,5,6,7,8 --nodes 10000,50000,200000
```

Each record contains the best move, score, PV, completed depth, stop reason, nodes, qnodes,
in-check qnodes, terminal mates, negative-SEE checks, and repetition hits.

## Mate-sacrifice search diagnosis

`mate_sacrifice_search_probe` compares production search choices with rule-only mate evidence. It
uses the production quiescence setting and accepts development-suite JSONL records containing
`source_index`, `sfen`, `first_move`, and `mate_horizon`:

```console
cargo run --release --bin mate_sacrifice_search_probe -- \
  --input data/search_quality/generated/dev_mate_sacrifice.jsonl \
  --weights policy_weights_v2.1.0.binary \
  --depth 7 --nodes 20000 --proof-node-limit 2000000
```

The primary metrics are exact expected-first-move matches and mate-acceptable choices. A choice is
mate-acceptable only when the rule-only oracle proves mate after the selected root move within the
record's remaining horizon. The report also includes total search nodes, average completed depth,
mate scores, unknown oracle results, and the `source_index` values of mate-acceptable records;
those indices support failure-set mining across revisions.

This command is **dev-only**. Do not pass a holdout suite, inspect holdout record results, or use
holdout source indices for tuning. Holdout is reserved for a single release-candidate gate after
the implementation and thresholds have been frozen.

### Rule-only mate search budget

Production search runs a rule-only mate probe before evaluation search and selects its first move
only for `ProvenMate`. It searches 1, 3, 5, then 7 ply with checking moves as attacker OR nodes and
all legal evasions as defender AND nodes. Repetition is not mate, budget or deadline exhaustion is
`Unknown`, and `Unknown` is neither accepted nor rejected. The default root cap is 8192 nodes. On
the current dev suite its actual p95 is 3967 nodes; ordinary `taya36` positions terminate early
with a measured p95 of 291 nodes.

After evaluation search, at most three leading root candidates receive an opponent-mate probe of
128 nodes each. Only a proven opponent mate rejects a candidate. Mate nodes are included in the
global search node limit, and time probes share the global deadline. `SearchReport` exposes
`mate_nodes`, `mate_probes`, `mate_proven`, `mate_unknown`, and `mate_rejected`.

Use the dev-only offline profiler to compare caps without evaluation values:

```console
cargo run --release --bin mate_search_profile -- \
  --input data/search_quality/generated/dev_mate_sacrifice.jsonl \
  --budgets 128,256,512,1024,2048,4096,8192
```

The profiler rejects holdout-named inputs. Its results are development diagnostics and must not be
used to inspect or tune against holdout records.

With the fixed depth-7 / 20,000-total-node dev gate, the integrated default produces 110/200 exact
expected first moves and 199/200 mate-acceptable choices, above the pre-Phase-1 baselines of 83 and
187. All 16 Phase-1 mate-acceptable regressions are recovered. These are dev results, not release
or holdout claims.

## Rule-only miners

```console
cargo run --release --bin mate_sacrifice_miner -- \
  --positions data/dev.sfen \
  --output data/search_quality/generated/dev_mate_sacrifice.jsonl \
  --split dev \
  --depths 1,3,5,7 --proof-node-limit 2000000 --seed 20260713

cargo run --release --bin quiet_evasion_miner -- \
  --positions data/dev.sfen \
  --output data/search_quality/generated/dev_quiet_evasion.jsonl \
  --split dev \
  --seed 20260713

cargo run --release --bin resource_cycle_miner -- \
  --positions data/dev.sfen \
  --output data/search_quality/generated/dev_resource_cycles.jsonl \
  --split dev \
  --depths 4,6,8 --node-limit 250000 --seed 20260713
```

The mate miner uses full-width checking-move OR / legal-defense AND proof. A node-limit result
is `Unknown` and is never emitted as positive or negative evidence. The cycle miner likewise
emits only replay-validated positive witnesses; failure to find a cycle is not a negative label.
The evasion miner stores the complete legal evasion set as its reference.

Every miner writes `<output-stem>.manifest.json` with the generator commit, dirty-worktree flag,
generator-source SHA-256, input and output SHA-256, seed, extraction filters, record count, and
duplicate count. The generator hash covers `Cargo.toml`, `Cargo.lock`, `src`, and `shogi_lib`.
Rule-only
manifests record `weight_sha256` as `null` because no weight participates in labeling.
The suite and sidecar are prepared in temporary files and published as one pair. If either publish
step fails, both old files are restored. Inputs, outputs, weights, suites, and sidecars are rejected
when paths or existing inodes collide. A run producing zero evidence records fails without replacing
an existing suite or sidecar. Existing pair destinations must be regular files; directories,
symlinks, and other file types are rejected before either destination is changed. Once both new
files are published, removal of the old backup files is best effort: cleanup failure is warned but
does not misreport the consistent new pair as a failed publication, and the backups remain available
for manual recovery.

After all dev and holdout suites are generated, freeze the aggregate manifest specified by the
evaluation plan:

```console
cargo run --release --bin search_suite_manifest -- \
  --source-files data/dev.sfen data/holdout.sfen \
  --suite-files data/search_quality/generated/dev_mate_sacrifice.jsonl \
    data/search_quality/generated/holdout_mate_sacrifice.jsonl \
    data/search_quality/generated/dev_quiet_evasion.jsonl \
    data/search_quality/generated/holdout_quiet_evasion.jsonl \
    data/search_quality/generated/dev_resource_cycles.jsonl \
    data/search_quality/generated/holdout_resource_cycles.jsonl \
  --weight policy_weights_v2.1.0.binary \
  --output data/search_quality/generated/suite_manifest.json
```

The aggregate command is a freeze gate, not only a hash writer. By default it requires a clean
generator worktree and matching generator-source hashes, all six dev/holdout classifications,
the planned minimum counts, mandatory sidecars,
matching schema/count/seed/filter/dedupe/input/output SHA metadata, and zero source-game/SFEN
intersection between the complete dev and holdout source pools. Every source row must have a game
key, and every suite row must match the exact source line's game key and canonical SFEN. The gate
also parses every classification-specific field and replays its evidence: mate lines must terminate
in checkmate and be re-proved by the rule-only oracle, quiet-evasion sets must equal the complete
legal move set, and resource cycles must replay from their source and reproduce the declared hand
dominance. `--allow-dirty --allow-incomplete` exists only for smoke validation; using either flag
makes `freeze_eligible: false`, as does any dirty sidecar, and the result must not be used as a
release artifact.

Dev and holdout inputs must already be split by source game. Do not inspect or tune against
individual holdout records; if that happens, move the record to dev and freeze a replacement.

### Safe small-batch verification

Heavy probes must be run through `scripts/safe_search_run.sh`. It serializes jobs
with `flock` (or an atomic `mkdir` fallback), defaults to CPUs `0-3`, `nice 10`,
300 wall-clock seconds, 600 CPU seconds, a 1.5 GiB RSS limit, and a 1.5 GiB
virtual-memory (`VM_LIMIT_KB`) limit. It polls the complete child tree,
records maximum RSS and VmSize, and kills every recorded descendant with TERM
followed by KILL at any limit. Exit reasons are `rss_limit`, `vm_limit`,
`cpu_limit`, `timeout`, or `child_exit_N`; a nonzero child exit is never logged
as `completed`.

Example command:

```console
scripts/safe_search_run.sh --timeout 120 --log data/search_quality/safe-mate16.log -- \
  cargo run --release --bin mate_sacrifice_search_probe -- \
  --input data/search_quality/generated/dev_mate_sacrifice.jsonl --limit 16 \
  --nodes 50000 --time-limit-ms 500
```

Before running the full dev suite, use only these bounded batches:

- mate sacrifice: 16--32 records, at most 50,000 total nodes per record;
- quiet evasion: 32 records (`--record-limit 32`), miner-only legal proof;
- resource cycle: 16 records (`--record-limit 16`, `--node-limit 50000`),
  with the wrapper's 120-second wall-clock cap.

The miners' node limits are proof-search limits rather than evaluation limits.
When a proof search reaches its limit, the record is omitted or marked
`Unknown`; it is never treated as a negative label.

Do not pass holdout files to smoke tests. Run one wrapper process at a time and
inspect the log's `max_rss_kb` and exit reason. The 200-record mate run and self-play
are later gates and are deliberately excluded from Phase 5 hardening.

The wrapper's process-tree integration smoke covers normal completion, lock
contention (exit 75), timeout descendant cleanup, RSS-limit termination, and
CPU-limit termination:

```console
scripts/safe_search_run_smoke.sh
```

The default virtual-memory cap is intentionally conservative. A model-loading
smoke that needs more address space may pass `--vm-kb 4194304` (and, if needed,
`--vm-limit-kb 4194304`), but the measured RSS limit remains 1.5 GiB and is
still enforced by the wrapper. `MAX_RSS_KB`, `CPU_LIMIT_SEC`, and
`SAFE_SEARCH_VM_LIMIT_KB` can also override the defaults; unlimited values are
never defaults.
