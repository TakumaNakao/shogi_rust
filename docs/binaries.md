# Binary inventory

`Cargo.toml`は`autobins = false`を指定し、この一覧にある実行targetだけを明示的に公開する。
既存のCLI名は変更していない。分類featureはtargetの用途と安定性を表し、評価方式の
`halfkp64` featureとは独立して組み合わせる。

production、supported training、benchmarkの`src/bin`は数行の互換entryだけを持つ。
実装は`usi_shogi`、`src/training_tools/`、`src/benchmark_tools/`のlibrary APIに置き、
testからCLI processを介さず利用できる。research targetは実験速度を優先し、この制約の対象外とする。

| 分類 | Cargo feature | 安定性 | 通常のbuild |
|---|---|---|---|
| production | 不要 | USIとweight formatの互換性を維持 | 含む |
| supported training | `training-tools` | CLI/manifest互換を管理 | 明示build |
| benchmark/profile | `benchmark-tools` | CIと性能検証用 | 明示build |
| research | `research-tools` | CLI/出力形式の互換保証なし | 除外 |

## Production

- `usi_engine`: ShogiHome等から起動するUSIエンジン。default-runでもある。

```bash
cargo build --release --features halfkp64 --bin usi_engine
```

production buildではtraining/research targetと、それらだけが使う`csa`、`glob`、
`plotters`、`sha2`を依存graphへ含めない。

## Supported training

- data preparation: `csa_policy_dump`, `csa_rate_stats`, `dataset_build`,
  `halfkp_dataset_shard`, `halfkp_rank_dataset`, `jsonl_shard`, `kpp_position_pool`,
  `rank_stats`
- teacher/trainer: `bonanza_pairwise_train`, `halfkp_kpp_train`,
  `halfkp_search_teacher`, `halfkp_search_train`, `halfkp_train`, `kpp_learn`,
  `kpp_self_learn`

```bash
cargo build --release --features halfkp64,training-tools --bin halfkp_search_teacher --bin halfkp_search_train
```

## Benchmark and profile

- `benchmark`, `eval_profile`, `halfkp_trace_bench`, `movegen_profile`, `perft`,
  `search_fingerprint`, `search_profile`, `usi_benchmark`

```bash
cargo build --release --features halfkp64,benchmark-tools --bin search_fingerprint
```

`search_fingerprint`はCIの意味的非退行gateであり、出力契約を変更するときはbaselineと
理由を同じcommitで更新する。

## Research

- weight/data experiments: `adjust_weights`, `distill_dump`, `distill_stats`,
  `distill_train`, `halfkp_feature_dump`, `kpp_weight_check`, `mmto_balance_dataset`,
  `mmto_dump`, `mmto_stability_filter`, `mmto_train`, `mmto_tree_dump`,
  `mmto_tree_train`, `nnue_feature_dump`, `nnue_rank_dump`, `record_finetune`,
  `value_dump`, `value_train`
- probes/gates: `mmto_probe`, `mmto_rerank_gate`, `mmto_score_gate`,
  `policy_regret_probe`, `position_probe`, `qsearch_candidate_probe`,
  `root_decision_probe`, `root_rescue_probe`, `see_probe`, `value_regret_probe`
- feedback/analysis/demo: `bench_failure_feedback`, `bench_failure_miner`,
  `record_analyze`, `rerank_feedback_collect`, `shogi_game`, `tree_feedback_collect`

```bash
cargo build --release --features research-tools --bin mmto_probe
```

## Maintenance contract

- binaryを追加・削除・分類変更するときは`Cargo.toml`とこの文書を同じcommitで更新する。
- supported trainingからresearchへの降格は利用者影響を記録する。
- CLI名を変更する場合は、旧名を同じ処理へ委譲するthin shimとして最低1 release残し、
  削除予定versionをこの文書へ記載する。
- 現時点でdeprecated targetはない。Phase 7では名称変更を行っていないためshimもない。
