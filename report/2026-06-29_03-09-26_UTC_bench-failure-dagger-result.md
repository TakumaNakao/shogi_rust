# Bench Failure DAgger Listwise 検証結果

- 作成日時: 2026-06-29 03:09:26 UTC
- 対象ブランチ: `training/strong-weight-learning-infra`

## 目的

bench failure miningで得た「実戦で候補が選んだ悪手」を、guardではなくtraining shardとして使う。
`mmto_tree_dump`へ明示的な`teacher_move`/`student_move`ペアを渡し、`student-leaf listwise`で現在の失敗分布を直接押し下げられるか確認した。

## 実装

以下を追加した。

- `tools/extract_bench_failure_pairs.py`
  - `bench_failure_miner`のJSONLを`mmto_tree_dump`が読めるJSONLへ変換する。
  - `teacher_move`を良手、`timed_move`または`actual_move`を悪手として出力する。
  - regret帯、最大regret、sample weight、dedupeを指定可能。
- `tools/run_bench_failure_dagger_listwise.sh`
  - bench failure JSONL -> explicit pair JSONL -> tree dump -> listwise train -> score/rerank/bench gate を実行する。

smokeでは、変換、tree dump、train引数接続、best_epoch=0時の重み削除まで確認した。

## 実験1: 高regret帯

- run dir: `data/mmto/runs/bench_failure_dagger_listwise_20260629_021233`
- regret帯: 150..100000cp
- extract: 184件中25件
- dump: 24 train records、1件skip

結果:

- baseline best metric: 0.864667
- best epoch: 0
- bench failure extra valid:
  - selected_regret_mean: 24947.41
  - bad100: 0.4167
- score/rerank/bench: 未実行

判断:

150cp以上では詰み級・大外れ値に寄りすぎ、通常の重み更新信号としては強すぎる。
best epochは出ず、不採用。

## 実験2: 通常regret帯

- run dir: `data/mmto/runs/bench_failure_dagger_normalband_20260629_023729`
- regret帯: 50..150cp
- extract: 184件中28件
- dump: 28 train records、skipなし

結果:

- baseline best metric: 0.769429
- best epoch: 3
- best value: 0.730714
- score gate: passed
  - mean_abs_delta_cp: 0.41
  - p95_abs_delta_cp: 1.24
  - max_abs_delta_cp: 1.80
- bench failure extra valid:
  - selected_regret_mean: 7170.80 -> 3622.81
  - bad100: 0.2857
- rerank gate: failed
  - p95 regret: 72.98 -> 73.06
  - match rate: 44.25% -> 43.12%
- 20局ベンチ: 未実行

判断:

通常regret帯は、bench failure shard上では明確に改善した。
しかし全体rerankのp95とmatch率をわずかに壊したため、採用できない。

## 次の方針

bench failure DAggerは、詰み級外れ値を外し、通常regret帯を使う方向がよい。
ただしhard shardだけを強めると全体の手選択一致率が落ちる。

次は以下を試す。

1. `REPLAY_WEIGHT`をさらに下げる。
2. `EXTRA_VALID_BEST_WEIGHT`は維持しつつ、`BEST_METRIC=capped-selected-regret`または`p95-regret`で選ぶ。
3. `CURRENT_TOP_MARGIN_WEIGHT`を弱め、teacher distribution CEとhard replayのバランスを取る。
4. dump depth/top-kを下げて反復速度を上げる。

採用条件は引き続き、score gate pass、rerank非悪化、最低20局で悪化なしとする。
