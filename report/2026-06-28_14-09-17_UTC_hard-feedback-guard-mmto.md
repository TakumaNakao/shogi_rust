# Hard Feedback Guard for MMTO Checkpoint Selection

- 作成日時: 2026-06-28 14:09:17 UTC
- ブランチ: `training/strong-weight-learning-infra`
- 目的: MMTO-lite系の学習で、valid lossは改善しても実探索のroot選択が悪化する問題を早期検出する。

## 実装

`mmto_tree_train` に、hard feedbackをbest checkpoint選択のガードとして使う機能を追加した。

- `--best-guard-feedback-loss-increase`
- `--best-guard-feedback-violation-increase`

既存の `--feedback-json` を読み込み、`FEEDBACK_WEIGHT=0` でも各epochのfeedback指標を評価できる。これにより、通常の `valid-loss` や `p95-regret` をbest metricに使いながら、過去に壊れた探索局面で違反率が悪化するepochを採用候補から除外できる。

また、複数の `mmto_rerank_gate` 結果からfeedback JSONを集約する `rerank_feedback_collect` と、運用用スクリプト `tools/collect_rerank_feedback_pool.sh` を追加した。

## Spark実験結果

### 既知失敗局面4件での検証

- RUN_DIR: `data/mmto/runs/feedback_guard_probe_20260628_140134`
- feedback samples: 4
- baseline feedback: `loss=97.388176`, `violation_ratio=0.2500`
- epoch1-3: `violation=0.5000`
- epoch4-5: `violation=0.7500`
- 全epochで `best_guard_passed=false`
- `best_epoch=0`

この条件では、valid-loss方向に進む学習epochが既知の探索悪化局面をさらに悪化させることを検出し、候補重みを出さずに止められた。

### rerank結果67件からのfeedback pool

- RUN_DIR: `data/mmto/runs/feedback_pool_probe_20260628_140641`
- 入力 `rerank_gate.json`: 67ファイル
- strict: 9 samples
- medium: 15 samples
- broad: 11 samples

`feedback_medium.json` を使った短時間検証:

- baseline feedback: `loss=94.108765`, `violation=0.266667`
- epoch1-3: `violation=0.3333`
- epoch4-5: `violation=0.4000`
- 全epochで `best_guard_passed=false`
- `best_epoch=0`

## 判断

現状のMMTO-lite学習は、通常のvalid lossや局所的な教師一致方向には進んでも、実際のAlphaBeta探索で選ばれる手のtail riskを改善できていない。少なくとも今回の条件では、単純に学習時間を延ばすだけでは改善より悪化の蓄積が起きる可能性が高い。

ただし今回の実装により、悪化候補を早期に拒否する仕組みはできた。今後の24時間級学習は、このhard feedback guardとrerank gateを必ず通してから実行する。

## 次の課題

1. hard feedback poolがまだ小さいため、失敗局面を増やす。
2. 現在の目的関数が「静的な候補順位」寄りなので、探索後root選択の悪化を直接最適化する方向へ寄せる。
3. valid lossを主目的にするのではなく、feedback violation、rerank mean/p90/p95、bad50/bad100を採用ゲートの主指標にする。
4. 長時間学習の前に、同一dumpで `feedback violation` がbaseline以下になる小実験を必須にする。
