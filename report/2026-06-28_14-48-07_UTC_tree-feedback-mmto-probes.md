# Tree-Derived Feedback MMTO Probe Results

- 作成日時: 2026-06-28 14:48:07 UTC
- ブランチ: `training/strong-weight-learning-infra`
- 目的: `mmto_tree_dump` のroot候補から、teacher-best手とstudent-selected高regret手のfeedback pairを作り、feedback-only学習の母集団を増やせるか確認する。

## 実装

`tree_feedback_collect` を追加した。

- 入力: `mmto_tree_v1` JSONL
- good: teacher rank最上位の手
- bad: student/current searchが選んだ手
- filter:
  - `--min-candidate-regret-cp`
  - `--min-regret-delta-cp`
  - `--max-good-regret-cp`
- `--guard-output` と `--guard-percent` でtrain/guard split可能

これにより、過去の `rerank_gate.json` 由来では15件程度だったhard feedbackを、既存tree dumpから100件規模へ増やせる。

## Spark実験

RUN_BASE:

`data/mmto/runs/tree_feedback_probe_20260628_143809`

入力:

- `data/mmto/runs/phase_gate_listwise_ce_mid_20260628_132336/train.tree.jsonl`
- `data/mmto/runs/phase_gate_listwise_ce_mid_20260628_132336/valid.tree.jsonl`

feedback生成:

- strict: train 46 / guard 16
- medium: train 99 / guard 33

採用poolはmedium。理由は、strictより件数が約2倍あり、guard評価に最低限の厚みがあるため。

## 結果

### H: conservative

- `FEEDBACK_WEIGHT=0.3`
- `LEARNING_RATE=0.00008`
- baseline feedback: `loss=110.541862`, `violation=0.7273`
- epoch5 feedback: `loss=110.534950`, `violation=0.7273`
- `best_epoch=5`
- score gate: PASS
  - mean abs delta: `0.012969cp`
  - p95 abs delta: `0.040839cp`
  - max abs delta: `0.054153cp`
- rerank gate: PASS
  - baseline mean: `6.798092`
  - candidate mean: `6.798092`
  - p90/p95/bad50/bad100: non-worsening

### I: stronger

- `FEEDBACK_WEIGHT=1.0`
- `LEARNING_RATE=0.00005`
- baseline feedback: `loss=110.541862`, `violation=0.7273`
- epoch8 feedback: `loss=110.523170`, `violation=0.7273`
- `best_epoch=8`
- score gate: PASS
  - mean abs delta: `0.035055cp`
  - p95 abs delta: `0.109801cp`
  - max abs delta: `0.146075cp`
- rerank gate: FAIL
  - baseline mean: `6.798092`
  - candidate mean: `6.799769`
  - p90/p95/bad50/bad100: unchanged
  - fail reason: `mean regret worsened`

## 判断

tree-derived feedbackは、rerank由来feedbackよりサンプル数を増やせる。保守的設定ではheld-out feedback violationを悪化させず、score/rerank gateも通る安全なdeltaを作れた。

一方で、強めるとrerank meanがわずかに悪化する。差分は小さいが、現行の採用基準では不採用でよい。

この結果は「単純にMMTO-liteを長く回す」のではなく、「search-aligned feedbackを増やし、feedback-lossを下げつつrerank tailをゲートする」方向が有望であることを示す。ただし現時点のdeltaはまだ小さく、対局強度に効く規模ではない。

## 次の方針

1. 20k規模dumpからtree feedbackをさらに増やす。
2. feedback-onlyを基本にし、通常tree/listwise lossは一旦混ぜない。
3. larger feedback poolで `feedback-loss` が下がり、held-out violationとrerank mean/p90/p95/bad50/bad100が非悪化になるか確認する。
4. safe deltaが安定して出るようになってから、少量のregularizerやblendを検討する。
