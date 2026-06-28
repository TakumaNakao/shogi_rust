# Large Tree Feedback MMTO Probe

- 作成日時: 2026-06-28 15:14:14 UTC
- ブランチ: `training/strong-weight-learning-infra`
- 目的: 20k規模dumpからtree-derived feedbackを増やし、feedback-only学習で安全かつ探索指標非悪化のdeltaを作れるか確認する。

## 入力

SOURCE_RUN_DIR:

`data/mmto/runs/mmto_rerank_long_20260624_140151`

feedback生成条件:

- strict: train 28 / guard 10
- medium: train 163 / guard 54
- loose: train 562 / guard 188

採用poolはmedium。strictは少なすぎ、looseは広げすぎでノイズが増える懸念があるため。

## 実験結果

RUN_BASE:

`data/mmto/runs/tree_feedback_large_probe_20260628_144914`

### J: conservative

- `FEEDBACK_WEIGHT=0.3`
- `EPOCHS=5`
- `LEARNING_RATE=0.00008`
- baseline feedback: `loss=85.059196`, `violation=0.5185`
- epoch5 feedback: `loss=85.059006`, `violation=0.5185`
- `best_epoch=5`
- score gate: PASS
  - mean abs delta: `0.003219cp`
  - p95 abs delta: `0.009506cp`
  - max abs delta: `0.043819cp`
- rerank gate: PASS
  - baseline mean: `8.361163`
  - candidate mean: `8.348711`
  - p90/p95/bad50/bad100: non-worsening

### K: stronger

- `FEEDBACK_WEIGHT=1.0`
- `EPOCHS=8`
- `LEARNING_RATE=0.00005`
- `MAX_WEIGHT_DELTA=0.01`
- baseline feedback: `loss=85.059196`, `violation=0.5185`
- epoch8 feedback: `loss=85.058701`, `violation=0.5000`
- `best_epoch=8`
- score gate: PASS
  - mean abs delta: `0.008667cp`
  - p95 abs delta: `0.025458cp`
  - max abs delta: `0.111698cp`
- rerank gate: PASS
  - baseline mean: `8.361163`
  - candidate mean: `8.343131`
  - baseline p90: `29.151466`
  - candidate p90: `29.100239`
  - p95 unchanged
  - bad50/bad100 unchanged

## 判断

20k dump由来のtree feedbackでは、feedback-only学習によりheld-out feedback violationを悪化させず、rerank mean/p90もわずかに改善する候補が出た。

これは、これまでのMMTO-lite混合lossとは異なり、search-aligned feedbackのみを使えば安全な方向へ進める可能性を示している。

ただし、score deltaは最大でも `0.111698cp` 程度であり、対局強度に直接出る規模ではまだない。現段階ではリリース候補ではなく、学習アルゴリズムの方向性確認と位置付ける。

## 次の方針

1. medium/loose tree feedbackで少し大きいdeltaを狙う。
2. held-out feedback violation、score gate、rerank gateをすべて通った候補だけ小規模ベンチへ進める。
3. ベンチ比較は同一エンジンで、candidate weight vs `policy_weights_v2.1.0.binary` として実施する。
4. 20局以下の結果で重み採用はしない。まずは悪化検出と候補探索に使う。
