# Feedback-Only MMTO Probe Results

- 作成日時: 2026-06-28 14:33:07 UTC
- ブランチ: `training/strong-weight-learning-infra`
- 目的: 通常MMTO lossを外し、hard feedback pairだけで安全な重みdeltaを作れるか確認する。

## 背景

直近のMMTO-lite実験では、valid lossやteacher matchは改善しても、実際のAlphaBeta root選択を `mmto_rerank_gate` で見るとtail riskが悪化していた。GPT-5.5 xhighの判断でも、これは「学習不足」ではなく「目的関数が探索選択とずれている」可能性が高いとされた。

そのため、`--allow-empty-train` と `--feedback-guard-json` を追加し、通常のtree lossを完全に外したfeedback-only学習を試した。

## 実験条件

- feedback train: `data/mmto/runs/feedback_split_direct_probe_20260628_141513/feedback_medium_train.json`
- feedback guard: `data/mmto/runs/feedback_split_direct_probe_20260628_141513/feedback_medium_guard.json`
- train samples: 10
- guard samples: 5
- 通常tree train: 0 samples
- `BEST_METRIC=feedback-loss`
- `BEST_GUARD_FEEDBACK_VIOLATION_INCREASE=0`

## 結果

RUN_BASE:

`data/mmto/runs/feedback_loss_select_probe_20260628_142429`

### F: conservative

- `FEEDBACK_WEIGHT=0.2`
- `LEARNING_RATE=0.00008`
- baseline feedback: `loss=111.937607`, `violation=0.2000`
- epoch5 feedback: `loss=111.934090`, `violation=0.2000`
- `best_epoch=5`
- score gate: PASS
  - mean abs delta: `0.004907cp`
  - p95 abs delta: `0.014254cp`
  - max abs delta: `0.027277cp`
- rerank gate: PASS
  - mean: `6.798092`
  - p90: `27.28862`
  - p95: `31.9415`
  - bad50: `0.009009`
  - bad100: `0.009009`

### G: stronger

- `FEEDBACK_WEIGHT=1.0`
- `LEARNING_RATE=0.00005`
- baseline feedback: `loss=111.937607`, `violation=0.2000`
- epoch5 feedback: `loss=111.926651`, `violation=0.2000`
- `best_epoch=5`
- score gate: PASS
  - mean abs delta: `0.015316cp`
  - p95 abs delta: `0.044505cp`
  - max abs delta: `0.085146cp`
- rerank gate: PASS
  - mean: `6.798092`
  - p90: `27.28862`
  - p95: `31.9415`
  - bad50: `0.009009`
  - bad100: `0.009009`

## 判断

通常MMTO lossを外すと、held-out hard feedback violationを悪化させずにfeedback lossを下げられた。これは、直近の悪化原因がfeedback pair更新そのものではなく、広いtree/listwise lossとの混合で探索選択に逆向きの勾配が混ざっていた可能性を示す。

ただし、今回のdeltaは最大でも `0.085cp` 程度で、対局強度に出る規模ではない。採用候補ではなく、次の学習設計の足場と見る。

## 次の方針

1. hard feedbackを15件規模から100-200件規模へ増やす。
2. feedback-onlyまたはfeedback主導のconstrained updateを継続し、通常tree lossはregularizer扱いに下げる。
3. `feedback-loss` 改善だけでなく、held-out feedback violation非悪化、rerank bad50/bad100/p90/p95非悪化を必須ゲートにする。
4. 次の長時間学習前に、実ベンチ敗局や追加root探索からhard feedback poolを拡張する。
