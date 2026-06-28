# Feedback-only MMTO の現状判断と line-search 計画

- 作成日時: 2026-06-28 16:26:26 UTC
- ブランチ: `training/strong-weight-learning-infra`
- 目的: KPP重み学習で「単純に長く回せば改善するか」を判断し、次の実験順序を固定する。

## 結論

現時点では、同じ MMTO-lite / tree loss を長時間回す方針は優先しない。

通常の tree/listwise loss は offline loss を改善しても、rerank/hard feedback で悪化する例が多い。一方、feedback-only は score/rerank gate を通しやすく、安全な方向には動くが、更新量が小さすぎて対局ベンチ上の強さにはまだ出ていない。

したがって次は、長時間学習ではなく delta line-search で「方向は正しいが小さすぎる」のか、「方向自体が弱い」のかを切り分ける。

## 最新実験

実験ディレクトリ:

```text
data/mmto/runs/tree_feedback_candidate_probe_20260628_151607
```

source dump:

```text
data/mmto/runs/mmto_rerank_long_20260624_140151
```

### medium feedback

- train: 163件
- guard: 54件
- best_epoch: 15
- feedback loss: 96.533943 -> 96.533394
- violation_ratio: 0.5370 -> 0.5370
- score gate: PASS
- rerank gate: PASS

### loose feedback

- train: 562件
- guard: 188件
- best_epoch: 10
- feedback loss: 100.222679 -> 100.221817
- violation_ratio: 0.4734 -> 0.4628
- score gate: PASS
- rerank gate: PASS

20局ベンチ:

```text
NewWin: 10
BaselineWin: 10
Draw: 0
score rate: 50.00%
paired starts:
  new sweeps: 1
  baseline sweeps: 1
  splits: 8
  draw/mixed: 0
```

この結果は採用不可。安全だが、強さとしては中立。

## なぜ長時間化だけでは弱いか

1. feedbackサンプル数が少ない。
   - 18k行以上の tree dump から loose でも 750件程度しか作れていない。
   - 実際の violation 改善は数サンプル分に留まる。

2. 更新量が小さすぎる。
   - score gate 上の p95_abs_delta は 0.04cp 程度。
   - 探索のroot選択を安定して変える規模ではない。

3. 通常 tree loss は探索整合性を壊しやすい。
   - offline loss は改善しても rerank mean/p90/p95 や hard feedback が悪化する。
   - これは目的関数が「実際にAlphaBetaが選ぶ手」とずれていることを示す。

## 次の優先順

### 1. delta line-search

baseline -> feedback-only candidate の差分を alpha 1, 2, 4, 8, 16 に拡大し、各alphaを以下で評価する。

- score gate
- rerank gate 1000局面
- feedback guard
- 通過した最大alphaのみ20局ベンチ

判定:

- alphaを上げても p95_abs_delta が 0.1cp 未満なら、方向が実用未満。
- alpha拡大で rerankが悪化するなら、長時間学習ではなく目的関数変更へ進む。
- 20局ベンチは採用判定に使わず、次の40/100局に進むかの一次判定にする。

### 2. rerank閉ループ feedback-only

line-searchで安全なalphaが見つかった場合、候補重みでrerankを再実行し、実際に候補が間違えた局面を feedback 化する。

通常 tree loss は混ぜず、feedback-only で1-2世代だけ回す。

### 3. PV/sibling feedback dump

閉ループでも差分が小さい場合は、rootだけでなく探索中のPV近傍・兄弟局面を学習対象にする。

目標:

- 20k root dump から数千件以上の feedback を作る。
- held-out feedback violation 改善と rerank非悪化を両立する。

## 運用メモ

- 実験は `GPT-5.3-codex-spark` に委任する。
- 大きな方針判断は `GPT-5.5 xhigh` に委任する。
- 採用候補でない `.binary` は削除する。
- 20局結果だけで重みを採用しない。
