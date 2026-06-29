# PV sibling feedback 3K 実験結果

- 作成日時: 2026-06-28 19:55:10 UTC
- 対象ブランチ: `training/strong-weight-learning-infra`
- run dir: `data/mmto/runs/pv_sibling_feedback_20260628_190857`

## 目的

root-only feedback が実戦採用に弱かったため、PV sibling node まで teacher/student の不一致を拾う feedback-only 学習を 3K 局面へ拡大した。

## 実験条件

- 入力局面: `data/mmto/positions/wdoor2023_2026_r4000_p16_120.sfen`
- `MAX_POSITIONS=3000`
- `TEACHER_DEPTH=3`
- `STUDENT_DEPTH=2`
- `POSITION_CHUNK_SIZE=128`
- `--emit-pv-sibling-nodes`
- baseline weights: `policy_weights_v2.1.0.binary`
- candidate: `best.raw.binary`

## Dump 結果

- total positions: 3000
- root records: 2798
- pv sibling records: 9928
- train records: 11454
- valid records: 1272
- skipped positions: 202
  - `exclude_in_check`: 171
  - `max_abs_root_score`: 21
  - `min_legal_moves`: 10

## Feedback 学習

- feedback train records: 1125
- feedback guard records: 375
- baseline feedback loss: 103.243195
- baseline violation ratio: 0.5760
- best epoch: 12
- best feedback loss: 103.232384
- best violation ratio: 0.5253

更新方向は一貫していたが、score delta は非常に小さかった。

## Offline Gate

Score gate:

- samples: 900
- mean abs delta: 0.05 cp
- p95 abs delta: 0.18 cp
- max abs delta: 0.22 cp
- result: PASS

Rerank gate:

- baseline mean regret: 15.22
- candidate mean regret: 14.68
- baseline p90/p95: 43.82 / 73.05
- candidate p90/p95: 43.54 / 72.61
- baseline match: 39.20%
- candidate match: 39.80%
- result: PASS

## 対局ベンチ

20局:

- new wins: 12
- baseline wins: 8
- draws: 0
- paired starts: new sweeps 2, baseline sweeps 0, splits 8

40局:

- new wins: 21
- baseline wins: 15
- draws: 4
- total score rate: 57.50%
- decisive win rate: 58.33%
- paired starts: new sweeps 3, baseline sweeps 1, splits 13, draw/mixed 3

100局:

- new wins: 51
- baseline wins: 44
- draws: 5
- total score rate: 53.50%
- decisive win rate: 53.68%
- total score rate 95% CI: 43.97%..63.03%
- decisive win rate 95% CI: 43.71%..63.37%
- new as black: 25-24-1
- new as white: 26-20-4
- paired starts: new sweeps 8, baseline sweeps 5, splits 33, draw/mixed 4
- end reasons: Resign 94, RepetitionDraw 5, MaxPliesAdjudication 1

## 判断

この候補重みは採用しない。

20局と40局では有望に見えたが、100局では 53.5% まで縮み、信頼区間も広い。paired starts も `new sweeps 8 / baseline sweeps 5 / splits 33` で、明確に強くなったとは判断できない。

ただし、root-only feedback と違い、PV sibling feedback は offline gate と短期ベンチの両方で小さなプラスを出した。方向性は完全な失敗ではない。

## 次の方針

単純に長く回すのではなく、PV sibling の候補集合全体を使う hard-node listwise/groupwise objective へ進む。

現状の feedback-only 学習は、各局面の情報を `teacher best vs student bad` の単一ペアに圧縮している。そのため、PV sibling dump が持つ候補集合全体の teacher score 分布をほとんど使えていない。

次に実装する内容:

1. PV sibling hard node を抽出し、候補 top16/top24 全体に対する listwise CE または KL loss を追加する。
2. teacher 上位手と student/current 悪手の margin 補助を残す。
3. mate 級や 10000cp 超の外れ値は除外または低重みにする。
4. score delta は今回の 0.18cp より大きく、まず p95 1..5cp 程度を狙う。
5. offline gate と 20/40/100局ベンチで採否を判断する。

