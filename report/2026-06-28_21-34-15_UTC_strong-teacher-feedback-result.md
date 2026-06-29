# Strong-teacher PV feedback 実験結果

- 作成日時: 2026-06-28 21:34:15 UTC
- 対象ブランチ: `training/strong-weight-learning-infra`
- 目的: PV sibling feedback の hard node だけを深めの teacher で再ラベルし、feedback-only 学習の信号品質を上げる

## 実装

`tools/run_pv_sibling_strong_teacher_feedback.sh` を追加した。

特徴:

- 既存 feedback JSON から hard node だけを抽出する。
- `teacher_best_move` と `candidate_move` を保持した JSONL を作る。
- `mmto_tree_dump` で hard node だけを `teacher_depth=4` で再探索する。
- `tree_feedback_collect` で feedback JSON を作る。
- feedback-only `mmto_tree_train`、score gate、rerank gate、短期ベンチまで流す。

また、`tree_feedback_collect` に以下を追加した。

- `--max-candidate-regret-cp`
- `--max-regret-delta-cp`

これは 100000cp 級の mate/大差外れ値を通常の重み更新から外すためのもの。

## 実験1: 外れ値込み strong-teacher feedback

- run dir: `data/mmto/runs/pv_sibling_strong_teacher_20260628_203933`
- hard relabel positions: 800
- teacher depth: 4
- student depth: 2

Dump:

- total positions: 800
- root records: 799
- pv sibling records: 1377
- train records: 1743
- valid records: 433
- skipped: 1 (`max_abs_root_score`)

Feedback:

- train records: 682
- guard records: 227
- train candidate regret mean/min/max: 528.79 / 30.02 / 100211.19
- guard candidate regret mean/min/max: 529.19 / 30.33 / 99737.80

Learning:

- baseline feedback loss: 99.797035
- baseline violation: 0.555066
- best epoch: 12
- best feedback loss: 99.790871
- best violation: 0.497797

Gates:

- score gate: PASS
  - mean abs delta: 0.0249 cp
  - p95 abs delta: 0.0824 cp
  - max abs delta: 0.2037 cp
- rerank gate: FAIL
  - reason: bad200 ratio worsened from 0.0208 to 0.0231

補足:

mean/p90/p95、bad50、bad100、match は改善していた。bad200 だけが1件相当悪化したため棄却した。

## 実験2: 外れ値除外 strong-teacher feedback

- run dir: `data/mmto/runs/pv_sibling_strong_teacher_filtered_20260628_212449`
- 既存 depth4 dump を再利用
- `--max-candidate-regret-cp 600`
- `--max-regret-delta-cp 600`

Feedback:

- train records: 679
- guard records: 226
- train candidate regret mean/min/max: 88.71 / 30.02 / 327.57
- guard candidate regret mean/min/max: 90.54 / 30.51 / 284.50

Learning:

- baseline feedback loss: 98.585358
- baseline violation: 0.5088
- best epoch: 12
- best feedback loss: 98.576920
- best violation: 0.4558

Gates:

- score gate: PASS
  - mean abs delta: 0.02 cp
  - p95 abs delta: 0.08 cp
  - max abs delta: 0.20 cp
- rerank gate: PASS
  - baseline mean/p90/p95: 270.11 / 113.26 / 145.35
  - candidate mean/p90/p95: 268.08 / 102.74 / 142.94
  - baseline match: 33.72%
  - candidate match: 35.33%
  - baseline bad50/bad100/bad200: 0.3095 / 0.1178 / 0.0208
  - candidate bad50/bad100/bad200: 0.2933 / 0.1085 / 0.0208

20局ベンチ:

- new wins: 9
- baseline wins: 11
- draws: 0
- score rate: 45.00%

判断:

棄却。offline gate はかなり良いが、20局ベンチで悪化した。

## 結論

hard node を depth4 で再ラベルする方針は、offline 指標には明確に効く。特に外れ値除外後は rerank gate をすべて通過した。

一方で、20局ベンチでは `9-11` と悪化した。これは以下のどれかが原因と考えられる。

1. hard node 局所の改善が、taya36 の通常対局分布に汎化していない。
2. 更新量が小さすぎて実戦差はノイズ域。
3. depth4 teacher が局所的には良いが、現行探索depth5対局の勝敗には直結していない。
4. feedback-only 更新が局所的すぎ、保護なしで他の局面をわずかに壊している。

## 次の方針

次は同じ strong-teacher filtered feedback を使い、以下のどちらかを試す。

1. `policy_anchor_weight` または `policy_anchor_margin_weight` を入れて、v2.1.0 の通常分布を保護する。
2. 20局で悪化した局面を `bench_failure_miner` / `tree_feedback_collect` に戻し、bench failure feedback と strong-teacher feedback を混ぜる。

採用条件は変えない。

- score gate PASS
- rerank gate PASS
- 20局で悪化なし
- 40局で改善傾向
- 100局で 55% 以上または複数seedで安定

