# ベンチ敗局フィードバック変換ツール

- 作成日時: 2026-06-27 18:32:03 UTC
- 対象ブランチ: `training/strong-weight-learning-infra`
- 目的: `bench_failure_miner` で抽出した実戦ベンチ敗局の失敗局面を、`mmto_tree_train --feedback-json` に投入できる形式へ変換する。

## 背景

直近の失敗マイニングでは、rootで深い教師探索なら救える局面は少なく、実際には「現行root探索が時間内に悪い手を選んでいる」ケースが主だった。

そのため、root rescue の探索変更を急ぐよりも、以下の形で重み学習へ戻す方が合理的と判断した。

- ベンチ敗局から `timed_move` と `teacher_move` が異なる局面を抽出する。
- `timed_regret_cp` が大きい局面だけを残す。
- `teacher_move` を良い手、`timed_move` を悪い手として `mmto_tree_train` の feedback loss に渡す。

## 追加内容

新規バイナリ:

```text
bench_failure_feedback
```

入力:

```text
bench_failure_miner の JSONL
```

出力:

```json
{
  "hard_positions": [
    {
      "sfen": "...",
      "teacher_best_move": "...",
      "baseline_move": "...",
      "candidate_move": "...",
      "candidate_regret": 150.0,
      "regret_delta": 150.0
    }
  ]
}
```

この形式は既存の `mmto_tree_train --feedback-json` と互換である。

## 検証

Spark サブエージェントに実験・検証を委任した。

実行結果:

```text
cargo build --release --bin bench_failure_feedback --bin mmto_tree_train: 成功

入力:
  /tmp/bench_failure_mining_suite_full_20260627_181158/all_failures.jsonl

閾値:
  --min-timed-regret-cp 150

input lines: 475
feedback samples: 52
filtered by regret: 421
filtered by moves: 0
filtered by dedupe: 2
```

JSON確認:

```text
hard_positions: 52
sfen 欠損: 0
teacher_best_move 欠損: 0
candidate_move 欠損: 0
candidate_regret 欠損: 0
regret_delta 欠損: 0
```

さらに、軽量な `mmto_tree_train --dry-run --feedback-json` 読み込み確認も成功した。

## 判断

この変更は探索アルゴリズムや評価関数の本番挙動には影響しない。次の目的は、ベンチで実際に負けた局面を feedback loss に混ぜ、単なる棋譜模倣ではなく「現行エンジンの具体的な悪手を避ける」方向へ学習を寄せることである。

次の学習実験では、以下を比較する。

1. 既存 tree dump のみ。
2. tree dump + `bench_failure_feedback` の軽い重み。
3. tree dump + `bench_failure_feedback` の強めの重み。

採用判定は offline gate のみでは行わず、最終的に `v2.1.0` baseline および現行固定版へのベンチで確認する。

## 追試: feedback混合小実験

Spark サブエージェントに、変換済みfeedbackを既存tree dumpへ混ぜる小実験を委任した。

条件:

```text
RUN_DIR=data/mmto/runs/mmto_bench_failure_feedback_w05_20260627_183504
SOURCE_RUN_DIR=data/mmto/runs/mmto_pv_sibling_cap025_3k_20260627_171155
TRAIN_LINES=7000
VALID_LINES=1000
FEEDBACK_JSON=data/mmto/feedback/bench_failures_gt150_20260627.json
FEEDBACK_WEIGHT=0.5
FEEDBACK_GOOD_MOVE=teacher
FEEDBACK_MIN_REGRET_DELTA_CP=150
FEEDBACK_MIN_CANDIDATE_REGRET_CP=150
FEEDBACK_MAX_SAMPLE_WEIGHT=3
EPOCHS=5
```

学習中の指標は少し改善した。

```text
valid p95: 177.92 -> 177.02
valid teacher_match: 24.00% -> 25.20% (best epoch 1)
feedback violation_ratio: 59.62% -> 53.85% (epoch 1)
score gate: PASSED
```

しかし探索込みのrerank gateでは悪化した。

```text
baseline:  mean=108.56 p90=29.47 p95=43.25 match=44.10%
candidate: mean=108.66 p90=29.51 p95=43.52 match=43.90%
bad50:  0.0370 -> 0.0380
bad100: 0.0040 -> 0.0050

RERANK GATE FAILED
```

失敗理由:

```text
mean regret worsened
p90 regret worsened
p95 regret worsened
bad50 ratio worsened
bad100 ratio worsened
match rate failed improvement requirement
```

判断:

- この候補重みは不採用。
- `score_gate` や静的なvalid指標だけでは不十分で、探索込みのrerank gateを必須にする方針を維持する。
- 今回の結果は「単純にもっとepochを増やせば改善する」というより、feedback局面の選別、重み付け、通常tree lossとの干渉、best metricの設計に問題が残っていることを示している。
- 次は、ベンチ敗局feedbackをそのまま混ぜるのではなく、探索上の候補集合内で teacher 手と candidate 手が直接競合している局面だけを使う、または序盤・中盤・終盤を分けて損失を調整する必要がある。
