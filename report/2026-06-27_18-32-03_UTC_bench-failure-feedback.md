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
