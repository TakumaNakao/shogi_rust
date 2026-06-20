# policy regret tiny blend rejected

- 作成日時: 2026-06-20 04:55:00 UTC
- 判断: 不採用

## 背景

`policy_regret_probe` の結果、wdoor高レート勝者データの教師手は現行探索から見ても大きく破綻していなかった。

GPT-5.5 xhigh の判断に従い、過去の外部policy-only学習との差分として、以下の安全策を入れた小実験を行った。

- 現行探索 depth3 で `teacher_move_regret <= 50cp` の外部教師だけ採用。
- 合法手数が少ない強制局面を除外。
- 王手回避局面を除外。
- 詰み級・極端評価局面を除外。
- 低学習率。
- 1 epochのみ。
- 直接置換せず、最終重みは5% blend。

## データ作成

train:

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/policy_regret_probe \
  --weights policy_weights_v2.1.0.binary \
  --input /tmp/shogi_external_kifu/datasets_quality/wdoor2026_policy_train_200k_r4000_winner.jsonl \
  --depth 3 \
  --max-positions 20000 \
  --seed 15101 \
  --jobs 4 \
  --show-worst 5 \
  --export-accepted /tmp/policy_regret_tiny_blend/train_accepted_20k_d3_r50.jsonl \
  --max-accepted-regret-cp 50 \
  --min-accepted-legal-moves 10 \
  --max-accepted-abs-score-cp 3000 \
  --exclude-accepted-in-check
```

```text
samples: 20000
mean_regret_cp: 65.55
p50_regret_cp: 8.99
p90_regret_cp: 88.80
p95_regret_cp: 130.28
bad_regret_count_gt_300: 33 (0.17%)
teacher_move_match: 4895 (24.48%)
exported accepted: 12375
```

valid:

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/policy_regret_probe \
  --weights policy_weights_v2.1.0.binary \
  --input /tmp/shogi_external_kifu/datasets_quality/wdoor2026_policy_valid_20k_r4000_winner.jsonl \
  --depth 3 \
  --max-positions 5000 \
  --seed 15102 \
  --jobs 4 \
  --show-worst 5 \
  --export-accepted /tmp/policy_regret_tiny_blend/valid_accepted_5k_d3_r50.jsonl \
  --max-accepted-regret-cp 50 \
  --min-accepted-legal-moves 10 \
  --max-accepted-abs-score-cp 3000 \
  --exclude-accepted-in-check
```

```text
samples: 2000
mean_regret_cp: 33.26
p50_regret_cp: 11.05
p90_regret_cp: 101.45
p95_regret_cp: 145.15
bad_regret_count_gt_300: 3 (0.15%)
teacher_move_match: 481 (24.05%)
exported accepted: 1328
```

## 学習

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/distill_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/policy_regret_tiny_blend/train_accepted_20k_d3_r50.jsonl \
  --valid /tmp/policy_regret_tiny_blend/valid_accepted_5k_d3_r50.jsonl \
  --extra-valid unfiltered=/tmp/shogi_external_kifu/datasets_quality/wdoor2026_policy_valid_20k_r4000_winner.jsonl \
  --output /tmp/policy_regret_tiny_blend/finetuned_lr0005_e1.binary \
  --epochs 1 \
  --batch-size 512 \
  --learning-rate 0.005 \
  --softmax-temperature 100 \
  --freeze-material
```

```text
baseline train samples=12375 ce=4.100164 top1=0.2151
baseline valid samples=1328 ce=4.149981 top1=0.2011
baseline extra_valid[unfiltered] samples=2000 ce=4.008378 top1=0.2095

epoch 1 train_ce=4.100162 train_top1=0.2149
epoch 1 valid_ce=4.149980 valid_top1=0.2011
epoch 1 extra_valid[unfiltered] ce=4.008376 top1=0.2095
```

5% blend:

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/adjust_weights \
  --input policy_weights_v2.1.0.binary \
  --blend-target /tmp/policy_regret_tiny_blend/finetuned_lr0005_e1.binary \
  --blend-ratio 0.05 \
  --output /tmp/policy_regret_tiny_blend/blend_lr0005_e1_r005.binary
```

## regret gate

taya36:

```text
samples: 72
bad_regret_count_gt_300: 0 (0.00%)
teacher_move_match: 72 (100.00%)
```

accepted外部valid 200件:

```text
samples: 200
bad_regret_count_gt_300: 0 (0.00%)
teacher_move_match: 200 (100.00%)
```

## 20局スモーク

new weights:

```text
/tmp/policy_regret_tiny_blend/blend_lr0005_e1_r005.binary
```

baseline weights:

```text
policy_weights_v2.1.0.binary
```

条件:

```text
positions: taya36.sfen
games: 20
depth: 5
time-limit-ms: 100
seed: 15121
record-dir: /tmp/shogi_bench_policy_regret_r50_blend5_20_seed15121
```

結果:

```text
new wins: 9
baseline wins: 9
draws: 2
new decisive win rate: 50.00%
new total score rate: 50.00%
```

`record_analyze`:

```text
end reasons:
  MaxPliesAdjudication: 1
  RepetitionDraw: 2
  Resign: 17
paired starts:
  new sweeps: 2
  baseline sweeps: 2
  splits: 5
  draw/mixed pairs: 1
average final score for new: 156.7
terminal result mismatches: 0
non-terminal score/result sign mismatches: 0
```

## 判断

不採用。

regret filter と5% blendにより悪化は抑えられたが、20局で完全に中立だった。offline top1も改善しておらず、100局へ拡大する根拠がない。

候補重みは削除済み。

## 次の示唆

`regret <= 50cp` の外部棋譜policyは安全には見えるが、低LR・5% blendでは信号が弱すぎる。

外部棋譜を使う場合は次のどちらかに進む必要がある。

- 50k-100k acceptedまで増やし、同じ5% blendで再試行する。
- policy CEではなく、探索値付き候補手比較やvalue/policy混合目的にする。

ただし、過去のpolicy-only失敗を踏まえると、単に件数を増やすだけでは95%目標には届きにくい。
