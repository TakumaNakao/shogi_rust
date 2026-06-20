# policy regret soft blend rejected

- 作成日時: 2026-06-20 05:30:00 UTC
- 実験ブランチ: `experiment/policy-regret-soft-blend`
- 判断: 不採用

## 背景

`regret <= 50cp` の外部棋譜教師を hard label として5% blendする実験は、20局で 9-9-2 と中立だった。

次に、外部教師手を単独の正解として押し込まず、現行探索最善手と外部教師手を同じ `teacher_scores` に入れた soft target データを試した。

これは過去の外部policy-only失敗との違いとして、探索最善と矛盾する方向へ強く押さないことを狙った。

## tooling

`policy_regret_probe` に `--export-soft` を追加した。

accepted条件を満たす局面について、以下の形式でJSONLを書き出す。

```json
{
  "sfen": "...",
  "teacher_move": "...",
  "teacher_scores": [
    {"move_usi": "探索最善手", "score": 123.4},
    {"move_usi": "外部教師手", "score": 100.0}
  ]
}
```

`distill_train` は既存のsoft target処理でこの形式を読める。

## データ作成

train:

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/policy_regret_probe \
  --weights policy_weights_v2.1.0.binary \
  --input /tmp/shogi_external_kifu/datasets_quality/wdoor2026_policy_train_200k_r4000_winner.jsonl \
  --depth 3 \
  --max-positions 20000 \
  --seed 15301 \
  --jobs 4 \
  --show-worst 3 \
  --export-soft /tmp/policy_regret_soft_blend/train_soft_20k_d3_r50.jsonl \
  --max-accepted-regret-cp 50 \
  --min-accepted-legal-moves 10 \
  --max-accepted-abs-score-cp 3000 \
  --exclude-accepted-in-check
```

```text
samples: 20000
mean_regret_cp: 50.82
p50_regret_cp: 8.49
p90_regret_cp: 94.32
p95_regret_cp: 130.90
bad_regret_count_gt_300: 27 (0.14%)
teacher_move_match: 4949 (24.75%)
exported soft accepted: 12284
```

valid:

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/policy_regret_probe \
  --weights policy_weights_v2.1.0.binary \
  --input /tmp/shogi_external_kifu/datasets_quality/wdoor2026_policy_valid_20k_r4000_winner.jsonl \
  --depth 3 \
  --max-positions 5000 \
  --seed 15302 \
  --jobs 4 \
  --show-worst 3 \
  --export-soft /tmp/policy_regret_soft_blend/valid_soft_d3_r50.jsonl \
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
exported soft accepted: 1328
```

## 学習

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/distill_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/policy_regret_soft_blend/train_soft_20k_d3_r50.jsonl \
  --valid /tmp/policy_regret_soft_blend/valid_soft_d3_r50.jsonl \
  --extra-valid hard=/tmp/policy_regret_tiny_blend/valid_accepted_5k_d3_r50.jsonl \
  --extra-valid unfiltered=/tmp/shogi_external_kifu/datasets_quality/wdoor2026_policy_valid_20k_r4000_winner.jsonl \
  --output /tmp/policy_regret_soft_blend/finetuned_soft_lr0005_e1.binary \
  --epochs 1 \
  --batch-size 512 \
  --learning-rate 0.005 \
  --softmax-temperature 100 \
  --teacher-temperature 100 \
  --freeze-material
```

```text
baseline train samples=12284 ce=4.070124 top1=0.2209
baseline valid samples=1328 ce=4.111819 top1=0.2011
baseline extra_valid[hard] samples=1328 ce=4.149981 top1=0.2011
baseline extra_valid[unfiltered] samples=2000 ce=4.008378 top1=0.2095

epoch 1 train_ce=4.070124 train_top1=0.2209
epoch 1 valid_ce=4.111817 valid_top1=0.2011
epoch 1 extra_valid[hard] ce=4.149980 top1=0.2011
epoch 1 extra_valid[unfiltered] ce=4.008377 top1=0.2095
```

5% blend:

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/adjust_weights \
  --input policy_weights_v2.1.0.binary \
  --blend-target /tmp/policy_regret_soft_blend/finetuned_soft_lr0005_e1.binary \
  --blend-ratio 0.05 \
  --output /tmp/policy_regret_soft_blend/blend_soft_lr0005_e1_r005.binary
```

## regret gate

```text
taya36:
samples: 72
bad_regret_count_gt_300: 0 (0.00%)
teacher_move_match: 72 (100.00%)
```

## 20局スモーク

```text
record-dir: /tmp/shogi_bench_policy_regret_soft_blend5_20_seed15321
seed: 15321
games: 20
depth: 5
time-limit-ms: 100
```

```text
new wins: 10
baseline wins: 10
draws: 0
new decisive win rate: 50.00%
new total score rate: 50.00%
```

`record_analyze`:

```text
end reasons:
  Resign: 20
paired starts:
  new sweeps: 0
  baseline sweeps: 0
  splits: 10
  draw/mixed pairs: 0
average final score for new: -40.1
terminal result mismatches: 0
non-terminal score/result sign mismatches: 0
```

## 判断

不採用。

soft target化によりhard labelより理屈は安全になったが、20局では全ペアsplitの完全中立だった。offline指標もCEがごくわずかに改善しただけで、top1は変化しなかった。

候補重みは削除済み。

## 次の示唆

低LR 5% blend は安全だが、20k規模では信号が弱すぎる。

外部棋譜policyを続けるなら、50k-100k acceptedへ拡張する必要がある。ただし、ここまでの結果から policy-only では大幅な勝率向上は期待しにくい。次は、評価表現力の改善、小型NNUE、または探索値を直接学ぶ比較/value混合目的へ寄せるべき。
