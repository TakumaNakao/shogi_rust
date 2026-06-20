# value regression v2 guarded rejected

- 作成日時: 2026-06-20 03:35:00 UTC
- 実験ブランチ: `experiment/value-regression-v2-guarded`
- 判断: 不採用

## 背景

探索ヒューリスティックと基礎高速化の短期候補が続けて不採用になったため、GPT-5.5 xhigh の提案に従い、重み更新を小さく再開した。

過去の value/drop-window regression は hard/drop を強く当てすぎて対局で悪化したため、今回は通常局面を主、hard/dropを補助にした guarded 設計にした。

## データ

作業ディレクトリ:

```text
/tmp/value_v2_guarded
```

### normal

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/value_dump \
  --weights policy_weights_v2.1.0.binary \
  --input converted_records2016_10818.sfen \
  --train-output /tmp/value_v2_guarded/normal_train.jsonl \
  --valid-output /tmp/value_v2_guarded/normal_valid.jsonl \
  --depth 4 \
  --max-positions 1200 \
  --valid-percent 20 \
  --jobs 4 \
  --seed 14001
```

```text
train records: 960
valid records: 240
skipped positions: 0
```

### hard/drop

入力:

```text
/tmp/shogi_hard_v241_12301_200/baseline_win_tails.sfen
/tmp/shogi_value_drop/drop_windows.sfen
```

重複除去して300局面に制限。

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/value_dump \
  --weights policy_weights_v2.1.0.binary \
  --input /tmp/value_v2_guarded/hard_input.sfen \
  --train-output /tmp/value_v2_guarded/hard_train.jsonl \
  --valid-output /tmp/value_v2_guarded/hard_valid.jsonl \
  --depth 4 \
  --max-positions 300 \
  --valid-percent 20 \
  --jobs 4 \
  --seed 14002
```

```text
train records: 240
valid records: 60
skipped positions: 0
```

### train mix

```text
normal_train: 960
hard_train: 240
total: 1200
hard ratio: 20%
```

## lr20

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/value_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/value_v2_guarded/train_mix.jsonl \
  --valid /tmp/value_v2_guarded/normal_valid.jsonl \
  --extra-valid hard=/tmp/value_v2_guarded/hard_valid.jsonl \
  --output /tmp/value_v2_guarded/candidate_lr20.binary \
  --epochs 1 \
  --batch-size 256 \
  --learning-rate 20 \
  --score-clip 3000 \
  --target-scale 600 \
  --huber-delta 1 \
  --sign-threshold 200 \
  --seed 14004
```

offline:

```text
baseline valid samples=240 huber=0.000472 rmse_cp=18.44 mae_cp=8.31 corr=0.6108
epoch valid    samples=240 huber=0.000472 rmse_cp=18.43 mae_cp=8.30 corr=0.6108

baseline extra_valid[hard] samples=60 huber=1.681602 rmse_cp=1572.28 mae_cp=1226.08 corr=0.8766
epoch extra_valid[hard]    samples=60 huber=1.681407 rmse_cp=1572.14 mae_cp=1225.94 corr=0.8766
```

regret probe:

```text
taya36:   bad_regret_count_gt_300 = 0 / 72
hard120:  bad_regret_count_gt_300 = 0 / 120
```

same-engine 20局:

```text
seed: 14101
new wins: 10
baseline wins: 10
draws: 0
new decisive win rate: 50.00%
```

`record_analyze`:

```text
paired starts:
  new sweeps: 2
  baseline sweeps: 2
  splits: 6
average final score for new: -79.6
terminal result mismatches: 0
non-terminal score/result sign mismatches: 0
```

## lr100

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/value_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/value_v2_guarded/train_mix.jsonl \
  --valid /tmp/value_v2_guarded/normal_valid.jsonl \
  --extra-valid hard=/tmp/value_v2_guarded/hard_valid.jsonl \
  --output /tmp/value_v2_guarded/candidate_lr100.binary \
  --epochs 1 \
  --batch-size 256 \
  --learning-rate 100 \
  --score-clip 3000 \
  --target-scale 600 \
  --huber-delta 1 \
  --sign-threshold 200 \
  --seed 14005
```

offline:

```text
baseline valid samples=240 huber=0.000472 rmse_cp=18.44 mae_cp=8.31 corr=0.6108
epoch valid    samples=240 huber=0.000471 rmse_cp=18.41 mae_cp=8.25 corr=0.6111

baseline extra_valid[hard] samples=60 huber=1.681602 rmse_cp=1572.28 mae_cp=1226.08 corr=0.8766
epoch extra_valid[hard]    samples=60 huber=1.680624 rmse_cp=1571.59 mae_cp=1225.40 corr=0.8767
```

regret probe:

```text
taya36:   bad_regret_count_gt_300 = 0 / 72
hard120:  bad_regret_count_gt_300 = 0 / 120
```

same-engine 20局:

```text
seed: 14121
new wins: 8
baseline wins: 12
draws: 0
new decisive win rate: 40.00%
```

`record_analyze`:

```text
paired starts:
  new sweeps: 1
  baseline sweeps: 3
  splits: 6
average final score for new: 37.9
terminal result mismatches: 0
non-terminal score/result sign mismatches: 0
```

## 判断

lr20 は offline と regret は安全だったが、20局で 10-10 と改善なし。

lr100 は offline 指標はわずかに改善し、regret も安全だったが、20局で 8-12 と悪化した。

今回の guarded value regression v2 は採用しない。offline の微小改善は対局強化に変換できなかった。

## 次の示唆

この規模の depth4 value regression では、現在の重みに対して実戦で意味のある改善を作るには信号が弱い可能性が高い。

重み更新を続ける場合は、次のどちらかに条件を変える。

- より大きい通常局面データと高品質な外部棋譜/強AI棋譜を使う。
- value回帰ではなく、探索最善手の policy/top-k 蒸留や、候補手比較型の学習に寄せる。

短期の次手としては、重みを壊さないため、大規模学習へ拡大する前にデータ品質と学習目的を再設計する。
