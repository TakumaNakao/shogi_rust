# soft探索蒸留 v2 500局面 smoke

- 作成日時: 2026-06-19 19:15:50 UTC
- ブランチ: `exp/soft-distill-v2-500-smoke`
- 目的: `--teacher-score-source searched` と保守的blendを使い、前回のsoft蒸留失敗を改善できるか確認する。

## 教師データ

```bash
target/release/distill_dump \
  --weights policy_weights_v2.1.0.binary \
  --input converted_records2016_10818.sfen \
  --train-output /tmp/shogi_soft_v2_d4_top8_500/train.jsonl \
  --valid-output /tmp/shogi_soft_v2_d4_top8_500/valid.jsonl \
  --depth 4 \
  --teacher-score-top 8 \
  --teacher-score-depth 4 \
  --teacher-score-source searched \
  --max-positions 500 \
  --valid-percent 10 \
  --seed 12101 \
  --jobs 4
```

結果:

```text
train records: 450
valid records: 50
```

探索スコアgap分布:

```text
n=500
p25=0.060
p50=0.421
p75=1.810
p90=6.265
p95=21.665
max=366.829
avg=3.884
>=5cp: 53
>=10cp: 40
>=20cp: 29
>=40cp: 6
>=80cp: 3
```

`min-teacher-gap 80` は厳しすぎるため、今回の対局候補はgapなしで作成した。

## 学習

```bash
target/release/distill_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/shogi_soft_v2_d4_top8_500/train.jsonl \
  --valid /tmp/shogi_soft_v2_d4_top8_500/valid.jsonl \
  --output /tmp/shogi_soft_v2_d4_top8_500/trained_gap0.binary \
  --epochs 2 \
  --batch-size 128 \
  --learning-rate 0.2 \
  --softmax-temperature 100 \
  --teacher-temperature 100 \
  --min-teacher-gap 0
```

offline:

```text
baseline train samples=450 ce=3.675130 top1=0.1267
baseline valid samples=50 ce=3.656548 top1=0.1400
epoch 2 train_ce=3.675001 train_top1=0.1289 valid_ce=3.656431 valid_top1=0.1600
```

full重みをそのまま使わず、`blend_ratio=0.25` の候補を作成。

```bash
target/release/adjust_weights \
  --input policy_weights_v2.1.0.binary \
  --blend-target /tmp/shogi_soft_v2_d4_top8_500/trained_gap0.binary \
  --blend-ratio 0.25 \
  --output /tmp/shogi_soft_v2_d4_top8_500/blend_gap0_r025.binary
```

full重み `trained_gap0.binary` は容量節約のため削除済み。候補として残したのは `blend_gap0_r025.binary` のみ。

## 固定セット確認

`loss_in_check_low_reply.sfen`:

```text
total: 27
in_check: 27
low_legal_in_check: 21
terminal: 4
search_win: 0
search_loss: 26
legal_without_bestmove: 0
bestmove_gives_check: 0
bestmove_limits_reply: 0
```

`taildrop_root_rescue.sfen`:

```text
total: 12
in_check: 1
low_legal_in_check: 1
terminal: 0
search_win: 3
search_loss: 1
legal_without_bestmove: 0
bestmove_gives_check: 6
bestmove_limits_reply: 4
```

## 対局結果

比較条件:

```text
engine: 現行masterの同一 usi_engine
new weights: /tmp/shogi_soft_v2_d4_top8_500/blend_gap0_r025.binary
baseline weights: policy_weights_v2.1.0.binary
positions: taya36.sfen
depth: 5
time-limit-ms: 100
max-plies: 200
adjudicate-at-max-plies
jobs: 4
```

20局 / seed 12201:

```text
new wins: 14
baseline wins: 6
draws: 0
new total score rate: 70.00%
```

40局 / seed 12221:

```text
new wins: 23
baseline wins: 17
draws: 0
new total score rate: 57.50%
95% CI total: 42.18%..72.82%
end reasons:
  Resign: 40
paired starts:
  new sweeps: 3
  baseline sweeps: 0
  splits: 17
  draw/mixed pairs: 0
average final score for new: 56.8
```

## 暫定判断

前回のsoft蒸留候補は40局で 18-21-1 まで落ちたが、今回のsearched/depth4/blend候補は40局で 23-17、paired startsでも baseline sweep が0だった。

まだ採用には不十分だが、100局ゲートへ進める価値はある。100局で55%以上かつpaired starts優位なら、次に v2.4.1 相手の100局比較へ進める。

100局ベンチ:

```text
seed 12241 / 100 games
new wins: 52
baseline wins: 45
draws: 3
new decisive win rate: 53.61%
new total score rate: 53.50%
95% CI total: 43.87%..63.13%
end reasons:
  MaxPliesAdjudication: 2
  RepetitionDraw: 3
  Resign: 95
paired starts:
  new sweeps: 9
  baseline sweeps: 6
  splits: 33
  draw/mixed pairs: 2
average final score for new: 45.4
```

## 最終判断

不採用。

20局は70%、40局は57.5%だったが、100局では53.5%まで落ちた。paired startsは new sweeps 9 / baseline sweeps 6 とわずかに良いが、事前に置いた採用基準の55%を下回った。

今回の結果は、前回のdepth3/static候補よりは改善しているが、現行固定重みを明確に超えるほどではない。候補重み `/tmp/shogi_soft_v2_d4_top8_500/blend_gap0_r025.binary` は削除済み。

次にsoft蒸留を続けるなら、500局面ではなく局面数を増やすか、教師局面を `converted_records2016` のランダム局面だけでなく現行エンジンの敗局・互角局面に寄せる必要がある。ただし、現時点では探索/評価基盤の次候補と比較して優先度を再評価する。
