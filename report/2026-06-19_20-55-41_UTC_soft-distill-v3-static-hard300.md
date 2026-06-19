# soft探索蒸留 v3 static hard300 smoke

- 作成日時: 2026-06-19 20:55:41 UTC
- ブランチ: `exp/soft-distill-v3-static-hard`
- 目的: hard局面に対して、全合法手searchedではなく teacher best + 静的上位候補をdepth5で読む方式が有効か確認する。

## 背景

hard200の全合法手searched方式は、teacher生成コストが高く、40局で 20-20 に留まった。

全816 hard局面を `teacher-score-source static / depth5 / top8` で生成する試行は15分超で未完了だったため中断した。反復可能な範囲として300件に絞った。

## 教師データ

```bash
target/release/distill_dump \
  --weights policy_weights_v2.1.0.binary \
  --input /tmp/shogi_soft_v3_hard/hard_unique.sfen \
  --train-output /tmp/shogi_soft_v3_static_hard300/train.jsonl \
  --valid-output /tmp/shogi_soft_v3_static_hard300/valid.jsonl \
  --depth 5 \
  --teacher-score-top 8 \
  --teacher-score-depth 5 \
  --teacher-score-source static \
  --max-positions 300 \
  --valid-percent 10 \
  --seed 12511 \
  --jobs 4
```

結果:

```text
train records: 269
valid records: 30
skipped positions: 1
```

gap分布:

```text
n=284
avg=18572.540
p25=0.000
p50=28.117
p75=159.836
p90=99171.036
p95=99471.566
max=200000.000
>=5=165
>=10=160
>=20=151
>=40=133
```

## 学習

条件:

```bash
target/release/distill_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/shogi_soft_v3_static_hard300/train.jsonl \
  --valid /tmp/shogi_soft_v3_static_hard300/valid.jsonl \
  --output /tmp/shogi_soft_v3_static_hard300/trained_lr100_e2_t100.binary \
  --epochs 2 \
  --batch-size 128 \
  --learning-rate 1.0 \
  --softmax-temperature 100 \
  --teacher-temperature 100 \
  --min-teacher-gap 0
```

結果:

```text
baseline train samples=269 ce=3.233119 top1=0.3717
baseline valid samples=30 ce=3.405010 top1=0.2667
epoch 1 train_ce=3.233005 train_top1=0.3903 valid_ce=3.405007 valid_top1=0.3000
epoch 2 train_ce=3.232887 train_top1=0.3941 valid_ce=3.405005 valid_top1=0.3000
```

valid top1 は 0.2667 から 0.3000 へ改善した。

full重みは使わず、blend 0.20 を作成して対局評価した。

```text
sha256 blend r0.20:
b4d06d4b0c0b4f7aee3fff20e821f177a04a811fa465f7b4b80f62a24f7cfbb6
```

## 40局スモーク

条件:

```text
new weights: /tmp/shogi_soft_v3_static_hard300/blend_lr100_e2_t100_r020.binary
baseline weights: policy_weights_v2.1.0.binary
engine: same current usi_engine
positions: taya36.sfen
games: 40
depth: 5
time-limit-ms: 100
max-plies: 200
adjudicate-at-max-plies
seed: 12531
```

結果:

```text
new wins: 20
baseline wins: 20
draws: 0
new total score rate: 50.00%
95% CI total: 34.50%..65.50%
```

## 判断

不採用。

offlineではvalid top1が改善したが、対局40局では50.0%に留まった。採用ゲートの40局52%以上を満たさない。

候補重み `/tmp/shogi_soft_v3_static_hard300/blend_lr100_e2_t100_r020.binary` は削除済み。

hard局面だけに寄せる蒸留は、現時点では現行重みを明確に超えない。次に重み更新を続けるなら、hard局面のみではなくランダム安定化局面を混ぜ、かつ対局ゲート前に「通常局面validが悪化していないか」を測る必要がある。
