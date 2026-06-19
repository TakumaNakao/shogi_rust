# soft探索蒸留 v3 mixed smoke

- 作成日時: 2026-06-19 21:08:28 UTC
- ブランチ: `exp/soft-distill-v3-mixed-smoke`
- 目的: hard-only蒸留が40局50%に留まったため、random searched局面とhard static局面を混合し、通常局面を壊さずhard局面も改善できるか確認する。

## データ

既存のJSONLを再利用した。

```text
random searched:
  /tmp/shogi_soft_v2_d4_top8_500/train.jsonl
  /tmp/shogi_soft_v2_d4_top8_500/valid.jsonl

hard static:
  /tmp/shogi_soft_v3_static_hard300/train.jsonl
  /tmp/shogi_soft_v3_static_hard300/valid.jsonl
```

混合後:

```text
train records: 719
valid records: 80
```

## 学習

条件:

```bash
target/release/distill_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/shogi_soft_v3_mixed/train.jsonl \
  --valid /tmp/shogi_soft_v3_mixed/valid.jsonl \
  --output /tmp/shogi_soft_v3_mixed/trained_lr100_e2_t100.binary \
  --epochs 2 \
  --batch-size 128 \
  --learning-rate 1.0 \
  --softmax-temperature 100 \
  --teacher-temperature 100 \
  --min-teacher-gap 0
```

結果:

```text
baseline train samples=719 ce=3.509760 top1=0.2184
baseline valid samples=80 ce=3.562222 top1=0.1875
epoch 1 train_ce=3.509540 train_top1=0.2295 valid_ce=3.562053 valid_top1=0.2125
epoch 2 train_ce=3.509322 train_top1=0.2392 valid_ce=3.561886 valid_top1=0.2125
```

分離評価:

```text
random valid:
  trained ce=3.656018 top1=0.1400

hard valid:
  trained ce=3.404999 top1=0.3333
```

random側は悪化せず、hard側は改善した。

full重みは使わず、blend 0.20 を作成した。

```text
sha256:
699b90c6c5a525f4e46e811e828d20d42dcbddece9c4e9a35e7b4fddf1197e70
```

## 40局スモーク

条件:

```text
new weights: /tmp/shogi_soft_v3_mixed/blend_lr100_e2_t100_r020.binary
baseline weights: policy_weights_v2.1.0.binary
engine: same current usi_engine
positions: taya36.sfen
games: 40
depth: 5
time-limit-ms: 100
max-plies: 200
adjudicate-at-max-plies
seed: 12601
```

結果:

```text
new wins: 21
baseline wins: 19
draws: 0
new total score rate: 52.50%
end reasons:
  MaxPliesAdjudication: 1
  PerpetualCheckLoss: 1
  Resign: 38
paired starts:
  new sweeps: 4
  baseline sweeps: 3
  splits: 13
  draw/mixed pairs: 0
average final score for new: 124.7
```

40局ゲートは最小限通過したため、100局へ進めた。

## 100局ゲート

条件:

```text
games: 100
seed: 12621
record-dir: /tmp/shogi_bench_softv3_mixed_r020_12621_100
```

結果:

```text
new wins: 49
baseline wins: 51
draws: 0
new total score rate: 49.00%
95% CI total: 39.20%..58.80%
end reasons:
  MaxPliesAdjudication: 4
  Resign: 96
paired starts:
  new sweeps: 5
  baseline sweeps: 6
  splits: 39
  draw/mixed pairs: 0
average final score for new: -14.5
terminal final positions: 96
terminal result mismatches: 0
non-terminal score/result sign mismatches: 0
```

## 判断

不採用。

40局では52.5%だったが、100局では49.0%まで落ちた。paired startsでも baseline sweeps が上回り、平均最終評価も負になった。

候補重み `/tmp/shogi_soft_v3_mixed/blend_lr100_e2_t100_r020.binary` は削除済み。

hard-only、static hard、mixedのいずれも対局ゲートを通らなかった。次は重み更新の実行そのものではなく、教師データ基盤を見直す。特に以下が必要:

- random valid と hard valid を常に分離評価する。
- 詰み級・大差gap局面の比率を制御する。
- 1反復5-6分以内で教師生成できる固定データ設計にする。
- offline top1改善だけで採用しない。
