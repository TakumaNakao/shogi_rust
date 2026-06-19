# soft探索蒸留 v3 mixed gap80 smoke

- 作成日時: 2026-06-19 21:29:45 UTC
- ブランチ: `exp/soft-distill-v3-mixed-gap80`
- 目的: mixedデータから `gap >= 80` の大差局面を除外し、通常局面を壊しにくい蒸留候補になるか確認する。

## 背景

直前のmixed候補は、40局で52.5%だったが100局で49.0%まで落ち、不採用になった。

`distill_stats` で確認した mixed データのgap分布:

```text
records: 799
with_gap: 784
gap_p50: 0.558
gap_p75: 12.887
gap_p90: 144.024
gap_p95: 98959.680
gap_ge_80: 102 (13.01%)
gap_ge_1000: 52 (6.63%)
gap_ge_10000: 52 (6.63%)
```

今回は `--max-teacher-gap 80` で、p90以上の大差局面を落として学習した。

## 学習

条件:

```bash
target/release/distill_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/shogi_soft_v3_mixed_gap80/train.jsonl \
  --valid /tmp/shogi_soft_v3_mixed_gap80/valid.jsonl \
  --extra-valid random=/tmp/shogi_soft_v2_d4_top8_500/valid.jsonl \
  --extra-valid hard=/tmp/shogi_soft_v3_static_hard300/valid.jsonl \
  --output /tmp/shogi_soft_v3_mixed_gap80/trained_gap80_lr100_e1_t100.binary \
  --epochs 1 \
  --batch-size 128 \
  --learning-rate 1.0 \
  --softmax-temperature 100 \
  --teacher-temperature 100 \
  --min-teacher-gap 0 \
  --max-teacher-gap 80
```

結果:

```text
baseline train samples=630 ce=3.524539 top1=0.1921
baseline valid samples=67 ce=3.458897 top1=0.1642
baseline extra_valid[random] samples=50 ce=3.656548 top1=0.1400
baseline extra_valid[hard] samples=17 ce=2.877572 top1=0.2353

epoch 1 train_ce=3.524318 train_top1=0.2016 valid_ce=3.458699 valid_top1=0.1940
epoch 1 extra_valid[random] samples=50 ce=3.656285 top1=0.1600
epoch 1 extra_valid[hard] samples=17 ce=2.877563 top1=0.2941
```

offlineでは mixed / random / hard のすべてで top1 が改善した。

full重みは使わず、blend 0.20 を作成した。

```text
sha256:
9a3b084d8a089b54e8fa9df387c8d9b9e494148bd5be79a11f0f5405a88bd3d1
```

## 40局スモーク

条件:

```text
new weights: /tmp/shogi_soft_v3_mixed_gap80/blend_gap80_lr100_e1_t100_r020.binary
baseline weights: policy_weights_v2.1.0.binary
engine: same current usi_engine
positions: taya36.sfen
games: 40
depth: 5
time-limit-ms: 100
max-plies: 200
adjudicate-at-max-plies
seed: 12701
```

結果:

```text
new wins: 19
baseline wins: 20
draws: 1
new total score rate: 48.75%
end reasons:
  MaxPliesAdjudication: 2
  RepetitionDraw: 1
  Resign: 37
paired starts:
  new sweeps: 4
  baseline sweeps: 4
  splits: 11
  draw/mixed pairs: 1
average final score for new: 45.5
terminal final positions: 37
terminal result mismatches: 0
non-terminal score/result sign mismatches: 0
```

## 判断

不採用。

offlineでは分離validがすべて改善したが、対局40局では48.75%に落ちた。paired startsも完全に中立で、採用ゲートを満たさない。

候補重み `/tmp/shogi_soft_v3_mixed_gap80/blend_gap80_lr100_e1_t100_r020.binary` は削除済み。

示唆:

- offline top1改善は、現行のKPP policy蒸留では対局勝率を十分に予測できない。
- `gap >= 80` を落としても改善しなかったため、問題は巨大gap局面だけではない。
- 次に重み更新を続けるなら、policy手一致ではなく、評価値回帰・探索勝敗・局面価値を含む教師に寄せる必要がある。
