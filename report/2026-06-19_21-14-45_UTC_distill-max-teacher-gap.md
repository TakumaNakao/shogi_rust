# distill_train max teacher gap と gap制御mixed smoke

- 作成日時: 2026-06-19 21:14:45 UTC
- ブランチ: `tooling/distill-max-teacher-gap`
- 目的: soft蒸留データに含まれる詰み級・大差gap局面を抑制し、通常局面を壊しにくい学習ゲートを作る。

## 実装

`distill_train` に `--max-teacher-gap` を追加した。

既存の `--min-teacher-gap` は、探索上位1位と2位の差が小さい局面を除外する。今回の `--max-teacher-gap` は逆に、差が大きすぎる局面を除外する。

狙い:

```text
詰み級・大差局面だけに学習が引っ張られることを防ぐ。
random valid と hard valid の分離評価をしやすくする。
```

不正な指定はエラーにする。

```text
--max-teacher-gap < 0: error
--max-teacher-gap < --min-teacher-gap: error
```

## 検証

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release \
  --bin distill_train \
  --bin adjust_weights \
  --bin usi_benchmark \
  --bin record_analyze \
  --bin usi_engine
```

結果:

```text
cargo test --all-targets: pass
release build: pass
```

## gap制御mixed smoke

直前のmixed smokeは100局で49.0%まで落ちた。そこで同じ混合データに `--max-teacher-gap 1000` を適用し、巨大gap局面を除外した。

条件:

```bash
target/release/distill_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/shogi_soft_v3_mixed_gapcap/train.jsonl \
  --valid /tmp/shogi_soft_v3_mixed_gapcap/valid.jsonl \
  --output /tmp/shogi_soft_v3_mixed_gapcap/trained_gapcap1000_lr100_e1_t100.binary \
  --epochs 1 \
  --batch-size 128 \
  --learning-rate 1.0 \
  --softmax-temperature 100 \
  --teacher-temperature 100 \
  --min-teacher-gap 0 \
  --max-teacher-gap 1000
```

結果:

```text
baseline train samples=674 ce=3.516245 top1=0.2092
baseline valid samples=73 ce=3.459810 top1=0.1918
epoch 1 train_ce=3.516016 train_top1=0.2166 valid_ce=3.459623 valid_top1=0.2192
```

分離dry-run:

```text
random valid:
  samples=50 ce=3.656278 top1=0.1600

hard valid:
  samples=23 ce=3.032115 top1=0.3478
```

full重みは使わず、blend 0.20 を作成した。

```text
sha256:
8c940fa0c7468bc137190d1ea3391cb680e1972affea3510e866626768972ea4
```

## 40局スモーク

条件:

```text
new weights: /tmp/shogi_soft_v3_mixed_gapcap/blend_gapcap1000_lr100_e1_t100_r020.binary
baseline weights: policy_weights_v2.1.0.binary
engine: same current usi_engine
positions: taya36.sfen
games: 40
depth: 5
time-limit-ms: 100
max-plies: 200
adjudicate-at-max-plies
seed: 12641
```

結果:

```text
new wins: 20
baseline wins: 19
draws: 1
new total score rate: 51.25%
end reasons:
  RepetitionDraw: 1
  Resign: 39
paired starts:
  new sweeps: 3
  baseline sweeps: 2
  splits: 14
  draw/mixed pairs: 1
average final score for new: -22.0
terminal final positions: 39
terminal result mismatches: 0
non-terminal score/result sign mismatches: 0
```

## 判断

候補重みは不採用。

40局ゲートの52.5%に届かず、平均最終評価も負だった。候補重み `/tmp/shogi_soft_v3_mixed_gapcap/blend_gapcap1000_lr100_e1_t100_r020.binary` は削除済み。

一方で `--max-teacher-gap` は、教師データ基盤として残す価値がある。今後の蒸留では巨大gap局面の比率を制御し、random/hard validを分離して監視する。
