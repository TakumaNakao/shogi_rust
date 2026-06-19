# soft探索蒸留 v2 ツール追加

- 作成日時: 2026-06-19 19:05:07 UTC
- ブランチ: `exp/soft-distill-v2-searched`
- 目的: 1000局面 / depth3 / 静的上位候補のsoft蒸留が40局で不採用になったため、教師候補の選び方と候補重みの出し方を改善する。

## 背景

前回のsoft蒸留smokeでは、offline valid top1は 0.07 から 0.16 に上がったが、同一エンジン・重み差分のみの40局で 18-21-1、総合46.25%に落ちた。

主な問題は以下。

- `teacher_scores` の候補が、教師bestmove以外は現行モデルの静的上位手だった。
- depth3教師が浅く、評価関数へ戻す信号として弱い可能性が高い。
- full更新をそのまま対局に出しており、更新が強すぎる可能性がある。
- 探索上位1位と2位が僅差の局面も学習していた。

## 追加内容

### `distill_dump --teacher-score-source searched`

`teacher_score_source` を追加した。

```text
static   : 従来通り、教師bestmove + 現行モデル静的上位手
searched : 全合法手を探索評価し、探索スコア上位N手を teacher_scores に保存
```

`searched` は重いが、静的評価上位ではなく探索上位の分布を教師にできる。

### `distill_train --min-teacher-gap`

`teacher_scores` の探索上位1位と2位の差が指定値未満なら、その局面を学習から除外する。

狙い:

- 僅差で教師分布が曖昧な局面を避ける。
- 小規模学習でノイズを拾いすぎる問題を抑える。

### `adjust_weights --blend-target --blend-ratio`

候補重みをそのまま対局に出さず、以下の保守的blendを作れるようにした。

```text
output = input + blend_ratio * (blend_target - input)
```

例:

```bash
target/release/adjust_weights \
  --input policy_weights_v2.1.0.binary \
  --blend-target /tmp/trained_candidate.binary \
  --blend-ratio 0.25 \
  --output /tmp/blended_025.binary
```

## smoke確認

実行:

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin distill_dump --bin distill_train --bin adjust_weights
```

searched dump:

```bash
target/release/distill_dump \
  --weights policy_weights_v2.1.0.binary \
  --input converted_records2016_10818.sfen \
  --train-output /tmp/shogi_soft_v2_smoke/train.jsonl \
  --valid-output /tmp/shogi_soft_v2_smoke/valid.jsonl \
  --depth 3 \
  --teacher-score-top 4 \
  --teacher-score-depth 3 \
  --teacher-score-source searched \
  --max-positions 20 \
  --valid-percent 20 \
  --seed 12101 \
  --jobs 4
```

結果:

```text
train records: 16
valid records: 4
```

`min-teacher-gap 0` のdry-run:

```text
baseline train samples=16 ce=3.425538 top1=0.0625
baseline valid samples=4 ce=3.592565 top1=0.2500
```

小さなdepth3 smokeでは `min-teacher-gap 80` だと全局面が除外された。これはdepth3の探索スコア差が小さいためで、本番候補では局面数を増やし、depth4で確認する。

blend smoke:

```bash
target/release/adjust_weights \
  --input policy_weights_v2.1.0.binary \
  --blend-target policy_weights_v2.1.0.binary \
  --blend-ratio 0.25 \
  --output /tmp/shogi_soft_v2_smoke_blend.binary
```

同一重み同士のblendとして正常に保存できることを確認し、出力ファイルは削除した。

## 次の実験案

重いため最初は2000局面ではなく、まずは小さめの探索教師で挙動を見る。

候補:

```bash
target/release/distill_dump \
  --weights policy_weights_v2.1.0.binary \
  --input converted_records2016_10818.sfen \
  --train-output /tmp/shogi_soft_v2_d4_top8/train.jsonl \
  --valid-output /tmp/shogi_soft_v2_d4_top8/valid.jsonl \
  --depth 4 \
  --teacher-score-top 8 \
  --teacher-score-depth 4 \
  --teacher-score-source searched \
  --max-positions 500 \
  --valid-percent 10 \
  --seed 12101 \
  --jobs 4
```

学習:

```bash
target/release/distill_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/shogi_soft_v2_d4_top8/train.jsonl \
  --valid /tmp/shogi_soft_v2_d4_top8/valid.jsonl \
  --output /tmp/shogi_soft_v2_d4_top8/trained.binary \
  --epochs 2 \
  --batch-size 128 \
  --learning-rate 0.2 \
  --softmax-temperature 100 \
  --teacher-temperature 100 \
  --min-teacher-gap 80
```

その後、`blend_ratio=0.25` の候補から20局smokeに進める。

採用条件は前回と同じく、20局で50%未満なら破棄、40局で55%未満なら不採用。
