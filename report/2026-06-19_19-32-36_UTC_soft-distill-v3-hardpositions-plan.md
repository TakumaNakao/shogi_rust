# soft探索蒸留 v3 hard局面計画

- 作成日時: 2026-06-19 19:32:36 UTC
- ブランチ: `exp/soft-distill-v3-hardpositions`
- 目的: soft探索蒸留 v1/v2 の失敗を踏まえ、ランダム局面ではなく v2.4.1 比較で実際に弱点化した局面を中心に重み更新を試す。

## 背景

soft蒸留 v1 は `converted_records2016` のランダム1000局面、depth3/static top4で生成し、40局で 18-21-1 まで悪化した。

soft蒸留 v2 は searched teacher depth4/top8、500局面、blend 0.25に改善し、40局では 23-17 だったが、100局では 52-45-3、total 53.5% で採用基準を下回った。

このため、単にランダム局面を増やすよりも、現行エンジンが v2.4.1 相手に落とした局面、評価急落局面、score/result mismatch局面を優先して教師データ化する。

## 採用する方針

GPT-5.5 xhigh の判断により、当面の比重は以下とする。

```text
重み・棋譜/蒸留: 80%
探索・高速化: 20%
```

探索側は既に多くの候補を棄却しており、現行約71%から v2.4.1 比95%へ進めるには評価関数側の改善が必要と判断した。ただし、自己対局学習は過去に評価関数を壊しているため、固定局面と対局ゲートを前提にする。

## hard局面ソース

`record_analyze` に以下のエクスポートを追加した。

```text
--export-baseline-win-tails
--baseline-win-tail-plies
--export-baseline-sweep-starts
```

既存の以下も併用する。

```text
--export-drops
--export-mismatches
```

200局の現行master vs v2.4.1比較を実行し、以下を抽出する。

```text
/tmp/shogi_hard_v241_12301_200/drops.sfen
/tmp/shogi_hard_v241_12301_200/mismatches.sfen
/tmp/shogi_hard_v241_12301_200/baseline_win_tails.sfen
/tmp/shogi_hard_v241_12301_200/baseline_sweep_starts.sfen
```

実行結果:

```text
seed 12301 / 200 games
new wins: 143
baseline wins: 53
draws: 4
new total score rate: 72.50%
decisive win rate: 72.96%
end reasons:
  MaxPliesAdjudication: 7
  RepetitionDraw: 4
  Resign: 189
paired starts:
  new sweeps: 49
  baseline sweeps: 6
  splits: 41
  draw/mixed pairs: 4
average final score for new: 30.1
average final score for NewWin: 360.4
average final score for BaselineWin: -866.9
terminal final positions: 57
terminal result mismatches: 0
non-terminal score/result sign mismatches: 21
```

抽出件数:

```text
baseline_sweep_starts.sfen: 6
baseline_win_tails.sfen: 643
drops.sfen: 199
mismatches.sfen: 21
total: 869
```

## 蒸留候補

最初の候補は、hard局面を中心にした小さな実験に限定する。

```text
teacher source: searched
teacher depth: hard局面 depth 5、安定化局面 depth 4
teacher top: 8 または 12
min_teacher_gap: 0 / 5 / 10
learning rate: 0.05 / 0.1
epochs: 1-2
material: freeze
blend ratio: 0.10 / 0.20 / 0.25
```

full重みは対局候補にしない。必ずblend後の重みだけを評価する。

## 採否ゲート

1. offline:
   - hard valid CE/top1 が改善する。
   - random valid が大きく悪化しない。

2. same-engine weight比較:
   - 40局で total 52%以上。
   - 可能なら paired baseline sweeps が増えない。
   - 通過候補のみ100局へ進める。

3. 100局:
   - 現行固定重み比 total 55%以上を最低条件。
   - 56%以上なら v2.4.1比較へ進める。

4. v2.4.1比較:
   - 現行約71%から最低 +5pt、つまり76%以上を中間ゲートにする。
   - 95%は最終目標だが、ここでは一段階の明確な改善を採用条件にする。

## 容量運用

候補重みは `/tmp` に置き、棄却時点で削除する。

基準重み `policy_weights_v2.1.0.binary` は現行検証に必要なため削除しない。

## hard200 smoke

全816 hard局面に `searched depth 5` で教師スコアを付ける試行は20分超で未完了だったため中断した。

全816 hard局面に `searched depth 4` で教師スコアを付ける試行も10分超で未完了だったため中断した。

まず現実的な反復単位として、hard局面200件の `searched depth 4 / top8` を生成した。

```bash
target/release/distill_dump \
  --weights policy_weights_v2.1.0.binary \
  --input /tmp/shogi_soft_v3_hard/hard_unique.sfen \
  --train-output /tmp/shogi_soft_v3_hard200/train.jsonl \
  --valid-output /tmp/shogi_soft_v3_hard200/valid.jsonl \
  --depth 4 \
  --teacher-score-top 8 \
  --teacher-score-depth 4 \
  --teacher-score-source searched \
  --max-positions 200 \
  --valid-percent 10 \
  --seed 12421 \
  --jobs 4
```

結果:

```text
train records: 179
valid records: 20
skipped positions: 1
```

gap分布:

```text
n=184
avg=11909.438
p25=0.000
p50=6.479
p75=79.991
p90=98784.071
p95=99396.084
max=100509.944
>=5=95
>=10=87
>=20=80
>=40=63
```

詰み・大差局面由来の巨大gapが多く、通常局面と二極化している。

### 学習

lr 0.05 / 0.1, temperature 200, epoch 1 はoffline指標がほぼ動かなかった。

強め条件として lr 1.0, temperature 100, epoch 2 を試した。

```text
baseline train samples=179 ce=3.520616 top1=0.3855
baseline valid samples=20 ce=3.397170 top1=0.3000
epoch 1 train_ce=3.520559 train_top1=0.3911 valid_ce=3.397165 valid_top1=0.3000
epoch 2 train_ce=3.520500 train_top1=0.3966 valid_ce=3.397160 valid_top1=0.3000
```

full重みは使わず、blend 0.10 / 0.20 を作成した。40局スモークは、より効果が出やすい blend 0.20 のみ実施した。

### 40局スモーク

条件:

```text
new weights: /tmp/shogi_soft_v3_hard200/blend_lr100_e2_t100_r020.binary
baseline weights: policy_weights_v2.1.0.binary
engine: same current usi_engine
positions: taya36.sfen
games: 40
depth: 5
time-limit-ms: 100
max-plies: 200
adjudicate-at-max-plies
seed: 12441
```

結果:

```text
new wins: 20
baseline wins: 20
draws: 0
new total score rate: 50.00%
end reasons:
  Resign: 40
paired starts:
  new sweeps: 2
  baseline sweeps: 2
  splits: 16
  draw/mixed pairs: 0
average final score for new: 32.9
terminal final positions: 40
terminal result mismatches: 0
non-terminal score/result sign mismatches: 0
```

## hard200 smoke判断

不採用。

hard200単体のsoft蒸留は、現行固定重み比で40局50.0%に留まった。採用ゲートの40局52%以上を満たさず、paired startsも完全に中立だった。

候補重み `blend_lr100_e2_t100_r010.binary` と `blend_lr100_e2_t100_r020.binary` は削除済み。

次に蒸留を続けるなら、hard局面だけでなくランダム安定化局面を混ぜるか、`teacher_score_source searched` の全合法手探索をやめて teacher best + 静的上位候補だけを深く読む方式に戻し、計算量を抑えて局面数を増やす必要がある。
