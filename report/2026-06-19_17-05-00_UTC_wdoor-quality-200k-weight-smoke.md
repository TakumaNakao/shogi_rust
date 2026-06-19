# wdoor高レート勝者200k重み更新スモーク結果

- 作成日時: 2026-06-19 17:05 UTC
- ブランチ: `exp/wdoor-quality-200k-weight-smoke`
- 目的: `csa_policy_dump` の品質フィルタで作った200k教師データが、通常wdoor教師より重み更新に有効か確認する。

## データ生成

条件:

```bash
target/release/csa_policy_dump \
  --input /tmp/shogi_external_kifu/wdoor/extract/2026 \
  --train-output /tmp/shogi_external_kifu/datasets_quality/wdoor2026_policy_train_200k_r4000_winner.jsonl \
  --valid-output /tmp/shogi_external_kifu/datasets_quality/wdoor2026_policy_valid_200k_r4000_winner.jsonl \
  --seed 20260619 \
  --valid-percent 10 \
  --max-records 200000 \
  --min-ply 8 \
  --max-ply 160 \
  --min-player-rate 4000 \
  --winner-only \
  --decisive-only
```

結果:

```text
games used: 3263
games skipped: 88
games filtered: 49948
records filtered: 4730269
train records: 180000
valid records: 20000
```

## 学習

条件:

```bash
target/release/distill_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/shogi_external_kifu/datasets_quality/wdoor2026_policy_train_200k_r4000_winner.jsonl \
  --valid /tmp/shogi_external_kifu/datasets_quality/wdoor2026_policy_valid_200k_r4000_winner.jsonl \
  --output /tmp/shogi_wdoor_quality_200k_lr005/policy_weights_wdoor2026_200k_r4000_winner_lr005_t100_e1.binary \
  --epochs 1 \
  --batch-size 512 \
  --learning-rate 0.05 \
  --softmax-temperature 100
```

結果:

```text
baseline train samples=180000 ce=3.991430 top1=0.2243
baseline valid samples=20000 ce=4.007719 top1=0.2206

epoch 1 train_ce=3.991217 train_top1=0.2303 valid_ce=4.007457 valid_top1=0.2269
material_coeff=0.145648
```

差分:

```text
valid CE:   4.007719 -> 4.007457  改善 +0.000262
valid top1: 0.2206   -> 0.2269    改善 +0.0063
```

通常wdoor 200kではvalid top1が悪化したが、高レート勝者200kではvalid top1が改善した。オフライン上は、品質フィルタの方向性は通常全手学習より良い。

## 20局スモーク

条件:

```text
new weights: /tmp/shogi_wdoor_quality_200k_lr005/policy_weights_wdoor2026_200k_r4000_winner_lr005_t100_e1.binary
baseline weights: policy_weights_v2.1.0.binary
engine: same current usi_engine
positions: taya36.sfen
games: 20
depth: 5
time-limit-ms: 100
seed: 9821
```

結果:

```text
new wins: 13
baseline wins: 7
draws: 0
new decisive win rate: 65.00%
new total score rate: 65.00%
95% CI total: 44.10%..85.90%

paired starts:
  new sweeps: 4
  baseline sweeps: 1
  splits: 5
  draw/mixed pairs: 0

record_analyze:
  non-terminal score/result sign mismatches: 1
```

20局では有望だったが、CIが広いため採用不可。40局へ進めた。

## 40局スモーク

条件:

```text
games: 40
seed: 9841
record-dir: /tmp/shogi_bench_wdoor_quality200k_9841_40
```

結果:

```text
new wins: 18
baseline wins: 21
draws: 1
new decisive win rate: 46.15%
new total score rate: 46.25%
95% CI total: 30.99%..61.51%

end reasons:
  MaxPliesAdjudication: 2
  RepetitionDraw: 1
  Resign: 37

paired starts:
  new sweeps: 3
  baseline sweeps: 4
  splits: 12
  draw/mixed pairs: 1

record_analyze:
  non-terminal score/result sign mismatches: 6
```

## 判断

高レート勝者フィルタは通常wdoor全手学習よりオフライン指標を改善したが、40局では悪化した。

結論:

```text
policy_weights_wdoor2026_200k_r4000_winner_lr005_t100_e1.binary は不採用。
重み更新は引き続き慎重扱い。
```

示唆:

- 外部棋譜の品質フィルタは有効そうだが、policy-onlyの単純CE更新では強さに直結しない。
- 20局の13-7はノイズだった可能性が高い。
- 次に重みを触るなら、手一致率だけでなく探索後の勝率ゲートを前提に、より小さい学習率または正則化を使うべき。
- 当面は探索・速度改善へ戻る。重み更新は「品質フィルタ + より保守的な学習率 + 40局以上ゲート」の形でのみ再試行する。
