# TinyNNUE depth4 rank学習候補棄却

- 作成日時: 2026-06-20 01:26:27 UTC
- ブランチ: `experiment/nnue-rank-depth4`
- 目的: `nnue_rank_dump` で作った同一root内子局面rankingデータを使い、対局ゲートへ進める候補が作れるか確認する。

## データ生成

```bash
target/release/nnue_rank_dump \
  --input taya36.sfen \
  --output /tmp/nnue_rank_taya36_d4_top8.jsonl \
  --weights policy_weights_v2.1.0.binary \
  --depth 4 \
  --max-positions 36 \
  --top-k 8 \
  --jobs 4
```

結果:

```text
roots: 36
records: 288
avg records/root: 8.00
real 0m13.979s
file size: 153K
```

分割:

```text
train: 28 roots / 224 records
valid: 8 roots / 64 records
```

## H=64

```text
initial rank valid roots=8 top1=12.50% selected_regret=8.25 bad_selected=0
best epoch by valid RMSE: 7
valid_rmse=11.07
valid_mae=7.56
valid_sign=54.69%
valid_top1=12.50%
valid_sel_regret=7.80
```

途中で最も良かったrank指標:

```text
epoch 012 valid_top1=37.50% valid_sel_regret=7.35
```

## H=96

```text
initial rank valid roots=8 top1=12.50% selected_regret=7.91 bad_selected=0
best epoch by valid RMSE: 10
valid_rmse=11.29
valid_mae=7.12
valid_sign=54.69%
valid_top1=25.00%
valid_sel_regret=4.62
```

途中を含めてもvalid top1は25%止まりだった。

## 判断

H=64/H=96候補は棄却する。rank gateが弱く、20局対局へ進める根拠がない。

今回の結果から、単純なvalue MSEだけでrankデータを学習しても、同一root内の手順序を安定して学べないことが分かった。次は以下のどちらかが必要:

1. root単位のpairwise/listwise ranking lossを学習器に入れる。
2. データを36 rootsから増やし、敗局tailや外部低regret局面を混ぜる。

現時点では候補重みをリリース・タグ対象にしない。
