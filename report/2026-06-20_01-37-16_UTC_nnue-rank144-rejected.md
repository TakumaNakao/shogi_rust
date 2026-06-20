# TinyNNUE rank144 depth4候補棄却

- 作成日時: 2026-06-20 01:37:16 UTC
- ブランチ: `experiment/nnue-rank-depth4-144`
- 目的: rankデータを36 rootsから144 rootsへ増やし、rank checkpoint基準でTinyNNUE候補を作る。

## データ生成

```bash
target/release/nnue_rank_dump \
  --input taya36.sfen \
  --output /tmp/nnue_rank_taya144_d4_top8.jsonl \
  --weights policy_weights_v2.1.0.binary \
  --depth 4 \
  --max-positions 144 \
  --top-k 8 \
  --jobs 4
```

結果:

```text
roots: 144
records: 1152
avg records/root: 8.00
real 1m7.657s
file size: 612K
```

分割:

```text
train: 128 roots / 1024 records
valid: 16 roots / 128 records
```

## H=64

条件:

```text
hidden: 64
epochs: 30
lr: 0.0015
rank-loss-weight: 0.001
rank-temperature-cp: 50
checkpoint-metric: valid_rank_selected_regret
```

結果:

```text
best epoch: 16
valid_rmse=37.82
valid_mae=19.09
valid_sign=58.59%
valid_top1=100.00%
valid_sel_regret=0.00
```

root regret 64局面:

```text
mean_regret_cp: 43.55
p50_regret_cp: 4.28
p90_regret_cp: 232.58
p95_regret_cp: 233.28
max_regret_cp: 233.66
teacher_move_match: 11 (17.19%)
```

## H=96

条件:

```text
hidden: 96
epochs: 30
lr: 0.0015
rank-loss-weight: 0.001
rank-temperature-cp: 50
checkpoint-metric: valid_rank_selected_regret
```

結果:

```text
best epoch: 15
valid_rmse=38.20
valid_mae=24.60
valid_sign=58.59%
valid_top1=100.00%
valid_sel_regret=0.00
```

root regret 64局面:

```text
mean_regret_cp: 27.14
p50_regret_cp: 2.73
p90_regret_cp: 87.93
p95_regret_cp: 154.95
max_regret_cp: 236.20
teacher_move_match: 12 (18.75%)
```

## 10局スモーク

H=96のみ実施:

```text
new wins: 0
baseline wins: 10
draws: 0
new decisive win rate: 0.00%
new total score rate: 0.00%
```

## 判断

H=64/H=96候補は棄却する。rank validでは良く見えても、root regretがまだ重く、H=96の10局スモークで0-10と明確に崩壊した。

学習基盤としては前進しているが、現状のTinyNNUEは「手順序の局所模倣」はできても、探索全体を支える評価スケール・終盤評価・静止探索末端評価が弱い。

次の方針:

1. TinyNNUE単体でKPPを置き換える前に、KPP + TinyNNUE補正のハイブリッドを試す。
2. 候補手順位だけでなく、既存KPP評価からの小さな残差を学習させる。
3. 対局ゲート前に、KPPとの評価相関と探索NPSを同時に見る。

候補重みは削除し、リリース対象にしない。
