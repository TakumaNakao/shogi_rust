# value regression候補のregret gate検証

- 作成日時: 2026-06-19 22:09:43 UTC
- ブランチ: `exp/value-regret-gate-smoke`
- 結論: 小規模mixed value regression候補は不採用。100局で優位が53%まで縮んだ。

## 目的

`value_regret_probe` が、評価値回帰候補を対局前に選別するgateとして使えるか確認した。

対象候補は、前回のdepth 4 value regressionスモークでhard validを改善した `mixed lr500 / 5epoch` 相当の重み。

## データ

通常局面:

```text
taya36.sfen から300局面
train records: 240
valid records: 60
```

hard局面:

```text
/tmp/shogi_bench_records_qdelta_vs_v241_5201_100 の BaselineWin tail
train records: 320
valid records: 80
```

mixed:

```text
train records: 560
valid records: 140
```

## offline指標

baseline:

```text
mixed valid huber=0.669987 rmse_cp=1054.89 mae_cp=493.58 sign_acc=0.9600 corr=0.7604
random valid huber=0.000239 rmse_cp=13.12 mae_cp=6.26 corr=0.4473
hard valid huber=1.172298 rmse_cp=1395.44 mae_cp=859.06 sign_acc=0.9600 corr=0.7613
```

candidate `lr500 / 5epoch`:

```text
mixed valid huber=0.659440 rmse_cp=1043.69 mae_cp=486.13 sign_acc=0.9733 corr=0.7626
random valid huber=0.000279 rmse_cp=14.18 mae_cp=9.70 corr=0.4303
hard valid huber=1.153810 rmse_cp=1380.62 mae_cp=843.46 sign_acc=0.9733 corr=0.7631
```

hardとmixedは改善したが、random validは悪化した。

## regret gate

同一重み同士:

```text
samples: 140
mean_regret_cp: 0.00
max_regret_cp: 0.00
bad_regret_count_gt_300: 0 (0.00%)
teacher_move_match: 140 (100.00%)
```

候補重み:

```text
samples: 140
mean_regret_cp: 1.01
p50_regret_cp: 0.00
p90_regret_cp: 3.05
p95_regret_cp: 5.44
max_regret_cp: 14.64
bad_regret_count_gt_300: 0 (0.00%)
teacher_move_match: 93 (66.43%)
```

regret上は大きな悪手は出ていない。したがって、この候補はregret gateだけでは落ちない。

## 対局ベンチ

同一エンジンで重みだけを比較した。

設定:

```text
depth 5
time-limit-ms 100
max-plies 200
adjudicate-at-max-plies
positions taya36.sfen
```

20局:

```text
new wins: 11
baseline wins: 8
draws: 1
new total score rate: 57.50%
```

40局:

```text
new wins: 24
baseline wins: 16
draws: 0
new total score rate: 60.00%
paired starts:
  new sweeps: 8
  baseline sweeps: 4
  splits: 8
```

100局:

```text
new wins: 52
baseline wins: 46
draws: 2
new decisive win rate: 53.06%
new total score rate: 53.00%
decisive win rate 95% CI: 43.25%..62.64%
total score rate 95% CI: 43.32%..62.68%
paired starts:
  new sweeps: 14
  baseline sweeps: 11
  splits: 24
  draw/mixed pairs: 1
```

40局では良く見えたが、100局では優位がほぼ消えた。採用しない。

## 判断

今回のvalue regression候補は不採用。

理由:

- 100局で53%に縮み、採用基準に届かない。
- regretは低いが、対局優位に変換できていない。
- random validの悪化があり、評価全体をhard tailへ寄せすぎている可能性が高い。

## 次の方針

value regressionを続ける場合は、以下が必要。

1. 通常局面データを最低1000局面以上に増やす。
2. hard tailだけでなく、`worst_drop` 周辺の中間局面を混ぜる。
3. regret gateは「巨大悪手の除外」には有効だが、採用判定には対局ベンチが必要。
4. offline random valid悪化を許容しない設定を優先する。

短期の強さ改善としては、重み更新だけではなく、再び探索・基礎高速化の候補へ戻る価値が高い。
