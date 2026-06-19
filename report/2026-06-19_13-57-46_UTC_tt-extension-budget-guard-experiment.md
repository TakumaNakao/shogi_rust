# TT延長予算guard 実験メモ

- 作成日時: 2026-06-19 13:57:46 UTC
- 実験ブランチ: `exp/tt-extension-budget-guard`
- 結論: 棄却。速度ゲートは通過したが、v2.4.2直接比較40局で互角止まり。

## 実装内容

v2.4.2で導入した `check_evasion_extension_budget` をTT entryへ保存し、TTのscore/bound再利用条件を以下に変更した。

```text
entry.generation == current_generation
entry.depth >= depth
entry.extension_budget >= check_evasion_extension_budget
```

TT best moveはordering用途として従来通り使った。qsearch、SEE、評価関数、重み、mate score、千日手判定には触れていない。

## 速度確認

条件:

```bash
target/release/search_profile \
  --weights policy_weights_v2.1.0.binary \
  --positions taya36.sfen \
  --samples 36 \
  --depth 5 \
  --seed 9501
```

v2.4.2基準:

```text
total nodes: 8638168
check evasion extensions: 11652
elapsed ms: 約28700
```

実験結果:

```text
total nodes: 8627918
quiescence nodes: 7792208
check evasion extensions: 11474
tt extension budget rejects: 1377
elapsed ms: 28725.70
```

TT budget rejectは発生しており、速度・ノード増加の問題はなかった。

## v2.4.2直接比較

条件:

```text
positions: taya36.sfen
depth: 5
time-limit-ms: 100
max-plies: 200
jobs: 4
seed: 5201
```

40局:

```text
new wins: 20
baseline wins: 20
draws: 0
new decisive win rate: 50.00%
new total score rate: 50.00%
paired starts:
  new sweeps: 2
  baseline sweeps: 2
  splits: 16
  draw/mixed pairs: 0
```

## 判断

採用しない。TT再利用の正当性としては自然だが、少なくともこの条件・seedでは強さの改善が見えなかった。

今後この方向を再試行する場合は、単純なbudget guardではなく、TT entryの「延長済みかどうか」や境界王手局面専用の統計を追加して、実際にどの局面で悪影響を避けられるかを先に可視化する。
