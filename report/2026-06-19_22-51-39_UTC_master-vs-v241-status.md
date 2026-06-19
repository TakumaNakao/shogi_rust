# master vs v2.4.1 現状確認

- 作成日時: 2026-06-19 22:51:39 UTC
- 対象: `master` vs `v2.4.1`
- 結論: v2.4.1には明確に優勢だが、目標95%にはまだ遠い。

## 目的

KPP評価内の冗長境界チェック削除を採用した後、現在のmasterがv2.4.1に対してどの程度優勢かを再確認した。

## 設定

```text
new-engine: /home/nami_ride_trade/shogi_rust/target/release/usi_engine
baseline-engine: /tmp/shogi_rust_v241_compare/target/release/usi_engine
weights: 両方 policy_weights_v2.1.0.binary
positions: taya36.sfen
games: 40
depth: 5
time-limit-ms: 100
max-plies: 200
adjudicate-at-max-plies
jobs: 4
seed: 10101
record-dir: /tmp/shogi_bench_master_vs_v241_40_seed10101
```

## 結果

```text
new wins: 28
baseline wins: 10
draws: 2
new decisive win rate: 73.68%
new total score rate: 72.50%
decisive win rate 95% CI: 57.99%..85.03%
total score rate 95% CI: 59.10%..85.90%
```

終局理由:

```text
MaxPliesAdjudication: 1
RepetitionDraw: 2
Resign: 37
```

paired starts:

```text
new sweeps: 9
baseline sweeps: 0
splits: 10
draw/mixed pairs: 1
```

## 所見

baseline sweepsが0であり、現行masterはv2.4.1に対して安定した優位を維持している。

一方、40局でtotal 72.5%なので、目標の95%には大きな差がある。短期的な小改善を積み上げるだけでは到達しにくく、次のどちらかが必要になる可能性が高い。

1. 敗局tailを狙った具体的な探索改善。
2. より大きな評価関数更新。ただし、直近のvalue regression小規模候補は100局で53%に縮み不採用。

## 次の方針

GPT-5.5 xhighへ、直近の棄却実験とこの40局結果を踏まえ、次に試すべき小さな探索改善候補の再選定を依頼した。
