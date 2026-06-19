# 境界王手応手延長 <=4 実験メモ

- 作成日時: 2026-06-19 14:10:00 UTC
- 実験ブランチ: `exp/boundary-check-evasion-replies4`
- 結論: 棄却。速度ゲートは通過したが、v2.4.2直接比較40局で負け越した。

## 実装内容

v2.4.2で採用した境界王手応手延長の条件を、合法応手数 `<=3` から `<=4` へ広げた。

変更点:

```text
CHECK_EVASION_EXTENSION_MAX_REPLIES: 3 -> 4
```

拡張予算は1回のまま。`depth == 1` 条件、通常探索で1手読む方針、qsearch/SEE/評価関数/重みは変更していない。reply count 4で発火した回数を `search_profile` に追加した。

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
total nodes: 8635731
quiescence nodes: 7795904
check evasion extensions: 25078
check evasion reply4 extensions: 13430
elapsed ms: 27905.70
```

ノード数と速度は問題なかったが、発火数はv2.4.2比で2倍強に増えた。

## v2.4.2直接比較

条件:

```text
positions: taya36.sfen
depth: 5
time-limit-ms: 100
max-plies: 200
jobs: 4
seed: 5301
```

40局:

```text
new wins: 19
baseline wins: 21
draws: 0
new decisive win rate: 47.50%
new total score rate: 47.50%
paired starts:
  new sweeps: 4
  baseline sweeps: 5
  splits: 11
  draw/mixed pairs: 0
```

## 判断

採用しない。`<=4` は探索量の面では許容範囲だったが、強さではv2.4.2を下回った。

今後は `<=5` 以上、延長予算2、qsearch内王手応手生成へは進めない。v2.4.2の `<=3` 条件を現時点の安定点として扱う。
