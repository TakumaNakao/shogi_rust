# 王手応手延長閾値4の棄却

- 作成日時: 2026-06-19 22:23:51 UTC
- ブランチ: `exp/check-evasion-extension-4`
- 結論: 採用しない。profile上のコストは小さいが、40局で悪化した。

## 目的

採用済みの低分岐王手応手延長を少し広げ、終盤の読み抜けを減らす。

変更内容:

```rust
CHECK_EVASION_EXTENSION_MAX_REPLIES: usize = 3 -> 4
```

## profile

変更前:

```text
samples: 72
total nodes: 19067196
quiescence nodes: 17270535
check evasion extensions: 26959
elapsed ms: 64146.45
nodes/sec: 297244.76
```

変更後:

```text
samples: 72
total nodes: 19277889
quiescence nodes: 17450228
check evasion extensions: 60739
elapsed ms: 64398.92
nodes/sec: 299351.13
```

ノード増は約1.1%、elapsed増は約0.4%で、profile上は許容範囲だった。

## 対局ゲート

比較対象:

- new: `CHECK_EVASION_EXTENSION_MAX_REPLIES = 4`
- baseline: master `CHECK_EVASION_EXTENSION_MAX_REPLIES = 3`
- weights: 両方 `policy_weights_v2.1.0.binary`
- positions: `taya36.sfen`
- depth 5 / time-limit-ms 100 / max-plies 200

20局:

```text
new wins: 11
baseline wins: 9
draws: 0
new total score rate: 55.00%
```

40局:

```text
new wins: 19
baseline wins: 20
draws: 1
new total score rate: 48.75%
```

## 判断

採用しない。

理由:

- 20局ではわずかに良く見えたが、40局で互角未満に落ちた。
- 延長回数が2倍以上に増えた割に、勝率へつながらなかった。
- 王手応手拡張は現行閾値3が局所最適に近い可能性がある。

## 次の方針

王手応手拡張の閾値拡大は再試行しない。次は、戦略変更ではなく以下を優先する。

1. 探索挙動を変えないprofile主導の高速化。
2. qsearchの大きな設計変更ではなく、評価関数呼び出しや局面生成の局所改善。
3. 必要なら、敗局tailから具体的な詰み逃し・王手回避ミスを抽出してから個別パッチを作る。
