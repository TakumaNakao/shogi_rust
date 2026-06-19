# qsearch終端詰み検出の高速合法応手判定つき再実験

- 作成日時: 2026-06-19 17:59:31 UTC
- ブランチ: `exp/qsearch-terminal-fast-evasion`
- 目的: 以前に速度悪化で棄却した qsearch 終端詰み検出を、専用の高速合法応手判定で再評価する。

## 背景

以前の `qsearch terminal mate` 実験では、qsearch中に王手かつ合法手ゼロの局面を `-inf` として扱うことで終端整合性は改善したが、王手中qsearchノードで毎回 `legal_moves()` を先行生成したため、profileが約 `+9.07%` 悪化して棄却した。

今回の再実験では `Position::has_legal_evasion()` を追加した。王手中に合法応手が1つでも見つかれば生成を打ち切るため、終端判定専用に使える。

## 実装

- `shogi_lib::Position::has_legal_evasion()` を追加。
- `quiescence_search` 冒頭で、`position.in_check() && !position.has_legal_evasion()` の場合だけ `-inf` を返す。
- `search_profile` に `quiescence terminal mates` カウンタを追加。
- qsearchの候補範囲、SEE skip、通常探索、評価関数、重みは変更していない。

## 固定検証セット

GPT-5.5 xhigh サブエージェントの提案に従い、探索変更の前処理ゲートとして以下を追加した。

```text
data/search_quality/loss_in_check_low_reply.sfen
data/search_quality/taildrop_root_rescue.sfen
```

`loss_in_check_low_reply.sfen` は score/result mismatch から抽出した、王手中・低応手の終局近傍局面を中心にした固定セットである。

## 検証

テスト:

```text
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

結果: pass。

固定セットprobe:

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/position_probe \
  --weights policy_weights_v2.1.0.binary \
  --positions data/search_quality/loss_in_check_low_reply.sfen \
  --depth 5 \
  --summary
```

結果:

```text
total: 27
in_check: 27
low_legal_in_check: 21
terminal: 4
search_loss: 26
legal_without_bestmove: 0
```

profile条件:

```text
target/release/search_profile \
  --weights policy_weights_v2.1.0.binary \
  --positions taya36.sfen \
  --samples 36 \
  --depth 5 \
  --seed 9501
```

最新master:

```text
total nodes: 8638168
quiescence nodes: 7802474
elapsed ms: 27925.43
```

本実験:

```text
total nodes: 8627638
quiescence nodes: 7792404
quiescence terminal mates: 380
elapsed ms: 28384.63
```

速度悪化は約 `+1.65%`。以前の `+9.07%` より大きく改善し、事前ゲート内に収まった。

72 samples profile:

```text
total nodes: 19067196
quiescence nodes: 17270535
quiescence terminal mates: 424
elapsed ms: 62513.28
```

## 対局ベンチ

比較対象: 最新master `de8965e`

条件:

```text
depth 5
time-limit-ms 100
max-plies 200
adjudicate-at-max-plies
jobs 4
weights: policy_weights_v2.1.0.binary
positions: taya36.sfen
```

20局 seed 10101:

```text
new wins: 11
baseline wins: 7
draws: 2
new total score rate: 60.00%
paired starts:
  new sweeps: 2
  baseline sweeps: 0
  splits: 7
  draw/mixed pairs: 1
```

40局 seed 10121:

```text
new wins: 27
baseline wins: 13
draws: 0
new total score rate: 67.50%
paired starts:
  new sweeps: 9
  baseline sweeps: 2
  splits: 9
```

100局 seed 10141:

```text
new wins: 53
baseline wins: 45
draws: 2
new total score rate: 54.00%
paired starts:
  new sweeps: 9
  baseline sweeps: 6
  splits: 33
  draw/mixed pairs: 2
```

100局 seed 10161:

```text
new wins: 59
baseline wins: 40
draws: 1
new total score rate: 59.50%
paired starts:
  new sweeps: 14
  baseline sweeps: 5
  splits: 30
  draw/mixed pairs: 1
```

200局合算:

```text
new wins: 112
baseline wins: 85
draws: 3
new total score rate: 56.75%
decisive win rate: 56.85%
paired starts:
  new sweeps: 23
  baseline sweeps: 11
```

## 判断

採用する。

単独100局 seed 10141 は `54.00%` で採用ラインに1局届かなかったが、別seedの100局で `59.50%`、200局合算で `56.75%` となり、paired starts でも new sweeps が優勢だった。

大幅改善ではないが、以下の理由で採用価値がある。

- 以前の棄却理由だった速度悪化を専用APIで解消した。
- qsearch中の終端詰みを静的評価で流す不整合を減らす。
- 対最新masterで200局合算が採用ラインを超えた。

今後の探索変更は、追加した固定セットを事前ゲートとして使い、20局だけの良化で採用しない。
