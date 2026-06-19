# 単応手チェーン延長 実験レポート

- 日時: 2026-06-19 14:27:49 UTC
- ブランチ: `exp/single-reply-chain-extension`
- 比較対象: `v2.4.3`
- 重み: `policy_weights_v2.1.0.binary`
- 結論: 不採用

## 仮説

`v2.4.2` で採用した境界王手応手延長は、depth境界で「指した手が王手になり、相手の合法応手が3手以下」の場合だけ1手延長する。

今回の仮説は、その延長で入った応手側ノードが「王手を受けていて合法応手が1手だけ」の場合に限り、さらに1回だけ延長すれば、終盤の強制線を低コストで拾えるというものだった。

## 実装内容

- `CHECK_EVASION_EXTENSION_MAX_REPLIES=3` は変更しない。
- 既存の境界王手応手延長とは別に `single_reply_chain_budget=1` を追加。
- `depth == 1`、既存延長予算使用済み、局面が王手、合法応手が1手だけの場合に、その1手の先をさらに1 ply探索する。
- `search_profile` に `single reply chain extensions` カウンタを追加。

## 速度ゲート

コマンド:

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/search_profile \
  --weights policy_weights_v2.1.0.binary \
  --positions taya36.sfen \
  --samples 36 \
  --depth 5 \
  --seed 9501
```

結果:

```text
samples: 36
total nodes: 8663106
quiescence nodes: 7826355
quiescence moves considered: 4276997
quiescence moves searched: 1627326
quiescence see skips: 1475675
check evasion extensions: 11554
single reply chain extensions: 1165
elapsed ms: 28261.47
nodes/sec: 306534.17
```

`v2.4.3` 系の基準は total nodes 8,638,168、elapsed 約28.1秒だったため、ノード増は約0.29%、時間はほぼ横ばい。速度ゲートは通過した。

## 対局ゲート

コマンド:

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/usi_benchmark \
  --new-engine /home/nami_ride_trade/shogi_rust/target/release/usi_engine \
  --baseline-engine /tmp/shogi_rust_v243_compare/target/release/usi_engine \
  --new-weights /home/nami_ride_trade/shogi_rust/policy_weights_v2.1.0.binary \
  --baseline-weights /home/nami_ride_trade/shogi_rust/policy_weights_v2.1.0.binary \
  --positions /home/nami_ride_trade/shogi_rust/taya36.sfen \
  --games 40 \
  --depth 5 \
  --time-limit-ms 100 \
  --max-plies 200 \
  --adjudicate-at-max-plies \
  --jobs 4 \
  --seed 5401 \
  --record-dir /tmp/shogi_bench_single_reply_chain_5401
```

結果:

```text
new wins: 15
baseline wins: 25
draws: 0
new decisive win rate: 37.50%
new total score rate: 37.50%
decisive win rate 95% CI: 24.22%..52.97%
total score rate 95% CI: 22.50%..52.50%
```

`record_analyze` 要約:

```text
end reasons:
  Resign: 40
paired starts:
  new sweeps: 1
  baseline sweeps: 6
  splits: 13
  draw/mixed pairs: 0
average final score for new: -89.2
average final score for NewWin: 452.2
average final score for BaselineWin: -414.1
terminal final positions: 0
terminal result mismatches: 0
non-terminal score/result sign mismatches: 3
```

## 判断

速度コストは非常に小さかったが、40局の直接比較で明確に悪化した。paired starts でも baseline sweeps が new sweeps を大きく上回っており、採用候補には残さない。

この結果から、境界王手応手延長の追加拡張は、単に強制線を深く読むだけでは改善にならない可能性が高い。今後この方向を再試行する場合は、詰み・必至・評価急落などの明確な局面条件でさらに絞る必要がある。
