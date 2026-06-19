# 合法手あり全敗局面の bestmove fallback 修正

- 作成日時: 2026-06-19 17:26:16 UTC
- ブランチ: `fix/all-losing-bestmove`
- 対象: `find_best_move` が合法手あり局面で `None` を返す問題

## 背景

GPT-5.5 xhigh サブエージェントの指摘により、全候補が `-inf` と評価される局面で `bestmove none` になり得ることを確認した。

再現用局面:

```text
/tmp/qdelta_mismatch_positions_10.sfen
```

修正前の `position_probe --depth 5 --summary` では、10局面中8局面が合法手ありにもかかわらず `bestmove=none` だった。

```text
legal_without_bestmove: 8
terminal: 2
```

## 修正内容

探索内部の `alpha_beta_search` は変更せず、rootの `find_best_move` だけを保守的に修正した。

反復深化のある深さで全候補が `-inf` となり `current_best_move_for_depth` が残らない場合、既存の `best_move` を `None` で上書きしない。初期値としては従来どおりrootの並び順先頭合法手を保持するため、合法手がある限りUSIで `bestmove resign` を返しにくくなる。

## 検証

固定probe:

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/position_probe \
  --weights policy_weights_v2.1.0.binary \
  --positions /tmp/qdelta_mismatch_positions_10.sfen \
  --depth 5 \
  --summary
```

結果:

```text
legal_without_bestmove: 0
terminal: 2
```

通常局面profile:

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/search_profile \
  --weights policy_weights_v2.1.0.binary \
  --positions taya36.sfen \
  --samples 72 \
  --depth 5 \
  --seed 9501
```

masterとの比較で探索木は一致した。

```text
total nodes: 19079358
quiescence nodes: 17282476
check evasion extensions: 26968
```

テスト:

```text
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

結果: 10 tests passed。

## 対局ベンチ

現masterとの同一重み比較。

20局 seed 10061:

```text
new wins: 13
baseline wins: 7
draws: 0
new total score rate: 65.00%
```

40局 seed 10081:

```text
new wins: 20
baseline wins: 20
draws: 0
new total score rate: 50.00%
paired starts:
  new sweeps: 3
  baseline sweeps: 3
  splits: 14
  draw/mixed pairs: 0
```

## 判断

強さ改善としては中立で、v2.4.1に対する勝率95%へ直接近づく変更ではない。

一方、合法手がある局面で `bestmove resign` を返す可能性を下げるUSI安全性修正として価値がある。探索内部の `-inf` PV処理や有限mate score化は過去に悪化したため今回は再導入しない。
