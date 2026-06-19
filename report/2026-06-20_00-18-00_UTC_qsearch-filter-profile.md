# qsearch filter 内訳計測

- 日時: 2026-06-20 00:18 UTC
- ブランチ: `tooling/qsearch-filter-profile`
- 目的: qsearch が全合法手生成後に capture/check だけを残す現状で、どの程度の手を捨てているか確認する。

## 追加したもの

`ShogiAI` と `search_profile` に以下の計測値を追加した。

```text
quiescence moves generated
quiescence moves discarded
quiescence discard rate
```

探索候補、順序、SEE skip、評価値は変更していない。

## 検証

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin search_profile
```

結果: 通過。

## profile

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/search_profile \
  --weights policy_weights_v2.1.0.binary \
  --positions taya36.sfen \
  --samples 72 \
  --depth 5 \
  --seed 9501
```

結果:

```text
total nodes: 19067196
quiescence nodes: 17270535
quiescence moves considered: 8841983
quiescence moves generated: 138833345
quiescence moves discarded: 129991362
quiescence moves searched: 3264998
quiescence see skips: 3096541
check evasion extensions: 26959
aspiration researches: 0
quiescence discard rate: 93.63%
```

従来基準と nodes/qnodes/considered/searched/see skips/extensions は一致した。

## 判断

qsearch は探索ノードの約90.6%を占め、生成した合法手の約93.6%を capture/check フィルタで捨てている。したがって `legal_moves() -> retain(capture/check)` を同一候補集合の直接生成へ置き換えられれば、速度改善余地は大きい。

ただし王手手の同一生成は実装リスクが高い。次に進む場合は、まず `Position::legal_quiescence_moves` を追加し、旧方式との集合一致テストを広い局面セットで通してから `ai.rs` の qsearch を差し替える。
