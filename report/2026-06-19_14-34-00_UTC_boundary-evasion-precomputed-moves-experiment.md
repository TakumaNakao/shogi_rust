# 境界王手応手ノードの合法手再利用 実験レポート

- 日時: 2026-06-19 14:34:00 UTC
- ブランチ: `perf/boundary-evasion-precomputed-moves`
- 比較対象: `v2.4.3`
- 重み: `policy_weights_v2.1.0.binary`
- 結論: 不採用

## 仮説

`v2.4.2` で採用した境界王手応手延長では、親ノード側で「王手がかかった子局面の合法応手数」を調べるために `position.legal_moves()` を呼び、その後、延長先の子ノードで再び同じ合法手を生成する。

この重複生成を避けるため、親ノードで生成済みの合法応手を子ノードへ渡せば、探索木を変えずに速度だけ改善できると考えた。

## 実装内容

- `alpha_beta_search_internal` に `precomputed_moves: Option<ArrayVec<Move, 593>>` を追加。
- 通常ノードでは従来通り `position.legal_moves()` を呼ぶ。
- 境界王手応手延長が発火する場合のみ、親ノードで生成した応手一覧を子ノードへ渡す。
- 探索順序、枝刈り、評価関数、置換表ポリシーは変更しない。

## 探索木不変チェック

コマンド:

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/search_profile \
  --weights policy_weights_v2.1.0.binary \
  --positions taya36.sfen \
  --samples 36 \
  --depth 5 \
  --seed 9501
```

実験版:

```text
total nodes: 8638168
quiescence nodes: 7802474
quiescence moves considered: 4271670
quiescence moves searched: 1625940
quiescence see skips: 1475088
check evasion extensions: 11652
```

`v2.4.3` baseline とノード数、qsearchノード、qsearch手数、SEE skip、延長回数は一致した。探索木は不変と判断できる。

## 速度比較

同一条件で3回測定した。

`v2.4.3` baseline:

```text
elapsed ms: 28153.12
elapsed ms: 27864.50
elapsed ms: 28011.56
median: 28011.56
```

実験版:

```text
elapsed ms: 29572.32
elapsed ms: 28136.93
elapsed ms: 28526.08
median: 28526.08
```

中央値では実験版が約1.84%遅い。ゲート条件だった「中央値で2%以上改善、悪化ランなし」を満たさない。

## 判断

探索木は変えずに済んだが、速度改善は得られなかった。低分岐応手の再生成コストよりも、`Option<ArrayVec>` の受け渡し、分岐、コード形状変化による最適化阻害の方が大きかった可能性がある。

この実験は不採用。対局ベンチには進めない。

今後この方向を再試行するなら、探索関数シグネチャを変えずに、合法応手数だけを早期打ち切りで数える `legal_moves_up_to(limit)` のような生成器側APIを作り、親側の完全生成そのものを避ける方が筋が良い。
