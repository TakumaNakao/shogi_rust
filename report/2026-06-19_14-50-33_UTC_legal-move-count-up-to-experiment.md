# legal_moves_count_up_to 実験レポート

- 日時: 2026-06-19 14:50:33 UTC
- ブランチ: `perf/legal-move-count-up-to`
- 比較対象: `v2.4.3`
- 重み: `policy_weights_v2.1.0.binary`
- 結論: 不採用

## 仮説

境界王手応手延長では、depth境界で「指した手が王手になり、相手の合法応手が3手以下か」を調べるために `position.legal_moves().len()` を呼んでいる。

この判定では正確な全合法手数は不要で、4手以上あると分かった時点で延長しないと判断できる。そのため `legal_moves_count_up_to(4)` を追加すれば、探索木を変えずに速度改善できると考えた。

## 実装内容

- `shogi_lib::Position` に `legal_moves_count_up_to(limit)` を追加。
- 生成された疑似合法手を `is_legal` で数え、合法手数が `limit` に達した時点で打ち切る。
- 境界王手応手延長の判定を以下に変更。

```rust
position.legal_moves_count_up_to(CHECK_EVASION_EXTENSION_MAX_REPLIES + 1)
    <= CHECK_EVASION_EXTENSION_MAX_REPLIES
```

追加テスト:

- 初期局面で `count_up_to(4) == 4`
- 最大合法手局面で `count_up_to(4) == 4`
- 単応手局面で `count_up_to(4) == 1`
- `usize::MAX` 指定では `legal_moves().len()` と一致

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

`v2.4.3` baseline と完全一致したため、探索木は変わっていない。

## 速度比較

`v2.4.3` baseline 直近測定:

```text
elapsed ms: 28153.12
elapsed ms: 27864.50
elapsed ms: 28011.56
median: 28011.56
```

実験版:

```text
elapsed ms: 28181.18
elapsed ms: 28200.28
elapsed ms: 46012.33
median: 28200.28
```

3回目は外れ値の可能性があるが、中央値でもbaselineより速くない。2%以上改善の速度ゲートを満たさない。

## 判断

探索木は不変だったが、速度改善は得られなかったため不採用。対局ベンチには進めない。

今回の実装は、疑似合法手の生成後に `is_legal` の途中打ち切りを行うだけで、`generate_evasions` 自体は全候補を生成する。そのため、境界王手応手延長判定の主要コストが疑似合法手生成側にある場合は効果が出にくい。

この方向を再試行するなら、`generate_evasions` 自体を上限付きで止めるAPIが必要。ただしmovegen本体の変更はバグリスクが高いため、perftと専用テストを厚くしてから行うべき。
