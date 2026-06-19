# 探索側ArrayVecソート 実験レポート

- 日時: 2026-06-19 14:44:38 UTC
- ブランチ: `perf/search-inplace-move-sort`
- 比較対象: `v2.4.3`
- 重み: `policy_weights_v2.1.0.binary`
- 結論: 不採用

## 仮説

探索中の指し手順序付けでは、合法手を `ArrayVec` で生成した後、`Vec<(Move, score)>` を作ってスコア順に並べ、さらに `Vec<Move>` に戻している。

このheap `Vec` 生成を避ければ、探索木を変えずに速度改善できると考えた。

## 実装内容

試した内容:

1. `sort_unstable_by_key` で `ArrayVec<Move, 593>` を直接ソート。
2. 直接ソートではスコア計算が比較中に何度も呼ばれて約59秒まで悪化したため中止。
3. スコアを1回だけ計算する形に戻し、`Vec<(Move, score)>` と `Vec<Move>` を `ArrayVec` に置換。

変更対象:

- `src/ai.rs` の qsearch 指し手順序付け
- `src/ai.rs` の内部 alpha-beta 指し手順序付け
- `src/ai.rs` のroot指し手順序付け

触っていないもの:

- qsearch候補条件
- SEE skip
- 王手ボーナス
- TT
- 境界王手応手延長
- 評価関数
- movegen本体

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
elapsed ms: 28736.58
elapsed ms: 28895.98
elapsed ms: 29010.06
median: 28895.98
```

中央値で約3.16%遅い。速度ゲートである「中央値2%以上改善、悪化ランなし」を満たさない。

## 判断

探索木は不変だったが、速度は改善しなかった。heap allocation削減よりも、`ArrayVec` の大きな値の扱い、スタック使用、コード生成、collect処理のコストが勝った可能性がある。

この実験は不採用。対局ベンチには進めない。

今後この系統を再試行するなら、ソート用バッファを `ShogiAI` 内に再利用可能なワークスペースとして持つ、または小分岐ノード専用に挿入ソートや手数1の特殊化を行う方が現実的。
