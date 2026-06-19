# 単一合法手ノードのソート省略 実験レポート

- 日時: 2026-06-19 14:53:56 UTC
- ブランチ: `perf/skip-single-move-sorting`
- 比較対象: `v2.4.3`
- 重み: `policy_weights_v2.1.0.binary`
- 結論: 不採用

## 仮説

合法手が1手しかないノードでは、指し手スコア計算、ソート、killer move並べ替え、TT move並べ替えは不要である。

境界王手応手延長によって低分岐ノードが増えているため、手数1ノードの処理を特殊化すれば探索木を変えずに速度改善できると考えた。

## 実装内容

- 内部alpha-betaで `legal_moves().len() == 1` の場合、スコア計算とソートを省略。
- rootでも同様に、初期合法手が1手だけならroot move orderingを省略。
- qsearchは初回実験では触らない。

触っていないもの:

- qsearch
- SEE
- 王手ボーナス
- TT probe/store
- 境界王手応手延長条件
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
elapsed ms: 28374.46
elapsed ms: 28202.64
elapsed ms: 28129.99
median: 28202.64
```

中央値でbaselineより速くない。速度ゲートを満たさない。

## 判断

探索木は不変だったが、速度改善は得られなかったため不採用。対局ベンチには進めない。

合法手1手ノードで省けるorderingコストは、このprofile条件では全体に対して小さい。qsearchが全ノードの約90%を占めるため、内部alpha-beta側だけを特殊化しても効果が出にくい。
