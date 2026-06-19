# forced-evasion static fold 実験結果

## 目的

終盤の `final_in_check=true` かつ合法手少数の mismatch が多いため、GPT-5.5 xhigh の提案に基づき「王手中・合法手少数局面の静的評価折り畳み」を試した。

狙いは、王手を受けている局面そのものをKPP raw評価せず、合法応手後の最善静的評価へ1手だけ折り畳むことで、終盤の静的評価ノイズを減らすことだった。

## 実装

実験ブランチ:

```text
exp/forced-evasion-static-fold
```

コミット:

```text
7ade248 Experiment with forced evasion static fold
```

条件:

```text
if position.in_check():
  legal_moves == 0 -> -1_000_000
  legal_moves <= 3 -> max(-raw_eval(child))
  otherwise -> raw_eval(position)
```

`SparseModelEvaluator` と `search_profile` 用評価器を検索用評価へ切り替え、`position_probe` では raw static と search static を両方表示した。

## 事前検証

`qdelta` 100局から抽出した mismatch 上位10件に `position_probe` を当てた。

観測:

- mismatch 上位10件はすべて `in_check=true`
- 合法手数は `0..3`
- 全件で `search_score=-inf`
- `legal_moves=1..3` の局面でも `search_static_eval` は raw static からほぼ動かなかった

例:

```text
static_eval=510.0 search_static_eval=510.2 search_score=-inf
static_eval=430.6 search_static_eval=430.3 search_score=-inf
static_eval=365.0 search_static_eval=365.7 search_score=-inf
```

つまり、今回の1手静的折り畳みは、問題局面の評価符号やmarginを実質的に改善しなかった。

## プロファイル

条件:

```text
positions: taya36.sfen
samples: 36
depth: 5
seed: 9501
```

結果:

```text
total nodes: 8,560,737
qsearch nodes: 7,736,068
elapsed: 30,785.10 ms
nodes/sec: 278,080.53
```

以前の同条件 baseline は約 `8,511,421` nodes / `27,851.75 ms` だったため、ノードはほぼ同程度で経過時間は悪化した。

## 判断

採用しない。40局ベンチへ進めない。

理由:

- GPT-5.5が提示した足切り条件「mismatch局面で有効評価がraw staticから十分動くこと」を満たさなかった。
- 合法手生成を評価中に呼ぶため、目的に刺さらないまま速度だけ悪化した。
- mismatch上位は探索上すでに `-inf` の必敗局面であり、1手のraw静的折り畳みでは不十分だった。

## 次の方針

今回の結果から、mismatch上位は「静的評価が悪い」というより「探索上は必敗だが、raw static分類が追いつかない局面」として扱うべき可能性が高い。

次は探索本体を変える前に、`position_probe` と exported SFEN を使って以下を分類する。

- `search_score=-inf` の必敗分類
- `search_score=inf` の勝ち筋分類
- raw static と search score が大きく乖離する局面
- tail drop 上位のうち、探索PVが強制王手/詰み筋を示す局面

pruning追加や王手周辺の雑な補正は引き続き避ける。
