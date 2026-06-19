# v2.1.0 baseline 100局ベンチ結果

作成日時: 2026-06-19 00:04:20 UTC  
対象ブランチ: `improve-self-play-learning`  
対象HEAD: `33272aa`  
baseline: `v2.1.0` worktree `/tmp/shogi_rust_v210_bench`

## 目的

現行HEADが `v2.1.0` baseline に対して、100局以上の標準ベンチで明確に勝ち越すか確認した。

## 条件

- new engine: `/home/nami_ride_trade/shogi_rust/target/release/usi_engine`
- baseline engine: `/tmp/shogi_rust_v210_bench/target/release/usi_engine`
- weights: 両方 `/home/nami_ride_trade/shogi_rust/policy_weights_v2.1.0.binary`
- positions: `/home/nami_ride_trade/shogi_rust/taya36.sfen`
- games: `100`
- depth: `5`
- time-limit-ms: `100`
- max-plies: `200`
- adjudicate-at-max-plies: enabled
- jobs: `4`
- seed: `2401`
- record-dir: `/tmp/shogi_bench_records_2401`

## 結果

```text
NewWin: 77
BaselineWin: 20
Draw: 3

new decisive win rate: 79.38%
decisive win rate 95% CI: 70.29%..86.24%

new total score rate: 78.50%
total score rate 95% CI: 70.63%..86.37%
```

終局理由:

```text
Resign: 97
RepetitionDraw: 3
```

paired starts:

```text
new sweeps: 30
baseline sweeps: 1
splits: 16
draw/mixed pairs: 3
```

先後:

```text
new as black: 35-13-2
  decisive win rate: 72.92%
  total score rate: 72.00%

new as white: 42-7-1
  decisive win rate: 85.71%
  total score rate: 85.00%
```

`record_analyze --tail-plies 12` の要約:

```text
average final score for new: 372.8
average final score for NewWin: 562.8
average final score for BaselineWin: -382.1
score/result sign mismatches: 10
```

## 判断

P0採用目安の `100局で total score rate >= 70%` を満たした。95% CI の下限も `70.63%` で、`v2.1.0` baseline に対して現行HEADは明確に優勢と判断できる。

baseline sweep は1件のみで、特定開始局面への極端な弱さは限定的だった。一方で black/white の偏りはあり、今回条件では new 白番が特に強く、new 黒番は相対的に低い。

## 敗局傾向

`BaselineWin` は20局。終盤 `tail_scores` の `worst_drop` は中央値約145、75パーセンタイル約163、最大316だった。20局中9局で150点以上の急落があり、安定して押し切られるより、終盤数手で評価が崩れる敗局が目立つ。

大きな急落例:

```text
game_042: worst_drop=316
game_025: worst_drop=298
game_053: worst_drop=233
game_011: worst_drop=168
game_007: worst_drop=163
```

`BaselineWin` なのに最終評価が new 側に正の局:

```text
game_007 final_score_for_new=259.8
game_052 final_score_for_new=44.2
game_084 final_score_for_new=320.7
```

これらは詰み逃し、王手回避ミス、または終盤の危険度を評価関数が見落としている可能性がある。

## 次の優先作業

短期の探索改善として、過去に悪化した qsearch 全王手拡張や check extension の再試行は避ける。まずは限定的で検証しやすい以下を候補にする。

1. root での即詰み・即詰まされ回避の軽量確認
2. 王手回避局面に限定した合法手orderingの改善
3. BaselineWin棋譜の具体的な敗着前後を分類する補助出力の追加

採用条件は従来通り、20局で悪化しないこと、できれば40局で改善傾向、最終採用は100局または複数seedで確認とする。
