# root前回実測スコア順序付けの棄却

- 作成日時: 2026-06-19 22:58:58 UTC
- ブランチ: `exp/root-score-ordering`
- 結論: 採用しない。root順序変更により探索ノードが増え、search profileが悪化した。

## 目的

反復深化のroot探索で、現状は前depthのbest moveだけを先頭へ移動している。

これを、前depthで実測したroot score順に並べ替えることで、次depthのroot探索効率を上げることを狙った。

## 実装概要

- 各depthでroot手と評価値を `root_scores_for_depth` に保存。
- aspiration failで再探索した場合はスコアをクリアして再計測。
- depth完了後、評価済みroot手をscore降順に並べ、未評価手は元順序を維持。

## 検証

変更前:

```text
samples: 72
total nodes: 19067196
quiescence nodes: 17270535
check evasion extensions: 26959
elapsed ms: 62230.49
nodes/sec: 306396.37
```

変更後:

```text
samples: 72
total nodes: 20890530
quiescence nodes: 18779976
check evasion extensions: 26250
elapsed ms: 69860.51
nodes/sec: 299032.04
```

探索ノードが約9.6%増え、elapsedも悪化した。

## 判断

採用しない。

理由:

- 前depthのroot score順は、現在のmove ordering / TT / aspirationとの相性が悪く、枝刈り効率を下げた。
- 100ms条件ではroot順序の悪化がそのまま探索量増につながる。
- 前回bestだけ先頭へ出す現行方式の方が安定している。

## 次の方針

root全体のスコア順並べ替えは再試行しない。root改善を行う場合は、全手順序を変えるのではなく、特定の tactical root move を限定的に補正する形にする。
