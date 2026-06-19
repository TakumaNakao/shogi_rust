# null-move pruning 実験結果

## 目的

`v2.4.1` に対して、探索ノード削減による実戦強度向上を狙って null-move pruning を検証した。

この候補は GPT-5.5 xhigh による探索方針レビューで、qsearch delta pruning の次候補として挙げられた。ただし将棋では zugzwang 的な危険や終盤の誤枝刈りリスクがあるため、高リスク pruning 候補として扱った。

## 実装

実験ブランチ:

```text
exp/null-move-pruning
```

主要コミット:

```text
64f6eeb Experiment with null move pruning
24c34f6 Track null move pruning by depth
```

実装内容:

- `Position::do_null_move` / `undo_null_move` を追加
- null move の do/undo 復元テストを追加
- alpha-beta 本体に `allow_null` を渡し、連続 null move を禁止
- `search_profile` に null move attempt / cutoff と depth 別集計を追加

最終的に40局ベンチへ進めた条件:

```text
depth >= 3
margin = 0
R = 2
alpha.is_finite()
beta.is_finite()
!position.in_check()
sennichite_detector.get_position_count(position) <= 1
recursive allow_null = false
```

## プロファイル

固定 depth 5:

```text
positions: taya36.sfen
samples: 36
seed: 9501
```

`v2.4.1` baseline:

```text
total nodes: 8,511,421
elapsed: 27,851.75 ms
nodes/sec: 305,597.31
```

null-move pruning:

```text
total nodes: 4,893,526
elapsed: 16,591.62 ms
nodes/sec: 294,939.65
null move attempts: 7,118
null move cutoffs: 2,169
null move cutoff rate: 30.47%
```

depth 別:

```text
depth 3: attempts 5,332 / cutoffs 983 / rate 18.44%
depth 4: attempts 1,786 / cutoffs 1,186 / rate 66.41%
```

ノード数と経過時間は大きく改善した。一方、depth 3 での attempt が多く、浅い探索での誤枝刈りリスクが高い形だった。

## 対局ベンチ

共通条件:

```text
baseline: v2.4.1
weights: policy_weights_v2.1.0.binary
positions: taya36.sfen
depth: 5
time-limit-ms: 100
max-plies: 200
adjudicate-at-max-plies
```

20局 smoke:

```text
seed: 5301
NewWin: 12
BaselineWin: 8
Draw: 0
total score rate: 60.00%
new as black: 6-4
new as white: 6-4
paired starts:
  new sweeps: 4
  baseline sweeps: 2
  splits: 4
```

40局:

```text
seed: 5401
NewWin: 18
BaselineWin: 21
Draw: 1
total score rate: 46.25%
95% CI total: 30.99%..61.51%
decisive win rate: 46.15%
end reasons:
  Resign: 39
  RepetitionDraw: 1
```

先後:

```text
new as black: 6-13-1, total score rate 32.5%
new as white: 12-8-0, total score rate 60.0%
```

paired starts:

```text
new sweeps: 2
baseline sweeps: 4
splits: 13
draw/mixed: 1
```

`record_analyze` 要約:

```text
non-terminal score/result sign mismatches: 3
largest tail drops:
  game_020: 324
  game_008: 323
  game_004: 320
  game_025: 295
  game_013: 290
```

## 判断

null-move pruning は採用しない。

理由:

- 40局で `46.25%` と `v2.4.1` に負け越した。
- paired starts で baseline sweep が new sweep を上回った。
- new黒番が `32.5%` と大きく崩れた。
- depth 3 の発火が多く、浅い探索での誤枝刈りリスクが強く出た可能性がある。
- 速度改善は大きいが、実戦強度へ変換できなかった。

20局では `12-8` と良く見えたが、40局で再現しなかった。qsearch delta pruning と同様に、短いベンチだけで pruning 系変更を採用しない方針を再確認した。

## 次の方針

null-move の追加変種を深追いしない。

候補としては `depth >= 4, margin = 0` や `depth == 3` だけ margin を戻す案もあるが、現時点では優先度を下げる。直近の分析では、敗局・mismatch が終盤の王手中かつ合法手少数の局面に集中しているため、次は以下を優先する。

- `record_analyze` で抽出した mismatch / tail drop SFEN を固定検証セット化する
- 終盤・王手周辺の非終端 mismatch を分類する
- pruning 追加ではなく、終端判定・王手周辺の探索/評価の不整合を低リスクに潰す

関連して、`record_analyze` には以下の検証基盤改善を `master` に入れた。

```text
66e4613 Export analyzed benchmark positions
84532a7 Separate terminal records in analysis
```
