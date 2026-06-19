# qsearch delta pruning 実験結果

## 目的

`v2.4.1` に対する次の探索改善候補として、GPT-5.5 xhigh の提案に基づき qsearch delta pruning を検証した。

狙いは、明らかに alpha を上回れない静かな捕獲を qsearch 内で枝刈りし、同一時間内の探索量を増やすことだった。

## 実装条件

実験ブランチ: `exp/qsearch-delta-pruning`

コミット:

```text
b6b5531 Prune quiet qsearch captures by delta margin
```

枝刈り条件:

- qsearch 内のみ
- 自玉が王手されていない
- 通常の捕獲手
- 成りではない
- 王手ではない
- 捕獲される駒の価値が取得できる
- `stand_pat + victim_value + margin <= alpha` の場合に探索しない

最初に `margin = 1200` を試したが skip が 0 だったため、実効性を見るため `margin = 0` で検証した。

## プロファイル結果

条件:

```text
search_profile seed 9501
36 samples
```

`v2.4.1` baseline:

```text
total nodes: 8,511,421
qsearch nodes: 7,686,858
elapsed: 29,677.80 ms
nps: 286,794.20
```

qsearch delta pruning:

```text
total nodes: 8,302,480
qsearch nodes: 7,479,512
elapsed: 28,106.42 ms
nps: 295,394.47
delta candidates: 1,571,189
delta skips: 365,406
candidate skip rate: 23.26%
```

速度・ノード数の面では小さい改善が見えた。

## 対局ベンチ結果

比較対象: `v2.4.1`

共通条件:

```text
positions: taya36.sfen
depth: 5
time-limit-ms: 100
max-plies: 200
adjudicate-at-max-plies
weights: policy_weights_v2.1.0.binary
```

結果:

```text
20 games / seed 5001: 9-10-1, total score rate 47.5%
40 games / seed 5101: 26-14-0, total score rate 65.0%
100 games / seed 5201: 48-51-1, total score rate 48.5%
```

100局ベンチ:

```text
NewWin: 48
BaselineWin: 51
Draw: 1
decisive CI: 38.88%..58.20%
total CI: 38.75%..58.25%
end reasons:
  Resign: 92
  MaxPliesAdjudication: 7
  RepetitionDraw: 1
paired starts:
  new sweeps: 6
  baseline sweeps: 8
  splits: 35
  draw/mixed: 1
```

`record_analyze` では score/result sign mismatch が 16 局あり、最大級の mismatch は `final_in_check=true` かつ合法手数 0-3 の終局付近が多かった。

## 判断

qsearch delta pruning は採用しない。

理由:

- 100局で `v2.4.1` に対して互角以下だった。
- 40局では良く見えたが、100局で再現しなかった。
- 速度改善は小さく、強さ低下リスクを上回る根拠がない。
- 終局付近の mismatch 傾向を改善する変更ではなかった。

今回の結果は、短い20局・40局で良化して見える探索変更でも、最低100局または複数seed確認を通す必要があることを再確認するものだった。

## 次の候補

GPT-5.5 xhigh の前回提案では、qsearch delta pruning の次候補は null-move pruning だった。

ただし qsearch delta pruning の100局失敗を反映し、次の実装前に GPT-5.5 xhigh へ再判断を依頼した。null-move を進める場合でも、最初は保守的な条件で小さく実装し、速度プロファイルと対局ベンチを分けて評価する。

## 今後の評価方針

速度改善と戦略改善を混同しないため、今後は原則として次の順に確認する。

1. `search_profile` で nodes、qsearch nodes、elapsed、nps、枝刈り回数を比較する。
2. 同一 depth 条件で対局し、探索手法そのものの強さを確認する。
3. 同一 time 条件で対局し、速度改善が実戦強度へ変換されるか確認する。
4. 20局で明確に悪ければ撤退し、良さそうでも100局または複数seedまで採用しない。
