# KPP guarded CE学習結果

- 作成日時: 2026-06-22 11:59:25 UTC
- 対象run: `data/wdoor/runs/wdoor2023_2026_r4000_guarded_ce_20260621_122248`
- 結論: 不採用。`best.binary` も最終重みも現行baseline重みを上回らなかった。

## 学習設定

Wdoor/WCSC系CSA棋譜 2023-2026、rate 4000以上、decisive-only、負け側100手目以降除外で4 epoch学習した。

主な安全設定:

- `--freeze-material`
- `--anchor-l2 0.0005`
- `--max-weight-delta 0.05`
- `--early-stop-min-accuracy-drop 0.5`
- `--best-checkpoint-path .../best.binary`

## 学習ログ要約

baseline validation:

- accuracy: 22.83% (2874/12590)
- CE: 3.980881

epoch validation:

- epoch 1: 22.37% (2816/12590), CE 3.980835
- epoch 2: 22.37% (2817/12590), CE 3.980831
- epoch 3: 22.37% (2817/12590), CE 3.980830
- epoch 4: 22.38% (2818/12590), CE 3.980830

重み制御:

- 最終 `max|w-w0|`: 0.002660
- `clamped_weights`: 0
- material coefficient: 0.145648
- 最大重み: 1.762908
- 最小重み: -2.947418
- 非ゼロ要素: 49.2777% (109606268/222425676)

安全制御は効いており、重み暴走は起きなかった。一方で、検証精度はbaselineから約0.45ポイント低下したまま改善しなかった。

## 単独重み検査

`kpp_weight_check` で `best.binary` と最終重みを確認した。

- ファイル読込成功
- weight count: 222425676
- NaN/infによる異常終了なし
- 統計値は学習ログと一致

## 対局ベンチ

比較条件:

- engine: 同一 `target/release/usi_engine`
- new weights: 学習重み
- baseline weights: `policy_weights_v2.1.0.binary`
- positions: `taya36.sfen`
- depth: 5
- time-limit-ms: 100
- max-plies: 200
- adjudicate-at-max-plies

### best.binary seed 3301 / 40局

- NewWin: 27
- BaselineWin: 13
- Draw: 0
- new total score rate: 67.50%
- 95% CI total: 52.98%..82.02%
- end reasons:
  - Resign: 39
  - MaxPliesAdjudication: 1
- paired starts:
  - new sweeps: 8
  - baseline sweeps: 1
  - splits: 11
  - draw/mixed: 0

この40局は有望に見えたが、追加100局で再現しなかった。

### final weight seed 3302 / 40局

- NewWin: 17
- BaselineWin: 20
- Draw: 3
- new total score rate: 46.25%
- 95% CI total: 31.39%..61.11%
- end reasons:
  - Resign: 36
  - RepetitionDraw: 3
  - MaxPliesAdjudication: 1
- paired starts:
  - new sweeps: 3
  - baseline sweeps: 5
  - splits: 10
  - draw/mixed: 2

最終重みは不採用。

### best.binary seed 3401 / 100局

- NewWin: 42
- BaselineWin: 54
- Draw: 4
- new decisive win rate: 43.75%
- new total score rate: 44.00%
- decisive 95% CI: 34.26%..53.72%
- total 95% CI: 34.47%..53.53%
- end reasons:
  - Resign: 95
  - RepetitionDraw: 4
  - MaxPliesAdjudication: 1
- paired starts:
  - new sweeps: 3
  - baseline sweeps: 10
  - splits: 34
  - draw/mixed: 3
- average final score for new: -55.3

`best.binary` も100局で明確に悪化寄り。採用しない。

## 判断

今回のguarded CEは、重みを壊さずに長時間学習を完走できた点では成功した。しかし、強さ改善には失敗した。

主な示唆:

- hard-label CEは、検証CEをわずかに下げても対局力を改善しない。
- validation accuracyがbaselineより低い重みは、短い40局で上振れしても100局で崩れる可能性が高い。
- `best-checkpoint` は有用だが、検証精度自体がbaselineを超えない限り採用判断には弱い。

## 次の方針

同じhard-label CEを長く回す実験は優先度を下げる。

次に試すべき候補:

1. regret-weighted soft CE
   - 教師手だけをone-hotで押すのではなく、探索評価差が小さい候補手にも確率を残す。
2. root候補手のpairwise/listwise ranking
   - 棋譜手の模倣より、探索で上位に来るべき手の順位を直接学習する。
3. validation gateの強化
   - baseline validation accuracyを下回る重みは原則ベンチ優先度を下げる。

今回の重みはリリースしない。
