# 探索境界の低分岐王手応手延長 実験報告

- 作成日時: 2026-06-19 13:47:02 UTC
- 実験ブランチ: `exp/boundary-check-evasion-extension`
- 結論: 採用。現行master直接比較100局で明確に勝ち越した。

## 実装内容

`alpha_beta_search` の深さ境界で、次局面が王手局面かつ合法応手が少ない場合だけ、qsearchへ落とさず通常探索を1手だけ延長する。

条件:

```text
depth == 1
check_evasion_extension_budget > 0
position.in_check()
position.legal_moves().len() <= 3
```

拡張予算は1回だけ。qsearch本体、候補手生成、SEE、評価関数、詰みスコアは変更していない。

これは過去に棄却した `qsearch all check evasions` や `qsearch capped check evasions` と異なり、qsearchを広げない。`forced evasion static fold` とも異なり、静的評価への折り畳みではなく通常探索で応手を読む。

## 速度確認

条件:

```bash
target/release/search_profile \
  --weights policy_weights_v2.1.0.binary \
  --positions taya36.sfen \
  --samples 36 \
  --depth 5 \
  --seed 9501
```

master基準:

```text
total nodes: 8511421
elapsed ms: 約28104
```

実験結果:

```text
total nodes: 8638168
quiescence nodes: 7802474
check evasion extensions: 11652
elapsed ms: 29258.25
```

ノード増加は約1.5%。経過時間は測定揺れ込みで約4%増に収まり、事前ゲートの「elapsed悪化+8%以内、nodes増加+15%以内」を満たした。

## v2.4.1 比較

条件:

```text
positions: taya36.sfen
depth: 5
time-limit-ms: 100
max-plies: 200
jobs: 4
seed: 3601
```

20局:

```text
new wins: 15
baseline wins: 4
draws: 1
new total score rate: 77.50%
paired starts:
  new sweeps: 6
  baseline sweeps: 0
  splits: 3
  draw/mixed pairs: 1
```

40局:

```text
new wins: 27
baseline wins: 11
draws: 2
new decisive win rate: 71.05%
new total score rate: 70.00%
paired starts:
  new sweeps: 10
  baseline sweeps: 2
  splits: 7
  draw/mixed pairs: 1
```

## 現行master直接比較

現行masterを `/tmp/shogi_rust_master_compare` に別worktreeでビルドし、実験ブランチをnew、masterをbaselineとして比較した。

条件:

```text
positions: taya36.sfen
depth: 5
time-limit-ms: 100
max-plies: 200
jobs: 4
seed: 4601
```

20局:

```text
new wins: 12
baseline wins: 8
draws: 0
new total score rate: 60.00%
paired starts:
  new sweeps: 3
  baseline sweeps: 1
  splits: 6
  draw/mixed pairs: 0
```

40局:

```text
new wins: 24
baseline wins: 14
draws: 2
new decisive win rate: 63.16%
new total score rate: 62.50%
paired starts:
  new sweeps: 5
  baseline sweeps: 1
  splits: 12
  draw/mixed pairs: 2
```

100局:

```text
new wins: 63
baseline wins: 36
draws: 1
new decisive win rate: 63.64%
new total score rate: 63.50%
decisive win rate 95% CI: 53.82%..72.44%
total score rate 95% CI: 54.11%..72.89%
end reasons:
  MaxPliesAdjudication: 2
  RepetitionDraw: 1
  Resign: 97
paired starts:
  new sweeps: 19
  baseline sweeps: 6
  splits: 24
  draw/mixed pairs: 1
```

100局でCI下限が50%を超え、paired startsでもnew sweepがbaseline sweepを大きく上回った。

## 判断

採用する。終盤の王手応手局面でqsearchへ早く落ちすぎる問題を、小さな探索延長で補正できている可能性が高い。

今後の注意:

- 全王手拡張へ一般化しない。
- qsearch内の王手応手生成へ戻さない。
- 拡張条件を広げる場合は、必ず今回の100局結果を固定baselineとして比較する。
