# MMTO-lite小規模検証結果

- 作成日時: 2026-06-22 13:53:45 UTC
- 対象ブランチ: `report/mmto-lite-small-validation`
- 目的: `docs/mmto_lite_validation.md` の手順に従い、MMTO-lite root listwise学習が小規模データでoffline gateと短時間対局gateを通るか確認する。

## 結論

今回のMMTO-lite候補は採用しない。

`blend_0.05` は20局smokeでは 11勝8敗1分、score rate 57.50% と一見有望だったが、40局gateで 19勝21敗、score rate 47.50% まで落ちた。手順書の40局gate条件である50%以上を満たさないため破棄する。

完全MMTOへ進む条件も満たしていない。次は完全MMTOではなく、MMTO-liteの学習信号と目的関数を改善する。

## 使用データ

既存run:

```text
data/mmto/runs/d3_top8_200_20260622_131843
```

内容:

- train samples: 180
- valid samples: 20
- teacher: searched
- depth: 3
- teacher-score-top: 8
- positions: `converted_records2016_10818.sfen`

## パラメータ探索

最初の低学習率設定:

```text
data/mmto/runs/tune_d3_top8_200_20260622_133726
```

結果:

| config | best epoch | valid regret | 判定 |
| --- | ---: | ---: | --- |
| lr0.05 mt150 tt100 delta0.02 | 0 | 2.12 | 不採用 |
| lr0.10 mt150 tt100 delta0.02 | 0 | 2.12 | 不採用 |
| lr0.05 mt100 tt100 delta0.02 | 0 | 2.12 | 不採用 |
| lr0.10 mt100 tt100 delta0.05 | 0 | 2.12 | 不採用 |

低学習率では `max_abs_delta` が最大でも約0.0002で、重みが実質動かなかった。

強い更新設定:

```text
data/mmto/runs/aggressive_d3_top8_200_20260622_134232
```

結果:

| config | 傾向 | 判定 |
| --- | --- | --- |
| lr10 mt50 tt100 delta0.02 | valid regret悪化 | 不採用 |
| lr50 mt50 tt100 delta0.02 | valid regret悪化 | 不採用 |
| lr10 mt30 tt100 delta0.02 | valid regret悪化 | 不採用 |
| lr50 mt30 tt100 delta0.02 | valid regret悪化 | 不採用 |
| lr100 mt50 tt100 delta0.05 | valid regret悪化 | 不採用 |
| lr100 mt30 tt100 delta0.05 | valid regret悪化 | 不採用 |

強く動かすとtrain CEは少し下がるが、valid selected regretが悪化した。

中間学習率設定:

```text
data/mmto/runs/midlr_d3_top8_200_20260622_134432
```

有望だった設定:

```text
learning-rate: 5
model-temperature: 30
teacher-temperature: 100
max-weight-delta: 0.02
anchor-l2: 0.0005
```

valid推移:

| epoch | valid CE | top1 | selected regret | p90 | p95 | expected regret |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.078625 | 25.00% | 2.12 | 4.39 | 5.84 | 3.21 |
| 1 | 2.078599 | 20.00% | 2.22 | 4.39 | 5.84 | 3.21 |
| 2 | 2.078594 | 20.00% | 2.13 | 3.92 | 5.84 | 3.21 |
| 3 | 2.078640 | 25.00% | 1.95 | 3.92 | 5.84 | 3.21 |
| 4 | 2.078702 | 20.00% | 3.61 | 6.57 | 12.63 | 3.21 |
| 5 | 2.078760 | 25.00% | 3.39 | 6.57 | 12.63 | 3.22 |

epoch 3だけselected regretが 2.12 から 1.95 に改善したため、最小候補として保存しblend評価へ進めた。ただしvalid 20局面だけの非常に弱い改善であり、安定性は低い。

## 候補重み

```text
data/mmto/runs/candidate_lr5_mt30_d002_20260622_134622
```

作成したblend:

- `blend_0.05.binary`
- `blend_0.10.binary`
- `blend_0.20.binary`

40局gate失敗後、巨大な `.binary` は削除した。ログと棋譜は残している。

## 20局 Smoke

```text
record-dir: data/mmto/runs/candidate_lr5_mt30_d002_20260622_134622/bench20_blend005_seed5101
weight: blend_0.05
seed: 5101
```

結果:

```text
new wins: 11
baseline wins: 8
draws: 1
new decisive win rate: 57.89%
new total score rate: 57.50%
decisive win rate 95% CI: 36.28%..76.86%
total score rate 95% CI: 36.40%..78.60%
```

paired starts:

```text
new sweeps: 2
baseline sweeps: 0
splits: 7
draw/mixed pairs: 1
```

20局smokeは通過と判断した。

## 40局 Gate

```text
record-dir: data/mmto/runs/candidate_lr5_mt30_d002_20260622_134622/bench40_blend005_seed5201
weight: blend_0.05
seed: 5201
```

結果:

```text
new wins: 19
baseline wins: 21
draws: 0
new decisive win rate: 47.50%
new total score rate: 47.50%
decisive win rate 95% CI: 32.94%..62.50%
total score rate 95% CI: 32.02%..62.98%
```

paired starts:

```text
new sweeps: 2
baseline sweeps: 3
splits: 15
draw/mixed pairs: 0
```

40局gateは不通過。20局の勝ち越しは揺らぎと判断する。

## 観察

1. 低学習率ではKPP重みがほぼ動かない。
2. 学習率を大きくすると重みは動くが、valid selected regretが悪化しやすい。
3. `lr5 / model_temperature30` だけ一時的にvalid selected regretが改善したが、次epochで崩れた。
4. 20局smokeは勝ち越したが、40局では負け越した。
5. 現在のroot listwise目的関数だけでは、まだ安定した学習信号になっていない。

## 運用上の注意

`kpp_weight_check` は全KPP重みをソートするため、巨大メモリを使う。複数候補で並列実行するとメモリを圧迫する。以後、ハイパーパラメータ探索では次を守る。

- `mmto_train --output /dev/null` でログだけ取る。
- 対局候補だけ `--best-checkpoint-path` で保存する。
- `kpp_weight_check` は候補1本に対して単独実行する。
- 不採用の `.binary` はログ確認後に削除する。

## 次の方針

完全MMTOへは進まない。次はMMTO-lite自体を改善する。

優先候補:

1. valid局面を20から100以上へ増やし、offline gateの揺らぎを減らす。
2. `candidate_scope=scored` のまま、teacher top候補の品質を改善する。
3. root rankデータに `teacher gap` と `candidate count` の統計を追加し、学習に向かない平坦局面を除外する。
4. listwise単独ではなく、teacher gapが大きい候補だけpairwise margin補助を加える。
5. 学習率をweight単位ではなく、局面のmodel score変化量で制御する。

次の実験は `depth3/top8/max_positions1000/valid100以上` で、まずdump品質を確認する。その後、pairwise補助を実装するか判断する。
