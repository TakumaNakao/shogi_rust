# Bonanza pairwise KPP 学習への方針転換

- 作成日時: 2026-06-27 11:25:44 UTC
- 対象ブランチ: `training/strong-weight-learning-infra`
- 目的: 長時間学習で強いKPP重みを作るため、現行MMTO-liteの延長ではなく、棋譜手と兄弟合法手を直接比較する学習器を追加する。

## 結論

現時点では、MMTO-lite/feedback/policy-anchor系を単純に長く回すべきではない。

複数実験で train/valid loss や feedback violation は改善したが、rerank gateや対局に結びつく改善は小さかった。さらに best epoch はほぼ1に寄り、epochを進めるほど clamp される重みが増える傾向があった。これは学習時間不足ではなく、目的関数と実際の指し手選択のズレが主因と判断する。

次の本命は、Bonanza系に近い「棋譜手 vs 兄弟合法手」のpairwise rankingである。KPP評価を続けるなら、局面値そのものではなく、同一rootで選ばれた手の子局面評価が兄弟手より高くなるように直接最適化する方が筋が良い。

## 実装

`bonanza_pairwise_train` を追加した。

主な仕様:

- 入力は `sfen` と `teacher_move` を持つJSONL。
- `dataset_build` / `csa_policy_dump` の出力を読める。
- 学習データはストリーミングで読み、全train sampleをメモリに載せない。
- 各局面で棋譜手の子局面と、現モデル評価で上位に来る兄弟合法手を比較する。
- 損失は `softplus((margin_cp - (teacher_score - sibling_score)) / softplus_temp_cp)`。
- 更新はKPP差分特徴に対するSGD。
- `--anchor-l2` と `--max-weight-delta` で初期重みからのドリフトを抑える。
- `--freeze-material` で素材係数更新を止められる。
- valid指標は top1 accuracy、pair accuracy、mean rank、loss。
- best checkpointは valid top1 最大、同点なら valid loss 最小で選ぶ。

補助スクリプトとして `tools/run_bonanza_pairwise_pipeline.sh` を追加した。小実験向けのデフォルトで、`csa_policy_dump` によるJSONL作成から `bonanza_pairwise_train` までを実行する。

## スモークテスト

GPT-5.3-codex-spark サブエージェントに極小スモークを委任した。

条件:

- input: `data/wdoor/extract/2026`
- `MAX_RECORDS=80`
- `VALID_PERCENT=20`
- `MIN_PLY=16`
- `MAX_PLY=60`
- `EPOCHS=1`
- `BATCH_SIZE=16`
- `VALID_MAX_SAMPLES=40`
- `HARD_NEGATIVES=2`
- `LEARNING_RATE=0.005`
- `MAX_WEIGHT_DELTA=0.0005`
- `ANCHOR_L2=0.0002`
- `FREEZE_MATERIAL=1`

結果:

- 実行成功。
- baseline valid: top1=6.25%, pair_acc=58.43%, mean_rank=23.50, loss=1.252107
- epoch1 train: top1=12.50%, pair_acc=69.97%, mean_rank=18.02, loss=1.327773
- epoch1 valid: top1=6.25%, pair_acc=58.43%, mean_rank=23.50, loss=1.252100
- best epoch: 1

これは強さ評価ではなく、入出力とストリーミング更新が動くことの確認である。生成された `best.binary` と `final.binary` は採用候補ではないため削除済み。

## 次の検証計画

次はサブエージェントに、同じ学習器で小規模パラメータ探索を並列実行させる。

優先する軸:

- `MAX_RECORDS`: 2k -> 10k
- `HARD_NEGATIVES`: 2 / 4 / 8
- `LEARNING_RATE`: 0.001 / 0.003 / 0.005
- `MAX_WEIGHT_DELTA`: 0.0005 / 0.001
- `WINNER_ONLY` と `DECISIVE_ONLY` の有無

採用判断はtrain lossではなく、held-out top1、pair accuracy、mean rank、既存rerank gate、短時間対局ベンチで行う。ここでも10局程度の良結果だけでは採用しない。

## 小規模A/B結果

GPT-5.3-codex-spark サブエージェントに、`MAX_RECORDS=2000` の小規模比較を2本委任した。どちらも候補重みは採用せず削除済み。

共通条件:

- input: `data/wdoor/extract/2026`
- `VALID_PERCENT=10`
- `MIN_PLY=16`
- `MAX_PLY=120`
- `EPOCHS=2`
- `BATCH_SIZE=128`
- `VALID_MAX_SAMPLES=500`
- `LEARNING_RATE=0.003`
- `MAX_WEIGHT_DELTA=0.001`
- `ANCHOR_L2=0.0002`
- `FREEZE_MATERIAL=1`

### A: hard negatives 4

- `RUN_DIR=data/bonanza_pairwise_runs/sweep_a_h4_lr003_20260627_112949`
- baseline valid: top1=20.00%, pair_acc=70.67%, mean_rank=22.89, loss=0.841781
- epoch1 valid: top1=20.00%, pair_acc=70.67%, mean_rank=22.88, loss=0.841780
- epoch2 valid: top1=20.00%, pair_acc=70.67%, mean_rank=22.88, loss=0.841780
- best epoch: 2

### B: hard negatives 8

- `RUN_DIR=data/bonanza_pairwise_runs/sweep_b_h8_lr003_20260627_112946`
- baseline valid: top1=20.00%, pair_acc=70.67%, mean_rank=22.89, loss=0.727741
- epoch1 valid: top1=20.00%, pair_acc=70.66%, mean_rank=22.89, loss=0.727741
- epoch2 valid: top1=20.00%, pair_acc=70.66%, mean_rank=22.89, loss=0.727741
- best epoch: 0

判断:

- top1、pair accuracy、mean rank はどちらも実質改善なし。
- A/Bのlossはhard negative数が違うため直接比較しない。
- 更新量が小さすぎる可能性がある。次は `LEARNING_RATE` と `MAX_WEIGHT_DELTA` を少し緩めるか、より多いrecordsで動くかを確認する。
- ただし、単にepochを増やすだけではなく、valid top1/pair/rankが動く設定を先に探す。
