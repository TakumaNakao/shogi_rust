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

## 追加C/D結果

更新量とデータ品質の仮説を分けて、さらに2本をサブエージェントに委任した。

### C: 更新量を強める

設定:

- `MAX_RECORDS=2000`
- `HARD_NEGATIVES=4`
- `LEARNING_RATE=0.02`
- `MAX_WEIGHT_DELTA=0.005`
- `ANCHOR_L2=0.0001`
- `EPOCHS=3`

結果:

- `RUN_DIR=data/bonanza_pairwise_runs/sweep_c_h4_lr02_delta005_20260627_113322`
- baseline valid: top1=20.00%, pair_acc=70.67%, mean_rank=22.89, loss=0.841781
- epoch1 valid: top1=20.00%, pair_acc=70.69%, mean_rank=22.86, loss=0.841779
- epoch2 valid: top1=20.00%, pair_acc=70.70%, mean_rank=22.85, loss=0.841777
- epoch3 valid: top1=20.00%, pair_acc=70.69%, mean_rank=22.86, loss=0.841775
- best epoch: 3

判断:

- 更新量を強めても、valid top1は動かなかった。
- pair_accとmean_rankは微小改善したが、採用判断に使える大きさではない。
- clampは0で、`MAX_WEIGHT_DELTA` は制約になっていない。

### D: 高レート・決着棋譜

設定:

- `MAX_RECORDS=10000`
- `MIN_PLAYER_RATE=4000`
- `DECISIVE_ONLY=1`
- `HARD_NEGATIVES=8`
- `LEARNING_RATE=0.01`
- `MAX_WEIGHT_DELTA=0.003`
- `ANCHOR_L2=0.0001`
- `EPOCHS=2`

dataset:

- games used: 141
- train records: 9000
- valid records: 1000

結果:

- `RUN_DIR=data/bonanza_pairwise_runs/sweep_d_r4000_h8_lr01_20260627_113341`
- baseline valid: top1=20.90%, pair_acc=73.36%, mean_rank=24.76, loss=0.727909
- epoch1 valid: top1=21.00%, pair_acc=73.35%, mean_rank=24.77, loss=0.727909
- epoch2 valid: top1=21.10%, pair_acc=73.35%, mean_rank=24.77, loss=0.727907
- best epoch: 2

判断:

- top1は +0.20pt だが、pair_accとmean_rankは非改善。
- データ品質を上げても、現SGD更新では明確な改善に届いていない。

## 次の実装判断

`bonanza_pairwise_train` は動作するが、現状の単純SGDでは更新が弱く、valid順位指標にほとんど反映されない。

次は学習器側を改良する。

- 第一候補: pairwise専用のAdagradを追加する。
- 第二候補: 評価済み子局面の特徴再計算を避け、同一batch内でより多くのhard negative pairを安定更新する。
- 第三候補: `max_weight_delta` ではなく、触れた特徴に対するproximal L2を強め、clampに頼らない更新制御へ寄せる。

この状態で長時間学習に進むのはまだ早い。

## Adagrad E/F結果

pairwise専用Adagradを実装し、SGD C/D と同系統の条件で確認した。

### E: small data + Adagrad

設定:

- `MAX_RECORDS=2000`
- `HARD_NEGATIVES=4`
- `OPTIMIZER=adagrad`
- `LEARNING_RATE=0.003`
- `MAX_WEIGHT_DELTA=0.005`
- `ANCHOR_L2=0.0001`
- `EPOCHS=3`

結果:

- `RUN_DIR=data/bonanza_pairwise_runs/sweep_e_adagrad_h4_lr003_20260627_113707`
- baseline valid: top1=20.00%, pair_acc=70.67%, mean_rank=22.89, loss=0.841781
- epoch1 valid: top1=20.00%, pair_acc=70.04%, mean_rank=23.35, loss=0.840860, clamped=36111
- epoch2 valid: top1=19.00%, pair_acc=70.34%, mean_rank=23.12, loss=0.840466, clamped=131791
- epoch3 valid: top1=19.50%, pair_acc=70.14%, mean_rank=23.27, loss=0.840408, clamped=281179
- best epoch: 1

### F: high-rate data + Adagrad

設定:

- `MAX_RECORDS=10000`
- `MIN_PLAYER_RATE=4000`
- `DECISIVE_ONLY=1`
- `HARD_NEGATIVES=8`
- `OPTIMIZER=adagrad`
- `LEARNING_RATE=0.003`
- `MAX_WEIGHT_DELTA=0.005`
- `ANCHOR_L2=0.0001`
- `EPOCHS=3`

結果:

- `RUN_DIR=data/bonanza_pairwise_runs/sweep_f_adagrad_r4000_h8_lr003_20260627_113732`
- dataset: train records=9000, valid records=1000
- baseline valid: top1=20.90%, pair_acc=73.36%, mean_rank=24.76, loss=0.727909
- epoch1 valid: top1=20.20%, pair_acc=72.71%, mean_rank=25.34, loss=0.727292, clamped=399475
- epoch2 valid: top1=20.80%, pair_acc=72.61%, mean_rank=25.43, loss=0.727154, clamped=757089
- epoch3 valid: top1=20.70%, pair_acc=72.93%, mean_rank=25.14, loss=0.727115, clamped=1395598
- best epoch: 0

判断:

- Adagradはtrain指標を大きく動かしたが、valid top1/pair_acc/mean_rankは悪化した。
- `clamped_weights` が急増しており、局所的な過更新が起きている。
- 「optimizerを強くするだけ」では解決しない。
- 次は、更新式の強化ではなく、教師信号の質とpair選別を見直すべきである。

## 次の方針

GPT-5.5 xhigh に再分析を依頼した結果、次の本命は **既存MMTO tree dumpを使い、teacher score差が大きいpairだけを使うBonanza風目的** と判断した。

理由:

- 生棋譜手を常に正解扱いすると、悪手、序盤依存、人間的妥協、探索深度差がノイズになる。
- 既存MMTO tree dumpには同一root内の候補手とteacher scoreがある。
- `teacher_score(best) - teacher_score(other)` が大きいpairだけを使えば、「明確に良い手を明確に悪い手より上げる」信号になる。
- KPPで続ける場合にも、将来NNUEへ進む場合にも、このteacher pair生成は再利用できる。

当面やらないこと:

- 生棋譜手pairwiseのSGD/Adagrad係数調整。
- Adagrad E/F系の延長。
- opening book / opening prior を本命にすること。
- 小型NNUEへの即移行。

次の実装対象:

- MMTO tree JSONLをstreamingで読むlarge-margin teacher pair trainer。
- `teacher_best_score - teacher_other_score >= margin` のpairだけ採用。
- pair重みはscore差で軽くスケールし、上限を置く。
- valid指標はteacher-best rank、large-margin pair accuracy、lossを主にする。
- clamp数、更新norm、局面重複率をログに出す。

## Large-margin teacher-pair probe

既存 `mmto_tree_train` には `all-candidates`、`score-gap`、`teacher-mismatch` が既にあるため、まず専用runnerを追加して小実験した。

追加:

- `tools/run_mmto_large_margin_teacher_pairs.sh`
- `tools/run_mmto_from_dump.sh` の `OPTIMIZER` / `ADAGRAD_EPSILON` 環境変数化。

実験:

- run: `data/mmto/runs/mmto_large_margin_teacher_pairs_probe_20260627_114408`
- source dump: `data/mmto/runs/mmto_rerank_long_20260624_140151`
- train/valid: `9000 / 1000`
- `TEACHER_TOP_K=1`
- `BAD_CANDIDATE_SCOPE=all-candidates`
- `MIN_REGRET_CP=100`
- `MAX_PAIRS_PER_SAMPLE=16`
- `PAIR_MINING=loss-top`
- `PAIR_WEIGHT_MODE=score-gap`
- `PAIR_WEIGHT_SCALE_CP=150`
- `BEST_METRIC=teacher-mismatch`
- `OPTIMIZER=adagrad`

結果:

```text
baseline valid:
  loss=27.188143
  selected_regret=684.24
  p90=231.39
  p95=262.88
  teacher_match=18.40%
  bad50=0.5290
  bad100=0.3750

epoch 1 valid:
  loss=27.186134
  selected_regret=684.30
  teacher_match=18.30%
  bad100=0.3750

epoch 2 valid:
  loss=27.185123
  selected_regret=684.50
  teacher_match=18.20%
  bad100=0.3760

epoch 6 valid:
  loss=27.183174
  selected_regret=684.57
  teacher_match=18.20%
  bad100=0.3770

best_epoch=0
```

`score_gate` / `rerank_gate` は未実施。理由は `best_epoch=0` でbaselineが最良と判定され、学習候補が早期棄却されたため。

判断:

- large-margin pair objective のlossはわずかに下がった。
- しかし teacher_match は下がり、selected_regret と bad100 は悪化した。
- これは「epochを増やせばよい」兆候ではない。
- 現設定を長時間化するより、目的関数・データ分割・teacher信号の作り方を再検討する。
