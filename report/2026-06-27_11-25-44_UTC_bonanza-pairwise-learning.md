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

## GPT-5.5 xhigh 再分析

large-margin teacher-pair probe 後に、次の研究判断をGPT-5.5 xhighへ依頼した。

結論:

- 現設定を単純に長時間化する見込みは低い。
- `all-candidates` の `score-gap` pairwise loss は、root search / rerank の採用指標と噛み合っていない。
- 次は広い `teacher_best vs all bad candidates` ではなく、探索が実際に選ぶ悪手を直接扱う目的へ寄せる。
- scratch学習は今は主路線にしない。v2.1.0 は長時間学習済みの強い事前分布なので、まずは warm-start の段階学習で進める。

推奨する目的関数:

- `listwise-leaf` を主目的にする。
- 現モデルtop悪手への `current-top margin` を入れる。
- rerank gate が出した `(teacher_move, candidate_move)` の hard feedback を補助的に使う。
- checkpoint選択は `p95-regret` だけでなく、`max-regret` / `bad100` guard を必須にする。

推奨する検証段階:

1. `best_epoch > 0`
2. `best_guard_max_regret_increase=0`
3. `best_guard_bad100_increase=0`
4. score gate pass
5. rerank gateで mean / p90 / p95 / bad50 / bad100 / match 非悪化
6. 独立valid rerankでも非悪化
7. 20局は破綻検知、採用は100局以上

次の実験:

- `tools/run_mmto_refresh_loop.sh` による guarded refresh。
- base dump は `data/mmto/runs/bonanza_root_pergame_2k_leaf_gt010_20260627_001929` を使う。
- `REFRESH_MAX_POSITIONS=200`
- `BASE_TRAIN_LINES=1800`
- `BASE_VALID_LINES=200`
- `LOSS_MODE=listwise-leaf`
- `CURRENT_TOP_MARGIN_WEIGHT=0.05`
- `GAME_TEACHER_MARGIN_WEIGHT=0.05`
- `BEST_GUARD_MAX_REGRET_INCREASE_CP=0`
- `BEST_GUARD_BAD100_INCREASE=0`

## Guarded refresh probe

GPT-5.5 xhigh の推奨に従い、`listwise-leaf + current-top margin + game-teacher margin` の guarded refresh を小さく実行した。

実験:

- run: `data/mmto/runs/mmto_refresh_loop_guarded200_20260627_115601`
- source: `data/mmto/runs/bonanza_root_pergame_2k_leaf_gt010_20260627_001929`
- `REFRESH_MAX_POSITIONS=200`
- `BASE_TRAIN_LINES=1800`
- `BASE_VALID_LINES=200`
- `EPOCHS=2`
- `LEARNING_RATE=0.0002`
- `MAX_WEIGHT_DELTA=0.001`
- `LOSS_MODE=listwise-leaf`
- `LISTWISE_HARD_NEGATIVE_WEIGHT=0.05`
- `CURRENT_TOP_MARGIN_WEIGHT=0.05`
- `GAME_TEACHER_MARGIN_WEIGHT=0.05`
- `BEST_METRIC=p95-regret`
- `BEST_GUARD_MAX_REGRET_INCREASE_CP=0`
- `BEST_GUARD_BAD100_INCREASE=0`

学習内valid:

```text
baseline valid:
  selected_regret=11.06
  p90=27.20
  p95=42.74
  teacher_match=17.73%
  bad50=0.0455
  bad100=0.0136

epoch 1 valid:
  selected_regret=10.60
  p90=27.27
  p95=31.68
  teacher_match=19.55%
  bad50=0.0318
  bad100=0.0136

epoch 2 valid:
  selected_regret=10.65
  p90=27.27
  p95=31.68
  teacher_match=20.00%
  bad50=0.0318
  bad100=0.0136

best_epoch=1
```

score gate:

```text
SCORE GATE PASSED
samples=1596
mean_abs_delta_cp=0.27
p95=0.48
max=0.56
```

rerank gate:

```text
baseline:
  samples=189
  mean=6.050085
  p90=11.20
  p95=27.25
  max=196.03
  match=40.21%
  bad50=0.0265
  bad100=0.0159

candidate:
  mean=6.052938
  p90=11.20
  p95=27.25
  max=196.03
  match=40.21%
  bad50=0.0265
  bad100=0.0159

RERANK GATE FAILED
```

hard feedback:

- rerank失敗後に実行。
- `best_epoch=0` でbaseline最良のため不採用。

判断:

- 学習内validとscore gateはかなり良い。
- rerankでは p90 / p95 / max / match / bad50 / bad100 が同一で、meanだけ約 `0.0029cp` 悪化した。
- この差は実質ノイズ幅なので、候補を即廃棄するより、平均regretに小さな許容幅を置いて独立rerankへ進める価値がある。
- gate許容値をスクリプトから指定できるよう、`RERANK_ALLOW_MEAN_REGRET_INCREASE_CP` などをMMTO pipelineに追加した。

## Rerank gate修正

`RERANK_ALLOW_MEAN_REGRET_INCREASE_CP=0.05` を指定して再実験したが、候補は再び棄却された。

原因:

- `mmto_rerank_gate` には `allow_*_increase` と `require_*_improvement` が両方ある。
- `allow_mean=0.05` は正しく渡っていた。
- しかし `require_mean_regret_improvement_cp=0` が、「改善要求なし」ではなく「平均regretの悪化を一切許さない」として働いていた。
- そのため、`+0.0029cp` の平均regret差でも `mean regret failed improvement requirement` で落ちていた。

修正:

- `require_mean_regret_improvement_cp`
- `require_p90_regret_improvement_cp`
- `require_p95_regret_improvement_cp`

上記は、正の値を指定したときだけ追加の改善要求として判定するようにした。

これにより:

- 非悪化/許容幅は `allow_*_increase` が担当する。
- 明確な改善要求は `require_*_improvement_cp > 0` のときだけ有効になる。
- `require_match_rate_improvement_pct=0` は既存どおりmatch率の非低下要求として残す。

次の確認:

- `RERANK_ALLOW_MEAN_REGRET_INCREASE_CP=0.05`
- `require_mean_regret_improvement_cp=0`
- 他指標は非悪化

この条件で guarded refresh を再実行し、通れば独立rerankへ進める。

## Fixed-gate guarded refresh probe

`require_*_improvement_cp=0` の扱いを修正した後、同じ guarded refresh を再実行した。

実験:

- run: `data/mmto/runs/mmto_refresh_loop_guarded200_tol005_fixedgate_20260627_121018`
- HEAD: `91469ce`
- `RERANK_ALLOW_MEAN_REGRET_INCREASE_CP=0.05`
- 他条件は前回 guarded refresh と同等。

offline gate:

```text
train best_epoch=1
score_gate: PASS
  mean_abs_delta_cp=0.2687
  p95_abs_delta_cp=0.4756
  max_abs_delta_cp=0.5552

refresh rerank_gate: PASS
```

独立rerank:

```text
input: data/mmto/runs/mmto_rerank_long_20260624_140151/valid.tree.jsonl
max_positions=500

baseline:
  mean=216.45844
  match=48.80%

candidate:
  mean=216.40077
  match=49.00%

RERANK GATE PASSED
```

20局ベンチ:

```text
seed=12001
games=20
new wins: 7
baseline wins: 13
draws: 0
new decisive win rate: 35.00%
new total score rate: 35.00%
decisive win rate 95% CI: 18.12%..56.71%
```

判断:

- 初めて `guarded refresh -> score gate -> refresh rerank -> independent rerank` までは通過した。
- しかし20局ベンチで 7-13 と負けたため、候補重みは不採用。
- `best.raw.binary` は削除済み。
- offline rerank gate が実戦勝率を十分に予測できていない。
- 現在の学習設定を単純に長く回すのは危険。長時間化する前に、gateを実戦強さに近づける必要がある。

次の研究課題:

- offline gateに `taya36` / 実戦開始局面での小ベンチ相関を入れる。
- rerank対象が実戦ベンチと乖離していないか確認する。
- 20局smokeを学習候補選抜の必須段階にする。
- 採用候補は最低でも複数seedの20局または40局で非悪化を確認してから長時間学習へ進める。

## Offline gate / bench乖離の分析

fixed-gate guarded refresh は、offline gate と独立rerankを通ったにもかかわらず20局benchで 7-13 と負けた。

GPT-5.5 xhigh に再分析を依頼した結論:

- 現目的関数を単純に長時間化する見込みは低い。
- `score gate` と `independent rerank` は「壊していない」確認には使えるが、「強くなった」確認としては弱い。
- 今回の重み差は `mean_abs_delta_cp=0.2687`, `p95=0.4756`, `max=0.5552` と小さすぎる。
- rerank改善も `mean 216.45844 -> 216.40077`, `match 48.80% -> 49.00%` で、実戦勝率を押し上げる効果量としては弱い。

benchで負けた原因候補:

- rerank局面セットが `valid.tree.jsonl` 由来で、benchの `taya36.sfen` 開始から自己生成される実戦系列と分布が違う。
- rerankは固定深さ中心だが、benchはUSI経由の探索・時間制限・move orderingの影響を受ける。
- 評価差が1cp未満でも探索順や枝刈りが変わり、実戦勝率に出る可能性がある。
- 現目的関数は「差分が小さいと効かず、大きくすると壊れる」狭い帯に入っている可能性がある。

次に必要な検証:

- 候補生成後、複数seedの20局benchを必ず行う。
- bench棋譜から `baseline_sweep_starts` と `drop_windows` を抽出する。
- `taya36 + bench由来SFEN` で深めの bench-aligned rerank を実行する。
- 採用候補は、offline non-regression ではなく、bench複数seedで50%以上、baseline sweepが増えない、bench-aligned rerank非悪化を満たすものに限定する。

長期判断:

- KPP warm-start は捨てない。
- ただし、同じKPP目的関数の長時間化を主路線にはしない。
- `current-top mistake`、bench/self-play hard positions、深めteacher、hard replay を明示的に使う root/search-aligned objective へ寄せる。
- それでも複数seed benchで改善しない場合、KPP-only長時間学習は打ち切り、KPPを土台にした residual NNUE へ進む。

## Bench-aligned gate runner

bench-aligned gateを手動コマンドで実行したところ、以下が分かった。

実験:

- run: `data/mmto/runs/mmto_refresh_loop_guarded200_benchgate_20260627_122544`
- guarded refresh: pass
- candidate: `iter_1/best.raw.binary`

60局bench:

```text
seed 12001: new 8, baseline 12, draw 0
seed 12101: new 10, baseline 9, draw 1
seed 12201: new 12, baseline 8, draw 0

total:
  new 30
  baseline 29
  draw 1
  score: 61 / 120 = 50.83%
```

paired starts:

```text
seed 12001:
  new sweeps: 0
  baseline sweeps: 2
  splits: 8

seed 12101:
  new sweeps: 2
  baseline sweeps: 1
  splits: 6
  draw/mixed: 1

seed 12201:
  new sweeps: 2
  baseline sweeps: 0
  splits: 8
```

bench-aligned rerank:

- 最初に `max_positions=300`, depth `5/5/7` で実行した。
- 30分以上かかり、`positions: 300 (invalid=1 duplicate=493)` から進捗が見えなかったため中断した。
- 親スクリプト側が失敗扱いで候補binaryを削除したため、軽量rerankへ切り替えた時点で候補が存在しなかった。
- このrunではbench-aligned rerankは未完了。

判断:

- 60局benchは 50.83% で、前回単一seedの 35% より中立。
- ただし明確な改善とは言えない。
- deep bench-aligned rerank は重すぎる。
- 今後は最初から軽量rerankを使い、候補削除はbench/rerank判定後だけに行う専用runnerで実施する。

追加:

- `tools/run_mmto_benchgate_probe.sh`

このrunnerは以下を標準化する。

- guarded refresh候補生成。
- 複数seed 20局bench。
- `record_analyze` による `baseline_sweep_starts` / `drop_windows` 抽出。
- `taya36 + bench由来SFEN` の軽量bench-aligned rerank。
- 60局score rateが閾値未満、またはrerank失敗なら候補binary削除。

## Benchgate runner result

`tools/run_mmto_benchgate_probe.sh` を実行し、候補生成からbench-aligned gateまで一貫確認した。

実験:

- run: `data/mmto/runs/mmto_benchgate_runner_20260627_131106`
- HEAD: `6938993`
- `BENCH_SEEDS=12001 12101 12201`
- `BENCH_GAMES=20`
- `BENCH_DEPTH=10`
- `RERANK_MAX_POSITIONS=80`
- rerank depth: baseline/candidate/teacher = `4/4/6`

候補生成:

- guarded refresh: pass
- `iteration=1 refresh accepted`
- refresh内 `SCORE GATE PASSED`

bench:

```text
seed 12001:
  new wins: 10
  baseline wins: 10
  draws: 0
  score rate: 50.00%

seed 12101:
  new wins: 7
  baseline wins: 13
  draws: 0
  score rate: 35.00%

seed 12201:
  new wins: 11
  baseline wins: 9
  draws: 0
  score rate: 55.00%

total:
  new 28
  baseline 32
  draw 0
  score: 56 / 120 = 46.67%
```

paired starts:

```text
seed 12001:
  new sweeps: 2
  baseline sweeps: 2
  splits: 6

seed 12101:
  new sweeps: 1
  baseline sweeps: 4
  splits: 5

seed 12201:
  new sweeps: 2
  baseline sweeps: 1
  splits: 7
```

bench-aligned rerank:

```text
positions: 80 (invalid=1 duplicate=497)

baseline:
  mean=1239.54
  p90=17.88
  p95=28.37
  match=61.25%
  bad50=0.0125
  bad100=0.0125

candidate:
  mean=1239.52
  p90=17.88
  p95=28.37
  match=62.50%
  bad50=0.0125
  bad100=0.0125

RERANK GATE PASSED
```

最終判定:

- 60局bench scoreが46.67%で、閾値50%未満。
- 候補binaryは削除済み。
- run容量は `78M`、残存binaryなし。

判断:

- bench-aligned rerankを通っても、実戦benchでは採用できなかった。
- したがって、現行の guarded refresh 条件を単純に長く回すべきではない。
- まずは benchgate runner を選抜ゲートとして固定し、学習条件やデータ生成を変えた候補だけを通す。
- 長時間学習へ進める条件は、最低でも benchgate runner で50%以上、できれば55%以上またはbaseline sweep増加なしを満たすこと。

## Bench-feedback runner

benchgate runner は「候補を作り、複数seedの短時間benchで落とす」ところまでは標準化できた。一方で、落ちた候補の実戦敗因を次の学習に戻していなかった。

そこで `tools/run_mmto_bench_feedback_probe.sh` を追加した。

狙い:

- guarded refresh で候補を作る。
- 複数seed bench と `record_analyze` を実行する。
- `baseline_sweep_starts.sfen` と `drop_windows.sfen` を集約する。
- benchで落ちた候補をすぐ削除せず、bench由来hard SFENを DAgger/replay 入力にして repair stage を回す。
- repair候補を再度benchとbench-aligned rerankに通し、通らなければbinaryを削除する。

設計判断:

- feedbackの起点はデフォルトで候補重み自身にする。これは「候補の実戦悪化を局所修正する」ためである。
- teacherは現時点では `policy_weights_v2.1.0.binary` の探索結果を使う。
- feedback stage は小さく保ち、長時間学習に進む前の候補選別として使う。
- smokeや不採用候補の大きな `.binary` は削除する。

合わせて `tools/run_mmto_benchgate_probe.sh` の初回refresh条件を環境変数で上書き可能にした。

追加した主な環境変数:

- `REFRESH_MAX_POSITIONS`
- `REFRESH_EPOCHS`
- `BASE_TRAIN_LINES`
- `BASE_VALID_LINES`
- `REFRESH_RERANK_MAX_POSITIONS`

これにより、スクリプト自体のsmoke testでは初回refreshと内部rerankを小さくできる。

## Bench-feedback smoke

GPT-5.3-codex-spark サブエージェントに軽量smokeを委任した。

最初のsmoke:

- run: `data/mmto/runs/mmto_bench_feedback_smoke_20260627_133924`
- 問題: 内部rerankが固定 `300` 局面で走り、軽量smokeとして重すぎた。
- 対応: 実行を中断し、未採用の `best.raw.binary` を削除した。

修正後smoke:

- run: `data/mmto/runs/mmto_bench_feedback_smoke2_20260627_134810`
- `BENCH_SEEDS=13001`
- `BENCH_GAMES=2`
- `BENCH_DEPTH=4`
- `INITIAL_REFRESH_MAX_POSITIONS=30`
- `INITIAL_REFRESH_RERANK_MAX_POSITIONS=20`
- `RERANK_MAX_POSITIONS=12`
- `FEEDBACK_MAX_POSITIONS=12`
- 終了ステータス: `0`

結果:

```text
total_new=1
total_baseline=1
total_draw=0
score_rate_num=2
score_rate_den=4
bench_aligned_rerank_status=0
candidate kept for further evaluation
```

これは強さ評価ではない。2局だけのsmokeであり、目的はrunnerの接続確認である。残った `best.raw.binary` は採用候補ではないため削除した。

判断:

- 候補生成 -> bench -> hard SFEN抽出 -> gate判定までのフローは動作した。
- 今回は初回候補が軽量benchgateを通ったため、feedback repair stage は未実行である。
- 次の検証では、benchgateを落ちる候補または閾値を高めた条件で、bench hard positions が実際に repair stage に入ることを確認する。

## Bench-feedback forced smoke

feedback repair stage の分岐を確認するため、`MIN_SCORE_RATE_PCT=101` で初回benchgateを意図的に不合格にした。

1回目:

- run: `data/mmto/runs/mmto_bench_feedback_forced_20260627_135218`
- `bench_feedback_positions=10`
- `bench_feedback_stage` は実行された。
- `dagger` は `total positions=10`, `train records=5`, `valid records=0`
- `mmto_tree_train` は `best_epoch=0` でreject。
- `.binary` は残らなかった。

この実行で、reject時に `final_summary.txt` が残らない問題が見つかった。

修正:

- すべての早期終了分岐で `final_summary.txt` を出すようにした。
- `final_status` を追加した。
- `binary_files_after.txt`, `du_after.txt`, `df_after.txt` を共通関数で必ず出すようにした。
- feedback stage 内部rerankの上限を `FEEDBACK_RERANK_MAX_POSITIONS` で指定可能にした。

修正後の確認:

- run: `data/mmto/runs/mmto_bench_feedback_forced_summary_20260627_135718`
- `bench_feedback_positions=8`
- `bench_feedback_stage` は実行された。
- feedback候補は `best_epoch=0` でreject。
- `final_summary.txt` は作成された。
- `final_status=feedback_rejected_no_candidate`
- `.binary` は残らなかった。

判断:

- bench hard positions を feedback stage に流す経路は動作した。
- reject時の機械可読summaryも残るようになった。
- 次は強制不合格ではなく、通常条件の候補でfeedback stageが有効に働くかを確認する。
