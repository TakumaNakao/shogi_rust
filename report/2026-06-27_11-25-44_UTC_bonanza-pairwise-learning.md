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

## Bench-feedback A/B probe

通常寄りの条件で、候補修正型とbaseline再学習型を比較しようとした。

共通条件:

- `BENCH_GAMES=10`
- `BENCH_DEPTH=8`
- `BENCH_TIME_LIMIT_MS=80`
- 3 seeds
- `MIN_SCORE_RATE_PCT=55`
- `INITIAL_REFRESH_MAX_POSITIONS=100`
- `INITIAL_REFRESH_EPOCHS=2`
- `FEEDBACK_MAX_POSITIONS=80`
- `FEEDBACK_EPOCHS=2`

実験A:

- run: `data/mmto/runs/mmto_bench_feedback_probeA_candidate_20260627_140134`
- `FEEDBACK_START_WEIGHTS=candidate`
- `final_status=no_initial_candidate`
- `.binary` 残存なし

実験B:

- run: `data/mmto/runs/mmto_bench_feedback_probeB_baseline_20260627_140147`
- `FEEDBACK_START_WEIGHTS=baseline`
- `final_status=no_initial_candidate`
- `.binary` 残存なし

初期refreshの状況:

```text
baseline best_metric_score=39.474091
epoch 1 best_metric_score=38.503777
epoch 1 best_guard_passed=false
epoch 2 best_metric_score=38.726536
epoch 2 best_guard_passed=false
best_epoch=0
```

判断:

- A/Bとも初期候補が作れず、feedback stage には進めなかった。
- p95-regret系のbest metric自体は改善しているが、guardで落ちた。
- 具体的には `teacher_match` や `max_regret` のguardが候補化を止めている。
- これは安全側ではあるが、候補生成が狭すぎる可能性がある。
- 比較目的の「候補修正型 vs baseline再学習型」は、この実験では判定不能。

対応:

- `tools/run_mmto_benchgate_probe.sh` の `NO_CANDIDATE` summary に、`best_epoch` と `best_guard` 関連ログを残すようにした。
- 次に試すなら、guardを完全に外すのではなく、`teacher_match` / `max_regret` のどちらが実戦悪化と相関するかを分けて確認する。

## Softguard benchgate probe

GPT-5.5 xhigh の分析では、今は単純に学習時間を伸ばす段階ではなく、まず候補生成gateを二段化するべきと判断された。

理由:

- A/B probeでは p95系metricは改善していた。
- しかし `max_regret` / `bad100` / `teacher_match` のguardがゼロ許容で、候補が外へ出なかった。
- 小規模validでゼロ許容にすると、候補をbenchで落とす前に止まりすぎる。

そこで、最小変更として `tools/run_mmto_benchgate_probe.sh` と `tools/run_mmto_bench_feedback_probe.sh` に best guard slack を環境変数で渡せるようにした。

追加:

- `BEST_GUARD_MAX_REGRET_INCREASE_CP`
- `BEST_GUARD_BAD100_INCREASE`
- `BEST_GUARD_TEACHER_MATCH_DROP_PCT`
- `INITIAL_BEST_GUARD_*`
- `FEEDBACK_BEST_GUARD_*`

実験:

- run: `data/mmto/runs/mmto_softguard_probe_20260627_140949`
- `BEST_GUARD_MAX_REGRET_INCREASE_CP=1.0`
- `BEST_GUARD_BAD100_INCREASE=0.01`
- `BEST_GUARD_TEACHER_MATCH_DROP_PCT=1.0`
- `BENCH_GAMES=10`
- seeds: `12001 12101 12201`

候補生成:

```text
best_epoch=1
best_value=38.503777
score gate: pass
refresh rerank: pass
bench-aligned rerank: pass
```

bench:

```text
seed 12001: 8-2
seed 12101: 2-8
seed 12201: 5-5

total:
  new 15
  baseline 15
  draw 0
  score rate 50.00%
```

判断:

- softguardにより、候補を外へ出してbenchで評価する二段gateは成立した。
- ただしbenchは50%ちょうどで、seed間の揺れが大きい。
- 採用候補ではないため、残った `best.raw.binary` は削除した。
- 次は `MIN_SCORE_RATE_PCT=55` で同系候補を落とし、bench hard positions を feedback repair へ送る。

## Softguard candidate extended validation

`MIN_SCORE_RATE_PCT=55` で同系のsoftguard候補を作ったところ、feedback repair には進まず、初期候補がbenchgateを通過した。

実験:

- run: `data/mmto/runs/mmto_softguard_feedback_probe_20260627_141905`
- candidate: `initial_benchgate/iter_1/best.raw.binary`
- `INITIAL_BEST_GUARD_MAX_REGRET_INCREASE_CP=1.0`
- `INITIAL_BEST_GUARD_BAD100_INCREASE=0.01`
- `INITIAL_BEST_GUARD_TEACHER_MATCH_DROP_PCT=1.0`
- 初回bench: seeds `12001 12101 12201`, 各10局

初回30局:

```text
new 20
baseline 10
draw 0
score rate 66.67%
bench-aligned rerank: pass
```

候補が通ったため、追加検証をGPT-5.3-codex-spark サブエージェントに委任した。

追加80局:

```text
seed 12601: 10-10-0, score 50.00%
seed 12701: 9-10-1, score 47.50%
seed 12801: 13-6-1, score 67.50%
seed 12901: 11-9-0, score 55.00%

total:
  new 43
  baseline 35
  draw 2
  score rate 55.00%
```

110局合算:

```text
new 63
baseline 45
draw 2
score rate 58.18%
```

追加80局の終局理由:

```text
Resign: 77
MaxPliesAdjudication: 1
RepetitionDraw: 2
```

追加80局のpaired starts:

```text
new sweeps: 9
baseline sweeps: 5
splits: 24
draw/mixed: 2
```

record_analyze所見:

- 追加80局でnewの敗局はすべて `BaselineWin` / `Resign`。
- 300cp超のdropが複数ある。
- 最大dropは `seed12901 game_001`, `ply64`, `drop=331`。
- 300cp超は `seed12801 game_006`, `seed12801 game_008`, `seed12901 game_001`, `seed12901 game_002`。

判断:

- これまでの重み学習候補の中では珍しく、追加80局でも50%未満に崩れなかった。
- ただし110局で58.18%は、重み更新を採用・リリースするにはまだ弱い。
- 初回30局の66.67%から優位幅が縮小しており、偶然要素は残る。
- 候補binaryは削除せず、追加検証を続ける。
- 次は標準条件寄り、または別seedで100局以上を追加し、少なくとも合計200局級で55%以上を維持できるか確認する。

## Softguard candidate standard-condition validation

同候補を標準寄りの `depth=5`, `time-limit-ms=100` でも確認した。

実験:

- run: `data/mmto/runs/mmto_softguard_feedback_probe_20260627_141905/validate_std_d5_seed14001`
- candidate: `data/mmto/runs/mmto_softguard_feedback_probe_20260627_141905/initial_benchgate/iter_1/best.raw.binary`
- baseline: `policy_weights_v2.1.0.binary`
- games: `100`
- seed: `14001`
- depth: `5`
- time-limit-ms: `100`

結果:

```text
new 53
baseline 46
draw 1
decisive win rate 53.54%
total score rate 53.50%
95% CI decisive: 43.76%..63.04%
95% CI total: 43.77%..63.23%
```

終局理由:

```text
Resign: 96
MaxPliesAdjudication: 2
PerpetualCheckLoss: 1
RepetitionDraw: 1
```

paired starts:

```text
new sweeps: 6
baseline sweeps: 3
splits: 40
draw/mixed pairs: 1
```

record_analyze所見:

- 300cp超のtail dropが複数ある。
- 最大dropは `game_044_new_white_BaselineWin.usi`, `ply95`, `drop=336`。
- その他、`game_058`, `game_038`, `game_052`, `game_094`, `game_045`, `game_023`, `game_059` などで320cp級drop。
- terminal result mismatchesは0。
- non-terminal score/result sign mismatchesは1。

参考合算:

- 条件違いのため厳密な合算ではない。
- 既存110局: `63-45-2`, score rate `58.18%`
- 標準寄り100局: `53-46-1`, score rate `53.50%`
- 参考合算210局: `116-91-3`, total score rate `55.95%`

判断:

- 標準寄り条件でも50%未満には崩れなかった。
- ただし100局単独ではCIが広く、有意な重み改善とは言えない。
- 候補binaryは保持し、同条件の別seedで追加100局を行う。
- 追加後も55%前後を維持するなら、200局級の標準bench候補として扱う。

## Softguard candidate second standard validation

同じ標準寄り条件で別seedを追加した。

実験:

- run: `data/mmto/runs/mmto_softguard_feedback_probe_20260627_141905/validate_std_d5_seed15001`
- games: `100`
- seed: `15001`
- depth: `5`
- time-limit-ms: `100`

結果:

```text
new 47
baseline 48
draw 5
decisive win rate 49.47%
total score rate 49.50%
95% CI decisive: 39.64%..59.35%
95% CI total: 39.95%..59.05%
```

終局理由:

```text
Resign: 94
MaxPliesAdjudication: 1
RepetitionDraw: 5
```

paired starts:

```text
new sweeps: 8
baseline sweeps: 8
splits: 30
draw/mixed pairs: 4
```

標準条件200局合算:

```text
seed 14001: 53-46-1
seed 15001: 47-48-5

total:
  new 100
  baseline 94
  draw 6
  decisive rate 51.54%
  total score rate 51.50%
  95% CI decisive 44.55%..58.48%
  95% CI total 46.61%..56.36%
```

参考合算:

- 条件違いのため参考扱い。
- 既存110局: `63-45-2`, score rate `58.18%`
- 標準200局: `100-94-6`, score rate `51.50%`

判断:

- 標準条件200局ではほぼ互角まで縮んだ。
- この候補を重み更新として採用・リリースする根拠はない。
- 一方で50%未満に明確崩壊したわけではないため、候補binaryは削除せず、direct bench feedback の素材として残す。
- 次の主眼は、softguard候補そのものではなく、benchで見えた失敗手・大drop局面を直接pair lossへ入れること。

## Direct bench-feedback smoke

標準benchで見えたdrop window局面を使い、`mmto_tree_train --feedback-json` の直接pair loss経路を小さく試した。

目的:

- softguard候補自体を採用するのではなく、bench上の失敗局面から「candidate moveよりteacher moveを上げる」信号を作れるか確認する。

入力:

- source run: `data/mmto/runs/mmto_direct_feedback_probe_20260627_151908`
- `validate_std_d5_seed14001` からdrop windowsをexport: `408` 行
- `validate_std_d5_seed15001` からdrop windowsをexport: `452` 行
- dedupe後: `848` 局面

最初に `max_positions=160`, depth `4/4/6` で `mmto_rerank_gate` を回したが、重すぎたため中断した。軽量条件へ切り替えた。

軽量rerank:

```text
max_positions: 20
depth: baseline/candidate/teacher = 3/3/5
hard_positions: 5
teacher != candidate: 5
rerank gate: pass
```

direct feedback学習:

```text
RUN_DIR=data/mmto/runs/mmto_direct_feedback_probe_20260627_151908/train_teacher_feedback
TRAIN_LINES=400
VALID_LINES=80
EPOCHS=1
FEEDBACK_WEIGHT=0.5
FEEDBACK_GOOD_MOVE=teacher
FEEDBACK_MIN_CANDIDATE_REGRET_CP=15
SEPARATE_AUX_ADAGRAD=1
```

学習結果:

```text
baseline feedback:
  samples=5
  loss=144.767395
  margin_mean=-64.05
  violation_ratio=0.6000
  candidate_regret_mean=99091.77

epoch 1 feedback:
  samples=5
  loss=144.763657
  margin_mean=-64.05
  violation_ratio=0.6000
  candidate_regret_mean=99091.77

best_epoch=1
best_value=30.671322
score gate: pass
rerank gate: pass
```

短ベンチ:

```text
seed 16001
games 20
depth 5
time-limit-ms 100

new 10
baseline 8
draw 2
total score rate 55.00%
95% CI total 34.33%..75.67%
```

判断:

- direct feedback の実装経路は既存コードで動作した。
- ただし今回の信号は5サンプルだけで、feedback lossの改善も極小。
- 20局benchは破綻検知としては通ったが、強さ判断には使えない。
- 次に続けるなら、rerank対象を `20 -> 40/80` に広げる。ただし depth `4/4/6` は重すぎるため、まず `3/3/5` または `3/3/4` でサンプル数を増やす。
- `train_teacher_feedback/best.raw.binary` は追加検証候補として一旦残した。中間の `candidate.raw.binary` は削除した。

## Direct feedback候補の40局追加検証

短ベンチで破綻しなかった direct feedback 候補を、同条件で40局追加検証した。

候補:

- `data/mmto/runs/mmto_direct_feedback_probe_20260627_151908/train_teacher_feedback/best.raw.binary`

追加検証:

```text
record-dir: data/mmto/runs/mmto_direct_feedback_probe_20260627_151908/validate_d5_seed17001_40
seed: 17001
games: 40
depth: 5
time-limit-ms: 100

new: 19
baseline: 18
draw: 3
decisive win rate: 51.35%
total score rate: 51.25%
95% CI decisive: 35.89%..66.55%
95% CI total: 36.35%..66.15%
```

終局理由:

```text
Resign: 37
RepetitionDraw: 3
```

paired starts:

```text
new sweeps: 2
baseline sweeps: 2
splits: 14
draw/mixed: 2
```

record_analyze:

```text
terminal final positions: 37
terminal result mismatches: 0
non-terminal score/result sign mismatches: 0
```

20局smokeとの合算:

```text
games: 60
new: 29
baseline: 26
draw: 5
total score rate: 52.50%
Wilson approx: 40.1%..64.6%
```

判断:

- direct feedback候補は明確な悪化ではないが、採用できる強さではない。
- 60局合算でも信頼区間が広く、v2.1.0重みを置き換える根拠にはならない。
- 今回の学習信号は5サンプルだけで、feedback lossの改善も極小だったため、重みを長く残す価値は低い。
- 次は候補重みを増やす前に、bench失敗局面からより多くのdirect feedbackサンプルを低コストに作る仕組みを整える。
- 容量節約のため、この候補重みは結果記録後に削除した。

## Direct feedbackサンプル生成プローブ

direct feedback候補を増やす前に、bench失敗局面から軽量rerankでどれだけfeedback信号を得られるか測定した。実験はサブエージェントに委任した。

追加した補助スクリプト:

- `tools/run_mmto_direct_feedback_sample_probe.sh`

入力:

```text
baseline weights: policy_weights_v2.1.0.binary
candidate weights: data/mmto/runs/mmto_softguard_feedback_probe_20260627_141905/initial_benchgate/iter_1/best.raw.binary
positions: data/mmto/runs/mmto_direct_feedback_probe_20260627_151908/direct_feedback_input.sfen
depth: baseline/candidate/teacher = 3/3/5
seed: 18001
```

実行:

```text
RUN_DIR=data/mmto/runs/direct_feedback_sample_probe_20260627_161147
MAX_POSITIONS_LIST="40 80"
BASELINE_DEPTH=3
CANDIDATE_DEPTH=3
TEACHER_DEPTH=5
```

結果:

```text
max_positions  status  seconds  samples  hard_positions  teacher_candidate_diff
40             0       46       40       5               5
80             0       55       80       9               9
```

出力サイズ:

```text
max40_d3_3_5.json: 21,408 bytes
max80_d3_3_5.json: 23,271 bytes
run_dir total: about 88 KiB
```

判断:

- depth `3/3/5` なら、80局面でも1分未満で完了し、容量負荷もほぼない。
- hard sample率は `9/80 = 11.25%` 程度で、前回の20局面5件よりは下がったが、全入力848局面へ広げれば数十から100件前後のfeedback候補を得られる可能性がある。
- 次は `MAX_POSITIONS_LIST="200 1000"` で同じ軽量条件を測り、十分なサンプル数が出るか確認する。
- ただしbaseline/candidate平均regretがほぼ同一で、今回のsoftguard候補はv2.1.0から探索選択が大きく変わっていない。feedback学習へ進む場合も、候補生成側を強く揺らす仕組みが必要になる可能性が高い。

追加で全入力近くまで拡大した。

実行:

```text
RUN_DIR=data/mmto/runs/direct_feedback_sample_probe_20260627_161423
MAX_POSITIONS_LIST="200 1000"
BASELINE_DEPTH=3
CANDIDATE_DEPTH=3
TEACHER_DEPTH=5
```

結果:

```text
max_positions  status  seconds  samples  hard_positions  teacher_candidate_diff
200            0       129      200      22              22
1000           0       342      848      113             113
```

`max_positions=1000` は入力848行を全件処理した。両方とも `RERANK GATE PASSED` で、`.binary` は生成していない。

出力サイズ:

```text
max200_d3_3_5.json: 29,281 bytes
max1000_d3_3_5.json: 70,563 bytes
run_dir total: about 152 KiB
```

判断:

- 848局面から113件の `teacher != candidate` pair が得られたため、feedback学習の最低限の信号密度は満たした。
- 所要時間は全件で342秒、実験全体で約8分未満で、日常的に回せる。
- 次は `max1000_d3_3_5.json` を `--feedback-json` に渡し、通常dumpの小さなreplayと組み合わせて1-2 epochだけ学習する。
- ただしcandidateとbaselineの平均regretが同一に見えるため、このfeedbackはsoftguard候補固有の弱点というより、探索同士の局所的な選択差だけを拾っている可能性がある。採用判断はbenchで見る必要がある。

## 学習停滞の分析

GPT-5.5 xhighサブエージェントに、これまでの重み学習が採用候補を作れていない理由と、単純に長く回すべきかを分析させた。

結論:

- 現時点の主因は計算資源不足ではなく、目的関数と実戦で評価されるroot探索後の指し手選択のずれである。
- 固定dumpのlossや静的rerankだけが改善しても、USI経由・時間制限・move ordering込みの対局勝率へ安定して移っていない。
- `score gate` / `rerank gate` は「壊していない」検査には有用だが、「強くなった」証拠としては弱い。
- v2.1.0自身の浅い探索をteacherにした自己蒸留だけを大規模化しても、新しい棋力情報が少ない。

単純にもっと学習を回す方針が期待できる条件:

```text
- best_epochが0/1に張り付かず、複数epochでvalid・hard-valid・rerankが同時に改善する。
- train lossだけでなく、searched rootのmatch、bad50/bad100、p90/p95が独立splitで非悪化になる。
- bench由来hard positionsで、実際のcandidate bad moveをforce includeできている。
- direct feedbackが数件ではなく、少なくとも数十から100件規模の teacher != candidate pair を安定生成できる。
- 60局benchで55%以上、標準条件200局で53-55%以上を維持する。
```

期待しにくい条件:

```text
- fixed dumpのlossだけが下がる。
- best_epochが1に張り付く。
- clamp数が増える。
- rerank meanだけ微改善し、matchやtailが動かない。
- feedback sampleを少数重複で何度も流す。
- v2.1.0 teacherの自己蒸留だけを大量化する。
```

次に実装・検証すべき優先事項:

1. bench/direct counterexample DAggerを学習データの一級入力にする。
   - benchのdrop windowから `sfen`, `candidate_move`, `teacher_move`, `regret`, `source_game`, `ply`, `sample_weight` を持つexplicit pairを作る。
   - teacher move と candidate move を必ず候補に入れ、「teacher move > candidate move」の明示pair lossを通常データとは別replayとして扱う。
   - 成功条件は848 drop windowsから50件以上の有効pair、feedback margin改善、通常valid/hard-valid非悪化、benchgate 60局55%以上、標準200局53%以上。
   - 撤退条件は160局面まで広げても有効pair20件未満、またはfeedbackだけ改善してbenchが50%前後へ戻ること。
2. weighted PV-sibling / Bonanza-rootをgroup-normalized化して補助データとして使う。
   - PV siblingは完全なノイズではないが、単純拡大はtail悪化を起こしやすい。
   - root 1件に対してPV sibling総重みをcapし、root/hard constraintsを満たす中でPV sibling loss最良のcheckpointを選ぶ。

v2.1.0重みの扱い:

- 採用候補作成はv2.1.0 warm-start継続が妥当。
- ゼロ初期化は診断・長期研究用に限定する。
- 巨大疎KPPをscratchから短期で育てるには、データ量・教師探索・目的関数の整備がまだ不足している。
- 今は `freeze_material`, anchor, max-delta, hard replay上限で、v2.1.0から制御されたdelta学習として扱う。

## Direct feedback 113件学習の初回結果

全848局面から得た113件のdirect feedback pairを使い、小規模学習を実行した。実験はサブエージェントに委任した。

実行:

```text
RUN_DIR=data/mmto/runs/mmto_direct_feedback_113_train_20260627_162339
SOURCE_RUN_DIR=data/mmto/runs/mmto_rerank_feedback_5k_gate_w1.0_20260627_074057
TRAIN_LINES=1800
VALID_LINES=200
EPOCHS=2
LOSS_MODE=listwise-leaf
LISTWISE_FEATURE_SOURCE=teacher-leaf
FEEDBACK_JSON=data/mmto/runs/direct_feedback_sample_probe_20260627_161423/max1000_d3_3_5.json
FEEDBACK_WEIGHT=1.0
FEEDBACK_GOOD_MOVE=teacher
BEST_METRIC=p95-regret
```

baseline:

```text
train loss: 5.144132
valid loss: 5.428699
valid p95: 159.35
valid teacher_match: 25.00%
feedback samples: 113
feedback loss: 129.967545
feedback margin_mean: -27.85
feedback violation_ratio: 0.5929
```

epoch推移:

```text
epoch 1:
  train loss: 5.140793
  valid loss: 5.428513
  feedback violation_ratio: 0.5487

epoch 2:
  train loss: 5.134140
  valid loss: 5.414764
  feedback violation_ratio: 0.5221
```

結果:

```text
best_epoch: 0
best_value: 159.350769
score gate: not run
rerank gate: not run
bench: not run
```

判断:

- 通常valid lossとfeedback violationは改善している。
- しかし `BEST_METRIC=p95-regret` ではvalid p95が同値のため、checkpoint選択でbaselineが最良扱いになった。
- これは学習そのものが完全に無効というより、counterexample feedbackを一級目的として採用候補へ残す選択条件が不足している。
- 対策として、`mmto_tree_train` に `BEST_METRIC=feedback-loss` と `BEST_METRIC=feedback-violation` を追加した。
- 通常validの安全性は既存のbest guardで担保しつつ、feedback改善をcheckpoint選択に使えるようにした。
- 次は同じ条件で `BEST_METRIC=feedback-violation` を使い、候補がgateを通るか確認する。

## Direct feedback 113件学習のfeedback-best結果

`BEST_METRIC=feedback-violation` を追加した後、同じ113件feedbackで再実験した。実験はサブエージェントに委任した。

実行:

```text
RUN_DIR=data/mmto/runs/mmto_direct_feedback_113_bestfb_20260627_162755
SOURCE_RUN_DIR=data/mmto/runs/mmto_rerank_feedback_5k_gate_w1.0_20260627_074057
TRAIN_LINES=1800
VALID_LINES=200
EPOCHS=2
LOSS_MODE=listwise-leaf
LISTWISE_FEATURE_SOURCE=teacher-leaf
FEEDBACK_JSON=data/mmto/runs/direct_feedback_sample_probe_20260627_161423/max1000_d3_3_5.json
FEEDBACK_WEIGHT=1.0
FEEDBACK_GOOD_MOVE=teacher
BEST_METRIC=feedback-violation
```

baseline:

```text
valid loss: 5.428699
valid selected_regret_mean: 36.86
valid p90: 119.64
valid p95: 159.35
valid teacher_match: 25.00%
feedback violation_ratio: 0.5929
feedback loss: 129.967545
```

epoch推移:

```text
epoch 1:
  valid loss: 5.428513
  valid p95: 159.35
  valid teacher_match: 25.50%
  feedback violation_ratio: 0.5487

epoch 2:
  valid loss: 5.414764
  valid p95: 159.35
  valid teacher_match: 26.00%
  feedback violation_ratio: 0.5221
```

結果:

```text
best_epoch: 2
best_value: 0.522124
score gate: passed
rerank gate: failed
bench: not run
```

score gate:

```text
mean_abs_delta_cp: 0.0416
p95_abs_delta_cp: 0.0980
max_abs_delta_cp: 0.1493
```

rerank gate:

```text
baseline mean regret: 524.58
candidate mean regret: 524.72
baseline p90 regret: 26.62
candidate p90 regret: 26.83
baseline p95 regret: 42.91
candidate p95 regret: 42.91
match: 43.30% -> 43.30%

failure:
  mean regret worsened by 0.14cp
  p90 regret worsened by 0.21cp
  required mean improvement was not met
```

判断:

- `feedback-violation` をbest metricにしたことで、feedback改善候補をcheckpointとして選べるようになった。
- 通常valid loss、teacher_match、feedback violationはいずれも改善しており、score gateも十分小さいdeltaで通った。
- ただしrerank gateで小幅に悪化したため採用しない。
- `regret_delta_mean=0.00` であることが重要で、今回のfeedbackは「candidateがbaselineより悪いcounterexample」ではなく「baselineとcandidateが同程度に間違うteacher correction」を多く含む。
- そのため、feedbackだけを押すと通常rerankへの副作用が小さく出るが、benchで強くなる根拠はまだ弱い。

次の方針:

- 同じ113件を強く押すのではなく、より保守的に `FEEDBACK_WEIGHT` とepoch数を下げる。
- `RERANK_REQUIRE_MEAN_REGRET_IMPROVEMENT_CP=0` にして、「改善必須」ではなく「非悪化」を見るプローブを1回だけ行う。
- それでもmean/p90が悪化するなら、このfeedback集合はそのままでは採用候補作成に使わない。
- さらに進める場合は、`regret_delta > 0` の候補、つまりcandidate固有の悪化局面を増やす必要がある。

## Direct feedback 113件の保守条件

`BEST_METRIC=feedback-violation` は候補を残せることが分かったため、同じ113件をより弱く押す保守条件を2本試した。実験はサブエージェントに委任した。

### 条件A: feedback weight 0.5 / 1 epoch

実行:

```text
RUN_DIR=data/mmto/runs/mmto_direct_feedback_113_fw05_e1_20260627_163244
FEEDBACK_WEIGHT=0.5
EPOCHS=1
LEARNING_RATE=0.00008
BEST_METRIC=feedback-violation
```

結果:

```text
best_epoch: 1
feedback violation_ratio: 0.592920 -> 0.575221
score gate: passed
rerank gate: failed
bench: not run
```

rerank:

```text
baseline mean: 524.58496
candidate mean: 524.73370
baseline p90: 26.615944
candidate p90: 26.830187
baseline p95: 42.910797
candidate p95: 42.910797
match: 43.30% -> 42.78%
```

判断:

- feedback violationは改善したが、mean/p90/matchが悪化して不採用。
- 候補 `.binary` は削除済み。

### 条件B: feedback weight 0.5 / lr half / 2 epochs

実行:

```text
RUN_DIR=data/mmto/runs/mmto_direct_feedback_113_fw05_lr4e5_e2_20260627_163417
FEEDBACK_WEIGHT=0.5
EPOCHS=2
LEARNING_RATE=0.00004
BEST_METRIC=feedback-violation
```

結果:

```text
best_epoch: 2
feedback violation_ratio: 0.592920 -> 0.575221
score gate: passed
rerank gate: passed
bench: 9-11-0
```

score gate:

```text
mean_abs_delta_cp: 0.02
p95_abs_delta_cp: 0.05
max_abs_delta_cp: 0.08
```

rerank:

```text
baseline mean: 524.58496
candidate mean: 524.58496
baseline p90: 26.615944
candidate p90: 26.615944
baseline p95: 42.910797
candidate p95: 42.910797
match: 43.30% -> 43.30%
```

20局bench:

```text
seed: 19201
games: 20
depth: 5
time-limit-ms: 100

new: 9
baseline: 11
draw: 0
total score rate: 45.00%
```

paired starts:

```text
new sweeps: 1
baseline sweeps: 2
splits: 7
draw/mixed: 0
```

判断:

- 条件Bはoffline gateを通過したが、20局で45%のため不採用。
- 候補 `.binary` は削除済み。
- 113件direct feedbackを単純に押すだけでは、feedback指標の改善が対局強化へ移らない。

総合判断:

- このfeedback集合は `regret_delta_mean=0.00` で、candidate固有の悪化ではなく、baselineとcandidateが同程度に間違う局面を多く含んでいる。
- そのため、教師手へ寄せる信号自体は作れるが、v2.1.0基準の実戦勝率改善にはつながっていない。
- この113件セットをさらに強く・長く学習する方針は打ち切る。
- 次は `regret_delta > 0`、つまり「candidateがbaselineより悪くなった局面」だけを集めるdirect counterexampleに切り替える。

## Direct feedback集合のdelta確認

113件feedbackが対局勝率へ移らなかった理由を確認するため、元JSONの `regret_delta` を集計した。

対象:

```text
data/mmto/runs/direct_feedback_sample_probe_20260627_161423/max1000_d3_3_5.json
```

結果:

```text
hard_positions: 113
regret_delta > 0: 0
regret_delta >= 1: 0
regret_delta >= 10: 0
regret_delta min/max/mean: 0.0 / 0.0 / 0.0
candidate_regret mean: 72127.0
baseline_regret mean: 72127.0
```

判断:

- このfeedback集合は、candidateがbaselineより悪くなった局面ではない。
- baselineとcandidateが同じ悪手を選ぶ局面に対して、teacher手を上げる補正だった。
- v2.1.0からの小さなdelta学習としては、既存baselineの弱点も同時に押すため、勝率改善に直結しにくい。
- 今後のdirect feedback候補は、学習前に `regret_delta > 0` の件数を必ず確認する。
- `tools/run_mmto_direct_feedback_sample_probe.sh` に `delta_gt0`, `delta_ge1`, `delta_ge10` のsummary列を追加した。

## PV sibling group cap

direct feedback 113件の単純学習は打ち切り、次の候補としてPV sibling信号を再検討する。過去の3K weighted PV sibling ultra-safe候補は100局で55.50%を出したが、PV sibling総重みがroot重みを超えやすい問題があった。

実装:

- `mmto_tree_dump` に `--pv-sibling-total-weight-cap` を追加した。
- 指定時は、同じrootから生成されるPV sibling局面の総 `sample_weight` がcapを超えないよう、各PV sibling weightを `min(pv_sibling_sample_weight, cap / sibling_count)` にする。
- 未指定時は従来通り、各PV siblingに固定 `--pv-sibling-sample-weight` を使う。

smoke:

```text
RUN_DIR=data/mmto/runs/pv_sibling_cap_smoke_20260627_164717
max_positions: 50
teacher_depth: 4
student_depth: 3
pv_sibling_max_plies: 2
pv_sibling_sample_weight: 0.25
cap: none vs 0.25
```

結果:

```text
records:
  no_cap:  root 50, pv sibling 200
  cap025:  root 50, pv sibling 200

sample_weight:
  no_cap:  root 1.0 x50, pv sibling 0.25 x200
  cap025:  root 1.0 x50, pv sibling 0.0625 x200

per-root PV sibling total:
  no_cap: max 1.0, mean 1.0
  cap025: max 0.25, mean 0.25
```

判断:

- capは期待通り動作した。
- record数は変えず、PV siblingの総学習圧だけをroot比で抑えられる。
- 次は3K PV sibling dumpをcapありで再生成し、過去ultra-safe条件に近い学習を行って、offline gateと短ベンチを見る。
