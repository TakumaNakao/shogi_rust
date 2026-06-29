# Search-grounded listwise学習への本線切り替え

作成日時: 2026-06-29 06:34:42 UTC

## 背景

直近のMMTO-lite、Bonanza-root、protected feedback、bench failure DAggerは、offline lossや一部rerank指標を改善しても、対局ベンチで採用可能な重みを安定して作れていない。

GPT-5.5 xhighサブエージェントにも検討を依頼した結論は、今の棄却済み系統を単純に長く回すだけでは改善確率が低い、というものだった。理由は、学習時間よりも「目的関数と実探索での手選択改善の相関」が不足しているためである。

## 判断

単純にepoch数や学習時間を伸ばすのではなく、長時間学習の入口を以下へ寄せる。

- `listwise-leaf` を本線にする。
- teacher score上位手と、現在モデルが高く見ている手を同じ候補集合に入れる。
- 現在モデルが選びやすい高regret手をhard negativeとして押し下げる。
- checkpoint選択はvalid lossではなく、`capped-selected-regret` など探索選択に近い指標を優先する。
- 長時間runはmanifestで入力、重み、git状態を復元可能にしてから実行する。

## 実装

`tools/run_mmto_from_dump.sh` を、既存dump再利用の長時間学習候補として更新した。
また、誤って旧pairwise既定の長時間runを起動しないように、標準の `tools/run_mmto_rerank_pipeline.sh` も同じ既定値へ揃えた。

主な既定値:

- `LOSS_MODE=listwise-leaf`
- `LISTWISE_TEACHER_TOP_K=16`
- `LISTWISE_CANDIDATE_TOP_K=16`
- `LISTWISE_MIN_SELECTED_REGRET_CP=30`
- `LISTWISE_WEIGHT_MODE=model-regret`
- `TEACHER_TOP_CE_WEIGHT=0.1`
- `CURRENT_TOP_MARGIN_WEIGHT=0.05`
- `BEST_METRIC=capped-selected-regret`
- `STREAM_TRAIN=1`

また、`tools/write_training_manifest.py` を追加し、各runで `manifest.json` を保存できるようにした。

manifestに記録する内容:

- git branch / commit / dirty状態
- run種別と主要パラメータ
- 入力dump、subset、score positionsのsha256と行数
- 初期重み、teacher重みのsha256
- 実行環境の基本情報

## 長時間学習へ進める条件

短時間probeで以下を満たすまでは、学習時間だけを伸ばさない。

- `best_epoch > 0`
- `mmto_score_gate` 通過
- `mmto_rerank_gate` で mean / p90 / p95 / bad50 / bad100 が非悪化
- teacher matchが低下しない
- 20局smokeで明確な悪化がない
- 40局以上で少なくとも非悪化

長く回す価値があるのは、loss改善ではなく「実探索で悪手を選びにくくなる」兆候が複数splitで確認できた構成だけである。

## 次の作業

1. 既存の100k/20k dumpから、5k/500と9k/1kの独立splitで `run_mmto_from_dump.sh` を実行する。
2. manifest、train log、score gate、rerank gateを比較する。
3. 2splitとも非悪化なら、10k級の数時間runへ進む。
4. そこで通過した構成だけ、24-48時間runの候補にする。

## Smoke結果

`GPT-5.3-codex-spark` サブエージェントに、既存PV sibling dumpを使った小さなfrom-dump smokeを依頼した。

条件:

- `SOURCE_TRAIN=data/mmto/runs/pv_sibling_feedback_20260628_190857/train.pv.tree.jsonl`
- `SOURCE_VALID=data/mmto/runs/pv_sibling_feedback_20260628_190857/valid.pv.tree.jsonl`
- `TRAIN_LINES=1000`
- `VALID_LINES=200`
- `EPOCHS=2`
- `STREAM_TRAIN=1`

結果:

- `RUN_DIR=data/mmto/runs/listwise_from_dump_smoke_20260629_063704`
- `best_epoch=2`
- `mmto_score_gate`: pass
- `mmto_rerank_gate`: fail
- 失敗理由: mean regretの改善幅が要求値0.5cpに届かなかった。
- manifestは生成され、source dump、subset、score positions、重みsha256、行数が記録された。
- 不採用のためraw binaryは残っていない。

このsmokeは、実装とmanifest出力が壊れていないことを確認するものとしては成功。ただし、長時間学習へ進める改善信号としては不足している。

標準pipeline側も小さくsmokeした。

条件:

- `MAX_POSITIONS=120`
- `VALID_PERCENT=20`
- `POSITION_CHUNK_SIZE=16`
- `JOBS=2`
- `TEACHER_DEPTH=2`
- `STUDENT_DEPTH=1`
- `EPOCHS=1`

結果:

- `RUN_DIR=data/mmto/runs/pipeline_listwise_smoke_20260629_064214`
- ログ上で `LOSS_MODE=listwise-leaf`、`STREAM_TRAIN=1`、`LISTWISE_TEACHER_TOP_K=16`、`LISTWISE_CANDIDATE_TOP_K=16` を確認。
- manifestは生成され、train 96行、valid 24行、初期/teacher重みのsha256が記録された。
- `mmto_rerank_gate` は改善要求0.5cpに届かずfail。
- 不採用のためraw binaryは残っていない。

このsmokeも、pipelineが新しい既定値で動くことを確認する目的としては成功。強さ改善の根拠には使わない。
