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
