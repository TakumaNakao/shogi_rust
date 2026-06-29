# Hard-node listwise MMTO 実験結果

- 作成日時: 2026-06-28 20:36:05 UTC
- 対象ブランチ: `training/strong-weight-learning-infra`
- 目的: PV sibling dump の候補集合全体を使う listwise/groupwise 学習を検証する

## 実装

`mmto_tree_train --loss-mode listwise-leaf` に hard-node listwise 用の制御を追加した。

- `--listwise-teacher-top-k`
- `--listwise-candidate-top-k`
- `--listwise-min-selected-regret-cp`
- `--listwise-max-teacher-score-abs-cp`
- `--listwise-regret-cap-cp`
- `--listwise-weight-mode none|selected-regret|model-regret`
- `--listwise-weight-scale-cp`
- `--listwise-max-sample-weight`

候補集合は `teacher top-k` と `model top-k` の union とし、teacher最善手を必ず含めるようにした。

## 実験1: 初期listwise実装

- run dir: `data/mmto/runs/pv_sibling_listwise_20260628_200426`
- 入力: `data/mmto/runs/pv_sibling_feedback_20260628_190857`
- 結果: `best_epoch=0`

原因:

初期実装では `teacher top-k` で候補を絞ってから `model top-k` を取っていた。そのため、学習したい「現行モデルが選ぶ悪手」が teacher top-k 外にあると候補集合から落ちていた。

このため、候補集合を `teacher top-k ∪ model top-k` に修正した。

## 実験2: 修正版listwise

- run dir: `data/mmto/runs/pv_sibling_listwise_20260628_201613`
- `BEST_METRIC=valid-loss`
- `LEARNING_RATE=0.00008`
- `MAX_WEIGHT_DELTA=0.02`

結果:

- baseline valid loss: 12.023541
- best valid loss: 12.021881
- best epoch: 1
- score gate: PASS
  - mean abs delta: 0.0321 cp
  - p95 abs delta: 0.0834 cp
  - max abs delta: 0.1359 cp
- rerank gate: FAIL
  - baseline mean/p90/p95: 14.9867 / 43.9966 / 72.6679
  - candidate mean/p90/p95: 15.0572 / 44.3405 / 72.6679
  - baseline bad50/bad100: 0.0900 / 0.0320
  - candidate bad50/bad100: 0.0920 / 0.0320

判断:

valid loss はわずかに改善したが、rerank 指標が悪化したため不採用。

## 実験3: 更新量を弱めた保護寄り設定

- run dir: `data/mmto/runs/pv_sibling_listwise_20260628_202606`
- `BEST_METRIC=bad50-regret`
- `LEARNING_RATE=0.00004`
- `MAX_WEIGHT_DELTA=0.01`
- `TEACHER_TOP_CE_WEIGHT=0.1`
- `CURRENT_TOP_MARGIN_WEIGHT=0.05`
- `LISTWISE_MAX_SAMPLE_WEIGHT=2`

結果:

- best epoch: 9
- score gate: PASS
  - mean abs delta: 0.05 cp
  - p95 abs delta: 0.14 cp
  - max abs delta: 0.23 cp
- rerank gate: FAIL
  - mean/p90/p95 regret が悪化
  - bad50/bad100 が悪化
  - match率も改善しなかった

判断:

更新量を弱めても rerank gate を通過しなかったため、この listwise 単体路線は現設定では不採用。

## 結論

候補集合全体を使う listwise 目的関数は、valid loss や teacher_match を少し動かせる。しかし、現時点では rerank gate を改善できず、実戦に近い探索指標を悪化させる傾向がある。

したがって、次は listwise を主目的にしない。

## 次の方針

PV sibling feedback の単一ペア学習は、100局で `51-44-5` と弱いながら実戦側にプラスを出した。一方、listwise 単体は rerank gate を壊した。

次は以下を試す。

1. pair feedback を主信号に戻す。
2. listwise は補助信号に抑えるか、offline rerank を壊さない anchor としてのみ使う。
3. 強teacher再ラベルは全局面ではなく hard node のみに限定して、depth 4/5 の信号品質を上げる。
4. 10000cp級の外れ値を通常学習から外し、別監視指標として扱う。

採用条件:

- score gate PASS
- rerank mean/p90/p95/bad50/bad100 非悪化
- 20局で悪化なし
- 40局で改善傾向
- 100局で55%以上、または複数seedで安定してbaseline sweepが増えないこと

