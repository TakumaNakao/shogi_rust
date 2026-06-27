# MMTO feedback + policy anchor 検証メモ

- 作成日時: 2026-06-27 09:55:31 UTC
- 対象ブランチ: `training/strong-weight-learning-infra`
- 目的: KPP重みを長時間学習へ進める前に、MMTO系目的関数が安定して改善するか確認する。

## 結論

現時点では、単純にepoch数や学習時間を増やすだけで強い重みになる可能性は低い。

理由は、直近のMMTO系実験で train/valid loss は改善しても、rerank gateや対局指標に移らないケースが多いためである。今回の `feedback + policy anchor` は複数splitで score/rerank gate を通過したが、改善幅は小さく、best epoch は1に張り付いた。長時間学習で改善が積み上がるより、副作用が増幅するリスクが高い。

## 実装した変更

### 負数ガード引数の修正

`--best-guard-* -1` がCLI上で壊れる可能性があったため、MMTO系スクリプトの該当引数を `--arg=value` 形式にした。

- commit: `81a3e03 Fix negative MMTO guard arguments`

### 補助損失用Adagrad状態の分離オプション

`mmto_tree_train` に `--separate-aux-adagrad` を追加した。通常のMMTO損失、replay、feedback、policy anchor が同じAdagrad累積状態を共有していると、補助損失の重みを変えても更新量に差が出にくい可能性があったためである。

`tools/run_mmto_from_dump.sh` からは `SEPARATE_AUX_ADAGRAD=1` で有効化できる。

- commit: `bae50e1 Allow separate Adagrad state for MMTO aux losses`

### policy anchor margin loss

`mmto_tree_train` に `--policy-anchor-margin-weight` を追加した。従来のpolicy anchorはbaselineの分布に寄せる補助損失だったが、top手を保護する力が弱かったため、baseline topと競合手のmarginを直接要求する損失を追加した。

`tools/run_mmto_from_dump.sh` からは以下で制御できる。

- `POLICY_ANCHOR_MARGIN_WEIGHT`
- `POLICY_ANCHOR_MARGIN_CP`
- `POLICY_ANCHOR_MARGIN_SOFTPLUS_TEMP_CP`

- commit: `4dfeca0 Add policy anchor margin loss to MMTO training`

### feedback SFEN dedupe

複数のrerank feedback JSONを混ぜると、同一SFENのhard positionが重複して補助損失を過大に支配する可能性がある。そのため `mmto_tree_train` に `--feedback-dedupe-sfen` を追加し、feedback sampleをSFEN単位で重複排除できるようにした。

`tools/run_mmto_from_dump.sh` からは `FEEDBACK_DEDUPE_SFEN=1` で有効化できる。

- commit: `08672b3 Deduplicate MMTO feedback samples by SFEN`

## 実験結果

### feedback + policy anchor 別split再現

既存dump `data/mmto/runs/mmto_loss_top_20260625_142715` の 7k/1k split を使用した。

設定:

- `FEEDBACK_WEIGHT=3.0`
- `POLICY_ANCHOR_WEIGHT=0.05`
- `TEACHER_TOP_CE_WEIGHT=0.5`
- `MAX_WEIGHT_DELTA=0.003`

結果:

- score gate: PASS
- rerank gate: PASS
- baseline: mean=8.36, p90=29.15, p95=43.52, match=49.00%, bad50=0.0380, bad100=0.0030
- candidate: mean=8.25, p90=29.09, p95=43.26, match=49.50%, bad50=0.0370, bad100=0.0030
- feedback violation: 0.3684 -> 0.1053
- policy anchor top_match: 100.00% -> 94.40%

改善はあるが小さい。policy anchorは baseline policy からの乖離を十分に抑えられていない。

### anchor weight 感度確認

同じ7k/1k splitで `POLICY_ANCHOR_WEIGHT=0.02 / 0.05 / 0.10` を比較した。

結果は主要指標が完全一致した。

- candidate mean=8.245475
- p90=29.087257
- p95=43.257140
- match=49.50%
- bad50=0.0370
- bad100=0.0030
- policy anchor top_match=94.40%

この設定では、anchor weight が実験上の有効な調整軸になっていない。

### separate aux Adagrad

`SEPARATE_AUX_ADAGRAD=1` で同じ7k/1k splitを再実行した。

結果:

- score gate: PASS
- rerank gate: PASS
- best_epoch=1
- baseline: mean=8.36, p90=29.15, p95=43.52, match=49.00%, bad50=0.0380, bad100=0.0030
- candidate: mean=8.25, p90=29.09, p95=43.26, match=49.50%, bad50=0.0370, bad100=0.0030
- feedback violation: 0.1053
- policy anchor top_match: 94.40%

Adagrad状態を分離しても改善しなかった。補助損失の干渉だけが原因ではない。

### policy anchor margin loss

同じ7k/1k splitで、通常のpolicy anchorを無効化し、margin型のanchorだけを追加した。

`POLICY_ANCHOR_MARGIN_WEIGHT=0.2` の結果:

- score gate: PASS
- rerank gate: PASS
- policy_anchor_margin top_match: 100.00% -> 95.80%
- baseline: mean=8.36, p90=29.15, p95=43.52, match=49.00%, bad50=0.0380, bad100=0.0030
- candidate: mean=8.25, p90=29.09, p95=43.26, match=49.50%, bad50=0.0370, bad100=0.0030

`POLICY_ANCHOR_MARGIN_WEIGHT=1.0` の結果:

- score gate: PASS
- rerank gate: PASS
- policy_anchor_margin top_match: 100.00% -> 95.90%
- candidate: mean=8.31, p90=29.15, p95=43.52, match=49.30%, bad50=0.0380, bad100=0.0030

margin weightを強めてもtop_matchの改善は小さく、rerankはむしろ悪化した。margin型anchorは単独では採用できない。

### multi-feedback dedupe

10本の過去rerank feedback JSONを混ぜ、`FEEDBACK_DEDUPE_SFEN=1` で同一SFENを除外した。少数hard feedbackへの過適合を避けるため、feedback weightは1.0に抑えた。

設定:

- `FEEDBACK_WEIGHT=1.0`
- `FEEDBACK_DEDUPE_SFEN=1`
- `SEPARATE_AUX_ADAGRAD=1`
- `POLICY_ANCHOR_WEIGHT=0`
- `POLICY_ANCHOR_MARGIN_WEIGHT=0`
- `BEST_METRIC=p95-regret`

結果:

- `RUN_DIR=data/mmto/runs/mmto_multifeedback_dedup_7k_w1_20260627_104704`
- score gate: PASS
- rerank gate: PASS
- best_epoch=1
- baseline feedback: samples=100, loss=94.937523, violation_ratio=0.5600
- epoch10 feedback violation_ratio=0.3800
- final feedback: samples=100, loss=94.887260, violation_ratio=0.4300
- baseline rerank: mean=8.361163, p90=29.151466, p95=43.522550, match=49.00%, bad50=0.0380, bad100=0.0030
- candidate rerank: mean=8.281066, p90=29.100239, p95=43.257140, match=49.30%, bad50=0.0370, bad100=0.0030

参照実験のcandidateは mean=8.245475, p90=29.087257, p95=43.257140, match=49.50%, bad50=0.0370, bad100=0.0030 だった。multi-feedback dedupeはfeedback violationを下げたが、参照実験よりrerank mean/p90/matchが悪く、改善なしと判断する。

## 判断

「もっと長く学習すれば改善する」という仮説は弱い。

今の目的関数では、短い学習で改善する局面はあるが、改善幅が小さく、epochを進めるほどclamp数が増え、best epochが初期に寄る。multi-feedbackでもこの傾向は変わらなかった。これは長時間化よりも目的関数設計の問題を示している。

## 次の実装候補

### 1. policy anchor を制約として扱う

現在のpolicy anchorは補助損失として弱い。margin型anchorも単独では足りなかった。次に進めるなら、単なる補助損失ではなく、offline gateで壊れた局面を明示的にhard replayし、baseline topを落とした局面を優先的に再学習する必要がある。

目的:

- top_match低下を直接抑える。
- anchor weightの感度を明確にする。
- feedbackで悪手を直しつつ、baseline policyの安定領域を壊さない。

### 2. feedback hard positions を学習データから分離する

現在は少数のfeedback JSONを補助損失として流している。multi-feedback dedupeでも改善が足りなかったため、通常データとは別の hard replay set として扱い、各epochで固定回数だけ強制更新する設計が必要になる。

目的:

- rerankで実際に壊れた手を確実に直す。
- 通常loss低下がhard局面の悪化を隠す問題を避ける。

### 3. 長時間学習前の条件

長時間学習へ進む条件は以下。

- 独立splitで rerank mean/p90/p95/match/bad50/bad100 が非悪化。
- anchor top_match低下を明確に抑制できる。
- best_epochが1固定ではなく、複数epochで改善が積み上がる。
- checkpointごとのscore/rerank gateを自動化し、悪化した時点で停止できる。

ただし、現在のMMTO-lite目的関数はこの条件を満たしていない。次の大きな候補は、現行の探索教師fine-tuneではなく、棋譜手と兄弟手を比較するBonanza寄りのroot ranking学習、またはKPP表現自体を超える小型NNUEである。

## 運用メモ

実験で生成された大きな `.binary` は採用候補ではないため削除済み。rootの `policy_weights_v2.1.0.binary` は保持している。
