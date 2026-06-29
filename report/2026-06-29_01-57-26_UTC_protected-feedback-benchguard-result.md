# Bench Failure Guard付きProtected Feedback検証結果

- 作成日時: 2026-06-29 01:57:26 UTC
- 対象ブランチ: `training/strong-weight-learning-infra`
- 実験run: `data/mmto/runs/protected_feedback_benchguard_fb050_i05_20260629_011337`

## 目的

前回40局で悪化したprotected feedback候補から抽出したbench failure feedbackを、次回の`FEEDBACK_GUARD_JSON`へ追加する。
これにより、offline gateでは良く見えるが実戦で悪化する候補を弾けるか確認した。

追加したguard:

- `data/mmto/runs/pv_sibling_strong_teacher_filtered_20260628_212449/feedback_guard.json`
- `data/mmto/runs/protected_feedback_phasefix_40gate_fb050_i05_20260628_233749/bench_failure_feedback_capped.json`

## 実験条件

- `FEEDBACK_WEIGHT=0.50`
- `INCUMBENT_PROTECTION_WEIGHT=0.05`
- `LOSS_MODE=aux-only`
- `BENCH_GAMES=40`
- `BENCH_SEED=37601`
- `KEEP_MIN_NEW_WINS=22`
- bench failure feedback上限: 100000cp

## 結果

学習とoffline gateは通過した。

- feedback samples: train 679 / eval 251 / guard json 2
- best epoch: 12
- baseline feedback loss: 100.771881
- best value: 100.770836
- score gate: passed
- rerank gate: passed

しかし40局ベンチでは悪化した。

- new wins: 18
- baseline wins: 22
- draws: 0
- new decisive win rate: 45.00%
- new total score rate: 45.00%
- end reasons: Resign 40
- paired starts: new sweeps 1 / baseline sweeps 3 / splits 16

採用条件を満たさないため、候補重みは削除した。

## 補足

実験中に同条件の重複run `data/mmto/runs/protected_feedback_benchguard_fb050_i05_20260629_015125` が開始された。
これは中断し、残っていた中間重みも削除した。

`011337` runは40局ベンチまでは完了したが、拒否後のbench failure miningは完了マーカー作成前に止まった。
`bench_failure.jsonl`は途中生成の50行のみ残っているため、次回の正式なguardには使わない。

## 判断

bench failure feedbackをguardに追加しても、今回の設定ではoffline評価と実戦結果のズレを十分に埋められなかった。

次の改善では、単に失敗局面を追加するだけでなく、以下のどちらかを優先する。

1. `best_metric`をfeedback lossではなく、実戦相関の高いrerank悪化指標に寄せる。
2. 候補採用前のgateに小規模ベンチ相当の「実探索での手選択変化」検査を入れる。

現時点ではこの系統の重み更新は採用しない。
