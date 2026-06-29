# Protected Feedback 40局ゲートと敗局フィードバック回収結果

- 作成日時: 2026-06-29 01:12:20 UTC
- 対象ブランチ: `training/strong-weight-learning-infra`
- 実験run: `data/mmto/runs/protected_feedback_phasefix_40gate_fb050_i05_20260628_233749`

## 目的

phase-balanced protectionを入れたprotected feedback学習候補が、20局だけでなく40局でも安定して改善するか確認した。
また、不採用時にbench failure miningで失敗局面を回収し、次回学習のfeedback guardへ再利用できるか確認した。

## 結果

offline gateは通過した。

- protection records: 1200
- phase: opening 400 / middle 400 / late 400
- best epoch: 12
- score gate: passed
- rerank gate: passed
- rerank mean regret: baseline 270.11 -> candidate 268.42
- rerank bad100: baseline 0.1178 -> candidate 0.1085

しかし40局ベンチでは悪化した。

- new wins: 16
- baseline wins: 23
- draws: 1
- new total score rate: 41.25%
- new decisive win rate: 41.03%
- paired starts: new sweeps 2 / baseline sweeps 6 / splits 11 / draw-mixed 1

採用条件を満たさないため、候補重みは削除した。

## 敗局フィードバック

不採用時のbench failure miningは正常に動作した。

- mined positions: 184
- raw feedback samples: 26
- raw candidate regret: min 152.57 / mean 91890.41 / max 200000.00

raw feedbackには200000cpの完全な詰みスイング外れ値が含まれていた。
そのため、`bench_failure_feedback`に`--max-timed-regret-cp`を追加し、スクリプト既定値を100000cpにした。

既存の失敗ログを100000cp上限で再変換した結果:

- capped feedback samples: 25
- capped candidate regret: min 152.57 / mean 87566.03 / max 99999.08
- filtered by max regret: 1

## 判断

protected feedback単独ではまだ実戦投入できる重みを作れていない。
ただし、40局ベンチで悪化した候補から具体的な失敗局面を回収し、次回のfeedback guardに混ぜる経路は機能した。

次は、既存のstrong-teacher feedback guardに今回のcapped bench failure feedbackを追加し、offline gateと40局ベンチで悪化候補をより強く弾けるか検証する。
