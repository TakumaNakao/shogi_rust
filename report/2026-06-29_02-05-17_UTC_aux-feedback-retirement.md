# Aux-only Feedback系の検証打ち切り判断

- 作成日時: 2026-06-29 02:05:17 UTC
- 対象ブランチ: `training/strong-weight-learning-infra`

## 結論

`aux-only` protected feedback微調整を本線から外す。
単純に学習時間を伸ばすのではなく、`listwise-leaf`の教師分布蒸留と実探索由来hard negativeを本線にする。

## 根拠

直近の`aux-only`系はoffline指標では改善しても、40局ベンチで採用できなかった。

- phase-balanced protected feedback:
  - 20局: 13-6-1
  - 40局: 19-18-3 または 16-23-1
- bench failure guard追加:
  - score gate: passed
  - rerank gate: passed
  - 40局: 18-22、total score rate 45.00%
- `bad100-regret`をbest metricにした検証:
  - baseline best metric: 0.438799
  - 学習後候補: 0.441109まで悪化
  - best_epoch=0で不採用

feedback lossやviolationはわずかに改善しているが、実戦勝率に結びついていない。
これは学習時間不足より、目的関数と実戦強さの相関不足が主因と判断する。

## 次の方針

1. `mmto_tree_train --loss-mode listwise-leaf`を本線化する。
2. teacher score上位候補とcurrent model上位候補を同じ候補集合に入れ、teacher分布へのlistwise CEで学習する。
3. current modelが選びやすい高regret手をhard negativeとして直接押し下げる。
4. checkpoint選択は`feedback-loss`ではなく、`bad100-regret`または`capped-selected-regret`を優先する。
5. bench failureはguard追加ではなく、実探索で選ばれた悪手をteacherで再ラベルしたtraining shardとして使う。

## しばらく避ける実験

- `FEEDBACK_WEIGHT`や`INCUMBENT_PROTECTION_WEIGHT`だけの微調整
- `aux-only`のepoch延長
- bench failure feedbackをguardへ追加するだけの実験

これらは既に実戦相関不足が見えており、同じ方向へ計算資源を使う優先度は低い。
