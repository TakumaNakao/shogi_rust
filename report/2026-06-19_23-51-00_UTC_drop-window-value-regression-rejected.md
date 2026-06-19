# drop-window value regression 棄却

- 日時: 2026-06-19 23:51 UTC
- ブランチ: `exp/drop-window-value-regression`
- 目的: `worst_drop window` 局面を使い、終盤評価急落に強い重みへ微調整できるか確認する。

## データ

`record_analyze --export-drop-windows` で `/tmp/shogi_bench_master_vs_v241_40_seed10101` から drop window を抽出した。

```text
/tmp/shogi_value_drop/drop_windows.sfen: 193
/tmp/shogi_value_drop/normal_train.jsonl: 960
/tmp/shogi_value_drop/normal_valid.jsonl: 240
/tmp/shogi_value_drop/hard_train.jsonl: 154
/tmp/shogi_value_drop/hard_valid.jsonl: 39
```

## 候補1: normal + hard 1x, lr20 e1

```text
baseline valid samples=240 huber=0.000573 rmse_cp=20.31 mae_cp=8.39 corr=0.6630
baseline extra_valid[hard] samples=39 huber=1.048265 rmse_cp=1357.80 mae_cp=774.59 corr=0.6657

epoch valid samples=240 huber=0.000573 rmse_cp=20.30 mae_cp=8.36 corr=0.6631
epoch extra_valid[hard] samples=39 huber=1.048227 rmse_cp=1357.77 mae_cp=774.57 corr=0.6657
```

改善が小さすぎ、regret probe でも候補手が変わらなかったため削除した。

## 候補2: normal + hard 4x, lr100 e1

```text
baseline valid samples=240 huber=0.000573 rmse_cp=20.31 mae_cp=8.39 corr=0.6630
baseline extra_valid[hard] samples=39 huber=1.048265 rmse_cp=1357.80 mae_cp=774.59 corr=0.6657

epoch valid samples=240 huber=0.000571 rmse_cp=20.27 mae_cp=8.34 corr=0.6645
epoch extra_valid[hard] samples=39 huber=1.047477 rmse_cp=1357.02 mae_cp=774.05 corr=0.6663
```

regret gate:

```text
hard windows:
samples: 193
mean_regret_cp: 0.01
max_regret_cp: 0.37
bad_regret_count_gt_300: 0
teacher_move_match: 183 (94.82%)

normal valid:
samples: 240
mean_regret_cp: 0.06
max_regret_cp: 1.48
bad_regret_count_gt_300: 0
teacher_move_match: 203 (84.58%)
```

offline 指標と regret は悪くなかったが、same-engine weight 対戦で悪化した。

```bash
target/release/benchmark \
  --new-weights /tmp/shogi_value_drop/drop_window_hard4x_lr100_e1.binary \
  --baseline-weights policy_weights_v2.1.0.binary \
  --positions taya36.sfen \
  --games 20 \
  --depth 5 \
  --time-limit-ms 100 \
  --max-plies 200 \
  --adjudicate-at-max-plies \
  --seed 13401
```

結果:

```text
new wins: 9
baseline wins: 11
draws: 0
new total score rate: 45.00%
```

## 判断

不採用。候補重みは `/tmp/shogi_value_drop/*.binary` から削除済み。

今回の結果から、少量の drop-window value regression は offline 指標をわずかに改善しても対局勝率に結びつかなかった。重み更新を続けるなら、より大きく質の高い教師データ、またはプロ棋譜・トップAI棋譜由来の policy/value 教師を使う必要がある。

次は探索・速度改善に戻るか、重み更新を行う場合はデータ品質を上げる方向で設計し直す。
