# TinyNNUE listwise rank loss基盤と小規模候補棄却

- 作成日時: 2026-06-20 01:29:29 UTC
- ブランチ: `tooling/nnue-listwise-rank-loss`
- 目的: value MSEだけでは同一root内の手順序が安定しなかったため、root単位のlistwise ranking lossを学習器に追加する。

## 実装内容

`tools/train_nnue_numpy.py` に以下を追加した。

- `--rank-loss-weight`
- `--rank-temperature-cp`
- rootごとのteacher score分布とmodel score分布のsoftmax cross entropy
- value loss更新後に、root group単位のranking更新を追加
- メタJSONにrank loss設定を保存

## 実験データ

前回作成したdepth4 rankデータ:

```text
/tmp/nnue_rank_taya36_d4_top8.jsonl
roots: 36
records: 288
train: 28 roots / 224 records
valid: 8 roots / 64 records
```

## rank-loss-weight=0.02

H=64で試したが、RMSEが大きく暴れて不安定だった。

代表:

```text
epoch 001 valid_rmse=55.83 valid_top1=37.50% valid_sel_regret=1.15
epoch 008 valid_rmse=14.92 valid_top1=25.00% valid_sel_regret=4.79
```

## rank-loss-weight=0.002 / temperature=50cp

H=64:

```text
best epoch: 11
valid_rmse=10.43
valid_sign=75.00%
valid_top1=25.00%
valid_sel_regret=7.68
```

H=96:

```text
best epoch: 15
valid_rmse=14.48
valid_sign=70.31%
valid_top1=62.50%
valid_sel_regret=0.24
```

H=96は小さいvalid setではrank指標が良く見えた。

## root regret確認

H=96候補を `value_regret_probe` で64局面確認した。

条件:

```text
teacher: policy_weights_v2.1.0.binary
candidate: /tmp/tiny_nnue_rank_d4_h96_listwise_w002t50.bin
teacher-depth: 4
candidate-depth: 4
max-positions: 64
```

結果:

```text
mean_regret_cp: 106.59
p50_regret_cp: 34.36
p90_regret_cp: 233.71
p95_regret_cp: 236.88
max_regret_cp: 291.90
bad_regret_count_gt_300: 0 (0.00%)
teacher_move_match: 4 (6.25%)
```

## 判断

listwise rank lossの実装は採用する。root内手順序を直接学習するための基盤として必要である。

ただし、今回のH=64/H=96候補は棄却する。H=96は8 rootのvalidでは良く見えたが、64局面のroot regretで平均106cpと悪く、20局ゲートへ進める根拠がない。

次の方針:

1. 36 rootsではvalidが小さすぎるため、最低でも数百rootのrank dumpへ増やす。
2. 既存のroot valueデータ、rankデータ、敗局tailデータを混合する。
3. rank checkpointをvalid RMSEではなく、`selected_regret` または複合指標で保存できるようにする。

候補重みは削除し、リリース対象にしない。
