# TinyNNUE学習メトリクス追加とdepth3初回確認

- 作成日時: 2026-06-20 01:08:59 UTC
- ブランチ: `tooling/nnue-training-metrics`
- 目的: NNUEが探索教師値に対して既存KPP静的評価を上回っているか、RMSE/MAE/sign accuracyで見られるようにする。

## 実装内容

`tools/train_nnue_numpy.py` に以下を追加した。

- `--baseline-field`
- train/validのbaseline RMSE
- train/validのbaseline MAE
- train/validのbaseline sign accuracy
- 各epochのtrain/valid sign accuracy

## データ生成

```bash
target/release/nnue_feature_dump \
  --input taya36.sfen \
  --output /tmp/nnue_taya_depth3_512.jsonl \
  --weights policy_weights_v2.1.0.binary \
  --depth 3 \
  --max-positions 512 \
  --jobs 4
```

結果:

```text
records: 512
skipped positions: 0
real 0m5.514s
```

## H=32

```text
baseline valid rmse=16.54 mae=12.22 sign=56.25%
epoch 020 train_rmse=12.30 train_mae=6.57 train_sign=84.15% valid_rmse=12.51 valid_mae=8.90 valid_sign=65.62%
```

純評価速度:

```text
sparse evals/sec: 275722.73
H=32 evals/sec: 485080.71
```

探索profile:

```text
sparse nodes/sec: 266766.33
H=32 nodes/sec: 312482.03
```

## H=64

```text
baseline valid rmse=16.54 mae=12.22 sign=56.25%
epoch 020 train_rmse=11.49 train_mae=6.18 train_sign=75.89% valid_rmse=12.87 valid_mae=9.24 valid_sign=60.94%
```

H=64のbest epochはepoch 16付近:

```text
epoch 016 train_rmse=10.31 train_mae=5.45 train_sign=85.04% valid_rmse=11.55 valid_mae=7.64 valid_sign=71.88%
```

純評価速度:

```text
sparse evals/sec: 275722.73
H=64 evals/sec: 438689.75
```

探索profile:

```text
sparse nodes/sec: 266766.33
H=64 nodes/sec: 291610.75
```

## 判断

depth3 512局面の小規模確認では、H=32/64とも既存KPP `static_eval` よりvalid RMSEが改善した。速度面でも、少なくともこの小型モデルでは即撤退条件に該当しない。

注意:

- これはoffline指標であり、対局強化の証明ではない。
- データが小さく、`taya36.sfen` 由来に強く偏っている。
- H=64はepoch 20より途中epochの方が良く、early stoppingまたはbest checkpoint保存が必要。

次:

1. best checkpoint保存を追加する。
2. benchmarkでTinyNNUEを使えるようにエンジン切替を追加する。
3. H=32/64の40局same-engineゲートを実行する。
