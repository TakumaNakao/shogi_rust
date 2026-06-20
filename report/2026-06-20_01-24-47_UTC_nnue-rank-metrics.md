# TinyNNUE rank JSONLメトリクス

- 作成日時: 2026-06-20 01:24:47 UTC
- ブランチ: `tooling/nnue-rank-metrics`
- 目的: `nnue_rank_dump` のJSONLを学習器で読み、同一root内のtop1一致率と選択手regretを対局前に評価できるようにする。

## 実装内容

`tools/train_nnue_numpy.py` を拡張した。

- 通常dumpの `sfen` とrank dumpの `child_sfen` の両方に対応。
- `root_index`, `rank`, `regret` がある場合、root単位のrankingメトリクスを表示。
- メタJSONにbest checkpointのrankingメトリクスを保存。

追加メトリクス:

- `top1`: モデルがroot内で最大評価した子局面がteacher rank 1だった割合。
- `selected_regret`: モデルが選んだ子局面のteacher regret平均。
- `bad_selected`: モデルが選んだ子局面のteacher regretが300cp超だったroot数。

## 確認

構文チェック:

```bash
python3 -m py_compile tools/train_nnue_numpy.py
```

小規模rankデータ:

```bash
head -n 224 /tmp/nnue_rank_taya36_d3_top8.jsonl > /tmp/nnue_rank_taya36_d3_train.jsonl
tail -n 64 /tmp/nnue_rank_taya36_d3_top8.jsonl > /tmp/nnue_rank_taya36_d3_valid.jsonl
```

学習確認:

```bash
python3 tools/train_nnue_numpy.py \
  --train /tmp/nnue_rank_taya36_d3_train.jsonl \
  --valid /tmp/nnue_rank_taya36_d3_valid.jsonl \
  --output /tmp/tiny_nnue_rank_metrics_check.npz \
  --binary-output /tmp/tiny_nnue_rank_metrics_check.bin \
  --hidden 32 \
  --epochs 6 \
  --batch-size 32 \
  --lr 0.002
```

結果:

```text
initial rank train roots=28 top1=42.86% selected_regret=6.71 bad_selected=0
initial rank valid roots=8 top1=25.00% selected_regret=8.20 bad_selected=0
epoch 005 ... valid_top1=75.00% valid_sel_regret=0.54
best epoch: 5 valid_rmse=10.31 valid_mae=6.81 valid_sign=59.38%
```

メタJSON:

```json
{
  "best_valid_rank_roots": 8,
  "best_valid_rank_top1": 0.75,
  "best_valid_rank_selected_regret": 0.5400727375000001,
  "best_valid_rank_bad_selected": 0
}
```

## 判断

rank JSONLを使った学習・評価の最小経路は通った。これはまだ強さ改善ではないが、前回の「offline RMSEは良いが対局で崩壊」を検出するための追加ゲートになる。

次:

1. depth4 rank dumpを生成する。
2. rank metricsで候補を絞る。
3. 候補が `valid_top1` と `selected_regret` の両方で良い場合だけ20局ゲートへ進める。
