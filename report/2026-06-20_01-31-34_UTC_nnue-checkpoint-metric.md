# TinyNNUE checkpoint基準選択

- 作成日時: 2026-06-20 01:31:34 UTC
- ブランチ: `tooling/nnue-checkpoint-metric`
- 目的: rank学習時にvalid RMSEだけでbest checkpointを選ぶと、rank指標の良いepochを取り逃がすため、保存基準を選択可能にする。

## 実装内容

`tools/train_nnue_numpy.py` に `--checkpoint-metric` を追加した。

選択肢:

- `valid_rmse`
- `valid_rank_selected_regret`
- `valid_rank_top1`

通常のvalue JSONLでは従来通り `valid_rmse` を使う。rank JSONLでは、対局前gateに近い `valid_rank_selected_regret` または `valid_rank_top1` を使える。

## 確認

```bash
python3 -m py_compile tools/train_nnue_numpy.py

python3 tools/train_nnue_numpy.py \
  --train /tmp/nnue_rank_taya36_d4_train.jsonl \
  --valid /tmp/nnue_rank_taya36_d4_valid.jsonl \
  --output /tmp/tiny_nnue_ckpt_metric_check.npz \
  --binary-output /tmp/tiny_nnue_ckpt_metric_check.bin \
  --hidden 32 \
  --epochs 4 \
  --batch-size 32 \
  --lr 0.002 \
  --rank-loss-weight 0.002 \
  --rank-temperature-cp 50 \
  --checkpoint-metric valid_rank_selected_regret
```

結果:

```text
best epoch: 3 checkpoint=valid_rank_selected_regret valid_rmse=20.81 valid_mae=17.62 valid_sign=42.19%
```

メタJSON:

```json
{
  "checkpoint": "valid_rank_selected_regret",
  "best_checkpoint_score": 1.01245364375,
  "best_valid_rank_selected_regret": 1.01245364375,
  "best_valid_rank_top1": 0.375
}
```

## 判断

これは学習基盤として採用する。rank学習候補の保存基準を対局前gateに近づけられるため、今後の大きめrankデータ実験で必要になる。
