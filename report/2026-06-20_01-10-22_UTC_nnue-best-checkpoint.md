# TinyNNUE best checkpoint保存

- 作成日時: 2026-06-20 01:10:22 UTC
- ブランチ: `tooling/nnue-best-checkpoint`
- 目的: valid RMSEが最良のepochを保存し、最終epochで悪化した重みを対局ゲートへ回さないようにする。

## 実装内容

`tools/train_nnue_numpy.py` を変更した。

- 各epoch後にvalid RMSEを評価。
- 最小valid RMSEのモデルをメモリ上に保持。
- `.npz` と `--binary-output` にはbestモデルを保存。
- メタJSONに以下を追加。
  - `checkpoint`
  - `best_epoch`
  - `best_valid_rmse`
  - `best_valid_mae`
  - `best_valid_sign`

## 確認

```bash
python3 -m py_compile tools/train_nnue_numpy.py

python3 tools/train_nnue_numpy.py \
  --train /tmp/nnue_taya_depth3_train448.jsonl \
  --valid /tmp/nnue_taya_depth3_valid64.jsonl \
  --output /tmp/tiny_nnue_best_check.npz \
  --binary-output /tmp/tiny_nnue_best_check.bin \
  --hidden 64 \
  --epochs 8 \
  --batch-size 32 \
  --lr 0.002
```

結果:

```text
best epoch: 8 valid_rmse=11.88 valid_mae=7.93 valid_sign=73.44%
```

メタJSON:

```json
{
  "best_epoch": 8,
  "best_valid_rmse": 11.880273795064518,
  "best_valid_sign": 0.734375,
  "checkpoint": "best_valid_rmse"
}
```

## 判断

H=64の初回確認では途中epochが最良になりやすかったため、この変更は対局評価前の必須基盤として採用する。次はTinyNNUEを実戦ベンチに接続する。
