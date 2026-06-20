# NumPy小型NNUE学習プロトタイプ

- 作成日時: 2026-06-20 01:01:12 UTC
- ブランチ: `tooling/nnue-train-prototype`
- 目的: `nnue_feature_dump` のJSONLから、小型NNUE風モデルを実際に学習できる最小経路を作る。

## 実装内容

- `tools/train_nnue_numpy.py` を追加。
- PyTorch未導入環境でも動くよう、依存はNumPyのみ。
- モデル形状:
  - `feature_emb[feature]` の和
  - `king_emb[king_bucket]`
  - `material_w * material`
  - hidden bias
  - clipped ReLU
  - 線形出力
- 出力:
  - `.npz`: `feature_emb`, `king_emb`, `material_w`, `hidden_b`, `out_w`, `out_b`
  - `.npz.json`: hidden数、特徴数、target scaleなどのメタ情報

## 確認

構文チェック:

```bash
python3 -m py_compile tools/train_nnue_numpy.py
```

小規模データ生成:

```bash
target/release/nnue_feature_dump \
  --input taya36.sfen \
  --output /tmp/nnue_train_depth2_80.jsonl \
  --weights policy_weights_v2.1.0.binary \
  --depth 2 \
  --max-positions 80 \
  --jobs 4
```

過学習確認:

```bash
python3 tools/train_nnue_numpy.py \
  --train /tmp/nnue_train_depth2_80.jsonl \
  --output /tmp/tiny_nnue_depth2_80.npz \
  --hidden 16 \
  --epochs 8 \
  --batch-size 16 \
  --lr 0.003
```

結果:

```text
train samples: 80
valid samples: 80
hidden: 16
epoch 001 train_rmse=11.81 train_mae=7.39 valid_rmse=11.81 valid_mae=7.39
epoch 008 train_rmse=7.42 train_mae=4.94 valid_rmse=7.42 valid_mae=4.94
```

簡易train/valid分割:

```text
train samples: 64
valid samples: 16
hidden: 16
epoch 001 train_rmse=10.89 train_mae=8.53 valid_rmse=19.74 valid_mae=11.44
epoch 006 train_rmse=6.96 train_mae=4.80 valid_rmse=16.40 valid_mae=7.09
```

## 判断

学習ループと保存形式は機能している。これはまだ強さ改善ではなく、次のRust側NNUE推論器の入力として使うための基盤である。

採用ゲート:

- Rust側で同じ `.npz` 由来の重みを読める、またはRust向け形式へ変換できる。
- 非差分float推論でNPS低下が許容範囲に収まる。
- 既存SparseModel単体に対して、同一探索条件の小規模ベンチで悪化しない。

撤退条件:

- 非差分推論のNPS低下が大きく、補正しても勝率が改善しない。
- 深さの浅い教師値には過学習できるが、検証局面や対局ベンチで一貫して悪化する。
- 特徴形式が大きすぎてCPU実戦探索に載せられない。
