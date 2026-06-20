# 小型NNUEバイナリexport

- 作成日時: 2026-06-20 01:03:30 UTC
- ブランチ: `tooling/nnue-export-binary`
- 目的: NumPyで学習した小型NNUEプロトタイプを、Rust推論器が読みやすい単純なバイナリ形式へ出力する。

## 実装内容

`tools/train_nnue_numpy.py` に `--binary-output` を追加した。

バイナリ形式:

- magic: `TNNUE001`
- header: little-endian `<IIIIf>`
  - format version
  - hidden size
  - feature count
  - king bucket count
  - target scale
- body: little-endian float32 row-major
  - `feature_emb`
  - `king_emb`
  - `material_w`
  - `hidden_b`
  - `out_w`
  - `out_b`

## 確認

```bash
python3 -m py_compile tools/train_nnue_numpy.py

python3 tools/train_nnue_numpy.py \
  --train /tmp/nnue_train_depth2_80.jsonl \
  --output /tmp/tiny_nnue_export_check.npz \
  --binary-output /tmp/tiny_nnue_export_check.bin \
  --hidden 8 \
  --epochs 2 \
  --batch-size 16 \
  --lr 0.003
```

結果:

```text
wrote model: /tmp/tiny_nnue_export_check.npz
wrote meta: /tmp/tiny_nnue_export_check.npz.json
wrote binary model: /tmp/tiny_nnue_export_check.bin
```

magic確認:

```text
b'TNNUE001'
```

## 次

次ブランチでRust側の非差分float推論器を追加し、`eval_profile` と `search_profile` で速度低下を測る。xhigh分析の採用ゲートに従い、NPS低下が15%以上ならこの路線はいったん停止する。
