# TinyNNUE residual target基盤

- 作成日時: 2026-06-20 01:38:40 UTC
- ブランチ: `tooling/nnue-residual-target`
- 目的: TinyNNUE単体でKPPを置き換える実験が対局で崩壊したため、KPP評価を土台にした残差補正を学習できるようにする。

## 実装内容

`tools/train_nnue_numpy.py` に `--target-offset-field` を追加した。

例:

```text
--target-field teacher_score --target-offset-field static_eval
```

この場合、学習targetは以下になる。

```text
teacher_score - static_eval
```

これにより、TinyNNUEは評価全体ではなく、既存KPP評価からの補正だけを学習できる。

## 確認

```bash
python3 -m py_compile tools/train_nnue_numpy.py

python3 tools/train_nnue_numpy.py \
  --train /tmp/nnue_taya_depth4_train448.jsonl \
  --valid /tmp/nnue_taya_depth4_valid64.jsonl \
  --output /tmp/tiny_nnue_residual_check.npz \
  --binary-output /tmp/tiny_nnue_residual_check.bin \
  --hidden 64 \
  --epochs 12 \
  --batch-size 32 \
  --lr 0.002 \
  --target-field teacher_score \
  --target-offset-field static_eval \
  --baseline-field ''
```

結果:

```text
best epoch: 6
valid_rmse=7.91
valid_mae=5.10
valid_sign=67.19%
```

## 判断

残差targetは学習可能。次はRust側にKPP + TinyNNUE residual evaluatorを追加し、補正係数を小さくした安全な候補を試す。

注意:

- 今回の確認モデルは採用候補ではない。
- 対局評価前に、補正係数別のroot regretとsearch_profileを確認する。
