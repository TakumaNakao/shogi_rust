# 評価値回帰基盤の追加

- 作成日時: 2026-06-19 21:41:41 UTC
- ブランチ: `value-regression-tooling`
- 目的: 方策蒸留が対局性能に結びつかなかったため、KPP評価関数をroot探索評価値へ回帰する実験基盤を追加する。

## 背景

直近の方策蒸留では、offlineのtop1やCEは改善しても対局では安定して強くならなかった。

- mixed soft distillation: 100局で 49-51
- max teacher gap 80: 40局で 19-20-1

GPT-5.5 xhighによる分析では、方策の手一致を追うより、現行探索のroot評価値を教師にしたKPP value regressionへ移る方が有望と判断された。

## 追加した道具

### `value_dump`

固定局面からroot探索評価値をJSONLへ出力する。

主な出力項目:

- `sfen`
- `teacher_score`
- `depth`
- `legal_moves`
- `pv`

評価値の向きは、局面の手番側視点で保存する。これは `alpha_beta_search(root, ...)` と `SparseModel::predict_from_position(root)` の向きが一致しているため、符号反転しない。

### `value_train`

JSONLの `teacher_score` を教師として、KPP評価値をHuber損失で回帰する。

初期方針:

- `score_clip=3000`
- `target_scale=600`
- `huber_delta=1.0`
- `freeze_material=true`
- `freeze_bias=false`

`material_coeff` は密な1特徴として残差を吸収しやすいため、初期実験では固定する。

## スモーク検証

使用データ:

```bash
taya36.sfen
```

教師生成:

```bash
target/release/value_dump \
  --weights policy_weights_v2.1.0.binary \
  --input taya36.sfen \
  --train-output /tmp/shogi_value_train_smoke.jsonl \
  --valid-output /tmp/shogi_value_valid_smoke.jsonl \
  --depth 2 \
  --max-positions 24 \
  --valid-percent 25 \
  --jobs 4 \
  --seed 9801
```

結果:

```text
train records: 18
valid records: 6
skipped positions: 0
```

dry-run:

```text
baseline train samples=18 huber=0.000080 rmse_cp=7.59 mae_cp=4.22 sign_acc=0.0000 sign_samples=0 corr=0.9169
baseline valid samples=6 huber=0.000234 rmse_cp=12.98 mae_cp=7.40 sign_acc=0.0000 sign_samples=0 corr=0.8532
```

1epoch保存スモーク:

```text
epoch 1 batch_loss=0.000080 material_coeff=0.145648 bias=-0.000000
epoch train samples=18 huber=0.000080 rmse_cp=7.59 mae_cp=4.22 sign_acc=0.0000 sign_samples=0 corr=0.9169
epoch valid samples=6 huber=0.000234 rmse_cp=12.98 mae_cp=7.40 sign_acc=0.0000 sign_samples=0 corr=0.8532
saved /tmp/shogi_value_smoke.binary
```

深さ2・少数局面では教師値が小さく、`|teacher| >= 100cp` の符号一致対象は出なかった。これはスモークとしては問題ないが、実験ではより深い教師または負け棋譜由来局面を含める。

## 採用ゲート

候補重みは、以下を満たすまで採用しない。

1. held-out局面で `valid_huber`, `RMSE(cp)`, `MAE(cp)` がbaselineより改善する。
2. `extra_valid` でrandom局面・敗局tail局面のどちらも悪化しない。
3. `|teacher| >= 200cp` の符号一致率と相関が悪化しない。
4. 候補重みの探索手を教師深さで再採点した平均regretと `regret > 300cp` 率がbaseline以下になる。
5. 40局の現行固定版対戦で55%以上を目安にし、100局以上で再確認する。

## 次の作業

1. depth 4から5の小規模value datasetを作る。
2. random局面と敗局tail局面を分けて `extra_valid` にする。
3. value regression候補を複数学習し、offline gateで落とす。
4. 通過候補だけ40局ベンチへ進める。
