# depth 4評価値回帰スモーク

- 作成日時: 2026-06-19 21:47:43 UTC
- ブランチ: `exp/value-regression-d4-taya300`
- 結論: 候補重みは採用しない。通常局面と敗局tailの両立がまだ不十分。

## 実験目的

`value_dump` / `value_train` を使い、現行KPP重みをroot探索評価値へ回帰したときにoffline指標が改善するか確認した。

## データ

### 通常局面

```bash
target/release/value_dump \
  --weights policy_weights_v2.1.0.binary \
  --input taya36.sfen \
  --train-output /tmp/shogi_value_d4_taya300_train.jsonl \
  --valid-output /tmp/shogi_value_d4_taya300_valid.jsonl \
  --depth 4 \
  --max-positions 300 \
  --valid-percent 20 \
  --jobs 4 \
  --seed 9811
```

結果:

```text
train records: 240
valid records: 60
```

分布:

```text
train mean_abs=7.58 max_abs=119.45 ge100=1 ge200=0
valid mean_abs=6.61 max_abs=79.95 ge100=0 ge200=0
```

通常局面だけでは教師値の幅が狭く、value regressionの採否判断には弱い。

### 敗局tail局面

過去の100局ベンチ記録からBaselineWin終盤局面を抽出した。

```bash
target/release/record_analyze \
  --weights policy_weights_v2.1.0.binary \
  --record-dir /tmp/shogi_bench_records_qdelta_vs_v241_5201_100 \
  --tail-plies 12 \
  --export-baseline-win-tails /tmp/shogi_value_hard_tails_qdelta5201.sfen \
  --baseline-win-tail-plies 12
```

抽出:

```text
exported baseline win tail positions: 645
```

このうち400局面をdepth 4教師化した。

```text
hard train records: 320
hard valid records: 80
```

hard側にはmate由来の `100000cp` が含まれるため、`value_train` 側では `score_clip=3000` にクリップして学習した。

## 学習結果

### mixed train

通常240件 + hard320件。

baseline:

```text
mixed valid huber=0.669987 rmse_cp=1054.89 mae_cp=493.58 sign_acc=0.9600 corr=0.7604
random valid huber=0.000239 rmse_cp=13.12 mae_cp=6.26 corr=0.4473
hard valid huber=1.172298 rmse_cp=1395.44 mae_cp=859.06 sign_acc=0.9600 corr=0.7613
```

lr500 / 5epoch:

```text
mixed valid huber=0.659440 rmse_cp=1043.69 mae_cp=486.13 sign_acc=0.9733 corr=0.7626
random valid huber=0.000279 rmse_cp=14.18 mae_cp=9.70 corr=0.4303
hard valid huber=1.153810 rmse_cp=1380.62 mae_cp=843.46 sign_acc=0.9733 corr=0.7631
```

hard側は改善したが、random側が悪化した。

### balanced train

通常局面を4倍にして、通常960件 + hard320件にした。

lr100 / 5epoch:

```text
mixed valid huber=0.667729 rmse_cp=1052.50 mae_cp=491.44 sign_acc=0.9600 corr=0.7611
random valid huber=0.000243 rmse_cp=13.24 mae_cp=6.54 corr=0.4429
hard valid huber=1.168344 rmse_cp=1392.28 mae_cp=855.11 sign_acc=0.9600 corr=0.7620
```

lr500 / 5epoch:

```text
mixed valid huber=0.659578 rmse_cp=1043.75 mae_cp=485.05 sign_acc=0.9733 corr=0.7629
random valid huber=0.000266 rmse_cp=13.84 mae_cp=7.29 corr=0.4249
hard valid huber=1.154063 rmse_cp=1380.69 mae_cp=843.37 sign_acc=0.9733 corr=0.7634
```

通常局面を厚くしてもrandom側の悪化は残った。

## 判断

今回の候補重みは対局ベンチへ進めない。

理由:

- hard局面では改善するが、random extra_validが悪化する。
- 通常局面の教師値が小さく、評価関数全体を安全に動かすにはデータが偏っている。
- まだregret評価がないため、offline改善が探索手の改善に結びつくか判断できない。

## 実装側の反映

`value_train` のデフォルト学習率を `0.02` から `100.0` に変更した。

理由:

- `target = score / 600` のHuber損失では、raw評価値に戻した勾配が小さい。
- `0.02` では3epoch回しても実質的に指標が動かなかった。
- lr100では小さく安定して更新され、lr500ではhard改善がより見える。

## 次の作業

1. 通常局面側を300ではなく1000以上に増やす。
2. hard局面はmate値を含むtailだけでなく、`worst_drop` 周辺の中間局面を使う。
3. `regret_probe` を追加し、候補重みの選ぶ手を教師探索で再採点する。
4. random extra_validが悪化しない候補だけ40局ベンチへ進める。
