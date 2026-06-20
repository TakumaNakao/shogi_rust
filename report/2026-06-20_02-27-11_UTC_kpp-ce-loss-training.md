# KPP教師あり学習のCE loss対応

- 作成日時: 2026-06-20 02:27:11 UTC
- ブランチ: `training/kpp-learn-ce-loss`
- 対象: `src/kpp_learn.rs`

## 背景

GPT-5.5 xhighサブエージェント `Mencius` に、v2.1.0時代のKPP教師あり学習アルゴリズム改善を検討させた。

主な結論:

- 現行の `update_batch_for_moves` は、モデル最善手が教師手と違う時だけ特徴差分を足し引きするmargin/perceptron型更新。
- 正解時は更新されず、不正解時も確信度や合法手全体の分布を使わない。
- まず `kpp_learn` に温度付きcross entropyを選択可能にするのが最優先。
- material係数は初期実験ではfreeze推奨。
- 棋譜教師あり学習を主軸にし、探索蒸留やvalue回帰は後段の微調整に回す。

## 実装

`kpp_learn` に以下を追加した。

```text
--loss margin|ce
--softmax-temperature <f32>
```

`margin` は従来方式。`ce` は既存の `SparseModel::update_batch_with_cross_entropy_temperature` を使う。

CSVログには以下を追加した。

```text
loss_mode
train_loss
valid_ce
```

validation表示も top1 accuracy に加えて CE を出す。

## スモーク検証

`/tmp/shogi_kpp_smoke_csa` の最小CSA 2局で、`margin` と `ce` の両方を実行した。生成された一時重みは削除済み。

共通条件:

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/kpp_learn \
  --input-dir /tmp/shogi_kpp_smoke_csa \
  --weights /home/nami_ride_trade/shogi_rust/policy_weights_v2.1.0.binary \
  --epochs 1 \
  --batch-size 4 \
  --chunk-size 1 \
  --valid-percent 50 \
  --valid-max-files 1 \
  --seed 42 \
  --checkpoint-every-batches 1 \
  --freeze-material \
  --no-graph \
  --softmax-temperature 600
```

`margin` ログ:

```text
epoch,batch,loss_mode,train_loss,train_accuracy,train_correct,train_total,valid_ce,valid_accuracy,valid_correct,valid_total,material_coeff,min_w,max_w
1,1,Margin,0.000000,25.0000,1,4,3.586798,33.3333,2,6,0.145648,-2.950039,1.762825
1,2,Margin,0.000000,50.0000,1,2,3.587995,0.0000,0,6,0.145648,-2.950036,1.762824
```

`ce` ログ:

```text
epoch,batch,loss_mode,train_loss,train_accuracy,train_correct,train_total,valid_ce,valid_accuracy,valid_correct,valid_total,material_coeff,min_w,max_w
1,1,Ce,3.525620,25.0000,1,4,3.589370,0.0000,0,6,0.145648,-2.950040,1.762827
1,2,Ce,3.448618,50.0000,1,2,3.589360,16.6667,1,6,0.145648,-2.950040,1.762827
```

## 検証コマンド

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin kpp_learn
```

両方通過。

## 長時間学習への推奨

初回は従来 `margin` と新 `ce` を同じCSAコーパス・同じseed・同じvalidation splitで比較する。

推奨小実験:

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/kpp_learn \
  --input-dir /data/shogi/csa/2016 \
  --input-dir /data/shogi/csa/2017 \
  --weights /home/nami_ride_trade/shogi_rust/policy_weights_v2.1.0.binary \
  --output /tmp/policy_weights_kpp_ce_t600_lr005_seed20260620.binary \
  --epochs 1 \
  --batch-size 1024 \
  --chunk-size 1024 \
  --learning-rate 0.05 \
  --l2-lambda 0.00001 \
  --loss ce \
  --softmax-temperature 600 \
  --valid-percent 5 \
  --valid-max-files 512 \
  --seed 20260620 \
  --checkpoint-dir /tmp/kpp_ce_t600_lr005_seed20260620_checkpoints \
  --checkpoint-every-batches 100 \
  --log-path /tmp/kpp_ce_t600_lr005_seed20260620.csv \
  --freeze-material
```

比較候補:

```text
loss=ce temperature=100 lr=0.05
loss=ce temperature=200 lr=0.05
loss=ce temperature=600 lr=0.05
loss=ce temperature=600 lr=0.1
loss=margin lr=0.1
```

## 採用ゲート

1. `valid_ce` がbaselineより改善。
2. `valid_accuracy` が大きく悪化しない。
3. `--freeze-material` 時にmaterial係数が維持される。
4. `kpp_weight_check` でNaN/infなし。
5. `search_profile` でNPSや探索ノードが破綻しない。
6. 候補重み vs v2.1.0重みを同一エンジンで40局。
7. 40局で悪化しない候補だけ100局以上。

短い10局では採用しない。
