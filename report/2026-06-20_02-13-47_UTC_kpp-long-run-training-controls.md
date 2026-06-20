# KPP長時間学習制御の整備

- 作成日時: 2026-06-20 02:13:47 UTC
- ブランチ: `training/kpp-learn-long-run-controls`
- 対象: `src/kpp_learn.rs`

## 背景

v2.1.0の重みはCSA棋譜を使った長時間の教師ありKPP学習で作られている。直近の探索改善、TinyNNUE単体、policy蒸留、hybrid residualでは大きな伸びが出ていないため、次の大きな改善候補として「v2.1.0時代の学習方法を安全に長時間回せる形へ強化する」方針に切り替える。

ただし、過去に自己対局学習で評価関数を壊した経緯があるため、いきなり採用候補重みとして長時間学習を開始しない。まず学習プログラムを再現可能・中断復帰可能・検証可能にする。

## 変更内容

`kpp_learn` を年指定の固定実行から、長時間学習向けCLIへ変更した。

追加した主な制御:

- 複数 `--input-dir` からCSAを読み込む
- `--weights` と `--output` を分離
- `--epochs`
- `--batch-size`
- `--chunk-size`
- `--learning-rate`
- `--l2-lambda`
- `--seed`
- `--valid-percent`
- `--valid-max-files`
- `--checkpoint-dir`
- `--checkpoint-every-batches`
- `--log-path`
- `--freeze-material`
- `--no-graph`

学習アルゴリズム自体は、まずv2.1.0系のプロ棋譜指し手学習を維持した。今回の目的は、長時間学習を安全に実行するための制御と観測を追加すること。

## スモーク検証

リポジトリには実戦用CSAコーパスが見つからなかったため、`/tmp/shogi_kpp_smoke_csa` に最小CSAを2局作成して検証した。

実行コマンド:

```bash
rm -rf /tmp/kpp_learn_smoke_out
mkdir -p /tmp/kpp_learn_smoke_out/checkpoints

env RUST_FONTCONFIG_DLOPEN=1 target/release/kpp_learn \
  --input-dir /tmp/shogi_kpp_smoke_csa \
  --weights /home/nami_ride_trade/shogi_rust/policy_weights_v2.1.0.binary \
  --output /tmp/kpp_learn_smoke_out/policy_weights_smoke.binary \
  --epochs 1 \
  --batch-size 4 \
  --chunk-size 1 \
  --valid-percent 50 \
  --valid-max-files 1 \
  --seed 42 \
  --checkpoint-dir /tmp/kpp_learn_smoke_out/checkpoints \
  --checkpoint-every-batches 1 \
  --log-path /tmp/kpp_learn_smoke_out/kpp_learn_smoke.csv \
  --freeze-material \
  --no-graph
```

結果:

```text
baseline validation accuracy: 0.00% (0/6)
epoch 1 batch 1: 正解率: 25.00%
epoch 1 最後のバッチ 2: 正解率: 50.00%
epoch 1 validation accuracy: 0.00% (0/6)
checkpoint generated: policy_weights_epoch001_batch0000001.binary
checkpoint generated: policy_weights_epoch001_batch0000002.binary
final output generated: policy_weights_smoke.binary
```

CSVログ:

```text
epoch,batch,train_accuracy,train_correct,train_total,valid_accuracy,valid_correct,valid_total,material_coeff,min_w,max_w
1,1,25.0000,1,4,33.3333,2,6,0.145648,-2.950039,1.762825
1,2,50.0000,1,2,0.0000,0,6,0.145648,-2.950036,1.762824
```

スモーク用の一時重みは削除済み。

## 長時間学習コマンド案

CSAコーパスを、例えば `/data/shogi/csa/2016`、`/data/shogi/csa/2017` のように配置してから実行する。

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin kpp_learn

env RUST_FONTCONFIG_DLOPEN=1 target/release/kpp_learn \
  --input-dir /data/shogi/csa/2016 \
  --input-dir /data/shogi/csa/2017 \
  --weights /home/nami_ride_trade/shogi_rust/policy_weights_v2.1.0.binary \
  --output /tmp/policy_weights_longrun_kpp_seed20260620.binary \
  --epochs 1 \
  --batch-size 1024 \
  --chunk-size 1024 \
  --learning-rate 0.1 \
  --l2-lambda 0.00001 \
  --valid-percent 5 \
  --valid-max-files 512 \
  --seed 20260620 \
  --checkpoint-dir /tmp/kpp_longrun_checkpoints_seed20260620 \
  --checkpoint-every-batches 100 \
  --log-path /tmp/kpp_longrun_seed20260620.csv \
  --freeze-material
```

初回の長時間学習では `--freeze-material` を推奨する。v2.1.0重みのmaterial係数を壊さず、KPP重みだけを棋譜指し手へ寄せるため。

## 採用ゲート

長時間学習後の候補重みは、最低限以下を通す。

1. `kpp_weight_check` でNaN/inf、ファイルサイズ、material係数を確認。
2. `search_profile` で探索速度とノード傾向を確認。
3. 現行固定版同士で候補重み vs v2.1.0重みを40局。
4. 40局で悪化しない候補だけ100局。
5. v2.4.1 baseline比較で100局以上。

短い10局だけでは採用しない。

## 注意

このPC上では実戦用CSAコーパスは見つからなかった。

```text
/home/nami_ride_trade 以下で見つかった .csa は csa crate のfixtureのみ
```

したがって、実際の長時間学習にはCSAコーパスの配置が必要。重みファイルは `.gitignore` 対象なので、生成された候補重みは `/tmp` などに置き、採用時だけ手動でGitHub Release assetとしてアップロードする。
