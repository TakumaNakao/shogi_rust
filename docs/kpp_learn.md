# KPP supervised training

`kpp_learn`はCSA棋譜からsparse KPP modelを学習するsupported training toolである。
棋譜収集、split、勝敗・手番解釈には`training_data`の共通規約を使う。

## Build

```bash
RUST_FONTCONFIG_DLOPEN=1 cargo build --release --features training-tools --bin kpp_learn
target/release/kpp_learn --help
```

## Initialization

- `--init-mode auto`: `--weights`があればloadし、なければ新規modelを作る。
- `--init-mode load`: 既存weightを必須とする。
- `--init-mode scratch`: weightを読まず、`--scratch-material-coeff`から始める。

scratchでbaselineを上書きしないよう、`--output`は`--weights`と別pathにする。既存modelの
継続学習は`load`を使い、入力weightのhashをrun記録へ残す。

## Loss and filters

- `--loss margin`: 棋譜手とmodel最善手のmarginを学習する従来方式。
- `--loss ce`: 合法手softmaxのcross entropy。`--softmax-temperature`で尺度を指定する。
- `--freeze-material`: material係数を固定しKPPだけを更新する。
- `--decisive-only`, `--winner-only`, `--min-player-rate`,
  `--exclude-loser-after-ply`, `--loser-sample-rate`: 共通CSA metadataに基づくfilter。

filterと乱択は`--seed`を固定する。同条件比較では入力file集合、weight、seed、filterをすべて
同じにする。

## Example

```bash
RUST_FONTCONFIG_DLOPEN=1 target/release/kpp_learn \
  --init-mode load \
  --input-dir data/wdoor/extract/2026 \
  --weights ./policy_weights_v2.1.0.binary \
  --output runs/kpp-wdoor-2026/policy_weights.binary \
  --loss ce \
  --softmax-temperature 150 \
  --epochs 1 \
  --batch-size 2048 \
  --load-file-batch-size 256 \
  --valid-percent 5 \
  --valid-max-files 500 \
  --seed 20260620 \
  --checkpoint-dir runs/kpp-wdoor-2026/checkpoints \
  --checkpoint-every-batches 200 \
  --freeze-material \
  --min-player-rate 4000 \
  --decisive-only \
  --exclude-loser-after-ply 100 \
  --no-graph
```

`--load-file-batch-size`は同時に局面化するCSA file数を制限する。メモリが少ない環境では
256以下から開始する。長時間runは`runs/`以下に隔離し、最終weight、checkpoint、引数、seed、
入力hashとvalidation結果を一緒に保存する。artifact配置の正本は
[`artifact_policy.md`](artifact_policy.md)である。
