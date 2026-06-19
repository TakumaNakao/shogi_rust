# distill_train extra valid 表示

- 作成日時: 2026-06-19 21:17:40 UTC
- ブランチ: `tooling/distill-extra-valid`
- 目的: 蒸留学習時に random valid と hard valid を同じコマンドで分離監視できるようにする。

## 背景

hard-only、static hard、mixed蒸留はいずれも対局ゲートを通らなかった。

主因の一つは、offline指標が改善しても、それが特定bucketだけの改善なのか、通常局面を壊していないのかを毎回別コマンドで確認していたことにある。

今後の教師データ基盤では、以下を標準にする。

```text
main valid: 混合valid
extra valid random: 通常局面
extra valid hard: 敗局・評価急落局面
```

## 実装

`distill_train` に `--extra-valid LABEL=PATH` を追加した。

複数指定できる。

```bash
--extra-valid random=/path/to/random_valid.jsonl \
--extra-valid hard=/path/to/hard_valid.jsonl
```

出力例:

```text
baseline valid samples=73 ce=3.459810 top1=0.1918
baseline extra_valid[random] samples=50 ce=3.656548 top1=0.1400
baseline extra_valid[hard] samples=23 ce=3.032117 top1=0.3043
```

学習時は各epoch後にも同じ形式で表示する。

## 検証

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin distill_train
```

結果:

```text
cargo test --all-targets: pass
release build: pass
```

実データdry-run:

```bash
target/release/distill_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/shogi_soft_v3_mixed/train.jsonl \
  --valid /tmp/shogi_soft_v3_mixed/valid.jsonl \
  --extra-valid random=/tmp/shogi_soft_v2_d4_top8_500/valid.jsonl \
  --extra-valid hard=/tmp/shogi_soft_v3_static_hard300/valid.jsonl \
  --output /tmp/shogi_soft_v3_mixed/extra_valid_dry_unused.binary \
  --epochs 1 \
  --batch-size 128 \
  --learning-rate 0.0 \
  --softmax-temperature 100 \
  --teacher-temperature 100 \
  --min-teacher-gap 0 \
  --max-teacher-gap 1000 \
  --dry-run
```

結果:

```text
baseline train samples=674 ce=3.516245 top1=0.2092
baseline valid samples=73 ce=3.459810 top1=0.1918
baseline extra_valid[random] samples=50 ce=3.656548 top1=0.1400
baseline extra_valid[hard] samples=23 ce=3.032117 top1=0.3043
```

## 判断

採用。

強さを直接上げる変更ではないが、今後の重み更新で「randomを壊さずhardを改善する」条件を1コマンドで監視できる。過去の失敗であるoffline top1のみの採用判断を避けるための基盤として有用。
