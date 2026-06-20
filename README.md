# shogi_ai

これは、Rustで書かれた将棋AIです。
ほとんどがGemini-CLIを使用してコーディングされています．

USI (Universal Shogi Interface) プロトコルに対応した思考エンジン `usi_engine` と、評価関数学習ツール `kpp_learn` が含まれています。

## 主な機能

- `usi_engine`: USIプロトコル対応の将棋思考エンジンです。ShogiGUIなどのGUIに導入して対局や検討ができます。
- `kpp_learn`: 評価関数KPP (Komi, Piece, Position) の学習を行います。

## 使い方 (ShogiHome)

### 1. エンジンのダウンロード

1.  本リポジトリの [Releasesページ](https://github.com/TakumaNakao/shogi_rust/releases) にアクセスします。
2.  最新のリリースから、お使いのOSに合ったファイルをダウンロードします。
    *   **Windows:** `usi_engine.exe`
    *   **Linux:** `usi_engine`

### 2. [ShogiHome](https://sunfish-shogi.github.io/shogihome/)へのエンジン登録

1.  ShogiHomeを起動します。
2.  「エンジン設定」を選択します。
3.  「追加」ボタンをクリックします。
4.  ファイル選択ダイアログが表示されるので、先ほどダウンロードした `usi_engine.exe` (または `usi_engine`) を選択します。
5.  エンジン一覧に `usi_engine` が追加されたことを確認し、「設定」をクリックします。
6.  「評価関数ファイル」にパソコンに保存されている評価関数の重みファイル (weights.binary) のパスを設定します．
7.  「OK」をクリックします．
8.  「保存して閉じる」をクリックします．

これで、対局や検討の際に `usi_engine` を選択して使用できるようになります。

## 開発者向け情報

### ビルド方法

```bash
# リリースビルド
cargo build --release

# 特定のバイナリのみビルド
cargo build --release --bin usi_engine
cargo build --release --bin kpp_learn
```

### KPP評価関数の学習

`kpp_learn` はCSA棋譜からKPP評価関数の重みを学習するツールです。
現在は2種類の学習方式を選べます。

- `--loss margin`: 従来方式。モデル最善手が棋譜の手と違う時だけ、棋譜の手へ近づけます。
- `--loss ce`: 新方式。合法手全体のsoftmax分布を使うcross entropy学習です。正解手にどれだけ確率を割り当てているかを見て更新します。

新しく重みを試す場合は、まず `ce` 方式と `--freeze-material` の組み合わせを推奨します。駒得係数を固定し、KPP重みだけを安全に更新します。

#### 実行例

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin kpp_learn

env RUST_FONTCONFIG_DLOPEN=1 target/release/kpp_learn \
  --input-dir /path/to/csa/2016 \
  --input-dir /path/to/csa/2017 \
  --weights ./policy_weights_v2.1.0.binary \
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

主な出力:

- `--output`: 最終的な候補重み
- `--checkpoint-dir`: 途中checkpoint
- `--log-path`: 学習ログCSV

CSVには `train_loss`、`valid_ce`、`valid_accuracy`、`material_coeff`、重みの最小/最大値が出ます。長時間学習中は `valid_ce` が改善しているか、`valid_accuracy` が大きく悪化していないかを確認してください。

#### 採用前の検証

学習後の重みは短い対局だけで採用しないでください。最低限、以下を確認します。

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release \
  --bin kpp_weight_check --bin search_profile --bin usi_engine --bin usi_benchmark

env RUST_FONTCONFIG_DLOPEN=1 target/release/kpp_weight_check \
  --weights /tmp/policy_weights_kpp_ce_t600_lr005_seed20260620.binary

env RUST_FONTCONFIG_DLOPEN=1 target/release/search_profile \
  --weights /tmp/policy_weights_kpp_ce_t600_lr005_seed20260620.binary \
  --positions ./taya36.sfen \
  --samples 72 \
  --depth 5 \
  --seed 9501
```

候補重みは、まず現行重みとの40局以上の比較で悪化しないことを確認し、良い候補だけ100局以上で検証します。採用する重みはGit管理せず、GitHub Releaseのassetとして手動アップロードします。
