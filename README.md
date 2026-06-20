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

#### wdoor/floodgate棋譜の取得

東京大学のwdoor/floodgateアーカイブからCSA棋譜を取得できます。棋譜本体は大きいため `data/wdoor/` に保存し、Git管理対象にはしません。

```bash
tools/download_wdoor_kifu.sh 2026
```

取得後の主なパス:

- `data/wdoor/archive/wdoor2026.7z`
- `data/wdoor/extract/2026`

レート分布を確認して学習用の閾値を決めるには、以下を実行します。

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin csa_rate_stats

target/release/csa_rate_stats \
  --input data/wdoor/extract/2026 \
  --thresholds 3000,3500,4000,4500 \
  --top-players 20
```

#### 実行例

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin kpp_learn

env RUST_FONTCONFIG_DLOPEN=1 target/release/kpp_learn \
  --input-dir data/wdoor/extract/2026 \
  --weights ./policy_weights_v2.1.0.binary \
  --output /tmp/policy_weights_wdoor2026_r4000_ce_seed20260620.binary \
  --epochs 1 \
  --batch-size 2048 \
  --chunk-size 20000 \
  --load-file-batch-size 256 \
  --learning-rate 0.005 \
  --l2-lambda 0.0001 \
  --loss ce \
  --softmax-temperature 150 \
  --valid-percent 5 \
  --valid-max-files 500 \
  --seed 20260620 \
  --checkpoint-dir /tmp/kpp_wdoor2026_r4000_ce_seed20260620_checkpoints \
  --checkpoint-every-batches 200 \
  --log-path /tmp/kpp_wdoor2026_r4000_ce_seed20260620.csv \
  --freeze-material \
  --min-player-rate 4000 \
  --decisive-only \
  --exclude-loser-after-ply 100
```

主な出力:

- `--output`: 最終的な候補重み
- `--checkpoint-dir`: 途中checkpoint
- `--log-path`: 学習ログCSV

CSVには `train_loss`、`valid_ce`、`valid_accuracy`、`material_coeff`、重みの最小/最大値が出ます。長時間学習中は `valid_ce` が改善しているか、`valid_accuracy` が大きく悪化していないかを確認してください。

`--chunk-size` は進捗表示と大きな処理区切り、`--load-file-batch-size` は一度に局面化するCSAファイル数です。メモリ不足を避けるため、通常は `--load-file-batch-size 256` 以下から始めてください。メモリが少ないPCでは `RAYON_NUM_THREADS=4` と `--batch-size 512` も有効です。

#### 24時間程度の長時間学習

家を空けている間などに長く回す場合は、直近4年分のwdoor/floodgate棋譜を使います。まず必要な年の棋譜を取得します。

```bash
tools/download_wdoor_kifu.sh 2023
tools/download_wdoor_kifu.sh 2024
tools/download_wdoor_kifu.sh 2025
tools/download_wdoor_kifu.sh 2026
```

以下は、`2023-2026` の `rate >= 4000` の手を対象に、4 epoch 学習するコマンドです。`--log-path` は付けません。付けると各バッチで検証計算が走り、長時間学習がかなり遅くなるためです。

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin kpp_learn

RUN_DIR="data/wdoor/runs/wdoor2023_2026_r4000_ce_t150_lr005_e4_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR/checkpoints"

nohup env RAYON_NUM_THREADS=4 RUST_FONTCONFIG_DLOPEN=1 target/release/kpp_learn \
  --input-dir data/wdoor/extract/2023 \
  --input-dir data/wdoor/extract/2024 \
  --input-dir data/wdoor/extract/2025 \
  --input-dir data/wdoor/extract/2026 \
  --weights /home/nami_ride_trade/shogi_rust/policy_weights_v2.1.0.binary \
  --output "$RUN_DIR/policy_weights_wdoor2023_2026_r4000_ce_t150_lr005_e4.binary" \
  --loss ce \
  --softmax-temperature 150 \
  --epochs 4 \
  --batch-size 2048 \
  --chunk-size 20000 \
  --load-file-batch-size 256 \
  --valid-percent 5 \
  --valid-max-files 500 \
  --min-player-rate 4000 \
  --decisive-only \
  --exclude-loser-after-ply 100 \
  --learning-rate 0.005 \
  --l2-lambda 0.0001 \
  --freeze-material \
  --checkpoint-dir "$RUN_DIR/checkpoints" \
  --checkpoint-every-batches 5000 \
  --no-graph \
  > "$RUN_DIR/train_stdout.log" 2>&1 &

echo $! > "$RUN_DIR/pid"
echo "$RUN_DIR"
```

進捗確認:

```bash
tail -f "$RUN_DIR/train_stdout.log"
```

停止:

```bash
kill "$(cat "$RUN_DIR/pid")"
```

#### 勝敗を使ったサンプリング

棋譜には勝った側の手と負けた側の手が両方含まれます。負けた側の手をすべて正解手として学習すると、終盤の敗着まで強く覚えてしまう可能性があります。

`kpp_learn` では以下のオプションで調整できます。

- `--decisive-only`: 勝敗が推定できる棋譜だけ使います。
- `--winner-only`: 勝った側の手だけ学習します。
- `--exclude-loser-after-ply N`: 負けた側のN手目以降を除外します。
- `--loser-sample-rate R`: 負けた側の手を確率Rで残します。`0.25` なら25%だけ残します。

最初の長時間学習では、勝者手だけに絞りすぎず、負け側の序中盤は残す設定を推奨します。

```bash
--decisive-only --exclude-loser-after-ply 100
```

より強く絞る比較実験としては、以下も候補です。

```bash
--decisive-only --winner-only
--decisive-only --loser-sample-rate 0.5
```

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
