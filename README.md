# shogi_ai

これは、Rustで書かれた将棋AIです。
ほとんどがGemini-CLIを使用してコーディングされています．

USI (Universal Shogi Interface) プロトコルに対応した思考エンジン `usi_engine` と、評価関数学習ツール `kpp_learn` が含まれています。

## 主な機能

- `usi_engine`: USIプロトコル対応の将棋思考エンジンです。ShogiGUIなどのGUIに導入して対局や検討ができます。
- `kpp_learn`: 評価関数KPP (Komi, Piece, Position) の学習を行います。

## 使い方 (ShogiGUI)

### 1. エンジンのダウンロード

1.  本リポジトリの [Releasesページ](https://github.com/TakumaNakao/shogi_rust/releases) にアクセスします。
2.  最新のリリースから、お使いのOSに合ったファイルをダウンロードします。
    *   **Windows:** `usi_engine.exe`
    *   **Linux:** `usi_engine`

### 2. ShogiGUIへのエンジン登録

1.  ShogiGUIを起動します。
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
