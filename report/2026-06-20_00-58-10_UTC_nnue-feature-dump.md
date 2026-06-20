# NNUE特徴dump基盤

- 作成日時: 2026-06-20 00:58:10 UTC
- ブランチ: `tooling/nnue-feature-dump`
- 目的: 小型NNUEプロトタイプへ進む前段として、既存局面を手番基準に正規化した疎特徴JSONLへ変換できるようにする。

## 実装内容

- `evaluation::extract_nnue_features` を追加。
- `src/bin/nnue_feature_dump.rs` を追加。
- 初期局面でNNUE特徴indexが宣言範囲内に収まり、ソート済み重複なしであることをユニットテスト化。

特徴形式:

- `king_bucket`: 手番側玉位置と相手玉位置を、手番側を先手向きに正規化して `own_king * 81 + opponent_king` で表現する。
- `features`: 盤上駒と持ち駒の疎index列。色は手番側をBlack、相手側をWhiteとして正規化する。
- `material`: 既存評価と同じ駒価値による手番側から見た駒得。
- 任意で `--weights` による `static_eval`、`--depth` による探索教師値 `teacher_score` と `pv` を同時に出力できる。

## 確認

ビルド:

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin nnue_feature_dump
```

小規模dump:

```bash
target/release/nnue_feature_dump \
  --input taya36.sfen \
  --output /tmp/nnue_features_static.jsonl \
  --max-positions 5

target/release/nnue_feature_dump \
  --input taya36.sfen \
  --output /tmp/nnue_features_depth2.jsonl \
  --weights policy_weights_v2.1.0.binary \
  --depth 2 \
  --max-positions 5 \
  --jobs 2
```

結果:

```text
records: 5
skipped positions: 0
```

全体テスト:

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

結果:

```text
11 passed; 0 failed
```

## 判断

これは強さ改善ではなく、学習・評価関数更新に向けた基盤整備として採用する。直近のpolicy-only blend実験は20局で改善を示せなかったため、次は評価関数の表現力を上げる小型NNUEプロトタイプへ進む。

次の候補:

1. JSONLを読むPython/PyTorchの小型MLP学習器を追加する。
2. Rust側に非差分float推論のNNUE評価器を追加し、既存SparseModelと切り替え可能にする。
3. 小規模探索教師値で過学習テストを行い、NPS低下と評価精度を測る。
