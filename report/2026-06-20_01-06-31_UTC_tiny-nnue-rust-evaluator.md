# TinyNNUE Rust非差分評価器

- 作成日時: 2026-06-20 01:06:31 UTC
- ブランチ: `feature/tiny-nnue-evaluator`
- 目的: NumPy学習器が出力した小型NNUEバイナリをRustで読み、既存KPP評価と同じprofile経路で速度を比較できるようにする。

## 実装内容

- `evaluation::TinyNnueModel` を追加。
- `TNNUE001` 形式のloaderを追加。
- `TinyNnueModel::predict_from_position` を追加。
  - `extract_nnue_features`
  - feature embedding和
  - king bucket embedding
  - material項
  - clipped ReLU
  - linear output
  - `target_scale` でcpへ戻す
- `eval_profile --nnue-weights` を追加。
- `search_profile --nnue-weights` を追加。
- loaderと推論の最小ユニットテストを追加。

## 確認

ビルド:

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin eval_profile --bin search_profile
```

全体テスト:

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

結果:

```text
12 passed; 0 failed
```

## 速度計測

使用モデル:

- `/tmp/tiny_nnue_export_check.bin`
- hidden 8
- `taya36.sfen` 80局面・depth2教師値で2epochだけ学習した動作確認用モデル
- 強さ評価用ではない

純評価:

```text
sparse:
  evals/sec: 292465.27
tiny-nnue:
  evals/sec: 533690.77
```

探索profile:

```text
sparse:
  samples: 8
  depth: 4
  total nodes: 427563
  nodes/sec: 274720.21

tiny-nnue:
  samples: 8
  depth: 4
  total nodes: 189933
  nodes/sec: 290633.76
```

探索profileは評価値が異なるため探索木も変わる。ここでは「非差分float推論が即座に致命的なNPS低下を起こしていない」ことだけを確認した。

## 判断

xhigh分析の採用ゲートでは、NNUE推論がKPPより明確に遅い場合は撤退だった。今回のH=8動作確認モデルでは少なくとも速度面の即撤退条件には該当しない。

次にやるべきこと:

1. H=32/64/96のモデルで、depth3/4教師値のoffline valid RMSEとsign accuracyを測る。
2. 速度が許容範囲なら、USIまたはベンチ用エンジンでTinyNNUEを実戦評価できる切替を追加する。
3. 40局same-engineゲートで55%以上を目標にする。
