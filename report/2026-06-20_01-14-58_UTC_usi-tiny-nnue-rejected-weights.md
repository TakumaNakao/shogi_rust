# USI TinyNNUE接続と小規模重み棄却

- 作成日時: 2026-06-20 01:14:58 UTC
- ブランチ: `feature/usi-tiny-nnue-evalfile`
- 目的: `usi_engine` の `EvalFile` で既存KPP `.binary` とTinyNNUE `TNNUE001` バイナリの両方を読めるようにし、対局ゲートを実行する。

## 実装内容

- `evaluation::EngineEvaluator` を追加。
- `EvalFile` の先頭8 bytesが `TNNUE001` なら `TinyNnueModel` として読む。
- それ以外は従来通り `SparseModelEvaluator` として読む。
- `usi_shogi` の探索評価器を `EngineEvaluator` に差し替えた。

## 確認

ビルド:

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin usi_engine --bin usi_benchmark
```

全体テスト:

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

結果:

```text
12 passed; 0 failed
```

USI確認:

```text
info string Loaded sparse evaluation file: policy_weights_v2.1.0.binary
bestmove 8g8f

info string Loaded tiny-nnue evaluation file: /tmp/tiny_nnue_depth3_h32.bin
bestmove 2h3h
```

## H=32 depth3小規模重み

条件:

```text
new: /tmp/tiny_nnue_depth3_h32.bin
baseline: policy_weights_v2.1.0.binary
games: 20
depth: 5
time-limit-ms: 100
seed: 6101
```

結果:

```text
new wins: 1
baseline wins: 19
draws: 0
new decisive win rate: 5.00%
new total score rate: 5.00%
paired starts:
  new sweeps: 0
  baseline sweeps: 9
  splits: 1
  draw/mixed pairs: 0
average final score for new: -1020.8
```

## H=64 depth3 best checkpoint

条件:

```text
new: /tmp/tiny_nnue_depth3_h64_best.bin
baseline: policy_weights_v2.1.0.binary
games: 20
depth: 5
time-limit-ms: 100
seed: 6102
```

結果:

```text
new wins: 2
baseline wins: 17
draws: 1
new decisive win rate: 10.53%
new total score rate: 12.50%
paired starts:
  new sweeps: 0
  baseline sweeps: 7
  splits: 2
  draw/mixed pairs: 1
average final score for new: -743.3
```

## 判断

USI接続基盤は採用する。既存KPP重みを壊さず、TinyNNUE候補を通常の `usi_benchmark` に載せられるようになったため、今後の評価関数実験に必要な土台である。

一方、今回のH=32/H=64小規模重みは棄却する。offline valid RMSEはKPP `static_eval` より良く見えたが、20局対局で明確に崩壊した。原因は、512局面・depth3教師値だけでは探索中の評価分布を支えられず、局面供給も `taya36.sfen` に偏りすぎていることが濃厚。

次の方針:

1. TinyNNUE自体は継続。
2. 重み更新は、より広い局面集合とdepth4/5教師値を使う。
3. 対局ゲート前に、root候補手ランキング/regret系のoffline評価を追加する。
4. 直接20局で崩壊した小規模重みは削除し、リリース・タグ対象にしない。
