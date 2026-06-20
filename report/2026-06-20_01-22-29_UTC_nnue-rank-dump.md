# NNUE子局面ranking dump基盤

- 作成日時: 2026-06-20 01:22:29 UTC
- ブランチ: `tooling/nnue-rank-dump`
- 目的: root局面だけのvalue学習ではTinyNNUEが対局で崩壊したため、同一root内の合法手ごとの子局面比較を学習・検証できるデータを作る。

## 実装内容

`src/bin/nnue_rank_dump.rs` を追加した。

各root局面について:

1. 合法手を列挙する。
2. 各合法手を指した子局面を作る。
3. 既存KPP教師探索で子局面を `depth - 1` 探索し、root手番から見た `teacher_score` を付ける。
4. `teacher_score` 順にrankを付ける。
5. 子局面のNNUE特徴をJSONLへ出力する。

主なフィールド:

- `root_sfen`
- `child_sfen`
- `move_usi`
- `rank`
- `legal_moves`
- `teacher_score`
- `best_score`
- `regret`
- `king_bucket`
- `features`
- `material`

## 確認

ビルド:

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin nnue_rank_dump
```

小規模dump:

```bash
target/release/nnue_rank_dump \
  --input taya36.sfen \
  --output /tmp/nnue_rank_taya_d3_top8.jsonl \
  --weights policy_weights_v2.1.0.binary \
  --depth 3 \
  --max-positions 4 \
  --top-k 8 \
  --jobs 2
```

結果:

```text
roots: 4
records: 32
avg records/root: 8.00
```

36局面top8:

```bash
target/release/nnue_rank_dump \
  --input taya36.sfen \
  --output /tmp/nnue_rank_taya36_d3_top8.jsonl \
  --weights policy_weights_v2.1.0.binary \
  --depth 3 \
  --max-positions 36 \
  --top-k 8 \
  --jobs 4
```

結果:

```text
roots: 36
records: 288
avg records/root: 8.00
real 0m6.832s
file size: 153K
```

全体テスト:

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

結果:

```text
12 passed; 0 failed
```

## 判断

これは強さ改善ではなく、TinyNNUE再実験のためのデータ基盤として採用する。前回のdepth3/depth4 root value学習はoffline RMSEやroot regretが良くても対局で大きく負けたため、次は同一root内の子局面rankingを学習・評価に入れる。

次:

1. `train_nnue_numpy.py` にrank JSONL対応を追加する。
2. value lossだけでなく、同一root内の上位手と下位手のmargin/rankingを評価する。
3. 対局前にroot内rank accuracyやbad top1 regretをgateにする。
