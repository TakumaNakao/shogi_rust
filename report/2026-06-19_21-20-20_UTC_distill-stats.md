# distill_stats による教師gap監視

- 作成日時: 2026-06-19 21:20:20 UTC
- ブランチ: `tooling/distill-stats`
- 目的: 蒸留JSONLの teacher score gap 分布を学習前に確認し、詰み級・大差局面の混入率を制御できるようにする。

## 背景

直近の重み更新は、hard-only、static hard、mixed、gap制御mixedのいずれも対局ゲートを通らなかった。

特にhard局面では、探索スコアgapが極端に大きい局面が多かった。これが学習を詰み級・大差局面へ寄せすぎ、通常局面の強さに結びついていない可能性がある。

今後の教師データ基盤では、学習前に以下を確認する。

```text
gap >= 1000 の比率
gap >= 10000 の比率
p50 / p75 / p90 / p95 / p99
teacher_scores がない行、1手しかない行
```

## 実装

`distill_stats` を追加した。

使用例:

```bash
target/release/distill_stats \
  --input /tmp/shogi_soft_v3_mixed/train.jsonl \
  --input /tmp/shogi_soft_v3_mixed/valid.jsonl
```

出力内容:

```text
records
with_teacher_scores
without_teacher_scores
with_gap
one_score
non_finite_scores
gap_avg
gap percentiles
gap_ge_<bucket>
```

bucketは `--buckets 5,10,20,40,80,1000,10000` のように指定できる。

## 実データ確認

### random searched 500

```text
records: 500
with_gap: 500
gap_avg: 3.884
gap_p50: 0.421
gap_p75: 1.802
gap_p90: 6.265
gap_p95: 21.665
gap_p99: 47.523
gap_p100: 366.829
gap_ge_80: 3 (0.60%)
gap_ge_1000: 0 (0.00%)
gap_ge_10000: 0 (0.00%)
```

### hard static 300

```text
records: 299
with_gap: 284
one_score: 15
gap_avg: 18572.539
gap_p50: 28.588
gap_p75: 159.836
gap_p90: 99191.062
gap_p95: 99577.414
gap_p99: 100188.844
gap_p100: 200000.000
gap_ge_80: 99 (34.86%)
gap_ge_1000: 52 (18.31%)
gap_ge_10000: 52 (18.31%)
```

### mixed

```text
records: 799
with_gap: 784
one_score: 15
gap_avg: 6730.285
gap_p50: 0.558
gap_p75: 12.887
gap_p90: 144.024
gap_p95: 98959.680
gap_p99: 99760.297
gap_p100: 200000.000
gap_ge_80: 102 (13.01%)
gap_ge_1000: 52 (6.63%)
gap_ge_10000: 52 (6.63%)
```

## 検証

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin distill_stats
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

結果:

```text
release build: pass
cargo test --all-targets: pass
```

## 判断

採用。

`distill_stats` は強さを直接上げる変更ではないが、次の重み更新で教師データを作る前に、詰み級局面の比率を機械的に確認できる。現時点では、hard static 300 の `gap >= 1000` が18.31%と高く、mixedでは6.63%まで下がる。この情報を使って、今後の教師データは以下を目安にする。

```text
gap >= 1000: 10%以下を目標
gap >= 10000: 10%以下を目標
one_score: 少なめに保つ
random/hard validを別々に監視
```
