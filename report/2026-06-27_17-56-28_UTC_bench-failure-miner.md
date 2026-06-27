# bench_failure_miner 追加

- 作成日時: 2026-06-27 17:56:28 UTC
- ブランチ: `training/strong-weight-learning-infra`
- 目的: 小手先の探索高速化から一度離れ、ベンチ棋譜由来の敗局 counterexample を体系的に抽出する。

## 背景

直近で以下の小さな高速化を試したが、採用できなかった。

- qsearch 1候補 fast path: 探索カウンタは完全一致したが、速度改善が不安定。
- TT best move delayed scoring: 短い固定深さ profile で探索カウンタが不一致。

GPT-5.5 xhigh の判断では、現行探索は ordering / TT / history / killer / aspiration が絡み、微小な高速化の期待値が低い。次は実装前に敗局分類と bench 由来 counterexample 抽出を強化するべき、という結論になった。

## 追加したもの

`bench_failure_miner` を追加した。

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin bench_failure_miner
```

主な機能:

- `usi_benchmark` の保存棋譜 `.usi` を読み込む。
- 既定では `BaselineWin`、つまり new 側の敗局tailだけを対象にする。
- `--all-records` で勝局も含めて見る。
- tail局面について、以下を比較する。
  - 実戦手
  - timed root手
  - teacher depth探索手
- teacher視点での regret を計算する。
- `actual_bad`, `timed_bad`, `root_rescuable` を分類する。
- JSONLとSFEN exportを出せる。

代表コマンド:

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/bench_failure_miner \
  --record-dir /tmp/shogi_weight_bench_pv_sibling_3k_ultrasafe_11001 \
  --weights policy_weights_v2.1.0.binary \
  --tail-plies 12 \
  --timed-depth 5 \
  --teacher-depth 6 \
  --time-limit-ms 100 \
  --top 20 \
  --jsonl-output /tmp/bench_failure_miner.jsonl \
  --export-timed-bad-sfens /tmp/bench_failure_miner_timed_bad.sfen \
  --export-root-rescue-sfens /tmp/bench_failure_miner_root_rescue.sfen
```

## スモーク検証

実験は `GPT-5.3-codex-spark` サブエージェントに委任した。

使用棋譜:

```text
/tmp/shogi_weight_bench_pv_sibling_3k_ultrasafe_11001
```

軽量条件:

```text
max-records: 6
tail-plies: 6
timed-depth: 3
teacher-depth: 4
time-limit-ms: 30
```

結果:

```text
records loaded: 6
positions probed: 6
samples mined: 6
paired starts: 3
baseline sweep starts: 0
actual teacher mismatches: 5 (83.33%)
timed teacher mismatches: 5 (83.33%)
root_rescuable: 1
in_check: 3 (50.00%)
legal_moves_le_3: 3 (50.00%)
timed mean_regret_cp: 7.34
```

`--all-records` の軽量確認:

```text
records loaded: 4
positions probed: 8
samples mined: 8
actual teacher mismatches: 7 (87.50%)
timed teacher mismatches: 8 (100.00%)
actual bad_regret_gt_300: 0 (0.00%)
timed bad_regret_gt_300: 0 (0.00%)
```

JSONLは全行 `jq` でparseでき、SFEN exportも生成された。

## 判断

`bench_failure_miner` は採用する。

これは直接棋力を上げる変更ではないが、次の学習・root救済実験の入力を作るための基盤である。特に、過去の direct feedback 113件では `regret_delta > 0` が0件で、candidate固有の悪化を押せていなかった。このツールで、まず timed root miss や root-rescuable が実際にどれだけあるかを測る。

## 次の手順

1. 60局以上の既存benchに対して `bench_failure_miner` を走らせる。
2. `timed_bad`, `actual_bad_timed_not_bad`, `root_rescuable`, `in_check`, `legal_moves <= 3` を集計する。
3. root-rescuable が多ければ、限定的なroot救済案を検討する。
4. timed_bad が多ければ、深いteacherとの差分を学習データ化する。
5. counterexampleが少なければ、探索改善ではなく評価関数・序盤対策へ戻る。
