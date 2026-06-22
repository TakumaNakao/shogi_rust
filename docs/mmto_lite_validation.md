# MMTO-lite検証手順書

この文書は、Bonanza/MMTO系の完全実装へ進む前に、現在実装済みの `mmto_dump` / `mmto_train` で「探索結果と整合するKPP学習」が有効かを小さく検証するための手順である。

結論として、完全MMTOへすぐ進まない。まずroot候補手の探索スコア分布をKPPへlistwise蒸留し、offline regretと短時間対局の両方で改善傾向を確認する。

## 1. 基本方針

- `policy_weights_v2.1.0.binary` は必ず固定し、上書きしない。
- 学習後の重みはすべて `data/mmto/runs/...` 以下へ別名保存する。
- `data/mmto/` はgit管理しない。
- validation CEだけで採用しない。
- 主指標は `valid selected_regret_mean`, `p90`, `p95`, `bad_regret`, `expected_regret` とする。
- 対局検証なしで本格学習へ進まない。
- 20局はsmoke test、40局は初期gate、100局以上を採用判断に使う。

現在の `mmto_dump --teacher-score-source searched` は、全合法手を探索してから `--teacher-score-top` に切る。そのため、最初は必ず少数局面・浅いdepthから始める。

## 2. 事前確認

```bash
cd /home/nami_ride_trade/shogi_rust
git status --short --branch
test -f policy_weights_v2.1.0.binary
```

ビルドする。

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release \
  --bin mmto_dump \
  --bin mmto_probe \
  --bin mmto_train \
  --bin adjust_weights \
  --bin kpp_weight_check \
  --bin usi_engine \
  --bin usi_benchmark \
  --bin record_analyze
```

## 3. Smoke Test

目的は、JSONL生成、dry-run、1 epoch保存が壊れていないことを確認すること。

```bash
RUN_DIR="data/mmto/runs/smoke_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_dump \
  --weights policy_weights_v2.1.0.binary \
  --input taya36.sfen \
  --train-output "$RUN_DIR/train.rank.jsonl" \
  --valid-output "$RUN_DIR/valid.rank.jsonl" \
  --depth 1 \
  --teacher-score-top 4 \
  --teacher-score-source searched \
  --max-positions 20 \
  --valid-percent 10 \
  --jobs 1 \
  --seed 4101

head -n 1 "$RUN_DIR/train.rank.jsonl"
wc -l "$RUN_DIR/train.rank.jsonl" "$RUN_DIR/valid.rank.jsonl"
```

dry-runでbaseline指標を確認する。

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_train \
  --weights policy_weights_v2.1.0.binary \
  --train "$RUN_DIR/train.rank.jsonl" \
  --valid "$RUN_DIR/valid.rank.jsonl" \
  --output "$RUN_DIR/smoke.binary" \
  --dry-run \
  --epochs 1 \
  --log-path "$RUN_DIR/dry_run.log"
```

1 epochだけ実際に保存する。

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_train \
  --weights policy_weights_v2.1.0.binary \
  --train "$RUN_DIR/train.rank.jsonl" \
  --valid "$RUN_DIR/valid.rank.jsonl" \
  --output "$RUN_DIR/smoke.binary" \
  --epochs 1 \
  --batch-size 4 \
  --learning-rate 0.02 \
  --model-temperature 600 \
  --teacher-temperature 150 \
  --anchor-l2 0.0005 \
  --max-weight-delta 0.05 \
  --best-checkpoint-path "$RUN_DIR/best.binary" \
  --log-path "$RUN_DIR/train.log"

env RUST_FONTCONFIG_DLOPEN=1 target/release/kpp_weight_check "$RUN_DIR/best.binary" \
  > "$RUN_DIR/weight_check.txt"
```

注意:

- `kpp_weight_check` は全KPP重みを並べ替えるためメモリ消費が大きい。複数候補に対して並列実行しない。
- ハイパーパラメータ探索だけが目的なら、`mmto_train --output /dev/null` とし、`--best-checkpoint-path` を指定しないことで巨大な重みファイル生成を避けられる。
- 対局へ進める候補だけ `--best-checkpoint-path "$RUN_DIR/best.binary"` を指定して保存する。

ここで見るもの:

- `train.rank.jsonl` と `valid.rank.jsonl` が空でない。
- `baseline valid` が表示される。
- `best.binary` と `smoke.binary` が保存される。
- `weight_check.txt` が出る。
- `best_epoch=0` でも異常ではない。smokeでは改善を期待しない。

## 4. 小規模MMTO-lite実験

最初の本実験は、探索コストを抑えるため `max-positions 200` 程度から始める。

```bash
RUN_DIR="data/mmto/runs/d3_top8_200_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_dump \
  --weights policy_weights_v2.1.0.binary \
  --input converted_records2016_10818.sfen \
  --train-output "$RUN_DIR/train.rank.jsonl" \
  --valid-output "$RUN_DIR/valid.rank.jsonl" \
  --depth 3 \
  --teacher-score-top 8 \
  --teacher-score-source searched \
  --max-positions 200 \
  --valid-percent 10 \
  --min-legal-moves 2 \
  --exclude-in-check \
  --max-abs-root-score 3000 \
  --jobs 4 \
  --seed 4201
```

学習する。

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_train \
  --weights policy_weights_v2.1.0.binary \
  --train "$RUN_DIR/train.rank.jsonl" \
  --valid "$RUN_DIR/valid.rank.jsonl" \
  --output "$RUN_DIR/mmto_d3_top8.binary" \
  --epochs 2 \
  --batch-size 64 \
  --learning-rate 0.02 \
  --model-temperature 600 \
  --teacher-temperature 150 \
  --anchor-l2 0.0005 \
  --max-weight-delta 0.05 \
  --bad-regret-cp 300 \
  --best-metric selected-regret \
  --best-checkpoint-path "$RUN_DIR/best.binary" \
  --log-path "$RUN_DIR/train.log"
```

重みを検査する。

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/kpp_weight_check "$RUN_DIR/best.binary" \
  > "$RUN_DIR/weight_check.txt"
tail -n 5 "$RUN_DIR/train.log"
```

## 5. Offline Gate

`train.log` のepoch 0がbaselineである。候補を対局へ進める条件:

- `valid_selected_regret_mean` がepoch 0以下。
- `valid_p90_regret` と `valid_p95_regret` がepoch 0以下、または悪化がごく小さい。
- `valid_bad_regret_ratio` がepoch 0以下。
- `valid_expected_regret` がepoch 0以下。
- `max_abs_delta` が `--max-weight-delta` 以下。
- `material_coeff` が変わっていない。
- `best_epoch=0` の場合は、学習で改善していないため原則対局へ進めない。

CEだけが改善してregretが悪化した候補は破棄する。

### 5.1 Hard-negative診断

offline regretが改善しても対局に効かない場合は、full-legal validで「現モデルが高評価して選びそうな高regret悪手」を抽出する。

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_probe \
  --weights policy_weights_v2.1.0.binary \
  --input "$RUN_DIR/valid.rank.jsonl" \
  --output "$RUN_DIR/probe_valid_regret100.jsonl" \
  --min-regret 100 \
  --top 50 \
  --format jsonl
```

CSVで目視する場合:

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_probe \
  --weights policy_weights_v2.1.0.binary \
  --input "$RUN_DIR/valid.rank.jsonl" \
  --output "$RUN_DIR/probe_valid_regret100.csv" \
  --min-regret 100 \
  --top 50 \
  --format csv
```

見るべき点:

- `selected_regret` が大きい局面がどの程度あるか。
- `model_rank_by_teacher` が大きい手をモデルが選んでいないか。このrankは0-basedで、0がteacher bestである。
- `teacher_gap` が小さい局面だけに偏っていないか。
- `candidate_count` が十分大きいか。top8/top16だけでは危険手を見落とすことがある。

`mmto_probe` は該当局面が0件でも正常終了する。JSONLは空ファイル、CSVはヘッダのみになる。

### 5.2 Hard-negative学習

`listwise` で全体順位を整えつつ、現モデルが選びそうな高regret悪手を直接下げる補助損失を使う。

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_train \
  --weights policy_weights_v2.1.0.binary \
  --train "$RUN_DIR/train.rank.jsonl" \
  --valid "$RUN_DIR/valid.rank.jsonl" \
  --output "$RUN_DIR/mmto_hard_negative.binary" \
  --epochs 5 \
  --batch-size 128 \
  --learning-rate 10 \
  --model-temperature 30 \
  --teacher-temperature 100 \
  --loss listwise-hard-negative \
  --hard-negative-weight 0.1 \
  --hard-negative-min-regret 100 \
  --hard-negative-margin 0.5 \
  --hard-negative-top-model 5 \
  --hard-negative-top-teacher 1 \
  --anchor-l2 0.0001 \
  --max-weight-delta 0.2 \
  --bad-regret-cp 300 \
  --best-metric selected-regret \
  --best-checkpoint-path "$RUN_DIR/best_hard_negative.binary" \
  --log-path "$RUN_DIR/train_hard_negative.log"
```

初期探索範囲:

- `--hard-negative-weight`: 0.05, 0.1, 0.2
- `--hard-negative-min-regret`: 50, 100, 200
- `--hard-negative-margin`: 0.5, 1.0
- `--hard-negative-top-model`: 3, 5
- `--hard-negative-top-teacher`: 1, 2

採用条件:

- `valid_selected_regret_mean` がepoch 0より5%以上改善。
- `valid_p90_regret`, `valid_p95_regret`, `valid_bad_regret_ratio`, `valid_expected_regret` が悪化しない。
- `hard_negative_samples` と `hard_negative_pairs` が0ではない。
- `probe_valid_regret100` の高regret局面が減る。
- 20局smokeと40局gateを必ず通す。

## 6. Blend候補の作成

MMTO-liteで直接学習した重みが強すぎる場合に備え、v2.1.0とのblendを作る。

```bash
for RATIO in 0.05 0.10 0.20; do
  env RUST_FONTCONFIG_DLOPEN=1 target/release/adjust_weights \
    --input policy_weights_v2.1.0.binary \
    --blend-target "$RUN_DIR/best.binary" \
    --blend-ratio "$RATIO" \
    --output "$RUN_DIR/blend_${RATIO}.binary"
done
```

対局へ進める順序:

1. `blend_0.05.binary`
2. `blend_0.10.binary`
3. `blend_0.20.binary`
4. `best.binary`

小さいblendが弱いのに大きいblendだけ強い、という結果は過学習やノイズの疑いがあるため慎重に扱う。

## 7. 安定性フィルター（depth3/depth4一致）

depth4 の `kpp_rank_v1` JSONL を depth3 と照合して、安定性で `stable` / `unstable` を振り分ける前処理。

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_stability_filter \
  --depth3 path/to/d3.rank.jsonl \
  --depth4 path/to/d4.rank.jsonl \
  --output-stable path/to/stable.rank.jsonl \
  --output-unstable path/to/unstable.rank.jsonl \
  --stats-output path/to/stable_stats.json \
  --min-d4-gap-cp 15 \
  --min-d3-gap-cp 5 \
  --min-legal-moves 2
```

soft条件を許可する例:

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_stability_filter \
  --depth3 path/to/d3.rank.jsonl \
  --depth4 path/to/d4.rank.jsonl \
  --output-stable path/to/stable.rank.jsonl \
  --output-unstable path/to/unstable.rank.jsonl \
  --stats-output path/to/stable_stats.json \
  --require-best-match \
  --max-d4-best-rank-in-d3 3 \
  --max-d3-best-regret-in-d4-cp 25 \
  --min-legal-moves 2
```

主な出力:
- `stable.rank.jsonl`: 条件をすべて満たした depth4 record（JSON行をそのまま保持）
- `unstable.rank.jsonl`: 不適合 record（同じく元JSON行）
- `stable_stats.json`: reject 理由ごとの件数、ヒストグラム、分布統計を含む集計

best不一致を診断したい場合（best条件を無視してその他条件のヒスト分布を確認）:
```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_stability_filter \
  --depth3 path/to/d3.rank.jsonl \
  --depth4 path/to/d4.rank.jsonl \
  --output-stable path/to/stable.rank.jsonl \
  --output-unstable path/to/unstable.rank.jsonl \
  --stats-output path/to/stable_stats.json \
  --allow-best-mismatch \
  --max-d4-best-rank-in-d3 3 \
  --max-d3-best-regret-in-d4-cp 25 \
  --min-legal-moves 2
```

`--allow-best-mismatch` を付けない場合は、既定（`--require-best-match`）の厳密チェックがそのまま有効です。

## 8. 20局 Smoke対局

重みだけの効果を見るため、同じ `usi_engine` を両側に使い、重みだけを変える。

```bash
WEIGHT="$RUN_DIR/blend_0.10.binary"
BENCH_DIR="$RUN_DIR/bench20_blend010_seed5101"
rm -rf "$BENCH_DIR"

env RUST_FONTCONFIG_DLOPEN=1 target/release/usi_benchmark \
  --new-engine /home/nami_ride_trade/shogi_rust/target/release/usi_engine \
  --baseline-engine /home/nami_ride_trade/shogi_rust/target/release/usi_engine \
  --new-weights "$WEIGHT" \
  --baseline-weights /home/nami_ride_trade/shogi_rust/policy_weights_v2.1.0.binary \
  --positions /home/nami_ride_trade/shogi_rust/taya36.sfen \
  --games 20 \
  --depth 5 \
  --time-limit-ms 100 \
  --max-plies 200 \
  --adjudicate-at-max-plies \
  --jobs 4 \
  --seed 5101 \
  --record-dir "$BENCH_DIR"

env RUST_FONTCONFIG_DLOPEN=1 target/release/record_analyze \
  --weights "$WEIGHT" \
  --record-dir "$BENCH_DIR" \
  --tail-plies 12 \
  > "$BENCH_DIR/record_analyze.txt"
```

20局gate:

- score rate 45%未満なら破棄。
- 明確な実装バグ、異常な反則、終盤の大崩れがあれば破棄。
- 50%以上なら40局へ進める。
- 45%から50%は、offline指標が強い場合だけ別seedで再確認する。

## 9. 40局 Gate

```bash
WEIGHT="$RUN_DIR/blend_0.10.binary"
BENCH_DIR="$RUN_DIR/bench40_blend010_seed5201"
rm -rf "$BENCH_DIR"

env RUST_FONTCONFIG_DLOPEN=1 target/release/usi_benchmark \
  --new-engine /home/nami_ride_trade/shogi_rust/target/release/usi_engine \
  --baseline-engine /home/nami_ride_trade/shogi_rust/target/release/usi_engine \
  --new-weights "$WEIGHT" \
  --baseline-weights /home/nami_ride_trade/shogi_rust/policy_weights_v2.1.0.binary \
  --positions /home/nami_ride_trade/shogi_rust/taya36.sfen \
  --games 40 \
  --depth 5 \
  --time-limit-ms 100 \
  --max-plies 200 \
  --adjudicate-at-max-plies \
  --jobs 4 \
  --seed 5201 \
  --record-dir "$BENCH_DIR"

env RUST_FONTCONFIG_DLOPEN=1 target/release/record_analyze \
  --weights "$WEIGHT" \
  --record-dir "$BENCH_DIR" \
  --tail-plies 12 \
  > "$BENCH_DIR/record_analyze.txt"
```

40局gate:

- score rate 50%未満なら破棄。
- 55%以上なら局面数やdepthを少し増やす。
- 50%から55%は、seed違い40局でもう一度確認する。
- baseline sweepsがnew sweepsより多い場合は破棄寄りに判断する。

## 10. 次のスケールアップ

40局gateを通った場合だけ、次の順で拡大する。

現在の推奨は、trainもvalidもfull-legalに近い候補集合で検証すること。`searched` は内部的に全合法手を探索してから `--teacher-score-top` に切るため、`--teacher-score-top 128` は多くの通常局面で実質full-legal trainになる。

### 10.1 局面数を増やす

```bash
RUN_DIR="data/mmto/runs/d3_top128_1000_fullvalid_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_dump \
  --weights policy_weights_v2.1.0.binary \
  --input converted_records2016_10818.sfen \
  --train-output "$RUN_DIR/train.rank.jsonl" \
  --valid-output "$RUN_DIR/valid.rank.jsonl" \
  --depth 3 \
  --teacher-score-top 128 \
  --teacher-score-source searched \
  --max-positions 1000 \
  --valid-percent 10 \
  --score-all-legal-for-valid \
  --min-legal-moves 2 \
  --exclude-in-check \
  --max-abs-root-score 3000 \
  --jobs 4 \
  --seed 4301
```

`depth 4` は探索コストが大きいため、`depth 3 / 1000局面` が非悪化を示してから試す。

### 10.2 ハイパーパラメータを小さく振る

優先順:

- `teacher-temperature`: 100, 150, 300
- `learning-rate`: 0.005, 0.01, 0.02
- `max-weight-delta`: 0.02, 0.05
- `teacher-score-top`: 8, 16

MMTO-lite改善版では、以下も試す。

- `--loss listwise-pairwise`
- `--pairwise-weight`: 0.01, 0.05, 0.1
- `--pairwise-gap`: 1, 2, 5
- `--pairwise-margin`: 0.5, 1, 2
- `--train-min-teacher-gap`: 0, 0.5, 1
- `--train-min-score-span`: 0, 5, 10
- `--valid-filter none` を標準にし、validを簡単にしない。
- `--loss listwise-hard-negative`
- `--hard-negative-weight`: 0.05, 0.1, 0.2
- `--hard-negative-min-regret`: 50, 100, 200
- `--hard-negative-margin`: 0.5, 1.0
- `--hard-negative-top-model`: 3, 5
- `--hard-negative-top-teacher`: 1, 2

一度に複数要素を変えない。

## 11. 100局 Gate

40局で有望な候補だけ100局へ進める。

```bash
WEIGHT="$RUN_DIR/blend_0.10.binary"
BENCH_DIR="$RUN_DIR/bench100_blend010_seed5301"
rm -rf "$BENCH_DIR"

env RUST_FONTCONFIG_DLOPEN=1 target/release/usi_benchmark \
  --new-engine /home/nami_ride_trade/shogi_rust/target/release/usi_engine \
  --baseline-engine /home/nami_ride_trade/shogi_rust/target/release/usi_engine \
  --new-weights "$WEIGHT" \
  --baseline-weights /home/nami_ride_trade/shogi_rust/policy_weights_v2.1.0.binary \
  --positions /home/nami_ride_trade/shogi_rust/taya36.sfen \
  --games 100 \
  --depth 5 \
  --time-limit-ms 100 \
  --max-plies 200 \
  --adjudicate-at-max-plies \
  --jobs 4 \
  --seed 5301 \
  --record-dir "$BENCH_DIR"

env RUST_FONTCONFIG_DLOPEN=1 target/release/record_analyze \
  --weights "$WEIGHT" \
  --record-dir "$BENCH_DIR" \
  --tail-plies 12 \
  > "$BENCH_DIR/record_analyze.txt"
```

100局gate:

- score rate 55%以上を最低条件にする。
- できればseed違いでも50%を下回らないことを確認する。
- 100局で50%前後なら、完全MMTOではなく学習信号・データ選別・温度・blendを見直す。
- 100局で明確に勝ち越すまで、長時間学習へ進めない。

## 12. アルゴリズム込みの最終比較

重みだけで有望な候補が出たあと、現行HEADの探索アルゴリズム込みで `v2.1.0` baselineと比較する。

```bash
git worktree add /tmp/shogi_rust_v210_bench v2.1.0
cd /tmp/shogi_rust_v210_bench
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin usi_engine

cd /home/nami_ride_trade/shogi_rust
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin usi_engine --bin usi_benchmark --bin record_analyze

WEIGHT="data/mmto/runs/<run>/blend_0.10.binary"
BENCH_DIR="data/mmto/runs/<run>/bench100_vs_v210_seed5401"
rm -rf "$BENCH_DIR"

env RUST_FONTCONFIG_DLOPEN=1 target/release/usi_benchmark \
  --new-engine /home/nami_ride_trade/shogi_rust/target/release/usi_engine \
  --baseline-engine /tmp/shogi_rust_v210_bench/target/release/usi_engine \
  --new-weights "$WEIGHT" \
  --baseline-weights /home/nami_ride_trade/shogi_rust/policy_weights_v2.1.0.binary \
  --positions /home/nami_ride_trade/shogi_rust/taya36.sfen \
  --games 100 \
  --depth 5 \
  --time-limit-ms 100 \
  --max-plies 200 \
  --adjudicate-at-max-plies \
  --jobs 4 \
  --seed 5401 \
  --record-dir "$BENCH_DIR"
```

この比較は「現行探索 + 新重み」対「v2.1.0探索 + v2.1.0重み」であり、リリース候補の判断に使う。重み単体の効果とは分けて記録する。

## 13. 完全MMTOへ進む条件

完全MMTOまたは探索木内兄弟比較へ進むのは、以下を満たした後にする。

- MMTO-liteが複数設定でoffline regretを改善する。
- 20局/40局で非悪化、少なくとも1候補が明確に有望。
- 100局でv2.1.0重みへ勝ち越す候補がある。
- dump局面数を増やしても性能が崩れない。
- 失敗時に敗因を追える `record_analyze` とtrain logが残っている。

次に実装するなら、完全MMTOではなく以下の順に進む。

1. hard-position replay
2. listwise + pairwise補助
3. root近傍の兄弟比較
4. PV leaf特徴の保存
5. exact/boundを区別した探索木MMTO

## 14. 破棄条件

以下に該当する候補は破棄する。

- `best_epoch=0` のまま。
- valid regretがbaselineより悪化。
- `bad_regret` が増える。
- 20局で45%未満。
- 40局で50%未満。
- 100局で50%前後に留まる。
- 先後どちらかだけ極端に悪い。
- baseline sweepsがnew sweepsより多い。
- `kpp_weight_check` で重みが異常に偏る。

## 15. 結果記録

有望候補または方針転換が出たら、`report/` に日本語で報告書を作る。

最低限記録する項目:

- run directory
- dump条件
- train条件
- offline baseline指標
- best epoch指標
- blend ratio
- 20局/40局/100局結果
- end reasons
- paired starts
- 採用/破棄判断
- 次の実験
