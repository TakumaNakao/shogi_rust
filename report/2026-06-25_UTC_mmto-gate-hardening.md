# MMTO学習ゲート強化レポート

- 作成日時: 2026-06-25 UTC
- ブランチ: `improve-mmto-training-density`
- 目的: offline gateを通過したMMTO候補が100局ベンチで悪化したため、弱い候補を学習パイプライン内で早期棄却できるようにする。

## 背景

`data/mmto/runs/mmto_from_dump_20260625_110902/best.raw.binary` は以下のoffline結果でgateを通過した。

- baseline mean regret: `8.36`
- candidate mean regret: `8.30`
- 改善量: `0.06cp`
- match: `49.00% -> 49.30%`

しかし同一エンジン・重み差分のみの100局ベンチでは以下となり、不採用とした。

- NewWin: `41`
- BaselineWin: `57`
- Draw: `2`
- new total score rate: `42.00%`

この結果から、従来の「悪化しない」offline gateだけでは、実戦強度に足りない微小改善候補を通してしまうことが分かった。

## 実装内容

### 1. 不要な候補重みの削除

不採用候補の大きな重みファイルを削除した。

- `data/mmto/runs/mmto_from_dump_20260625_110902/best.raw.binary`
- `data/mmto/runs/mmto_from_dump_20260625_110902/blend_0.02.binary`
- `data/mmto/runs/mmto_from_dump_20260625_110902/blend_0.05.binary`

残っている `.binary` は `policy_weights_v2.1.0.binary` のみで、baselineとして必要なため保持した。

### 2. rerank gateに最低改善条件を追加

`mmto_rerank_gate` に以下のCLIを追加した。

- `--require-mean-regret-improvement-cp`
- `--require-p90-regret-improvement-cp`
- `--require-p95-regret-improvement-cp`
- `--require-match-rate-improvement-pct`
- `--require-bad-regret-improvement threshold:ratio`

既定値はすべて `0` で後方互換を維持する。スクリプトの標準設定では `mean regret` に `0.5cp` 以上の改善を要求する。

今回の失敗候補は改善量が `0.06cp` しかないため、新ゲートではFAILになる。

### 3. best epoch選択を頑健化

`mmto_tree_train --best-metric` に以下を追加した。

- `p90-regret`
- `p95-regret`
- `bad50-regret` / `bad-regret-50`
- `capped-selected-regret`

また `--selected-regret-cap-cp` を追加し、外れ値をcapした平均regretを使えるようにした。

標準パイプラインは `--best-metric p95-regret` に変更した。これにより、極端な外れ値を含むmeanだけが少し改善した候補ではなく、悪い分位を改善する候補を選びやすくする。

## 検証

実行済み:

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo fmt --check
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin mmto_tree_train --bin mmto_rerank_gate --bin mmto_score_gate --bin adjust_weights
```

旧失敗runのJSONを使った確認:

```text
baseline mean=8.36 candidate mean=8.30 improvement=0.06
new gate decision: FAIL
```

baseline同士の小規模rerank gate確認:

```text
RERANK GATE FAILED: ["mean regret failed improvement requirement: 0.00 > 0.00 - 0.50"]
status=2
```

## 次の方針

次のMMTO学習は、既存の20k dumpを再利用して短めに回す。

推奨:

```bash
SOURCE_RUN_DIR=data/mmto/runs/mmto_rerank_long_20260624_140151 \
TRAIN_LINES=7000 \
VALID_LINES=1000 \
bash tools/run_mmto_from_dump.sh
```

offline gateを通過した場合でも、まず20局smoke、その後100局ベンチで確認する。新しいゲートは弱い候補を落とすためのものなので、通過だけで採用しない。
