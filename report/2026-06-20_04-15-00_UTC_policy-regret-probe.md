# policy regret probe

- 作成日時: 2026-06-20 04:15:00 UTC
- ブランチ: `tooling/policy-regret-probe`
- 目的: 外部高品質棋譜の教師手が、現行探索から見てどの程度悪手扱いされるかを測る。

## 背景

探索小改良、合法手生成高速化、小規模 value regression は直近で不採用が続いた。

外部wdoor高レート勝者データは offline の policy top1 を改善したが、40局ベンチでは悪化した。次に重み更新を行う前に、外部教師手そのものが現行探索とどの程度矛盾しているかを診断する必要がある。

## 追加ツール

`src/bin/policy_regret_probe.rs` を追加した。

入力は `distill_train` と同じJSONL形式:

```json
{"sfen":"...","teacher_move":"..."}
```

各局面について以下を行う。

1. 現行重みで root を探索し、探索最善手と探索値を得る。
2. 外部教師手を1手指した子局面を探索し、教師手の探索値を得る。
3. `search_score - teacher_score` を regret として集計する。

これにより、外部棋譜手が「現行探索から見ても自然な手」なのか、「探索上は明確に悪い手」なのかを学習前に確認できる。

## 検証

```text
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin policy_regret_probe
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

結果: pass

## データ

```text
/tmp/shogi_external_kifu/datasets_quality/wdoor2026_policy_valid_20k_r4000_winner.jsonl
```

これは wdoor 2026 から高レート勝者・決着局を条件に抽出した教師データ。

## depth3 / 1000件

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/policy_regret_probe \
  --weights policy_weights_v2.1.0.binary \
  --input /tmp/shogi_external_kifu/datasets_quality/wdoor2026_policy_valid_20k_r4000_winner.jsonl \
  --depth 3 \
  --max-positions 1000 \
  --seed 15002 \
  --jobs 4 \
  --show-worst 8
```

```text
samples: 1000
mean_regret_cp: 32.39
p50_regret_cp: 11.98
p90_regret_cp: 100.24
p95_regret_cp: 138.06
max_regret_cp: 320.68
bad_regret_count_gt_300: 1 (0.10%)
teacher_move_match: 218 (21.80%)
```

## depth4 / 300件

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/policy_regret_probe \
  --weights policy_weights_v2.1.0.binary \
  --input /tmp/shogi_external_kifu/datasets_quality/wdoor2026_policy_valid_20k_r4000_winner.jsonl \
  --depth 4 \
  --max-positions 300 \
  --seed 15003 \
  --jobs 4 \
  --show-worst 8
```

```text
samples: 300
mean_regret_cp: 29.11
p50_regret_cp: 3.93
p90_regret_cp: 88.46
p95_regret_cp: 124.35
max_regret_cp: 305.39
bad_regret_count_gt_300: 1 (0.33%)
teacher_move_match: 81 (27.00%)
```

## 解釈

外部高品質棋譜の教師手は、現行探索から見ても大きく破綻していない。

- bad regret は depth3で0.10%、depth4で0.33%。
- p95 regret はおおむね120-140cp程度。
- 一方で探索最善との完全一致率は22-27%程度に留まる。

これは、外部棋譜データが明確な悪手を大量に含むわけではないが、単純な hard-label policy CE で現行探索の最善手分布に寄せるには信号がずれていることを示す。

## 次の方針

外部棋譜を使う場合、単純な `teacher_move` への手一致学習ではなく、以下のように探索値で重み付けする必要がある。

- regretが小さい教師手だけを学習対象にする。
- regretが大きい教師手は除外する。
- 教師手を hard target にせず、現行探索の上位候補と外部教師手を混ぜた比較学習にする。
- policy-onlyで重み全体を動かすのではなく、学習率をさらに落とし、40局以上のゲートを必須にする。

現時点では、外部高品質棋譜は「利用可能だが、そのまま教師手として押し込むのは弱い」という判断。
