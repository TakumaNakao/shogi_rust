# root decision probe 追加と初回解析

- 日時: 2026-06-19 23:24 UTC
- ブランチ: `tooling/root-decision-probe`
- 目的: `v2.4.1` 比較の `BaselineWin` tail で、実戦の `find_best_move(depth=5, time=100ms)` が時間なし探索 teacher と比べてどれだけ悪い root 決定をしているかを分類する。

## 追加したもの

`src/bin/root_decision_probe.rs` を追加した。

棋譜を順に再生し、new 側手番の tail 局面ごとに以下を比較する。

- 棋譜上の実戦手
- 同じ重み・同じ探索での `find_best_move(depth, time_limit_ms)`
- 時間制限なし `alpha_beta_search(teacher_depth)` の teacher 手
- teacher から見た実戦手 regret と timed 手 regret
- timed 探索の `nodes` / `qnodes`

探索本体と評価関数には触れていない。解析専用ツールである。

## 検証

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin root_decision_probe
```

結果: 通過。

## 初回 probe

対象:

```bash
target/release/root_decision_probe \
  --weights policy_weights_v2.1.0.binary \
  --record-dir /tmp/shogi_bench_master_vs_v241_40_seed10101 \
  --only-new-losses \
  --tail-plies 8 \
  --depth 5 \
  --time-limit-ms 100 \
  --teacher-depth 5 \
  --bad-regret-cp 300 \
  --top 20
```

要約:

```text
records probed positions: 40
candidate positions: 40
timed teacher mismatches: 31 (77.50%)
actual teacher mismatches: 30 (75.00%)
timed bad_regret_gt_300: 1 (2.50%)
actual bad_regret_gt_300: 0 (0.00%)
timed mean_regret_cp: 2483.34
timed p90_regret_cp: 0.00
timed p95_regret_cp: 29.44
actual mean_regret_cp: 2.97
actual p90_regret_cp: 0.00
actual p95_regret_cp: 1.86
```

最大の timed regret:

```text
game_010_new_white_BaselineWin.usi ply=139
actual=S*7f
timed=P*3e
teacher=S*7f
teacher_score=-785.2
timed_score=-inf
actual_score=-785.2
timed_regret=99214.8
legal=95 checking=4 timed_nodes=21234 timed_qnodes=19673
sfen 7g1/1R5+bs/2nps2pk/p1P1pp1Pp/9/1P1P1B2P/N2K2+R2/9/L8 w 3GS2N3L2Ps5p 140
```

## 判断

この seed の `BaselineWin` tail では、`timed bad_regret > 300cp` は 1/40 だけだった。したがって、現時点で root ordering や root verify 系の探索変更を広く入れる根拠は弱い。

ただし1局面だけ、100ms探索が teacher と実戦手から外れて即敗勢に落ちるケースがある。これは局所的には有用なテストケースなので、今後 root 時間切れ・aspiration・root候補評価数の計測を追加する場合の固定局面として使える。

`teacher-depth 6` のフル probe は重く、対話作業中には完走しなかった。今後は `--max-records` で分割するか、安価サブエージェントに長時間実験として回す。

## 次の方針

root 系の追加パッチは一旦保留する。次は GPT-5.5 xhigh の第2候補である `worst_drop window` 付き value regression のデータ抽出を検討する。実装する場合も、まず `record_analyze` の window export を追加し、offline valid と regret gate を通してから重み比較へ進む。
