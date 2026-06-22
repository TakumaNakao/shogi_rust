# MMTO-lite hard-negative実装と検証結果

- 作成日時: 2026-06-22 15:32:00 UTC
- 対象ブランチ: `feature/mmto-hard-negative`
- 目的: offline regret改善が対局力に変換されない問題に対し、現モデルが選びそうな高regret悪手を直接叩くhard-negative MMTO-liteを検証する。

## 結論

`mmto_probe` と `mmto_train --loss listwise-hard-negative` の実装は成功したが、今回の重み候補は実戦投入しない。

offline full-legal validでは大きく改善した。最良configは `valid selected regret mean` を 76.99 から 46.68 へ下げた。

しかし、同一engineで重みだけを変えた20局smokeでは全blendが不通過だった。

```text
blend 0.01: 8-11-1, score rate 42.50%
blend 0.02: 9-10-1, score rate 47.50%
blend 0.05: 7-13-0, score rate 35.00%
```

40局gateへは進めない。リリース、タグ付け、重み採用は行わない。

## 実装内容

追加:

```text
src/bin/mmto_probe.rs
```

`mmto_probe` は `kpp_rank_v1` JSONLを読み、現KPP重みが選ぶmodel argmaxとteacher rankingのズレを出力する。

主な出力:

- `model_move`
- `model_rank_by_teacher`
- `model_teacher_score`
- `teacher_best_move`
- `teacher_best_score`
- `selected_regret`
- `model_score`
- `teacher_gap`
- `candidate_count`

`model_rank_by_teacher` は0-basedで、0がteacher bestである。

`mmto_train` には以下を追加した。

```text
--loss listwise-hard-negative
--hard-negative-weight
--hard-negative-margin
--hard-negative-min-regret
--hard-negative-top-model
--hard-negative-top-teacher
--hard-negative-max-pairs-per-sample
```

hard-negativeはlistwise CEを維持したまま、teacher上位手とmodel上位の高regret悪手に対するhinge補助損失を加える。

## 検証

検証データ:

```text
data/mmto/runs/d3_top128_1000_fullvalid_20260622_140845
```

実験run:

```text
data/mmto/runs/hard_negative_sweep_20260622_150747
```

baseline probe:

```text
min-regret 100 / top20: 20件
min-regret 300 / top20: 0件
```

このデータではp95 regretが約265cpのため、`bad-regret-cp 300` は判別力が弱い。

## Offline結果

共通条件:

```text
epochs 5
batch-size 128
learning-rate 50
model-temperature 30
teacher-temperature 100
anchor-l2 0.0001
max-weight-delta 0.2
```

代表結果:

| config | best epoch | valid mean | p90 | p95 | expected | 判定 |
|---|---:|---:|---:|---:|---:|---|
| listwise baseline | 4 | 76.99 -> 69.23 | 260.09 -> 230.18 | 265.19 -> 264.14 | 43.70 -> 41.61 | offline通過 |
| hard-negative C | 1 | 76.99 -> 46.68 | 260.09 -> 204.38 | 265.19 -> 263.00 | 43.70 -> 41.05 | offline通過 |
| hard-negative E | 4 | 76.99 -> 54.56 | 260.09 -> 209.76 | 265.19 -> 264.14 | 43.70 -> 40.16 | offline通過 |

全configがoffline gateを通過した。最良は `hard-negative C`:

```text
--hard-negative-weight 0.10
--hard-negative-min-regret 100
--hard-negative-margin 0.5
--hard-negative-top-model 5
--hard-negative-top-teacher 1
```

## 対局結果

同一 `usi_engine` で重みだけを比較した。

条件:

```text
positions: taya36.sfen
games: 20
depth: 5
time-limit-ms: 100
max-plies: 200
adjudicate-at-max-plies
```

| blend | NewWin | BaselineWin | Draw | score rate | paired starts |
|---|---:|---:|---:|---:|---|
| 0.01 | 8 | 11 | 1 | 42.50% | new sweeps 1 / baseline sweeps 3 / splits 5 / draw-mixed 1 |
| 0.02 | 9 | 10 | 1 | 47.50% | new sweeps 1 / baseline sweeps 1 / splits 7 / draw-mixed 1 |
| 0.05 | 7 | 13 | 0 | 35.00% | new sweeps 0 / baseline sweeps 3 / splits 7 / draw-mixed 0 |

終局理由:

```text
blend 0.01: Resign 19, RepetitionDraw 1
blend 0.02: Resign 19, RepetitionDraw 1
blend 0.05: Resign 20
```

反則や終局矛盾は検出されなかった。

## 判断

hard-negativeはoffline指標を強く改善したが、今回も対局改善へ変換できなかった。

現時点の解釈:

1. `depth3 / 1000局面` のteacher分布に過適合している可能性が高い。
2. full-legal validでも同一dump由来のため、対局分布への汎化を十分に測れていない。
3. `max-weight-delta 0.2` の更新方向には、有害成分がまだ多い。
4. `bad-regret-cp 300` は今回のデータでは高すぎて、有害局面検出指標として弱い。
5. root静的評価だけを合わせても、探索中に本当に必要な評価修正と一致していない可能性がある。

## 後処理

不採用のため、以下の巨大重みは削除した。

```text
best_hard_negative.binary
blend_0.01.binary
blend_0.02.binary
blend_0.05.binary
final_hard_negative.binary
```

run directoryにはCSVログ、probe出力、対局棋譜、record_analyze結果のみを残した。

## 次の方針

hard-negativeそのものは診断・実験基盤として残す。ただし、同じ `depth3/top128/1000` データでさらにweightやmarginを微調整する優先度は低い。

次に優先する案:

1. depth安定教師データを作る。depth3とdepth4でbest/top順位が安定する局面だけを学習する。
2. train dumpとは独立したhard validを作る。対局敗局tailや別seed/別年度局面をvalidに入れる。
3. `bad-regret-cp` をデータ分布に合わせ、100cp/200cpのbad ratioもログに出す。
4. score-space trust regionを追加し、候補手評価値の変化量そのものを制限する。
5. それでも対局へ効かなければ、root近傍sibling dumpまたはKPP residual型の小型NNUEへ進む。

## 検証コマンド

通過:

```text
env RUST_FONTCONFIG_DLOPEN=1 cargo fmt --check
env RUST_FONTCONFIG_DLOPEN=1 cargo check --bin mmto_probe --bin mmto_train
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

