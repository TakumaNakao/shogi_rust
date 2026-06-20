# Hybrid NNUE residual evaluator

- 作成日時: 2026-06-20 01:47:52 UTC
- ブランチ: `tooling/hybrid-nnue-evaluator`
- 目的: 既存KPP評価を置き換えず、TinyNNUEを残差補正として足す実験基盤を追加する。

## 実装

以下を追加した。

- `HybridNnueEvaluator`: `SparseModel + residual_scale * TinyNnueModel`
- `EngineEvaluator::HybridNnue`
- `usi_engine` の `ResidualEvalFile` / `ResidualScale` オプション
- `usi_benchmark` の `--new-residual-weights` / `--baseline-residual-weights` / scale指定
- `eval_profile` / `search_profile` / `value_regret_probe` のhybrid評価対応

この変更は評価関数の新しい読み込み経路と実験CLIを追加するもので、既定動作は従来のKPP評価のまま。

## 残差モデル

一時モデル:

```text
/tmp/tiny_nnue_residual_d4_h64.bin
```

学習条件:

```text
train: /tmp/nnue_taya_depth4_train448.jsonl
valid: /tmp/nnue_taya_depth4_valid64.jsonl
hidden: 64
epochs: 20
target: teacher_score - static_eval
best epoch: 6
valid_rmse: 7.91
valid_mae: 5.10
valid_sign: 67.19%
```

## オフライン評価

`eval_profile`

```text
sparse evals/sec: 281372.05
hybrid scale 0.25 evals/sec: 155793.99
hybrid scale 0.5 evals/sec: 152883.51
hybrid scale 1.0 evals/sec: 149767.71
```

`search_profile`

```text
sparse nodes/sec: 276305.54
hybrid scale 0.25 nodes/sec: 174830.84
hybrid scale 0.5 nodes/sec: 175562.89
```

`value_regret_probe`

```text
scale 0.25:
  mean_regret_cp: 0.20
  p95_regret_cp: 1.05
  max_regret_cp: 2.48
  teacher_move_match: 46/64 = 71.88%

scale 0.5:
  mean_regret_cp: 1.48
  p95_regret_cp: 3.78
  max_regret_cp: 29.03
  teacher_move_match: 38/64 = 59.38%

scale 1.0:
  mean_regret_cp: 14.93
  p95_regret_cp: 72.58
  max_regret_cp: 100.63
  teacher_move_match: 20/64 = 31.25%
```

scale 0.25はroot候補の後悔値が小さく、探索上の破壊は少ない。一方で、評価・探索速度は大きく低下する。

## USI確認

hybrid指定後も通常のUSI出力を返すことを確認した。

```text
info depth 1 score cp -4 time 0 nodes 33 pv 8g8f
bestmove 8g8f
```

## 正規スモークベンチ

誤ってresidual単体を `EvalFile` に指定したベンチは無効とした。正しい比較は、base KPP重みを両者に読み込ませ、new側だけ `ResidualEvalFile` と `ResidualScale` を指定した。

条件:

```text
new: sparse KPP + residual H64 scale 0.25
baseline: sparse KPP
positions: taya36.sfen
games: 10
depth: 5
time-limit-ms: 100
max-plies: 200
seed: 6302
jobs: 4
```

結果:

```text
new wins: 5
baseline wins: 5
draws: 0
new decisive win rate: 50.00%
new total score rate: 50.00%
decisive win rate 95% CI: 23.66%..76.34%
total score rate 95% CI: 19.01%..80.99%
end reasons:
  Resign: 10
paired starts:
  new sweeps: 2
  baseline sweeps: 2
  splits: 1
  draw/mixed pairs: 0
```

## 判断

今回のresidualモデルは不採用。理由は、scale 0.25でも探索速度が約37%落ち、10局スモークで5-5に留まったため。

ただし、hybrid evaluatorとCLI/USI/ベンチ対応は、今後の「KPPを壊さずに小型NNUEや蒸留残差を試す」ための基盤として残す価値がある。次に重み更新を進める場合は、まず残差の学習データを増やし、固定KPP平均との差分をより安定して近似する方針が妥当。
