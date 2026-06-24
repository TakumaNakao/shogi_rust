# MMTO hard-negative 学習基盤の改善

- 作成日時: 2026-06-24 14:30 UTC
- ブランチ: `improve-mmto-training-density`

## 背景

`mmto_rerank_long_20260623_223743` は `score gate` を通過したが、`rerank gate` で以下が悪化したため不採用になった。

- mean regret: `4.08 -> 4.14`
- p90 regret: `11.76 -> 12.22`
- p95 regret: `22.99 -> 24.01`

主因は、学習ペアが少なすぎたこと。10000局面の長時間runで `train pairs=2450`、`valid pairs=279` しかなく、ほとんどの局面が更新に寄与していなかった。

## 実装内容

`mmto_tree_train` に `--bad-candidate-scope` を追加した。

- `student-top`: 従来互換。dump時点のstudent上位手だけをbad候補にする。
- `model-top`: 現在の学習中モデルが高く評価するbad候補を使う。
- `all-candidates`: 候補集合内の全bad候補を使う。

長時間用スクリプト `tools/run_mmto_rerank_long.sh` は `model-top` を標準にした。

主な新デフォルト:

- `BAD_CANDIDATE_SCOPE=model-top`
- `STUDENT_BAD_TOP_K=12`
- `MIN_REGRET_CP=15`
- `MAX_PAIRS_PER_SAMPLE=32`
- `SCORE_ALL_LEGAL_FOR_VALID=1`
- `EPOCHS=10`

また、`rerank gate` 失敗時でも `hard_positions.sfen` を保存するようにした。これにより失敗局面を次段のDAgger風hard-position学習へ回せる。

追加スクリプト:

- `tools/make_wdoor_mmto_positions.sh`
  - Wdoor高レートCSAからMMTO用SFEN集合を作る。
- `tools/run_mmto_hard_stage.sh`
  - 失敗runの `hard_positions.sfen` と `best.raw.binary` を使って2段目のhard-position学習を回す。

## Smoke結果

同一の300局面dumpでペア密度を比較した。

- 従来設定: train `74` pairs / valid `22` pairs
- `all-candidates`: train `1372` pairs / valid `1112` pairs
- `model-top`: train `1018` pairs / valid `570` pairs

`model-top` は従来比で10倍以上の学習信号を作れている。小規模1epoch更新も正常終了した。

## 次の実行方針

まずは既存2016局面集合で新しい長時間runを行う。

```bash
bash tools/run_mmto_rerank_long.sh
```

Wdoor高レート局面を使う場合:

```bash
bash tools/make_wdoor_mmto_positions.sh
POSITIONS=data/mmto/positions/wdoor2023_2026_r4000_p16_120.sfen bash tools/run_mmto_rerank_long.sh
```

`rerank gate` が失敗したが `hard_positions.sfen` が得られた場合:

```bash
BASE_RUN_DIR=data/mmto/runs/mmto_rerank_long_<timestamp> bash tools/run_mmto_hard_stage.sh
```

採用条件は従来通り、`score gate` と `rerank gate` の両方を通過し、その後にUSI対局ベンチで確認すること。
