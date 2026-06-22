# MMTO-lite KPP学習アルゴリズム実装計画

- 作成日時: 2026-06-22 12:15:16 UTC
- 目的: Bonanza/MMTO系の思想に沿って、KPP三駒関係重みを「探索結果と整合するように」最適化する学習基盤を実装する。
- 対象: `/home/nami_ride_trade/shogi_rust`
- 結論: いきなり完全MMTOを作らず、root候補手の探索スコア分布を学ぶ `regret-aware listwise KPP trainer` を最小実装とし、その後に探索木内の兄弟手比較を保存するMMTO-liteへ拡張する。

## 1. 背景

直近のKPP guarded CE学習は、重みを壊さずに完走したが、強さ改善には失敗した。

- baseline validation: 22.83%
- epoch 4 validation: 22.38%
- `best.binary` 100局: 42勝54敗4分、score rate 44.00%
- 最終重み40局: 17勝20敗3分、score rate 46.25%

原因は、現在の `kpp_learn --loss ce` が「棋譜手を一手後の静的評価softmaxで当てる」目的であり、AlphaBeta探索後の手の価値や兄弟手の相対順位を直接学んでいないことだと判断する。

Bonanza/MMTO系の要点は、評価関数のパラメータを、探索結果が棋譜手や強い手選択と整合するように最適化することである。Hoki/Kaneko 2014では、線形/非線形の評価特徴を対象に、探索結果に基づく目的関数を設計し、大規模な評価関数最適化を行っている。

## 2. 実装方針

### 2.1 完全MMTOではなく段階実装にする

完全MMTOは、探索木、PV leaf、兄弟ノード、目的関数、局所最適化を密結合させるため実装負荷が大きい。まずは、既存探索器をteacherとして使い、root候補手の探索スコア分布をKPP静的評価へ蒸留する。

段階:

1. Root listwise KPP training
2. Pairwise ranking補助
3. Hard-position replay
4. Search-tree sibling dump
5. MMTO-lite iterative optimization

### 2.2 既存重みは捨てない

`policy_weights_v2.1.0.binary` は以下の用途で残す。

- teacher探索の初期評価
- 学習初期値
- anchor
- baseline対局相手
- fallback release weight

ゼロからKPPを作り直すのは、計算資源とデータ量に対してリスクが大きい。

### 2.3 採用指標をpolicy accuracyからregretへ移す

今後の主要offline指標:

- selected regret mean / p90 / p95
- bad regret > 300cp
- listwise KL / CE
- pairwise ordering accuracy
- hard valid上のselected regret

`validation accuracy` は参考値に降格する。

## 3. アルゴリズム設計

### 3.1 Root listwise objective

各root局面 `s` について合法手集合 `A(s)` を作る。

teacher探索で候補手を評価する。

```text
teacher_score_i = SearchTeacher(s, a_i)
best_score = max_i teacher_score_i
regret_i = best_score - teacher_score_i
```

教師分布:

```text
q_i = exp((teacher_score_i - best_score) / teacher_temperature) / Z
```

学習対象のKPP静的評価:

```text
model_score_i = EvalKPP(after(s, a_i))
p_i = softmax(model_score_i / model_temperature)
```

実装上の重要点:

- `Position::do_move()` は手番を反転する。
- 候補手後局面をroot手番視点で評価するため、既存 `distill_train` と同じく `do_move(mv)` 後に `switch_turn()` してからKPP特徴を抽出する。
- teacher探索側は、子局面探索結果をroot手番視点へ戻すため既存実装と同じく `-score` する。

主損失:

```text
L_listwise = - sum_i q_i * log(p_i)
```

補助指標:

```text
selected_move = argmax_i model_score_i
selected_regret = best_score - teacher_score_selected
expected_regret = sum_i p_i * regret_i
```

### 3.2 Pairwise margin objective

探索スコア差が十分大きい候補手ペアだけを使う。

```text
if teacher_score_good - teacher_score_bad >= pair_min_gap:
    L_pair = max(0, margin - (model_score_good - model_score_bad))
```

初期実装ではlistwiseのみでよい。pairwiseは、listwiseだけでselected regretが下がらない場合に追加する。

### 3.3 Anchor / clamp

既存重みからの逸脱を制限する。

```text
w <- w + anchor_l2 * (w0 - w)
|w - w0| <= max_weight_delta
```

既存 `kpp_learn` の安全制御を移植する。

### 3.4 Blend評価

学習済み重みを直接採用せず、まず既存重みとのblendを試す。

候補:

```text
blend_ratio = 0.05, 0.10, 0.20
```

`adjust_weights` を再利用する。

## 4. データ形式

### 4.1 JSONL root rank record

新形式 `kpp_rank_v1` を追加する。ただし初期実装では既存 `distill_train` の `teacher_scores` 形式を拡張する形でもよい。後方互換を重視し、既存JSONLを読める状態を保つ。

```json
{
  "schema": "kpp_rank_v1",
  "version": 1,
  "sfen": "position sfen",
  "teacher_depth": 4,
  "teacher_weights": "policy_weights_v2.1.0.binary",
  "root_score": 123.4,
  "candidates": [
    {
      "move": "7g7f",
      "teacher_score": 123.4,
      "rank": 0,
      "regret": 0.0,
      "searched": true
    },
    {
      "move": "2g2f",
      "teacher_score": 82.0,
      "rank": 1,
      "regret": 41.4,
      "searched": true
    }
  ]
}
```

### 4.2 既存distill形式との関係

`distill_dump` の `teacher_scores` は近い形式だが、以下が不足する。

- schema version
- root_score
- rank
- regret
- teacher設定メタデータ
- searched/staticの明示

後方互換のため、`distill_train` は既存形式を読み続ける。新しいMMTO-lite系は `rank_v1` を標準にする。

### 4.3 候補集合の扱い

初期学習では `candidate_scope=scored` を標準にする。

- `scored`: `teacher_scores` に含まれる候補手だけでsoftmaxを作る。
- `legal`: 全合法手をmodel側候補に含める。

`legal` は未採点の良手を誤って押し下げる危険があるため、validで全合法手を採点できる場合だけ使う。

## 5. 新規/拡張CLI設計

### 5.1 `mmto_dump`

目的: root局面から候補手探索スコア付きrankデータを作る。

最初の実装では、新規 `mmto_dump` ではなく既存 `distill_dump` へ `--jsonl-version 2` とrankメタデータを追加する方法でもよい。既存パイプラインを壊さないことを優先する。

入力:

- `--weights`
- `--input`
- `--output`
- `--valid-output`
- `--depth`
- `--teacher-score-top`
- `--teacher-score-source searched|static`
- `--max-positions`
- `--valid-percent`
- `--jobs`
- `--seed`
- `--min-legal-moves`
- `--exclude-in-check`
- `--max-abs-root-score`
- `--max-teacher-gap`
- `--min-teacher-gap`
- `--score-all-legal-for-valid`
- `--jsonl-version`

初期実装:

- `searched` top8/top16を優先。
- 全合法手探索は重いので、まず `teacher_score_top` で制限する。
- 候補集合は `teacher best + 静的上位N` または `全合法手を浅く探索して上位N`。
- validだけは `--score-all-legal-for-valid` で全合法手採点を可能にする。

出力:

- `train.rank.jsonl`
- `valid.rank.jsonl`
- 件数、平均合法手数、平均gap、除外理由集計

### 5.2 `mmto_train`

目的: rank_v1 JSONLからKPP重みを更新する。

主要オプション:

- `--weights`
- `--train`
- `--valid`
- `--extra-valid LABEL=PATH`
- `--output`
- `--epochs`
- `--batch-size`
- `--learning-rate`
- `--model-temperature`
- `--teacher-temperature`
- `--loss listwise|listwise-pairwise`
- `--pair-loss-weight`
- `--pair-min-gap`
- `--pair-margin`
- `--freeze-material`
- `--anchor-l2`
- `--max-weight-delta`
- `--best-checkpoint-path`
- `--best-metric selected-regret|valid-ce|bad-regret`
- `--checkpoint-dir`
- `--checkpoint-every-batches`
- `--log-path`
- `--candidate-scope scored|legal`
- `--bad-regret-cp`

出力ログ:

```text
epoch,batch,train_ce,train_selected_regret_mean,train_bad_regret_300,
valid_ce,valid_top1,valid_selected_regret_mean,valid_p90,valid_p95,
valid_bad_regret_300,valid_expected_regret,outside_scored_candidates,
max_abs_delta,p95_abs_delta,clamped_weights
```

### 5.3 `mmto_stats`

目的: rank_v1データの品質確認。

表示:

- records
- candidates per root
- teacher gap分布
- root score分布
- in-check比率
- high-regret候補比率
- duplicate sfen数

初期実装では `mmto_dump` の末尾集計だけでもよい。

## 6. 実装タスクリスト

### Phase 0: 設計固定

- [x] hard-label CE長時間学習の失敗を記録する。
- [x] 三駒関係KPP再学習方針を記録する。
- [ ] 本実装計画をレビューして `master` に保存する。

### Phase 1: rankデータ形式と統計

- [ ] `src/bin/mmto_dump.rs` を追加する、または既存 `distill_dump` にrank_v1出力を追加する。
- [ ] `RankRecord`, `RankCandidate` serde構造体を定義する。
- [ ] `sfen`, `teacher_score`, `rank`, `regret` を出力する。
- [ ] `--teacher-score-source searched|static` を実装する。
- [ ] `--teacher-score-top` を実装する。
- [ ] `--score-all-legal-for-valid` を実装する。
- [ ] `--jsonl-version` を実装する。
- [ ] `--valid-percent`, `--seed`, `--max-positions` を実装する。
- [ ] dump時に除外理由を集計する。
- [ ] `cargo build --release --bin mmto_dump` を通す。
- [ ] 100局面程度でdry runし、JSONLを目視確認する。

### Phase 2: listwise trainer

- [ ] `src/bin/mmto_train.rs` を追加する。
- [ ] rank_v1 JSONL loaderを実装する。
- [ ] 候補手を合法性チェックし、子局面KPP特徴を抽出する。
- [ ] 子局面特徴抽出では `do_move(mv)` 後に `switch_turn()` し、root手番視点を保つ。
- [ ] teacher distribution `q_i` を作る。
- [ ] model distribution `p_i` を作る。
- [ ] listwise CE/KL lossを実装する。
- [ ] sparse gradientをHashMapで集約する。
- [ ] `freeze_material` を実装する。
- [ ] `anchor_l2`, `max_weight_delta` を移植する。
- [ ] validation selected regretを計算する。
- [ ] `best_checkpoint_path` をselected regret基準で保存できるようにする。
- [ ] NaN/inf検査を入れる。
- [ ] `cargo test --all-targets` を通す。

### Phase 3: offline gate

- [ ] baseline重みのvalid selected regretを表示する。
- [ ] candidate重みのvalid selected regretを表示する。
- [ ] mean / p90 / p95 / bad_regret_300 を出す。
- [ ] `extra-valid random=...` と `extra-valid hard=...` を実装または移植する。
- [ ] `kpp_weight_check` を検証手順に入れる。
- [ ] `search_profile` で速度悪化がないか確認する。

### Phase 4: 小規模実験

- [ ] `taya36.sfen` からrankデータを作る。
- [ ] `converted_records2016_10818.sfen` から1k rootを作る。
- [ ] depth4 top8で学習する。
- [ ] depth4 top16で学習する。
- [ ] teacher temperature 100/150を比較する。
- [ ] 5%/10%/20% blendを作る。
- [ ] 20局 smokeで破棄候補を落とす。
- [ ] 40局 gateで候補を絞る。

### Phase 5: hard-position replay

- [ ] 40局/100局の敗局tailからhard SFENを抽出する。
- [ ] `value_regret_probe` または `root_decision_probe` で高regret局面を集める。
- [ ] random:hard = 3:1 程度のmixed rankデータを作る。
- [ ] hard validでselected regretが下がるか確認する。

### Phase 6: pairwise補助

- [ ] `--loss listwise-pairwise` を追加する。
- [ ] `--pair-min-gap` を追加する。
- [ ] `--pair-margin` を追加する。
- [ ] `--pair-loss-weight` を追加する。
- [ ] pairwise ordering accuracyをvalidationに表示する。
- [ ] listwise単独と20局/40局で比較する。

### Phase 7: MMTO-lite探索木兄弟比較

- [ ] `ShogiAI` に探索ログ収集用の任意フックを追加する。
- [ ] root以外の兄弟手候補を `SiblingRecord` として保存する。
- [ ] PV leafのKPP特徴差分を保存する。
- [ ] alpha-beta cutoffで不完全な候補を除外/低重み化する。
- [ ] exact scoreとbound scoreを区別し、bound scoreを通常の教師値として混ぜない。
- [ ] sibling比較データでpairwise学習する。
- [ ] root rankデータとsiblingデータを混合学習する。

## 7. 採否ゲート

### 7.1 offline gate

候補を対局へ進める条件:

- valid selected regret meanがbaseline以下。
- valid p90/p95 regretがbaseline以下。
- bad regret > 300cpがbaseline以下。
- hard validで悪化しない。
- listwise CE/KLだけが改善してregretが悪化する候補は破棄。
- `kpp_weight_check` が成功する。
- material固定時にmaterial coeffが変化しない。
- `max|w-w0|` とp95 deltaがログに出ている。

### 7.2 対局 gate

- 20局 smoke: 50%未満なら破棄。
- 40局 gate: 55%以上を目安。baseline sweepsがnew sweepsを上回るなら破棄寄り。
- 100局 gate: 55%以上、複数seedで片方が50%未満に落ちない。
- リリース候補: 100局以上で明確に改善し、現行固定版にも勝ち越す。

## 8. リスクと対策

### リスク: teacherが現行重み依存

同じ重みの探索結果を学ぶだけでは、局所最適から抜けにくい。

対策:

- hard-position replayを使う。
- depth5 teacherを一部混ぜる。
- 外部強AI/高rate棋譜手はregret filter後にsoft targetへ混ぜる。

### リスク: offline指標と対局力が乖離する

過去にvalidation CE改善が対局力に結びつかなかった。

対策:

- selected regretを主指標にする。
- 40局では採用しない。
- 100局・複数seedを必須にする。

### リスク: rankデータ生成が重い

全合法手depth4/5探索は高コスト。

対策:

- top8/top16に制限する。
- まずstatic上位候補 + teacher bestから始める。
- searched全合法手は小規模比較に限定する。

### リスク: topN候補漏れ

topN候補に良手が漏れると、未採点手を暗黙に押し下げてしまう。

対策:

- 初期学習は `candidate_scope=scored` に限定する。
- validだけ全合法手採点を行い、topN候補漏れを測る。
- `outside_scored_candidates` をログに出す。

### リスク: 探索木内scoreのbound混入

alpha-betaのcutoffにより、探索木内の兄弟手にはexact scoreとbound scoreが混在する。

対策:

- root listwiseでは明示的に子手を探索し、候補ごとに同じ条件でscoreを作る。
- MMTO-lite探索traceへ進むときは、exact/boundをデータ形式に保存する。
- bound scoreはpairwise教師として使わないか、低重みで扱う。

### リスク: KPPの表現力限界

三駒関係KPPだけでは非線形な局面評価を表現しにくい。

対策:

- KPP-onlyで3回以上gate失敗ならTinyNNUE residualへ比重を移す。
- KPPはfallback/初期知識として残す。

## 9. 最初に実行する具体コマンド案

実装後の想定コマンド:

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release \
  --bin mmto_dump --bin mmto_train --bin adjust_weights \
  --bin kpp_weight_check --bin usi_benchmark --bin record_analyze

RUN_DIR=data/mmto/runs/mmto_rank_d4_top8_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_dump \
  --weights policy_weights_v2.1.0.binary \
  --input converted_records2016_10818.sfen \
  --train-output "$RUN_DIR/train.rank.jsonl" \
  --valid-output "$RUN_DIR/valid.rank.jsonl" \
  --depth 4 \
  --teacher-score-top 8 \
  --teacher-score-source searched \
  --max-positions 5000 \
  --valid-percent 10 \
  --jobs 4 \
  --seed 4101

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_train \
  --weights policy_weights_v2.1.0.binary \
  --train "$RUN_DIR/train.rank.jsonl" \
  --valid "$RUN_DIR/valid.rank.jsonl" \
  --output "$RUN_DIR/mmto_rank_d4_top8.binary" \
  --epochs 2 \
  --batch-size 256 \
  --learning-rate 0.02 \
  --model-temperature 100 \
  --teacher-temperature 100 \
  --loss listwise \
  --freeze-material \
  --anchor-l2 0.0005 \
  --max-weight-delta 0.05 \
  --best-metric selected-regret \
  --best-checkpoint-path "$RUN_DIR/best.binary" \
  --log-path "$RUN_DIR/train.log"
```

## 10. 参考

- Hoki, Kaneko, "Large-Scale Optimization for Evaluation Functions with Minimax Search", JAIR 2014  
  https://www.jair.org/index.php/jair/article/view/10871
- Bonanza - Chessprogramming Wiki  
  https://www.chessprogramming.org/Bonanza
- Stockfish NNUE documentation  
  https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html
- Kaneko publications  
  https://www.graco.c.u-tokyo.ac.jp/~kaneko/papers/
