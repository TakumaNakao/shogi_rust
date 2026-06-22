# 三駒関係KPP重み再学習計画

- 作成日時: 2026-06-22 12:08:02 UTC
- 目的: hard-label CE学習の失敗を踏まえ、より質の高い三駒関係KPP重みを作るための研究方針を定める。
- 結論: `policy_weights_v2.1.0.binary` は捨てずに強い初期値・anchorとして残す。一方で、CSA棋譜手をone-hot正解にする長時間学習は凍結し、探索スコア付き候補手のlistwise/ranking学習へ移行する。

## 1. 調査結果

### ローカル実験から分かったこと

直近のguarded CE学習は安全に完走したが、強さは改善しなかった。

- baseline validation: 22.83% (2874/12590), CE 3.980881
- epoch 4 validation: 22.38% (2818/12590), CE 3.980830
- `max|w-w0|`: 0.002660
- `clamped_weights`: 0
- `best.binary` 100局: 42勝54敗4分、score rate 44.00%
- 最終重み40局: 17勝20敗3分、score rate 46.25%

安全制御は効いたが、検証CEの微改善は対局力に変換されなかった。

### 既存コードの性質

`kpp_learn` のCEは「合法手の一手後静的評価にsoftmaxをかけ、棋譜手を正解として押す」方式である。これは探索後の手の価値や兄弟手比較を直接学ぶものではない。

既存の `distill_dump` / `distill_train` は探索スコア付き候補手を扱えるため、次の研究の土台として再利用できる。

### 外部調査からの示唆

Bonanza/MMTO系では、実用AlphaBeta探索プログラムの評価関数を、探索結果が棋譜の指し手選択と整合するように最適化する。Hoki/Kaneko 2014は、線形/非線形の重み付き特徴を対象に、探索結果に基づく目的関数を設計し、4000万超のパラメータを調整したことを報告している。

Bonanza MethodはKPP/KPPTのような玉位置・駒配置の組み合わせ特徴を機械学習で調整し、NNUEにも影響した系統である。

NNUE系の資料では、疎な入力特徴と差分更新により、CPU探索中でも評価関数を強化できることが示されている。ただし本プロジェクトでは、まずKPPの学習目的を探索値に合わせる方が短期の費用対効果が高い。

## 2. なぜ今までの学習が効かなかったか

主因は、学習目標が「強い評価関数」ではなく「棋譜手を一手後静的評価で当てるpolicy」に寄りすぎていたことだと考える。

技術的仮説:

1. hard-label CEは同等に良い手を不正解として押し下げる。
2. 一手後のKPP静的評価だけで棋譜手を当てる目的は、AlphaBeta探索で勝つ目的とずれる。
3. 三駒関係の重みは巨大で疎なので、棋譜手だけの微小更新では局所的なノイズを増やしやすい。
4. 現行重みは探索と噛み合った局所最適に近く、弱い目的関数で動かすとバランスを壊す。
5. validation accuracyやCEは採用指標として弱く、100局以上の対局で崩れやすい。

したがって、今後は「棋譜手一致率」ではなく、「探索上位手を上位に並べる」「悪い手を選ばない」「selected regretを下げる」方向へ目的関数を変える。

## 3. 学習方式の優先順位

### P0: regret-aware listwise KPP trainer

最優先で実装する。

各root局面について、探索で候補手を評価し、次を作る。

- `score_i`: root手番から見た候補手iの探索スコア
- `regret_i = best_score - score_i`
- `q_i = softmax(score_i / teacher_temperature)`

KPP静的評価による候補手分布 `p_i` を `q_i` に近づける。

損失:

- `KL(q || p)` または soft cross entropy
- 補助指標として expected regret
- 必要なら高regret手を強く下げるpairwise marginを追加

狙い:

- 棋譜手one-hotではなく、探索で同等に良い手に確率を残す。
- 探索で明確に悪い手を下げる。
- 三駒関係重みを「探索の葉で有効な静的評価」に近づける。

### P1: pairwise ranking / margin loss

同一root内で、探索スコア差が十分ある候補手ペアだけを使う。

例:

- `score_good - score_bad >= 150cp`
- `eval_good - eval_bad >= margin`

hard-labelよりノイズに強く、KPPの相対順位を直接直せる。

### P2: DAgger型 hard-position replay

候補重みで高regretを出す局面を `value_regret_probe` / `root_decision_probe` で抽出し、次のrankデータへ混ぜる。

少ない計算資源では、全局面を均一に増やすより、失敗局面を再学習する方が期待値が高い。

### P3: MMTO-lite

rootだけでなく探索木内の兄弟局面を保存し、PV leafの特徴差分で更新する。Bonanza/MMTOに近い本命だが、実装負荷が高いため、まずroot listwiseで代替する。

### P4: TinyNNUE residual

KPP-onlyの再学習が複数回ゲートを通らない場合、KPP主評価を維持しつつTinyNNUE residualへ比重を移す。既にNNUE系の基盤はあるため、中長期候補として残す。

## 4. 短期研究計画

### Phase 1: データ・指標基盤

実装対象:

- `distill_train` を拡張、または `kpp_rank_train` を新設。
- teacher_scoresを使うlistwise CE/KLを正式なKPP学習器にする。
- validationに以下を追加:
  - listwise CE/KL
  - selected regret mean / p90 / p95
  - bad regret > 300cp
  - top1一致率
  - hard valid / random valid / loss-tail valid の分離表示
- `anchor_l2`, `max_weight_delta`, `best_checkpoint_path`, `freeze_material` を再利用。

最初のデータ:

- `taya36.sfen`
- `converted_records2016_10818.sfen` からランダム局面
- 現行baseline敗局tailから抽出したhard局面
- まず1k-5k root、depth4、top8/top16

### Phase 2: 小実験

候補:

1. listwise CE, top8, depth4, teacher temperature 100
2. listwise CE, top16, depth4, teacher temperature 150
3. listwise + pairwise margin, top8, depth4
4. 既存重みから5%/10%/20% blend

materialは固定する。既存重みを初期値・anchorとして使い、ゼロから学習しない。

### Phase 3: 採否ゲート

offline gate:

- validation selected regretがbaseline以下
- bad regret > 300cpがbaseline以下
- hard validで悪化しない
- `max|delta|`, p95 deltaが制御範囲内
- `kpp_weight_check` 成功

対局 gate:

- 20局 smokeで50%未満なら破棄
- 40局で55%以上、かつbaseline sweepsがnew sweepsを上回らない
- 100局で55%以上、複数seedの片方が50%未満に落ちない
- 採用前に現行固定重みとの同一engine比較を行う

## 5. 中期研究計画

1. rankデータを10k-50k rootへ拡大。
2. depth5 teacherを一部混ぜる。
3. DAgger式に高regret局面を追加収集。
4. value Huberを補助損失として少量混ぜる。
5. 100局で安定した候補のみ、v2.1.0基準と現行固定基準の両方で評価する。

## 6. 長期研究計画

1. 探索木内の兄弟手比較を保存するMMTO-liteへ進む。
2. PV leaf特徴差分で、探索値が逆転すべき手だけ更新する。
3. KPP-onlyの限界が明確ならTinyNNUE residualへ移行する。
4. KPP重みはNNUEの初期知識またはfallback評価として残す。

## 7. 捨てるもの・残すもの

### 凍結するもの

- `kpp_learn --loss ce` の長時間hard-label CSA学習。
- `Margin` の棋譜手perceptron更新。
- `kpp_self_learn` の盲目的なオンライン自己対局学習。
- validation accuracyだけでbest重みを選ぶ運用。

### 残すもの

- `policy_weights_v2.1.0.binary`
  - 強い初期値、anchor、baselineとして残す。
- `SparseModel`
  - KPP特徴抽出、binary形式、探索評価器として残す。
- `distill_dump` / `distill_train`
  - 探索スコア付き候補手学習の土台として残す。
- `policy_regret_probe` / `value_regret_probe` / `root_decision_probe`
  - offline gateとhard-position収集に使う。
- `usi_benchmark` / `record_analyze`
  - 最終採否に使う。

## 8. 次に作るもの

次の実装単位は `regret-aware listwise KPP trainer` とする。

最小実装:

1. `distill_train` に selected regret指標を追加。
2. teacher_scoresがある場合、探索スコア分布を教師分布として使う。
3. best checkpointを validation selected regret で保存できるようにする。
4. anchor/clamp/freeze_materialを導入する。
5. 小規模rankデータ生成コマンドをREADMEへ追加する。

この段階で、長時間学習には入らない。まず1k-5k rootの小実験でoffline gateと40局gateを通す。

## 参考

- Hoki, Kaneko, "Large-Scale Optimization for Evaluation Functions with Minimax Search", JAIR 2014  
  https://www.jair.org/index.php/jair/article/view/10871
- Bonanza - Chessprogramming Wiki  
  https://www.chessprogramming.org/Bonanza
- Stockfish NNUE documentation  
  https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html
- Yu Nasu NNUE translation repository  
  https://github.com/asdfjkl/nnue
- Kaneko publications  
  https://www.graco.c.u-tokyo.ac.jp/~kaneko/papers/
