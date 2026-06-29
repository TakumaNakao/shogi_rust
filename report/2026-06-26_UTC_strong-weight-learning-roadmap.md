# 本質的に強い評価関数重みを作るための研究計画

- 作成日: 2026-06-26 UTC
- 対象: `/home/nami_ride_trade/shogi_rust`
- 方針転換: 短期的な20局/40局改善ではなく、プロ棋士級を目標にした強い評価関数学習基盤の開発を主目的にする。

## 結論

短期ベンチで少し勝つ重みではなく、評価関数そのものの質を上げる。既存 `policy_weights_v2.1.0.binary` は必要なら初期値として使うが、最終目的は既存重みに小さなパッチを当てることではない。

本命は以下の2本立てにする。

1. Bonanza/MMTO系の探索整合学習を本格化し、KPP重みを「棋譜手」ではなく「探索結果と一致する評価関数」に近づける。
2. KPPの限界が見えた時点で、小型NNUEから本格NNUEへ移行し、CPU探索に耐える非線形評価関数を作る。

短期の勝率改善、序盤ブック、手書き囲いボーナスは補助策に留める。主目的は強い重みを作る学習アルゴリズムである。

## なぜ現在のCE棋譜手学習では足りないか

現在の `kpp_learn --loss ce` は、局面から棋譜手を当てる教師あり学習である。これは有用だが、強い評価関数を作る目的とはずれがある。

- 棋譜手一致は「評価値が正しい」ことを直接保証しない。
- 負け側の自然な手、序盤の定跡的手、終盤の唯一手が同じ目的関数で混ざる。
- 浅い静的評価で候補手をsoftmaxするため、探索後に良い手かどうかを十分に反映しない。
- KPPは巨大疎特徴なので、短時間のscratch学習では十分な特徴が更新されない。
- 既存v2.1.0は2日以上の学習で得られており、数epochの短期実験で超えられないのは自然である。

直近のscratch比較でも、ゼロ初期化はvalidation accuracyを `15.97%` から `19.00%` まで伸ばしたが、v2.1.0 warm-start初期値 `22.83%` には届かなかった。これはscratch学習が失敗したというより、巨大KPPを短時間CEだけで育てるには学習量も目的関数も不足していることを示す。

## 参考にする既存研究

### Bonanza / MMTO

Hoki and Kaneko, "Large-Scale Optimization for Evaluation Functions with Minimax Search", JAIR 2014 は、AlphaBeta探索プログラムの評価関数を、探索結果が専門家棋譜の手選択と一致するように最適化する。論文では4,000万を超えるパラメータを少数のハイパーパラメータで調整し、Bonanzaの2013年世界コンピュータ将棋選手権優勝に寄与したとされる。

本プロジェクトへの示唆:

- 棋譜手CEだけではなく、探索木上の兄弟手比較を目的関数に入れる。
- 静的評価ではなく、探索で選ばれる手が正しくなるように重みを調整する。
- 巨大KPPの重み更新には、局面数・探索木・hard negativeが必要である。

### NNUE

NNUEは将棋発祥のCPU向けニューラル評価であり、疎入力と差分更新によりAlphaBeta探索内で使える。現在のコードにはすでに `TinyNnueModel`, `HybridNnueEvaluator`, `nnue_feature_dump`, `nnue_rank_dump` があり、完全新規ではなく段階的に進められる。

本プロジェクトへの示唆:

- KPP線形評価だけでは、非線形な形勢判断に限界がある。
- 小型NNUEを残差評価として入れ、効果が見えたら本格NNUEへ拡張する。
- データ品質が重要で、静かな局面、探索値が安定した局面、戦型・局面段階の多様性を確保する必要がある。

## 長期研究フェーズ

### Phase 0: 学習基盤の再定義

目的:

- 短期ベンチ改善用の実験ではなく、数日単位で回せる再現可能な学習パイプラインを作る。

実装するもの:

- 学習run manifest: 入力データ、フィルタ、初期重み、学習率、seed、git commit、出力重みsha256を保存。
- 中断再開可能なteacher dump。
- データセット統計: 年、レート、手数帯、先後、戦型近似、勝敗、重複率。
- 重み検査: NaN/inf、最大差分、評価値分布、局面段階別の評価変化。

完了条件:

- 2日以上の学習runを再現可能に開始・停止・再開できる。
- 学習後に、重み・ログ・manifestから同じ実験条件を復元できる。

### Phase 1: 高品質データセット作成

目的:

- 棋譜をそのまま流すのではなく、評価関数学習に向いた局面集合を作る。

データ源:

- Wdoor高レートCSA 2023-2026。
- 追加で取得可能な強豪AI棋譜。
- ライセンス確認できるプロ棋譜または公開棋譜。
- 将来的には強化後エンジンの自己対局。

フィルタ:

- `min_player_rate >= 4000` を基本。
- decisive game優先。
- 負け側終盤手の除外。
- in-check局面の扱いを分離。
- qsearch/static差、depth違い探索差で不安定局面を除外または別ラベル化。
- 序盤、中盤、終盤を均等または意図的比率でサンプリング。

成果物:

- `data/.../positions/*.jsonl` または `.sfen`
- `dataset_manifest.json`
- `dataset_stats.json`

### Phase 2: 本格MMTO-liteからMMTOへ

目的:

- KPP重みを探索結果に整合させる。

現在のMMTO-liteの問題:

- teacher dumpが小さい。
- root候補だけに寄り、探索木全体の兄弟手比較が薄い。
- 微小差分を保守的にかけているため、対局に効くほどの更新になりにくい。
- selected regretやrerank gateは診断には有用だが、学習信号としてまだ弱い。

次に実装するもの:

1. 探索木兄弟ノードdump
   - rootだけでなく、PV上の複数深さで兄弟手と探索値を保存する。
   - `parent_sfen`, `root_turn`, `candidate_move`, `searched_score`, `static_features` を持つ。

2. Bonanza型ペアワイズ目的関数
   - expert/teacher best moveが、探索で悪い候補より高くなるようにmargin lossをかける。
   - regretが大きい候補を強く重み付けする。

3. 反復式 teacher refresh
   - 一度dumpして終わりではなく、世代ごとにteacher searchを更新する。
   - `weights_g0 -> dump_g1 -> train_g1 -> gate -> weights_g1` の形。

4. 大規模hard negative mining
   - 現行モデルが好むがteacherが嫌う手を優先的に集める。
   - 序盤、中盤、終盤で別々に集計する。

成功条件:

- offline gateだけでは採用しない。
- 重み差分が十分に生まれ、かつscore gateで破綻しない。
- 20/40局は途中確認に留め、最終は100局以上。
- 複数seedで改善しなければ世代更新を止める。

### Phase 3: NNUE評価関数の本格化

目的:

- KPP線形評価の表現力限界を超える。

最初の設計:

- 入力: 既存 `extract_nnue_features` を拡張したHalfKP/KingPiece系。
- 構造: 小型MLPから開始し、CPU探索で耐えるサイズにする。
- 学習対象: 探索値、WDL、teacher PV評価、棋譜手ランキング。
- 推論: 最初はfloatで正しさ優先、その後int16/int8量子化。
- リリース時は重みファイルを1つにまとめる。複数巨大重みを前提にしない。

段階:

1. 現行 `TinyNnueModel` の学習器を整備する。
2. KPP+NNUE residualで対局評価する。
3. 効果が見えたら単体NNUEまたはKPP互換ラッパーへ移行する。
4. 差分Accumulatorを実装する。
5. 量子化してNPS低下を抑える。

成功条件:

- eval/search profileでNPS低下を測る。
- 同一探索時間で対局改善する。
- 評価値分布がKPPより安定する。
- 序盤・中盤・終盤の局面別regretが改善する。

### Phase 4: 長時間学習run

目的:

- 2日以上の学習を前提にした本番runを行う。

実行前チェック:

- データセット固定。
- manifest生成。
- checkpoint保存間隔と容量上限を設定。
- 最良checkpointだけを残す自動清掃。
- validation split固定。
- 学習曲線をCSV/JSONで保存。

候補run:

- KPP-MMTO世代学習: 24-48時間。
- NNUE value/rank学習: 24-72時間。
- hybrid KPP+NNUE residual: 24時間。

採用条件:

- offline validation改善。
- score gate通過。
- probeで重大な悪化なし。
- 100局以上の対局検証。
- 重みファイルはリリース候補1つだけ残す。

## 直近で実装すべきタスク

優先度順:

1. `training_manifest` の導入
   - すべての長時間学習runにmanifestを出す。

2. `teacher_position_dump` の安定化
   - 中断再開、chunk、重複除去、quiet filter、局面段階タグを持つ。

3. `tree_sibling_dump` の拡張
   - rootだけでなくPV周辺の兄弟手を保存する。

4. `mmto_full_train` の設計
   - Bonanza型pairwise/listwise loss。
   - hard negative weighting。
   - 世代更新可能なcheckpoint形式。

5. `nnue_value_train` の整備
   - 既存TinyNNUEを本格学習できるPython/Rustパイプラインにする。

6. `weight_gate_suite`
   - KPP/NNUE共通で、score distribution、root regret、search profile、対局smokeを一括実行する。

## やらないこと

- 10局だけで候補重みを採用しない。
- 手書きで囲いKPPを大きく盛らない。
- 弱い現行評価だけで自己分析定跡を大量生成しない。
- 低品質な大量局面をそのまま学習に流さない。
- 重みファイルを複数巨大アセットとしてリリースしない。

## 判断

重み学習は諦めない。むしろ、本質的な強さを目指すなら重み学習が中心である。ただし、現在のCE教師あり学習を長く回すだけでは足りない。

今後は「強い重みを作るための学習基盤」を主開発対象にする。短期の探索改善や序盤ブックは補助に回し、評価関数学習のデータ品質、目的関数、teacher探索、NNUE化に開発資源を集中する。

## 参考文献・資料

- Hoki, K. and Kaneko, T. "Large-Scale Optimization for Evaluation Functions with Minimax Search", JAIR, 2014. https://www.jair.org/index.php/jair/article/view/10871
- Nasu, Y. "Efficiently Updatable Neural-Network-based Evaluation Function for Computer Shogi" translation repository. https://github.com/asdfjkl/nnue
- "Study of the Proper NNUE Dataset", arXiv:2412.17948. https://arxiv.org/html/2412.17948v1
