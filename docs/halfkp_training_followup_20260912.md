# HalfKP 学習方針の更新 — T13/T15後

決定日: 2026-09-12。調査対象commit: `f367d2b`。
本書は[v4方針](halfkp_training_strategy_v4_20260909.md)のT15以降を更新する。
今回は調査・計画・記録のみ。以下の修正・追加学習は未実行。

## 1. 次の本線

**既存の2,060,908件を使い、同じ価値学習を32 epoch、条件付き64 epochまで実行する。**
早期停止・checkpoint選択を計画どおりに直し、まず学習量不足を検証する。
次の最初の実験では、データ取得・教師生成・LR変更・新optimizerの実装は不要である。

外部AIの評価値・PV・重み、v2.5.0/KPP由来の重みは学習に使わない。
accepted gen6（SHA `162e17ba…`）を初期値・自作探索教師として維持する。
未昇格gen7（SHA `918a7e21…`）は既存実験の比較対象であり、教師へ昇格させない。
自己対局は、ユーザーの希望に従い棋譜学習を検証した後の選択肢とする。

## 2. 実験が示していること

| 実験 | 観測 | 判断 |
|---|---|---|
| gen6対gen5 | screen 55.7%、独立確認56.2%、確認CI 52.0–60.4% | 自作探索教師を更新した価値学習には親改善実績がある |
| gen6対v2.5.0 | taya36、各5秒、50局、17勝33敗 | v2.5.0超えは未達。旧gen5の14勝36敗との差だけでは改善を証明できない |
| T13 gen7対gen6 | 266勝225敗9分、54.1%、CI 49.8–58.4% | 優位性は未確認だが、悪化の証明でもない |
| T15 同じgen7対gen6 | 254勝236敗10分、51.8%、CI 47.5–56.1% | 小幅改善と同等程度の両方が観測と整合する |

T13/T15はいずれも探索失敗・missing root・非合法手・非有限値0。
合計は520勝461敗19分、得点率52.95%。これは記述的集計であり、screenを見て追加した試験を合算して新たな昇格判定にはしない。
「再現screen不合格でofflineと実戦の乖離が確定する」という前回の説明を撤回する。
現時点の適切な結論は、**offlineの改善は確認でき、棋力にも改善の兆候はあるが、規定の優位性確認には届いていない**、である。
同じ候補の確認対局を合格するまで繰り返す作業はここで終了する。

T13のvalidだけを親と再比較した結果（54,693件、同一温度・同一入力）:

| 指標 | gen6親 | gen7 epoch16 |
|---|---:|---:|
| 混合target二乗誤差 | 0.00946764 | 0.00938172 |
| 棋譜の終局結果Brier | 0.20926431 | 0.20790676 |
| 棋譜の終局結果log-loss | 0.63158748 | 0.62822363 |
| 自作探索値とのMAE | 119.51cp | 153.54cp |

Brier/log-lossは棋譜結果を探索leafに引き継いだラベルに対する指標であり、そのleafからの実際の勝率を直接測ったものではない。
MAE悪化も単独では失敗としない。現在の目的は探索値の完全コピーではなく、探索値と棋譜結果を混合した確率targetへの回帰だからである。
ただしvalidのphase別Brierは序盤`0.228724→0.228931`、中盤`0.225866→0.225827`、終盤`0.183199→0.179765`で、改善は主に終盤にある。
序中盤の改善不足は仮説として残し、次runでもphase別に観察する。phaseはrootの手数由来の区分である。

## 3. 学習量と実装上の食い違い

前回のgen6は1,425,255件×24 epoch＝34,206,120提示。
今回のgen7は2,060,908件×16 epoch＝32,974,528提示で、約3.6%少ない。
母集団・教師・初期値も異なるため純粋なデータ量比較ではなく、まして「大量学習でも飽和した」証拠にはならない。
教師生成の長時間稼働とoptimizerの更新量を区別する。

`run_halfkp_v4_gen6_2m_unattended.sh::train_epoch`、trainerソース、モデルmanifest、pipeline.logを照合し、次を確認した。

1. **停止設定:** 計画は微小な横ばいで止めない方針だったが、実際は`early_stop_patience=2`、`min_valid_improvement=1e-6`。
   epoch8→16のvalid二乗誤差は`0.00938571→0.00938172`、結果Brierも`0.20811736→0.20790676`へ改善している。
   32への継続条件を満たしていたのに、実際はepoch13で一度停止し、次stageで再開後epoch16で再停止した。
   「32 epochの実験を済ませた」とは扱えない。
2. **候補選択:** `--save-last-epoch`がbestを上書きする。epoch15のvalid値`0.00938164`はepoch16の`0.00938172`より小さいが、固定候補はepoch16だった。
   停止判定用の最小改善幅と、最良モデルの保存が同じ条件に結び付いている点も直す。対局済み候補の選び直しはしない。
3. **SWA:** 計画は無効化だったが実際はepoch4から計算した。選択モデルはlastで、SWAがこのモデルのAdaGrad更新に混入した証拠はない。
   次runでは明示的に無効化し、不要な評価計算と選択の曖昧さをなくす。
4. **評価記録:** 各stageでtestを評価し、`final_test epoch=32`の表示には実到達epochではなく要求値を使っている。
   testは既に閲覧済みで、今後「全く未使用の最終test」とは呼ばない。継続・checkpoint選定はvalidだけで行う。
5. **再開:** fingerprintにはbinary SHA、停止設定、SWA、test入力が含まれる。
   そのため設定を直して旧checkpointを単純resumeすることはできない。fingerprintの書換えや検証回避で続行しない。

これらは実行制御・選択・報告の問題であり、正しいモデルSHAで行ったT13/T15の対局成績を無効にはしない。
新たな致命的value勾配バグを発見したという意味でもない。過去のrank符号・尺度・旧教師形式の問題はv4の分類を引き継ぐ。

前回の診断過程にある`t14-eval2/registry-candidate.json`は診断用に作られた非正規registryであり、昇格根拠に使わない。
同ディレクトリの1レコード更新後モデルも評価正本にしない。正本は変更していない元モデル同士の`halfkp_value_compare`結果とする。
今後の評価は重み更新を伴わないCLIで行う。

## 4. 次runの具体的方法

### F01 — 最小限の実行制御修正

- trainerのbest保存を「validの厳密な最小値」に揃え、停止判定用min deltaと分離する。同値は早いepochを選ぶ。
- 再開に必要なcurrent/optimizerと、採用候補best/選択metadataを別々に保存する。再開を挟んでもbestのモデルSHA・epoch・scoreが一致することを検証する。
- runnerは4 epoch単位で完了したcheckpointを保存し、`completed_epoch`を基準に進める。非finite・artifact異常で後続stageへ進まない。
- `early-stop-patience=1000000`、`swa-start-epoch=1000000`を明示し、このrunの学習停止は下記のstage規則だけで管理する。
  これらは有限値の横ばいに関する制御であり、非finite等の異常停止は有効なまま。
- 学習中の`--test`を外し、candidate凍結後に既存評価CLIで一度だけ参照評価する。epochの表示も実到達値へ修正する。
- valueの手番反転・有限差分、best保存/再開、stage停止、評価CLIとtrainerの同一fixture指標一致を検証する。
  評価の共通target処理を再利用し、全体の再設計はしない。

### F02 — 同じ2Mで更新量を増やす

旧runを保存し、新しいrunでaccepted gen6から開始する。optimizerは開始時だけ初期化する。
旧epoch16からのresume移行機能を新設するより、同条件の先頭16 epochを再計算する方が作業が小さく、由来も明確である。
gen7を新しい教師・親として扱う必要はない。旧16 epochは再現確認の基準として残す。

| 項目 | 固定値 |
|---|---|
| train | T13の`train.dedup-expanded.hkst`、2,060,908件 |
| valid | T13の`valid.filtered.hkst`、54,693件 |
| init | `v4-value-gen5-20260910/candidate/model-epoch24.bin`（accepted gen6） |
| target | 自作gen6探索値80%＋棋譜終局結果20%、kappa 1352.7871 |
| optimizer / LR | AdaGrad、feature 3e-4、output 3e-5 |
| その他 | batch128、loss power2.5、rank全て0、clip1、output-limit0.20 |
| 並列 / seed | 12 worker、20260911（T13と同じ） |
| 評価保存 | 各epochのvalid/phase指標、4 epochごとのcurrentと再開state、全期間のbest |
| 上限 | 32 epoch、条件付き64 epoch。128は自動実行しない |

32 epochは65,949,056提示、64 epochは131,898,112提示となる。
教師生成済みデータを再利用するため、まず追加の計算時間をモデル更新に配分できる。
run名・データ/モデル/binary SHA・実際の全引数を実行前にconfigへ固定し、モデル/optimizerを途中で初期化し直さない。

4 epoch単位の判定で、全体の混合target誤差と結果Brierが両方とも過去best checkpointより`1e-4`超悪化した状態が2回続いたら、そのstageで終了してbestを保持する。
小さな横ばいだけでは32より前に止めない。
32→64は、32時点までのbestが16時点までのbestに対して混合target誤差を`1e-6`以上改善し、結果Brier/log-lossがそれぞれ`1e-4`超悪化せず、数値異常がない場合に自動続行する。
これらは今回固定する運用基準で、有意差の判定ではない。
phase別指標は毎回保存するが、結果を見ながらその場で別の選択指標へ切り替えない。

### F03 — 候補1個を評価する

候補はgen6初期値を含め、全期間のvalid混合target二乗誤差が最小のモデル1個に固定する。
真の結果指標も併記し、候補を凍結した後に対局する。親がbestなら新候補なしとする。

次runから、**開発screenと昇格確認の役割を分ける**。
開発screenは安全違反0かつ得点率52.5%以上で独立確認へ進める。screen側でCI下限>50%までは要求しない。
独立確認は250 pair/500局を最後まで固定条件で実施し、従来どおり得点率52.5%以上・paired 95%下限>50%・安全違反0の場合だけ昇格する。
50%近辺の数%差を500局ずつで確実に検出できるわけではなく、条件未達は優位性未確認と記録する。
T13/T15の判定にはこの新screen規則を遡及適用しない。

開発screenは使用済みT13の250 openingを開発用として再利用できるが、既知の集合であることを明記する。
確認は全過去runの使用履歴・family重複を調べ、未使用250 pairを学習前に固定する。
reserve 1001–1250は候補範囲であって、未使用を確認せずに採用しない。
各20k nodes/手・先後交換、bootstrap 100,000回、seedとcandidate SHAを事前固定し、途中成績で打切り/延長/候補差替えをしない。
確認で条件未達なら同じ候補の再確認は追加しない。

昇格後にtaya36・両者5秒・50局でv2.5.0と比較する。固定nodeの改善と実時間の優位性を区別する。
50局の僅差だけで「大きく超えた」とはせず、得点率・CI・全終局理由を記録してrelease可否を判断する。

## 5. その次に変えるもの

| 32/64 epochの観測 | 次の方法 |
|---|---|
| valid改善が続き、親改善も独立確認 | 新accepted親で再解析し、WDoor独立棋譜を増やして総unique train4Mへ。80%新規/20%replayを目安に実内訳を記録。10Mは4Mの結果後 |
| train誤差は減るがvalidが持続悪化 | epoch延長を止め、未使用game・年・phaseの多様性を増やす次実験を設計。同じ棋譜内の密度だけを増やさない |
| train/validとも横ばい、実効更新も極小 | 同じgen6初期値・凍結train subset 250k・8 epochでLR 1e-4/3e-4/9e-4（outputは1/10）を同条件比較。有望1設定のみ全2Mへ。短いsubsetで棋力を断定しない |
| 更新は十分でもofflineしか改善しない | train由来固定5k rootを同じgen6で100k対300k nodes、depth上限8対12で再解析。completed depth、root/leaf整合、結果較正、時間を比較し、より深い教師が有益か診断する |
| 結果重視による序中盤の悪化が持続 | 同じデータ・同じ初期値・同じ提示数でsearch mix0.8対0.9の単一比較を検討。現在は結果指標も改善しており、直ちに0.5へ変更する根拠は弱い |
| 稀な特徴だけに誤差集中 | 既存feature factorization実装の適用可能性を調べ、小規模比較後に判断 |

複数条件を同時変更しない。LR比較・深い教師の小規模診断は、その後に親Gateを伴う本実験へ進むための候補選別であり、それだけで有効性を認定しない。
棋譜量・十分な更新・教師品質を試しても改善しない場合に、accepted scratch同士の自己対局と棋譜replayを検討する。
現状から「新algorithmなら必ずv2.5.0を超える」とも「棋譜学習が限界」とも断定できない。

## 6. 作業順・完了条件

| ID | 作業 | 完了条件 | 状態 |
|---|---|---|---|
| F00 | T15集計・方針更新 | report SHA、実更新量、計画差分を記録 | 完了 |
| F01 | 停止・best保存・評価/再開の最小修正 | 上記回帰確認、configとコードが一致 | 実装・回帰テスト完了 |
| F02 | 同じ2Mで32、条件付き64 epoch | 中間重み/optimizer/指標保存、停止理由自動記録 | 実行中（extended runner） |
| F03 | 候補固定、親screen・条件付き独立確認 | 最多250＋250 pair、昇格判断記録 | 完了待ち（gate runner待機中） |
| F04 | 昇格時v2.5.0比較、次規模決定 | 50局結果、4M条件または原因別次実験を記録 | 条件付き |

実装: [extended runner](../scripts/run_halfkp_v4_gen6_2m_extended.sh)を使い、既存のlock・端末切断に耐える起動・完全artifact再開を使う。
学習完了後は[gen7 gate runner](../scripts/run_halfkp_v4_gen7_gate_unattended.sh)が完了artifactを検証し、F03のscreen→条件付き独立確認を自動で進める。条件未達や異常も終了理由を保存する。
放置中に停止理由を無視して別LR・別教師・4Mへ移る処理は作らない。

## 7. 根拠

- [T08学習結果](../training/halfkp64-scratch-v1/v4_t08_training_report_20260910.json)
- [T13モデルmanifest](../runs/halfkp64-scratch-v1/v4-value-gen6-2m-20260911/model.bin.manifest.json)、[pipeline.log](../runs/halfkp64-scratch-v1/v4-value-gen6-2m-20260911/pipeline.log)
- [T13/offline集計](../training/halfkp64-scratch-v1/v4_t13_gen7_screen_offline_20260912.json)、[T15と今回の判断記録](../training/halfkp64-scratch-v1/v4_t15_review_20260912.json)
- [runner](../scripts/run_halfkp_v4_gen6_2m_unattended.sh)、[trainer](../src/training_tools/halfkp_search_train.rs)
- [Stockfish公式NNUE解説・loss](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html#loss-functions-and-how-to-apply-them)（2026-09-12再確認）: 評価値を確率に変換して実結果と混合する方法を説明している。尺度はengine/data依存。現在の価値学習の方向を支持するが、このrepoのLR・混合率・epochを保証するものではない。
- [同・factorization](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html#real-effect-of-the-factorizer): 主に初期学習・疎な特徴への効果を説明している。今回の次実験に直ちに追加する根拠とはしない。
