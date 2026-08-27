# HalfKP scratch v2.5.0超え再計画 v2

- 制定日: 2026-08-23
- 対象lineage: `halfkp64-scratch-v1`
- 学習へ戻す基準モデル: generation 3、SHA-256 `a40af8e3ef913dc78e8272d3e6170d7a801453fa7dadd9b954bb7c945cbb431b`
- 目的: v2.5.0/KPP由来重みを学習へ一切使わず、CPU-onlyでv2.5.0を明確に上回るHalfKP重みを作る
- 旧計画: [`halfkp_v250_breakthrough_plan_20260820.md`](halfkp_v250_breakthrough_plan_20260820.md)
- 状態: 方針更新完了。次の作業はPhase 0のlineage・Gate修復

## 1. 決定

現在のgeneration 6はreleaseしない。generation 4～6は計画上の昇格条件を満たしていないため、accepted本線から外し、
artifactを残したexperimental系列として扱う。学習は、統計的な親改善が最後に確認できたgeneration 3から再開する。

次の主経路は「深い探索値を保持したqsearch PV leaf」と「終局結果」を混合して学習し、その目的関数が親棋力へ転移する
ことを小規模に確認してから、ユニーク局面数・epoch・自己対局世代を増やす方式とする。現行のTreeStrap-lite、棋譜手MMTO、
学習率の単純な上昇をそのまま大規模化しない。

「良いアルゴリズムを長時間動かせば必ず超える」とは実験前に保証できない。しかし、今回までの失敗はHalfKPや
スクラッチ学習の限界を示していない。意図した教師を実際には学習しておらず、昇格判定も計画より緩かったためである。
同じ推論系でv2.5.0が十分強いことから、まず表現容量ではなく教師・目的関数・学習量・評価を正す。

## 2. 実験結果の再解釈

| 証拠 | 結果 | 判断 |
|---|---:|---|
| generation 3 MMTO vs generation 2 | 500局、61.4%、paired 95% CI 56.9～65.8% | 明確なscratch内改善 |
| generation 3 vs v2.5.0 | taya36・両者5秒・50局、14-36、28.0% | 大差が残る |
| generation 4 / 5 / 6 vs各親 | 52.1 / 52.2 / 53.2% | 点推定だけでは改善を確定できない |
| 同paired 95% CI下限 | 47.9 / 48.0 / 48.8% | 3候補とも正本Gate不合格 |
| generation 6 vs v2.5.0 | taya36・両者5秒・50局、8-42、16.0% | release不可。gen3より見かけ上悪化 |
| generation 6教師 | 493,770 root、15,621,588 candidate | candidateは生成したが学習weight 0 |
| generation 6学習 | 2 epoch、`search_mix=1.0` | 約987,540 root提示。終局結果もweight 0 |
| v2.5.0蒸留部 | 32,117,302 sample/epoch × 12 | 385,407,624提示に加え、元KPP学習がある |

MMTO全体では約133.8時間・約249M pairを処理したが、これは主に棋譜手の順位信号であり、強い絶対局面価値とは同一でない。
直近500k生成は通常探索100B nodesとqsearch約106B nodesを消費した一方、学習した絶対価値はroot約100万提示だけだった。
したがって「計算時間をかけたのに弱い」のではなく、計算を未使用candidate生成と弱い自己模倣へ配分していた。

2M WDoor corpusは完成しているが、実学習済みなのは非重複250k pilotだけである。同じ生棋譜手を増やす効果は低そうだが、
2M局面を正しい探索値・終局結果で再ラベルする効果は未検証であり、データ量仮説を棄却してはならない。

## 3. 確定した実装・評価上の問題

### 3.1 qsearch leafを学習していなかった

`halfkp_search_teacher --qsearch-leaf`はleafをcandidateへ保存するが、value lossは常に`record.root`へ掛かる。
さらにteacherはcompleted alpha-betaのroot scoreを捨て、現在モデル自身のqsearch scoreへ置き換える。rank weightが0なので、
保存したleafには一切勾配が入らない。Phase 2の44.4%は、計画していたqsearch-leaf探索値学習の検証ではない。

### 3.2 TreeStrap-liteはTreeStrapではなかった

generation 4～6は、rootの深い探索値を同じrootのstatic評価へ回帰したRootStrap型自己蒸留である。最大32候補は
alpha-beta内部nodeではなく、最善手と手生成順の合法手を別々にqsearchしたものだった。`rank_weight=0`、
`game_rank_weight=0`、`game_vs_search_weight=0`なので全candidateが未使用で、`search_mix=1.0`のため勝敗も未使用だった。

本来のTreeStrapは探索木の全内部nodeを更新し、alpha-betaのexact値とlower/upper boundを区別する。現実装を
TreeStrapの先行研究で正当化することはできない。

### 3.3 可変長leafの手番符号が未実装

現trainerは全candidate utilityを無条件に`-child_eval`とする。これは1手後のchildにだけ正しく、qsearch PVが
可変長ならrootとleafの手番を比較して符号を決める必要がある。実データでもrootとleafが同じ手番の割合は27～35%あり、
rankを有効にすると誤学習する。

### 3.4 promotion Gateが正本より緩かった

正本はscore 52.5%以上かつopening-pair bootstrap 95%下限>50%を要求したが、実装はscore 50%以上と
unpaired Wilson下限47.5%以上の非劣性で昇格させ、paired下限を判定に使っていない。さらに3世代で同じ250 openingを
反復し、100k集合は500k集合に包含された。微差の逐次選択と固定suiteへの適合を、累積棋力向上と誤認しやすい状態だった。

加えてgeneration 3から6までの累積parameter deltaはfeature max `0.02724`、output max `0.004789`で、従来の安全目安
`0.02 / 0.004`を超えた。現在の`passes_safety`は探索失敗・非合法手・非finiteだけを見ており、重み・活性・materialの
healthを昇格判定へ接続していなかった。

### 3.5 真の自己対局学習はまだ行っていない

200局のself-play smokeは同一SHA同士の対局安全確認だけで、教師保存・終局ラベル・重み更新はない。したがって、
on-policy自己対局学習が失敗したという証拠はまだ存在しない。

### 3.6 artifactを上書きしていた

最新のv2.5.0比較は既存run directoryを`FORCE=1`で再利用し、旧artifactを削除した。今後はmodel SHAを含む一意な
directoryを使い、既存runの上書きを禁止する。

## 4. 先行事例から採る原則

1. NNUEの標準的なvalue学習へ戻る。評価値をWDL空間へ写し、search evalとgame resultを固定比率で混合する。
   Stockfishの公式資料もこの形とMSE系lossを説明している。
2. 教師局面は評価関数が探索中に呼ばれる分布へ合わせる。やねうら王の公式手順どおり、局面をqsearch PV leafへ
   置換しても元の評価値を保持し、PV長の偶奇に応じて評価値と勝敗を反転する。
3. 量は必要である。Hoki・KanekoのMMTOは数百万局面・数億手対を使い、Stockfishの強いNNUEも数十億局面規模である。
   50万rootを2 epochだけ回す量で、v2.5.0の長期学習を超える前提を置かない。
4. 自己対局はaccepted scratchモデルから新しいon-policy教師を作る世代型にする。同じ固定WDoor rootを親自身へ
   回帰し続けるだけでは、新しい戦略的情報が入らない。
5. 真のTreeStrapを行うなら、TT内部node、exact値、片側bound lossまで実装する。名称だけを流用しない。

参考一次資料:

- [Stockfish NNUE training documentation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)
- [やねうら王の学習手順](https://github.com/yaneurao/YaneuraOu/wiki/%E3%82%84%E3%81%AD%E3%81%86%E3%82%89%E7%8E%8B%E3%81%AE%E5%AD%A6%E7%BF%92%E6%89%8B%E9%A0%86/1e3194ec5f778d12650ef82a6f3c3001ce30d503)
- [Hoki & Kaneko, Large-Scale Optimization for Evaluation Functions with Minimax Search](https://mlanthology.org/jair/2014/hoki2014jair-largescale/)
- [Veness et al., Bootstrapping from Game Tree Search](https://proceedings.neurips.cc/paper/2009/hash/389bc7bb1e1c2a5e7e147703232a88f6-Abstract.html)
- [Stockfish 14 release notes](https://github.com/official-stockfish/Stockfish/releases/tag/sf_14)

## 5. 新しい作業計画

### Phase 0: lineage・Gate・再現性を修復する

1. generation 4～6をaccepted本線から外し、generation 3を再開点にする。model・教師・対局artifactは削除しない。
2. promotion実装を`score >= 52.5% && paired_bootstrap_lower > 50% && safety_violations == 0`へ統一する。
   safetyには探索異常だけでなく、parameter delta、material bit、activation/score分布、出力上限を含める。
3. 500局で点推定52.5%以上かつCIだけが不確定な候補は、条件を変えず最大1,000局まで延長する。それ以外は不合格とする。
4. openingを事前に少なくともA～Hのdisjoint集合へ分割し、候補ごとに未使用集合を使う。同じ集合を候補選択へ
   反復利用しない。
5. run directoryへ候補・親SHAを含め、既存directoryがあればfail closedにする。

完了条件は、gen4～6の再分類、Gate回帰test、opening split manifest、上書き禁止testである。

### Phase 1: preserved-target qsearch-leafを正しく実装する

新しいteacher semanticsを`deep_search_target_on_qsearch_leaf_v1`として固定する。

1. rootをalpha-beta探索し、最後に完了した反復のscoreとbest moveを保存する。
2. best moveからqsearch PV leafまで進め、学習入力をrootではなくleafにする。
3. deep root scoreは捨てない。`root.side_to_move == leaf.side_to_move`ならそのまま、異なれば符号を反転する。
4. root視点の勝敗も同様に、異なる手番なら`result_leaf = 1 - result_root`へ変換する。
5. qsearch自身のstatic scoreはtargetにせず、診断値としてだけ保存する。
6. mate・repetition・fail-hard boundは通常の両側回帰へ入れない。leaf kindをmanifestへ保存する。
7. search/resultは`0.8/0.2`、校正済み`kappa=1352.7871cp`、MSE系`loss_power=2.5`へ固定する。
8. qsearch候補順位は当面weight 0とし、1 rootにつきbest leafだけ生成する。

偶数・奇数PV、手番符号、勝敗反転、leafへだけ勾配が入ること、finite difference、save/resume identityをtestする。
既存245-rootのgeneration 3教師と500k教師は実装回帰に再利用してよいが、後者は弱いgen5教師なので本学習には使わない。

### Phase 2: 正しい目的関数が棋力へ転移することを証明する

generation 3から教師を新規生成する。まず25k rootをsemantic smoke、次に100k rootを唯一の棋力pilotとする。

- teacher: 200k nodes/root、best qsearch leaf、search/result `0.8/0.2`
- data: game-disjoint、勝敗・phaseを均衡化した2018～2025 WDoor。2026 reserveは触らない
- optimizer: 新目的用の状態を1回だけ初期化し、checkpoint/resumeで継承する。各epoch・世代でzero AdaGradへ戻さない
- LR: feature `3e-4`、output `3e-5`を出発点とし、10k固定subsetでparameter deltaとfinite値を確認後に固定する
- train: checkpoints 1/2/4/8 epoch。候補はfrozen valid lossと安全監査で1個だけ凍結する
- match: Phase 0の未使用opening 500局。必要な場合だけ1,000局へ延長する

25kは実装・loss・安全性の確認であり棋力判定には使わない。100kの点推定が50%未満、または安全違反なら大規模化しない。
点推定50%以上でvalidationも改善するがCI不確定の場合だけ、同じ設定の500k root・最大16 epochへ一度拡張する。
500k候補が厳格Gateを通過して初めて次へ進む。

### Phase 3: ユニーク局面と実学習量を増やす

Phase 2通過後、構築済み2M WDoor rootへ正しい教師を付け直す。生の棋譜手を正解にせず、局面・phase・勝敗だけを使う。

1. frozen 5k subsetで50kと200k nodes教師の符号・score・best move安定性を比較する。
2. 十分一致する最小budgetを50kまたは100k nodesから選び、2Mを50k shard単位で生成する。
3. one-root/one-leaf形式にして、未使用candidate qsearchへ計算を使わない。
4. 8/16/32 epochでatomic checkpointを保存し、game-disjoint validが悪化する前の1候補だけを選ぶ。
5. 2M候補を新しいopening集合で親Gateへ掛ける。不合格ならepochやLRを勝敗に合わせて再探索しない。

32 epochなら最大64Mの絶対価値提示となり、直近の約1M提示より大幅に増える。それでもv2.5.0の蒸留提示数より少ないため、
Phase 2/3の転移が確認できた場合に限り、古いWDoor年度を追加して2～5Mユニーク局面へ広げる。

### Phase 4: 本当の世代型自己対局学習を行う

Phase 3のaccepted scratchだけを対局者・teacherにし、初めてon-policy学習を行う。

1. WDoor由来の多様なopening ballotから20k nodes/手で自己対局する。序盤だけseed固定の小さな探索揺らぎを入れる。
2. 対局中のcompleted root score、best qsearch leaf、leaf手番、最終勝敗を保存する。
3. 最初の世代は250k unique rootで実装確認し、通過後は1～2M unique root/世代へ増やす。
4. 学習データは新規on-policy 80%、過去accepted WDoor/search教師20%とし、分布崩壊を防ぐ。
5. 各世代2～4 epoch、最大6世代。厳格な未使用opening親Gateを通ったモデルだけが次世代teacherになる。
6. 2世代連続不合格、またはheld-outと親対局が同時に悪化したら停止する。

これまでの200局smokeと異なり、教師生成・学習・親対局までを1世代として完結させる。v2.5.0は対局者にも使わない。

### Phase 5: plateau時だけ真のTreeStrapを実装する

preserved-target leafと2世代以上の自己対局が親を改善した後にplateauした場合だけ実施する。

- alpha-beta探索のTTから内部nodeを採取する
- exact値は通常回帰、lower/upper boundは片側lossにする
- rootあたり8～32 nodeへ制限し、重複を除く
- 10k searchのpilotで内部nodeへ勾配が入ることを確認する
- RootStrap・leaf方式と同じ親に対して比較し、厳格Gate合格時だけ拡大する

現在の`root_legal_moves_qsearch_leaf_bucket_v1`はTreeStrapと呼ばず、診断用artifactとして凍結する。

## 6. 評価とrelease条件

- v2.5.0/KPP由来重みはinit、teacher、target、anchor、optimizer、自己対局相手、候補選択に使わない。
- 開発中はaccepted scratch親だけと比較し、候補ごとに未使用openingを使う。
- promotionはscore 52.5%以上、paired bootstrap 95%下限>50%、安全違反0をすべて要求する。
- v2.5.0比較はscratch候補を凍結した後のtaya36・両者5秒・50局screenだけにする。50%以下ならreleaseしない。
- 「大きく上回る」の目標は50局screenで60%以上を得て、別openingの固定node 200局でも55%以上かつ95%下限>50%とする。
- 2026 WDoor reserveは最終候補1個にだけ使う。
- 全runでmodel、optimizer、dataset、binary、Git revisionのSHAを記録し、既存artifactを上書きしない。

## 7. 今は行わないこと

- 現行TreeStrap-lite 500kをそのまま増量する
- LRを`3e-4`より上へ単純に上げる、または大規模hyperparameter gridを行う
- 同じ教師をvalidation悪化後も大量epoch反復する
- 生の棋譜手MMTOだけを2Mへ広げる
- policy head、MCTS、AlphaZero型学習をノートPC CPUへ導入する
- objective修復前にHalfKP幅・層数を増やす

Phase 1～4を正しく実施しても親改善が2回連続で得られない場合に限り、評価値scale、feature表現、HalfKP幅、追加dense層を
NPSと合わせて再監査する。v2.5.0との差がある現時点では、architecture変更より教師修復の期待値が高い。

## 8. 進捗表

| 作業 | 状態 | 完了条件 |
|---|---|---|
| 全実験の再集計 | 完了 | gen3～6、2M、WDL、qsearch、v2.5.0比較を照合 |
| 学習コード監査 | 完了 | root/leaf、符号、candidate weight、optimizer、Gateを照合 |
| 一次資料調査 | 完了 | NNUE、やねうら王、MMTO、TreeStrapを照合 |
| generation 6最終評価 | 完了・不採用 | v2.5.0へ8-42、16.0% |
| generation 4～6 | 完了・実験扱い | strict paired Gate不合格としてregistryから除外 |
| Phase 0 Gate・artifact修復 | 完了（`11b2a73`） | 回帰test、Gate修正、artifact上書き禁止 |
| Phase 1 preserved-target leaf | 実装完了・25k smoke合格 | semantics v4、leaf符号反転、学習検証Brier改善 |
| Phase 2 25k/100k証明 | 完了・Gate未通過 | 100k: 50.8%、paired CI 46.6～55.0%、安全違反0 |
| Phase 3 500k/2M長期学習 | 2Mモデル完成・1.18Mとの直接Gate実行中 | 2Mが1.18Mを上回るかを同一openingで判定 |
| Phase 4 世代型自己対局 | 条件付き | accepted親から1世代を完走・昇格 |
| Phase 5 真TreeStrap | plateau時のみ | 内部node実装とpilot Gate合格 |
| v2.5.0再評価・release | 1.18M基準のtaya36・5秒50局を実行中 | 1.18M基準を確定し、次の修正版比較へ進む |

25k smokeは意味論・損失の技術確認に限定した。generation 3を固定親にした100k root候補はvalidation
Brierを改善し、固定20,000 nodes・500局で50.8%を得たが、paired CI下限が46.6%でstrict Gateを通らなかった。
計画の例外条件（点推定50%以上、validation改善、CI不確定）に該当するため、同一設定で500k root・最大16 epochを
一度だけ実施する。500kでもGate不通過なら、教師量だけを増やさずPhase 1目的関数を再監査する。
長時間処理の後段は`scripts/run_halfkp_preserved_target_phase3.sh`で再開可能にし、教師完了後のvalid生成・重複除外・
学習・Gateを既存artifactを上書きせず自動実行する。

### 500k監査結果（2026-08-24）

500k教師は492,934件（検索失敗2,802、resource limit 3,555）を生成し、train/valid入力重複除外後のvalidは47,947件だった。
教師manifestはsemantics v4 (`deep-search-target-v1`)、`qsearch_leaf=true`、`treestrap_lite=false`、rootの完了探索値保持を示している。
validのleaf手番はrootと同一16,254件、反転31,693件で、可変PVの符号反転経路が実データで使われている。

16 epochのvalidation Brierは0.01110860（25k: 0.01188181、100k: 0.01145858）まで改善したが、固定20,000 nodes・250 opening-pair
(500局) は52.6%、paired bootstrap 95% CI [48.3%, 56.9%]で、`paired_superiority`下限>50%を満たさず不採用となった。
安全違反は0件で、点推定52.5%以上かつCIのみ不確定というPhase 0の延長条件には該当するため、2Mへの増量はせず、未使用の
`final_openings_seed20260730_2000.jsonl`先頭1,000 pairで同条件の延長評価を行う。

目的関数監査では、`value-only`教師が各rootに最善qsearch leaf 1件だけを持つため、game-vs-searchの適格pairは0件となった。
従って`halfkp_search_objective_eval`の「no eligible game-vs-search records」は異常ではなく、このPhaseの設計どおりである。
train logでも`game_vs_search_eligible=0`、`game_rank_eligible=0`を確認した。目的関数の未適用を不具合と誤認してrank教師へ
切り替えず、まず統計的Gateを確定させる。

### 1,000局延長結果（2026-08-25）

未使用の`final_openings_seed20260730_2000.jsonl`から1,000 opening-pair（2,000局）を同じ20,000 nodes/手条件で実行した。
候補は56.2%、paired bootstrap 95% CI [54.025%, 58.375%]、探索失敗・非合法手・non-finite scoreは0件で、
`paired_superiority`を含む厳格Promotion Gateを通過した。これは500k候補をaccepted scratch親として扱う根拠になる。
次は2M raw rootを新規に作り、frozen 5k subsetで50k/200k nodes教師のscore・best move安定性を比較する。
2025年以前を含む2018～2025 WDoor（2026 reserveは未使用）から、accepted generation 4 scratchモデルをteacherとして使う。
KPP由来重みは初期値・教師・評価のいずれにも使用しない。

### 探索budget pilot結果（2026-08-25）

同一5,000-root pilot（実出力は検索完了件数が異なる）を50k/100k/200k nodesで比較した。
50kは4,894件、100kは4,934件、200kは4,953件を生成した。qsearch leaf network-input一致率は50k対100kで76.8%、
100k対200kで82.8%、root scoreのPearson相関はそれぞれ0.966、0.966だった。100k対200kのscore絶対差は平均48.2cp、
中央値0cpで、50k対200kの平均100.5cpより安定していた。
従って2Mの最小十分budgetは100k nodes/rootとし、200kへの倍増は行わない。教師生成は1 root/1 best qsearch leaf、
search/result=0.8/0.2、最大16 epochを固定する。
再開可能な実行本体は`scripts/run_halfkp_phase3_2m_preserved_target.sh`に固定し、raw manifest待機、50k shard化、
teacher生成、validation重複除外、16 epoch学習を順に実行する。各成果物は既存ファイルを上書きしない。

### 1.18M候補のGateと2Mフォールバック（2026-08-25）

1,185,388局面の候補（`model-2m-100k.bin`、実データ不足により2M目標へ未達）を、generation 4親と未使用の
reserve後半1,000 opening-pairで固定20,000 nodes/手評価している。前半1,000 pairとのrecord SHA重複は0である。
raw builderはrate 4,000条件ではopening/middle目標の95%に届かず終了したため、パイプラインが不完全manifestを受理しないよう
`run_halfkp_phase3_2m_preserved_target.sh`をfail-closed化した。Gateが不合格の場合は
`scripts/watch_phase3_gate_and_expand.sh`が自動的にrate 3,000の2018～2025 WDoorを再構築し、2M以上を満たすまで停止したうえで
同じ100k nodes教師・16 epoch学習を再開する。Gate合格時はこの増量を行わない。

### 2M拡張の実行開始（2026-08-26）

1.18M候補はGateを56.45%（paired 95% CI 54.28～58.63%、安全違反0件）で通過したが、データ量の効果を
確認するため、rate 3,000以上の2018～2025 WDoorから2M局面を別runへ構築している。2M候補はgen4親から、
同じteacher semantics v4・100k nodes/root・同じ16 epoch条件で再学習し、1.18M候補と同一Gateおよび
新規holdoutで比較する。既存1.18M成果物は上書きしない。

### 2M対1.18M直接Gate開始（2026-08-27）

2Mモデル（raw 2,000,000局面、train 1,814,343局面）は16 epoch学習を完了した。1.18Mモデルを親にした
同一の固定20,000 nodes/手・1,000 opening-pairで直接比較を開始している。候補優位の点推定52.5%以上、
paired bootstrap 95%下限>50%、安全違反0を要求し、同一データ量実験のため既存1.18M Gateと同じopeningを使う。

### 2M候補のv2.5.0比較開始（2026-08-27）

2M候補をv2.5.0/KPP由来重みと、学習系統外のevaluation-only authorizationで比較している。taya36から
seed固定・SHA選択した25 opening-pair（先後交換50局）、両者5秒/手、同一HalfKP探索バイナリを使用する。
この対局結果は学習・checkpoint選択へ戻さず、v2.5.0を上回る主張の診断およびrelease判定だけに使う。

### 1.18M基準のv2.5.0比較開始（2026-08-28）

2M候補が1.18M候補を上回らなかったため、同じtaya36・seed・5秒/手・25 opening-pair条件で1.18M候補を
v2.5.0と比較している。2Mとの差がデータ量によるものか、両候補に共通する教師・表現の限界かを切り分ける。
