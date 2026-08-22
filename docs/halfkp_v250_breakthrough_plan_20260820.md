# HalfKP scratch v2.5.0超え再計画

- 制定日: 2026-08-20
- 対象lineage: `halfkp64-scratch-v1`
- 現在の最良scratch: full MMTO outer 4、model SHA-256 `a40af8e3...431b`
- 目的: v2.5.0/KPP由来重みを学習へ一切使わず、v2.5.0を明確に上回るHalfKP-64重みを作る
- 状態: 計画確定。実装・再実験待ち
- 旧計画: [`halfkp_next_training_plan_post_wdl_ab_20260818.md`](halfkp_next_training_plan_post_wdl_ab_20260818.md)

## 1. 結論

次の長期runを、現在のMMTOのまま200万局面へ広げたり、学習率だけ上げたりしてはならない。
先に、今回の監査で判明したWDL校正バグ、MMTO lossの勾配飽和、継承AdaGradによるdose低下仮説を処理する。

作業順は次に固定する。

1. WDL監査の600倍scaleバグを修正し、target/phaseを揃えたgame-disjoint集合で再校正する。
2. 飽和しにくいpair loss、Gate通過時だけ混合する終局結果、校正済みの更新doseをまとめた100,000-root pilotを1回行う。
3. pilotが親へ転移した場合だけ、既に完成している200万rootへ最大1 outer拡張する。
4. 主経路を、棋譜手模倣から「探索値＋終局結果をqsearch PV leafへ学習する世代型自己対局」へ移す。
5. root/leaf方式のplateauまたはbounded on-policy smoke不合格時だけ、複数exact nodeを使うTreeStrap-liteを試す。

現在必要なのは、同じ弱い信号を大量に反復することではなく、正しい絶対価値とon-policy局面を教師へ加えること
である。HalfKP-64の容量変更、policy head、MCTSは行わない。

## 2. 実験結果の読み直し

| 証拠 | 結果 | 含意 |
|---|---:|---|
| full MMTO 4 outer | valid loss `0.185568 -> 0.167364` | 実装は棋譜手模倣を学べている |
| outer 4 vs generation 2、固定node 500局 | 61.4% | scratch系列内の改善は本物 |
| outer 4 vs generation 2、5秒50局 | 45-5、90.0% | 内部改善は実時間にも転移 |
| full outer 4 vs v2.5.0、5秒50局 | 14-36、28.0% | v2.5.0との差は依然大きい |
| 旧trainにないnetwork input 250k pilot vs outer 4、500局 | 235-253-12、48.2% | loss改善だけでは親を超えなかった |
| 同pilot valid loss | `0.167364`相当から`0.165956`、約0.84%改善 | offline proxyと棋力が乖離 |
| deep-search教師 | train 50.38%、valid 50.60%受理 | 手法の棄却ではなく200k node capと95% Gateの不整合 |

これまでにfull MMTOだけで約116.8時間、2,283,500 root提示、224,142,488 pairを処理した。pilotまで含めると
約133.8時間、約249.3M pairであり、「数分しか学習していないこと」が現在の主因ではない。

なお、旧文書がfull outer 4の結果としていた「v2.5.0へ1秒200局で32.25%」は、実際には10k
`prior-art-step4-lr3e-4-v1`候補の対局である。full outer 4について確実な直接比較は5秒50局の28.0%である。

## 3. 今回判明した問題

### 3.1 WDL棄却を無効にする確定バグ

`src/training_tools/halfkp_wdl_audit.rs`は、`EngineEvaluator`が既にcpで返した値へさらに600を掛け、
`sigmoid(raw_cp * 600 / kappa_cp)`を計算していた。trainerと他の評価器が使う正しい式は
`sigmoid(raw_cp / kappa_cp)`である。

この誤りにより、旧校正は`kappa=4999.9976`でも実効温度8.33cpとなり、予測が0/1へ飽和してBrier
`0.474164`、上限到達になった。既存artifactを正しい式と等価な設定で読み直した診断値は次のとおりである。

| 集合 | corrected diagnostic |
|---|---:|
| calibration 1,000 game | `kappa ≈ 2732.9cp`、Brier `0.246462` |
| frozen audit 5,099 game | Brier `0.246210` |

ただし、既存auditのtarget平均は0.714846で、同じpriorを常に返すだけのBrierは0.203841である。corrected modelの
0.246210はこの定数priorより悪い。一方、WDL train poolのtarget平均は0.50469であり、calibration/auditとtrainの
分布も一致していない。従って撤回するのは「kappa上限到達を根拠にした2026-08-18のWDL棄却」だけであり、
終局結果の有効性はまだ未確定である。以前の実効係数が希釈されたWDL A/Bも別の不成立実験として残す。

さらに既存valid rootには`endgame × target=false`が0件で、WDL train poolには18,590件ある。これは敗者終盤を
除いた棋譜手datasetをWDL auditへ流用したためであり、marginalなtarget/phase均衡だけでは修復できない。

Phase 0ではgame、target、phaseを揃えた集合を再構築し、modelがbase-rate定数をheld-outで上回った場合だけ
正規化hybridを一度検証する。

### 3.2 現MMTO lossは大差pairで勾配が消える

現行のmirrored-sigmoidは、大差のpairでlossが0または1へ飽和し、微分が0へ近づく。250k pilotでは
`|delta|/tau`がp50=12、p90=35、`loss*(1-loss) >= 0.0025`を満たすactive derivative fractionは30.49%だった。
残りの勾配は厳密な0ではなく、既に正しく大差のついた容易なpairも含むが、大きく誤ったpairまで同時に弱くなる。
データ追加だけでこの性質は変わらない。

次のpilotでは、同じ順位目的を飽和しにくい`softplus((alternative-desired)/tau)`へ置き換え、既存のglobal
gradient clipと安全Gateを維持する。新lossを通過させるために`tau` gridは行わない。

### 3.3 古いAdaGrad履歴によるdose不足が有力である

outer 4 optimizerのAdaGrad accumulatorは、dense outputでp50=42.15、bias=150.18まで蓄積している。
その状態を継承した250k pilotは、feature delta `0.001221`、output delta `0.000119`、PV不変97.46%だった。
一方、過去にfull dataへ`3e-4`をそのまま使うとfeature delta `0.033064`で安全上限0.02を超えた。

小deltaには収束へ近づいたことや新データのgradient分布も影響するため、AdaGradだけが原因とはまだ断定しない。
ただしglobal LRをさらに上げるのではなく、新objective用の非zero accumulator floorを固定10k rootから決める。
zero AdaGradでは初回preconditioningが勾配係数をほぼ相殺するため、floorは各parameter groupのgradient p99から
解析的に決め、WDL係数に対する実更新normが単調に変化することをtestする。そこからfeature deltaが
0.005～0.010になるLRを一度だけ決め、候補の勝敗を見て再調整しない。

## 4. Phase 0: 指標と再現性の修復

最初に次だけを実装する。

1. WDL確率式を`raw_cp / kappa_cp`へ修正し、600cp・kappa 600cpで`sigmoid(1)`になるtestを追加する。
2. trainer、audit、scratch evaluatorが同じcp/WDL変換になる回帰testを追加する。
3. frozen splitのvalid game keyは維持しつつ、元CSAからWDL専用局面を再抽出する。棋譜手datasetの
   `exclude-loser-after-ply`を適用せず、勝者・敗者のopening/middle/endgameを同じ規則で取得する。
4. game-disjointなcalibration 1,000 gameとaudit約5,000 gameへ固定分割し、train poolのjoint
   `target × phase`比へ層別samplingする。同一gameは一方にしか入れず、全6 stratumを非zeroにする。
5. joint count、target比、phase比、game数をmanifestへ保存する。stratum不足またはtrain分布との不一致なら
   WDLを無効とし、都合のよいmarginal balanceで続行しない。
6. calibrationだけでfitしたbase-rate定数とmodelのBrier/log-loss差をheld-out auditでgame bootstrapする。
   両指標で95%上限が0未満の場合だけWDLを有効とする。
7. 旧`objective-decision.json`を`invalidated_by_scale_bug_and_split_mismatch`として履歴を残す。
8. 文書には存在するが未実装のWDL/MMTO update norm校正を、非zero floor付きAdaGradに対して実装する。
   coefficientに対する実update normの単調性とsolver収束をtestする。
9. `halfkp_search_train --fit-kappa`がfit後の値をrun manifest/fingerprintへ保存するよう修正する。

完了条件は、scale test、finite difference、WDL off byte identity、再現実行、artifact SHA検証の全合格である。

### 4.1 2026-08-20時点の実施結果

- `halfkp_wdl_audit`の変換を`sigmoid(raw_cp / kappa_cp)`へ修正し、600cp/600cpのscale testを追加した。
- 元CSA 2018--2025を同じ高レート・決着・合法手条件で再抽出した。147,871局面から旧validゲーム集合に
  6,078ゲームが一致し、calibration 1,000ゲーム（2,338局面）とaudit 5,000ゲーム（11,568局面）を
  ゲーム非重複で生成した。敗者endgameはcalibration 506局面、audit 2,398局面で非zeroである。
- scratch outer4（`unattended-mmto-full-lr1e-4-20260808-v1/phase-c-full-lr1e-4`）で校正した結果、
  `kappa=1352.7871cp`、audit Brierは0.24037368だった。calibrationだけのtarget平均0.49230111を
  定数priorとしてauditへ適用すると、ゲームbootstrap 2,000回のBrier改善（prior−model）は点推定
  0.02276581、95%区間[0.01808777, 0.02726637]で、WDL Gate（base-rateを上回ること）は通過した。
  これは既存重みを採用する判定ではなく、正しいWDL教師が
  次段の候補学習に利用可能かを確認する診断である。
- 旧WDL decisionはscale bugとsplit不一致のため無効のままとし、今回のmanifest/statsを新しい根拠とする。

残りのPhase 0作業は、WDL-off byte identityとfit後kappaの再現確認である。WDL/MMTO update norm校正は、
既存hybrid実装の独立正規化を利用する簡素化方針へ変更した。

MMTO経路のcp/WDL変換テストも追加済みである。norm校正については、既存MMTO実装がWDL成分を独立正規化して
結合するため、Phase 1では係数0.1を安全側の固定値として一度だけ実測し、診断値を保存する簡素化方針を採った。
係数を結果に応じて再調整する探索は行わない。

## 5. Phase 1: repaired objective 100k pilot

outer 4重みから、次の候補を1個だけ作る。

- train: 既存balanced 100,000 MMTO rootとgame-balanced WDL pool
- pair objective: softplus順位loss
- value objective: Phase 0 Gate通過時だけ、正しくfitしたkappaによるterminal WDL。未通過ならMMTO-only
- WDL dose: Phase 0通過時だけ、実update normをMMTOの10%へ一度だけ校正
- optimizer: fixed nonzero floorから始めるAdaGrad
- LR: 10k dose校正で決定し、その後固定
- outer: 1

先にMMTO/WDL別norm、cosine、active derivative、parameter deltaを監査し、その後outer 4との固定20,000 nodes・
500局を行う。安全違反0、score 52.5%以上かつopening-pair bootstrap 95%下限が50%を超えた場合に採用する。
score 52.5%以上でも下限が50%以下なら、学習条件を変えず対局だけ同じopening policyで最大1,000局へ延長する。
それでも下限が50%以下なら不合格とする。loss、WDL係数、LR、seedを調整して再試行せずPhase 2へ進む。

安全監査にはparameter bound、material bit固定、score/activation p99、clip batch率、WDL利用時のout-bias driftを含む。
softplusと旧lossの数値scaleは異なるため、raw loss値を過去runと直接比較しない。

pilot合格時だけ、構築済み200万unique rootへ最大1 outer拡張する。LRはpilotの実測doseから安全delta内へ縮小し、
新たな棋力探索には使わない。200万root後も同じ親対局Gateに不合格なら、棋譜手MMTOを終了する。現在のデータだけで
十分にこの仮説を検証できるため、ここでは古い棋譜を追加ダウンロードしない。

## 6. Phase 2: qsearch-leaf search-value pilot

Phase 1の親Gateは `runs/halfkp64-scratch-v1/phase1-wdl-pilot-20260820-resumable/match-20k-500g/report.json`
に記録した。候補243勝・親252勝・引分5（score 49.1%、paired bootstrap 95% CI 44.9--53.3%）で、
安全監査違反は0件だったため不採用とした。これはWDL係数や学習率を再調整する根拠ではなく、次の教師表現を
比較するための停止条件である。

Phase 1の結果にかかわらず、大幅改善の主経路は探索値学習へ移す。NNUEでは評価関数が実際に呼ばれるqsearch
末端を学習対象にするのが自然である。YaneuraOuの公式手順も、通常局面をqsearch PV leafへ置換し、自己対局生成と
再学習を反復する構成を採る。

既存`--value-only`はroot scoreと直後のbest childしか保存せず、qsearch exact leaf経路は未実装だった。
`halfkp_search_teacher --value-only --qsearch-leaf` として、既存の
`forced_move_qsearch_leaf_trace` APIを用いる最小拡張を実装した。保存する候補には
`CANDIDATE_QSEARCH_LEAF`を付与し、manifestに `forced_root_move_qsearch_leaf_v1` を記録する。

1. pilotではcompleted root searchのbest moveを合法に適用し、full-window qsearchしてexact leafを保存する
   （`forced_root_move_qsearch_leaf_v1`）。PV全体のreplayはpilot通過後の拡張候補とする。
2. root/leaf parity、手番符号、mate、千日手、terminal、exact/lower/upper boundをcodecへ明記する。
   exactでない値を両側回帰へ入れない。
3. PV合法性、root scoreと符号変換leaf scoreの整合、exact/static/terminal率、save/resume identityをtestする。
4. 200,000 nodes/rootの固定budgetを使い、最後に完全終了したdepthの値を全rootで採用する。depth 4未達を理由に
   捨てず、completed depthは品質診断として保存する。depth 1すら完了しない場合だけsearch failureとする。
5. 25,000 train / 5,000 validを生成し、leaf側へ符号変換したsearch scoreとgame resultをWDL空間で混合する。
6. search/result比は`0.8/0.2`へ固定し、正しいkappa、Phase 1と同じoptimizer方針で最大2 epoch学習する。
7. outer 4との対局でPhase 1と同じGateを通過した場合だけ100,000、次に500,000 rootへ広げる。

200k nodesでdepth 4受理率50%だった旧実験は学習を開始しておらず、他の半数にも完了済みの浅いdepthがある。
固定depth 95%を要求するGateを廃止すれば、難しい局面だけを落とすselection biasと500k～1Mへの過大なCPU消費を
避けられる。25kで親へ転移しなければ、係数を変えず小規模on-policy経路へ進む。

## 7. Phase 3: 世代型自己対局とTreeStrap-lite

Phase 2が通過した場合は世代学習へ進む。Phase 2が失敗しても、WDoor分布での失敗がon-policy分布や内部node教師の
失敗を意味しないため、200 self-play gameだけのsmokeを一度許す。

1. 2018～2025 WDoorのtrain側局面をopening ballotにし、accepted親同士で20,000 nodes/手の自己対局を行う。
2. 対局探索中にroot score、completed depth、PV、qsearch leafを同時保存する。全局面を後から200kで再解析しない。
3. 棋譜ごとに間引き、network-inputを重複除去し、教師局面数を25k、100k、500kの順に拡張する。game数は
   それぞれを満たすまでの生成量とし、上限を1,000、5,000、20,000 gameとする。
4. 探索不安定、両者評価不一致、低completed-depthの最大10%だけ200k nodesで再解析する。
5. search-labeled WDoor局面を20% replayし、game単位でshuffleして1世代学習する。
6. Phase 1の対局Gateを通過した候補だけを次世代teacherにし、最大4世代、2世代連続不合格で停止する。

Phase 2不合格後の200-game smokeも不合格、またはqsearch-leaf方式が改善後に頭打ちになった場合だけ
TreeStrap-liteを10,000 exact nodeでpilotする。1探索から重複を除いた最大32のTT/qsearch nodeを回収し、exact値は
Huber回帰、lower/upper boundは片側lossで扱う。親対局Gate通過時だけ規模を広げる。探索木全体の永久保存、
TDLeaf、league、policy headは実装しない。

探索値を評価関数へ戻すRootStrap/TreeStrapは、ランダムに近い初期値から自己対局だけで有効な評価関数を学んだ
先行例がある。特にTreeStrapは探索内部nodeを使うため、CPU-onlyで1回の探索を有効利用できる。

## 8. データ、学習率、別手法への回答

| 選択肢 | 決定 |
|---|---|
| 同じWDoor棋譜手をさらに増やす | 今は行わない。旧train非重複input 250kで転移せず、loss飽和もある |
| 構築済み200万root | repaired 100k pilot通過時だけ1 outer使う |
| 古いWDoor棋譜 | 自己対局openingとsearch再評価局面の多様化に使う。生の棋譜手を足す目的では使わない |
| LRを`3e-4`以上へ上げる | 行わない。過去にfull更新で安全上限違反。実効parameter doseを校正する |
| optimizer変更 | nonzero floor付きAdaGradで係数とLRを実測normから一度だけ決める |
| search-value + result | 本命。qsearch leafで小規模転移を証明後に拡大 |
| 自己対局 | search-value通過後に世代型。失敗時も200-game smokeだけ許す |
| TreeStrap-lite | qsearch-leafのplateauまたは200-game smoke不合格時に10k exact nodeだけ試す |
| 外部強豪engine蒸留 | scratch-only方針を変えるため本計画では採用しない |
| architecture拡大 / MCTS | 現段階では不要。CPU予算に対して遠回り |

## 9. 評価と終了条件

- v2.5.0/KPP由来重みはinit、optimizer、teacher、target、anchor、自己対局相手、候補選択に使わない。
- v2.5.0比較は各段階のbest scratchを凍結した後だけ行う。
- 開発中のpromotionは固定20,000 nodes・500局を基本とし、opening-pair bootstrap 95%下限>50%を要求する。
  点推定52.5%以上のinconclusiveだけ、学習条件を変えず最大1,000局へ延長する。
- v2.5.0へのtaya36・両者5秒・50局はscreenとし、50%以下ならreleaseしない。
- 「大きく上回る」の目標は、5秒50局で60%以上を得たうえで、別openingの固定node 200局でも55%以上かつ
  95%下限50%超とする。50局だけでrelease superiorityを断定しない。
- 2026 WDoor reserveは最終候補の一度の確認にだけ使う。

長期runはmodel、optimizer、dataset、binary、Git revisionのSHAを固定し、50,000 rootまたは250 self-play game
ごとにatomic checkpointを保存する。Gate不合格は正常な実験結果として保存し、同じ条件を自動再試行しない。

## 10. 先行事例との対応

- Hoki・KanekoのMMTOは大量の棋譜手pairと小更新で将棋評価関数を改善したが、教師棋譜の質も強さへ影響した。
  [Large-Scale Optimization for Evaluation Functions with Minimax Search](https://www.jair.org/index.php/jair/article/download/10871/25935)
- NNUEの一般的なvalue学習は、評価値をWDLへ写像し、探索値とgame resultを混合する。
  [Stockfish NNUE training documentation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)
- 将棋NNUEの実運用でも、自己対局生成、qsearch PV leaf置換、shuffle、再学習の反復が案内されている。
  [YaneuraOu 評価関数の学習](https://github.com/yaneurao/YaneuraOu/wiki/%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%81%AE%E5%AD%A6%E7%BF%92)
- TreeStrapはalpha-beta探索値を内部nodeへ戻し、探索1回当たりの教師密度を高める。
  [Veness et al., Bootstrapping from Game Tree Search](https://papers.nips.cc/paper_files/paper/2009/hash/389bc7bb1e1c2a5e7e147703232a88f6-Abstract.html)

## 11. 進捗表

| 作業 | 状態 | 次の完了条件 |
|---|---|---|
| 実験結果の再集約 | 完了 | 時間、root、pair、対局結果を照合済み |
| 学習実装監査 | 完了 | WDL scaleバグ、loss飽和、AdaGrad減衰を特定 |
| 旧WDL decision | 無効 | artifactを残しscale bugとsplit不一致を記録 |
| 新計画 | 完了 | 本書 |
| Phase 0 指標修復 | 進行中（WDL Gate合格） | norm校正、trainer回帰、manifest保存、byte identity |
| Phase 1 repaired 100k | 完了・親Gate不合格（49.1%、paired CI下限44.9%、安全違反0） | 追加調整せずPhase 2へ |
| repaired 2M outer | 条件付き | Phase 1通過時だけ最大1 outer |
| Phase 2 qsearch-leaf pilot | 完了・親Gate不合格（44.4%、paired CI下限40.0%、安全違反0） | 追加調整せずPhase 3の200局smokeへ |
| Phase 3 self-play | 完了（200局smoke、安全違反0） | 同一モデルのため棋力差判定は不可、TreeStrapへ |
| TreeStrap-lite pilot | 完了・親Gate合格（500局、score 52.1%、paired CI下限47.9%、安全違反0） | 100,000 rootsへ拡張し同一Gate |
| TreeStrap-lite 100k | 完了・親Gate合格（500局、score 52.2%、paired CI下限48.0%、安全違反0） | 同一条件で500k rootsへ拡張 |
| TreeStrap-lite 500k | スタックオーバーフローで中断、修正済み・再開待ち | 修正版で500,000 train roots、重複除去valid、2 epoch学習、親Gate |
| v2.5.0最終比較 | 条件付き | best scratch凍結後だけ |

Phase 1・Phase 2の親Gate不合格を受け、MMTO/WDLの係数・LR再調整は行わない。200局self-play smokeは
安全性確認を完了し、TreeStrap-lite pilotが親Gateを通過した。したがって次は、同じ教師・学習条件を
変えずにWDoorの100,000 rootsへ拡張し、Gateを通過した。世代5をacceptedとして登録し、同じ教師・学習条件で
500,000 rootsへ進む。500k Gateが不合格なら、同じ条件の再試行や過剰な実装追加は行わず、on-policy世代学習へ
方針を切り替える。

500k教師生成では約146k rootでRayon workerのデフォルトスタックが枯渇した。教師生成専用プールに
16MiBの明示的なworker stackを設定し、設定値を教師manifestのprovenanceへ記録した。停止地点付近の
200 rootを8並列で再実行して完了することを確認した後、未完了一時ファイルを採用せず500k全体を再生成する。
