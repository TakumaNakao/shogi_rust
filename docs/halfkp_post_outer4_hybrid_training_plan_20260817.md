# HalfKP scratch outer 4 後の重み強化計画

- 決定日: 2026-08-17
- lineage: `halfkp64-scratch-v1`
- 対象モデル: full MMTO outer 4
- 目的: KPP由来のv2.5.0重みを学習へ一切使わず、scratch HalfKP-64の実戦棋力をv2.5.0より高くする
- 本書の範囲: 次回実装、データ構築、A/B、長期学習、評価、失敗時の次段を事前固定する

## 1. 結論

現outer 4はscratch系列の内部親へ昇格させるが、release候補にはしない。同じ570,875 rootへMMTO outerを
追加するだけの学習も続けない。

次の主経路を以下に固定する。

1. outer 4がscratch親を実時間でも上回るかを、taya36・5秒・先後交換50局で診断する。
2. 現行100k rootで、`MMTO-only`と`MMTO + 小さなterminal WDL補助`を一度だけA/Bする。
3. WDoor 2020～2022を取得し、既存2023～2025と年度横断で重複排除したexpanded dataを作る。2018～2019は
   2020～2022だけでは量が不足するとき、またはWDL用の局面多様性を補うときだけ追加する。
4. A/Bで選んだ目的関数をouter 4から再スタートし、expanded dataで最大3 outer、約1週間の無人学習を行う。
5. scratch内の評価だけで候補を凍結した後、新しい未使用opening seedでv2.5.0と5秒・50局を行う。
6. expanded学習でも差が十分縮まらなければ、WDoorをopening ballotに使うscratch-only自己対局と
   root/PV search bootstrapへ移る。

この順序は、現在確認された「棋譜手順位は改善するが、絶対的な勝敗価値と実時間探索効率が不足する」という
問題を、最小の実装変更から順に解く。architecture拡張、MCTS、policy head、LR grid、係数gridは行わない。

## 2. 現在の到達点

### 2.1 outer 4の正本

| artifact | SHA-256 |
|---|---|
| model | `a40af8e3ef913dc78e8272d3e6170d7a801453fa7dadd9b954bb7c945cbb431b` |
| optimizer | `99a7fb2356d94378712fb49c67d208bb3a85c6040f735e4f0fba00c7befd7fd3` |

このmodelとoptimizerの組をscratch generation 3の学習親として固定する。optimizerをzero-stateへ戻すと、
目的関数の差とoptimizer resetの差が混ざるため、次のA/Bとfull runは両方を継承する。trainerには、modelと
optimizerのSHAおよび同一lineageを検証して初期化する明示的な機能を追加する。

これはscratch系列内部の昇格であり、v2.5.0を上回ったというpromotionではない。

### 2.2 学習として得られた改善

full MMTO outer 1～4は安全に完走し、retryは0だった。

| outer | fixed valid loss | scratch親との500局score |
|---:|---:|---:|
| 1 | 0.185568 | 57.6% |
| 2 | 0.176325 | 57.9% |
| 3 | 0.170967 | 59.0% |
| 4 | **0.167364** | **61.4%** |

outer 4の親対局paired 95%区間は`[56.9%, 65.8%]`である。したがってcorrected MMTOは、少なくとも
固定20,000 nodes条件でscratch親の棋力へ転移した。学習不能、勾配符号反転、更新されていない、といった
致命的失敗を示す結果ではない。

### 2.3 v2.5.0との差

taya36互角局面、5秒/手、50局、25 opening pair先後交換では次の結果だった。

| model | 勝-敗-分 | score | 95%区間 |
|---|---:|---:|---:|
| outer 4 | 14-36-0 | 28.0% | 15.55～40.45% |
| v2.5.0 | 36-14-0 | 72.0% | - |

25 pairの内訳はouter 4の2勝0敗が2 pair、0勝2敗が13 pair、1勝1敗が10 pairだった。非split pairだけの
符号検定も両側`p ≈ 0.0074`であり、単なる先後差や少数openingの偏りでは説明しにくい。outer 4は約
`-164 Elo`相当で、現在のままreleaseしてはならない。

一方、outer 4のqnodes/rootはscratch親より15.95%増えている。固定nodeでの61.4%が5秒対局へ同じように
転移しない可能性があるため、長期学習の前に同じ5秒条件でscratch親との直接対局を一度だけ行う。

## 3. なぜ同じMMTOを増やすだけでは足りないか

現MMTOは、棋譜手のqsearch leafを他の合法手のleafより高くする順位学習である。これは有効だったが、次の
情報を直接与えない。

- その局面から最終的に勝ったか負けたかという絶対価値
- 敗者側の棋譜手が本当に良い手だったか
- 現モデルが自己対局で到達する局面の価値
- 評価値の尺度変化が実時間の枝刈り効率へ与える影響

実際、fixed auditでは勝者側だけでなく敗者側の棋譜手にも強く適合している。また、4 outerでlossと親対局は
単調改善したものの改善幅は逓減し、v2.5.0との5秒対局には大差が残った。従って現在の主因は「致命的実装ミス」
よりも、教師信号の不足、unique局面密度、proxy objectiveと実時間棋力の不一致である。

MMTOの先行研究でも、探索で棋譜手を選ぶよう反復学習し、強い棋譜を使うほど強い評価関数になった一方、数百万の
desired moveを利用している。本runは44,228棋譜を持つがtrain unique rootは570,875である。古い棋譜を
無条件で足すのではなく、高rate・年代・重複を管理してdecision coverageを増やす。
[Hoki & Kaneko, JAIR 2014](https://www.jair.org/index.php/jair/article/download/10871/25935)

## 4. 不変条件

次の条件は全段階で変更しない。

- KPP由来重み、v2.5.0の重みをinit、optimizer、teacher、target、棋譜生成、候補選択へ使わない。
- v2.5.0はscratch候補を凍結した後の最終対局相手にだけ使う。
- HalfKP-64 architecture、material係数、runtime評価形式は変更しない。
- 2026年WDoorは最終評価reserveであり、train、valid、係数校正へ使わない。
- 現行valid 90,759 rootと未開封test 62,233 rootは凍結し、expanded data構築時に再分割しない。
- taya36全体を開発結果として扱い、最終release主張には未使用の2026 WDoor reserveを使う。
- A/B後にLR、WDL係数、seedを結果に合わせてsweepしない。

## 5. Phase 0: 実時間転移の診断

outer 4と直前scratch親を、今回と同じ条件で直接対局させる。

- taya36互角局面集
- 5秒/手
- 25 opening pair、先後交換50局
- threads 1、jobs 4、最大depth 32、最大256 ply
- 新しい固定seed

この対局は候補選択ではなく、qnodes増加の診断である。

| 結果 | 判断 |
|---|---|
| outer 4が50%以上 | MMTO改善は実時間にも転移。Phase 1へ進む |
| 40～50% | 不明確。qnodes/score分布をwarningとしてPhase 1へ進む |
| 40%未満、かつ95%上限も50%未満 | 長期学習を開始せず、score尺度・枝刈り・評価分布を先に監査する |

50局で95%下限50%超を要求しない。小標本を過度に厳しいGateとして使わず、統計的に明白な退行だけを止める。

## 6. Phase 1: WDoor expanded data構築

### 6.1 取得順

WDoor公式archiveから、まず2020～2022を取得する。圧縮容量は合計約979 MBである。公式SHA-256を照合し、
rate統計と有効棋譜数を確認する。2020～2022だけで新規train rootが40万以上得られれば、最初のfull runの
MMTO streamには2018～2019を入れない。

| 年 | 公式表示容量 | 公式SHA-256 |
|---:|---:|---|
| 2018 | 107 MB | `d5bc1e7996ec3e9aa3895e665a2bea588683002a1dd7a5fd3be1598d85eb3867` |
| 2019 | 217 MB | `7e9d10b2e32f27eec57bbde0983a3eea5c847052a70b0dce58349cca6780624b` |
| 2020 | 337 MB | `50ca62725bf7ad96349705c33459929f8602d22d47bcc115d6c3e2dc93a5e3dd` |
| 2021 | 352 MB | `96182a29d30c4f851f20fbcd68a92afc06afa6b991092e78a6a39d5b14abaa60` |
| 2022 | 290 MB | `e0c367f2a5224eed8a70e6b82ca87aaa16cf578ba2c9478c493b78828068cb60` |

40万未満なら、同じ監査を通して2018～2019をMMTO streamにも追加する。40万以上でもWDL A/BがGoなら、
2018～2019をWDL stream用に追加取得するが、棋譜手MMTOには混ぜない。2018～2022全体の圧縮容量は約1.27 GiB、現行年度の
収率を仮定した展開量は約7.8 GiBであり、現在の空き約260 GiBに対して十分小さい。2008～2017はrating anchor、
持時間、エンジン世代の差が大きいため第一runでは使わない。WDoorは2008年から公式棋譜を公開しており、2017年末
以降のrating anchorを基準に2018年以降を最初の旧年度範囲とする。
[WDoor公式archive](https://wdoor.c.u-tokyo.ac.jp/shogi/)、
[WDoor運用履歴](https://wdoor.c.u-tokyo.ac.jp/shogi/history/history.html)

実取得した2020～2022のrate監査は次の通りだった。

| 年 | CSA | 両者rateあり | 両者rate≥4000 |
|---:|---:|---:|---:|
| 2020 | 20,194 | 13,119 | 196 |
| 2021 | 20,056 | 12,817 | 760 |
| 2022 | 20,036 | 14,421 | 1,123 |

したがって旧年度だけで現行trainを数倍にするという当初概算は成立しない。2018～2019を追加しても、旧年度は
MMTOの主データではなくWDLの結果多様化を主用途とする。rate条件を緩和して件数を増やすことはしない。

### 6.2 filter

旧年度にも現行と同じ最低品質条件を適用する。

- 両対局者rate 4,000以上
- 勝敗既知の決着局
- 不正なCSA、違法局面、rate欠損を除外
- in-check rootは静的評価の通常利用箇所と異なるためMMTO/WDLとも第一runでは除外
- 1棋譜の寄与上限を設け、特定の長い棋譜を支配させない
- 年度、勝敗、手番、opening/middle/lateの分布をmanifestへ保存

MMTO streamは現行どおり敗者側ply 80以降を除く。一方WDL streamでは終局結果に近い劣勢局面も価値教師になるため、
敗者側ply 80以降を一律除外せず、late phaseと1棋譜当たりweightの上限で偏りを抑える。

rate閾値は件数不足を理由に自動で下げない。2018～2019のrate 4,000以上が少ない場合、その年度はWDL用または
将来のself-play opening ballot用に留める。

### 6.3 splitと重複排除

異なるdatasetの単純concatは禁止する。既存の結合auditではcross-split HalfKP入力衝突3,790、split内重複
21,144が確認されている。

構築手順を次に固定する。

1. 現行train/valid/testのgame content SHA、opening family、HalfKP network-input hashを読み込む。
2. 現行と同じseed `20260821`、同じopening-family split規則で旧年度を分類する。
3. 既存valid/test familyへ割り当たる旧年度recordはtrainへ入れない。
4. 現行全splitと重なるgame contentおよびnetwork inputを旧年度から除く。
5. 旧年度内でもgame content、opening family、network inputを年度横断で重複排除する。
6. 現行valid/testはそのまま凍結し、旧年度からはtrain familyだけを追加する。
7. 結合後auditでgame、family、network inputのcross-split衝突とwithin-split重複がすべて0であることを
   必須にする。
8. URL、公式SHA、取得日時、license、filter件数、除外理由、出力SHAをmanifestへ固定する。

### 6.4 二つの教師stream

古い棋譜をすべて同じ強さの棋譜手教師として扱わない。

- **MMTO stream:** 2023～2025に、品質監査を通過した2020～2022を追加する。目標は100万～130万unique
  train root。旧年度が40万root未満なら、2018～2019を順に追加する。それでも不足するときだけ、2023～2025の
  train familyについて1棋譜24 rootの上限を48へ増やす。
- **WDL stream:** 2018～2025の決着局を使用できる。ただし年度・棋譜・phase・手番・結果を均衡化し、古い年度が
  総勾配を支配しないよう1棋譜当たりの総WDL weightを正規化する。

この分離により、旧年度は勝敗価値と局面多様性を提供するが、古い棋譜手の模倣がMMTOを無制限に支配しない。

## 7. Phase 2: terminal WDL補助の最小実装と100k A/B

### 7.1 実装内容

`halfkp_mmto_train`へ、任意で有効にできるroot terminal WDL補助を追加する。

- recordの`winner`をroot手番側のtarget `y ∈ {0, 1}`へ変換する。
- root静的評価`score`を`p = sigmoid(score / kappa)`で勝率空間へ写す。
- lossは`(p - y)^2`のBrier lossとする。
- `kappa`は凍結した既存validで1次元校正し、その後は固定する。
- WDL係数は、outer 4 optimizerでpreconditionした初期更新normがMMTO更新normの約10%になる値を一度だけ
  決めて固定する。
- A/Bとfullの両方でMMTO 1 rootにつきWDL 1 rootの`1:1` samplingとし、この比率を変えない。
- 同一棋譜から何root採っても、棋譜ごとのWDL総weightが同じになるよう正規化する。
- material係数、MMTO pair loss、`tau_cp=36.63`、feature LR `1e-4`、output LR比0.1は変えない。

NNUEでcp評価をWDLへ写し、探索評価と終局結果を混ぜる構造には先行実装がある。ただし本計画では過去の
result-only学習で起きた評価尺度の約100倍膨張を避けるため、WDLを10%程度の補助に限定する。
[Stockfish NNUE trainer](https://github.com/official-stockfish/nnue-pytorch)、
[公式NNUE学習文書](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)

追加する最低限の検証は次である。

- black/whiteとwinnerの符号test
- Brier lossとroot gradientのfinite-difference test
- `kappa`、WDL係数、data SHAがresume/checkpoint identityへ入るtest
- outer 4 model/optimizerの組が一致しないとfailするprovenance test
- KPP/v2.5.0をloadしていないことをmanifestで検証するtest
- WDL weight 0で現行MMTOとbyte-identicalになるgolden test
- 中断後resumeがuninterrupted runと同じmodel/optimizerになるtest

### 7.2 A/B条件

expanded full runの前に、既存のbalanced 100k train / 10k validで一度だけ比較する。

| 項目 | control | treatment |
|---|---|---|
| init model | outer 4 | outer 4 |
| init optimizer | outer 4 optimizer | outer 4 optimizer |
| MMTO root | 同じ100k、同じ順序 | 同じ100k、同じ順序 |
| WDL root | なし | 別のbalanced 100kを1:1で処理 |
| seed | 同一 | 同一 |
| MMTO | 有効 | 有効 |
| terminal WDL | 0 | 校正済み固定係数 |
| outer | 1 | 1 |

評価順は、固定valid、6,099-game fixed audit、parameter/score-scale audit、scratch親との固定node 500局、
treatment対controlの固定node 500局、treatment対controlのtaya36・5秒50局とする。pilot modelはfull runへ
継承せず、目的関数を選ぶためだけに使う。

### 7.3 Gate

Gateをhard safety、明白な退行、継続証拠に分ける。小標本の95%下限50%超をすべてに要求しない。

**即時停止:**

- non-finite、違法手、bound rejection、SHA/provenance不一致、material変化
- feature/hidden delta `> 0.02`、output delta `> 0.004`、parameter bound接触
- score絶対値quantileまたはoutput normがcontrol比15%以上膨張
- `delta Brier = treatment - control`のgame-bootstrap 95%区間上限が0以上で、改善を確認できない
- MMTO fixed-audit lossがcontrol比2%以上悪化

**明白な棋力退行として停止:**

- treatment対control 500局scoreが47.5%未満、かつpaired 95%上限も50%未満
- qnodes/rootがcontrol比20%以上増え、5秒対局も改善しない

**fullへ進むための主証拠:**

- hard safetyをすべて通過
- WDL Brierがgame-bootstrapで有意改善
- MMTO loss悪化が2%以内
- treatment対control fixed-node scoreが52.5%以上、またはpaired 95%下限が50%超

5秒50局は診断値とし、50%以上を必須にしない。40～50%はwarningとして条件付きfull outer 1へ進める。
40%未満かつ95%上限50%未満なら明白な実時間退行としてNo-Goにする。qnodes/rootはcontrol比10%超でwarning、
20%超かつ5秒対局も退行した場合だけNo-Goにする。

fixed-node scoreが47.5～52.5%で、offline WDLが明確に改善しscale/qnodesが正常なら、Gateを厳しくしすぎないため
full outer 1を確認試験として許可する。A/Bを再実行したり係数を微調整したりはしない。WDL treatmentがNo-Goでも、
controlが安全かつ棋力退行していなければ、目的関数はMMTO-onlyに戻し、未使用unique rootの効果だけをexpanded
outer 1で確認する。WDLとcontrolの両方がNo-GoならPhase 5へ移る。

## 8. Phase 3: expanded長期学習

### 8.1 固定条件

A/Bの勝者を使って開始する。WDLがGoまたは条件付きGoならhybrid、WDLがNo-Goでcontrolが安全なら
MMTO-onlyとする。

- init: frozen outer 4 model + outer 4 optimizer
- pilot checkpointは使わない
- MMTO train: 100万～130万unique root
- WDL train: hybrid時だけ2018～2025 balanced poolからMMTOと同数を1:1で処理し、1 outerで100万～130万root
- valid/test: 現行の凍結集合
- LR: 上限`1e-4`。後述の固定dose probeで一度だけ安全側へ縮小できる
- output LR ratio: `0.1`
- WDL係数、`kappa`、seed、root順序: A/Bから固定
- hybridは最大3 outer、MMTO-onlyは最大2 outer
- 各outer後にmodel、optimizer、checkpoint、metrics、manifestをimmutable archive化

full開始前に、expanded dataの固定50k MMTO rootと50k WDL rootを最終objective・outer 4 model/optimizerで
一度だけ学習するdose probeを行う。probe modelは破棄し、各parameter groupについて
`projected_delta = probe_delta * sqrt(full_root_count / 50,000)`を記録する。予測deltaが既存上限の75%を超える場合、
`full_lr = 1e-4 * min(1, 0.75 * limit / projected_delta)`の最小値へ一度だけ縮小する。超えなければ`1e-4`を使う。
これは棋力結果に合わせたLR sweepではなく、約60時間後のdelta失敗を避ける事前dose校正である。fullはprobeを
継承せずouter 4 model/optimizerから再開始し、以後LRを変更しない。

現在の実績から、110万rootなら学習約55～60時間/outer、auditと500局対局を含め約58～63時間/outerと見込む。
最大3 outerは約7～8日である。AdaGrad stateを継承し、各outerのdelta Gateでも安全性を確認する。

### 8.2 自動継続条件

各outerは「学習 → archive → fixed audit → scratch generation 3との500局 → Gate」の順で進む。

**hard stop:** Phase 2と同じ安全違反、archive不整合、audit未完了、dataset drift、サービス異常。

**通常停止:**

- generation 3とのscoreが47.5%未満、かつpaired 95%上限50%未満
- fixed MMTO auditが親より改善しない
- hybrid時にWDL Brierが前outerより改善しない
- qnodes/rootがgeneration 3の1.20倍を超え、5秒diagnosticでも改善しない

**次outerを許可:**

- safety、MMTO、scale Gateを通過し、hybrid時はWDL Gateも通過
- generation 3とのscoreが52.5%以上
- paired 95%下限48%以上
- 直前outerよりpoint scoreが2.5 pointを超えて低下していない

47.5～52.5%でofflineが明確に改善している場合、学習量不足を切り分けるため次outerを一度だけ許可する。同じ
不明確判定が2 outer続いたら停止する。outer 4以降、LR再探索、WDL係数再探索は自動実行しない。

### 8.3 無人運用

既存supervisorを再利用し、新workflow modeだけを最小追加する。汎用workflow DSLは作らない。

- locked binary、Git revision、config/data/model/optimizer SHAを起動時と各commit pointで検証
- flockによる二重起動防止、60秒heartbeat、AC電源と40 GiB以上の空き容量preflight
- model/optimizer/checkpoint-lastをfsyncし、outer archiveを上書き不可にする
- audit完了markerとpair単位resumeを維持
- Gate違反は`stopped_by_gate`として正常停止し、次outerを開始しない
- status JSONを進捗の正本にし、現在段階、経過、最新loss、対局数、ETA、artifact SHA、停止理由を保存
- network downloadとdataset buildはservice起動前に完了し、長期serviceからnetwork依存を除く

MMTOはouter途中のweight checkpointを持たない。110万rootでは電源断時に最大約60時間を再実行する可能性がある。
前runは5日2時間をretry 0で完走し、既存のouter単位checkpointは実運用で機能した。固定block checkpointを新設すると
batch境界とsearch snapshotの再現性を変えるため、今回は追加しない。

## 9. Phase 4: 候補選択とv2.5.0評価

長期run完了後、v2.5.0の情報を使わずに次の順でbest scratch checkpointを選ぶ。

1. hard safety合格
2. 凍結MMTO auditが親より改善し、hybrid時はWDL Brierも改善
3. score尺度とqnodesが正常
4. scratch generation 3との固定node 500局で最も高いscore
5. 同点なら直前checkpointとの直接対局で決める

候補をSHA固定した後だけv2.5.0をloadする。最終評価は今回と別の未使用opening集合で行う。taya36はseedを
変えても同じ36局面であり、すでに開発判断へ使ったため再利用しない。

- 2026年最終reserveから事前固定した未使用WDoor互角局面25 pair
- 5秒/手、先後交換50局
- threads 1、jobs 4、最大depth 32、最大256 ply
- model、binary、opening、config、全棋譜のSHAを保存

50局の判断を次に固定する。

| scratch score | 判断 |
|---:|---|
| 40%未満 | 差を十分縮めていない。releaseせずPhase 5へ |
| 40～50% | 明確な進展の可能性はあるがreleaseしない |
| 50%超、95%下限50%以下 | 点推定では上回るが証明不足。この評価だけではreleaseしない |
| 95%下限50%超 | v2.5.0超えのrelease候補 |

今回の28%から40%以上へ上がれば重要な進展だが、目標はあくまで50%超と統計的な再確認である。

## 10. Phase 5: scratch-only自己対局 + root/PV bootstrap

次のいずれかで移行する。

- Phase 2でWDLとMMTO-only controlの両方がNo-Go
- expanded outerがscratch親を改善しない
- 最終v2.5.0対局が40%未満
- offlineだけ改善し実時間棋力が改善しない

最初からTDLeaf、TreeStrap、全探索木保存、AlphaZero/MCTSを実装しない。まず既存alpha-betaへ合う最小構成を使う。

1. accepted scratch同士をWDoor局面から先後交換で自己対局させる。
2. 最初は200局で棋譜、終局label、千日手、投了、resumeを検証する。
3. 次に1,000局を固定20,000 nodes/手で生成し、数plyごとのroot/PVだけ保存する。
4. 保存rootを100,000～200,000 nodesで再探索し、最後に完了したiterative-deepeningのexact値だけを使う。
5. terminal WDLとdeeper-search WDLを混合し、現世代の浅い探索値へbootstrapする。
6. scratch親500局で52.5%以上かつpaired下限48%以上なら5,000局へ拡張する。
7. root bootstrapが改善しつつplateauした場合だけ、時系列を使うTDLeaf(λ)を検討する。

WDoor旧棋譜はこの段階ではopening ballotと終局分布の多様化に使える。v2.5.0/KPPは自己対局相手、teacher、
target、候補選択へ入れない。

## 11. Gitと実装順

実装は次の小さなcommitへ分ける。

1. `training: add bounded terminal WDL auxiliary to MMTO`
   - winner label、Brier gradient、`kappa`/係数固定、optimizer init、unit/golden tests
2. `training: freeze scratch generation 3 and expanded WDoor provenance`
   - outer 4 model/optimizer登録、旧年度download manifest、重複排除、dataset audit
3. `training: add post-outer4 A/B and unattended workflow`
   - A/B、Gate、status、専用config/service、supervisor tests
4. 長期run完了後に`docs: record post-outer4 hybrid results`
   - artifact SHA、各outer結果、停止理由、最終候補を記録

長期run中はGit revision lockを維持するためtracked fileを変更・commitしない。`runs/`とdownload dataはGitへ
入れず、manifest、compactな結果、再現手順だけをcommitする。

## 12. 進捗表

| 項目 | 状態 | 完了条件 |
|---|---|---|
| outer 4結果とv2.5.0 5秒50局監査 | 完了 | 14-36、28.0%、release不可 |
| 次期方針策定 | 完了 | 本書 |
| outer 4をscratch generation 3へ内部登録 | 未着手 | model/optimizer/provenance SHA固定 |
| outer 4対scratch親 5秒50局 | 実行中 | `match_outer4_vs_gen2_taya36_5s_50g_20260817` |
| 2020～2022 download / rate audit | 完了 | 公式SHA・年度別件数を記録 |
| 2018～2019 download / extract | 実行中 | 公式SHA照合済み、展開待ち |
| 2018～2019追加判断 | 決定 | MMTO主データにはせず、WDL用へ限定 |
| expanded dataset build / dedup audit | 未着手 | 全cross-split衝突・重複0 |
| WDL補助とoptimizer init実装 | 完了 | 関連unit test 5件合格、コミット `9826483` |
| 100k control / treatment A/B | 未着手 | fixed audit・500局・5秒50局完了 |
| A/B Gate判断 | 未着手 | full / self-playのどちらかへ一意に決定 |
| expanded outer 1 | 条件付き | A/B Go後に実行・archive・評価 |
| expanded outer 2 / 3 | 条件付き | 各outer Gate通過時だけ実行。outer 3はhybrid時のみ |
| best scratch checkpoint凍結 | 未着手 | v2.5.0非参照の規則で選択 |
| v2.5.0最終5秒50局 | 未着手 | 未使用2026 WDoor reserve 25 pairで完了 |
| releaseまたはself-play移行 | 未着手 | 95%下限50%超、またはPhase 5開始 |

## 13. 今回行わないこと

- 同じ570,875 rootへのouter 5以降
- 古い棋譜の無条件concat
- rate 4,000未満への自動緩和
- result-only学習
- LR、WDL係数、seedのgrid search
- HalfKP hidden拡大、追加layer、policy head
- AlphaZero/MCTSの新規実装
- v2.5.0/KPPをteacherまたは初期値にすること

この計画でv2.5.0超えを保証することはできない。しかし、現時点の証拠からは、同じMMTOをさらに反復するより、
終局価値を小さく加えて目的関数を修正し、高品質なunique局面を増やし、それでも不足した場合にscratch-onlyの
on-policy search bootstrapへ移る方が、CPU-only環境でv2.5.0へ到達する最短かつ検証可能な経路である。
