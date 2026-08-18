# HalfKP scratch 次期重み学習計画（WDL A/B後）

- 決定日: 2026-08-18
- lineage: `halfkp64-scratch-v1`
- 現行内部親: scratch generation 3 / full MMTO outer 4
- 目的: KPP由来重みを学習へ一切使わず、CPU-onlyでv2.5.0を上回るHalfKP-64重みを作る
- 実行方針: 小さな係数探索を繰り返さず、計画逸脱を修正した一度のA/B、Bonanza規模へ近づける
  dense MMTO、deep-search bootstrapの順に進む
- 実装・実験手順書:
  [`halfkp_post_wdl_dense_execution_runbook_20260818.md`](halfkp_post_wdl_dense_execution_runbook_20260818.md)

本書は目的、根拠、停止条件の正本である。実装file、CLI、command、artifact、Gate順、checkpoint、障害復旧、
進捗の運用正本は上記手順書とする。

## 1. 結論

現時点の重みはreleaseしない。scratch系列内では明確に強くなったが、v2.5.0に対する改善はまだ確認できない。
次の主経路は、次の三段階に固定する。

1. 実効寄与がほぼゼロだったterminal WDL補助を正規化し、計画どおりのA/Bを一度だけ完了する。
2. 2023～2025年の高rate棋譜から、1棋譜24 rootという人工的な上限を96 rootへ広げ、未使用局面を中心に
   200万～260万train rootを構築する。必要な場合だけ2020～2022年を最大25%まで補う。
3. outer 4 model/optimizerから最大2 outerのdense MMTOを行う。scratch親を改善できなければ停止し、最良候補が
   v2.5.0へ50%を超えない場合はscratch-onlyのdeep-search bootstrapへ移る。

新architecture、policy head、MCTS、AlphaZero型学習はこの段階では実装しない。HalfKP-64自体の表現能力は、
同一architectureのv2.5.0重みが既に示している。現在の主要な不足は、学習に使った棋譜局面密度と、終局価値・
探索値を正しく与える目的関数である。

## 2. 現在の証拠

### 2.1 MMTO改善は本物だが、外部目標との差は縮まっていない

| 比較 | 条件 | scratch側結果 | 判断 |
|---|---|---:|---|
| outer 4 vs generation 2 | fixed 20,000 nodes、500局 | score 61.4% | MMTO更新方向は有効 |
| outer 4 vs generation 2 | taya36、5秒、50局 | 45-5、90.0% | 実時間にも転移 |
| generation 2 vs v2.5.0 | 1秒、200局 | 61-139、30.5% | 大差あり |
| outer 4 vs v2.5.0 | 1秒、200局 | 64-135-1、32.25% | +1.75 point、未確定 |
| outer 4 vs v2.5.0 | taya36、5秒、50局 | 14-36、28.0% | release不可 |
| WDL treatment vs v2.5.0 | taya36、5秒、50局 | 16-34、32.0% | 別seedかつ小標本、改善証拠ではない |

outer 4の内部改善は固定nodeと実時間の両方で確認できる。一方、v2.5.0に対する1秒200局の改善は1.75 pointに
留まり、5秒50局の28%と32%は別seedで直接比較できない。従って「少しずつv2.5.0へ近づいている」と断定せず、
「scratch系列は改善したが、外部目標への改善は未証明」と扱う。

### 2.2 今回のWDL A/Bはterminal WDL仮説を検証できていない

controlとtreatmentの結果は次のとおりだった。

| 指標 | control | treatment | treatment-control |
|---|---:|---:|---:|
| valid MMTO loss | 0.162741052 | 0.162741448 | +0.000000396 |
| valid WDL Brier after | 0.263490477 | 0.263482985 | -0.000007492 |
| feature max delta | 0.000784032 | 0.000784032 | ほぼ同一 |
| qnodes | 2,427,546,499 | 2,427,630,812 | +0.0035% |

共通の学習前Brierは0.262956525であり、control、treatmentとも学習後は悪化した。treatmentの優位は、その
悪化量がcontrolより0.000007492小さかったという差に留まり、WDL校正が改善した証拠ではない。

比較対象を固定する正本SHAは次のとおりである。

| artifact | SHA-256 |
|---|---|
| control model | `fb87fa7627aa6d8f00be3e637a6ef1d18822e1d31f3898ff064ac26365b65359` |
| treatment model | `abb4761ed6f9bf1b0c600ea26018af42f57687092fb7d8984d0a1b93767de355` |
| treatment vs v2.5.0 match log | `45e9ab046c795537a1d62b7cbc481cfa95bba953489a6513752e317cff05017d` |

実装では100,000 WDL rootの勾配を10,375,690 MMTO pairと同じ勾配へ足し、最後にpair数だけで除算していた。
単純な件数換算でも、名目`wdl_weight=0.1`は

```text
0.1 * 100,000 / 10,375,690 = 0.0009638
```

に希釈される。実モデル比較でもtreatment-control差のL2 normはcontrol更新量の3.74%、更新方向cosineは
0.999300だった。さらに、事前計画にあった別WDL pool、棋譜ごとの総weight一定化、AdaGrad precondition後の
10% norm校正、game-cluster bootstrap、kappa校正が実装されていなかった。

従って棄却するのは現行の`wdl_weight=0.1`実装であり、正規化したterminal WDL仮説ではない。16-34モデルは
generation 4へ昇格させず、dense学習の親にも使わない。

### 2.3 学習量は先行MMTOより大幅に少ない

Hoki & KanekoのMMTOは47,566棋譜から4,467,116 desired movesと368,313,024 move pairsを作り、
nominal depth 1 + quiescence searchを反復した。現runは34,582 train棋譜を持つが、1棋譜24 rootの上限により
570,875 desired roots、約56,035,622 pair/outerに留まる。

| 項目 | Bonanza MMTO | 現run |
|---|---:|---:|
| train棋譜 | 47,566 | 34,582 |
| desired move/root | 4,467,116 | 570,875 |
| 1棋譜当たり | 約93.9 | 約16.5 |
| move pairs / outer | 368,313,024 | 約56,035,622 |
| search refresh | 最大200 iteration | 4 outer |

棋譜数は同程度だが、局面密度は約1/5.7、総rootは約1/7.8である。現MMTOが4 outer連続でvalid lossと
scratch親対局を改善した以上、同じ570,875 rootをさらに反復するのではなく、同じ高品質棋譜内の未使用局面を
初めて提示することが、先行事例に最も忠実な次手である。

[Hoki & Kaneko, Large-Scale Optimization for Evaluation Functions with Minimax Search](https://www.jair.org/index.php/jair/article/download/10871/25935)

## 3. 不変条件

- v2.5.0/KPP由来重みをinit、optimizer、teacher、target、anchor、自己対局相手へ使わない。
- v2.5.0はscratch内で候補を凍結した後の比較相手にだけ使う。
- 現行HalfKP-64 architecture、material係数、runtime codecを変えない。
- 2026年WDoorは最終評価reserveとし、train、valid、係数校正へ使わない。
- 現行valid 90,759 rootとtest 62,233 rootを凍結し、trainへ移さない。
- taya36は開発済み集合として扱い、最終release主張には使わない。
- seed、LR、WDL係数、mix率のgrid searchを行わない。
- すべての長期runでmodel、optimizer、dataset、binary、Git revisionのSHAを固定する。

## 4. Phase A: 計画準拠terminal WDL A/Bを一度だけ完了する

### 4.1 最小修正

`halfkp_mmto_train`のWDL経路だけを次のように修正する。WDL weight 0のMMTO経路は変更しない。

1. MMTO勾配とWDL勾配を別々に蓄積する。
2. MMTOはpair数、WDLは棋譜weight合計で別々に平均する。
3. WDL入力を`--wdl-train`で別に与える。2018～2025年の両者rate 4,000以上の決着局から、1棋譜につき
   opening/middle/late各最大1 rootを取り、凍結valid/test familyと2026年を除外する。network-inputを重複排除し、
   年度・棋譜・phase・手番・勝敗を固定seedで均衡化した最大100,000 rootを正本poolとする。
4. WDL samplerは棋譜を一様に選び、その棋譜内のrootを一様に選ぶ。これにより1棋譜の期待weightを一定にする。
   MMTO 1 rootにつきWDL 1 rootの`1:1` samplingをA/B、dense pilot、dense fullのすべてで維持する。
5. 左右対称projection後、outer 4 optimizerによるAdaGrad precondition後update normを別々に測る。
6. 固定10,000-root校正集合と正本の`1:1` samplerで、WDL update normがMMTOの10%になる係数を数式で一度だけ
   求める。dense時も係数、pool、sampling比を変えない。
7. raw norm、preconditioned norm、cosine、実効係数、棋譜数、WDL pool SHAをmanifestへ保存する。
8. validをgame hashでcalibration 1,000 gameとaudit 5,099 gameへ固定分割する。`kappa`は学習開始前にouter 4
   modelだけを使ってcalibration側のBrierを最小化する有界1次元最適化で一度だけ決める。control、treatment、
   denseの学習とauditで同じ値を使い、各modelへ再fitしない。
9. audit側ではgame単位のpaired Brier/log-loss差とbootstrap 95%区間を保存する。

回帰testは、winner/手番符号、finite difference、棋譜weight合計、pair/root別正規化、precondition後norm、
WDL 0 byte identity、resume identity、manifest identityを含める。

### 4.2 実行条件

- init: outer 4 model SHA `a40af8e3...431b`
- init optimizer SHA `99a7fb23...5fd3`
- MMTO: 既存balanced 100k、同じ順序
- WDL: 別のgame-balanced最大100k、同一棋譜総weight 1
- LR: `1e-4`、output比0.1、1 outer
- WDL norm比: 校正済み10%。結果を見て変更しない
- control: 既存`fb87fa76...5359`を使用する。ただし変更前binaryをSHA付きで保存し、変更後のWDL 0経路が
  10k replayで旧binaryとbyte一致した場合だけ

### 4.3 WDL採用Gate

次をすべて満たした場合だけdense runへWDLを入れる。

- audit 5,099 gameのBrier差`treatment-control`でbootstrap 95%上限 < 0
- log-lossが非退行
- 凍結90,759-root MMTO auditの相対悪化が0.5%以内
- NaN、Inf、違法手、bound rejection、provenance不一致が0
- feature/hidden delta ≤ 0.02、output delta ≤ 0.004
- controlとのfixed 20,000 nodes・500局でscore ≥ 52.5%、opening-pair bootstrap 95%下限 ≥ 48%
- 5秒50局は明白な実時間退行だけを止める。score < 40%かつopening-pair bootstrap 95%上限 < 50%ならNo-Go

一つでも失敗すればterminal WDLを本lineageで終了する。係数を100倍した再試験、kappa再調整、別seed試験は
行わず、以後はMMTO-onlyで進む。

## 5. Phase B: Bonanza規模へ近づけるdense dataset

### 5.1 recent dense train

2023～2025年の現行rate 4,000以上棋譜を再構築し、次だけを変更する。

- `max-records-per-game`: 24 → 96
- phase cap: `8/12/4` → `32/48/16`。現行のopening/middle/late比`2:3:1`を維持する
- 現manifestのglobal `--max-records 1100000`は撤廃する。先頭からの全体打切りで分布を歪めない
- それ以外のrate、決着、ply、in-check、合法手、敗者側late除外、split seedは維持する

全CSAを盤面依存filter前まで再集計すると、cap 24の候補1,079,848 rootに対しcap 96では4,101,852 root
（2023: 1,327,675、2024: 1,498,626、2025: 1,275,551）が上限になる。現行の候補からunique trainへの保持率を
機械的に当てた中心見積りは約217万train rootであり、300万は
重複を無視した理論上限に近い。従って目標を200万～260万unique train root、最低実行量を180万、旧trainに
存在しなかったnetwork inputを最低120万とする。実数はbuild後の監査で確定し、都合のよい件数を仮定しない。
260万を超えて得られた場合は捨てず、最大300万まで使用する。300万を超えた場合だけ、全filterと重複排除後に
固定seedで棋譜・phaseを均衡化して300万へ縮小し、入力順によるprefix cutoffは行わない。

既存570,875 train rootは1回だけreplayとして残す。現行trainに属するgame SHA/opening familyはtrainへ固定し、
同じ棋譜内の未使用positionを追加できる。現行valid/testのgame SHA/opening familyは全positionをtrainから除外する。
既存全splitのnetwork-input hashを先に読み、既存train rootそのものは1回だけ保持し、supplementで同じhashが出た
場合とvalid/test hashへ衝突した場合は除外する。生成後auditでは次をすべて0にする。

- game content cross-split collision
- opening family cross-split collision
- network-input cross-split collision
- split内network-input重複
- 違法teacher move、未知結果、非決着、provenance不一致

### 5.2 旧年度の扱い

完全展開後の全CSAを再監査した結果は次のとおりである。

| 年 | CSA | 両者rateあり | 両者rate≥4000 | 決着局 |
|---:|---:|---:|---:|---:|
| 2018 | 47,813 | 32,984 | 1,006 | 790 |
| 2019 | 106,411 | 81,466 | 1,490 | 1,233 |
| 2020 | 139,358 | 99,598 | 2,529 | 2,085 |
| 2021 | 123,717 | 85,069 | 9,595 | 7,685 |
| 2022 | 129,846 | 105,283 | 4,589 | 3,607 |

以前の2020～2022統計は展開途中の各約20,000 CSAだけを読んだ値であり無効とする。

recentだけでtrain 200万root未満の場合に限り、2020～2022を同じbuilderへ追加し、200万へ達するか候補を使い
切った時点で止める。ただし旧年度rootは最終train rootの25%以下とし、2018～2019はMMTO教師へ入れない。
2018～2019はWDL多様化または将来の自己対局opening ballotにのみ使う。rate閾値を3,500へ下げて量を作ることは
しない。追加後も総train 180万未満、または新規network input 120万未満ならdense MMTOを開始せずPhase Dへ移る。

### 5.3 unseen-root pilot

10日規模のfull outerを開始する前に、新datasetから旧trainとnetwork inputが重ならない250,000 rootを
game/phase-balancedに固定抽出する。

- init: outer 4 model + outer 4 optimizer
- objective: Phase A採用なら正本WDL poolを`1:1`で循環samplingするhybrid、不採用ならMMTO-only
- LR: `1e-4`
- outer: 1
- valid/audit: 現行凍結集合
- pilot重みは目的確認専用で、full runへ継承しない

Go条件は、安全Gate合格、凍結audit loss差のgame-bootstrap 95%上限 < 0、outer 4とのfixed-node 500局で
score ≥ 52.5%かつopening-pair bootstrap 95%下限 ≥ 48%とする。すべてのGo条件を満たした場合だけdense fullへ
進む。満たさなければ候補と理由を保存して`stopped_by_gate`とし、再試行せずPhase Dへ移る。

## 6. Phase C: dense long MMTO

pilot通過時だけ、outer 4 model/optimizerから再開始する。

- train: 想定200万～260万、最低180万、hard max 300万unique root
- valid: 凍結90,759 root
- objective: Phase Aで一意に決定したもの。hybrid時は同じWDL pool/SHA、`kappa`、係数、`1:1` samplerを固定
- LR上限: `1e-4`
- outer: 最大2
- search: 現行depth 1 + exact qsearch leaf
- checkpoint: 固定50,000 rootごとの決定論的milestone
- archive/Gate: outer完了単位

570,875 rootの1 outer実績は約30時間である。線形換算では200万rootで約4.4日、260万rootで約5.7日、
hard max 300万rootで約6.6日となる。
実際はphase構成でqnodes/rootが変わるため、250,000-root pilotの実測roots/sとqnodes/rootからETAを再計算する。
現時点ではauditと対局を含め、1 outerを5～7日、最大2 outerを10～15日と見積もる。

現行`halfkp_mmto_unattended.py`は旧dataset、generation 2 zero-AdaGrad、4 outerのpolicyを固定しているため、そのまま
再利用しない。既存の監視・archive骨格だけを使い、Phase A～C専用config/modeとsystemd unitを追加する。milestoneは
frozen search snapshot、入力cursorと順序、WDL sampler状態、model、optimizer、固定flush境界、dataset/binary/Git
SHAを保存し、中断再開がbit-identicalであることを小規模testで確認する。Gate判定はouter完了後だけ行う。
checkpointはatomic replaceし、直前世代を1個だけ残す。完了済みouter以外の全milestoneを永久保存しない。
50,000 root境界はtraining blockの完了位置へ固定し、wall-clock時間でgradientのflush位置を変えない。

pilotの実測deltaをpair doseの平方根でfullへ換算し、feature予測deltaが0.015を超える場合だけ、式でLRを一度
下げる。棋力結果を見てLRを変更しない。

各outer後に次を行う。

1. model/optimizer/checkpoint SHA検証
2. 凍結90,759-root fresh auditと6,099-game paired bootstrap
3. 直前accepted親とのfixed 20,000 nodes・500局
4. score分布、qnodes/root、parameter delta、評価値分位点の監査
5. qnodes/rootが直前親比20%を超えた場合だけ、同条件5秒50局を追加して実時間退行を確認

候補をacceptedにする条件は、安全違反0、audit差95%上限 < 0、直前accepted親へのscore ≥ 52.5%、
opening-pair bootstrap 95%下限 ≥ 48%である。qnodes/rootが20%を超えても単独では棄却せず、追加5秒対局が
score < 40%かつopening-pair bootstrap 95%上限 < 50%の場合だけ棄却する。すべての採用条件を満たさない候補は
artifactを保存して
`stopped_by_gate`とし、再試行しない。outer 1が不採用ならouter 4、outer 2が不採用ならouter 1を凍結する。
outer 1では同じ相手との500局を二重実行せず、outer 2でも直前accepted親だけと比較する。最大2 outer後、
この逐次採用規則で一意に決まった候補を凍結し、その後にtaya36開発対局でv2.5.0と5秒50局を行う。

| v2.5.0への結果 | 判断 |
|---|---|
| 40%未満 | 棋譜MMTOを終了しPhase Dへ |
| 40～50% | 重要な進展だがrelease不可。Phase Dへ |
| 50% | 同等止まり。Phase Dへ |
| 50%超 | 未使用2026 reserveの25 opening・先後交換50局へ |
| reserveのopening-pair bootstrap 95%下限 > 50% | v2.5.0超えのrelease候補 |
| reserveが上記未達 | releaseせずPhase Dへ。通常の50局score区間は補助値に限定 |

## 7. Phase D: scratch-only deep-search bootstrap

dense pilot/fullがscratch親を改善しない、または最良候補がv2.5.0へ50%を超えない場合に開始する。

### 7.1 既存失敗の再解釈

以前のdepth 6 / 100k nodes教師生成は100 root中100 rootがnode capへ到達し、`requested depth`完遂を必須にした
ため有効recordが0だった。これはdeep-search targetが無効だった証拠ではなく、保存条件が現実のCPU予算と
合わなかった結果である。

### 7.2 最小実装

既存`halfkp_search_teacher --value-only`と`halfkp_search_train`を再利用する。

- node cap時もlast fully completed iterative-deepening resultを採用する
- completed depth ≥ 4、full-window exact score、合法PV、finite scoreを必須にする
- recordへ実completed depth、nodes、PV length、teacher model SHAを保存する
- node cap途中の未完depth scoreとalpha/beta boundはtargetにしない
- 初回は`search_mix=1.0`のpure deep-search targetとする
- 初回pilotはWDLを加えない。MMTOに対して校正した係数を別objectiveへ流用しない
- material係数、score scale、qnodesを固定監視する

10,000 train / 2,000 valid root、200,000 nodes/root、最大2 epochでpilotする。見積りは教師生成4～6時間、
学習1時間未満、親500局を含め7～9時間である。

Go条件は、教師生成成功率95%以上、completed depth p10 ≥ 4、held-out deep-target loss 5%以上改善、安全Gate
合格、親500局score ≥ 52.5%かつopening-pair bootstrap 95%下限 ≥ 48%とする。通過時だけ50,000、次に
100,000 rootへ拡張する。
10,000で対局転移しなければ、mix、LR、depth閾値を調整せず停止する。

search bootstrappingは、深いalpha-betaの値を静的評価へ戻す方法である。RootStrapよりsearch tree内の多数nodeを
更新するTreeStrapが強かったという先行結果もあるが、TT nodeの局面保存を新設するのは大きな変更なので、まず
既存root/PV経路で転移を確認する。

[Veness et al., Bootstrapping from Game Tree Search](https://papers.nips.cc/paper_files/paper/2009/hash/389bc7bb1e1c2a5e7e147703232a88f6-Abstract.html)

### 7.3 通過時だけ自己対局分布へ広げる

deep-search pilotが通った場合だけ自己対局を追加する。

1. 2018～2022 WDoor局面をopening ballotにし、accepted scratch同士で200局smokeを行う。
2. 合法手、千日手、投了、resume、game SHAを検証後、1,000局を20,000 nodes/手で生成する。
3. ply 16～120を4 ply間隔でsampleし、game単位splitとnetwork-input dedupを行う。
4. 20,000～25,000 rootを200,000 nodesで再解析し、Phase Dと同じvalue targetで1世代学習する。
5. 親500局でscore ≥ 52.5%、opening-pair bootstrap 95%下限 ≥ 48%の場合だけ5,000局へ拡張する。
6. 最大3世代とし、各世代でaccepted親を更新してteacherを凍結する。

TDLeaf(λ)、全TT-node TreeStrap、league、MCTS、policy headは、root bootstrapが改善しながらplateauした場合だけ
別計画で検討する。最初から実装しない。

[Baxter, Tridgell & Weaver, TDLeaf(λ)](https://arxiv.org/abs/cs/9901001)

## 8. なぜこの順序か

- MMTOはscratch親に61.4～90.0%で勝っており、完全に失敗した方式ではない。
- 現datasetは先行MMTOと棋譜数が同程度なのに、1棋譜当たりrootを約1/6しか使っていない。
- terminal WDLは正しく試されていないが、単独で主戦略にする根拠もないため、一度の修正A/Bに限定する。
- dense unique rootは既存実装をほぼそのまま使え、CPU-onlyの長期休暇条件を直接活用できる。
- dense MMTOでも外部目標へ転移しなければ、相対棋譜手順位ではなく深い探索の絶対値を学ぶ必要がある。
- search bootstrapが成立してから自己対局を加えることで、大規模な生成基盤を先に作らずに済む。

NNUEで探索評価とgame resultをWDL空間で混合する構造自体は一般的だが、両者を別々に正規化し、尺度を固定する
必要がある。[Stockfish NNUE training documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)

## 9. 実行順序と概算時間

| 段階 | CPU実行時間の目安 | 自動分岐 |
|---|---:|---|
| Phase A corrected WDL | 12～16時間 | 全Gate通過ならhybrid、失敗ならMMTO-only |
| Phase B build + 250k pilot | 16～22時間 + build | 全Gate通過時だけPhase C |
| Phase C dense outer 1～2 | 10～15日 | 各outerを逐次採用、未達ならPhase D |
| Phase D 10k search pilot | 7～9時間 | 通過時だけ50k、100k、自己対局へ拡張 |

実装とdataset build時間を除けば、Phase A～Cは約12～17日である。supervisorはGateを満たした場合だけ次段を開始し、
中間帯を推測で進めない。`stopped_by_gate`ではartifactと理由を保存して正常終了し、同条件を自動再試行しない。

## 10. 実装とGit順序

1. `docs: record post-WDL A/B evidence and next training plan`
2. `training: normalize MMTO and WDL objectives independently`
3. `training: add game-cluster WDL calibration and audit`
4. `training: build dense high-rate WDoor train set`
5. `training: add resumable dense MMTO milestones and unattended workflow`
6. dense結果不十分時のみ`training: accept last completed deep-search target`
7. deep-search通過時のみ`training: add resumable scratch self-play extraction`

各commitでtargeted test、library test、release build、`git diff --check`を行う。長期run開始後はrevision lockを守り、
tracked fileを変更しない。run artifactは`runs/`へ保存し、完了後に小さいmanifest/reportだけをcommitする。

## 11. 進捗表

| 作業 | 状態 | 完了条件 |
|---|---|---|
| outer 4内部promotion | 完了 | generation 3 registry登録 |
| WDL 0.1 A/B | 完了・試験不成立 | 実効寄与約0.001、計画逸脱を記録 |
| treatment vs v2.5.0 | 完了 | 16-34、32.0%、release不可 |
| 旧WDoor全件rate再監査 | 完了 | 2018～2022両者4000以上19,209局 |
| 次期計画 | 完了 | 本書 |
| 実装・実験手順書 | 完了 | file/CLI/command/Gate/recoveryを固定 |
| WDL別正規化・監査実装 | 実装中 | I1/I2完了、10k byte identity・norm校正・WDL fit未完了 |
| corrected WDL treatment | 条件付き | 100k、Brier CI、500局、5秒50局 |
| dense dataset build | 実装中 | I4 builder/prepare完了、本番生成とGate未実施 |
| unseen-root 250k pilot | 未着手 | audit改善、親500局Gate |
| dense milestone / supervisor | 部分完了 | I3 checkpoint/resume基盤完了、専用supervisor未完了 |
| dense outer 1 / 2 | 条件付き | 各outer archive・audit・500局 |
| scratch候補凍結 | 未着手 | v2.5.0非参照で一意に選択 |
| v2.5.0開発対局 | 未着手 | taya36 5秒50局 |
| deep-search pilot | 条件付き | denseで未達時、10k root |
| scratch self-play | 条件付き | deep-search pilot通過時のみ |
| 2026 reserve最終対局 | 未着手 | 5秒50局、結果をrelease判定へ記録 |

## 12. 行わないこと

- 現行570,875 rootだけを再利用するsame-data-only outer 5以降
- 現在のWDL treatmentをgeneration 4へ昇格
- WDL係数の手動100倍、係数grid、seed再試行
- 2018～2019をMMTO主教師へ投入
- rate 4,000未満への自動緩和
- v2.5.0/KPPをteacher、target、init、self-play相手に利用
- hidden拡大、追加layer、policy head、AlphaZero/MCTS
- deep-search pilot通過前の大規模self-play基盤

本計画はv2.5.0超えを保証しない。しかし、現在判明したWDL試験の無効性を正し、先行MMTOとの最大の差である
desired-root量を埋め、それでも不足した場合に探索から新しい情報を得る。この順序が、CPU-only環境で既存実装を
最大限再利用しながら飛躍を狙う最短経路である。
