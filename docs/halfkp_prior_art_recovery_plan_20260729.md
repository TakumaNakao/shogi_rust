# HalfKP棋譜学習の失敗監査と先行事例準拠の再計画

- 作成日: 2026-07-29
- 状態: **Step 4条件付き完了・Step 5事前整備待ち**
- 対象lineage: `halfkp64-scratch-v1`
- 採用経路: **高rate WDoor棋譜を用いたBonanza-MMTO目的準拠のHalfKP適応**
- 初期model: accepted generation 2
  `9a1bfc9b7b317f451fa9a6efb350fed0a50e3020659686cefccc98646b9b3db6`
- 現行KPP/HalfKP重みの用途: **最終比較対局だけ**
- 計算機: AMD Ryzen 7 PRO 4750U、8 core / 16 thread、RAM 14 GiB、GPUなし

## 1. 結論

何度も失敗した最大の理由は、「棋譜学習が難しいから」でも「HalfKPに表現力がないから」でもない。
成功した棋譜学習の縮小版を実行せず、成功例の中心を省いた別のproxyを短期間で次々に試したためである。

特に、これまでHalfKPでMMTOと呼んだ処理は、Bonanza Methodの中心である次の処理を実装していなかった。

1. 棋譜手を含む全合法手を、**現在の学習中model**で探索する。
2. 各手から静止探索まで読み進め、探索が実際に評価したPV末端局面を得る。
3. 棋譜手のPV末端を全代替手のPV末端より高くする勾配を、両方の末端へ与える。
4. 重みを少し更新した後、変化した現在重みでPV末端を再生成する。

現行HalfKPの`halfkp_search_teacher`は探索前に指した直後の子局面をpackし、探索値だけをその子局面へ
関連付けていた。`halfkp_policy_train`も全合法手を列挙しているが、比較対象は指した直後の静的評価である。
これは正しく微分できていても、探索が静的評価関数を呼ぶ局面を学んでいない。

さらに、現行探索のquiescence searchは返り値のPVを常に空にしている。旧KPP用`mmto_tree_dump`には
`teacher_leaf_sfen` / `student_leaf_sfen`の器があるが、depth 1ではqsearch内の捕獲・王手列を辿れず、
本当のqsearch末端にはならない。この意味論を先に直さず、既存tree dumpを名前だけ移植してはならない。

従って次は、独自の順位loss、hard negative、search score回帰、型課題を追加しない。
WDoor棋譜手と全合法代替手の**current-model qsearch PV末端**を比較する一本の基準実装を作り、
小規模段階を実装検証だけに使った後、先行事例に見合う反復量まで止めずに学習する。

ただし、これは線形・整数weightとgrid-adjacent updateを用いるBonanza原法そのものではない。
非線形HalfKP、float weight、mini-batch AdaGradへ移すため、本書では以後
**MMTO目的準拠HalfKP適応**と呼ぶ。原法と同じ結果や計算量を仮定せず、相違点を事前に固定して測定する。

## 2. 今回の判断で変えること

### 2.1 撤回する結論

「高rate WDoorのall-legal CEが無効だった」という結論を撤回する。

Phase 14の計画はrate 4,000以上のWDoorを指定していたが、実行manifestが実際に読んだのは
`wdoor-phase3-250k-v1/{train,valid}.jsonl`だった。このdatasetにはrate下限がない。
pilot先頭50,000件の監査では、両対局者がrate 4,000以上の局面は約13.6%だけだった。

しかもepoch 2時点で次の全指標がまだ改善中だった。

| 指標 | parent | epoch 2 | 変化 |
|---|---:|---:|---:|
| valid CE | 4.256464 | 4.127225 | -3.04% |
| top-1 | 7.33% | 7.84% | +0.51 point |
| 棋譜手margin | -703.59 cp | -637.85 cp | +65.74 cp |
| margin改善 / 悪化 | 0 / 0 | 8,395 / 1,603 | 改善多数 |

これを、根拠を実証していない「2 epochでtop-1を1 point改善」という一点Gateだけで打ち切った。
従ってこれは方法の反証ではなく、**条件違反かつ未収束の実験**である。

ただし、正しいdatasetで同じone-ply CEをやり直すことは次の本線にしない。
Hokiらの論文が、one-plyの直感的比較を出発点として示した上で、探索PV末端を使うMMTOの方が安定性と
棋力で優れることを実験しているためである。既存CE実装はgradientとdataset readerの診断材料として残す。

### 2.2 「失敗」の再分類

| 系列 | 観測 | 現在の分類 |
|---|---|---|
| accepted generation 2 | scratch parentに200局96%、v2.5.0には約30% | 有効だが狭いcurriculumの初期値 |
| high-rate direct mix | offline改善、v2.5.0に20～30% | 有効な負結果。目的と棋力が乖離 |
| long-direct 32～128 | 同じ固有局面の反復、後半はexact/static/棋力悪化 | 同じproxyとglobal decayの負結果 |
| 旧rank/MMTO/Phase A | output 104/129個を無勾配でもclamp | 無効または交絡 |
| corrected search-value | Brier改善、search MAE悪化 | 有効な目的不一致 |
| game-rank correct dose | proxy loss改善、親400局45.25% | 有効なproxy不一致 |
| Phase 14 all-legal CE | 誤dataset、改善中2 epochで停止、対局未実施 | 未検証 |
| MMTO目的準拠HalfKP適応 | Step 4で正しいdataset・PV leaf・3 LR校正を完走 | 実装成立、効果は条件付き判定中 |

「独立な学習法が何度も正しく試され、すべて失敗した」と解釈してはならない。実装不具合で無効な実験、
意味の異なるproxyの負結果、未収束pilotが混在している。

## 3. 先行事例との比較

### 3.1 Bonanza / MMTO

Hoki and KanekoのMMTOは、47,566棋譜から4,467,116の棋譜手と368,313,024の
棋譜手対代替手pairを作った。各外側反復では、全training局面の全合法手を現在重みで浅く探索し、
PV末端を選び直してから勾配更新した。原法は1回のPV生成につき、同じPV集合を用いた微分・更新を
32回繰り返した。主な探索はnominal depth 1とquiescence searchであり、位置評価は200回のPV生成、
当時のIntel X5690 workstationで約1週間を要した。

重要なのは規模だけでなく、次の組合せである。

- 棋譜手対**全合法代替手**
- raw child評価でなく**current-model minimax / qsearch値**
- 勾配は**PV末端局面**
- 外側反復ごとの**PV再生成**
- 小さな更新
- 評価尺度を固定する等式制約
- 大規模特徴の過学習を抑えるL1
- 別棋譜holdoutの探索一致率
- checkpointごとの多数局対局

論文のfull modelは約4,000万parameterである。本repoのHalfKP-64はdirect featureだけで約675万parameterなので、
parameter比は約0.17である。既存高rate datasetのtrain 570,875 rootはBonanzaの約0.13、
1回のPV生成あたり56,035,622合法手pairは3.68億pairの約0.15である。

ただし、この二つの比から「parameter当たりの学習量が同等」と結論してはならない。feature分布、
sparse row coverage、ClippedReLU、optimizer、原法の32更新が異なるためである。これらの数値は
workloadの桁を決める参考値に限定し、十分性はcoverage、fresh loss、PV変化、対局で実測する。
これまでの50,000 root × 2 epochは約900万child提示にすぎず、実装sanityの規模だったことだけは明確である。

一次資料:

- [Large-Scale Optimization for Evaluation Functions with Minimax Search](https://www.jair.org/index.php/jair/article/download/10871/25935)
- [局面評価の学習を目指した探索結果の最適制御](https://ipsj.ixsq.nii.ac.jp/records/97627)
- [Bonanza式学習法](https://www.jstage.jst.go.jp/article/jsoft/26/5/26_216_1/_article/-char/ja/)

### 3.2 Apery

Aperyは53,810局のプロ棋譜をBonanza Methodで学習し、棋譜手と他手から浅く読み進めた**探索末端**を
正例・誤例にした。駒価を固定し、頻出する2駒関係から3駒関係へ進み、一致率が停滞してから探索を深く、
更新を小さくする3段階を約2か月実行した。

これは「型を先に覚える」というより、低次元・高頻度のparameter群から始め、同じ探索整合目的を
段階的に大規模化した例である。本repoでは新しい戦型lossを増やす代わりに、scratch generation 2の
material/exact基礎を固定scaleの初期値として使い、全HalfKP direct rowをPV末端目的で更新する。

一次資料:

- [Aperyの技術，世界コンピュータ将棋選手権におけるパフォーマンス解説](https://www.jstage.jst.go.jp/article/jsoft/26/5/26_200/_pdf)

### 3.3 現代NNUE

Stockfishの標準NNUEは、棋譜手のone-ply policyを主に学ぶのではない。大量局面に付けた探索scoreを
WDL空間へ写し、最終勝敗と混合してscalar valueを回帰する。公式手順の代表値はbatch 16,384、
1 epoch 1億sample、100 epochで競争力が出始め、約400 epochで成熟する。公式dataset例はdepth 9、
5,000 nodes/position、さらに数十億～160億局面の規模である。

従って「HalfKPだから、数万局面の棋譜手CEでもNNUE成功例と同じ」とは言えない。標準NNUE経路を採るには、
十分強い探索score教師を数千万～億局面へ付ける必要がある。

一次資料:

- [NNUE technical documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)
- [Basic training procedure](https://github.com/official-stockfish/nnue-pytorch/wiki/Basic-training-procedure-%28train.py%29)
- [Training datasets](https://github.com/official-stockfish/nnue-pytorch/wiki/Training-datasets)

## 4. 失敗原因の優先順位

### P0: 直接原因

#### P0-1. 学習対象が探索PV末端でなかった

`halfkp_search_teacher`はroot moveを指した直後に`PackedHalfKpPosition`を作り、その後のdepth探索scoreを
その即時childへ付けていた。trainerもその即時childへbackpropした。

この方法は、交換を読み切った先のscoreを交換前の局面へ直接押し込み、評価関数が探索中に実際に呼ばれる
局面分布と勾配局面を一致させない。finite differenceが通っても、異なる目的を正確に微分しているだけである。

#### P0-2. 現在重みによるPV再生成がなかった

既存HKSTはaccepted generation 2で一度生成して凍結した。学習で重みが変わっても、どの枝がPVになるか、
どの代替手が現在最も危険かを更新しなかった。

MMTOの中心は「探索→小更新→再探索」の反復である。凍結teacherを何epoch回しても、
古くなった局所近似を反復するだけになる。

#### P0-3. 成功例に相当する学習量へ到達する前に方法を変えた

2 epochや数分のpilotは、符号、loss、serializationを確認するsanityとしては有用だが、
50時間以上学習したv2.5.0を超える棋力実験ではない。

Bonanzaは200回のPV refresh、Aperyは約2か月、現代NNUEは数十億以上のsampleを使う。
これまでの運用は、小さいpilotを方法の永久棄却に使い、別lossを実装することを繰り返した。

### P1: 結果を悪化・混同した原因

#### P1-1. 実験条件とmanifestの不一致

Phase 14は高rate datasetを使わなかった。これだけで高rate CEに関する結論は無効になる。
今後は計画のdataset path、実manifestのpath、SHA-256、rate条件を実行開始前に機械照合する。

#### P1-2. 実装不具合と過剰制約

過去の監査で次を確認している。

- `halfkp_search_train`がzero gradientでも104/129のoutput weightを`±0.06`へ投影した。
- factor-only symmetryが正しい盤面対称性より強い制約を課した。
- long-directのglobal decayが未観測rowまで縮小し、128 epoch後のfeature RMSを56.8%落とした。
- train objectiveとcheckpoint選択用validationのweightingが一致しなかった。
- rootだけでなくcandidate childまで含めるとtrain/valid衝突があった。
- qsearchがPVを返さないため、既存のleaf fieldだけでは真のqsearch leafを表せない。

現在のimmediate-child gradientに新たな致命的符号反転は見つかっていない。しかし、符号が正しいことは
教師意味論が正しいことを保証しない。

#### P1-3. accepted generation 2が狭いspecialistだった

generation 2はfactor-only、exact/expert/result 70/15/15、2 epochで作られた。
scratch parentを大幅に上回った一方、v2.5.0には約30%である。以降、この大差を埋める候補へ
exact約90%、固定課題search約98%をほぼ維持することを要求し、広い棋力獲得より安全課題を強く拘束した。

次のMMTOではgeneration 2をscratch-onlyの初期値とmaterial scale anchorに使うが、exact、型、resultを
主lossへ混ぜない。これらは安全診断に限定する。

### P2: 判定運用の問題

#### P2-1. 小標本対局と一点AND Gate

20～40局では1勝が2.5～5 pointであり、小差を判定できない。それにもかかわらず、1勝差やpoint 50%を
永久停止条件に使った。親に22勝18敗の候補や、親400局ちょうど50%の候補も早期に終了した。

今後、小規模対局はクラッシュ・大崩れscreenにだけ使う。学習方法の有効性は、事前に確保した反復量、
fresh-search holdout、十分な対局数で判定する。

#### P2-2. proxyの改善を棋力の代理にしすぎた

long-directでは棋譜順位精度が上がってもv2.5.0対局が悪化した。exact/static/search課題も多くの候補が
同じ範囲に入り、候補識別力が低かった。top-1、Brier、marginは原因診断であり、最終目的ではない。

#### P2-3. 方針変更の頻度が高すぎた

短期間に多数のloss、trainer、Gateを追加した結果、各方法を成熟させる前に次へ移った。
これは「過剰なhyperparameter探索を避けた」のではなく、方法そのものを過剰に増やした状態である。

## 5. 代替案の比較

| 案 | 先行成功例への忠実度 | 現行重み不使用 | CPU適合 | 実装量 | 判断 |
|---|---:|---:|---:|---:|---|
| 正しい高rateでone-ply CEを延長 | 低～中 | 可 | 高 | 最小 | 診断のみ |
| MMTO目的準拠HalfKP適応 | 高 | 可 | 中～高 | 中 | **採用** |
| 外部の強いengineでvalue教師生成 | NNUEとして高 | 可。ただし外部重みを教師利用 | 低～中 | 中 | 明示承認なしでは不採用 |
| scratch self-play + deep score/result | 高 | 可 | 低 | 大 | MMTO成熟後のfallback |
| network拡大・policy head・新feature | 未確定 | 可 | 低 | 大 | 保留 |

純粋に最短の成功確率だけを考えるなら、現行とは独立した強い外部engineで数千万局面へ探索scoreを付け、
標準NNUE value回帰を行う案が有力である。しかしこれは学習済み外部評価器を教師として使う。
現在の「WDoor棋譜とscratch lineageだけ」という制約を広げるため、本計画では採用しない。

## 6. 採用する学習仕様

### 6.1 絶対条件

1. v2.5.0、KPP由来HalfKP、その他現行weightをinit、score、PV、best move、negative選択へ使わない。
2. v2.5.0は最終候補を凍結した後の比較対局だけに使う。
3. 初期値はaccepted generation 2に固定する。これは同じscratch lineageだけから作られている。
4. teacher moveはWDoor棋譜に記録された指し手だけである。
5. teacher searchとstudent searchを別modelに分けない。各outer iterationの**現在model一つ**で全手を探索する。
6. exact、詰み、戦型、result、旧HKST scoreを主lossへ混ぜない。

### 6.2 dataset

本線は既に監査済みの次を使う。

```text
runs/halfkp64-scratch-v1/wdoor-highrate4000-1m-v2/train.jsonl
runs/halfkp64-scratch-v1/wdoor-highrate4000-1m-v2/valid.jsonl
runs/halfkp64-scratch-v1/wdoor-highrate4000-1m-v2/test.jsonl
```

manifest上の条件と件数:

- 2023～2025 WDoor
- 両対局者rate 4,000以上
- 44,228 game、723,867 unique network input
- train 570,875 root / 34,582 game
- valid 90,759 root / 6,099 game
- test 62,233 root / 3,547 game
- game / opening family / network inputのsplit衝突0
- train内重複0
- dataset stage fingerprint:
  `bb1363e7fe376653a4c99437db2645cc930cb2518a7282ea7638819b0925f0b6`
- manifest SHA-256:
  `d8da8088b02d3d0200aad2ecb061691c6c418173a16fe32819f03088af48708f`
- train SHA-256:
  `c35673d69c9306f1dd4b1f2f32714adce4a43e7c1a086141bd6b7246b680eee6`
- valid SHA-256:
  `3ce8745dbeb094a6a3f357494eb94aa5f57a716437bda66dd0ca13060f075f96`
- test SHA-256:
  `85c73cdf519deff2a51bc2a92e0a746047cb1deae2187f3a1445dd2411b49893`
- testの件数とprovenance監査は済んでいるが、modelによるscore・選択には最終候補凍結まで使わない

Phase 14が読んだ`wdoor-phase3-250k-v1`へfallbackしてはならない。起動時にdataset manifestの
`min_player_rate=4000`、`min_opponent_rate=4000`、input/output SHAを検証し、不一致なら開始しない。

rootは合法手2以上だけを使う。棋譜手が合法でないrecord、王がない局面、非有限評価、探索不能はerrorにする。
局面を黙って捨てず、terminal / bound / failureの理由と率を全件記録する。

### 6.3 true qsearch leaf

training専用の探索結果を次で定義する。

```text
LeafResult {
    root_oriented_score_cp,
    pv,
    packed_leaf_position,
    terminal_kind,
    bound_kind,
    nodes,
    qnodes
}
```

各root move `m`について、その手を強制してからnominal depth 1相当、すなわちchildでqsearchを行う。
capture、check、evasionを辿って評価関数が実際に呼ばれた局面を`packed_leaf_position`とする。
baselineではwall-clock limitと任意node capを置かず、既存の決定論的最大qplyだけを使う。

通常対局searchの挙動を変えないため、training専用のleaf traceを追加する。
既存qsearchが返す空PVを使わず、既存qsearchと同じstand-pat、SEE skip、check/evasion、最大qplyの
分岐を共有し、選択されたprincipal leafを追跡する。現行qsearchはTTを参照しないため、
training専用のTT処理は追加しない。

rootからのnon-terminal full-window searchでexact resultが得られた場合は、次を必須testにする。

```text
returned root-oriented score
  == root視点へ向きを合わせた packed leaf のruntime評価
```

許容差はfloat加算順に必要な最小値だけとし、初期値は`1e-3 cp`とする。
内部のfail-hard bound、mate、千日手、最大qply、resource limitはraw leaf evalと混同せず、
`terminal_kind` / `bound_kind`で分類する。学習pairにはrootでexactな結果だけを使う。

### 6.4 目的関数

root `p`、棋譜手`d`、それ以外の全合法手`m`について、

```text
delta = s(p.d, w) - s(p.m, w)
T(delta) = 1 / (1 + exp(delta / tau))
Z = sum over p of (legal_moves(p) - 1)
J(w) = (1 / Z) * sum over all (p, m) of T(delta) + constraint
```

`s`は現在modelで選んだqsearch PV末端のroot視点評価である。各pairを等重みとし、合法手の多いrootも
Hokiらと同じくpair数だけ寄与する。mini-batchもroot数ではなくpair数で正規化する。棋譜手のleafへ正方向、
代替手のleafへ負方向の勾配を与える。棋譜手と現在bestが一致していても、全代替手とのmarginを学ぶ。
「scratch探索が棋譜手に同意したpairだけ」のような選別をしない。

Hokiらのscore単位で用いた`a=0.0273`に対応する`tau`を、runtimeのpawn=100 cpと大きく矛盾しない
移植初期値として`1 / 0.0273 = 36.63 cp`へ固定する。
10k段階でscore単位がruntime cpと一致することだけを監査し、対局結果を見てtemperatureを調整しない。

terminal mateは固定探索値としてlossへ入れてよいが、存在しないraw leaf勾配は作らない。
容易な強制詰み・合法手1手だけのrootはagreement指標から除く。

### 6.5 制約とoptimizer

先行例の尺度制約に対応して、accepted generation 2のmaterial output coefficientを固定する。
これは現行KPP値ではなくscratch lineage内で得た値であり、評価scaleの発散・ゼロ縮退を防ぐanchorである。

optimizerは既にfinite differenceとno-opを検証したpersistent AdaGradを再利用する。
optimizer stateをouter iteration間で保持し、全gradient 0のparameterを変更しない。
Bonanzaのgrid-adjacent updateをそのままfloat NNへ移す新optimizerは作らないが、1 outer iterationの
parameter最大変化を制限し、PVを急変させないという同じ条件を満たす。

BonanzaのL1係数は整数線形model用で、layer間のscale自由度を持つHalfKPへそのまま移せない。
batch依存のproximal L1や全row decayを新たに導入すると、過去と同じ未観測row収縮を再発させる。
従ってbaselineでは`l1_lambda=0`、`l2_lambda=0`、decoupled decayなしに固定する。この相違は隠さず、
feature coverageとtrain/full-valid乖離を記録する。成熟runで明確な過学習が出た場合だけ、
regularizationを別計画・別lineageとして検討する。

固定optimizer仕様:

```text
optimizer                 = persistent AdaGrad
search_block_roots        = 1,024
pair_batch_size           = 8,192
gradient_normalization    = equal weight per pair
gradient_clip_global_norm = 1.0
output_lr                 = feature_lr * 0.1
feature / hidden limit    = [-1.0, 1.0]
output / output-bias limit= [-0.20, 0.20]
material output weight    = frozen
leaf_reuse_passes         = 1
```

1 outerで許すsnapshotからの最大絶対変化はfeature/hidden bias `0.02`、output/output bias `0.004`とする。
固定1,024 valid probeのroot best-move一致率を90%以上に保つ。上限に触れたmodelをclipして続行せず、
そのlearning rateを不適格とする。material scalar固定だけでNN全体のscaleが数学的に一意になるとは
みなさず、score RMS、activation飽和率、parameter bound接触率も監視する。

各outerの開始時modelをimmutableな`search snapshot`とし、全rootのleaf選択はこのsnapshotだけで行う。
更新側modelは、生成したleaf blockを一度だけstreamしながらmini-batchごとに**更新後の現在weightで
leafをforwardし直し**、lossと勾配を計算する。snapshot scoreを固定labelとして回帰しない。
search snapshotと更新側を分けることで、後半blockのPVだけが更新済みmodelで選ばれる混在を防ぐ。

Bonanza原法は1 PV生成につき32回のfull-gradient更新を行う。一方、この適応の1 streaming passは
56,035,622 pair / 8,192、すなわち約6,841回のmini-batch更新を含む。両者の「更新1回」は同じ単位ではなく、
32 passへ機械的に増やすとblock順序を過大に反復する。従ってbaselineは`leaf_reuse_passes=1`に固定し、
原法と量的に同等とは主張しない。PV安定性とfresh objectiveでこの適応自体を検証する。

学習率の試行は10k rootで次の3点だけに限定する。

```text
3e-6, 1e-5, 3e-5  (feature / hidden)
output learning rate = 0.1 × feature learning rate
```

各2 outer iterationで、finite、scale固定、fresh-search loss低下、parameter/PV変化の安定性を比較し、
安定な最大値を一度選んで固定する。棋力やtestを見たgrid searchは行わない。

### 6.6 outer iteration

1 outer iterationは次のatomic単位とする。

1. 現checkpointとpersistent optimizer stateを読込む。
2. 現checkpointをimmutableなsearch snapshotとして固定する。
3. trainをblock単位で読み、全rootの全合法手をsearch snapshotで探索する。
4. block内のcurrent qsearch leaf全pairを一度streamし、各batchで更新側modelをforwardし直して
   mini-batch AdaGradを掛ける。
5. material coefficient固定とparameter/PV変化の上限を検査する。
6. modelとoptimizerをatomic checkpointへ保存する。
7. 毎outerは固定10,000 validを**更新後modelで再探索**し、fresh-search指標を出す。
   full 90,759 validは指定checkpointだけで再探索する。
8. model、dataset、binary、設定、統計、前checkpoint SHAをmanifestへ保存する。

leaf datasetを何epochも固定再利用しない。毎outer iterationで必ず再探索する。
大容量leaf JSONLを永続化する必要はなく、gradientをstream集計してよい。ただし再現監査用に固定した
少数rootのroot/move/PV/leaf/score traceを各checkpointへ保存する。

## 7. 実装計画

### Step 0: 記録の訂正

- Phase 14を`rejected`から`invalid_source_and_inconclusive`へ訂正する。
- 旧MMTO-liteを`true_mmto=false`と明記する。
- 既存artifactは削除せず、昇格・教師利用を禁止する。

完了条件:

- 文書、機械可読結果、manifest pathの三者が同じ分類を示す。

### Step 1: training専用qsearch leaf trace

最小変更:

- `src/ai/`へtraining専用leaf resultとforced-root-move searchを追加する。
- normal game searchのpublic behaviorとfingerprintを変えない。
- qsearchでstand-patまたは探索手のどちらが値を作ったかを追跡する。
- qsearch内PVを捨てず、最終raw評価局面をpackできるようにする。
- root full-windowでexact resultを作ったprincipal leafだけを受理する。

必須test:

1. captureが1手以上続くfixtureでleafがimmediate childと異なる。
2. non-terminal full-window scoreとleaf runtime評価が一致する。
3. 先手・後手、奇数・偶数PVのroot向きが一致する。
4. check evasion、stand-pat、capture、promotion、dropを含む。
5. mate / repetition / max-qplyを別terminalとして識別する。
6. search後にroot局面、history、incremental HalfKP contextが完全復元される。
7. normal search fingerprintが不変。

### Step 2: `halfkp_mmto_train`一本を実装

新しいdump、convert、trainerを複数作らず、一つのnative streaming executableにまとめる。
既存から再利用するもの:

- `PackedHalfKpPosition`
- `halfkp_search_train`のHalfKP forward/backpropとAdaGrad
- scratch lineage / promotion registry検証
- dataset manifest / SHA / environment記録
- center-king mirror gradient
- atomic checkpoint

追加するもの:

- 全合法手のforced qsearch leaf
- 棋譜手対全代替手のmirrored-sigmoid loss
- leaf両側へのgradient
- immutable search snapshotと更新側modelの分離
- 一回のstreaming mini-batch pass / outer
- material coefficient固定
- persistent outer checkpoint / resume
- fresh-search valid evaluator

追加しないもの:

- policy head
- value regression
- game result mix
- hard-negative mining
- search teacher model
- factor architecture
- SWA
- dashboard
- L1 / L2 / decoupled weight decay

Step 3開始前に、最終対局用opening 2,000組をseed `20260730`で生成し、過去の学習・screen・
parent対局openingとの重複を除いてSHA-256をmanifestへ固定する。以後、凍結候補対v2.5.0の
固定4,000局まで内容も結果も参照しない。途中のparent対局は別seed・別opening集合を使う。

### Step 3: 64 fixture / 1,024 root verification

64 fixtureで全leaf意味論、1,024 rootでend-to-endを確認する。

必須Gate:

- analytical gradientと固定PV finite differenceが一致する。
- pairの並べ方やrootごとの合法手数が違っても、全pair等重みの集約gradientが許容差内で一致する。
- 先後を反転した同型局面のloss/gradientが一致する。
- loss weight 0でmodelとoptimizerがbyte identity。
- save/reload後のmodel、optimizer、fresh validが一致する。
- dataset path/SHA/rate条件不一致を拒否する。
- NaN/Inf、違法手、root復元失敗0。

ここでは棋力を判定しない。

### Step 4: 10k root学習率校正

- train 10,000 / valid 2,000
- feature learning rate `3e-6 / 1e-5 / 3e-5`、outputは各0.1倍
- 各2 outer iteration
- 同じroot、決定論的な入力順、初期model
- batch 8,192 pair、clip norm 1.0、leaf reuse 1 pass
- `|delta| / tau`のp50 / p90 / p99と、mirrored-sigmoid derivativeが最大値の1%以上あるpair率を
  診断値として記録する（選択Gateにはしない）

選択順:

1. 全semantic/safety Gate通過
2. fresh-search valid lossが低下
3. material scale不変
4. parameter上限非接触
5. 固定probeのroot best-move一致率90%以上
6. feature/hidden最大delta 0.02、output最大delta 0.004未満
7. 条件を満たす最大learning rate

top-1を何point上げる、対局を何勝する、というGateは置かない。

### Step 5: 記録修正、選択LR再現、thread benchmark、100k検証

Step 4後監査で、outer 0とouter 1のfresh-valid履歴、計画済み飽和診断、outer別checkpoint、
区間wall timeが保存されていないことが分かった。長時間実行の前に、学習式を変えず次を追加する。

- outer 0と全outerのfixed-valid履歴
- milestoneごとのimmutable model / optimizer / manifest
- search / update / valid / saveの単調時計による経過時間
- `|delta|/tau`分位点、有効derivative率、regret分位点、gradient clip率
- per-outer feature / hidden / output最大delta
- 固定probeのqsearch PV length、PV sequence hash、leaf position hash一致率
- qnodes / root、binary SHA

per-root valid指標とparameter RMS / p99 / maxはfull production開始前に追加し、100k開始のblocking要件にはしない。

追加後、三つのlearning rateを再実行せず、`3e-5`だけを同じ10,000 train / 2,000 validで
4 outer再実行する。outer 2のmodel / optimizerが旧Step 4とbyte一致し、outer 4の同一valid
fresh lossがouter 0より低いことを確認する。

jobs benchmarkは固定1,024 train / 1,024 valid、`3e-5`、1 outerへ短縮してjobs 1 / 4 / 8を一度測る。
決定論と速度を確認した後、
accepted generation 2から独立にtrain 100,000 / valid 10,000、`3e-5`、4 outerを実行する。
まずouter 1を検証し、同じrunをresumeしてouter 4まで進める。

これはfull 1 outerの時間、node/qnode、legal moves、terminal、retry率と、複数outerの改善曲線を測る工程である。
agreementや小規模対局の改善が小さいことだけを理由に本線を止めない。fixed-valid fresh lossが親より逆向き、
leaf不一致、非有限、provenance不一致など、実装仮説そのものが壊れた場合は修正して同じStepを再実行する。
outer 4後の親500局は診断値とする。ただしopening pair単位paired scoreの95%上限が45%未満という
明白な棋力崩壊だけはfull開始の保留・監査Gateとし、別戦略へ自動移行しない。
対局前に、ply 8～40・opening family一意の250 pair suiteをcurrent strategy記載の規則で凍結し、
対局reportへopening-pair clustered percentile bootstrapを実装する。

### Step 6: full MMTO目的準拠HalfKP適応をmilestone運用

- train 570,875 root
- valid probe 10,000 root / checkpoint full valid 90,759 root
- 全合法手
- nominal depth 1 + qsearch
- 200 outer iterationを最終目標とする
- 初期checkpoint: 1, 2, 4, 8, 16
- iteration 16がouter 0より低く、`loss(4) < loss(8) < loss(16)`の連続悪化がなければ
  32, 64, 100, 144, 200へ継続する
- 小規模対局だけを理由に学習停止しない

full開始前に、modelを更新せず指定checkpointをfull validで評価して別artifactへ保存するeval-only経路を追加する。
per-root / parameter統計を含む観測機能追加後は、no-op byte identityと短い既知runのbyte回帰を再確認する。
accepted generation 2のouter 0 full-valid評価をproduction更新前に同じ90,759 rootで凍結する。

既存高rate trainは実datasetの`legal_moves - 1`合計で56,035,622 pair / outerである。
実時間はStep 5の実測から算出し、wall-clock、disk、温度、再開時間を開始前manifestへ書く。
200 outerで11,207,124,400 pair、
Step 4のqnode比では約20 trillion qnodeになる。実時間は1 outer約40～50時間、
200 outer約330～420日と外挿された。従って約1年の単一jobとして
起動せず、milestoneごとにatomic resume、CPU温度、filesystem容量、fresh loss曲線を確認する。
長い見積りだけを理由に別の短期proxyへ戻らない。

実行はtarget outer 1 / 2 / 4 / 8 / 16ごとに別invocationでresumeする。outer途中のresumeは実装せず、
障害時に最大1 outerを再実行するRPOを受容する。CPU温度、空き容量、process heartbeatは外部監視する。

各checkpointで記録する:

- train search-snapshot / valid fresh MMTO loss
- searched-move agreement
- 棋譜手search regretのmean / median / p90 / p95
- PV length、qnodes、PV refresh率
- phase別、合法手数bin別、手番別
- terminal / retry / failure率
- feature row coverage
- parameter RMS / p99 / max、bound接触率
- activation / runtime health
- float save/reload / runtime forward一致

### Step 7: 対局評価

checkpoint対局はiteration 8から行う。通常の小差は途中停止条件にしない。

1. iteration 8 / 16 / 32 / 64 / 100 / 144 / 200をaccepted generation 2と
   20,000 nodes / move、max depth 64、max 256 plies、先後交換500局で比較する。
   paired scoreの95%上限が45%未満の場合だけ、明白な棋力崩壊として次の長期区間を保留して監査する。
2. safety Gateを通過した指定checkpointのうち、**full valid fresh MMTO lossが最小**の一つを凍結する。
   同値なら早いiterationを選ぶ。親対局は診断値であり選択には使わない。
3. test 62,233 rootは、この凍結後にmodelで一度だけ評価する。test結果でcheckpointを選び直さない。
4. 凍結候補とv2.5.0をStep 3前に予約したopening 2,000組、先後交換、20,000 nodes / move、
   max depth 64、max 256 pliesの
   **固定4,000局**で比較する。
5. 1 opening pairをclusterとして100,000回のpaired bootstrap
   （seed 20260729、win=1 / draw=0.5 / loss=0）を一度だけ行う。
6. scoreの95% percentile CI下限が50%を超えた場合だけ「v2.5.0より強い」と判定する。
   跨いだ場合はinconclusive、下回った場合は未達とし、同じ対局列へのoptional stoppingをしない。

v2.5.0の対局結果を見てlearning rate、loss、datasetを調整しない。調整する場合は新しい候補系列とし、
別の最終openingを予約する。

## 8. 停止条件と次の分岐

### 8.1 実装修正して同じ計画を続ける条件

- qsearch scoreとleaf評価が一致しない。
- gradient finite differenceが通らない。
- fresh valid lossがtrainと逆向きに発散する。
- material scale、NaN/Inf、serializationが壊れる。
- dataset provenanceが一致しない。

これは手法の失敗ではなく実装不成立なので、別戦略へ移らず修正する。

### 8.2 MMTOを完了とみなす条件

200 outer iterationを最終目標とするが、Step 5完了前に約1年分を一括起動しない。まずfull iteration
1 / 2 / 4 / 8 / 16を完了し、fixed-valid曲線、実測時間、運用安全性を記録する。事前定義した
outer 16 Gateを満たせば後半milestoneへ継続する。
例外は、数値的に安全な学習率でもobjectiveが親から逆向きに動く、leaf同一性が壊れる、
provenanceが不一致になる等、実装仮説そのものが成立しない場合である。

20～40局の負け、2 iterationのtop-1不足、任意margin不足では終了しない。

### 8.3 結果別の次戦略

| 成熟後の結果 | 解釈 | 次 |
|---|---|---|
| fresh lossもagreementも改善しない | 実装・optimizer・scale問題 | 同じMMTOを修正 |
| agreement改善、親対局非改善 | WDoor模倣と棋力の乖離 | data品質を再監査 |
| 親に改善、v2.5.0へ未達、曲線改善中 | 単純に量不足の可能性 | 次系列でdataset拡大 |
| 親に改善、v2.5.0へ未達、full run収束 | expert imitationの限界 | scratch self-play/elmoへ |
| v2.5.0の95%下限超過 | 目標達成 | promotion / release評価 |

fallbackのscratch self-playでは、promoted scratch model自身が定跡なし対局を生成し、深い探索値と最終結果を
混合してqsearch PV末端へ学習する。現行KPP/v2.5.0を教師にしない。ただしこの実装はMMTO成熟結果が出るまで
始めない。

## 9. 進捗表

| Step | 内容 | 状態 | 完了証拠 |
|---:|---|---|---|
| 0 | 誤分類・誤dataset記録の訂正 | completed | docs / JSON |
| 1 | exact qsearch principal-leaf trace | completed | `bacd856` / unit / fingerprint |
| 2 | streaming `halfkp_mmto_train` | completed | `4275e2e` / tests / release build |
| 3 | 64 fixture / 1,024 root | completed | `a916fb3` / execution manifest |
| 4 | 10k、3 LR、各2 outer | conditional | 実行完了、`3e-5`暫定選択。履歴診断をStep 5で補完 |
| 5 | 記録修正、3e-5 10k×4再現、short jobs benchmark、100k×4 | pending | post-Step 4 strategy / benchmark report |
| 6 | full milestone 1 / 2 / 4 / 8 / 16、Gate通過時は後半継続 | pending | checkpoint manifests |
| 7 | parent / test / v2.5.0評価 | pending | match reports |

現在の実行判断と進捗の機械可読な正本は
`training/halfkp64-scratch-v1/mmto_post_step4_strategy_20260730.json`とする。
`prior_art_recovery_plan_20260729.json`は元計画と来歴を保持する。

### Step 1 実行記録（2026-07-29）

commit `bacd856`でtraining専用`forced_move_qsearch_leaf_trace`を実装した。通常探索は従来どおり
qsearch PVを外部へ返さず、depth 3 search fingerprintも基準と完全一致した。capture継続、
stand-pat、check evasion、promotion、drop、mate、先後score向き、root局面復元をunit testで確認した。
さらに実HalfKP-64 fixtureを使い、forced qsearch後の増分評価値とroot局面のfull refresh評価値が
bit一致することを確認した。詳細なコマンドと結果は
`training/halfkp64-scratch-v1/qsearch_leaf_trace_step1_report_20260729.json`に固定した。

### Step 2 実行記録（2026-07-29）

commit `4275e2e`で単一native executable `halfkp_mmto_train`を実装した。各outer開始時の
HalfKPをimmutableな`Arc<EngineEvaluator>`として固定し、1,024 root blockを並列forced-qsearchした後、
pair順を保持して更新側weightでleafを再forwardする。棋譜手対全代替手の`T(delta)`をpair数で正規化し、
両leafへbackpropする。persistent AdaGrad、material bit固定、center-king mirror、parameter/outer delta
上限、fresh valid再探索、hash検証付きatomic resumeを含む。補助loss、hard negative、weight decay、
SWA、中間leaf dumpは追加していない。

frozen WDoor rate-4000 manifestとaccepted scratch generation 2を用いた実データsmokeを完走し、
jobs 1 / 4のmodelおよびoptimizerがbyte一致した。詳細は
`training/halfkp64-scratch-v1/mmto_trainer_step2_report_20260729.json`に固定した。

### Step 3 実行記録（2026-07-29）

最終対局用openingは、学習に使わない2026年WDoor棋譜からseed `20260730`で2,000局面を予約し、
SHA-256 `385b7f70...262d2329`としてcommit `15b7634`に凍結した。2023～2025年のtrainおよび
過去parent match用validとのexact SFEN重複は0であり、途中対局には使わない。

64 root no-opでは5,305 train pairと4,316 valid pairを全合法手探索し、bound誤受理、違法手、
root復元失敗、非有限値は0だった。loss weight 0のmodelとoptimizerは更新前とbyte一致した。
1,024 root更新ではtrain 92,055 pair / 98,515,597 qnode、fresh-valid 89,865 pair /
179,737,746 qnodeを処理し、bound誤受理0、固定probe best-move不変率98.242%だった。
feature / hidden / output最大deltaはそれぞれ`2.111e-5 / 1.872e-5 / 1.721e-6`で上限を通過し、
material係数はbit不変だった。

両leaf backpropの先後別finite difference、T(delta) derivative、repetition、mate、max-qply、
save/reload、jobs決定論も通過した。詳細は
`training/halfkp64-scratch-v1/mmto_step3_verification_report_20260729.json`に固定した。

### Step 4 実行記録（2026-07-30）

同一10,000 train / 2,000 valid、各2 outer、pair batch 8,192、jobs 8で3 learning rateを実行した。
全runでrejected boundは0、material係数はbit固定、PV probeは90% Gateを通過した。
fresh-valid lossは`3e-6: 0.2289882`、`1e-5: 0.2268628`、`3e-5: 0.2221976`となり、
最終候補間では安全Gateを満たす最大値`3e-5`が最良だった。棋力対局やv2.5.0は選択に使っていない。
詳細は`training/halfkp64-scratch-v1/mmto_step4_calibration_report_20260730.json`に固定した。

後監査では、同じ2,000 rootの親model基準値、outer 1履歴、`|delta|/tau`分位点、有効derivative率、
outer別wall timeが保存されていないことを確認した。このため`3e-5`は暫定選択とし、Step 4を
「実行完了・判定は条件付き」と訂正する。learning rateを上げるほどloss / agreement / regretが同方向へ
改善し、`3e-5`の各outerのparameter deltaも上限を十分下回ったので、選択を撤回したり
3 LRを再実行したりはしない。
Step 5冒頭で記録機能を補い、`3e-5`だけを10k × 4 outerで再現する。

評価の説明は`docs/halfkp_mmto_post_step4_strategy_20260730.md`へ、現在の機械可読な実行正本は
`training/halfkp64-scratch-v1/mmto_post_step4_strategy_20260730.json`へ固定した。

## 10. 今回行わないこと

- Phase 14の閾値だけを緩めた再実行
- 旧HKSTの追加epoch
- depth scoreをimmediate childへ回帰
- current-search合意pairだけのhard negative
- exact / 戦型 / 詰み / resultのloss混合
- global weight decay
- factor-only再学習
- optimizer総当たり
- architecture拡大
- policy head
- 複数の並行training strategy
- 外部engine teacher
- v2.5.0を使った途中選択

過剰実装を避けるとは、短いpilotを次々に作ることではない。一つの既知の成功原理を意味論どおり実装し、
判定可能な量まで実行することである。
