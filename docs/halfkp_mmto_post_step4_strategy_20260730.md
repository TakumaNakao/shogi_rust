# HalfKP MMTO Step 4結果評価と次段階の実行方針

- 作成日: 2026-07-30
- 対象lineage: `halfkp64-scratch-v1`
- 状態: **2026-08-02のGate再評価により実行正本から置換済み**
- 初期model: accepted scratch generation 2
  `9a1bfc9b7b317f451fa9a6efb350fed0a50e3020659686cefccc98646b9b3db6`
- 暫定learning rate: feature / hidden `3e-5`、output `3e-6`
- 現行KPP由来重みおよびv2.5.0: 学習・選択に使わず、凍結候補の最終比較だけに使う
- 元計画: `docs/halfkp_prior_art_recovery_plan_20260729.md`
- Step 4集約結果: `training/halfkp64-scratch-v1/mmto_step4_calibration_report_20260730.json`
- 選択run一次成果物:
  `runs/halfkp64-scratch-v1/prior-art-step4-lr3e-5-v1/run_manifest.json`
- 現在の機械可読な実行正本:
  `training/halfkp64-scratch-v1/mmto_lr1e4_reassessment_plan_20260802.json`
- 置換後の説明: `docs/halfkp_mmto_lr1e4_reassessment_20260802.md`

## 1. 結論

Step 4の結果は、**learning rate候補間の相対校正としては良好**である。3条件を同じ初期model、同じ
train / valid root、同じ決定論的な入力順で比較したところ、learning rateを上げるほど候補最終時点の
fresh-search valid loss、棋譜手一致率、平均regretが同じ方向へ改善した。
outer 1からouter 2への固定probe root best-move維持率は97.17%であり、最後の1 outerが
大きすぎる兆候もない。この指標は親から最終modelまでの累積安定性でも、PV列やPV末端局面の同一率でもない。

ただし、**より強い重みができたとはまだ判定できない**。保存済み成果物には同じ2,000 valid rootでの
親model（outer 0）とouter 1のfresh評価がなく、親からの改善曲線を証明できない。対局評価もまだ行っていない。
従って現在の総合判定は次のとおりとする。

| 判定対象 | 判定 |
|---|---|
| qsearch leaf・勾配・dataset provenance | 合格。既知の致命的な再発なし |
| 候補最終値間のlearning rate dose response | 良好 |
| 親からの絶対的な学習効果 | 未証明 |
| `3e-5`の数値・parameter安全性 | 合格 |
| `3e-5`のroot best-move安定性 | 保存済みouter 1→2だけ合格 |
| `3e-5`の選択 | **暫定採用** |
| accepted generation 2からの改善 | 未証明 |
| v2.5.0を上回る棋力 | 未評価 |
| 開発全体 | **学習基盤は前進、棋力目標はまだ未判定** |

ここで別loss、別architecture、型課題、外部teacherへ移る根拠はない。一方、記録不足のまま約1年規模の
full 200 outerへ直行するのも妥当でない。最短経路は、判定に必要な記録だけを追加し、`3e-5`一本で
10k再現、100k検証、full段階実行へ進むことである。

## 2. Step 4の実測

### 2.1 学習率別結果

| feature / hidden LR | fresh-valid loss | 棋譜手一致率 | 平均regret | fixed-probe best-move変化率（outer 1→2） |
|---:|---:|---:|---:|---:|
| `3e-6` | 0.22898825 | 12.50% | 353.52 cp | 1.66% |
| `1e-5` | 0.22686275 | 12.75% | 351.53 cp | 1.95% |
| `3e-5` | **0.22219761** | **13.20%** | **347.61 cp** | 2.83% |

`3e-6`から`3e-5`への変化は次のとおりである。

- fresh-valid loss: `-0.00679064`、相対`-2.97%`
- 棋譜手一致率: `+0.70 point`、2,000 root中`+14`
- 平均regret: `-5.91 cp`

loss、agreement、regretの三つが同じ順序を示したことは正の証拠である。ただしagreementの差は14 rootだけで、
単独では強い統計的根拠にしない。per-root / game単位のpaired記録がないため、現時点では信頼区間も算出しない。

### 2.2 保存済み範囲の安全性

| 指標 | `3e-5`実測 | per-outer上限 | 判定 |
|---|---:|---:|---:|
| outer 1 feature / hidden / output最大delta | 0.000506818 / 0.000482589 / 0.000052813 | 0.02 / 0.02 / 0.004 | 合格 |
| outer 2 feature / hidden / output最大delta | 0.000213608 / 0.000218540 / 0.000022842 | 0.02 / 0.02 / 0.004 | 合格 |
| outer 1→2固定probe root best-move維持率 | 97.17% | 下限90% | 合格 |
| qsearch bound rejection | 0 | 0 | 合格 |
| material係数 | bit不変 | bit不変 | 合格 |

root best-move安定性Gateが厳しすぎるという兆候はない。各runの保存済みouter 1→2では97%以上を維持し、
90%下限は保存されたouter 1→2の比較では選択を制限していない。ただしouter 0→1の値は保存されておらず、
全outerの安定性合格はPhase B後に確定する。trainerはparameter delta上限超過時に停止するため、
完走した両outerのparameter Gateは通過している。

### 2.3 学習量と計算量

1条件の校正は10,000 rootを2 outer、899,932 pair / outer処理した。full trainは実datasetの
`legal_moves - 1`合計で56,035,622 pair / outer、200 outerでは11,207,124,400 pairになる。
従って1条件の校正量はその約0.01606%でしかない。10,000 rootもfull train
570,875 rootの1.75%である。2 outerの校正を50時間以上学習したv2.5.0との棋力比較に使ってはならない。

run directoryの生成時刻から最終manifest時刻までの差は、jobs 8で約51～55分 / outerだった。
区間wall timeを保存していないため、これはartifact時刻からの推定である。qsearch node数からの線形外挿は
次の範囲になる。

| 作業 | 概算 |
|---|---:|
| 10k × 4 outer再現 | 約3.5～4時間 |
| 100k train + 10k valid × 4 outer | 約32～36時間 |
| full 570,875 trainの1 outer | 約40～50時間 |
| full 200 outer | 約8,000～10,000時間、約330～420日 |

これはノートPCの温度制御、停止、full-valid評価時間を含まない概算である。Step 5の区間計測で更新する。
2026-07-30時点の空き容量は約271 GiB、model + optimizerは1 checkpoint約54 MBなので、
指定milestoneだけを残す方式ではdisk容量は律速にならない。

## 3. 現成果物から断定できないこと

### 3.1 親からの改善曲線

`run_manifest.json`はouterごとに上書きされ、最終outerしか残していない。`model.previous`にouter 1のmodelは
残るが、そのfresh-valid指標は残らない。また親modelを同じ2,000 valid rootで測ったouter 0基準値もない。

従って「三つの最終候補では`3e-5`が最良」とは言えるが、次はまだ証明できない。

- 親modelよりouter 1が良い
- outer 1よりouter 2が良い
- 同じvalid root上で改善が単調または持続している

Step 3のvalid値は1,024 rootであり、今回の2,000 root値と直接比較しない。train lossは更新前search
snapshotで生成したleaf上の値、valid lossは更新後のfresh search値なので、両者の差を通常の
train / validation gapとしても解釈しない。

### 3.2 勾配飽和

計画で必須とした次の診断値が実装・保存されていない。

- `|desired_score - alternative_score| / tau`のp50 / p90 / p99
- mirrored-sigmoid derivativeが最大値の1%以上あるpair率
- regretのmedian / p90 / p95 / p99
- gradient clippingのbatch数と率

`tau=36.63 cp`に対してvalid平均regretが約348 cp、train平均が約1,155 cpである。平均はmate等の外れ値に
影響されるため、この値だけで飽和とも健全とも判定できない。上記分位点を記録してから判断する。

### 3.3 棋力

Step 4はlearning rate校正であり、対局は行っていない。fresh MMTO lossの改善は必要な学習信号だが、
棋力改善の十分条件ではない。accepted generation 2やv2.5.0より強いという表現は、所定の対局を完了するまで
使わない。

## 4. 修正後の実行計画

### 4.0 追加した高learning-rate短期screen

`3e-5`は校正範囲の上端であり、そこで改善が続いているため、長期実行の前に探索範囲を一段だけ広げる。
同じaccepted generation 2、凍結dataset、10,000 train root、2,000 valid root、2 outer、同じoptimizerと
安全上限で、次を順に実行する。

1. feature/hidden `1e-4`（output `1e-5`）
2. 1が安全かつ固定probe維持率90%以上なら feature/hidden `3e-4`（output `3e-5`）

非有限値、parameter bound接触、qsearch拒否、または固定probe維持率90%未満で即停止する。安全に完走した候補だけを
fresh-valid loss、agreement、regret、更新量分位点で比較し、最小lossの候補をouter 4まで延長する。
`1e-3`はこのscreenでは実行しない。`3e-4`でも改善が続き、更新診断が飽和していない場合に限り、別判断として再検討する。
このscreenは校正であり、v2.5.0を上回ったことを意味しない。

以下はscreen直後の旧判定であり、2026-08-02の再評価で置換した。`1e-4`のscreenは完了した。
valid fresh lossはouter 1で`0.21655246`、outer 2で`0.21251359`まで改善したが、
outer 1の固定probe best-move維持率が`0.876953125`であり、90% Gateに違反した。outer 2は`0.9482421875`まで
戻ったものの、全outer合格条件を満たさないため、この候補は採用せず、`3e-4`も実行しない。更新診断では
`|delta|/tau`のp50/p90/p99がouter 2で`10/31/81`、active derivative率`0.3612`、gradient clip率`11/110`
であり、単純な飽和よりも初回更新の不安定性が問題と判断する。結果は
`runs/halfkp64-scratch-v1/prior-art-step4-lr1e-4-v2/`に保存した。

### Phase A: 長期実行に必要な最小記録を追加する

`halfkp_mmto_train`の学習式、探索条件、optimizer、parameter更新順は変えず、次だけを追加する。

1. outer 0の固定valid fresh評価と、全outerの集約履歴を保存する。
2. 指定outer完了後にmodel、optimizer、manifestをimmutableな名前で保存する。
3. search / update / valid / saveの単調時計による経過時間を保存する。
4. `|delta|/tau`分位点、有効derivative率、regret分位点、gradient clip率を保存する。
5. per-outer feature / hidden / output最大deltaを保存する。
6. 固定probeでqsearch PV length、PV sequence hash、leaf position hashの更新前後一致率を保存する。
7. qnodes / rootと実行binary SHAを保存する。

`|delta|/tau`と有効derivative率はsearch snapshotの固定scoreではなく、`add_pair_gradient`が各batchで
leafを再forwardした時点の**更新側weightによるdelta**から集計する。gradient clip率は`apply_batch`で
実際にclipしたbatch数 / 全更新batch数とする。

成果物は次の形に固定する。

```text
output-dir/
  checkpoint.json
  run_manifest.json
  metrics/
    outer_0000.json
    outer_0001.json
    outer_0002.json
  checkpoints/
    outer_0001/{model.bin,optimizer.bin,manifest.json}
    outer_0002/{model.bin,optimizer.bin,manifest.json}
```

各`outer_NNNN.json`は一時fileからrenameしてatomicに保存し、checkpointのouter番号とmodel /
optimizer SHAを照合する。追記logの独自復旧機構は作らない。full runでは指定milestoneだけをarchiveし、
全200 modelを複製しない。

最低限のtestは次の三つとする。

- loss weight 0でmodel / optimizerがbyte identityのまま履歴とbaselineを保存できる。
- outer 1完了後のresumeが、連続outer 2実行とmodel / optimizer byte一致する。
- archiveのSHA不一致、metricsとcheckpointのouter不一致、dataset provenance不一致を拒否する。

per-root / game JSONL、parameter RMS / p99はfull production開始前に追加すれば足り、100k前のblocking要件にしない。
新loss、新regularizer、新architecture、dashboard、block途中resumeは追加しない。

### Phase B: `3e-5`だけを10k × 4 outerで再現する

三つのlearning rateは再実行しない。accepted generation 2から、Step 4と同じtrain 10,000 /
valid 2,000、`3e-5`、jobs 8、pair batch 8,192で4 outer実行する。

outer 2では旧Step 4の次のSHAとのbyte一致を確認する。

- model:
  `fd82e4e028f8aa9b41b94387058a6c3105c5302ca2dd9b5b6e80ac6b755daaac`
- optimizer:
  `6119bb2ea8cb51264b7463abfd96626a86569e4851eb4ee1dacaaaa5d1eb1744`

旧Step 4ではbinary SHAを保存していないため、binary provenanceは新run内とresume間の一致を意味する。
同じcompiler条件で計測追加が学習算術へ影響していなければouter 2成果物も一致する。不一致なら100kへ進まず、
更新順や並列決定性の差を調べる。compiler差だけが原因と再現・立証できた場合に限り、全metricとparameterの
tight tolerance比較へ切り替え、理由をmanifestへ残す。

100kへ進むGate:

- dataset / parent / binary provenance一致
- 非有限値、違法手、root復元失敗、qsearch bound rejectionが0
- material係数bit不変、parameter上限非接触
- outer 2が旧成果物とbyte一致
- outer 4の同一valid fresh lossがouter 0より低い
- 各outerの固定probe root best-move維持率90%以上

agreement、regret、飽和率は診断値として併記する。agreementが小幅という理由だけでは止めない。
qnodes / rootがouter 0比10%以上増えた場合は警告し、探索暴走か局面差かを確認するが、単独の方法棄却条件にはしない。

### Phase C: jobs benchmarkを短縮して実行する

決定論は既にjobs 1 / 4で確認済みなので、10,000 rootを3回使わない。同一1,024 train /
1,024 valid、accepted generation 2、`3e-5`、1 outerでjobs 1 / 4 / 8を一度ずつ測り、
model / optimizer byte一致を確認して最速設定を採用する。
現在の第一候補はjobs 8である。

### Phase D: 100k × 4 outerで学習成立性を判定する

校正modelを継続学習せず、accepted generation 2から独立に開始する。

- train 100,000 / valid 10,000
- feature / hidden LR `3e-5`、output LR `3e-6`
- 4 outer、pair batch 8,192
- Phase Cで選んだjobs
- checkpoint 0 / 1 / 2 / 3 / 4

まずouter 1を完了・検証し、安全Gate通過後に同じrunを`--resume`してouter 4まで進める。
outer 4で次を満たせばMMTO目的の学習成立と判定する。

- outer 4のfixed-valid fresh lossがouter 0より低い
- lossが継続的に発散していない
- semantic / safety Gateが全outerで通過
- parameter、qnodes、飽和率に継続不能な異常がない

outer 4後にaccepted generation 2との先後交換500局を行う。これは最初の棋力診断であり、
通常の小差や互角はcheckpoint選択や早期棄却に使わない。ただし、opening pair単位paired scoreの
95%信頼区間上限が45%未満なら、明白な棋力崩壊としてfull開始を保留し、目的と棋力の乖離を監査する。
これは別戦略への自動移行条件ではない。Gateに触れずfresh objectiveが改善中なら、学習量不足の仮説を維持する。

中間診断は凍結済み高rate WDoor valid splitから250 opening pairをseed `20260731`で一度だけ生成・凍結し、
train splitおよび予約済み最終2,000 pairとのexact SFEN重複を0にする。test splitと、最終用に全体を
予約した2026年sourceは読まない。同じsuiteを全parent診断で使い、20,000 nodes / move、
max depth 64、max 256 plies、先後交換500局に固定する。paired intervalはopening pairをclusterとし、
100,000 resample、seed `20260731`で計算する。このsuiteをv2.5.0の最終比較には使わない。

opening生成順は対局loaderの制約を含め、次へ固定する。

1. sourceから`split == valid`かつ`8 <= ply <= 40`だけを残す。
2. trainと最終reserveにexact SFENがあるrecordを除く。
3. `sha256(seed\0game_key\0sfen)`の昇順に並べる。
4. 既出`opening_family`または既出exact SFENを飛ばし、先頭250件を採用する。
5. 250件未満なら実行せず失敗とする。
6. manifestへsource / output SHA、records 250、unique family 250、ply範囲、各overlap 0を保存する。

計画時のread-only監査ではfilter後14,206 recordから250 unique family / 250 unique SFENを選択でき、
ply範囲は8～40だった。実artifact生成時にも同じ検査を再実行する。

現行`halfkp_promotion_match`はgame単位Wilson下限しか出さない。Phase D対局前に、2局を1 opening pairとして
pair scoreを作り、pairを復元抽出するpercentile bootstrapの2.5% / 97.5%点を出力する。resample数とseedを
manifestへ固定し、既知の小配列、同一seed決定論、pair順入替えでtestする。破局監査にはこの95%上限だけを使う。

### Phase E: full runをmilestone単位で運用する

100kの成立確認後に、accepted generation 2から新しいproduction runを開始する。100k pilot modelを
初期値に使わない。full開始前にvalidのper-root loss / agreement / regret / qnodesと`game_key`、
parameter RMS / p99 / maxを保存できるようにする。また、学習modelを更新せず、指定checkpointを
full-valid 90,759 rootで評価して独立artifactへ保存するeval-only経路を追加する。これらの観測機能追加後、
loss weight 0のmodel / optimizer byte identityと、短い既知runが変更前成果物とbyte一致することを
再確認する。productionの更新を始める前に、accepted generation 2（outer 0）を同じ90,759 rootで
eval-only評価し、後続milestoneとの固定baselineとして凍結する。

- train 570,875 root
- valid probe 10,000 root / milestone full valid 90,759 root
- learning rate `3e-5` / output `3e-6`
- checkpoint milestone 1 / 2 / 4 / 8 / 16
- 親対局診断: 初期8 / 16、後半32 / 64 / 100 / 144 / 200で各500局
- 16まで通過後に32 / 64 / 100 / 144 / 200を継続

200 outerはBonanza実験に由来する上限milestoneとして残すが、原法は1回のPV生成につき32回の全体更新を
行っており、現実装の1-pass mini-batch outerと同じ学習量ではない。約1年の単一jobとして起動せず、
atomic resume、disk、温度、wall time、fresh loss曲線を各milestoneで確認する。

iteration 16から後半へ継続する条件は事前に次へ固定する。

- provenance、semantic、数値、parameterの全safety Gateが通過している。
- iteration 16のfull-valid fresh lossがouter 0より低い。
- `loss(4) < loss(8) < loss(16)`という2区間連続の悪化がない。
- resume、disk、温度に長期運用上の異常がない。

条件を満たせば32へ進み、同じ基準で次の事前定義milestoneへ自動継続する。iteration 8 / 16では
accepted generation 2との先後交換500局を診断として行う。通常の小差は停止・選択条件にしないが、
Phase Dと同じ「opening pair単位paired scoreの95%上限が45%未満」だけは次区間を保留する破局監査Gateとする。
32以降も、current lossがouter 0より低く、直近2 milestone区間が連続悪化していなければ次へ進む。

実行は`--outer-iterations 1`、`--resume --outer-iterations 2 / 4 / 8 / 16`の別invocationに分け、
各milestoneで終了状態を確認する。現行checkpoint粒度ではouter途中から再開できないため、障害時に最大
1 outer、約40～50時間を再実行するRPOを受容する。block単位resumeは実装しない。外部からCPU温度、
空き容量、process heartbeatを監視し、異常停止時は既存checkpoint SHAを検証してresumeする。

同じPV leafの複数pass再利用は現在の1-pass固定条件を変えるため、本計画には含めない。100kと初期full
milestoneの結果を得た後に必要性が生じた場合だけ、Bonanza原法のglobal reuseとstreaming block-local近似を
区別した別計画を作る。

### Phase F: 最終評価

checkpoint選択、test、v2.5.0比較は元計画を維持する。

1. safety Gateを通過した指定checkpointのうちfull-valid fresh loss最小の一つを凍結する。
2. 凍結後にtest 62,233 rootを一度だけ評価する。
3. 予約済み2,000 opening pair、先後交換4,000局でv2.5.0と比較する。
4. opening pairをclusterとしたpaired bootstrapの95%下限が50%を超えた場合だけ
   「v2.5.0より強い」と判定する。

v2.5.0の結果をlearning rate、loss、dataset、checkpoint選択へ戻さない。
親診断と最終比較はいずれも20,000 nodes / move、max depth 64、max 256 pliesに固定する。

## 5. 判断の要約

現在の結果は失敗ではない。過去の実験と異なり、正しい高rate WDoor dataset、current-model qsearch
PV leaf、両leafへの勾配、全合法代替手という本線を完走し、候補最終値のlearning rate dose responseが
初めて得られた。致命的な学習バグを示す証拠もない。一方、親からの絶対改善はPhase Bまで未証明である。

ただし「順調」を二つに分ける必要がある。

- **学習基盤の開発**: 前進している。Step 4は条件付き合格。
- **v2.5.0より強い重みの開発**: まだ未判定。学習量と対局証拠が不足。

次に必要なのは新戦略の追加ではなく、`3e-5`一本の改善曲線を失わず記録し、100kからfullへ段階的に
学習量を増やすことである。
