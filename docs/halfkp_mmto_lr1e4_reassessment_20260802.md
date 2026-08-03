# HalfKP MMTO `1e-4` Gate再評価と次期学習計画

- 作成日: 2026-08-02
- 対象lineage: `halfkp64-scratch-v1`
- 状態: **`1e-4`を棄却から「安定性警告付き継続評価」へ変更**
- 現在の実行正本:
  `training/halfkp64-scratch-v1/mmto_lr1e4_reassessment_plan_20260802.json`
- 学習初期値: accepted scratch generation 2
  `9a1bfc9b7b317f451fa9a6efb350fed0a50e3020659686cefccc98646b9b3db6`
- 現行KPP由来重み・v2.5.0: 学習、教師生成、hyperparameter選択に使用しない

## 1. 結論

旧計画に記載した90% Gateへ従って`3e-4`への自動進行を止めた手続きは正しい。結果を見てから事前条件を
無視しなかったためである。しかし、`1e-4`を「不安定なので不採用」と確定した判断は妥当でない。
正しい現在の判定は、**数値安全性を満たしobjectiveが最も改善した、有力だが追加確認が必要な候補**である。

90% Gateが測っているのはPV全体、leaf局面、score margin、棋力ではなく、更新前後の固定1,024 rootにおける
top-1 moveの同一率だけである。MMTOは棋譜手が選ばれるよう着手順位を変える学習なので、best moveの変化自体を
失敗とはみなせない。さらに現probeはvalid file先頭1,024 root、48 gameだけからなり、game内の相関を無視した
hard thresholdとしては根拠が弱い。

従って次のように訂正する。

- `1e-4`: rejectではなく`held_for_review`へ戻す。
- root best-move維持率90%: hard rejectionからwarning / 追加監査triggerへ変更する。
- `3e-4`: 永久中止せず、`1e-4`の方向評価とouter 4確認後に一度だけ上限確認する。
- 採否: fixed-valid paired objectiveと固定node対局で決める。

ただし、Gateを結果確認後に変更するため、既存`1e-4` outer 2をそのままpromotionしてはならない。再評価手順を
凍結してから既存成果物を確認し、その後だけ学習を再開する。

## 2. 実測結果の再評価

| 指標 | `3e-5` outer 2 | `1e-4` outer 1 | `1e-4` outer 2 | 解釈 |
|---|---:|---:|---:|---|
| fresh-valid loss | 0.22219761 | 0.21655246 | **0.21251359** | `3e-5`比4.36%改善 |
| 棋譜手一致率 | 13.20% | 13.35% | **13.55%** | 小幅だが悪化なし |
| 平均regret | 347.61 cp | 343.96 cp | **341.34 cp** | `3e-5`比6.27 cp改善 |
| valid qnodes / root | 159,221 | 161,436 | 163,693 | `3e-5`比+2.81%、10%警告未満 |
| best-move維持率 | 97.17% | **87.70%** | 94.82% | outer 1だけ旧Gate未達 |
| qsearch rejected bound | 0 | 0 | 0 | 合格 |

outer 1は898 / 1,024 rootが同一、126 rootが変化した。90%境界を満たすには922 root必要なので、違反は
24 root分である。違反自体は実在するが、変更した126 rootが棋譜手へ近づいたのか、離れたのか、非棋譜手間で
入れ替わっただけかは現記録から分からない。これを調べずに棄却することはできない。

parameter安全性にも破局の兆候はない。

| outer | feature最大delta | hidden最大delta | output最大delta | 上限に対する最大使用率 |
|---:|---:|---:|---:|---:|
| 1 | 0.00169079 | 0.00142598 | 0.00015946 | feature 8.45%、output 3.99% |
| 2 | 0.00075829 | 0.00067669 | 0.00007096 | feature 3.79%、output 1.77% |

gradient clipはouter 1で8 / 110 batch、outer 2で11 / 110 batchだった。active derivative率は37.54%から
36.12%である。`|delta|/tau p99=81`は、現実装が80以上を最終bucketへまとめ、bucket上端を返した表示であり、
正確なp99=81とは解釈しない。表示を「80以上の比率」へ直す。

zero accumulatorから始まるpersistent AdaGradでは最初のouterが最大更新になりやすい。outer 2でparameter deltaが
縮小し、best-move維持率が94.82%へ回復し、lossもさらに1.87%改善したことは、発散が加速する形と整合しない。

## 3. 先行事例との整合

Hoki・KanekoのMMTO原論文は、固定PVを用いる局所近似のため更新を十分小さくする必要を述べるが、90%という
採否閾値は定義していない。性能は目的関数、独立test setの棋譜手一致率、最終的なElo対局で評価している。
また、論文のfull PV保持と本実装のroot top-1同一率は別物である。
[MMTO原論文](https://www.jair.org/index.php/jair/article/download/10871/25935)

quiescence searchとAdaGradを用いた将棋の方策勾配教師あり学習も、学習対象を棋譜手選択としている。
best moveを変えないこと自体を性能目標にはしていない。
[GPW 2017論文](https://ipsj.ixsq.nii.ac.jp/record/183828/files/IPSJ-GPWS2017001.pdf)

Stockfish公式NNUE trainerもvalidation lossでcheckpointを比較し、最終的なnet採否には対局を用いる。
実装、optimizer、score尺度が異なるのでlearning rateの数値は転用しないが、offline objectiveと対局を分ける
原則は採用する。
[公式trainer](https://github.com/official-stockfish/nnue-pytorch)、
[FishtestのNNUE net test](https://github.com/official-stockfish/fishtest/wiki/Creating-my-first-test#nnue-net-tests)

調査した一次資料には、更新前後のroot best-move同一率90%を単独のmodel採否Gateにする根拠は見つからなかった。

## 4. 修正後のGate

### 4.1 即時停止するhard safety Gate

次のいずれかが発生したmodelは保存して停止し、次段階へ進めない。

1. dataset、parent、binary、checkpoint provenance不一致
2. NaN / Inf、違法手、root復元失敗
3. full-window qsearchのbound rejection
4. material係数のbit変化
5. parameter bound接触
6. 1 outerのfeature / hidden delta 0.02以上、output delta 0.004以上
7. model / optimizer / checkpoint SHA不一致

### 4.2 品質の進行Gate

- 同じ固定validでcandidate-parentのpair loss差をgame単位でpaired bootstrapする。
- candidateのloss悪化が95%区間で明確な場合は次outerへ進めない。
- 二つの連続checkpoint区間でfixed-valid lossが悪化した場合は、そのrunを終了して最良checkpointへ戻る。
- 同程度なら低いlearning rateを選ぶ。

### 4.3 warningに変更する指標

- root best-move維持率90%未満
- qnodes / rootが基準比10%以上増加
- clip率、active derivative率、parameter deltaの急変

warningは原因調査を開始するが、それ単独で棄却しない。best-move変化はteacherへ向かった、teacherから離れた、
非teacher間で変化、の三種類に分けて解釈する。別の根拠のない固定閾値へ置き換えない。

## 5. 作業計画

### Phase 0: 成果物保全と判定訂正

- `1e-4` outer 1の`model.previous` / `optimizer.previous`とouter 2成果物をimmutable名で保存する。
- SHA、dataset manifest、parent lineage、実行commandをmanifestへ固定する。
- 旧`reject_stability_gate`を履歴として残し、現在判定を`held_for_review`と明記する。

完了条件: resumeしてもouter 1 / 2を上書きせず再評価できる。

### Phase 1: 採否に必要な最小評価を実装する

新lossやdashboardは作らず、MMTOのfresh eval-only出力だけを追加する。

- valid 90,759 root、6,099 gameから、seed固定hashで各game 1 rootを選ぶ6,099-root audit setを作る。
- final reserveは読まず、既存validだけを使う。
- 各rootについてgame key、root hash、teacher move、best move、teacher score、best score、pair loss、regret、
  qnodesをJSONLへ保存する。
- model間でwrong→teacher、teacher→wrong、non-teacher→non-teacher、unchangedを集計する。
- game-key単位paired bootstrapでloss差の95%区間を出す。
- parent、`3e-5` outer 1 / 2、`1e-4` outer 1 / 2を同じaudit setで評価する。
- histogram最終bucketは`>=80`として保存し、偽のp99値を出さない。

現在の先頭1,024 probeは48 game、先頭2,000 validは97 gameしか含まない。audit setをgame横断にするのは、
新しい評価課題を増やすためではなく、同一game内の連続局面を独立sampleとして扱う偏りを減らすためである。

完了条件: `1e-4`の改善が一部gameだけでなくgame横断で確認でき、変更したbest moveの方向を説明できる。

Phase 1は完了した。6,099 game各1 rootのauditでは、parentのroot-equal mean pair lossは`0.22240457`だった。
`3e-5` outer 2は`0.21644027`、`1e-4` outer 1は`0.21189074`、`1e-4` outer 2は`0.20841469`まで低下した。
parentとの差のgame単位bootstrap 95%区間は、それぞれ`[-0.00667850,-0.00525955]`、
`[-0.01162620,-0.00938168]`、`[-0.01552393,-0.01246939]`で、全候補がparentより明確に改善した。
`1e-4` outer 2ではwrong→teacherが137 root、teacher→wrongが60 rootで、変更方向も正である。
qnodes/rootはparent比6.63%増で、10% warning域内だった。従って旧90% Gate違反は品質悪化を示しておらず、
Phase 2の`1e-4` outer 4へ進む。

### Phase 2: `1e-4`をouter 4まで確認する

- 既存outer 2 checkpointからouter 3、4へresumeする。
- 各outerでhard safety、固定audit、集約validを評価する。
- lossが改善する限り継続する。outer 4が悪化しても、outer 1～4の最良checkpointを候補として保持する。
- 90%未達は記録・方向分析するが、単独停止条件にしない。

概算: 約1.8時間。

実績: outer 3はvalid loss `0.20995277`、outer 4は`0.20806211`まで改善した。PV probeの
best-move維持率も`0.9736`、`0.9854`へ回復し、hard safety違反はなかった。outer 3/4の
checkpointは`checkpoints/archive_manifest.json`へSHA-256付きで保全した。従ってPhase 2は完了とし、
次の`3e-4` bounded screenへ進む。

### Phase 3: `3e-4`で上限を一度だけ確認する

- accepted generation 2から独立に`3e-4`、10k train、2k集約validでouter 1を実行する。
- hard safetyを通り、同じouter 1の`1e-4`より固定audit lossが低い場合だけouter 2へ進む。
- outer 2でも`1e-4`より良い場合だけouter 4へ延長する。
- 差が不明確なら低い`1e-4`を選ぶ。
- `1e-3`、中間learning rate、新optimizer、手動scheduleは試さない。

概算: outer 1だけ約0.9時間、最大outer 4まで約3.6時間。

実績: `3e-4` outer 1は集約valid loss `0.20427213`、固定6,099局面audit loss `0.20189649`、
regret `707.29cp`だった。`1e-4` outer 1のaudit loss `0.21189074`との差は`-0.01051382`で、
game bootstrap 95%区間は`[-0.02211544,-0.01883100]`。rejected boundsは0、qnodes増加も約7.5%で
hard safetyを満たしたため、outer 2へ進む。

### Phase 4: learning rateを凍結する

`3e-5 / 1e-4 / 3e-4`の安全なcheckpointを、同じouter数、同じ固定audit、同じfresh searchで比較する。

選択順:

1. hard safety全合格
2. game-paired fixed-audit loss最小
3. regretとteacher方向の変化が非悪化
4. 差が不明確なら低いlearning rate

ここでlearning rate探索を終了する。棋力が出るまで小刻みに率を増減する作業は行わない。

### Phase 5: 100kを1 outerだけ確認する

- accepted generation 2から、凍結LR、100,000 train root、10,000集約valid、1 outerを実行する。
- 10k校正のcheckpointを初期値に使わない。
- 累積更新量、qnodes、温度、resume、固定audit改善を確認する。
- 失敗時だけ一段低いscreen済みLRへ戻す。

従来の100k × 4は省略する。pilotを約32～36時間回してからfullを親からやり直すより、1 outer約8～9時間で
規模依存の異常だけ確認し、実data量へ計算時間を使う。

### Phase 6: full datasetを段階学習する

- accepted generation 2から凍結LRでproductionを開始する。
- 100k modelは継承せず、100kは規模確認に限定する。
- milestoneはouter `1 / 2 / 4 / 8 / 16`。
- 毎outerで6,099-root audit、milestoneでfull 90,759-root validを実行する。
- 各milestoneのmodel / optimizer / manifest / metricsをimmutable保存する。
- final outerではなく、fresh-valid最良の安全なcheckpointを候補にする。
- manual LR decayは追加せずpersistent AdaGradへ任せる。

full 1 outerは約40～50時間であり、50時間以上学習したv2.5.0と初めて計算時間の桁が揃う。outer 1を評価せず
「短時間で勝てない」と結論しない。outer 16以降は事前予約せず、valid曲線と対局結果を見て追加する。
旧計画の200 outerへ現時点でcommitしない。

### Phase 7: 棋力評価と停止条件

- full outer 1以降、full-validを更新したcheckpointをaccepted generation 2と先後交換500局で診断する。
- paired score 95%上限が45%未満なら明白な棋力崩壊として停止する。
- paired score 95%下限が50%を超えればscratch parentをpromotionする。
- 中間結果が不明確でoffline曲線が改善中なら次milestoneへ進む。
- 最終checkpoint凍結後だけ、予約済み4,000局でv2.5.0と比較する。
- v2.5.0は学習、model選択、途中LR調整に使わない。

full outer 1、2、4まで十分な学習量を与えてもMMTO lossだけ改善し、scratch parentとの対局が一貫して改善しない
場合は、実装失敗ではなくWDoor棋譜手模倣objectiveと棋力の乖離を第一仮説とする。その時点で初めて、scratch
lineage自身の深い探索または自己対局から教師を作る次計画へ移る。現段階ではarchitecture、regularizer、型課題、
現行重みteacherを追加しない。

## 6. 進捗表

| Phase | 状態 | 次の成果物 |
|---|---|---|
| 0 成果物保全・判定訂正 | completed | immutable outer 1 / 2、訂正manifest |
| 1 per-root fixed audit | completed | 6,099-root set、paired report |
| 2 `1e-4` outer 4 | completed | outer 3 / 4 metrics、SHA-256 archive |
| 3 `3e-4` bounded screen | in progress | outer 1 fixed-audit判定 |
| 4 LR凍結 | blocked by Phase 3 | selection report |
| 5 100k × 1 | blocked by Phase 4 | scale confirmation |
| 6 full staged run | blocked by Phase 5 | outer 1 / 2 / 4 / 8 / 16 |
| 7 対局・最終比較 | milestone連動 | parent match、最終v2.5.0 match |

この計画の最初の目的は、`1e-4`を無条件採用することではない。学習による有益な着手変更を「不安定」と誤認せず、
かつ小さなvalid subsetの改善だけで長期学習へ飛ばない判定系を、最小限の追加実装で作ることである。
