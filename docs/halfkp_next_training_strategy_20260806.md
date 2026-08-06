# HalfKP scratch 次期学習方針（MMTO 10k・対局結果後）

- 決定日: 2026-08-06
- 対象lineage: `halfkp64-scratch-v1`
- 機械可読な実行正本:
  `training/halfkp64-scratch-v1/next_training_plan_20260806.json`
- 旧計画:
  `docs/halfkp_mmto_lr1e4_reassessment_20260802.md`
- 制約: Phase A〜Cの無人実行でKPP由来重みおよびv2.5.0を初期値、教師、target、対局相手、選別器のいずれにも使用しない

## 1. 決定

`3e-4 outer4`をrelease候補として採用しない判断は維持する。ただし、現MMTO経路そのものを失敗として
打ち切る判断もしない。正しい現状認識は次の通りである。

> **learning-rate校正とMMTO objectiveの学習・汎化には成功したが、棋力への転移はまだ未判定である。**

次の主経路を以下に固定する。

1. `3e-4 outer4`とaccepted generation 2を直接500局対局させ、MMTO更新方向を診断する。
2. game横断で均衡化した100k rootを`3e-4`で1 outerだけ学習し、規模安全性を確認する。
3. 100k modelを継承せず、accepted generation 2から既存570,875 root全量をouter 1、条件付きでouter 2まで学習する。outer 2で明確な棋力・audit改善を確認できた場合だけ、同じGateでouter 3 / 4を一段ずつ自動許可する。
4. full outer 1 / 2でもofflineだけ改善して親対局が改善しなければ、棋譜手MMTOと棋力の乖離を確定する。
5. その場合だけ、終局WDL補助を加えた単一A/Bを行う。それも失敗した場合に、scratch-only自己対局と
   TDLeaf / RootStrap型の探索bootstrapへ移る。

`100k × 4`、LR grid、architecture変更、AlphaZero型policy/MCTS実装は行わない。v2.5.0は最終候補を
凍結した後の比較相手に戻し、途中のhyperparameterやcheckpoint選択には使わない。

## 2. 今回どこまで進展したか

### 2.1 objective学習には明確な進展がある

固定6,099 game各1 rootのauditでは次の改善が得られた。

| 指標 | accepted generation 2 | `3e-4 outer4` | 変化 |
|---|---:|---:|---:|
| root-equal pair loss | 0.22240457 | **0.19081546** | -14.20% |
| 棋譜手一致率 | 14.527% | **16.445%** | +1.918 point |
| mean regret | 731.03 cp | **692.83 cp** | -38.20 cp |
| wrong → teacher | - | 229 root | 正方向 |
| teacher → wrong | - | 112 root | 負方向 |
| qnodes / root | 190,060 | 212,032 | +11.56% warning |
| rejected bounds | 0 | 0 | hard safety合格 |

loss差のgame bootstrap 95%区間は`[-0.03393915,-0.02913128]`で0を含まない。学習に使った
先頭10,000 rootは459 gameだけだが、auditは独立した6,099 gameを横断している。従って、単なる
同一game内暗記ではなく、棋譜手を選ぶMMTO objectiveが別gameへ汎化したことは確認できた。

### 2.2 棋力向上はまだ証明できていない

同じ1秒・200局・seed・opening集合でv2.5.0と比較した結果は次の通りだった。

| model | 勝-敗-分 | score | 95%区間 | v2.5.0比Elo概算 |
|---|---:|---:|---:|---:|
| accepted generation 2 | 61-139-0 | 30.50% | 24.12–36.88% | -143 |
| `3e-4 outer4` | 64-135-1 | 32.25% | 25.79–38.71% | -129 |

候補の点推定は親より+1.75 point（約+14 Elo）だが、これは候補と親の直接対局ではない。同じgame番号を
介したproxy paired再集計でも、候補改善28 opening pair、悪化26、同値46、平均差+1.75 point、
bootstrap 95%区間は`[-7.00,+10.25]` pointだった。親より強いとも弱いとも言えない。

色別の候補－親差は先手で+13.0 point、後手で-9.5 pointと不均質だった。一方、fixed audit lossは
先後とも改善しているため、単純なside符号バグを示す形ではない。探索trajectoryと有限局数の揺れを、
親との直接・先後交換対局で切り分ける必要がある。

したがって「v2.5.0を超えなかったので今回の学習は無価値」でも、「offline lossが改善したので順調」でもない。
実装とobjective方向は前進したが、最終目標である棋力については未証明である。

### 2.3 学習量は明らかに不足している

`3e-4 outer4`は同じ10,000 rootを4 outer処理し、学習pairは合計3,599,728だった。既存full trainを
1 outer処理した場合の56,035,622 pairに対して**6.42%**でしかない。unique rootは10,000、gameは459で、
利用可能な570,875 root / 34,582 gameの一部である。

MMTO原論文は48,566棋譜、4,467,116 desired move、368,313,024 move pairを用い、200 iterationの結果を
対局で評価している。本リポジトリとfeature・optimizer・pair定義は異なるため回数を直接コピーできないが、
現在の3.6M pairだけで「棋譜MMTOでは強くならない」と結論する規模ではない。
[Hoki & Kaneko, JAIR 2014](https://www.jair.org/index.php/jair/article/download/10871/25935)

一方、旧long-direct経路は同じ棋譜をepoch 128まで反復し、valid objectiveを改善しながら棋力を落とした。
従って必要なのは同じ10kの反復ではなく、**corrected MMTOで未使用のgameを広く一度処理すること**である。

## 3. 原因仮説の優先順位

### 3.1 第一仮説: corrected MMTOの学習量・game coverage不足

根拠はfull 1 outer比6.42%、459 / 34,582 gameしか見ていないこと、loss改善がauditへ汎化したこと、
候補の対局点推定が親より低下していないことである。まず既存data全量で検証する。

### 3.2 第二仮説: 棋譜手模倣と棋力のproxy mismatch

MMTOは「記録手が他の合法手より上になる」ことを学ぶ。1局面に通常1手しかなく、代替手の真の価値、
終局勝敗、on-policyで将来訪れる局面分布は直接教師にならない。固定auditの改善が対局へ直結しないことは、
この限界と整合する。

今回のauditを終局結果別に分けると、loss改善は勝者側手で`-0.02946`、敗者側手で`-0.03692`だった。
敗者側手にも強く適合している。これは直ちに誤学習を証明しないが、最終結果を持たないobjectiveにWDL anchorを
加える候補理由になる。将棋の方策勾配教師あり学習でも、棋譜手一致率最大の設定が最強とは限らず、qsearchを
含む探索条件が対局結果へ大きく影響したと報告されている。
[Konemura et al., GPW 2017](https://ipsj.ixsq.nii.ac.jp/record/183828/files/IPSJ-GPWS2017001.pdf)

またqnodes/rootは全体で+11.56%、openingで+20.99%、middlegameで+12.01%、endgameで+11.45%だった。
固定node対局で改善して1秒対局で改善しない場合は、評価値の順位だけでなく、評価変化により枝刈り効率が落ちた
可能性を優先する。Phase Aをfixed-nodeにするのはこの要因を分離するためでもある。

### 3.3 第三仮説: 教師品質と局面分布

現在のWDoor datasetは両対局者rate 4,000以上、決着局、最大24 root/gameに限定されており、低品質dataを
無差別に集めたものではない。ただし対局エンジンの探索時間・棋力は混在し、現在engineが実戦で到達する
局面分布とも一致しない。MMTO原論文でも強い棋譜から学んだ評価関数の方が強かったため、現時点でrate閾値を
下げて件数だけ増やすことはしない。

また現datasetは`--exclude-in-check`で王手中局面を除き、敗者側のply 80以降も除外している。fixed auditも
同じ分布なので、王手応手、劣勢防御、敗勢終盤の価値精度を測っていない。これは今回の10k結果を否定する
ものではないが、full MMTO後にも実戦転移しない場合は、単なる件数追加よりon-policy自己対局でこのcoverageを
補う理由になる。

### 3.4 architectureは第一原因ではない

v2.5.0比較weightも同じHalfKP-64 runtime architectureで動く。表現容量の絶対上限は将来検討対象だが、
現時点では教師、objective、学習量の差の方が強い説明になる。hidden拡大や追加layerは延期する。

### 3.5 致命的実装ミスの可能性は低下したが、棋力Gateは不可欠

zero-loss byte identity、provenance、material固定、finite値、違法手、qsearch bound、parameter delta、
fresh-search auditを通過している。doseを上げるほどloss・regret・棋譜手方向が単調に改善したため、符号反転や
更新不能の兆候はない。ただしobjective自体が棋力proxyなので、direct matchを外すことはできない。

## 4. 棋譜を増やすか、アルゴリズムを変えるか

### 4.1 直ちに新しい棋譜を増やさない

既存trainには570,875 root / 34,582 gameがあり、corrected MMTOはその1.75%のunique rootしか使っていない。
まず既存全量を使う。100k検証ではファイル先頭を切り出さず、game・phase横断のdeterministic sampleを作る。

full outerで親棋力も改善した場合に限り、2026年分を同じrate 4,000基準で追加し、game/content/network-inputを
deduplicateする。rate閾値を下げる、同じgameからroot数を増やす、同じ10kを反復する、の順では増やさない。

### 4.2 現MMTOをfull規模で1～2 outer、明確な改善時のみ最大4 outer試す

現在の結果だけではalgorithm変更を正当化できない。100kはscale safety、full outer 1は初めてv2.5.0の
50時間超学習とwall-timeの桁が揃う本試験である。outer 2はouter 1が明白に崩壊しない場合だけ行う。
outer 3 / 4は後述の棋力・paired safety・fixed audit Gateを毎outer通過する場合だけ行う。
outer 5以降は事前予約しない。

### 4.3 自己対局は視野に入れるが、純AlphaZeroは採用しない

AlphaZeroは自己対局だけで将棋を学べることを示したが、policy/value network、MCTS、大量のself-playと
専用計算資源を前提とする。本リポジトリのalpha-beta + static HalfKPとノートPC CPUにそのまま移植するのは
最短経路ではない。
[Silver et al., AlphaZero](https://arxiv.org/abs/1712.01815)

この構成に合うのは、自己対局で得たon-policy局面へ終局WDLと、より深いalpha-beta探索のPV leaf値を与える
TDLeaf / RootStrap型である。TDLeafはminimax searchとtemporal-difference学習を組み合わせた先行例で、
TreeStrapは探索木内の整合性を利用する。ただし最初から全探索木を保存せず、root/PVと終局結果だけのpilotから
始める。
[TDLeaf(λ)](https://arxiv.org/abs/cs/9901001)、
[TreeStrap](https://proceedings.neurips.cc/paper_files/paper/2009/file/389bc7bb1e1c2a5e7e147703232a88f6-Paper.pdf)

Stockfish NNUEもcp評価をWDL空間へ写し、探索評価とgame resultを混合できるlossを採用し、checkpointの
最終評価を対局で行う。数値設定は転用しないが、「dense search value + result anchor + match selection」という
構造は次段へ採用する。
[公式NNUE文書](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)、
[公式trainer](https://github.com/official-stockfish/nnue-pytorch)

## 5. 実行計画

### 長期無人実行の共通手順

Phase A→B→Cはsupervisorで自動段階実行する。計画書は実行中に書き換えず、起動後の正確な進捗と
terminal状態はruntimeの`status.json`を正本とする。設定・復帰・操作の詳細は
`docs/halfkp_unattended_training_20260806.md`に固定する。

- 完了した各outerでmodel、optimizer、checkpoint、metrics、run manifestのimmutable archiveを作る。
- archiveしたmodelに対し、固定6,099-game auditと同じ250 opening pair / 500局の親直接対局を完了してから次のouterを許可する。
- 進捗、heartbeat、artifact SHA、Gate判定、停止理由をatomicなstatus JSONへ保存し、中断時は最後にcommit済みのouterから再開する。
- Gate違反、provenance不一致、archive破損、不完全なaudit / 対局では自動停止し、次段に進まない。
- Phase D / Eは保存結果の人手再検討と別の実行承認が必要であり、supervisorは自動実行しない。
- この無人workflowはKPP由来重みとv2.5.0を一度もloadしない。

### Phase A: 現候補と親の直接診断対局

1. validからtrain/test/final reserveとexact SFENが重ならない250 opening familyをseed `20260731`で固定する。正本は`training/halfkp64-scratch-v1/mmto_dev_openings_seed20260731_250/openings.jsonl`（SHA-256 `06b6d7739a5973e391e6a275b8dd8be09842f0f061b598c5c30da40b578ba190`）と同directoryの`manifest.json`（SHA-256 `e9485ee47fc4bba73490399710f2e8724646fe1dfc01676c2826b0ed3a7bd22a`）である。
2. `3e-4 outer4`とaccepted generation 2を20,000 fixed nodes/手、先後交換500局で対局させる。
3. opening pair平均を単位に100,000回bootstrapし、score差の95%区間を保存する。
4. この対局はpromotionではなく、10k MMTOの方向診断とする。

判定:

- safety failure、またはscore 95%上限が45%未満: `3e-4`方向を停止する。
- それ以外: 10kでは学習量仮説を棄却できないためPhase Bへ進む。
- v2.5.0との追加対局は行わない。

概算: 1～3時間。

### Phase B: game-balanced 100k × 1 scale確認

既存10k×4とpair数だけを揃える約38.5k×1ならdata diversityを厳密に分離できる。しかし、それ自体では
full規模の学習量を判定できず小pilotを一段増やすため、今回は100k×1がdiversity確認とscale safetyを兼ねる。

1. accepted generation 2とzero-state AdaGradから開始する。
2. train全34,582 gameをhash順に巡回し、各gameからphase別に最大1 rootずつ選び、残りを別hash順で補充する。
3. seed `20260731`でexact 100,000 / 10,000 rootを作り、game数・phase数・SHAをmanifestへ固定する。正本は`runs/halfkp64-scratch-v1/unattended-mmto-20260806-v1/derived-balanced-100k/`下で、`train.jsonl` SHA-256は`79cc64ced54eb95418dc0f2d6b78cfb9ad2102f3ad0545e5015cf3ffb5d99458`、`valid.jsonl`は`84fbd8233f0e2baa3ee96d9aa152edcf224555bcd947ce0758e84327f2f7db0d`、`manifest.json`は`f1c388c35747599a83716b2de170c7828362281aecd25923ba5614072d482066`である。
4. feature LR `3e-4`、output比0.1、1 outerだけ実行する。
5. 全6,099 valid gameを横断する10,000-root aggregate valid、固定6,099-game audit、parameter/qnodes、
   安全性を評価する。
6. 同じ250 opening pairで親と500局対局する。

判定:

- `outer delta limit exceeded`または`parameter bound touched`: `3e-4`成果を採用せず、accepted generation 2 + zero-state AdaGradから`1e-4`を一度だけfallback実行する。これ以外のhard safety、provenance、I/O、未知の失敗はfallbackせず停止する。
- 親対局scoreが47.5%未満かつ95%上限も50%未満: 現MMTOをfullへ拡大しない。
- それ以外: Phase Cへ進む。100k modelはproductionへ継承しない。

概算: 学習8～9時間 + audit・対局。

### Phase C: 既存full dataset outer 1 / 2（明確な改善時のみouter 3 / 4）

1. accepted generation 2とzero-state AdaGradから、570,875 train root全量を開始する。
2. LRはPhase Bでsafeだった`3e-4`、fallback時だけ`1e-4`を使う。
3. outer 1でfull 90,759 valid、固定audit、immutable checkpoint、親500局を実行する。
4. 明白な親退行でなければouter 2へresumeし、同じ評価を繰り返す。

outer 1 → 2 Gate:

- score 95%上限が45%未満、またはpoint score 47.5%未満かつ上限50%未満: 停止。
- 47.5～52.5%で不明確かつoffline改善中: 学習量確認のためouter 2を一度だけ許可。
- 52.5%以上、または95%下限50%超: outer 2へ進む。

outer 2後:

- 両outerでofflineだけ改善し、親scoreが一度も50%を超えない: objective乖離を確定しPhase Dへ移る。
- outer 3自動許可: outer 2の親対局scoreが52.5%以上、outer 1以上、paired safety合格、かつfixed audit loss差の95% CI上限が0未満。
- outer 4自動許可: outer 3でもscore 52.5%以上、直前outer以上、paired safety合格、fixed audit loss差95% CI上限0未満の同じGateを再確認する。
- outer 3 / 4の各完了後にもimmutable archive、fixed audit、250 pair / 500局を作り、同じGate違反で即時停止する。outer 5以降は自動実行しない。

概算: 40～50時間/outer。

### Phase D: MMTO + root WDLの単一A/B

Phase Cでobjective乖離を確認した場合だけ実行する。
Phase A〜Cの無人supervisorはPhase Dを自動実行しない。

1. architecture、optimizer、dataset schema、MMTO pair lossは変えない。
2. 各rootの既存`winner`からside-to-move WDL targetを作り、root valueの補助lossを加える。
3. 10k固定subsetで初期WDL gradient normがMMTO gradient normの約10%になる係数を一度だけ決め、sweepしない。
4. Phase Bと同じ100k rootで`MMTO-only`をcontrol、`MMTO+WDL`をcandidateとして各1 outer比較する。
5. WDL Brier、MMTO fixed-audit、親500局を評価する。

MMTO lossを大きく悪化させず、WDLを改善し、親対局もcontrolより良い場合だけfullへ進む。失敗時にmix、LR、
seedを微調整しない。

### Phase E: scratch-only自己対局 + TDLeaf / RootStrap

Phase A〜Cの無人supervisorはPhase Eを自動実行しない。

導入triggerは次のどちらかである。

- Phase Dまで行っても親を改善しない。
- MMTO系で親をpromotionした後、追加WDoor学習がplateauする。

最小構成:

1. accepted scratch checkpoint群だけを対戦相手にする。KPP/v2.5.0は生成・教師・leagueへ入れない。
2. WDoorの局面はopening ballotにだけ利用し、記録手や既存scoreを教師にしない。
3. 先後反転、複数scratch checkpoint、固定node budgetで局面分布を広げる。
4. まず200 self-play gameを生成し、次に1,000、通過時だけ5,000へ増やす。
5. 学習targetは終局WDLを主とし、studentより大きい固定node budgetのfull-window alpha-beta + qsearch値を補助にする。
6. depth完了を必須にせず、固定nodeで最後に完了したiterative-deepening値を使う。bound値は捨てる。
7. 最初はroot/PVだけを保存する。失敗時だけexact internal nodeを限定保存するTreeStrapへ進む。
8. 各generationを親500局で判定し、offline WDLだけでpromotionしない。

1秒/手・平均120 plyなら1,000局生成は概算30～40時間で、CPU-onlyでも段階実行可能である。純粋な
result-only clone self-playは誤りを自己増幅しやすいため行わない。

### Phase F: 最終v2.5.0比較

親promotionを通過し、設定を凍結したmodelだけをv2.5.0と比較する。まず開発用400局、score 45%以上なら
追加評価し、最終claimは予約済み2,000 opening pair・4,000局で行う。v2.5.0の結果を見て学習率、WDL係数、
checkpointを選び直さない。

## 6. 停止条件と進捗表

| Phase | 状態 | 完了条件 |
|---|---|---|
| A 親直接500局 | 常設service実行対象（実状態はruntime参照） | paired reportと方向判定 |
| B balanced 100k × 1 | A通過後に自動実行 | scale safety、immutable archive、fixed audit、親500局 |
| C full outer 1～4 | B通過後に条件付き自動実行 | 各outerのarchive / audit / 250 pairとobjective・親棋力の転移判定 |
| D MMTO + WDL | 条件付き・無人実行対象外 | 単一100k A/B |
| E scratch self-play | 条件付き・無人実行対象外 | 200 → 1,000 → 5,000 game |
| F v2.5.0最終比較 | blocked by parent promotion | frozen 4,000-game report |

この計画で最も重要なのは、短いpilotのlossだけでもv2.5.0対局だけでも方針を決めないことである。次の
分岐は、既存full dataをcorrected MMTOで処理したときに、**accepted scratch親との直接棋力が動くか**で決める。
