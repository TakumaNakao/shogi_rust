# HalfKP full MMTO 再実行計画

- 決定日: 2026-08-08
- lineage: `halfkp64-scratch-v1`
- 対象run: `runs/halfkp64-scratch-v1/unattended-mmto-full-lr1e-4-20260808-v1/`
- 機械可読な実行正本: `training/halfkp64-scratch-v1/full_mmto_recovery_plan_20260808.json`
- 実行設定: `training/halfkp64-scratch-v1/unattended_mmto_recovery_20260808.json`
- 旧run: `runs/halfkp64-scratch-v1/unattended-mmto-20260806-v1/`（読取専用で保存）

## 1. 結論

次の重み更新は、accepted scratch generation 2とzero-state AdaGradから、WDoor高レート棋譜の
570,875 train root全量を使うcorrected MMTO Phase Cをfeature LR `1e-4`で再実行する。
outer 1と2を順に学習・評価し、明確な改善が続く場合だけouter 3と4を許可する。outer 5は実行しない。

今回変更する実験変数はPhase Cの学習率だけである。objective、dataset、network architecture、optimizer、
output LR比、探索条件、評価集合、安全上限は変更しない。旧Phase A/Bを再計算せず、完了済みartifactのSHAと
意味をsupervisorが検証してからPhase Cだけを新しいrun rootで開始する。

KPP由来重みとv2.5.0の重みは、初期値、optimizer、教師、target、棋譜生成、対局相手、checkpoint選択、
hyperparameter選択のいずれにも使用しない。v2.5.0はscratch候補を確定した後の最終比較にだけ使用できる。

## 2. 旧runの結果と停止原因

### 2.1 Phase A/Bは成功している

| 段階 | 結果 | 判断 |
|---|---:|---|
| Phase A 10k×4候補 vs 親 | score 0.535、paired 95% CI [0.491, 0.579] | catastrophic regressionなし |
| Phase B balanced 100k×1 | 281勝209敗10分、score 0.572 | 親より改善 |
| Phase B paired CI | [0.531, 0.612] | 下限が0.5を超える |
| Phase B fixed audit loss差 | -0.037957、95%上限 -0.035810 | 強いoffline改善 |
| Phase B feature / hidden / output delta | 0.014717 / 0.010908 / 0.001382 | 安全上限内 |

Phase Bは親に対する500局の直接対局と、独立した6,099-game fixed auditの両方で改善した。これは
corrected MMTOの更新方向が有効であるという初めての統計的に明確な証拠であり、MMTO方式やWDoor datasetを
この時点で捨てる理由はない。Phase B modelは診断上のincumbentとしてimmutable archiveに保存するが、
subset選択バイアスを避けるためfull学習の初期値にはしない。

### 2.2 Phase Cの停止はdose scalingの誤りである

旧Phase CはPhase Bで安全だったLR `3e-4`を、100,000 rootから570,875 rootへそのまま拡大した。
outer 1の最後に次のdelta safety違反で停止した。

| parameter group | 観測delta | 上限 | 上限比 |
|---|---:|---:|---:|
| feature | 0.0330639 | 0.020 | 165.3% |
| hidden | 0.0198727 | 0.020 | 99.4% |
| output | 0.0027042 | 0.004 | 67.6% |

fullの学習pairは約56,035,622、Phase Bは10,375,690で、dose比は約5.40である。AdaGradの累積更新を
`LR × sqrt(update数)`で近似すると、Phase Bと同程度の変位にするLRは
`3e-4 / sqrt(5.40) ≈ 1.29e-4`となる。旧fullの実測deltaをLRへ線形換算すると`1e-4`で予測されるdeltaは
feature 0.01102、hidden 0.00662、output 0.000901で、上限に十分な余裕がある。

`1.5e-4`ならfeature delta予測は0.01653で上限内だが、余裕は約17%しかなく、新しい未検証の中間率を増やす。
`1e-4`は旧計画ですでに定義・小規模確認された率であり、平方根換算値の77%のdoseを持つ。安全性と学習量の
両方を満たす最短の一択として`1e-4`を採用する。

これはtrainerが安全に停止した結果である。旧Phase Cのcheckpointはouter 0、model SHAはscratch親と同一、
optimizerはzero-stateのままで、違反更新はcommitされていない。したがって旧Phase Cをresumeしたり、partialを
救済したり、旧runの設定を変更したりしない。

## 3. 新runの固定条件

### 3.1 初期状態とdata

- init: accepted scratch generation 2、SHA-256
  `9a1bfc9b7b317f451fa9a6efb350fed0a50e3020659686cefccc98646b9b3db6`
- optimizer: zero-state AdaGrad。Phase B optimizerや旧Phase C directoryを継承しない
- train: WDoor両対局者rate 4,000以上、570,875 root / 34,582 game
- valid: 固定90,759 root
- feature LR: `1e-4`
- output LR ratio: `0.1`
- gradient clip norm: `1.0`
- parameter delta上限: feature/hidden `0.02`、output `0.004`
- architectureと探索・MMTO pair生成条件: 変更なし

### 3.2 prior evidence gate

Phase A/Bを省略する代わりに、新supervisorは起動時と学習直前に旧runの次のartifactをSHA固定で再検証する。

- blocked `status.json`
- Phase B archive manifestとmodel
- Phase B fixed-audit completed marker
- Phase B 500局paired-match report
- Phase C failure log
- outer 0 checkpointとtrain lock

Phase Bのmodel SHA、audit強改善、paired CI下限>0.5、旧Phase CのLR `3e-4`、delta違反、outer 0への
rollback、親modelとzero optimizerの一致まで検査する。旧artifactは実験を省略する根拠にだけ使い、trainerの
`--init`、optimizer、教師、targetには渡さない。1 byteでも変化した場合は新runを開始しない。

## 4. 自動実行とGate

各outerは「学習 → immutable archive → fixed 6,099-game audit → 親との250 opening pair / 500局」の順に
完了し、次へ進む。model、optimizer、checkpoint、metrics、run manifest、audit、各対局pairを途中保存する。
OS再起動やclean stop後は最後に検証済みのcommit pointから再開する。

### outer 1から2

outer 1で次をすべて満たせばouter 2を実行する。

- non-finite、bound rejection、provenance不一致、material変化、parameter bound、delta上限違反がない
- fixed auditのloss差が負
- paired match safety合格
- score < 0.475かつpaired CI上限 < 0.5という明白な親退行でない
- catastrophic CI上限 < 0.45でない

`1e-4`でもouter 1がdeltaまたはparameter boundで停止した場合、自動fallback、追加LR sweep、上限緩和は
行わない。dose近似またはtrainer挙動の前提が外れた証拠として人手調査へ戻す。

### outer 2から3、outer 3から4

outer 3/4は、直前outerが上記共通Gateに加えて次をすべて満たす場合だけ実行する。

- 親対局scoreが0.525以上
- scoreがさらに一つ前のouter以上
- fixed audit loss差の95% CI上限が0未満
- paired match safety合格

outer 4完了後にも同じ条件をouter 3との比較で判定する。満たさないouter 4は保存するが、最終候補適格とは
扱わない。これは旧supervisorで欠けていた最終outerの品質判定を補うものであり、追加outerを許可しない。

### hard stopとwarning

hard stopはprovenance/SHA不一致、non-finite、違法手、search failure、bound rejection、material変化、
parameter bound、delta超過、archive不整合、固定audit不成立、統計的に明白な親退行である。qnodes増加、
PV変化率、scoreの不明確な揺れは記録するが、単独ではhard stopにしない。

## 5. 完了後の重み選択と次段

このrunは自動promotionしない。完了後は保存された全outerとPhase B incumbentを比較し、次の順で決める。

1. hard safetyをすべて満たす。
2. fixed auditが親より改善する。
3. 親との直接対局で最も強い候補を選ぶ。
4. 最良full候補とPhase B incumbentを、独立した先後交換対局で直接比較する。
5. scratch候補を固定した後だけv2.5.0と最終比較する。

full outer 1/2が親を改善しない、またはofflineだけ改善して対局scoreが動かない場合は、単なる棋譜量不足を
第一原因とする段階を終了する。同じMMTOのLR/seed/outerを細かくsweepせず、既存計画のPhase Dで終局WDLを
補助targetにした単一100k A/Bを行う。それでも改善しなければ、scratch checkpointだけを使う自己対局と
TDLeaf / RootStrap型search bootstrapへ移る。純AlphaZero型の新architectureはCPU-onlyの最短経路ではないため
導入しない。

## 6. 運用

```bash
scripts/manage_halfkp_mmto_unattended.sh validate
scripts/manage_halfkp_mmto_unattended.sh start
scripts/manage_halfkp_mmto_unattended.sh status
scripts/manage_halfkp_mmto_unattended.sh logs
scripts/manage_halfkp_mmto_unattended.sh stop
scripts/manage_halfkp_mmto_unattended.sh resume
```

新unitは`halfkp-mmto-full-recovery-20260808`である。旧unit
`halfkp-mmto-unattended-20260806`は起動時にmanagerがinactiveを確認してdisableする。旧unitがactiveなら
新unitを起動せず停止する。旧runは削除・移動・上書きしない。user lingeringと
systemd inhibitorによりlogout、idle、lid closeによる停止を防ぐ。ただし電源断にはcheckpoint単位の再開で対応する。

## 7. 進捗表

runtimeの`status.json`を実行状態の正本とし、この表は計画・実装の履歴として更新する。

| 項目 | 状態 | 記録 |
|---|---|---|
| 旧run failure / rollback監査 | 完了 | Phase C outer 0、親model・zero optimizerを確認 |
| Phase B有効性再評価 | 完了 | score 0.572、CI [0.531, 0.612]、audit強改善 |
| LR再設計 | 完了 | full Phase Cを`1e-4`一択に固定 |
| recovery evidence gate実装 | 完了 | SHAとsemanticの二重検証 |
| outer 4最終品質Gate修正 | 完了 | extended outerと同じ判定を追加 |
| 新config / service / tests | 完了 | 新run rootに分離、全自動テスト合格 |
| new run preflight / 起動 | runtime参照 | clean Git commit後に実行し、以後は`status.json`を正本とする |
| full outer 1 | 未実行 | 学習・audit・親500局 |
| full outer 2 | 未実行 | outer 1 Gate通過時に自動実行 |
| full outer 3 / 4 | 条件付き | score 0.525、非低下、audit strong時のみ |
| candidate選択 / v2.5.0最終比較 | 未実行 | unattended run完了後に別承認 |
