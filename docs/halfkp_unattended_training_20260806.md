# HalfKP MMTO 長期無人学習運用設計

> 2026-08-08更新: このunitのrunはPhase C outer 1で安全停止した。旧runを変更せず、
> `docs/halfkp_full_mmto_recovery_20260808.md`の新config・新unitでPhase Cを再実行する。

- 制定日: 2026-08-06
- 対象lineage: `halfkp64-scratch-v1`
- 実行計画: `docs/halfkp_next_training_strategy_20260806.md`
- 機械可読設定: `training/halfkp64-scratch-v1/unattended_mmto_20260806.json`
- runtime: `runs/halfkp64-scratch-v1/unattended-mmto-20260806-v1/`

## 1. 目的と結論

休暇中のノートPCでPhase AからPhase Cまでを自動実行し、各milestoneを再現・再開可能な形で保存する。
単に長時間回し続けるのではなく、次を同時に満たすことを運用要件とする。

1. accepted generation 2から毎回開始し、KPP由来重みとv2.5.0を学習・教師・候補選択に使わない。
2. 学習outer、固定audit、親との先後交換対局を一組として保存する。
3. safety違反、破損、不完全評価、明確な親退行では次の学習を開始しない。
4. process停止後は、完了済みのouterとopening pairを再実行せずに再開する。
5. 最終評価用に予約した2026年の2,000 opening pairを途中評価へ流用しない。

長期化そのものは目的ではない。100kで規模安全性を確認し、既存570,875 rootをouterごとに初めて広く処理する。
full outer 3 / 4は、outer 2までに親棋力の明確な正方向が得られた場合だけ自動で許可する。

## 2. 自動実行する範囲

| Phase | 入力と処理 | 自動継続条件 | terminal条件 |
|---|---|---|---|
| A | 既存`3e-4 outer4`対accepted gen2、250 pair / 500局 | safety合格かつpaired 95%上限が45%以上 | safety違反または上限45%未満 |
| B | game/phase-balanced train 100k、valid 10k、1 outer | hard safety、fixed audit、親対局の非退行Gate | 明確な親退行または未知の失敗 |
| C1 | accepted gen2 + zero AdaGradからfull outer 1 | offline改善かつ親対局で明確に退行しない | safety、offline、親対局Gate違反 |
| C2 | 同じrunを`--resume --outer-iterations 2` | outer 2 scoreが52.5%以上、outer 1以上、offline/safety合格なら延長可 | objective乖離または退行 |
| C3/C4 | 上記を満たす場合だけouter 3、4を一つずつ実行・評価 | milestoneごとに同じGateを再確認 | どのGate違反でも即時終了 |

Phase DのWDL補助とPhase Eの自己対局は、結果を見ずに無人で新規実装・実行しない。Phase Cで停止した場合は、
保存済みの対局とauditから次の単一仮説を決める。候補をpromotion registryへ自動登録する処理も行わない。

### 2.1 balanced dataset

元の先頭100k rootは一部gameへ偏るため使わない。派生trainは全game×phaseの代表90,993件を先に採り、
残り9,007件を別domainのSHA-256順位で補う。派生validは全6,099 gameから最低1件を採り、未採用の
game×phase代表を加えて10,000件にする。元JSON lineは変更せず、選択と最終順序をseed付きhashで固定する。

派生manifestには元manifestのpath/SHA、seed、選択規則、root/game/phase数、train-valid overlap、出力SHAを保存する。
同じ設定で再生成した際にbyte一致しない場合は学習を開始しない。

### 2.2 開発用opening 250 pair

2023--2025 frozen validからply 8--40、unique exact SFEN、unique opening familyを満たす250件を固定する。
元train、test、2026 final reserve 2,000件とのexact SFEN overlapを0にする。選択manifestを対局設定とともに固定し、
Phase A/B/Cの比較では同じ順序を使う。

## 3. 保存と再開の契約

### 3.1 trainer checkpoint

`halfkp_mmto_train`の`--outer-iterations N`は追加回数ではなく、累計outerの到達値である。supervisorは必ず
`N=1`、評価、`N=2 --resume`、評価、のように一段ずつ呼ぶ。

trainerは次を`checkpoint.json`へ記録する。

- completed outer
- init model SHA-256
- dataset manifest SHA-256
- model / optimizer SHA-256
- 一つ前のmodel SHA-256とmaterial bits

model・optimizer・metrics・run manifestをfsyncした後、`checkpoint.json`を最後のcommit markerとして更新する。
outer途中またはcommit直前で停止した場合、次回は`.previous`から直前の完了outerを復元し、そのouterだけを
最初から再実行する。gradient trajectoryを変えないため、outer途中のblock単位checkpointは導入しない。
このためfull outer中の障害では最大約40--60時間を再実行し得るが、完了済みouterは失わない。

### 3.2 immutable outer archive

各outerのcheckpoint commit確認後、次を一時directoryへcopyし、全fileのbytesとSHA-256を列挙したmanifestを
最後に書いてdirectory renameする。

- `model.bin`
- `optimizer.bin`
- `checkpoint.json`
- 対応する`metrics/outer_NNNN.json`
- `run_manifest.json`
- 対局用candidate manifest

既存archiveは上書きしない。同名archiveがある場合は全hashを再検証し、一つでも違えば`blocked`で停止する。
auditと対局は必ずこのarchive内modelを対象にするため、次outerの更新から隔離される。

### 3.3 auditと対局

固定6,099-game auditは、JSONL・Rust sidecar・比較結果のbytes/SHAを列挙した`audit_completed.json`だけを
commit markerとする。markerがなければ部分auditを再利用せず全件再生成する。対局はopening pairごとに2局を実行し、
pair JSONをatomic保存する。再起動時はcandidate/parent/opening/configのfingerprintを検証し、未完pairだけを再開する。

250 pair完了後、各pairの2局scoreをclusterとしてseed固定100,000回percentile bootstrapし、95%下限・上限を
保存する。binary失敗、JSON欠落、SHA不一致、局数不足、NaNは「不採用」ではなく`gate_error`であり、次段へ進まない。

## 4. supervisorの状態機械

runtimeの`status.json`が進捗の正本である。少なくとも次を記録する。

- config SHA-256、Git revision、使用binary SHA-256
- `running` / `completed` / `stopped_by_gate` / `blocked` / `stopped_by_user`
- 現在のPhase、step、outer、child PID、開始・更新時刻
- 完了step、再試行数、直近error、停止理由
- dataset、archive、audit、match reportのpathとSHA-256

長いchild processの実行中もheartbeatを更新する。同一runtimeに対する二重起動は`fcntl.flock`で拒否する。
同一stepの一時的失敗は最大3回だけ再試行し、同じ条件で繰り返す場合は無限loopせず`blocked`で正常停止する。
SIGTERMではchild process groupを終了し、既存checkpointを変更せずに終了する。

supervisorのconfig fingerprintは初回実行時に固定する。learning rate、tau、batch、jobs、root上限、delta上限、
binary、Git revision、dataset/親model/Gate reportのSHAが変わったresumeは拒否する。

## 5. 操作

通常操作は管理scriptだけを使う。

```bash
scripts/manage_halfkp_mmto_unattended.sh validate
scripts/manage_halfkp_mmto_unattended.sh start
scripts/manage_halfkp_mmto_unattended.sh status
scripts/manage_halfkp_mmto_unattended.sh logs
scripts/manage_halfkp_mmto_unattended.sh stop
scripts/manage_halfkp_mmto_unattended.sh resume
```

`start`はGit管理したuser systemd常設serviceをenableして起動し、terminalから分離する。user lingerも起動条件とし、
logout・OS再起動後にも同じunitを自動起動する。serviceは
`Restart=on-failure`、`KillMode=mixed`、低いscheduler優先度で動かし、`systemd-inhibit`でsleep、idle、
lid-switchをblockする。supervisorがGateにより停止した場合は成功終了にするため、systemdが同じ失敗を再起動しない。
明示的な`stop`は停止markerを書いたうえでunitをdisableし、次回bootでの意図しない再開も防ぐ。`resume`はmarkerと
retry状態を明示的に解除し、unitを再enableする。OS shutdown由来のSIGTERMではdisableしないため、boot後に自動再開する。

### 5.1 電源とloginに関する制約

- 起動前にAC online、十分な空き容量、親model/dataset SHA、必要commandを検証する。
- AC電源を接続し、吸排気口を塞がない。CPU温度とclock制御はOSのthermal controlへ委ねる。
- 起動前に`loginctl enable-linger $USER`を一度実行し、`Linger=yes`を確認する。常設user unitはlogout後も動作し、
  OS再起動後は最後のcommit済みmilestoneから自動再開する。
- 停電・kernel panic・強制電源断では、最後にcommit済みのouter/pairまで復旧する。filesystem自体の故障は対象外である。

## 6. runtime layout

```text
runs/halfkp64-scratch-v1/unattended-mmto-20260806-v1/
  status.json
  logs/
  derived-balanced-100k/
  locked_binaries/
  evaluations/
    phase-a-direct/match/
    phase-b-outer-0001/{audit.jsonl,comparison.json,match/}
  phase-b-mmto100k-lr3e-4/
  phase-b-mmto100k-lr1e-4/   # safety fallback時だけ
  phase-c-full/
  archives/
    phase-b/outer_0001/
    phase-c/outer_0001/
    phase-c/outer_0002/
  evaluations/
```

runtimeは`.gitignore`対象であり、数十MBのmodelや大量のpair reportをGitへ入れない。設定、実装、運用文書だけを
Gitで管理し、休暇後に採否と要約reportを`training/halfkp64-scratch-v1/`へ記録する。

## 7. 検証項目

無人起動前に次を確認する。

1. Rust MMTO unit testとrelease binary build。
2. selectorのdeterminism、game/phase coverage、exact count、overlap 0、tamper検出。
3. opening-pair集約、bootstrap境界、provenance不一致拒否、partial resume。
4. supervisorのconfig固定、archive hash検証、Gate境界、二重起動拒否。
5. 小さなfixtureまたはdry-runで全command、status遷移、stop/resumeを確認する。
6. service起動後にsystemd active、sleep inhibitor、heartbeat、child process、AC onlineを確認する。

## 8. 休暇後の判断

`status.json`のterminal理由から次を一つだけ選ぶ。

- clear strength trendでouter 4まで完了: 最良milestoneを追加開発対局へ進める。
- offlineと親棋力がともに改善: accepted parentへのpromotion計画を別commitで作る。
- offlineだけ改善し親scoreが動かない: MMTO objective乖離としてPhase Dの単一WDL A/Bへ進む。
- 親より明確に悪化: 同じMMTOのouter追加やLR sweepを行わず、自己対局TDLeaf / RootStrap計画を再評価する。
- `blocked`: artifactを修復せず保存し、原因を再現してから同じcheckpointをresumeする。

どの場合も、途中のv2.5.0対局結果からcheckpointやhyperparameterを選び直さない。
