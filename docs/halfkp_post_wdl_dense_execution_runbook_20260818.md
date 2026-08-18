# HalfKP scratch post-WDL dense 実装・実験手順書

- 制定日: 2026-08-18
- 状態: 実装開始前の正本
- 対象lineage: `halfkp64-scratch-v1`
- 対象branch: `training/halfkp-scratch-v1`
- 実験run root: `runs/halfkp64-scratch-v1/post-wdl-dense-20260818-v1`
- 上位計画: [`halfkp_next_training_plan_post_wdl_ab_20260818.md`](halfkp_next_training_plan_post_wdl_ab_20260818.md)

## 0. この文書の役割

上位計画は「なぜWDLを一度だけ正しく再試験し、dense MMTOへ進むか」を定義する。本書はそれを、実装者が
途中で設計判断やparameter探索をせず、上から順に実装・検証・実験できる作業単位へ落とした手順書である。

文書間で差がある場合の優先順位は次のとおりとする。

1. 本書の実装契約、CLI、Gate、再開契約
2. 上位計画の目的、不変条件、停止条件
3. 既存の過去run手順書

本書に記載する新しいCLIやschemaは、現時点では未実装である。各実装Stepの完了条件を満たす前に、そのCLIを
本番dataへ実行してはならない。Phase A～Cの長期run開始後は、固定したGit revisionのtracked fileを変更しない。

本書の自動実行範囲は、corrected WDL A/B、dense dataset、250k pilot、最大2 outerのdense MMTO、最良scratch
候補の凍結までである。v2.5.0比較は候補凍結後の別評価であり、Phase A～Cのtrainer、teacher、selectionへ
結果を戻さない。Phase Dのdeep-search bootstrapはtriggerだけを本書で固定し、A～Cの結果が出る前に実装しない。

## 1. 完了時の状態

本書を完了したとき、次が成立していなければならない。

- 旧WDL実装が希釈されていた原因を修正し、`WDL off`の既存MMTO経路は旧binaryとbyte-identicalである。
- WDL係数は固定10,000 rootで一度だけ数式校正され、候補結果を見て変更されない。
- 現行valid 90,759 rootとtest 62,233 rootはbyte-for-byteで凍結され、trainとの衝突が0である。
- dense trainは180万～300万unique root、新規network inputは120万以上である。
- trainのbase/new順序はgame単位の決定論的interleaveであり、baseを先頭へ固めない。
- pilotとfullはouter 4のmodelだけでなくoptimizerも継承する。
- 50,000 rootごとのcheckpointから、model、optimizer、metrics、samplerをbit-identicalに再開できる。
- 各候補はimmutable archiveを作ってからfresh audit、対局、Gate判定の順に評価される。
- Gate不合格、入力改変、停電、明示停止の各場合に、次に何をするかが一意に決まる。
- 最終的に`best-scratch.json`が、v2.5.0を参照せず一意に選ばれたscratch候補を指す。

## 2. 絶対に変えない条件

- v2.5.0/KPP由来重みをinit、optimizer、teacher、target、anchor、hard-negative選択、自己対局相手へ使わない。
- v2.5.0は`best-scratch.json`確定後の比較相手にだけ使う。
- 2026年WDoorをtrain、valid、WDL pool、kappa/norm校正へ使わない。
- 現行HalfKP-64 architecture、material係数、runtime codecを変更しない。
- dataset seed `20260821`、WDL seed `20260822`、audit split seed `20260823`、pilot seed `20260824`を
  結果に応じて変えない。
- LR、WDL係数、sampling比、kappa、seedのgrid searchを行わない。
- WDL A/Bは一度だけ行い、不合格なら以後MMTO-onlyに固定する。
- dense fullは最大2 outerとし、不採用outerの先へ進まない。
- live modelを評価せず、必ずSHA固定済みarchiveを評価する。
- global promotion registryはworkflow中に更新しない。workflow内の親はimmutable decisionで認可する。

## 3. 開始時の正本artifact

pathはrepository rootからの相対pathである。実行configにはpath、SHA-256、bytesをすべて記録する。

| 用途 | path / 件数 | SHA-256 |
|---|---|---|
| outer 4 model | `runs/halfkp64-scratch-v1/unattended-mmto-full-lr1e-4-20260808-v1/archives/phase-c/outer_0004/model.bin` | `a40af8e3ef913dc78e8272d3e6170d7a801453fa7dadd9b954bb7c945cbb431b` |
| outer 4 optimizer | 同directoryの`optimizer.bin` | `99a7fb2356d94378712fb49c67d208bb3a85c6040f735e4f0fba00c7befd7fd3` |
| outer 4 checkpoint | 同directoryの`checkpoint.json` | `806b1b2dd751c8dba727c0a6c49ac8c55dd2bb47cfaa06b1d43b7fa86bfec0f7` |
| outer 4 model manifest | 同directoryの`model.bin.manifest.json` | `b64f4a0e53a91ca4f207914bad3e9fccf1ed9b45537d55018c9cbcc3e7621b6b` |
| outer 4 archive manifest | 同directoryの`archive_manifest.json` | `c29076a01cbff837bdf41655edc7c1b4ffc27c5c23166a80c13cbfaa3d51fe72` |
| generation 3 Gate | `training/halfkp64-scratch-v1/outer4_internal_gate_20260817.json` | `a15fde711a17292f409f8bd988b6fc2c41deede1afae23f1b877135af146ce3b` |
| promotion registry | `training/halfkp64-scratch-v1/promotion_registry.json` | `129297e6505cde9ba2486100149b9c7d11ad67b778823ca4a609c8f6a35dbe82` |
| base dataset manifest | `runs/halfkp64-scratch-v1/wdoor-highrate4000-1m-v2/manifest.json` | `d8da8088b02d3d0200aad2ecb061691c6c418173a16fe32819f03088af48708f` |
| base train 570,875 | 同directoryの`train.jsonl` | `c35673d69c9306f1dd4b1f2f32714adce4a43e7c1a086141bd6b7246b680eee6` |
| frozen valid 90,759 | 同directoryの`valid.jsonl` | `3ce8745dbeb094a6a3f357494eb94aa5f57a716437bda66dd0ca13060f075f96` |
| frozen test 62,233 | 同directoryの`test.jsonl` | `85c73cdf519deff2a51bc2a92e0a746047cb1deae2187f3a1445dd2411b49893` |
| balanced MMTO train 100k | `runs/halfkp64-scratch-v1/unattended-mmto-20260806-v1/derived-balanced-100k/train.jsonl` | `79cc64ced54eb95418dc0f2d6b78cfb9ad2102f3ad0545e5015cf3ffb5d99458` |
| balanced MMTO valid | 同directoryの`valid.jsonl` | `84fbd8233f0e2baa3ee96d9aa152edcf224555bcd947ce0758e84327f2f7db0d` |
| balanced manifest | 同directoryの`manifest.json` | `f1c388c35747599a83716b2de170c7828362281aecd25923ba5614072d482066` |
| one-root/game audit 6,099 | `training/halfkp64-scratch-v1/mmto_audit_valid_20260802.jsonl` | `e0a710c116c5507df566a9716f28015efd1511d51016ebdc7a2b780e5a312cbe` |
| one-root/game manifest | 同directoryの`mmto_audit_valid_20260802.manifest.json` | `43dc73b284180290410054ff9bc49f9b58983635ea9f4822714553b518c85900` |
| fixed-node openings 250 | `training/halfkp64-scratch-v1/mmto_dev_openings_seed20260731_250/openings.jsonl` | `06b6d7739a5973e391e6a275b8dd8be09842f0f061b598c5c30da40b578ba190` |
| fixed-node openings manifest | 同directoryの`manifest.json` | `e9485ee47fc4bba73490399710f2e8724646fe1dfc01676c2826b0ed3a7bd22a` |
| 2026 reserve 2,000 | `training/halfkp64-scratch-v1/final_openings_seed20260730_2000.jsonl` | `385b7f70d63ddfb17ea9b39e53d316d7dd1b6d5946e6b4c28a74230e262d2329` |
| 2026 reserve manifest | 同directoryの`final_openings_seed20260730_2000.manifest.json` | `c5745163db94efa36cdcddc25b5d13eec83feb25446d1b604db33783b9653d71` |
| taya36 | `taya36.sfen` | `22950f42a2d65292fa689a8c84e3af7dc160c571e9aa559699b0beaf1bc9adce` |

旧WDL controlの正本は次である。新実装の10k replayが一致した場合だけ再利用する。

| artifact | SHA-256 |
|---|---|
| `runs/halfkp64-scratch-v1/wdl_ab_control_100k_20260817/model.bin` | `fb87fa7627aa6d8f00be3e637a6ef1d18822e1d31f3898ff064ac26365b65359` |
| 同directoryの`optimizer.bin` | `6e9db90e22e0accad03f1d473af978696829456057499576ddbb50083a393805` |
| 同directoryの`checkpoint.json` | `57ea53ceb08fcf97681dae6e74870b3ced5bb6e4315dd36c436c9f53442a2bba` |
| 同directoryの`run_manifest.json` | `da72eea1e145d41950cf43acb43a3604f744246c99d6a813fe56eb96921af91e` |

現在の`target/release/halfkp_mmto_train`は、旧A/Bで使った可能性が極めて高いbinaryである。新しいrelease
buildより前に必ず保全する。

- SHA-256: `329e32f5b79e6ce2b1dc56a14e1711da4b6c4d88157dc1d472a0f3a86129439f`
- bytes: `2,041,304`
- mtime: 2026-08-17 10:29:07

2018～2026 archiveの正本SHAは次である。2026は存在確認と最終reserveだけに使い、A～Cのsource listへ入れない。

| 年 | archive SHA-256 |
|---:|---|
| 2018 | `d5bc1e7996ec3e9aa3895e665a2bea588683002a1dd7a5fd3be1598d85eb3867` |
| 2019 | `7e9d10b2e32f27eec57bbde0983a3eea5c847052a70b0dce58349cca6780624b` |
| 2020 | `50ca62725bf7ad96349705c33459929f8602d22d47bcc115d6c3e2dc93a5e3dd` |
| 2021 | `96182a29d30c4f851f20fbcd68a92afc06afa6b991092e78a6a39d5b14abaa60` |
| 2022 | `e0c367f2a5224eed8a70e6b82ca87aaa16cf578ba2c9478c493b78828068cb60` |
| 2023 | `d10ba4a2b9d67f751a66ea92cb23f24eeafd69e744dab1e62da3d8729dcfd5b9` |
| 2024 | `4847d98cb297283f8c53e135107afcc083b90cd8933cb78fa8d1120e37f4e3e9` |
| 2025 | `423504903f211316ec40f9c3d8dcc2e5faf392fc4d1334ede1489242a9d09baa` |
| 2026 | `fa224cffe32205de96e9ff907b2ff4203dd965e16e6294329d4333fd2dbe27dd` |

## 4. 新runのartifact layout

```text
runs/halfkp64-scratch-v1/post-wdl-dense-20260818-v1/
├── status.json
├── run_identity.json
├── STOP_REQUESTED.json
├── locked_binaries/
├── logs/
├── derived/
│   ├── wdl-raw/
│   ├── wdl-pool/
│   │   ├── train.jsonl
│   │   ├── manifest.json
│   │   └── completed.json
│   ├── wdl-valid/
│   │   ├── calibration.jsonl
│   │   ├── audit.jsonl
│   │   ├── manifest.json
│   │   └── completed.json
│   ├── dense-recent-membership/
│   ├── dense-recent/
│   │   ├── composition_report.json
│   │   └── completed.json
│   ├── dense-old-membership/
│   ├── dense-final/
│   │   ├── train.jsonl
│   │   ├── valid.jsonl
│   │   ├── test.jsonl
│   │   ├── manifest.json
│   │   ├── dataset_audit.json
│   │   └── completed.json
│   └── unseen-250k/
├── phase-a/
│   ├── kappa.json
│   ├── norm-calibration.json
│   ├── replay-old/
│   ├── replay-new/
│   └── treatment-live/
├── phase-b/pilot-live/
├── phase-c/full-live/
├── archives/
│   ├── phase-a/control/outer_0001/
│   ├── phase-a/treatment/outer_0001/
│   ├── phase-b/outer_0001/
│   └── phase-c/outer_0001|outer_0002/
├── evaluations/
│   └── <candidate-label>/
│       ├── mmto-audit/
│       ├── wdl-audit/
│       ├── fixed-match/
│       └── time-match/
└── decisions/
    ├── phase-a-objective.json
    ├── phase-b-dataset.json
    ├── phase-b-pilot.json
    ├── phase-c-outer-0001.json
    ├── phase-c-outer-0002.json
    ├── accepted-parent-outer-0001.json
    └── best-scratch.json
```

`completed.json`とdecisionはatomic createし、既存fileを上書きしない。最低限、schema、config SHA、command
fingerprint、入力artifact SHA、出力artifactのpath/SHA/bytes/件数、作成時刻を持つ。再起動時は`status.json`
ではなく、immutable markerを先頭から検証してstateを再構成する。`status.json`は表示用cacheに限定する。

各Gate decisionのexact共通fieldは次とする。

```text
schema
config_sha256
created_at
candidate = {model_sha256, archive_manifest_sha256}
parent = {model_sha256, authorization_or_registry_sha256}
input_artifacts[]
evaluation_artifacts[]
gate_values
gate_results
passed
reasons[]
next_state
```

candidateを持たないdataset decisionではcandidate/parentをnullにし、canonical dataset artifactを追加する。
値だけを複写せず、元reportのSHAを必ず記録する。

decision固有fieldは最小限、Phase Aでは`objective`とWDL校正artifact（MMTO-only時は`null`）、datasetでは
`canonical_dataset`、pilot/outerでは`selected_best`、LR決定では値とf32 bits、`best-scratch.json`では
選択archive/modelと選択理由を持つ。これにより停止経路でも次工程の入力を手作業で補わない。

## 5. 実装対象と責務

### 5.1 `halfkp_mmto_train`

対象:

- `src/training_tools/halfkp_mmto_train.rs`
- 必要な共通処理を置く既存training module
- 対応unit/integration test

追加するCLI契約:

```text
--objective mmto-only|hybrid
--wdl-train PATH
--wdl-pool-manifest PATH
--expected-wdl-pool-manifest-sha256 SHA256
--wdl-kappa-manifest PATH
--expected-wdl-kappa-manifest-sha256 SHA256
--wdl-objective-manifest PATH
--expected-wdl-objective-manifest-sha256 SHA256
--wdl-sampler-seed U64
--calibrate-wdl-objective-report PATH
--wdl-calibration-roots N
--wdl-target-update-ratio FLOAT
--checkpoint-roots N
--expected-binary-sha256 SHA256
--expected-git-revision REV
```

`--wdl-weight`と`--wdl-kappa-cp`は旧CLI互換のため残す。新runでは`wdl-weight=0`だけを許可する。
corrected hybridは`wdl-kappa-cp`の既定600やcommand line値を一切使わず、objective manifest内のfit済みkappa
value/f32 bitsだけを使う。

validation modeは次の3つを排他的にする。

| mode | 必須 | 禁止 |
|---|---|---|
| MMTO-only training | `objective=mmto-only` | 全corrected WDL引数 |
| hybrid calibration | pool、kappa manifest、init optimizer、calibration report path | objective manifest、model更新、resume |
| hybrid training | pool、objective manifest、init optimizer、`objective=hybrid` | kappa再fit、calibration report path |

hybrid calibrationは`outer-iterations`に依存しない単一passとし、
`wdl-calibration-roots == max-train-roots == 10000`を要求する。model、optimizer、checkpointを生成しない。
MMTO-onlyは既存`Gradient`/`apply_batch`経路をそのまま使う。

- `checkpoint-roots=0`は旧outer単位checkpointであり、Phase A replay/treatmentだけに使う。
- pilot/fullでは`checkpoint-roots=50000`以外をconfig validationで拒否する。
- `outer-iterations`は到達するouter番号とする。これだけはtrajectory fingerprintから除外し、outer 1のGate後に
  同一runを`--outer-iterations 2 --resume`できるようにする。

内部gradientを次へ分離する。

```text
ParameterGradient
├── hidden_b
├── out_w / out_b
└── sparse

MmtoBatchGradient
├── ParameterGradient
├── pairs / loss_sum
└── delta histogram / active pairs

WdlBatchGradient
├── ParameterGradient
├── roots / weight_sum
└── weighted Brier sum
```

hybridの更新順序は固定する。

1. MMTOとWDLを別のgradientへ蓄積する。
2. sparseの左右対称projectionを別々に行う。
3. MMTOをpair数、WDLをweight合計で別々に平均する。
4. 両方のmaterial gradientを0にする。
5. `combined = mmto + coefficient * wdl`とする。
6. combinedへ既存global clipを一度だけ適用する。
7. 既存AdaGradで更新する。

既存8,192 pair境界は変えない。現dataset契約の`min_legal_moves >= 2`とexact qsearchにより、受理MMTO rootは
必ず1 pair以上を生成することをassertする。違反時はWDLを別batchへ移さずfail closedとする。rootごとにWDL
samplerを1回進め、そのWDL gradientを「そのMMTO rootの最初のpairが入るbatch」へ一度だけ加える。これを
`root_first_pair_batch_v1`としてmanifestへ記録する。WDL off経路は旧batch適用順を完全に維持する。

WDL samplerは`sha256_game_cycle_v1`とし、byte列まで固定する。

```text
game_digest =
  SHA256("halfkp-wdl-game-v1\0" || seed_u64_le || game_key_utf8)

root_digest =
  SHA256("halfkp-wdl-root-v1\0" || seed_u64_le || cycle_u64_le ||
         game_key_utf8)
root_index = u64_le(root_digest[0..8]) mod roots_in_game
```

- gameは`(game_digest, game_key UTF-8 bytes)`の辞書順にする。
- game内rootは`(phase order opening/middle/late, network_input_hash, source line SHA)`順にする。
- `game_index = draw_index mod game_count`、`cycle = draw_index / game_count`とする。
- hash衝突時は上記の第2/第3 keyでtie-breakする。
- gameごとのdraw数差は常に最大1とする。
- `draw_index`は各outer開始時に0へ戻す。
- active checkpointで`next_draw_index == roots_completed`を必ず検証する。
- completed outerには当該outerの最終draw数をmetricsへ記録し、次outerは0から開始する。

10k norm校正では更新を行わず、MMTOとWDLのaggregate projected/normalized gradientを作る。materialを除去後、
MMTO単独gradientと`coefficient * WDL`単独gradientをそれぞれproductionと同じglobal clipへ通す。outer 4
accumulator、feature/output LR比、epsilonを使い、各々について
`lr * clipped_g / (sqrt(accumulator + clipped_g^2) + epsilon)`の仮想update normを計算する。有界二分法で次を
満たす一つの係数を求める。

```text
norm(AdaGrad(coefficient * WDL)) / norm(AdaGrad(MMTO)) = 0.10
```

solverの許容差はratioで`1e-4`、係数範囲は`[0, 1000]`、最大iterationは80とする。上限で10%へ達しない、
0以外が非finite、MMTO/WDL normが0の場合はfail closedとする。reportにはkappa/coefficientのf32 bits、
target/achieved ratio、clip前raw norm/cosine、clip後preconditioned norm/cosine、root/pair/weight数、全入力SHAを
保存する。combined実学習のclipは両成分の結合後に一度だけ行うため、校正reportのisolated normは係数決定用の
固定尺度であり、各batchで10%を強制する値ではない。

`kappa.json`と`norm-calibration.json`はsingle pass完了後にatomic createする。既存時はrequest fingerprintと
全入力SHAを検証して同一なら再利用し、一つでも違えば上書きせず`blocked_integrity`とする。

### 5.2 50k-root checkpoint v2

checkpointは次を保存する。

```text
CheckpointV2
├── trajectory_fingerprint
├── generation
├── completed_outer
├── model / optimizer artifact
├── material_bits
└── active_outer
    ├── outer / roots_completed / milestone_index
    ├── JSONL file index / byte offset / next line number
    ├── raw and eligible records seen
    ├── WDL next_draw_index
    ├── outer-parent model SHA
    ├── immutable search_snapshot SHA
    ├── accumulated SearchMetrics
    └── complete UpdateDiagnostics including delta histogram
```

outer開始時に`search_snapshot.bin`をatomic copyし、resume時も必ずこれをEngineEvaluatorへ読む。途中更新された
`model.bin`をsearch snapshotに使ってはならない。50,000 root境界ではpartial search blockとpending gradientを
必ずflushする。このflushはtrajectoryの一部なので、`checkpoint_roots`をfingerprintへ含める。wall-clock時間を
理由にflushしてはならない。

output初期化時は学習開始前のmodel/optimizer、cursor 0、draw index 0、search snapshotを束ねたgeneration-0
checkpointをcurrentとしてcommitし、同じtupleを`model.previous`、`optimizer.previous`、
`checkpoint.previous.json`にもcommitする。それからactive outerを開始する。これにより最初の50k milestone
途中の障害にも有効なprevious tupleが存在する。

milestoneのcommit順は次に固定する。

1. 現在commit済みmodelを`model.previous`へatomic copy
2. 現在commit済みoptimizerを`optimizer.previous`へatomic copy
3. 現在checkpointを`checkpoint.previous.json`へatomic copy
4. 新modelをtmpへwrite、flush、fsync、rename、parent fsync
5. 新optimizerを同様に保存
6. 両SHAを再計算
7. 新`checkpoint.json`を最後にatomic replace

current tupleはcheckpoint欠落、JSON parse失敗、schema不正、参照artifact欠落、SHA/bytes不一致のいずれでも
無効とし、previous一世代を検証する。`checkpoint.previous.json`が記録するmodel/optimizer SHAは、復旧時には
`model.previous`/`optimizer.previous`へ対応付けて検証する。currentとpreviousの両方が不正なら再学習せず
`blocked_integrity`とする。ただしtrajectory fingerprint不一致は過去世代へfallbackせず即
`blocked_integrity`にする。

trajectory fingerprintの入力型は次のexact setとする。

```text
executable SHA / build Git revision / target / feature profile
init model SHA / init optimizer SHA / material bits
train and valid manifest SHA / train and valid file SHA / max train and valid roots
feature LR / output LR ratio / tau / loss weight / clip / all parameter and delta limits
pair batch / search block / jobs / PV probe roots / checkpoint roots
objective mode
kappa value and f32 bits / coefficient value and f32 bits
WDL pool manifest and file SHA / sampler seed and version / sampling ratio
WDL-to-MMTO assignment version
```

timestamp、elapsed、PID、resume command、到達目標だけを示す`outer_iterations`はfingerprintから除外する。

timestamp、wall-clock elapsed、resume command履歴はoperational metadataとしてtrajectory fingerprintと数値学習
metricsから分離する。中断有無で一致を要求するのはmodel、optimizer、cursor、sampler、loss/count/histogram、
objective geometry、trajectory fingerprintであり、実時間やPIDではない。

outer末尾が50kの倍数でない場合も、残りrootとpending gradientをflushしたactive-final checkpointを先にcommit
する。その後fresh valid audit、metrics、run/trajectory manifestを書き、最後に
`active_outer=null, completed_outer=N`のcompleted checkpointへ遷移する。audit中の停止で末尾学習を
やり直さない。

outer末尾delta上限違反は、active-final model/optimizerと`outer_safety.json`を保存した上で、completed
checkpointの`training_status=rejected_by_safety`としてcommitする。supervisorは証拠archiveとdecisionを作るが、
fresh audit/matchや次outerへ進まない。NaN、material変更、absolute parameter boundのようにbatch中に続行不能な
違反は、そのbatchをcommitせず直前milestoneを保持し、dedicated safety failureとしてretryせずdecisionへ変換する。

### 5.3 WDL校正・audit

新規:

- `src/bin/halfkp_wdl_audit.rs`
- `src/training_tools/halfkp_wdl_audit.rs`
- `scripts/analyze_wdl_audit.py`

CLIは次へ固定する。

```text
halfkp_wdl_audit fit-kappa \
  --model PATH --input PATH --input-manifest PATH \
  --kappa-min-cp 50 --kappa-max-cp 5000 --tolerance-cp 0.01 \
  --output PATH

halfkp_wdl_audit evaluate \
  --model PATH --input PATH --input-manifest PATH \
  --kappa-manifest PATH --output PATH

analyze_wdl_audit.py \
  --parent PATH --candidate PATH --require-exact-root-set \
  --cluster game_key --bootstrap-resamples 100000 \
  --bootstrap-seed 20260823 --output PATH
```

`fit-kappa`はouter 4だけに実行し、game-equal Brierを有界golden-sectionで最小化する。候補modelごとの再fitを
拒否する。`evaluate`はroot key、game key、side、winner、raw score、probability、Brier、log-lossを1 root
1行で出す。analyzerはroot集合の完全一致を要求し、root差をgame内平均してからgame単位bootstrapする。
intersectionだけで比較する実装は禁止する。

### 5.4 dataset生成・整列

変更:

- `src/training_tools/dataset_build.rs`
- `src/training_tools/game_dataset_audit.rs`

新規:

- `src/bin/halfkp_dataset_prepare.rs`
- `src/training_tools/halfkp_dataset_prepare.rs`

`dataset_build`へ次の3引数を追加する。

```text
--base-dataset-manifest PATH
--freeze-base-eval
--max-train-records N
```

base manifest/schema/output SHA/行数を先に検証し、base全splitのgame、opening family、PackedHalfKP
network-input hashをpreloadする。base trainは一度だけmembership outputへ保持する。base valid/testは
byte-for-byte copyし、新規valid/test candidateを出力しない。base valid/testのgame/family/hashに衝突する
train候補、base trainと重複するnetwork input、supplement内重複を除外する。

membership manifest schemaは`rank_value_dataset_v5_frozen_membership_manifest`とする。最低限、従来のrate、
filter、input/output artifactに加え、次を必須fieldにする。

```text
base_manifest = {path, sha256, bytes, schema}
frozen_eval.valid/test = {path, sha256, bytes, records}
train_layers[] = {
  id, role(base|supplement), source_years,
  first_line_inclusive, last_line_exclusive, records
}
collision_counters
output_files[] = {path, sha256, bytes, records}
```

baseはtrainの`[0, base_records)`、supplementは`[base_records, total_records)`というcontiguous rangeにする。
composerはmanifest SHAと各rangeを検証してからoffsetを読む。old追加時のbaseにはcomposed manifestも使えるよう、
builder内のmanifest readerをv4/v6共通contractへ変換する。

`halfkp_dataset_prepare`は次の4 subcommandだけを持つ。

```text
wdl-pool
wdl-valid-split
compose-interleave
unseen-pilot
```

全subcommandの選択順は、manifestに固定したseed、game key、phase、network-input hash、source line SHAを
SHA-256へ入れた辞書順とする。同率時はsource file順・line番号で決める。phase quotaはopening/middle/lateの
`2:3:1`をlargest-remainder（同率はこのphase順）で整数化し、不足phase分だけ同じ規則で再配分する。
同じ入力manifestとseedからはbyte-identicalな出力とmanifestが得られることをtestで保証する。WDL poolは
gameを先にhash選択してから各game各phase最大1 root、valid splitはgame hash順の先頭1,000/残り5,099、
unseen pilotはbaseと重ならないgame-round-robin順の250,000 rootとする。

`compose-interleave`はJSON行を変更せず、次の`game_round_robin_weighted_fair_v1`で最終train順を作る。

1. 各layerをgame単位にまとめる。
2. gameを`SHA256(seed || game_key)`順にする。
3. game内recordを`SHA256(seed || phase || sfen || teacher_move)`順にする。
4. 各gameから1 recordずつround-robinする。
5. base列とsupplement列を総record比のweighted-fair mergeで混ぜる。
6. 任意prefixで各layerの割当誤差を最大1 recordとする。

ただしold追加時のbaseが、同じseed/algorithmで作られたv6 composed manifestなら、その既存train streamを
canonical base列として順序を変えない。新しいold supplementだけをgame/phase順へ変換し、base列へweighted-fair
mergeする。v6 manifest内の元layer provenance/countはnested sourceとして引き継ぐ。これによりrecent作成時の
original-base/new-recent公平性を壊さない。

hard max 300万へ縮小するときもbaseは全件を予約する。削減対象はsupplementだけであり、
`supplement_limit = 3,000,000 - base_records`とする。supplementはgame単位round-robinに加え、最終全体の
opening/middle/lateが`2:3:1`へ最も近くなるlargest-remainder quotaで選ぶ。あるphaseが不足した場合だけ、
不足数を残りphaseへ同じ固定順で再配分する。入力prefixやinterleave後の先頭300万を切り取らない。

composed manifest schemaは`rank_value_dataset_v6_composed_manifest`とし、v4と共通のrate/filter/output fields、
base/frozen artifact、flatten前後のlayer counts、compose seed/algorithm、phase quota、dropped supplement数、
全output SHA/bytes/recordsを必須にする。`composition_report.json`にはさらに次を保存する。

- base raw line SHA multisetの完全一致と`base_records_exactly_once`
- supplement選択前/後の年度・phase・game・root数
- baseにないnetwork-input数
- frozen valid/testの期待/実SHA
- cross/within split collision
- old fraction
- manifest/outputの期待/実SHA、bytes、行数

dataset publish順は`train/valid/test + manifest + composition_report + composition_ready.json`、
`game_dataset_audit`、supervisorによる全dataset Gate判定、`completed.json`のatomic create、
`phase-b-dataset.json`の順に固定する。`completed.json`をGate前提にして循環させない。
`game_dataset_audit`単体でbase件数、新規件数、old比、manifest整合を判定できると仮定しない。

base 570,875 rootを先頭へ置く方式は禁止する。MMTOはJSONL順に8,192 pairごとAdaGrad更新するため、
base-firstでは小さいaccumulatorによる大きな実効stepをbaseが独占し、newが後半の小さいstepになる。

`game_dataset_audit`の`passed`へ
`duplicate_network_inputs_within_split == 0`を追加する。trainerは新しいfrozen/composed manifest schemaを
厳密に検証し、declared train/valid/testのSHAと行数が一致しない場合は拒否する。

### 5.5 MMTO audit

変更:

- `src/training_tools/halfkp_mmto_audit.rs`
- `scripts/analyze_mmto_root_audit.py`

audit出力へPackedHalfKP network-input hash由来の`root_key`を追加する。analyzerは90,759 root、6,099 gameを
受理し、parent/candidateのroot集合完全一致を要求する。各rootのloss差を同一game内で平均し、その6,099値を
bootstrapする。duplicate、missing、extra root、game key不一致は即エラーにする。

### 5.6 対局

変更:

- `src/training_tools/halfkp_promotion_match.rs`
- `scripts/run_halfkp_paired_match.py`
- `tests/test_run_halfkp_paired_match.py`

Rust matchとwrapperへ次を追加する。

```text
--time-limit-ms N                     # --nodesと排他
--parent-authorization PATH           # --registryと排他のworkflow内認可
--openings-format dataset-jsonl|usi-lines
--opening-selection first-unique|sha256-seed
--opening-seed U64
```

budget type/value、opening format/selection/seed、authorization SHAをpair artifactとrun fingerprintへ含める。
既存fixed-node比較は`dataset-jsonl + first-unique`を使い、従来の先頭250 unique familyを変えない。
`usi-lines + sha256-seed`では全行を合法局面へ変換し、重複network inputを除外してから
`SHA256(seed || normalized line)`順の先頭25を選ぶ。これをtaya36の内部time diagnosticに使い、各局面を必ず
先後交換する。

workflow内parentは、global registryのaccepted entryか、`--parent-authorization`で渡すimmutable
authorizationのexactly oneで認可する。両方指定と両方未指定をCLI validationで拒否する。outer 4を親にする
通常学習・対局だけregistryを使い、control、workflow内accepted parent、v2.5.0外部評価はauthorizationだけを使う。

```text
schema = halfkp_workflow_parent_authorization_v1
authorization_kind =
  accepted_within_workflow | diagnostic_control | evaluation_only_external_baseline
lineage_id
config_sha256
model = {path, sha256, bytes}
model_manifest = {path, sha256, bytes}
root_evidence = {path, sha256, bytes}
gate_decision = null | {path, sha256, bytes}
allowed_budgets = [{kind, value}]
allowed_candidate_lineage
status
```

`accepted_within_workflow`では`root_evidence`がarchive manifest、`gate_decision`がそのcandidateをacceptedにした
decision、statusが`accepted_within_workflow`でなければならない。outer 2対局のparent=outer 1に使う。

`diagnostic_control`では`root_evidence`が旧control run manifest、`gate_decision`がWDL-off replay completed
markerであり、control model/optimizer/init SHAとの整合が必要である。Phase A treatment対controlだけに使う。

`evaluation_only_external_baseline`はPhase A～C configでは禁止し、best scratch確定後の外部評価configだけで
許可する。candidate manifestはscratch lineageを要求するが、parent manifestは
`schema=evaluation_only_external_baseline_v1`、`lineage_id=v2.5.0-kpp-derived-evaluation-only`、
`allowed_usage=evaluation_only`を要求し、scratch lineageだと偽装しない。v2.5.0 model SHA、
evaluation-only manifest、許可したtime budgetを根にする。match reportはcandidate/parent lineageを別fieldへ
保存する。

wrong config、decision、archive、model、budgetの組合せは拒否する。opening selectorは、選んだsource line、
normalized position、family/network hash、seed、順序を`selected_openings.manifest.json`へ保存する。既存の
opening pair単位artifact、欠けたpairだけのresume、bootstrap、安全件数検証は維持する。

### 5.7 candidate archive契約

supervisorの既存staging→rename骨格を再利用し、control package、treatment、pilot、dense outerの各完了後に
次を行う。

1. live checkpointがouter完了を示し、active milestoneがないことを確認する。
2. model、optimizer、checkpoint、該当outer metrics、run manifest、trajectory manifestをstagingへcopyする。
3. `halfkp_mmto_candidate_v2`の`model.bin.manifest.json`を作る。
4. 全fileを列挙した`halfkp_mmto_outer_archive_v2`の`archive_manifest.json`を最後に作る。
5. source/live SHAを再確認し、staging directoryを最終archive pathへatomic renameする。
6. archiveをread-only扱いにし、以後のaudit/matchはarchive pathだけを受け付ける。

candidate manifestはlineage、diagnostic generation 4、phase、completed outer、training parent SHA、
evaluation parent SHA、training objective evidence SHA、dataset manifest SHA、model artifact、
status=`diagnostic_candidate`を持つ。treatmentのobjective evidenceはnorm-calibration completed marker、
pilot/fullはPhase A objective decisionである。treatmentのtraining parentはouter 4、evaluation parentはcontrol、
pilot/full outer 1は両方outer 4、full outer 2のevaluation parentはouter 1とする。archive manifestは
model/optimizer top-level SHAと、全fileの相対path/SHA/bytesを持つ。既存archiveと同名fileを上書きしない。

既存controlにはmodel manifestがないため、WDL-off replay通過後に
`archives/phase-a/control/outer_0001`へmodel、optimizer、legacy checkpoint、outer metrics、run manifestをcopyする。
これは`archive_kind=diagnostic_control`のv2 archiveとし、`provenance_wrapper.json`へinit outer 4、balanced100k、
旧binary、WDL-off replay markerのSHAを記録してからmodel/archive manifestを生成する。通常candidateに必須の
trajectory/search snapshotは要求せず、それ以外のfileを許可しない。verifierはarchive kindごとのallow-listで
検証し、通常candidateへこの例外を適用しない。元control artifactは変更しない。

評価command中の`CANDIDATE_ARCHIVE`、`CANDIDATE_MANIFEST`、`PARENT_AUTHORIZATION`は、このarchive completed
markerまたは直前decisionからsupervisorが解決する。live directoryや手入力pathから解決してはならない。

### 5.8 無人workflow

変更:

- `scripts/halfkp_mmto_unattended.py`
- `scripts/manage_halfkp_mmto_unattended.sh`
- `tests/test_halfkp_mmto_unattended.py`

新規:

- `training/halfkp64-scratch-v1/unattended_post_wdl_dense_20260818.json`
- `scripts/systemd/halfkp-mmto-post-wdl-dense-20260818.service`

既存v1 modeへ条件分岐を足し続けず、schema
`halfkp_mmto_unattended_config_v2`、mode`post_wdl_dense_v1`を追加する。v1 testsと旧profileは維持する。

configのtop-level keyは次のexact setとし、extra keyを拒否する。

```text
schema, workspace, run_root, lineage_id, workflow, policy, plan,
service, build, parent, sources, derivations, trainer, evaluation, gates
```

重要な固定値:

| section | 値 |
|---|---|
| policy | external weightをtrainingへ使用禁止、dense最大2 outer、grid禁止、外部評価をworkflow外へ分離 |
| service | heartbeat 60秒、transient retry最大3、retry間隔60秒、開始時空き50 GiB以上 |
| parent | generation 3、outer 4 modelとoptimizerの両方 |
| WDL | pool最大100k、1 game各phase最大1、sampling 1:1、preconditioned norm比10% |
| dense | target 2m、min 1.8m、hard max 3m、新規min 1.2m、old最大25% |
| trainer | jobs 8、block 1024、pair batch 8192、LR `1e-4`、output比0.1、tau 36.63 |
| checkpoint | pilot/fullとも50,000 root |
| fixed match | 250 pairs、20,000 nodes、bootstrap 100,000、seed 20260806 |
| time match | `taya36.sfen`、25 pairs、5,000 ms/手、bootstrap 100,000、seed 20260816 |

`build`は新binaryをpath/nameと`lock_on_first_run=true`で、tracked scriptと保全済みlegacy trainerを
path/SHA/bytesで宣言する。P1より前に未知のbinary SHAをconfigへ先書きせず、P1のrun identityが実SHA/bytesを
固定する。入力SHA、seed、algorithm version、output dirはconfigで先に固定する。

### 5.9 config v2のnested contract

top-levelだけでなくnested keyもallow-listで検証する。入力artifactの実path/SHA/bytesはI8で埋め、
build outputはP1のrun identityで固定する。

| section | 必須key |
|---|---|
| `workflow` | `mode`, `state_schema`, `phase_order` |
| `policy` | `forbid_kpp_and_v250_in_training`, `max_dense_outer`, `allow_hyperparameter_grid`, `external_evaluation_in_this_workflow` |
| `plan` | artifact型`path`, `sha256`, `bytes` |
| `service` | `unit_name`, `unit_file`, `heartbeat_seconds`, `max_attempts`, `retry_seconds`, `minimum_free_gib` |
| `build` | `command`, `artifacts`, `scripts`, `legacy_trainer`, `require_clean_git`, `git_revision_policy` |
| `parent` | `generation`, `model`, `optimizer`, `checkpoint`, `archive_manifest`, `model_manifest`, `promotion_registry`, `gate_report` |
| `sources` | `balanced100k`, `frozen_current`, `control`, `wdoor_archives`, `dev_openings`, `time_openings` |
| `derivations` | `wdl_pool`, `wdl_valid`, `dense`, `pilot` |
| `trainer` | jobs/block/batch/LR/clip/tau/delta limit/checkpoint/WDL固定値 |
| `evaluation` | `mmto_audit`, `wdl_audit`, `fixed_match`, `time_match` |
| `gates` | `A`, `B`, `C` |

固定値:

```text
schema = halfkp_mmto_unattended_config_v2
workflow.mode = post_wdl_dense_v1
workflow.state_schema = halfkp_mmto_unattended_status_v2
workflow.phase_order = [A, B, C]

policy.forbid_kpp_and_v250_in_training = true
policy.max_dense_outer = 2
policy.allow_hyperparameter_grid = false
policy.external_evaluation_in_this_workflow = false

service.unit_name = halfkp-mmto-post-wdl-dense-20260818
service.heartbeat_seconds = 60
service.max_attempts = 3
service.retry_seconds = 60
service.minimum_free_gib = 50

parent.generation = 3

derivations.wdl_pool.years = [2018,2019,2020,2021,2022,2023,2024,2025]
derivations.wdl_pool.max_roots = 100000
derivations.wdl_pool.phase_caps = [1,1,1]
derivations.wdl_pool.max_year_game_fraction = 0.25
derivations.wdl_valid.calibration_games = 1000
derivations.wdl_valid.audit_games = 5099

derivations.dense.recent_years = [2023,2024,2025]
derivations.dense.supplement_years = [2020,2021,2022]
derivations.dense.target_roots = 2000000
derivations.dense.min_roots = 1800000
derivations.dense.hard_max_roots = 3000000
derivations.dense.min_new_network_inputs = 1200000
derivations.dense.supplement_max_fraction = 0.25
derivations.pilot.roots = 250000

trainer.jobs = 8
trainer.search_block_roots = 1024
trainer.pair_batch_size = 8192
trainer.learning_rate_max = 0.0001
trainer.output_learning_rate_ratio = 0.1
trainer.gradient_clip_norm = 1.0
trainer.tau_cp = 36.63
trainer.pv_probe_roots = 1024
trainer.max_feature_delta_per_outer = 0.02
trainer.max_output_delta_per_outer = 0.004
trainer.wdl_target_preconditioned_update_ratio = 0.10
trainer.wdl_sampling_ratio = 1.0
trainer.milestone_roots = 50000

evaluation.mmto_audit = 90759 roots / 6099 games / 100000 bootstrap / seed 20260823
evaluation.wdl_audit = 1000 calibration games / 5099 audit games / 100000 bootstrap / seed 20260823
evaluation.fixed_match = 250 pairs / 20000 nodes / seed 20260806
evaluation.time_match = 25 pairs / 5000 ms / seed 20260816

gates.A.brier_delta_ci95_upper_lt = 0.0
gates.A.log_loss_delta_max = 0.0
gates.A.mmto_relative_loss_degradation_max = 0.005
gates.A.fixed_score_min = 0.525
gates.A.fixed_ci95_lower_min = 0.48
gates.A.time_catastrophic_score_below = 0.40
gates.A.time_catastrophic_ci95_upper_below = 0.50

gates.B.audit_delta_ci95_upper_lt = 0.0
gates.B.fixed_score_min = 0.525
gates.B.fixed_ci95_lower_min = 0.48

gates.C.audit_delta_ci95_upper_lt = 0.0
gates.C.fixed_score_min = 0.525
gates.C.fixed_ci95_lower_min = 0.48
gates.C.qnodes_ratio_trigger_above = 1.20
gates.C.time_catastrophic_score_below = 0.40
gates.C.time_catastrophic_ci95_upper_below = 0.50
```

入力artifact、tracked script、legacy trainerのartifact型はexactな`{path, sha256, bytes}`とする。新binaryだけは
configで`{path, name, lock_on_first_run:true}`とし、P1後はrun identityのexact SHA/bytesを必須にする。output pathはrun root配下だけを許す。config preflightは
親model/optimizer/checkpoint/archiveの相互整合、generation 3 registry entry、controlがouter 4 init由来で
あることを検証する。`training/teacher/source/derivation`のpath、SHA、command tokenへv2.5.0/KPPが現れた場合は
即拒否する。

## 6. 必須test

### 6.1 Rust unit test

`halfkp_mmto_train`:

1. 黒/白手番と黒/白勝者の4組合せでWDL derivativeの符号を確認する。
2. dense、sparse、output biasについてfinite differenceとanalytic gradientを照合する。
3. MMTO pair数を複製しても、別正規化したWDL成分が変化しない。
4. 同じWDL rootを複製しても、WDL平均gradientが変化しない。
5. MMTO/WDLを別々にprojectionした後、mirror rowが一致する。
6. material gradientがraw norm、preconditioned norm、updateのすべてから除外される。
7. scalar fixtureでAdaGrad候補update normを手計算と照合する。
8. coefficient solverがtarget 10%へ`1e-4`以内で収束する。
9. samplerのgolden sequence、game draw数差最大1、draw-index resume identityを確認する。
10. WDL offが既存`apply_batch`だけを通る。
11. JSONL byte cursorがfilterでskipした行を挟んでも正しいsuffixへ戻る。
12. trajectoryのLR、binary、Git、dataset、WDL SHA、sampler、checkpoint幅の各不一致を拒否する。
13. current checkpoint破損時にprevious model/optimizer/checkpointを一組で復旧する。
14. current/previous両方の破損を拒否する。

dataset:

1. base trainの全recordをexactly once保持する。
2. valid/testがbyte-for-byte一致する。
3. base train familyをtrainへ固定し、base valid/test familyをtrainから除外する。
4. base network-input hashをsupplementで上書きしない。
5. `max-train-records`がbase込みのexact値になる。
6. base manifest SHA不一致を拒否する。
7. interleaveを同requestでbyte-identicalに再生成できる。
8. 任意prefixのlayer比誤差が最大1である。
9. recentが200万以上ならoldを0件とする。
10. oldを使う場合に`4 * old_roots <= final_roots`を満たす。
11. 300万超の縮小がgame/phase-balancedで、入力prefix cutoffではない。
12. `game_dataset_audit`がwithin-split duplicateを実際にfailさせる。
13. unseen pilotがbase trainとのnetwork-input overlap 0、exact 250,000になる。

audit/match:

1. 90,759 root/6,099 gameのMMTO fixtureを受理する。
2. duplicate、missing、extra rootを拒否する。
3. root差→game平均→game bootstrapの順を小fixtureの手計算と照合する。
4. WDL calibration/auditのgame overlapが0である。
5. fixed kappaを候補ごとに再fitできない。
6. fixed-nodeとtime budgetのpair artifactを相互reuseできない。
7. registry/authorizationのexactly-oneを受理し、両方指定・両方なしを拒否する。
8. workflow authorizationのwrong config/decision/archive/modelを拒否する。
9. diagnostic control archiveだけがlegacy file allow-listを使えることを確認する。
10. 色交換、pair resume、bootstrap seed、安全件数を既存testに加えて検証する。

### 6.2 integration / reproducibility test

1. 旧binaryと新binaryで同じ10k MMTO-only、checkpoint offを実行する。
2. 最終modelとoptimizerのSHAが両方完全一致する。
3. 一致した場合だけ既存100k controlの再利用を許す。
4. tiny runを中断なしで完了する。
5. 同じtiny runをmilestone直前/直後でSIGTERMし、resumeする。
6. model、optimizer、時間を除く数値metrics、WDL draw index、trajectory fingerprintが中断なしrunと一致する。
7. milestone commitの各段階へfault injectionし、currentまたはpreviousから一意に復旧する。
8. auditはmarkerなしなら再生成し、markerありtamperならblockする。
9. end-to-end fake childで`A fail → MMTO-only → B pass → C outer1 pass → outer2 fail`を通す。
10. end-to-end fake childで`A pass → hybrid → B dataset fail`を通し、pilotを開始せず、
    `best-scratch.json`がouter 4を選ぶことを確認する。

実装完了時に最低限、次をすべて通す。

```bash
cargo fmt --all -- --check
cargo test --features halfkp64,training-tools
cargo clippy --all-targets --features halfkp64,training-tools -- -D warnings
python3 -m unittest \
  tests.test_halfkp_mmto_unattended \
  tests.test_run_halfkp_paired_match \
  tests.test_analyze_mmto_root_audit \
  tests.test_analyze_wdl_audit
python3 scripts/tests/test_mmto_selection.py
git diff --check
```

長時間testを通常unit testへ混ぜない。tiny実binary milestone resumeはI7の明示的integration reportとし、その
report SHAをI8 configが要求する。旧/新10k replayはfinal P1 binaryでPhase A中に実行し、startup preflightの
入力にはせず、control再利用前のimmutable markerにする。

## 7. Git実装順

各行を独立commitにし、対応testと文書を同じcommitへ含める。前commitのtestが通らない状態で次へ進まない。

| ID | commit subject | 主な内容 | 完了条件 |
|---|---|---|---|
| I1 | `training: separate MMTO and WDL objective normalization` | gradient分離、WDL off分岐、sampler、norm校正 | Rust数学test |
| I2 | `training: add game-cluster WDL and MMTO audits` | WDL binary/analyzer、MMTO root key/analyzer | audit unit test |
| I3 | `training: add deterministic MMTO milestone resume` | checkpoint v2、snapshot、cursor、atomic recovery | tiny resume identity |
| I4 | `training: build frozen-eval dense WDoor datasets` | builder 3引数、prepare subcommands、audit bug修正 | dataset unit/integration test |
| I5 | `training: generalize resumable paired evaluation` | workflow authorization、time budget、USI opening selector | pair runner test、fixed/time smoke |
| I6 | `training: add post-WDL dense unattended workflow` | config v2、state machine、manager、systemd | supervisor E2E test |
| I7 | `test: verify post-WDL dense workflow readiness` | full test、tiny resume identity review | 全report合格 |
| I8 | `training: lock post-WDL dense experiment configuration` | 最終runbook/input SHA、config、進捗を同時固定 | clean worktree |

I8では本書のI1～I8を完了へ更新した最終bytesを先に確定し、そのSHAをconfigへ入れて同じcommitで固定する。
config自身のSHA、I8 commitのGit SHA、新binary SHAは自己参照を避けてP1のrun identityで固定する。
`git_revision_policy=exact_clean_head_at_p1`とし、I8後はtracked fileを変更せずP1へ進む。I1～I8の途中で学習結果を見てparameterやGateを変更しない。実装上の
誤りが見つかった場合は、原因、影響、
修正、再実行が必要なtestを本書へ追記してから修正commitを作る。

## 8. Step P0: 旧trainerを最初に保全する

このStepだけは、新しい`halfkp_mmto_train`を含む`cargo build --release`より前に行う。現在のbinaryが
上書き済みなら作業を止め、Git revision `d6a7f60`のworktreeと別target directoryから旧binaryを再構築して
SHA一致を確認する。違うSHAのbinaryで代用しない。

```bash
set -euo pipefail
cd /home/kuro/shogi_rust

RUN=runs/halfkp64-scratch-v1/post-wdl-dense-20260818-v1
LEGACY_SHA=329e32f5b79e6ce2b1dc56a14e1711da4b6c4d88157dc1d472a0f3a86129439f
LEGACY_DST="$RUN/locked_binaries/halfkp_mmto_train.legacy"

test -z "$(git status --porcelain)"
test "$(sha256sum target/release/halfkp_mmto_train | awk '{print $1}')" = "$LEGACY_SHA"
test "$(stat -c '%s' target/release/halfkp_mmto_train)" = 2041304
install -D -m 0555 target/release/halfkp_mmto_train "$LEGACY_DST"
test "$(sha256sum "$LEGACY_DST" | awk '{print $1}')" = "$LEGACY_SHA"
cmp target/release/halfkp_mmto_train "$LEGACY_DST"
```

保全先は`runs/`配下なのでcommitしない。I8 configへpath/SHA/bytesを記録し、supervisorが毎回検証する。

### 8.1 Step P1: test、release build、run identity

P0後にI1～I8を完了し、clean worktreeになってから実行する。P1より前の実装中buildは許されるが、保全済み
legacy binaryへ触れてはならない。

```bash
set -euo pipefail
cd /home/kuro/shogi_rust

RUN=runs/halfkp64-scratch-v1/post-wdl-dense-20260818-v1
mkdir -p "$RUN/logs" "$RUN/locked_binaries"

cargo fmt --all -- --check
cargo test --features halfkp64,training-tools \
  2>&1 | tee "$RUN/logs/p1-cargo-test.log"
cargo clippy --all-targets --features halfkp64,training-tools -- -D warnings \
  2>&1 | tee "$RUN/logs/p1-cargo-clippy.log"
python3 -m unittest \
  tests.test_halfkp_mmto_unattended \
  tests.test_run_halfkp_paired_match \
  tests.test_analyze_mmto_root_audit \
  tests.test_analyze_wdl_audit \
  2>&1 | tee "$RUN/logs/p1-python-test.log"
python3 scripts/tests/test_mmto_selection.py \
  2>&1 | tee "$RUN/logs/p1-mmto-selection-test.log"
git diff --check
test -z "$(git status --porcelain)"

cargo build --release --locked --features halfkp64,training-tools \
  --bin dataset_build \
  --bin game_dataset_audit \
  --bin halfkp_dataset_prepare \
  --bin halfkp_mmto_train \
  --bin halfkp_mmto_audit \
  --bin halfkp_wdl_audit \
  --bin halfkp_promotion_match

for name in \
  dataset_build game_dataset_audit halfkp_dataset_prepare \
  halfkp_mmto_train halfkp_mmto_audit halfkp_wdl_audit \
  halfkp_promotion_match
do
  install -m 0555 "target/release/$name" "$RUN/locked_binaries/$name"
done
```

supervisorのP1処理は、上記log、binary、実行に使うPython script、config、plan、Git HEADについてSHA/bytesを採取し、
`run_identity.json`と`p1-build-completed.json`をatomic createする。binary copyが0555、Git HEADがI8、
worktree clean、legacy SHAがP0正本であることを再確認する。既存run identityが同一なら再利用し、一つでも違えば
上書きせず`blocked_integrity`とする。

## 9. Phase A: corrected WDL A/B

### 9.1 共通変数

以降のcommandはI8とP1完了後、supervisorがlockしたbinaryを使用する。説明のため次を使う。

```bash
ROOT=/home/kuro/shogi_rust
RUN="$ROOT/runs/halfkp64-scratch-v1/post-wdl-dense-20260818-v1"
LOCKED="$RUN/locked_binaries"
PARENT_DIR="$ROOT/runs/halfkp64-scratch-v1/unattended-mmto-full-lr1e-4-20260808-v1/archives/phase-c/outer_0004"
BASE="$ROOT/runs/halfkp64-scratch-v1/wdoor-highrate4000-1m-v2"
BAL="$ROOT/runs/halfkp64-scratch-v1/unattended-mmto-20260806-v1/derived-balanced-100k"
REGISTRY="$ROOT/training/halfkp64-scratch-v1/promotion_registry.json"
GATE="$ROOT/training/halfkp64-scratch-v1/outer4_internal_gate_20260817.json"
```

手動実行時も`target/release`ではなく`$LOCKED`を使う。supervisorは同じcommand fingerprintを生成する。

### 9.2 WDL rawとgame-balanced pool

WDLでは終盤の敗者局面も必要なので、dense MMTOと異なり`--exclude-loser-after-ply`を付けない。2026はinputへ
含めない。

A0開始前にarchive SHAとextract CSA数を検証する。必須件数は2018=47,813、2019=106,411、2020=139,358、
2021=123,717、2022=129,846、2023=121,730、2024=110,043、2025=128,843である。一件でも違えばbuildを
開始せず`blocked_integrity`とする。Phase Bはこのpreflight markerの2020～2025部分を再検証して再利用する。

```bash
"$LOCKED/dataset_build" \
  --input "$ROOT/data/wdoor/extract/2018" \
  --input "$ROOT/data/wdoor/extract/2019" \
  --input "$ROOT/data/wdoor/extract/2020" \
  --input "$ROOT/data/wdoor/extract/2021" \
  --input "$ROOT/data/wdoor/extract/2022" \
  --input "$ROOT/data/wdoor/extract/2023" \
  --input "$ROOT/data/wdoor/extract/2024" \
  --input "$ROOT/data/wdoor/extract/2025" \
  --output-dir "$RUN/derived/wdl-raw" \
  --license-id user-authorized:wdoor-public-archive \
  --source-uri https://wdoor.c.u-tokyo.ac.jp/shogi/ \
  --seed 20260822 \
  --shuffle-games \
  --valid-percent 0 \
  --test-percent 0 \
  --opening-prefix-plies 16 \
  --max-records-per-game 3 \
  --phase-records-per-game 1 1 1 \
  --min-ply 8 \
  --max-ply 160 \
  --min-player-rate 4000 \
  --min-opponent-rate 4000 \
  --known-result-only \
  --decisive-only \
  --min-legal-moves 2 \
  --exclude-in-check

"$LOCKED/halfkp_dataset_prepare" wdl-pool \
  --input "$RUN/derived/wdl-raw/train.jsonl" \
  --input-manifest "$RUN/derived/wdl-raw/manifest.json" \
  --frozen-dataset-manifest "$BASE/manifest.json" \
  --output-dir "$RUN/derived/wdl-pool" \
  --seed 20260822 \
  --max-roots 100000 \
  --max-roots-per-game-phase 1 \
  --max-year-game-fraction 0.25
```

selectorはsource SHAをraw manifestのinput pathへ逆引きして年度を決める。年度、game、phase、手番、winnerを
決定論的に層化し、game/family/network inputがfrozen valid/testと重なるrecordを除外する。各game各phase最大1、
全network input unique、2026件数0、単一年度game比25%以下をGateにする。

### 9.3 kappa calibration/audit split

既存one-root/game 6,099をgame hashで1,000/5,099へ一度だけ分割する。

```bash
"$LOCKED/halfkp_dataset_prepare" wdl-valid-split \
  --input "$ROOT/training/halfkp64-scratch-v1/mmto_audit_valid_20260802.jsonl" \
  --input-manifest "$ROOT/training/halfkp64-scratch-v1/mmto_audit_valid_20260802.manifest.json" \
  --output-dir "$RUN/derived/wdl-valid" \
  --seed 20260823 \
  --calibration-games 1000 \
  --audit-games 5099

"$LOCKED/halfkp_wdl_audit" fit-kappa \
  --model "$PARENT_DIR/model.bin" \
  --input "$RUN/derived/wdl-valid/calibration.jsonl" \
  --input-manifest "$RUN/derived/wdl-valid/manifest.json" \
  --kappa-min-cp 50 \
  --kappa-max-cp 5000 \
  --tolerance-cp 0.01 \
  --output "$RUN/phase-a/kappa.json"
```

`kappa.json`にはmodel/input SHA、game数、範囲、algorithm、最適kappaとf32 bits、Brier、境界到達有無を保存する。
境界が最適ならWDL校正不成立としてPhase A decisionを`mmto_only`にし、範囲を広げずPhase Bへ進む。非finite、
game数/SHA不一致は実装・data integrity failureなので`blocked_integrity`とする。

### 9.4 10k AdaGrad norm校正

```bash
"$LOCKED/halfkp_mmto_train" \
  --train "$BAL/train.jsonl" \
  --valid "$BAL/valid.jsonl" \
  --dataset-manifest "$BAL/manifest.json" \
  --expected-dataset-manifest-sha256 f1c388c35747599a83716b2de170c7828362281aecd25923ba5614072d482066 \
  --init "$PARENT_DIR/model.bin" \
  --init-optimizer "$PARENT_DIR/optimizer.bin" \
  --model-manifest "$PARENT_DIR/model.bin.manifest.json" \
  --promotion-registry "$REGISTRY" \
  --gate-report "$GATE" \
  --parent-generation 3 \
  --output-dir "$RUN/phase-a/norm-calibration-work" \
  --objective hybrid \
  --wdl-train "$RUN/derived/wdl-pool/train.jsonl" \
  --wdl-pool-manifest "$RUN/derived/wdl-pool/manifest.json" \
  --expected-wdl-pool-manifest-sha256 FROM_COMPLETED_MARKER \
  --wdl-kappa-manifest "$RUN/phase-a/kappa.json" \
  --expected-wdl-kappa-manifest-sha256 FROM_COMPLETED_MARKER \
  --wdl-sampler-seed 20260822 \
  --calibrate-wdl-objective-report "$RUN/phase-a/norm-calibration.json" \
  --wdl-calibration-roots 10000 \
  --wdl-target-update-ratio 0.10 \
  --learning-rate 0.0001 \
  --output-learning-rate-ratio 0.1 \
  --gradient-clip-norm 1 \
  --tau-cp 36.63 \
  --search-block-roots 1024 \
  --jobs 8 \
  --pair-batch-size 8192 \
  --checkpoint-roots 0 \
  --max-train-roots 10000 \
  --max-valid-roots 10000 \
  --pv-probe-roots 1024 \
  --expected-binary-sha256 FROM_RUN_IDENTITY \
  --expected-git-revision FROM_RUN_IDENTITY
```

`FROM_*`はshell文字列として実行せず、supervisorがmarker/run identityから実SHAを展開するplaceholderである。
校正reportへ`kappa.json`のSHAも束ね、以後のhybridはこのreportをobjective manifestとして使う。
solverが有限な10%係数を求められない場合はWDL objective不成立として`mmto_only`でPhase Bへ進み、係数上限を
広げたり別ratioを試さない。入力/SHA/fingerprint不一致はintegrity blockとする。

### 9.5 旧/新WDL-off 10k replay

旧/新で共通に固定する引数は次である。

```bash
COMMON_MMTO=(
  --train "$BAL/train.jsonl"
  --valid "$BAL/valid.jsonl"
  --dataset-manifest "$BAL/manifest.json"
  --expected-dataset-manifest-sha256 f1c388c35747599a83716b2de170c7828362281aecd25923ba5614072d482066
  --init "$PARENT_DIR/model.bin"
  --init-optimizer "$PARENT_DIR/optimizer.bin"
  --model-manifest "$PARENT_DIR/model.bin.manifest.json"
  --promotion-registry "$REGISTRY"
  --gate-report "$GATE"
  --parent-generation 3
  --outer-iterations 1
  --search-block-roots 1024
  --jobs 8
  --pair-batch-size 8192
  --learning-rate 0.0001
  --output-learning-rate-ratio 0.1
  --gradient-clip-norm 1
  --tau-cp 36.63
  --loss-weight 1
  --wdl-weight 0
  --feature-limit 1
  --output-limit 0.2
  --max-feature-delta-per-outer 0.02
  --max-output-delta-per-outer 0.004
  --max-train-roots 10000
  --max-valid-roots 10000
  --pv-probe-roots 1024
)

"$LOCKED/halfkp_mmto_train.legacy" \
  "${COMMON_MMTO[@]}" \
  --output-dir "$RUN/phase-a/replay-old"

"$LOCKED/halfkp_mmto_train" \
  "${COMMON_MMTO[@]}" \
  --output-dir "$RUN/phase-a/replay-new" \
  --objective mmto-only \
  --checkpoint-roots 0 \
  --expected-binary-sha256 FROM_RUN_IDENTITY \
  --expected-git-revision FROM_RUN_IDENTITY

cmp "$RUN/phase-a/replay-old/model.bin" "$RUN/phase-a/replay-new/model.bin"
cmp "$RUN/phase-a/replay-old/optimizer.bin" "$RUN/phase-a/replay-new/optimizer.bin"
```

両`cmp`が成功しない場合、既存controlを再利用しない。差分原因を修正してI1/I3へ戻り、新Git revisionと
新run rootでreplayをやり直す。許容誤差比較へ緩和してはならない。

両`cmp`成功後、前節のcontrol archiveと`diagnostic_control` parent authorizationをatomic publishする。この
authorizationがない限りtreatment対controlの対局を開始しない。

### 9.6 corrected treatment 100k

replay通過後だけ、既存controlを比較相手として一度実行する。

```bash
"$LOCKED/halfkp_mmto_train" \
  --train "$BAL/train.jsonl" \
  --valid "$BAL/valid.jsonl" \
  --dataset-manifest "$BAL/manifest.json" \
  --expected-dataset-manifest-sha256 f1c388c35747599a83716b2de170c7828362281aecd25923ba5614072d482066 \
  --init "$PARENT_DIR/model.bin" \
  --init-optimizer "$PARENT_DIR/optimizer.bin" \
  --model-manifest "$PARENT_DIR/model.bin.manifest.json" \
  --promotion-registry "$REGISTRY" \
  --gate-report "$GATE" \
  --parent-generation 3 \
  --output-dir "$RUN/phase-a/treatment-live" \
  --objective hybrid \
  --wdl-train "$RUN/derived/wdl-pool/train.jsonl" \
  --wdl-pool-manifest "$RUN/derived/wdl-pool/manifest.json" \
  --expected-wdl-pool-manifest-sha256 FROM_COMPLETED_MARKER \
  --wdl-objective-manifest "$RUN/phase-a/norm-calibration.json" \
  --expected-wdl-objective-manifest-sha256 FROM_COMPLETED_MARKER \
  --wdl-sampler-seed 20260822 \
  --outer-iterations 1 \
  --search-block-roots 1024 \
  --jobs 8 \
  --pair-batch-size 8192 \
  --learning-rate 0.0001 \
  --output-learning-rate-ratio 0.1 \
  --gradient-clip-norm 1 \
  --tau-cp 36.63 \
  --feature-limit 1 \
  --output-limit 0.2 \
  --max-feature-delta-per-outer 0.02 \
  --max-output-delta-per-outer 0.004 \
  --max-train-roots 100000 \
  --max-valid-roots 10000 \
  --pv-probe-roots 1024 \
  --checkpoint-roots 0 \
  --expected-binary-sha256 FROM_RUN_IDENTITY \
  --expected-git-revision FROM_RUN_IDENTITY
```

trainer完了後、model/optimizer/checkpoint/run manifestをstagingへcopyし、全SHA検証後に
`archives/phase-a/treatment/outer_0001`へrenameする。それより前にauditや対局を開始しない。

### 9.7 Phase A評価順

1. control/treatmentをaudit 5,099 gameへ`halfkp_wdl_audit evaluate`
2. `analyze_wdl_audit.py`でpaired Brier/log-loss
3. control/treatmentをfrozen valid 90,759 rootへ`halfkp_mmto_audit`
4. 改修済み`analyze_mmto_root_audit.py`でgame-cluster比較
5. fixed 20,000 nodes、250 opening pair
6. それまでのGate通過時だけ5秒、25 opening pair
7. `phase-a-objective.json`をcommit

MMTO auditの入力はone-root fileではなく`$BASE/valid.jsonl`である。

```bash
"$LOCKED/halfkp_mmto_audit" \
  --model CANDIDATE_ARCHIVE/model.bin \
  --input "$BASE/valid.jsonl" \
  --output CANDIDATE_AUDIT.jsonl \
  --tau-cp 36.63 \
  --block-roots 1024 \
  --jobs 8 \
  --max-roots 90759
```

fixed matchのcommand contract:

```bash
python3 "$ROOT/scripts/run_halfkp_paired_match.py" \
  --binary "$LOCKED/halfkp_promotion_match" \
  --candidate "$RUN/archives/phase-a/treatment/outer_0001/model.bin" \
  --candidate-manifest "$RUN/archives/phase-a/treatment/outer_0001/model.bin.manifest.json" \
  --parent "$RUN/archives/phase-a/control/outer_0001/model.bin" \
  --parent-manifest "$RUN/archives/phase-a/control/outer_0001/model.bin.manifest.json" \
  --parent-authorization "$RUN/decisions/phase-a-control-authorization.json" \
  --openings "$ROOT/training/halfkp64-scratch-v1/mmto_dev_openings_seed20260731_250/openings.jsonl" \
  --openings-format dataset-jsonl \
  --opening-selection first-unique \
  --pairs 250 \
  --nodes 20000 \
  --max-depth 64 \
  --max-plies 256 \
  --output-dir "$RUN/evaluations/phase-a-treatment/fixed-match" \
  --bootstrap-resamples 100000 \
  --bootstrap-seed 20260806
```

それまでのGate通過時だけ、同じcandidate/parent/authorizationでopeningsを`taya36.sfen`、
formatを`usi-lines`、selectionを`sha256-seed`、opening seedを`20260816`、pairsを25、
budgetを`--time-limit-ms 5000`へ変えて実行する。fixedとtimeのoutput directoryは分ける。

Phase A Gate不合格はworkflow全体の失敗ではない。decisionへ
`{"objective":"mmto_only","passed":false,"reasons":[...]}`を保存してPhase Bへ進む。全条件通過時だけ
`objective=hybrid`とし、pool/kappa/objective manifestのSHAをdecisionへ固定する。係数やseedを変えた再試験は
行わない。

## 10. Phase B: dense datasetと250k pilot

### 10.0 extract完全性preflight

過去に展開途中のdirectoryを全件として数えた事故があったため、builder開始前に年度別CSA数をexact検証する。

| 年 | 必須CSA数 |
|---:|---:|
| 2020 | 139,358 |
| 2021 | 123,717 |
| 2022 | 129,846 |
| 2023 | 121,730 |
| 2024 | 110,043 |
| 2025 | 128,843 |

recent合計は360,616、old合計は392,921でなければならない。archive SHA、extract directory、CSA数を
preflight markerへ保存する。builder reportの盤面依存filter前cap96候補は、recent 4,101,852
（2023: 1,327,675、2024: 1,498,626、2025: 1,275,551）、old 1,210,430
（2020: 188,443、2021: 694,336、2022: 327,651）と照合する。差があればdatasetをpublishせず
`blocked_integrity`とする。

### 10.1 recent membership build

現行datasetとの差はcap 24→96、phase 8/12/4→32/48/16、global max-records撤廃である。それ以外を変えない。
`phase-records-per-game`の合計が実効capなので、両flagをmanifestへ残しつつ合計96をtestする。

```bash
"$LOCKED/dataset_build" \
  --input "$ROOT/data/wdoor/extract/2023" \
  --input "$ROOT/data/wdoor/extract/2024" \
  --input "$ROOT/data/wdoor/extract/2025" \
  --output-dir "$RUN/derived/dense-recent-membership" \
  --base-dataset-manifest "$BASE/manifest.json" \
  --freeze-base-eval \
  --license-id user-authorized:wdoor-public-archive \
  --source-uri https://wdoor.c.u-tokyo.ac.jp/shogi/ \
  --seed 20260821 \
  --shuffle-games \
  --valid-percent 10 \
  --test-percent 10 \
  --opening-prefix-plies 16 \
  --max-records-per-game 96 \
  --phase-records-per-game 32 48 16 \
  --min-ply 8 \
  --max-ply 160 \
  --min-player-rate 4000 \
  --min-opponent-rate 4000 \
  --known-result-only \
  --decisive-only \
  --exclude-loser-after-ply 80 \
  --min-legal-moves 2 \
  --exclude-in-check

"$LOCKED/halfkp_dataset_prepare" compose-interleave \
  --membership-manifest "$RUN/derived/dense-recent-membership/manifest.json" \
  --output-dir "$RUN/derived/dense-recent" \
  --seed 20260821 \
  --algorithm game_round_robin_weighted_fair_v1 \
  --hard-max-train-roots 3000000
```

base 570,875を含むcomposed recent train総数を`R`とする。supplementだけの件数ではない。
`R >= 2,000,000`なら旧年度を使わず、`dense-recent`をfinalとして扱う。directoryをrename/copyしてmanifest内
pathを壊さず、後述の`phase-b-dataset.json`へcanonical dataset artifactとして`dense-recent`のpath/SHAを
記録する。

### 10.2 条件付き2020～2022 supplement

`R < 2,000,000`の場合だけ実行する。

```text
ADD = min(2,000,000 - R, floor(R / 3))
TARGET = R + ADD
```

`floor(R/3)`は`old / final <= 25%`の厳密な上限である。

```bash
"$LOCKED/dataset_build" \
  --input "$ROOT/data/wdoor/extract/2020" \
  --input "$ROOT/data/wdoor/extract/2021" \
  --input "$ROOT/data/wdoor/extract/2022" \
  --output-dir "$RUN/derived/dense-old-membership" \
  --base-dataset-manifest "$RUN/derived/dense-recent/manifest.json" \
  --freeze-base-eval \
  --max-train-records TARGET_FROM_DECISION \
  --license-id user-authorized:wdoor-public-archive \
  --source-uri https://wdoor.c.u-tokyo.ac.jp/shogi/ \
  --seed 20260821 \
  --shuffle-games \
  --valid-percent 10 \
  --test-percent 10 \
  --opening-prefix-plies 16 \
  --max-records-per-game 96 \
  --phase-records-per-game 32 48 16 \
  --min-ply 8 \
  --max-ply 160 \
  --min-player-rate 4000 \
  --min-opponent-rate 4000 \
  --known-result-only \
  --decisive-only \
  --exclude-loser-after-ply 80 \
  --min-legal-moves 2 \
  --exclude-in-check

"$LOCKED/halfkp_dataset_prepare" compose-interleave \
  --membership-manifest "$RUN/derived/dense-old-membership/manifest.json" \
  --output-dir "$RUN/derived/dense-final" \
  --seed 20260821 \
  --algorithm game_round_robin_weighted_fair_v1 \
  --hard-max-train-roots 3000000
```

2018～2019をdense MMTOへ追加しない。recentが300万を超える場合はfull membership生成後にcomposerが
base全件を残してsupplementだけをgame/phase-balancedに縮小し、生入力またはinterleave後の先頭で停止しない。

### 10.3 dataset Gate

supervisorは次のようにcandidateを一意に選ぶ。

```text
DENSE_CANDIDATE = derived/dense-recent  if R >= 2,000,000
DENSE_CANDIDATE = derived/dense-final   if R < 2,000,000
```

```bash
"$LOCKED/game_dataset_audit" \
  "$DENSE_CANDIDATE/train.jsonl" \
  "$DENSE_CANDIDATE/valid.jsonl" \
  "$DENSE_CANDIDATE/test.jsonl" \
  --expected-license-id user-authorized:wdoor-public-archive \
  --expected-source-uri https://wdoor.c.u-tokyo.ac.jp/shogi/ \
  --report "$DENSE_CANDIDATE/dataset_audit.json"
```

次をすべて満たす。

- valid/test SHAが開始時正本と完全一致
- base train 570,875 recordがexactly once
- train 1,800,000以上3,000,000以下
- baseにないnetwork inputが1,200,000以上
- game/family/network-input cross-split collisionが0
- within-split network-input duplicateが0
- illegal teacher、unknown/nondecisive result、provenance/schema mismatchが0
- old使用時`4 * old_roots <= final_roots`
- manifestの行数/SHA、seed、rate/ply/check/loser filterが実出力と一致

`game_dataset_audit`に加え、candidateの`composition_report.json`と`composition_ready.json`を検証する。
全Gate合格時だけ`completed.json`を作る。
`phase-b-dataset.json`はcanonical datasetのpath/SHA/bytes/recordsを持ち、以降はこのdecisionだけからdataset
pathを解決する。不合格なら全countと理由を同decisionへ保存し、outer 4を選ぶ`best-scratch.json`も作って
`stopped_by_gate`で正常終了する。閾値を緩めず、pilotを開始しない。

### 10.4 unseen 250k

```bash
"$LOCKED/halfkp_dataset_prepare" unseen-pilot \
  --input DENSE_TRAIN_FROM_PHASE_B_DATASET_DECISION \
  --input-manifest DENSE_MANIFEST_FROM_PHASE_B_DATASET_DECISION \
  --exclude-network-inputs "$BASE/train.jsonl" \
  --frozen-valid "$BASE/valid.jsonl" \
  --frozen-test "$BASE/test.jsonl" \
  --output-dir "$RUN/derived/unseen-250k" \
  --roots 250000 \
  --seed 20260824 \
  --balance game,phase
```

exact 250,000、base train overlap 0、game/phase層化、valid/test SHA不変をcompleted markerで固定する。

### 10.5 pilot

pilotはouter 4 model/optimizerから開始し、fullへ継承しない。Phase A decisionが`hybrid`なら同じpool、
kappa、coefficient、samplerを渡し、`mmto_only`ならWDL引数を一切渡さない。この分岐以外の引数は同じである。

```bash
"$LOCKED/halfkp_mmto_train" \
  --train "$RUN/derived/unseen-250k/train.jsonl" \
  --valid "$RUN/derived/unseen-250k/valid.jsonl" \
  --dataset-manifest "$RUN/derived/unseen-250k/manifest.json" \
  --expected-dataset-manifest-sha256 FROM_COMPLETED_MARKER \
  --init "$PARENT_DIR/model.bin" \
  --init-optimizer "$PARENT_DIR/optimizer.bin" \
  --model-manifest "$PARENT_DIR/model.bin.manifest.json" \
  --promotion-registry "$REGISTRY" \
  --gate-report "$GATE" \
  --parent-generation 3 \
  --output-dir "$RUN/phase-b/pilot-live" \
  --outer-iterations 1 \
  --search-block-roots 1024 \
  --jobs 8 \
  --pair-batch-size 8192 \
  --learning-rate 0.0001 \
  --output-learning-rate-ratio 0.1 \
  --gradient-clip-norm 1 \
  --tau-cp 36.63 \
  --feature-limit 1 \
  --output-limit 0.2 \
  --max-feature-delta-per-outer 0.02 \
  --max-output-delta-per-outer 0.004 \
  --max-train-roots 250000 \
  --max-valid-roots 90759 \
  --pv-probe-roots 1024 \
  --checkpoint-roots 50000 \
  --expected-binary-sha256 FROM_RUN_IDENTITY \
  --expected-git-revision FROM_RUN_IDENTITY \
  OBJECTIVE_ARGS_FROM_PHASE_A_DECISION
```

archive後、outer 4とのfull frozen auditとfixed 250-pair matchを行う。全安全Gate、game-bootstrap loss差95%上限
`< 0`、score `>= 0.525`、opening-pair bootstrap 95%下限`>= 0.48`をすべて満たした場合だけPhase Cへ進む。
不合格ならpilot artifactを保持し、bestをouter 4として`stopped_by_gate`にする。同条件のpilotを再試行しない。
この場合も先にouter 4を選ぶ`best-scratch.json`をatomic createする。

## 11. Phase C: dense full

### 11.1 LRを一度だけ固定する

pilotのfeature max deltaを`Dpilot`、pilot pairsを`Ppilot`、dense root数を`Nfull`とする。

```text
Pfull_est = Ppilot * Nfull / 250000
Dpred = Dpilot * sqrt(Pfull_est / Ppilot)
LRfull = min(1e-4, 1e-4 * 0.015 / Dpred)  if Dpred > 0
LRfull = 1e-4                              if Dpred = 0
```

すなわち予測feature deltaが0.015を超える場合だけLRを式で下げる。値とf32 bits、全入力値をimmutable
`phase-c-lr-decision.json`へ保存する。対局結果やouter 1結果を見てouter 2のLRを変えない。

### 11.2 outer 1

fullもpilotを継承せず、outer 4 model/optimizerから再開始する。trainer commandはpilotのdataset path、
root limit、output dir、LRだけを次へ変える。

```text
train/manifest = phase-b-dataset.jsonのcanonical dataset artifact
valid = frozen valid 90,759
max_train_roots = dense manifestの全train root
output_dir = phase-c/full-live
learning_rate = phase-c-lr-decision.jsonのLRfull
outer_iterations = 1
checkpoint_roots = 50000
```

objectiveはPhase A decisionの値をそのまま使う。各50k milestoneでcurrent/previous checkpointを検証し、
12時間milestoneが進まない場合はwarningを出すが、wall-clock flushや自動LR変更をしない。

outer 1完了後の順序:

1. 最終checkpoint/model/optimizer/metrics/run manifestの整合確認
2. immutable archive publish
3. frozen 90,759-root auditとgame bootstrap
4. outer 4とのfixed 250-pair match
5. `candidate_qnodes_per_root / parent_qnodes_per_root > 1.20`の場合だけ5秒25-pair match
6. `phase-c-outer-0001.json`をcommit

不合格ならbest=outer 4として`best-scratch.json`を書き、`stopped_by_gate`にする。

### 11.3 outer 2

outer 1がacceptedの場合だけ、`accepted-parent-outer-0001.json`を作り、同じlive directoryで実行する。

```bash
"$LOCKED/halfkp_mmto_train" \
  SAME_TRAJECTORY_ARGUMENTS_AS_OUTER_1 \
  --outer-iterations 2 \
  --resume
```

resume前にouter 1 archive/decision/authorizationとlive checkpointが一致することを確認する。outer 2はouter 1
archiveとだけ比較し、outer 4との内部500局を重複実行しない。Gate不合格ならbest=outer 1、合格ならbest=outer 2
として`best-scratch.json`をcommitする。最大2 outerで必ず終了する。

## 12. Gateのexact定義

### 12.1 共通安全Gate

すべて0または上限内を要求する。

- NaN / Inf
- illegal move / missing root result / search failure
- provenance / SHA / schema mismatch
- rejected non-exact bound
- feature max delta `<= 0.02`
- hidden max delta `<= 0.02`
- output max delta `<= 0.004`
- material bits変更

安全違反を「統計的な中間帯」として扱わない。

### 12.2 Phase A

frozen MMTO lossはroot数の多いgameを過大評価しない。model `m`について
`L_m = mean_game(mean_root(loss_m(root)))`、relative degradationは
`(L_treatment - L_control) / L_control`とする。root集合・game集合の完全一致と`L_control > 0`を要求し、
analyzer reportへ両`L_m`とrelative値を保存する。

| 指標 | 条件 |
|---|---:|
| paired Brier treatment-control | bootstrap 95%上限 `< 0` |
| paired log-loss treatment-control | mean `<= 0` |
| frozen MMTO relative loss degradation | `<= 0.005` |
| fixed-node score | `>= 0.525` |
| fixed-node pair bootstrap lower | `>= 0.48` |
| time catastrophic stop | score `< 0.40`かつpair bootstrap upper `< 0.50` |
| safety | 全項目通過 |

time matchは明白な退行だけを止める。catastrophic条件の両方が成立した場合だけtime Gate違反とする。

### 12.3 Phase B pilot / Phase C outer共通

| 指標 | 条件 |
|---|---:|
| fresh audit candidate-parent loss差 | game bootstrap 95%上限 `< 0` |
| fixed-node score | `>= 0.525` |
| fixed-node pair bootstrap lower | `>= 0.48` |
| safety | 全項目通過 |

Phase B pilotはここまでで判定し、time matchを行わない。Phase C outerだけ次を追加する。

| 指標 | 条件 |
|---|---:|
| time match実行条件 | `candidate_qnodes_per_root / parent_qnodes_per_root > 1.20` |
| time catastrophic stop | score `< 0.40`かつpair bootstrap upper `< 0.50` |

平均lossが改善してもCI上限が0なら不採用である。score 52.5%未満やCI下限48%未満も中間帯として次outerへ
進めない。このGateは「release強さの証明」ではなく、CPUを次outerへ投入する最低条件である。

## 13. 無人state machine

```text
P0 preflight / legacy lock verification
P1 build, test-report verification, run identity and binary lock
A0 WDL raw/pool
A1 WDL calibration/audit split
A2 kappa fit
A3 10k norm calibration
A4 old/new WDL-off replay
A5 treatment train and archive
A6 WDL audit
A7 full MMTO audit
A8 fixed match
A9 conditional time match
A10 objective decision: hybrid or mmto_only
B0 recent dense membership and interleave
B1 conditional old supplement and interleave
B2 dataset audit and decision
B2 fail -> F0(outer 4) -> stopped_by_gate
B3 unseen 250k selection
B4 pilot train and archive
B5 pilot audit/match
B6 pilot decision
B6 fail -> F0(outer 4) -> stopped_by_gate
C0 immutable LR decision
C1 outer 1 train/archive/audit/match/decision
C1 fail -> F0(outer 4) -> stopped_by_gate
C2 conditional outer 2 train/archive/audit/match/decision
C2 pass/fail -> F0(outer 2/outer 1) -> completed
```

各stateはcompleted markerが存在すれば、全SHAと意味を再検証してno-opにする。markerなしの部分出力はstaging
として捨てて再実行する。markerありの改変は削除・再生成せず`blocked_integrity`にする。

status enum:

```text
not_started
running
stopped_by_user
stopped_by_gate
blocked_integrity
blocked_environment
crashed
completed
```

status v2のexact field:

```text
schema
config_path / config_sha256 / run_identity_sha256
status / current_state / current_phase / current_outer / current_step
child_pid
progress = {
  roots_completed, roots_total, last_milestone,
  elapsed_seconds, eta_seconds
}
attempts / completed_markers
objective / accepted_parent / best_candidate
warnings / last_error / terminal_reason
created_at / started_at / updated_at
supervisor_crash_count
```

`resume`を許すのは`stopped_by_user`、環境回復後の`blocked_environment`、preflight再通過後の`crashed`だけである。
`stopped_by_gate`、`blocked_integrity`、`completed`は同じconfigでresumeしない。
正常なGate判定で終了する全経路は、terminal statusより先に`best-scratch.json`を作る。Phase C候補が未採用なら
outer 4、outer 1だけ採用済みならouter 1、outer 2採用時はouter 2を選ぶ。integrity/environment blockは
実験結果ではないため、このfileを新規確定しない。

## 14. failure / recovery

| 状況 | retry | 処置 |
|---|---:|---|
| config/Git/binary/input SHA不一致 | 0 | `blocked_integrity`。新commit/config/run rootで再審査 |
| 親model/optimizer/checkpoint不整合 | 0 | `blocked_integrity` |
| build/transient child exit | 最大3 | attempt別log、60秒間隔。3回失敗でblock |
| dataset staging中断 | 最大3 | 未publish stagingを削除して再生成 |
| dataset markerありtamper | 0 | 再生成せずintegrity block |
| dataset件数Gate不合格 | 0 | decisionとbest=outer 4を保存、`stopped_by_gate` |
| trainer SIGTERM/停電 | 自動再開 | current milestone、不正ならprevious一世代から`--resume` |
| current/previous両checkpoint不正 | 0 | `blocked_integrity` |
| 12時間milestoneなし | 0 | warningのみ。固定root境界まで継続 |
| outer末尾delta上限違反 | 0 | rejected archive。AはMMTO-only、B/CはGate stop |
| batch中NaN/material/bound違反 | 0 | 直前milestone保持。retryせず安全Gate不合格 |
| Phase A Gate不合格 | 0 | `mmto_only` decisionでPhase Bへ継続 |
| Phase B pilot不合格 | 0 | `best-scratch.json`=outer 4、Gate stop |
| Phase C outer 1不合格 | 0 | `best-scratch.json`=outer 4、outer 2なし |
| Phase C outer 2不合格 | 0 | `best-scratch.json`=outer 1 |
| audit中断・markerなし | 再実行 | 部分auditを破棄 |
| paired match中断 | 再開 | fingerprint検証後、欠けたpairだけ実行 |
| 明示stop | 0 | STOP marker→child group停止→`stopped_by_user` |
| OS shutdown | 自動再開 | STOP markerなし。boot後最後のmilestone/pairから |
| AC offline/空き容量不足 | 0 | 次境界で`blocked_environment`。回復後resume可 |
| supervisor例外 | 最大3 restart | crash loop 3回でblock |

provenance mismatch、invalid schema、tamperのような決定論的failureは3回retryせず即blockする。Gate不合格は
process failureではなく正常な実験結果なので、decisionを書いてexit 0にする。

## 15. systemd運用

managerへ新configを3つ目のreviewed profileとして登録し、旧2 unitのどちらかがactiveなら開始を拒否する。
systemd unitは既存の`systemd-inhibit`、user linger、`KillMode=control-group`、`Restart=on-failure`を維持する。

実装・I8 config確定後の運用command:

```bash
cd /home/kuro/shogi_rust
CONFIG=training/halfkp64-scratch-v1/unattended_post_wdl_dense_20260818.json

scripts/manage_halfkp_mmto_unattended.sh -c "$CONFIG" validate
loginctl enable-linger "$USER"
loginctl show-user "$USER" --property=Linger
scripts/manage_halfkp_mmto_unattended.sh -c "$CONFIG" start

scripts/manage_halfkp_mmto_unattended.sh -c "$CONFIG" status
scripts/manage_halfkp_mmto_unattended.sh -c "$CONFIG" logs

scripts/manage_halfkp_mmto_unattended.sh -c "$CONFIG" stop
scripts/manage_halfkp_mmto_unattended.sh -c "$CONFIG" resume
```

start直後に、unit active、sleep inhibitor、`run_identity.json`、0555のlocked binaries、child PID、60秒heartbeatを
確認する。Gate stop/completedはexit 0とし、systemdが再起動しないことを確認する。

## 16. `best-scratch`後のv2.5.0評価

これはPhase A～C configとserviceの外で行う。先に`best-scratch.json`、candidate archive/model manifestのSHAを
凍結し、その後で初めてv2.5.0 pathを評価configへ入れる。

### 16.1 taya36開発対局

既存条件と直接比較できるよう、両者1 thread、jobs 1、5秒/手、50局、seed `20260816`に固定する。

```bash
NEW_WEIGHTS=BEST_SCRATCH_MODEL \
BASELINE_WEIGHTS=/home/kuro/shogi_rust/policy_weights_halfkp64_kpp_distilled_v2.5.0.binary \
POSITIONS=/home/kuro/shogi_rust/taya36.sfen \
GAMES=50 \
TIME_LIMIT_MS=5000 \
DEPTH=32 \
MAX_PLIES=256 \
SEED=20260816 \
THREADS=1 \
JOBS=1 \
OUT_DIR=/home/kuro/shogi_rust/runs/halfkp64-scratch-v1/post-wdl-dense-vs-v250-taya36-5s-50g \
scripts/bench_halfkp64_learned_vs_v250.sh
```

| score | 次 |
|---:|---|
| `< 0.40` | release不可、Phase D |
| `0.40～0.50未満` | 進展だがrelease不可、Phase D |
| `= 0.50` | 同等止まり、Phase D |
| `> 0.50` | 2026 reserveへ進む |

taya36は開発済み集合なので、50%超だけでreleaseしない。

### 16.2 2026 reserve

fixed manifestから`sha256-seed` selector、seed `20260816`で25 unique familyを選び、先後交換50局を5秒/手で
行う。time-budget対応済みpaired wrapperを使い、v2.5.0には
`evaluation_only_external_baseline_v1` manifestと
`evaluation_only_external_baseline` parent authorizationを作る。このmanifestはmodel SHA
`91784d6f03f70580468a1509f968cbaaf598fbce084c22a739cb870344fb7c00`と
`allowed_usage=evaluation_only`だけを認可し、training configから参照できないようpreflight testする。

外部評価workflowがbest scratch、baseline manifest、candidate manifest、reserve manifestの全SHAを検証して
authorizationをatomic createした後、次を実行する。

```bash
python3 scripts/run_halfkp_paired_match.py \
  --binary LOCKED_HALFKP_PROMOTION_MATCH \
  --candidate BEST_SCRATCH_MODEL \
  --candidate-manifest BEST_SCRATCH_MODEL_MANIFEST \
  --parent policy_weights_halfkp64_kpp_distilled_v2.5.0.binary \
  --parent-manifest V250_EVALUATION_ONLY_MANIFEST \
  --parent-authorization EXTERNAL_EVALUATION_AUTHORIZATION \
  --openings training/halfkp64-scratch-v1/final_openings_seed20260730_2000.jsonl \
  --openings-format dataset-jsonl \
  --opening-selection sha256-seed \
  --opening-seed 20260816 \
  --pairs 25 \
  --time-limit-ms 5000 \
  --max-depth 64 \
  --max-plies 256 \
  --output-dir runs/halfkp64-scratch-v1/post-wdl-dense-vs-v250-reserve-5s-50g \
  --bootstrap-resamples 100000 \
  --bootstrap-seed 20260823
```

wrapperは`selected_openings.manifest.json`へ25 source line/family/SHAとselection seedを保存し、再開時に完全一致を
要求する。`first-unique`へfallbackしない。

reserveのopening-pair bootstrap 95%下限が`> 0.50`の場合だけ、v2.5.0超えのrelease候補とする。それ以外は
releaseせずPhase Dへ進む。通常の50局score区間は補助値に限定する。

## 17. Phase D trigger

次のどれかでPhase Dを開始する。

- dense dataset Gateを満たせない。
- unseen pilotがouter 4を改善しない。
- dense outer 1/2でaccepted候補が増えない。
- best scratchのtaya36 scoreが50%以下。
- taya36は50%超だが2026 reserveの下限が50%を超えない。

trigger時は上位計画Section 7に従い、`halfkp_search_teacher --value-only`がnode cap時にlast fully completed
iterative-deepening resultを保存できる最小修正だけを計画・実装する。最初の規模は10,000 train / 2,000 valid、
200,000 nodes/root、completed depth 4以上、最大2 epochである。A～Cの結果が出る前にself-play、TDLeaf、
TreeStrap、policy head、MCTSを実装しない。

## 18. 進捗表

状態は`未着手`、`作業中`、`完了`、`条件不成立`のいずれかにする。完了時はcommitまたはartifact SHAを備考へ
追記する。

| ID | 作業 | 状態 | 完了条件 / 記録 |
|---|---|---|---|
| H0 | 上位計画 | 完了 | `7caae5d` |
| H1 | 本実装・実験手順書 | 完了 | 本書 |
| P0 | 旧WDL trainer保全 | 完了 | SHA `329e32f5...439f`、run専用0555 copy |
| I1 | WDL別正規化・sampler基盤 | 完了 | MMTO/WDL分離、固定manifest、数学test |
| I2 | WDL/MMTO game-cluster audit | 完了 | root_key、exact-root、game bootstrap test |
| I3 | 50k milestone resume | 完了 | active checkpoint、snapshot、current/previous復旧、build/test |
| I4 | frozen-eval dense dataset tool | 完了 | base/eval凍結引数、prepare 4 subcommand、duplicate Gate、Rust test/build |
| I5 | paired time/authorization | 完了 | nodes/time排他、registry/authorization排他、USI sha256 selector、fixed/time smoke。report SHA `7946bb315a9ceefaacbf136aaee61adf398a889694117c16b5dd877fe0e4e3c3` / `a0d38f06de0e5892cc92ac58d8e5d98ae838bba299ac82a48aa7b650f49680fe` |
| I6 | unattended config v2/state machine | 未着手 | E2E test |
| I7 | implementation ready review | 未着手 | full test、tiny resume report |
| I8 | 最終config/runbook/input SHA lock | 未着手 | clean worktree、以後tracked変更なし |
| A0 | WDL pool / valid split | 完了 | pool 100,000 roots / 39,531 games、valid 1,000/5,099。pool manifest SHA `200d4e1a3c6713b9c5d3c74fd1087eea8bea05b4080cb889e9b7c33a5312c55e` |
| A1 | kappa / norm校正 | 条件不成立 | kappa `4999.9976` が上限到達、calibration Brier `0.4741644350`、objective decisionはMMTO-only |
| A2 | WDL-off replay | 未着手 | model/optimizer byte一致 |
| A3 | corrected treatment | 条件付き | archive完了 |
| A4 | corrected WDL Gate | 完了・MMTO-only | kappa境界到達のためhybridを採用せず、dense MMTOへ進む |
| B0 | recent dense build | 未着手 | full membership/interleave |
| B1 | old supplement | 条件付き | recent 2m未満時だけ、最大25% |
| B2 | dense dataset Gate | 未着手 | 1.8m/new 1.2m、全衝突0 |
| B3 | unseen 250k pilot | 条件付き | audit + 250-pair Gate |
| C1 | dense outer 1 | 条件付き | archive/audit/match/decision |
| C2 | dense outer 2 | 条件付き | outer 1 accepted時だけ |
| F0 | best scratch凍結 | 未着手 | `best-scratch.json` |
| E0 | taya36 5秒50局 | 条件付き | best凍結後のみ |
| E1 | 2026 reserve 5秒50局 | 条件付き | taya score >50%時だけ |
| D0 | deep-search pilot手順書 | 条件付き | Phase D trigger時だけ |

## 19. 実験終了後のGit記録

serviceが`completed`または`stopped_by_gate`で終了し、child processがないことを確認してからtracked fileを更新する。
長期run中の進捗は`status.json`とimmutable markerへ保存し、途中commitのためにrevision lockを壊さない。

commit対象:

- Phase A objective decisionの小さいsummary
- dense datasetのmanifest要約、件数、collision/audit report
- pilot/outerごとのmodel SHA、audit/match/Gate decision
- `best-scratch.json`のtracked summary
- 本書の進捗表

model、optimizer、巨大JSONL、pair個別artifactは`runs/`に残しcommitしない。summaryは元artifactのpath/SHAを
持たせる。結果記録commitの例は
`training: record post-WDL dense experiment results`とする。release候補になった場合だけ、別のrelease手順で
weight配布、release note、tagを行う。

## 20. 次に行う唯一の作業

次回「計画に従って実装」と指示されたら、最初にStep P0を実行する。その後I1から順に進める。
P0より先にrelease buildを行わず、I8完了前に本番dataset buildや学習を開始しない。
