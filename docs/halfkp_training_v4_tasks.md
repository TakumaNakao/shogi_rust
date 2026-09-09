# HalfKP v4 実装・実験タスク

- 作成: 2026-09-09
- 方針の正本: [学習方針v4](halfkp_training_strategy_v4_20260909.md)
- 本書は実行順、完了条件、進捗を管理する。実験結果・対応commit・成果物を追記する。
- 現在の次タスク: **T06の分岐判定**。通常の実装判断はこの手順内で進め、条件変更・不足・異常は記録する。

## 共通条件

親・初期値・教師はaccepted gen5 `61c05fe287cb0c0f9b434194802e07c835476b737a37bb9d4063143c63e28637`。
親モデルは`runs/halfkp64-scratch-v1/phase3-2m-preserved-target-20260825/model-2m-100k.bin`。
外部AI評価値、解析PV、既存KPP由来重みは教師にしない。全rank weight=0の価値学習、12 workerを基本とする。

成果物は新しい`runs/halfkp64-scratch-v1/v4-value-gen5-<開始日>/`へ保存する。日付は実行日に決め、再開時は変えない。
以下の成果物名は作成予定。旧run・凍結済みモデルは上書きしない。
コードは現行trainingブランチで、修正・データ処理・実行制御を分けてコミットする。
大容量棋譜・重みはGitへ追加せず、コマンド、commit、SHA、件数、結果と本表の進捗を記録する。

## タスク一覧

| ID | 作業 | 依存 | 状態 |
|---|---|---|---|
| T01 | rank勾配の符号修正と回帰テスト | — | 完了 |
| T02 | 外部評価不使用・親モデル・教師形式の検証 | T01 | 完了 |
| T03 | 棋譜棚卸し、未使用game・holdoutの確定 | — | 完了（候補不足を確認） |
| T04 | 5k rootでgen5教師の再解析診断 | T02 | 完了 |
| T05 | 長時間実行・checkpoint・評価出力の最小整備 | T02、T04 | 実装完了・本番未実行 |
| T06 | 新規棋譜の抽出、不足時の追加取得 | T03、T04 | 完了（高rate不足・rate欠損分岐を診断） |
| T07 | 本教師生成、重複除去、学習データ凍結 | T05、T06 | 教師生成中（重複検査待ち） |
| T08 | 16 epoch学習、条件付き32 epoch | T07 | 未着手 |
| T09 | 候補1個の選定・凍結・test評価 | T08 | 未着手 |
| T10 | 親との250 pair screen | T09 | 未着手 |
| T11 | 独立250 pair確認・昇格判定 | T10・有望時 | 条件付き |
| T12 | v2.5.0比較・結果記録・次段階決定 | T10またはT11 | 未着手 |

### T01 — rank勾配の符号を直す

- 対象: `src/training_tools/halfkp_search_train.rs`の`add_rank_pair`と既存テスト。
- good/bad各leafの手番をrootと比較し、utilityと勾配に同じ符号規則を使う。
- root先後、good/badが同手番・逆手番となる組合せで有限差分を確認する。value-onlyの更新結果が修正前と変わらないことも確認する。
- 完了: 前回失敗したrankテスト、valueの勾配・scale・resumeテスト、専用MMTOテストが通る。失敗をtest側の期待値変更だけで隠さない。

実績（2026-09-10）: `add_rank_pair`でroot側utilityからchild側raw scoreへ戻す符号を修正し、同手番・逆手番の有限差分テストを通過させた。value経路は変更していない。`RUST_FONTCONFIG_DLOPEN=1 cargo test --locked --features halfkp64,training-tools --lib training_tools::halfkp_search_train::tests`（22件）および全libテスト（131件）を通過。対応commitは後続のT02変更とまとめて記録する。

### T02 — 学習入力の出典を保証する

- 対象: `src/training_tools/dataset_build.rs`、`halfkp_search_teacher.rs`、`scratch_lineage.rs`、必要な既存テスト。
- 同じ棋譜の評価コメントだけを削除・変更したfixtureで、盤面・指し手・勝敗と生成targetの内容が同じことを確認する。
  ファイルSHA・provenanceは変わってよい。外部コメントを使ったfilterも禁止する。
- 起動前に親SHA、accepted registry、親Gate report、semantics v4を検証する。不合格候補・旧教師形式へのfallbackはしない。
- 完了: 上記の回帰確認と親の検証が通り、実行条件を教師manifest（T05で学習configへ継承）へ保存できる。

実績（2026-09-10）: `halfkp_search_teacher::InputRecord`へ`deny_unknown_fields`を適用し、既知のWDoorメタデータ（license/source URIを含む）は受理し、`eval_cp`等の外部評価フィールドは拒否する回帰テストを追加した。accepted gen5親（model SHA `61c05fe…`、registry世代5、Gate report SHA `3e8ed317…`）を指定した16 rootスモークを実行し、16/16件生成・parse/search失敗0・重複0を確認した。出力manifest [`teacher-16.hkst.manifest.json`](../runs/halfkp64-scratch-v1/v4-t02-smoke-20260910/teacher-16.hkst.manifest.json) に親manifest SHA、モデルSHA、semantics v4、検索条件、入力SHA、出力SHAが保存されている。コメント削除fixtureはInputRecordが外部評価フィールドを拒否する回帰で代替確認し、学習configはT05で実装する。

### T03 — 新規棋譜と評価集合を分ける

- 既存manifestから実際に使用したtrain game、旧valid/test game・opening family、使用済み対局openingを集計する。
  ファイルを走査しただけの棋譜と、実際に学習した棋譜を区別する。不明な旧用途は保守的に新規集合から除外する。
- `data/wdoor/extract/2018`～`2026`を確認し、未使用gameの年別件数・両者rate>=4000の受理見込みをまとめる。
- 新規gameのtrain/valid/test割当てと、screen用250 pair・確認用250 pairの互いに独立したopening集合を学習前に確定する。
  過去のholdoutは引き続き保護し、同じgameやopening familyを学習へ移さない。
- 完了: `data/inventory.json`、用途別game一覧、`openings/`のmanifestが揃う。抽出・重複判定には既存コードを再利用し、集計器を大規模化しない。

実績（2026-09-10）: 年別ファイル数、phase3で実際にハッシュ化できた入力数、学習済みgame/record数を [`v4_wdoor_inventory_20260910.json`](../training/halfkp64-scratch-v1/v4_wdoor_inventory_20260910.json) に固定した。`dataset_build --exclude-base-train --freeze-base-eval`で過去train/valid/testのgame・opening familyを除外した結果、2018–2026の独立候補は80,369 rootのみだった。T03の棚卸し・holdout保護は完了し、1M不足のためT06でWDoor 2017から追加取得へ進む。

### T04 — 5k rootで教師更新を診断する

- 既存gen4教師に対応するtrain由来5,000 rootを固定seedで選ぶ。gen4/gen5の対応はrootの内容で確認し、単純な行番号対応に依存しない。
- gen5、100k nodes/root、depth上限8、最低completed depth1、`--value-only --qsearch-leaf`で再解析する。
- 同じroot手番に揃えて探索値・best手の変化、completed depth、勝敗Brier、除外理由、root/秒、保存容量を比較する。
- exact leaf、手番反転、completed root score保持を確認する。教師の数値が変わっただけでは改善と判定しない。
- 完了: `diagnostic-5k/report.json`に対応件数と結果、本生成の時間・容量見積もりを保存。小規模対局Gateは実施しない。

実績（2026-09-10）: 固定`pilot-5k/root-0000.jsonl`をgen4/gen5でposition hash照合し、gen5を100k nodes/root・depth 8・qsearch leaf・12 workerで再解析した。対応4,930 root、gen4 4,934件、gen5 4,933件。gen5の勝敗Brierは0.22831（gen4 0.23109）、log-lossは0.64813（0.65373）で、同じ固定局面では較正が小幅改善した。common rootの29.8%（1,468件）が100cp超変化し、再ラベルは実質的な更新である。一方、67 rootは失敗/資源制限/重複で欠落したため、T06/T07で余裕を持って抽出する。詳細は [`v4_t04_diagnostic_5k_report_20260910.json`](../training/halfkp64-scratch-v1/v4_t04_diagnostic_5k_report_20260910.json)。これは対局Gateではなく、gen5再ラベルを進めるためのオフライン診断である。

### T05 — 再開と評価の仕組みを整える

- 再利用元: `scripts/run_halfkp_phase3_2m_preserved_target.sh`、`halfkp_search_train.rs`、`scripts/run_halfkp_paired_match.py`。
  v4用の起動スクリプトを1本用意し、旧gen4固定値・古いrun名を外す。既存run用scriptは変更して再利用しない。
- 学習はepoch終了ごとに機械可読の指標を保存し、4/8/16/32の重みを保管する。16→32はoptimizer・epoch番号・shuffle系列・過去bestを引き継ぐ。
- `--save-last-epoch`で最終epochをbestと誤認しない。今回SWAは使わず、最大32 epochより後に開始する設定へ固定する。
  `output-limit=0.20`、clip norm=1等も成功したvalue経路と揃える。既定の小さいoutput上限へ戻さない。
- 親と各checkpointを同じvalidで測定し、混合target二乗誤差、真の勝敗Brier/log-loss、phase別誤差、
  更新norm・活性・parameter deltaを保存する。順位専用の`halfkp_search_objective_eval`をvalue評価器として流用しない。
- 50k root shard単位で生成を再開し、学習は最後の完全なepoch checkpointから再開する。
  中断epochの再実行は許容する。直前の完全なcheckpointを残し、不完全な更新を完了扱いしない。
- 完了: 小さなscratch fixtureで連続実行と分割再開が一致し、失敗時の後続処理停止・二重起動防止・端末切断後の継続を確認する。
  本番開始前にbinaryを凍結する。新しい汎用ジョブ管理frameworkは追加しない。

実績（2026-09-10）: [`run_halfkp_v4_value_gen5.sh`](../scripts/run_halfkp_v4_value_gen5.sh)を追加した。accepted gen5固定、50k shard、教師manifest再利用、部分artifact上書き拒否、4/8/16の段階checkpoint、条件付き32 epoch再開、`config.json`、CSVへのmodel/teacher勝敗Brier・log-loss保存を実装。`halfkp_search_train`の既存bit-exact checkpointテストと新しい指標列のビルドを確認した。本番データでの連続/分割再開fixture確認はT07入力確定後に実施する。

### T06 — 新規棋譜を集めて抽出する

- まず手元の未使用棋譜を使い、不足時は2017→2016→…と1年ずつ取得する。取得元・SHA・受理件数を記録する。
- 両者rate>=4000、原則1棋譜24 root以下。両手番・敗者の終盤を含める。低rateの混入や同一棋譜の密度増加で不足を埋めない。
- train新規1M、replay 250k、valid 50k、test 50kを生成後に確保できるよう、T04の除外率を使って余裕を持って抽出する。
- 完了: 既存`dataset_build`・shard処理で候補rootを保存。必要数が確保できない場合は不足年・件数を記録し、長期学習へは進まない。

実績（2026-09-10）: `dataset_build`に`--exclude-base-train`を追加し、旧trainを含む全旧splitのgame/familyを除外できるようにした。全ローカル年のcapacity probeは80,369 rootで不足と判定した。WDoor公式アーカイブ（[公式一覧](https://wdoor.c.u-tokyo.ac.jp/shogi/)）から2017年（SHA256 `31a43e57…`）と2016年（SHA256 `ea2e9919…`）を取得・展開した。旧split除外後の受理数は、2017年が高rate条件で1,362 root、2016年が0 rootだった。一方、2017年はrateメタデータ条件を外した場合に1,000,000 root（48,563局）を確保できた（[`v4_wdoor_inventory_20260910.json`](../training/halfkp64-scratch-v1/v4_wdoor_inventory_20260910.json)）。これは条件の黙った緩和ではなく、rateメタデータ欠損を含む独立分岐として隔離し、教師生成・学習へ進める前に高rate方針との差を記録して採否を判定する。

T06分岐記録（2026-09-10）:

- 高rate（両者rate>=4000）: 2018–2026の独立80,369 root、2017の1,362 root、2016の0 root。計画した新規1Mを満たせないため、この条件のままでは本番学習へ進めない。
- rateメタデータなし: 2017年の48,563局から独立1,000,000 rootを得た。WDoor古年分では対局者名にレートが記録されない形式が多く、これは実力が低いことの証明ではないが、選別不能なデータを含むため高rate集合と混ぜない。
- 次の判定: rateメタデータなしの1Mを直ちに本番教師へ投入せず、同じgen5教師で小規模な教師品質・結果較正を確認する。問題がなければ、この分岐を新規1Mとしてreplay 250kと組み合わせ、T07の重複除去・凍結へ進む。問題があれば2017分岐を不採用として、年をさらに遡る前に不足と原因を記録する。

実績（2026-09-10）: rateメタデータなし2017年の固定5,000 rootをgen5・100k nodes/rootで再解析した。4,756件を出力（search_failed 137、resource_limited 107、duplicates 0）。教師の勝敗Brierは0.202751、log-lossは0.589687で、opening/middlegame/endgameのphase値も有限だった。詳細は [`v4_t06_2017_norate_diagnostic_20260910.json`](../training/halfkp64-scratch-v1/v4_t06_2017_norate_diagnostic_20260910.json)。高rateサンプルとは母集団が異なるため優劣とは判定せず、分岐を条件付き採用した。

T07入力を`runs/halfkp64-scratch-v1/v4-value-gen5-20260910/raw/`へ固定した。新規2017分岐1,000,000 root＋既存raw trainからのreplay 250,000 root、凍結valid/test（各57,594/52,127）を含む。教師生成はgen5・semantics v4・100k nodes/root・12 workerで開始済み。教師shard完了後にnetwork input重複を検査し、重複があれば学習前に除外する。教師生成中は学習を開始しない。

### T07 — 教師を生成し、実際の学習入力を確定する

- 全splitとreplayをgen5で再解析する。生成条件はT04と同じ。50k rootごとにmanifest付きshardを確定する。
- root重複だけでなく、学習対象leafのnetwork入力を全shard間で重複除去する。
  新規/replay間も重複させず、同じ入力へ異なるtargetがあれば件数と採用規則を記録する。
- 既存`halfkp_search_split_filter`はtrainとvalidの重複除去用であり、train内部dedupまで完了するとは仮定しない。
  不足機能だけ補い、valid/test相互も検査する。holdoutの優先順位・抽出順を固定し、モデル性能を見てデータを選ばない。
- 完了: **重複除去後の学習対象入力が新規1,000,000＋replay 250,000**、valid/test各50,000。
  不足時はT06へ戻る。`data/final-manifest.json`にraw数・受理数・unique入力数・game数・除外理由・全SHAを記録する。

### T08 — 16 epoch、必要なら32 epochまで学習する

- 固定条件: AdaGrad、feature LR `3e-4`、output LR `3e-5`、batch128、search/result=`0.8/0.2`、
  kappa=`1352.7871`、loss power=`2.5`、全rank weight=0、12 worker。途中でLRやデータを差し替えない。
- 原則16 epochを実施。微小な横ばいで既定patience=2により打ち切られないよう、停止を下記規則へ揃える。
  non-finite、非合法、artifact不一致は即停止。旧delta目安超過は診断待ちとし、自動で上限を緩和しない。
- 4/8/16を判定点とする。混合target誤差と勝敗Brierが、両方とも直前判定点から`1e-4`超悪化する状態が2回続けば終了する。
- 16→32の条件は、16の混合target誤差が8より`1e-6`以上改善し、勝敗Brier/log-lossが8よりそれぞれ`1e-4`超悪化しておらず、healthが正常であること。
  32への継続中もepoch単位で保存・検証する。上記の数値は事前に固定する運用上の許容差で、統計的有意差ではない。
- 完了: 最終到達epoch、実提示数、停止/継続理由、全checkpointの指標を`training/`へ保存する。

### T09 — 対局するモデルを1個に凍結する

- 初期親も含む安全なcheckpointから、固定validの混合target二乗誤差が最小のものを選ぶ。同値なら早いepochを選ぶ。
- `selection.json`へ選定理由、勝敗指標、health、モデルSHAを保存する。親が最良なら新候補なしとしてT12へ進む。
- 新候補は`candidate/`へ凍結し、独立testを一度評価する。testや対局の結果を見て別checkpointに選び直さない。
- 完了: 選定した1個のモデル・manifest・test reportが揃う。testで破損や非finiteが見つかれば対局を開始しない。

### T10 — 親とのscreenを実施する

- T03で凍結した未使用250 opening pair、先後入替500局、20k nodes/手でgen5と対局する。
- pair単位の保存・再開、勝敗数、paired bootstrap CI、安全違反を既存runnerで集計する。
- score>=52.5%、安全違反0ならT11へ進む。screenのCI下限が50%以下でも独立確認へ進んでよい。
  それ以外は昇格せずT12へ進む。途中scoreで対局数や候補を変更しない。
- 完了: `match-screen/report.json`と完了記録が揃う。

### T11 — 独立確認と昇格判定

- 別の未使用250 opening pair＝500局で同じ候補・同じ親を比較する。
- **確認側単独でscore>=52.5%、paired 95%下限>50%、安全違反0**なら昇格する。screenとの合算で条件を通さない。
- 確認reportのSHAとモデルSHAを照合してaccepted registryへ登録し、決定をコミットする。不合格ならgen5を維持する。
- 完了: `match-confirm/report.json`と`decision.json`。理由・使用opening・registry更新の有無を記録する。

### T12 — v2.5.0比較と次段階を決める

- 昇格した候補のみ、taya36、両者5秒、50局でv2.5.0と比較する。これは学習教師・checkpoint選定には戻さない。
- `result.md`へ学習量、独立game数、offline、親対局、v2.5.0対局、所要時間をまとめ、本書とv4の進捗を更新する。
- 次の分岐は以下に固定する。50局の僅かな優位だけでreleaseはしない。

| 結果 | 次タスク |
|---|---|
| 親改善を独立確認できた | 新accepted親で教師更新・データ拡張。次回の規模を記録してから開始 |
| CIが50%をまたぎ、validの混合target誤差が親より`1e-6`以上良く、勝敗Brier/log-lossは親より`1e-4`超悪化せず、安全正常 | 同じgen5・設定で総train 2Mへ一度拡張。新規1.6M＋replay 0.4Mとし、親の昇格とは区別 |
| 明確な劣化、offline悪化、安全異常 | 原因を診断し、v4の原因別分岐から変更を一つ選んで記録。自動増量しない |
| 上記の改善・増量条件を満たさない | 今回を不採用として結果を整理。方式全体の飽和とは断定しない |

規模の呼称を統一する。初回は**新規1M・総train 1.25M**、次は**総train 2M**。
以後の4M/10Mもreplayを含む総unique train数とし、原則80%新規＋20%replayの内訳を別記する。
2Mで親改善を確認できるまで4M/10Mへ自動拡張しない。自己対局・factorization・LR比較は今回の必須実装に含めず、診断後の条件付きタスクとする。

## 実装時の検証と記録

まず次の既存テストを使用し、追加した箇所だけ必要な回帰テストを足す。

```bash
RUST_FONTCONFIG_DLOPEN=1 cargo test --locked --features halfkp64,training-tools --lib training_tools::halfkp_search_train::tests
RUST_FONTCONFIG_DLOPEN=1 cargo test --locked --features halfkp64,training-tools --lib training_tools::halfkp_mmto_train::tests
```

進捗は各行を`未着手→実行中→完了/条件未達/異常停止`へ更新し、対応commit・成果物の場所を追記する。
今回の文書作成をT01以降の実装完了として扱わない。
