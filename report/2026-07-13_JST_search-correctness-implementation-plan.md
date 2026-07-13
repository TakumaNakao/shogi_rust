# 探索の正確性改善: 実装計画と評価計画

- 作成日: 2026-07-13 JST
- baseline: `v2.4.4` (`f79ec42`) + weight SHA-256 `8d2ad6ebd65afd9bdd921f7c03205582421f00cbe2c83ccbda984fbbe13747b3`
- 制約: 他将棋エンジンの評価値、指し手分布、PVは学習にも評価oracleにも使用しない
- 使用可能: 公開棋譜、自己対局、現行エンジン自身の探索、対局結果、ルールに基づく詰み証明
- リリース前評価: 自動生成した証明付き固定suiteと自己対局だけで完結させる

## 目的

実戦で観測された次の二症状を、重み変更より先に探索側で解消する。

1. 一見すると相手の駒損だが、その犠牲が詰みに直結している筋を見落とす。
2. 同じ地点への歩打ちと捕獲を繰り返し、盤面をほぼ戻しながら持ち駒だけを失う。

この二つを単なる「探索が浅い」でまとめない。前者は王手中qsearch、SEE pruning、詰み証明の問題、後者は完全同一局面ではない資源損失サイクルと水平線効果の問題として扱う。

## コードから分かる直接原因

### 王手・犠牲・詰み

現行 `quiescence_search_internal` には次の性質がある。

- 王手中でも静的評価をstand-patとして使う。
- qsearch候補は捕獲または王手だけで、quietな王手回避を落とす経路がある。
- `see(mv) < 0` の手を無条件にskipする。
- 現在のSEEは交換列ではなく、単に `victim_value - attacker_value` である。

したがって「駒損に見える王手」「大駒を捨てる詰み」「負SEEだが応手を強制する手」をqsearch境界で捨て得る。bench failure miningでも大regret局面の約6割が王手中で、合法手4-5手の詰み級例が記録されている。

### 歩を捨て続ける系列

現行千日手検出は盤面と持ち駒を含む完全局面hashが4回一致した場合だけ発火する。歩を打って取られるたびに自分の持ち歩が減り、相手の持ち歩が増える系列は完全同一局面ではないため、千日手ではなく検出対象外である。

さらに連続王手判定は、4回目の局面で `position.in_check()` かどうかだけを見ている。実際に同じ側が連続して王手していたかを履歴で証明していないため、資源サイクル実装の前に規則判定を厳密化する。

## 実装原則

1. 一つの候補で複数の探索規則を変えない。
2. 正しい低速参照実装を先に作り、同じ探索結果を保つ高速版を後から作る。
3. path依存の判定結果を通常TTへ保存しない。
4. mate、千日手、資源損失を通常cp評価と混ぜず、内部では明示的な終端種別を持つ。
5. 20局や40局の勝率だけで採用しない。症状別固定セットを必須gateにする。

## Phase 0: 再現局面と決定論的計測

### 0-A. 公開棋譜と自己対局から証明付きsuiteを自動生成する

人間対局棋譜には依存しない。局面源は次に限定する。

- Wdoor 2023-2025: 開発用mine pool。
- Wdoor 2026: 最終holdout pool。開発中の個別局面調整には使わない。
- 現行baselineの自己対局: 現エンジンが実際に到達する分布。
- `converted_records2016_10818.sfen` と `taya36.sfen`: 既に繰り返し観測済みなので開発用だけに使い、holdoutには入れない。

source game単位のhashでsplitし、同一棋譜や同一SFENがdev/holdoutをまたがないようdedupeする。

成果物:

- `data/search_quality/generated/dev_mate_sacrifice.jsonl`
- `data/search_quality/generated/holdout_mate_sacrifice.jsonl`
- `data/search_quality/generated/dev_quiet_evasion.jsonl`
- `data/search_quality/generated/holdout_quiet_evasion.jsonl`
- `data/search_quality/generated/dev_resource_cycles.jsonl`
- `data/search_quality/generated/holdout_resource_cycles.jsonl`
- `data/search_quality/generated/suite_manifest.json`

manifestにはgenerator commit、weight SHA、入力棋譜hash、seed、抽出条件、重複除外数、各出力SHA-256を記録する。一度凍結したholdoutは、探索候補ごとに作り直さない。

### 0-B. node制限探索を実装する

時間制限だけではCPU負荷と温度で比較が揺れる。`SearchLimits` に `node_limit` を追加し、USI `go nodes N` と `search_profile --nodes N` を実装する。

対象:

- `src/ai.rs`: node limitと停止理由
- `src/usi_shogi.rs`: `go nodes`
- `src/bin/search_profile.rs`: 固定nodes profile

固定depth、固定nodes、固定timeを別々に測り、同時指定時の優先規則もテストする。

### 0-C. root探索traceを構造化する

標準出力の文字列解析ではなく、探索結果を構造体で返せるようにする。

```text
SearchReport {
  best_move, score, pv, completed_depth, nodes, qnodes,
  terminal_mates, in_check_qnodes, negative_see_check_searches,
  repetition_hits, resource_cycle_hits, stop_reason
}
```

新しい `search_failure_probe` は、同じSFENをdepth `3..8`、nodes `10k/50k/200k` で読み、最善手、PV、scoreの変化をJSONLへ出す。

### 0-D. 評価関数を使わないsuite minerを作る

`search_failure_probe` とは別に、正解ラベルをルールだけで作る三つのminerを用意する。

1. `mate_sacrifice_miner`
   - 各局面の合法王手から、現行単純SEEが負になる手を列挙する。
   - 深さ上限1/3/5/7 plyの全幅AND/OR探索で強制詰みを証明する。
   - 証明手順、詰み手数、攻方初手、全防御数を保存する。
   - node上限に達した探索は `Unknown` とし、正例にも負例にも入れない。`ProvenNoMateWithinHorizon` は指定深さを全幅で完走した場合だけ付ける。
2. `quiet_evasion_miner`
   - 王手中局面から、非捕獲・非王手の合法な玉逃げ/合駒/移動合いを含む局面を抽出する。
   - 全合法回避集合そのものを正解ラベルとして保存する。
3. `resource_cycle_miner`
   - sampled positionから深さ4/6/8の制限付きDFSを行う。
   - 同じ盤面key・同じ手番へ戻り、一方の持ち駒がcomponent-wiseに減り相手が増えた手順を証明列として保存する。
   - 初期段階は歩打ち、捕獲、直前位置へ戻る手を優先して探索するが、出力前に全手順の合法性を再検証する。
   - 制限付きDFSで見つからなかった局面を「cycleなし」とはラベル付けしない。suiteには合法性を再検証できた正例と、その正例に対する探索の選択結果だけを収録する。

この全幅詰み探索はsuite生成用の遅いoracleであり、KPP評価値や他エンジンを使わない。Phase 5では同じ証明結果を保ったまま、実戦時間内で使える詰み探索へ高速化する。

完了条件:

- 同じcommit、weight、seed、node数ならbest move、score、PV、counterが再現する。
- dev/holdoutの両方に証明付きmate sacrifice、quiet evasion、resource cycleが生成される。
- 目標件数はmate sacrifice `dev>=200 / holdout>=100`、quiet evasion `dev>=500 / holdout>=200`、resource cycle `dev>=100 / holdout>=50` とする。
- 実棋譜由来で件数が不足する分類だけ、合法局面からの自動探索生成で補う。手作業で候補に都合のよい局面を追加しない。

## Phase 1: 王手中qsearchの正当化

### 1-A. 低速参照版

qsearchを二経路に分ける。

```text
if in_check:
    stand-pat禁止
    全合法王手回避を生成
    合法手ゼロなら詰み負け
    全応手を探索し、SEE pruning禁止
else:
    従来どおりstand-pat
    tactical moveを探索
```

まず `legal_moves()` を使う参照版を作る。速度が遅くても、固定局面の正解oracleとする。

### 1-B. 高速版

参照版と全く同じ手集合を返す専用evasion経路へ置き換える。既存 `generate_evasions` / `has_legal_evasion` を再利用し、TT move、捕獲、玉移動、合駒の順序を計測して調整する。

変更対象:

- `src/ai.rs`: `quiescence_search_in_check`
- `shogi_lib/src/movegen.rs`: 必要ならpublicな合法evasion API

採用条件:

- 王手中の全テスト局面で参照版とscore/best moveが一致する。
- quiet evasionが唯一手の局面を正しく読む。
- 全体 `search_profile` のelapsed悪化が10%以内、または同一時間でcompleted depthが低下しない。

## Phase 2: 負SEE王手を捨てない

Phase 1とは別候補として、qsearchのskip条件を次に限定する。

```text
if !gives_check && is_capture && see < 0:
    skip
```

王手、王手になる駒打ち、詰みが証明された手にはSEE pruningを適用しない。単純SEE自体の本格化はこの候補に混ぜない。

テスト局面:

- 大駒を捨てる1手詰め/3手詰め。
- 負SEEの王手を取ると詰む局面。
- 負SEE王手が単なる無駄王手で、探索爆発しやすい局面。

計測counter:

- `negative_see_checks_considered`
- `negative_see_checks_searched`
- `negative_see_check_mates`
- 追加qnodesと最大qply

採用条件:

- 犠牲詰み固定セットのsolve数が増える。
- qnodes増加p95を記録し、全体elapsed悪化15%以内。
- 200局screenで明確な悪化がない。

## Phase 3: 千日手規則の厳密化

`SennichiteDetector` の履歴要素をhashだけから次へ変更する。

```text
RepetitionEntry { hash, side_to_move, gave_check }
```

同一局面4回を検出した区間で、片側の全着手が連続王手だった場合だけ、その王手側の負けとする。それ以外は引き分けとする。結果も単なる `PerpetualCheckLoss` ではなく、どちらの負けかを表現する。

対象:

- `src/sennichite.rs`
- `src/ai.rs`
- `docs/sennichite.md`

テスト:

- 通常千日手。
- 先手の連続王手。
- 後手の連続王手。
- 途中に非王手を含む反復。
- 4回目だけ王手になっているが連続王手ではない反復。

これは棋力実験ではなく規則の正しさとして、全テスト通過後に採用する。

## Phase 4: 資源損失サイクル検出

### 4-A. まず検出だけ行う

探索pathに次を保持する。

```text
PathState {
  board_key_with_side,
  hands_black[7], hands_white[7],
  static_eval, move, gave_check
}
```

現在局面と同じ盤面配置・同じ手番の祖先があり、現在手番側の持ち駒がcomponent-wiseに減少し、相手側が同等以上で、少なくとも1枚の厳密な悪化がある場合を `ResourceLossCycle` とする。

初回候補では探索動作を変えず、次だけ記録する。

- cycle長
- 失った駒種と枚数
- 王手を含むか
- root PVに含まれるか
- 同じ駒打ち地点が反復したか

証明付きdev resource-cycle suiteで検出し、通常ベンチでの誤検出候補を確認してからcutoffへ進む。

### 4-B. 保守的cutoff

次の条件を全て満たす場合だけ、現在局面の静的評価を返してcycleを打ち切る。

- 完全に同じ盤面配置と手番。
- 自分の持ち駒はcomponent-wise subset。
- 相手の持ち駒はcomponent-wise superset。
- strict lossが1枚以上。
- cycle区間に王手がない。
- 現在局面が王手中でない。

これは「同じ盤面なのに選択肢である持ち駒だけを一方的に失った局面は、祖先より良くない」というdominance cutoffである。恣意的な「同じ手ペナルティ」は入れない。合法な連続王手や詰み筋を誤って抑えないため、王手を含む系列はPhase 5の詰み探索へ任せる。

重要:

- path依存なのでTTへ保存しない。
- 最初は歩だけで有効化し、金銀角飛へ一般化するのは実測後にする。
- static評価への追加二重ペナルティは加えない。持ち駒損は既存material/KPPに既に含まれる。

採用条件:

- dev/holdout resource-cycle suiteで同じ歩打ちの反復回数と持ち歩損失がゼロまたは明確に減る。
- 合法な歩の連打による詰み、連続王手、攻め継続の固定セットを壊さない。
- `resource_cycle_hits` のroot PV例を全件機械出力し、少なくとも初回100件を目視分類する。
- 400局評価で非悪化。

## Phase 5: オリジナルの短手数詰み探索

評価関数を使わない、ルールベースの詰み証明器を実装する。

段階:

1. 1手詰め検出: 合法王手を指し、相手に合法応手がなければ詰み。
2. attackerは王手だけ、defenderは全合法回避を読むAND/OR探索。
3. transposition、反復、node budgetを追加し、DFPNへ発展させる。

配置:

- 新規 `src/mate_search.rs`
- `Position` と合法手生成だけに依存し、KPP評価には依存しない。

探索への接続は二段階にする。

- root開始時に自分の詰みをnode制限付きで探索する。
- 通常探索のroot上位候補へ進んだ後、相手側の短手数詰みが証明される候補を棄却する。

「証明できなかった」を「詰みなし」と扱わない。結果は `ProvenMate / ProvenNoMateWithinHorizon / Unknown` に分ける。root候補確認は上位3手程度、合計node予算を思考時間の10-15%から始める。

採用条件:

- suite minerが証明した1/3/5/7手詰めで証明結果が正しい。
- 同一局面でnode budgetを増やした時に証明済み結果が反転しない。
- 通常局面でmate probe時間p95が予算内。
- 200局screen非悪化後、400局以上で採否。

## Phase 6: PVS

上記の正しさを固定した後でPVSを実装する。

- 最初の手はfull window。
- 後続手はzero window。
- alphaを超えた場合だけfull windowで再探索。
- PV node、mate、repetition、resource cycleの終端種別を維持する。

固定depthでbaselineとbest move/PVが一致する局面と、ordering改善によりnode数だけ減る局面を分けて測る。PVS単体で探索意味を意図せず変えた場合は採用しない。

## Phase 7: 保守的LMR

LMRは最後に行う。初回条件は狭くする。

- depth >= 3
- move index >= 4
- 非PV node
- 非王手、非捕獲、非成り、非TT move、非killer
- in-checkでない
- resource cycle関連手でない
- reductionは1 plyだけ
- fail-high時は必ずfull depthで再探索

犠牲手や詰み筋を再び水平線外へ追い出さないことを、mate sacrifice corpusで必須確認する。

## 評価計画

### Gate A: 正しさ

- `cargo test --all-targets`
- movegen/perft不変
- mate proofの正解率100%
- 千日手規則fixture 100%
- qsearch参照版と高速版の手集合・score一致

一件でも失敗したら対局ベンチへ進めない。

### Gate B: 症状別検索品質

各SFENを固定nodes `10k/50k/200k` と固定depth `4/5/6/7` で測る。

主要指標:

- mate sacrifice solve率
- 詰みを許すroot moveの選択率
- depth/node増加時のbest move安定率
- resource cycle検出率
- 同一地点への歩打ち回数
- cycle中の持ち駒純損失
- worst root score drop

採用候補は、対象症状を最低1つ明確に改善し、もう一方と既存 `taildrop_root_rescue.sfen` / `loss_in_check_low_reply.sfen` を悪化させないこと。

### Gate C: 性能

三種類を分けて測る。

1. 固定depth: node数、qnodes、elapsed、PV。
2. 固定nodes: elapsed、best move、score。
3. 固定time: completed depth、nodes、best move。

正しさ修正はnode数が増えても即棄却しない。ただし通常局面のelapsed悪化が15%を超える場合は、高速化を先に行い対局へ進めない。PVSは固定depth node削減を要求する。

### Gate D: 自己対局

外部エンジンは使わず、候補commit対固定baselineを同じ重みで比較する。

- opening: 学習・調整に使っていない500局面以上から先後入替。
- screen: 200局、100ms/手。
- confirmation: 400局以上、100ms/手。
- long-TC確認: 最低100局、1000ms/手。
- seedを固定し、候補ごとにopening集合を選び直さない。

最終採用はpaired SPRT相当で `Elo0=-5, Elo1=+10, alpha=beta=0.05` を基本とする。計算量が上限へ達した場合も、400局未満の点推定だけでは採用しない。

### Gate E: 凍結holdout

dev suiteと既存 `taya36` で候補と閾値を決めた後、Wdoor 2026由来holdoutを一度だけ評価する。

- mate sacrifice solve率。
- 詰みを許すroot move率。
- quiet evasion手集合の一致率。
- resource-loss move選択率。
- baselineが解けていた局面のregression数。

holdoutの個別局面を見て閾値を再調整した場合、そのholdoutは以後dev扱いとし、別source gameからtestを再凍結する。最終リリース判定では、集計値と失敗分類だけを確認し、個別局面へのパッチを行わない。

### リリース判定

「ルール上の不具合を直した」と「総合棋力が上がった」を分ける。後者としてリリースするには、次を全て満たす。

1. Gate Aを100%通過し、quiet evasionのholdout手集合一致率が100%である。
2. mate sacrifice holdoutのsolve数がbaselineより増え、baselineが解けた局面の未解決化と、詰みを許さなかったroot手から詰み許容手へのregressionが0件である。
3. baselineが資源を捨てたholdout subsetで、同じ地点への歩打ち回数または持ち駒純損失を80%以上削減し、正当な連打・連続王手fixtureを1件も壊さない。
4. 固定nodesで症状改善が再現し、固定timeで通常局面のcompleted depthとNPSに許容外の低下がない。
5. 同一重み・同一opening・先後入替の自己対局でSPRTが改善側 `H1` を採択する。最大局数でinconclusiveなら、「強くなった」とは判定せず追加対局または候補見直しとする。
6. 1000ms/手の長時間対局でも、短時間だけに依存した逆転悪化や特定openingへの偏りがない。

正しさだけを改善する候補は、自己対局がinconclusiveでも不具合修正版として別途リリースできる。ただし、その場合は総合棋力向上をリリース理由にしない。これにより、人間対局がなくても、証明可能な局所改善と統計的な総合改善を分けて判断できる。

### リリース後: 人間対局

人間対局はリリース前gateに含めない。全自動gateを通過してリリースした後に依頼し、次版の分布外failure miningとして扱う。記録可能な場合は問題手前後のSFEN、depth、nodes、score、PV、同じ駒打ちの反復回数を保存する。

## 実験順序と停止条件

| 順序 | 候補 | 次へ進む条件 | 停止条件 |
| ---: | --- | --- | --- |
| 0 | trace + node limit | 再現可能 | 同一条件で結果が揺れる |
| 1 | in-check qsearch | 参照版と一致 | quiet evasion/mate fixture失敗 |
| 2 | 負SEE王手を探索 | mate solve改善 | elapsed +15%超か通常局面悪化 |
| 3 | 千日手厳密化 | 全規則fixture通過 | 連続王手の勝敗誤り |
| 4 | resource cycle検出/cutoff | 歩捨て減少、誤検出なし | 正当な攻め筋を抑制 |
| 5 | mate search | 証明正確、予算内 | UnknownをNoMate扱いする挙動 |
| 6 | PVS | 固定depth node減 | PV/終端意味の不整合 |
| 7 | LMR | self-play通過 | 犠牲詰みを再び見落とす |

## 最初に実装する単位

最初の開発単位はPhase 0だけとする。

1. `go nodes` と停止理由。
2. `SearchReport` と追加counter。
3. 証明付きsuite minerと凍結manifest。
4. depth/node sweepを行う `search_failure_probe`。

ここで観測された実際のPVを確認してから、Phase 1とPhase 2のどちらが各失敗を直すかを測る。資源損失サイクルも、いきなりペナルティを足さず、Phase 4-Aの検出ログで仮説を検証してから探索を変える。
