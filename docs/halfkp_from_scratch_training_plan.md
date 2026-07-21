# HalfKP-64 重みゼロスタート学習方針

- Status: **proposed / implementation not started**
- 作成日: 2026-07-21
- 対象: v2.6.0以降のHalfKP-64評価関数
- 前提計算機: AMD Ryzen 7 PRO 4750U、8 core / 16 thread、RAM 14 GiB、GPUなし

関連文書:

- [`halfkp_performance_training_design.md`](halfkp_performance_training_design.md)
- [`halfkp_training_start_gate.md`](halfkp_training_start_gate.md)
- [`artifact_policy.md`](artifact_policy.md)
- [`refactoring_handoff_v2.5.4.md`](refactoring_handoff_v2.5.4.md)

## 1. 結論

「いきなり大量棋譜の最終勝敗だけを学ばせず、詰みや戦型など構造の明確な課題から始める」
という方向は、**条件付きで妥当**である。カリキュラム学習として、初期の不安定な最適化を
助け、学習器・評価尺度・特徴表現の誤りを小さなデータで検出できる。

ただし、次のように解釈し直す必要がある。

- 戦型名や囲い名に人手でcentipawn値を付けない。戦型はデータの層別化と網羅性確認に使う。
- 詰み局面へ`+MATE_SCORE`を回帰しない。ルールだけで証明した勝敗と、良い手・悪い手の
  順位制約として学習する。
- 終端局面そのものではなく、探索の葉に現れ得る非終端の詰み接続局面を主対象にする。
- 後段へ進んでも前段データを一定割合残し、破滅的忘却を防ぐ。
- 型の学習だけで完成とはしない。最終的な強さには、広い局面分布、棋譜の指し手と勝敗、
  新系列自身の深い探索、自己対局が必要である。

推奨する全体経路は次である。

```text
独立乱数初期化
  -> ルール由来の対称性・合法性
  -> 証明済み短手数詰み・詰み回避・短手数戦術
  -> 戦型別の指し手順位と最終勝敗
  -> 新系列自身による深い探索の順位教師
  -> 新系列内の自己対局・TD
  -> 全データを混合した収束学習
```

この計画では、現行のHalfKP/KPP重みを一切学習へ利用しない。既存モデルを改善するのではなく、
別の系譜を新たに作る。

## 2. 絶対条件: 現行重みを完全に隔離する

### 2.1 禁止事項

次の利用をすべて禁止する。

- 現行HalfKPまたはKPP重みを`--init`へ渡す。
- 現行重みの静的評価値、探索score、PV、最善手を教師にする。
- 現行重みから生成済みのKPP蒸留dataset、HKST教師、rank datasetを混ぜる。
- 現行重みで作ったcheckpoint、optimizer、SWAを引き継ぐ。
- 新系列の探索教師へ現行モデルを対戦相手、補助評価器、fallbackとして組み込む。
- 現行重みとの距離、相関、MAEを学習目的やearly stoppingへ用いる。

既存の次の経路は、そのままでは使用しない。

- `scripts/train_halfkp64_kpp.sh`
- `scripts/train_halfkp64_search.sh`
- `scripts/train_halfkp64_large.sh`
- `halfkp_kpp_train`が読むKPP label付きdataset
- 現行重みを親に持つ既存の`data/halfkp_search_learning_*`教師・checkpoint

ソースコード、局面表現、探索、棋譜parser、model codecは再利用してよい。禁止するのは学習済みの
数値と、その数値を使って生成した教師情報である。

### 2.2 系譜allowlist

ファイル名や配置だけでは、重みをrenameした場合の混入を防げない。そのためdenylistではなく、
新系列の系譜allowlistを使う。

新しい全model manifestへ次を記録する。

```text
lineage_id = "halfkp64-scratch-v1"
generation = 0, 1, 2, ...
parent_model_sha256 = null | <同じlineageの直前model>
initialization = "scratch_random_v1"
teacher_model_sha256 = null | <同じlineageの凍結model>
```

`generation=0`は親modelを持ってはならない。trainerとteacher generatorは、入力modelのmanifestが
同じ`lineage_id`へ到達できない場合に実行を拒否する。`--allow-legacy-*`のような迂回optionは
新pipelineには設けない。

### 2.3 比較対局の扱い

本計画中は、現行重みを比較相手としても起動しない。性能評価は、新系列の親子比較、固定された
ルール課題、棋譜holdout、探索性能で行う。

この制約下では「現行重み以上の棋力」を直接証明できない。将来、置換判断のために比較が必要に
なった場合は、最終候補を完全に凍結した後、学習へ結果を戻さない独立release acceptanceとして
別途明示承認を得る。比較結果を見て再学習するなら情報漏洩になるため、その候補は最終候補として
扱わない。

## 3. 「ゼロから」の定義

本計画のゼロスタートは次を意味する。

- model parameterは新しいseedから生成し、既存model fileを読まない。
- 教師はゲーム規則、証明済み探索、棋譜に記録された指し手・最終結果、新系列自身の探索・対局だけ。
- 他modelの評価値を模倣しない。
- 初代modelのparameterと全optimizer stateを新規作成する。

現行HalfKPは入力表現の一部として、ソースコードに定義された駒価値からmaterial scalarを計算する。
これは重みfileではなくmodel architectureの一部だが、初代modelではmaterial出力係数も`0`から始め、
教師データから更新する。これにより、既存評価器のmaterial係数を初期値として移植しない。

ゲーム規則以外を全く使わない厳密なtabula rasaは、棋譜の指し手・結果も利用できない。本計画は
前半をtabula rasa、後半をKPP非依存の棋譜学習とする。棋譜を完全禁止する実験は比較用の研究branch
に分ける。

## 4. 現行実装の調査結果

### 4.1 推論model

`halfkp64` featureのmodelは概ね次の構造である。

```text
HalfKP input: 45 king buckets * 2,344 piece states = 105,480 rows
feature transformer: 105,480 -> 64
black/white accumulator: 各64
activation: ClippedReLU [0, 1]
output: STM 64 + NSTM 64 + material 1 + bias
target scale: 1,000
```

feature embeddingだけで`6,750,720` parameterあり、f32 modelは約27 MBである。王位置は手番側へ
正規化され、盤の左右はmirror共有される。左右反転を単純に複製しても新情報が増えないため、
augmentationは正規化後のposition hashで重複排除する。

HalfKPは静的なvalue evaluatorであり、policy headを持たない。したがって、戦型の「次の手」は
合法手後局面のvalue順位として学ばせる必要がある。また、多手数の詰みは本質的に探索課題であり、
静的評価器だけへ完全な詰み手順を記憶させることはできない。

### 4.2 利用できる既存部品

| 部品 | 利用可否 | 判断 |
|---|---|---|
| `dataset_build` | 利用可 | content hash split、終局理由、結果既知、phase層別を再利用できる |
| `training_data` | 利用可 | CSA規約、重複排除、manifest作成に使える |
| `HalfKpFlatModel` codec/forward | 利用可 | runtimeとtrainerの同一forwardを維持できる |
| `halfkp_train` | 参考のみ | 乱数初期化できるが、逐次AdaGrad・勝敗専用で本学習には不十分 |
| `halfkp_search_teacher` | 改修後に利用 | 現在は入力重み必須。新系列modelだけを許可する必要がある |
| `halfkp_search_train` | 改修後に利用 | init必須かつsearch cp前提。汎用順位・証明labelへ拡張が必要 |
| `usi_benchmark` | 利用可 | 新系列の親子対局に使える |
| `shogi_lib`合法手生成 | 利用可 | 詰み証明器の規則実装の基礎になる |

### 4.3 不足しているもの

- manifest付きの独立乱数initializer
- 現行modelを受け付けないlineage検証
- 短手数詰みを完全に証明し、証明失敗と不詰みを区別するsolver
- 詰み、戦術、棋譜手、勝敗、自己探索を同一意味で表す汎用dataset schema
- 戦型・囲いを評価値ではなくsampling stratumとして扱う分類器
- HalfKP sparse rowの初期学習を助けるtraining-only factorization
- feature出現数、ClippedReLU飽和、score分布を記録する監視
- 新系列内の世代管理と、parentを破棄しないpromotion gate

### 4.4 過去実験から守るべき教訓

`halfkp_training_start_gate.md`では、棋譜最終結果だけの学習はholdout BCEを改善した一方、
2M局面の候補がKPPとの対局で大敗した。出力weightが約100倍へ膨らみ、探索尺度が壊れていた。
したがって、BCEや棋譜手一致率だけで次世代を採用してはならない。

既存の約1M局面search-teacher pipelineでは、教師生成が約1日、1,133,350 recordを8 epoch学習する
時間は約12分だった。CPU環境ではbackpropagationより探索教師の生成が支配的である。計算計画は
epoch数より、証明・探索するroot数とnode budgetを管理する。

## 5. 提案された「型から学ぶ」方針の妥当性

### 5.1 有効な部分

カリキュラム学習は、簡単な課題から難しい課題へ進めることで、収束速度や局所解を改善できる
可能性がある。HalfKPでは特に次の利点がある。

- 正解をルールで検証でき、誤labelを減らせる。
- 小規模datasetでforward、gradient、保存・再開、手番反転のbugを発見できる。
- 詰み、王手回避、成り、捕獲、駒打ちを各king bucketへ意図的に配置できる。
- modelが全く意味を持たない初期状態でも、自己評価を教師にせず学習を開始できる。
- rare featureへ早期にgradientを与えられる。

### 5.2 そのままでは危険な部分

#### 戦型へ固定評価値を付ける問題

同じ矢倉や美濃囲いでも、手番、駒組みの進行、相手の攻撃形、持ち駒で価値は変わる。
「穴熊だから+200」のようなlabelは、modelへ人間の先入観と誤った絶対尺度を固定する。

戦型は次にだけ使用する。

- 各戦型の局面数を均すsampling stratum
- ある戦型をtrain、別の棋譜をtestに分けるleak防止
- 戦型別のWDL、指し手順位、calibrationを報告するslice
- 自己対局開始局面を多様化するopening seed

#### 詰み値を静的評価へ回帰する問題

探索は終端詰みを有限のmate scoreとして別処理する。静的評価器へ巨大なmate scoreを学ばせると、
通常局面の尺度、aspiration、pruning marginを壊す。終端局面は通常のstatic eval対象でもない。

詰みから作る教師は次へ限定する。

- 非終端rootの`win/loss`確率target
- 証明済み勝ち手と証明済み非勝ち手のpairwise preference
- 詰まされる側では、すべて負けなら長く逃れる手の補助順位
- 固定node budgetでの詰み発見率を測るtest

`+MATE_SCORE`、`-MATE_SCORE`、未証明局面への推測labelは使わない。

#### 型だけの分布偏り

詰み問題は終盤・王周辺・持ち駒が多い分布へ偏る。先にそれだけを長く学ぶと、序中盤の大部分の
feature rowが未学習のままになり、出力層だけが詰み判定へ特化する。各段階で過去データと
中立anchorを混ぜ、stageを進めるたびに全体holdoutを確認する。

## 6. 他のゼロスタート手法の比較

| 方法 | 主な教師 | CPU適性 | 長所 | 主な問題 | 採否 |
|---|---|---:|---|---|---|
| 人手pattern回帰 | 人手評価値 | 高 | すぐ開始できる | 尺度と先入観を固定 | 主経路には不採用 |
| 証明済み戦術curriculum | 規則・完全探索 | 高～中 | 現行重み不要、labelが明確 | 分布が狭い | 初期段階に採用 |
| 棋譜最終結果学習 | 勝/分/負 | 高 | 安価で評価値模倣ではない | credit assignmentが長い | 混合signalとして採用 |
| Bonanza/MMTO型 | 棋譜手とminimax | 中 | 将棋で実績、value evaluatorと相性がよい | 探索生成が重い | 中核として採用 |
| search bootstrapping | 自身の深い探索 | 中～低 | 浅い評価へ探索知識を圧縮 | 自己誤りの再生産 | 世代gate付きで採用 |
| TDLeaf(λ) | 後続PV leaf・最終結果 | 中 | alpha-betaと直接結合できる | 不安定、実装検証が必要 | 小規模A/B候補 |
| Expert Iteration | tree search policy | 低 | planning/generalizationの反復 | policy headがなく計算量大 | 将来候補 |
| AlphaZero型 | MCTS自己対局 | 極低 | 完全tabula rasaが可能 | GPU/大規模計算前提、現architecture不一致 | 不採用 |
| 進化計算/SPSA | 対局結果 | 極低 | gradient不要 | 675万parameterには非効率 | 不採用 |
| 他model蒸留 | 他評価器score | 中 | 初期強度を得やすい | 今回の目的と矛盾 | 禁止 |

Bonanza式学習は、棋譜の選択手とminimax探索結果が一致するよう評価関数を調整する。将棋の大規模
parameterで実績があり、現行の合法手後局面rank lossと方向が合う。探索によるbootstrappingは、
small random weightsからself-playだけで評価関数を学んだ報告もある。ただし本repoでは、最初から
自己探索を信頼せず、証明済み課題で最低限の評価器を作ってから導入する。

AlphaZeroは規則だけから将棋を学べることを示したが、論文の「24時間」は大規模専用計算資源での
結果であり、ノートPC CPUの見積りには使えない。さらに現modelはpolicy headを持たず、alpha-beta
用の小さな静的評価器である。AlphaZeroを模倣するより、MMTO、search bootstrapping、TDLeafの方が
実装と計算資源に適合する。

## 7. 推奨する学習信号

model scoreを`e`、手番側勝率を`p`とする。

```text
p = sigmoid(e / kappa)
u(move) = -e(child(move))
```

### 7.1 結果loss

既知の最終結果`r in {0, 0.5, 1}`に対し、Brierまたは境界をclampしたBCEを使う。

```text
L_result = (p - r)^2
```

連続局面を独立sampleとして数えず、同一棋譜内の総weightを制限する。評価区間は棋譜単位bootstrap
で出す。

### 7.2 順位loss

良い手`g`、悪い手`b`について次を使う。

```text
L_pair = softplus((u(b) - u(g) + margin) / temperature)
```

証明済み詰みpair、棋譜手pair、自己探索pairを同じ数式で扱える。ただし`source_kind`ごとにconfidence
とmarginを分ける。根拠の異なるpairを同じcp差と解釈しない。

2026-07-21のexact pilot監査により、bounded mateへ任意の`mate_target`を与えるBCEは採用しないことに
変更した。3手以内に詰むことは勝率80%などを意味せず、正例しかないBCEはoutput biasとscore尺度を
押し上げるためである。Phase 2では次を固定する。

- exact mateは原則pairwise lossだけで学習し、絶対WDL校正はPhase 4の実戦結果まで行わない。
- `out_b`とmaterial bypass係数はPhase 1～2では凍結する。
- terminal childはruntime探索で評価関数を呼ばないため、value networkの学習入力にしない。
- familyごとのloss総量を正規化し、合法手数やproof treeの大きさで一問題のweightが増えないようにする。
- exact pilotで使った`mate_target=0.8/0.9`は診断実験として記録だけ残し、後続modelの親にしない。

trainer v2では次も修正する。

- record数でなく有効objective weightとfamily数でgradientを正規化する。objective 0のbounded no-mate recordが
  learning rateを薄めてはならない。
- family単位でseed固定shuffleをepochごとに行い、常に同じ並びで学習しない。
- `margin_class`をsource/horizon別marginへ明示的に写像する。使わないならschemaから削除する。
- valid lossを一つに混ぜず、source・mate distance・move種別・king bucket別metricでbest modelを選ぶ。
- sparse rowだけへL2を掛ける現実装を廃止し、lazy decayまたはexport時decayで未観測rowにも同じ意味を持たせる。

### 7.3 自己探索loss

初期の自己探索では絶対cp回帰を使わない。新modelのleaf値で作ったroot scoreへ同じmodelを回帰する
だけでは、尺度と誤りを自己複製するからである。まず深い探索が選んだ手と明確に劣る手の順位だけを
圧縮する。

絶対scoreは、世代が安定し、異なる深さでscore校正が一致することを確認した後のA/B候補とする。

### 7.4 TD loss

自己対局段階で、PV leafまたは次時点のsearch valueを用いる。

```text
target_t = (1 - lambda) * sum(lambda^n * search_value_{t+n})
           + terminal_result contribution
```

実装前に手番反転、mate score除外、千日手、持将棋、投了labelを明文化する。TDは結果・順位baselineを
上回った場合だけ採用する。

### 7.5 正則化と尺度anchor

過去のWDL学習ではoutput層膨張が実戦崩壊につながった。次を毎batchまたはepochで監視する。

- `out_w` L2 normと最大絶対値
- material係数
- score p01/p50/p99と`|score| > 2,000`の割合
- black/white別bias
- ClippedReLUの0側・1側飽和率
- gradient normとNaN/Inf

最初のkappaは固定し、同じrun中にmodel scaleとkappaを同時学習しない。kappa候補は独立pilotで選ぶ。

## 8. training-only feature factorization

HalfKPはking bucketごとに同じpiece stateを別rowへ持つため、初期にはrare rowの多くがほぼ更新されない。
Stockfish NNUEの技術文書でも、HalfKPの初期学習にvirtual feature factorizationが特に重要と説明されて
いる。

最初に次の2 factorだけを実装する。

```text
E_effective(king, piece_state)
  = E_direct(king, piece_state)
  + E_piece(piece_state)
  + E_relative(relative_king_piece)  // 盤上駒のみ
```

- `E_piece`: king位置をまたいでpiece stateを共有する。
- `E_relative`: 王から見た相対file/rank、駒種、所有者を共有する。
- 持ち駒は`E_piece`だけを使う。
- factorは学習時だけ存在し、export時にdirect rowへ加算してfoldする。
- 推論model形式と評価速度は変えない。

factorなしの同seed baselineを必ず残す。factorを増やせば必ず良くなるわけではないため、king-only factor
などは初回に追加しない。

## 9. 新dataset schema

既存HKSTはsearch cp semanticsへ強く結び付いている。詰みlabelを偽のcpとして詰め込まず、汎用の
`HalfKpCurriculumRecord`を新versioned formatとして追加する。仮称magicは`HKCL0001`とする。

```text
header:
  schema/version, feature profile, record count
  lineage_id, producer revision, seed
  input hashes, generator configuration

record:
  source_kind {exact_mate, exact_tactic, expert_move, game_result,
               self_search, self_play, symmetry_anchor}
  group_id, position_hash, game/problem hash, split
  ply, phase, opening_stratum, confidence, sample_weight
  root HalfKP features
  optional result {loss, draw, win}
  optional proof {status, mate_distance, nodes, rule_version}
  zero or more candidate preferences:
      good child, bad child, preference_kind, margin_class
```

`unknown`、`not proven`、`proven no mate`を別値にする。node capへ達した局面を負けや不詰みとして扱わない。

### 9.1 split規約

- 棋譜は内容hash単位でsplitする。
- 詰み問題は元局面と全派生局面を同じ`problem_family_id`へまとめてsplitする。
- 同じ局面、左右mirror、色交換・180度回転で同値な局面をsplit間で重複させない。
- generator seedをtrain/valid/testで分ける。
- 2026年棋譜はout-of-time testとして凍結し、hyperparameter選択へ使わない。
- exact test、opening test、general game test、self-play testを別manifestにする。

追加の必須検査:

- random playoutをsourceにする場合、各局面hashではなくplayout開始時に作る`game_family_id`でsplitする。
- train/valid/testは別seed・別生成streamで作り、生成後にsplitする方式を標準経路にしない。
- split衝突keyはSFEN文字列でなく、networkが実際に見る`PackedHalfKP + side-to-move + material`を正規化して
  hash化する。手数表記の違い、左右mirror、色交換・180度回転後に同じ入力となる局面を同一視する。
- canonical network-input hashをglobal indexへ登録し、split間衝突を0件にする。
- proof root、全派生node、candidate childは同じfamilyへ所属させる。
- manifestへfamily数、canonical重複除去数、split間衝突検査結果、カテゴリ別件数を記録する。
- hyperparameter選択にはtrain/devだけを使う。frozen testは設定lock後に1回だけ開き、testを見て設定を
  変えた時点でそのsplitをdevへ降格する。

`halfkp_exact_generate` v1は局面SFEN hashをfamily hashとしていたため、同じrandom gameの隣接局面が別splitへ
入る可能性がある。v1 corpusは実装smoke以外へ使用せず、上記を満たすv2へ置き換える。

### 9.2 label provenance

全recordのlabel由来を検査可能にする。`self_search`は教師model hashを必須とし、そのmodelが同じ
scratch lineageでなければ拒否する。`exact_*`はmodel hashを持ってはならない。

## 10. カリキュラムの実行段階

### Phase 0: 独立genesisと検証基盤

目的は、現行modelを読まずに安全な初代modelを作り、後続実験を比較可能にすることである。

実装:

1. `halfkp_scratch_init`を追加する。
2. feature embedding、STM/NSTM outputをseed固定の小さい乱数で初期化する。
3. hidden biasはactive feature数を考慮してClippedReLU中央付近に置く。
4. material係数とoutput biasは0から開始する。
5. 初期化直後のscore標準偏差が過大にならないscaleを10k合法局面で決める。
6. genesis manifestを作り、親modelがないことを検証する。
7. initializerを2 seedで作り、どちらも有限かつ左右・色対称testを通す。

全parameterを厳密に0へするとhidden unitの対称性が破れず、feature gradientも止まり得る。ここでいう
ゼロスタートは「既存知識ゼロ」であり、parameter値は独立した微小乱数にする。

Gate:

- 現行model fileをopenした記録がない。
- save/loadとengine/trainer forwardが`1e-4 cp`以内で一致する。
- score meanがほぼ0で、p99が通常探索score範囲を超えない。
- activation飽和率が各側1%未満。
- 同seedのbyte hashが再現する。

### Phase 1: ルール・対称性anchor

いきなり勝敗を教える前に、position表現と基本的な不変条件を確認する。

dataset:

- 合法局面と正規化した左右mirror
- 色交換・180度回転した同値局面
- 合法な捕獲、成り、駒打ち前後
- 王手を合法に回避できる局面
- 千日手・連続王手・持将棋はvalue学習でなく規則testへ分離

対称局面のscore一致はsoft lossだけでなくunit testでも保証する。捕獲や成りに一律の大小関係を
付けない。局面依存で逆転し得るため、順位labelは次のexact phaseで証明できたものだけにする。

規模は10k、50k、200kの順に増やす。10kをほぼ完全にoverfitできない場合は学習器bugとして停止する。

Phase 2の初回pilotを先行してしまったが、これは順序違反である。以後はPhase 1の10k symmetry corpus、
canonical split監査、128件rank overfitを完了するまでexact学習を再開しない。

### Phase 2: 証明済み詰み・短手数戦術

#### 詰みsolver

現repoには汎用DFPN solverがないため、最初にbounded exact solverを追加する。1手・3手は単純な
完全列挙をreference実装とし、5手以上はproof/disproof numberを持つdf-pnを主経路にする。

- 攻め方は王手のみ、受け方は全合法応手を列挙する詰将棋mode
- 通常局面用には全合法手を扱う短手数mate modeも分ける
- 打ち歩詰め、連続王手の千日手、手番、持ち駒を`shogi_lib`規則と一致させる
- 証明済み詰み、証明済み不詰み、resource limitを区別する
- proof PVだけでなく、各OR/AND nodeの証明状態を再検証できるようにする
- depthだけでなくnode capと経過時間をmanifestへ記録する

#### curriculum

1. 1手詰みは合法手生成・探索発見のtest fixtureに使い、terminal childをvalue学習しない
2. 3手詰みの非終端内部nodeと詰み回避
3. 5手詰みの非終端leaf ordering
4. node budget内で証明できた7手以上
5. capture/promotion/dropを含む短手数の強制勝ち・強制回避

終端局面はrootにもcandidate childにも使わない。bounded mateへ絶対WDL targetを与えず、両候補の
同一horizon内の証明状態が確定し、かつ両childが非終端であるpairだけを使う。攻め方だけでなく、受け方の
「詰みを外す手 / 詰む手」も生成する。攻め方の短手数優先と受け方の長手数回避は、通常の勝敗順位が
安定した後に小さい補助weightで追加する。

`proven_no_mate`という名前は無期限の不詰みと誤読できるため、v2 schemaでは
`no_forced_mate_within_horizon`とし、各recordに`horizon_plies`とsolver modeを必須化する。bad candidateは
move generator順の最初の王手だけを採らず、全合法手を証明した上でmove種別・難度別に決定的samplingする。

各カテゴリはking bucket、攻め駒、持ち駒、盤上/持ち駒からの打ち、成りの有無で均す。同一詰み筋を
駒や盤位置だけ変えて大量複製しない。

Gateの初期目標:

- unseen 1手詰みの探索発見率 100%
- unseen 3手詰み 98%以上
- unseen 5手詰み 90%以上
- 証明済み勝ち/非勝ちpair accuracy 95%以上
- general legal-position score p99、activation飽和、material係数が安全範囲
- Phase 1の対称性testに退行なし

Gateは次の順で判定し、前段未達なら後段の数値を採用しない。

1. dataset integrity: canonical split衝突0、family跨ぎ0、proof全件再検証、resource limitの誤label 0
2. trainer: cleanな128 pairを100% overfitし、中断再開がbit一致
3. static holdout: 全positive rootを分母にしたexact-best top-1、pair coverage、カテゴリ別pair accuracy
4. search holdout: 評価関数を組み込んだ実際の探索で1/3/5手発見率を測る
5. health holdout: exactとは独立した一般合法局面でscore分布、activation、material係数、色biasを測る

従来のpair accuracyはpairを作れたrootだけが分母であり、search discoveryの代用にはしない。

5手以上の閾値はsolver corpusの難度で再校正する。未達ならデータを増やす前に、表現能力と
探索depthのどちらが原因かを分離する。

### Phase 3: 戦型・囲いを使った構造curriculum

戦型はlabelではなくsampling keyにする。初回分類は解釈可能な規則で十分である。

- 居飛車 / 振り飛車 / 相振り
- 飛車fileと早期移動
- 玉のfile/rank、玉周辺3x3の駒種配置
- 主要歩の前進状態
- opening prefix hash
- ply 16/24/32時点の正規化position cluster

「矢倉」「美濃」「穴熊」などの表示名はreport用aliasとし、学習器は安定したfeature keyだけを見る。

この段階から棋譜を導入するが、全局面の勝敗回帰をいきなり行わない。各戦型から少量ずつ選び、
次を学ぶ。

- 記録された指し手と明らかに劣る合法手のpairwise preference
- 同一戦型内の勝敗結果を小weightで補助
- forced move、終盤敗者手、異常終局、低品質棋譜は除外またはdownweight

棋譜手が常に最善とは仮定しない。初期にはrate条件を厳しくし、棋譜手以外の候補へ「悪手」labelを
付けるのは、短い独立探索または後の自己探索で明確なgapがある場合に限る。探索がまだ弱い間は、
棋譜手対ランダム合法手のsoft preferenceとして扱う。

10～20%のopening familyを完全holdoutにし、特定戦型の暗記と一般化を分けて測る。

### Phase 4: 棋譜手 + 最終結果の広域学習

Phase 2と3のgateを通過してから、広い棋譜分布へ進む。

data候補:

- train: 2023～2025年、内容hash split、既知終局のみ
- validation: train年から棋譜単位で固定
- test: 2026年out-of-time
- 1棋譜の上限24～32局面
- opening/middle/endgameをおおむね25/50/25
- opening stratumごとの上限を設定
- 引き分けは規則上明示されたものだけ0.5

250k pilot、1M selection、3M finalの順に増やす。比較候補は少なくとも次の3つとする。

1. resultだけ
2. expert move rank + result
3. 2 + feature factorization

過去にresult-onlyでoffline改善と棋力が乖離したため、result-onlyを本命にしない。material係数、output
norm、score分布へ上限を設け、offline値が改善しても探索smokeが悪化したmodelは破棄する。

Stage replayの初期比率案:

```text
exact/symmetry 20%
opening/expert move 40%
general game result 40%
```

比率は仮説であり、固定validでA/Bする。exact dataを0%へ落とさない。

### Phase 5: 新系列自身のsearch bootstrapping

この段階で初めてmodelを探索教師へ使う。教師は必ず同じscratch lineageの凍結済みparentである。

1世代の手順:

1. parent `G_n`を凍結しhashを確定する。
2. 棋譜局面、合法random perturbation、自己対局局面からrootを作る。
3. `G_n`で深い探索を行い、candidate順位をimmutable datasetへ保存する。
4. `G_n`を初期値として`G_(n+1)`を学ぶ。
5. exact/general holdoutと`G_n`とのpaired matchを行う。
6. gateを通った場合だけ`G_(n+1)`を次の教師にする。

探索depthは2、3、4、必要なら5へ段階的に上げる。最初は250k rootで、改善した世代だけ1Mへ拡張
する。depthを上げた際にbest moveが安定しないrootはconfidenceを下げる。

学習signalは次の優先順とする。

1. search bestと明確に劣るcandidateの順位
2. 棋譜最終結果
3. 棋譜手とのsoft順位
4. parentの絶対search cpは原則不使用

新世代が親に負けた場合、その世代から教師を生成しない。常に最後のaccepted parentへ戻る。

### Phase 6: 新系列内の自己対局

自己対局は初期化手段ではなく、Phase 5までで最低限の棋力ができた後の分布拡張に使う。

- current対currentだけでなくcurrent対過去accepted世代を混ぜる。
- opening strataから8～24 plyの開始局面を選び、同一初期局面を先後交換する。
- 指し手へ少量の温度または上位手randomizationを入れる。
- 投了は保守的にし、初期世代の誤評価による偽labelを避ける。
- 千日手、連続王手、持将棋、最大手数を別resultとして記録する。
- 自己対局結果だけで既存のexact/棋譜datasetを置き換えない。

10k game pilotでparent leagueに勝てなければ停止する。改善時だけ50k、100kへ増やす。

TDLeafはこの段階で、terminal-result baselineと独立A/Bする。TD candidateが強くても、score尺度や
詰み判定を壊した場合は不採用とする。

### Phase 7: consolidationと最終候補

accepted世代だけを対象に、全sourceを混ぜて短い収束学習を行う。

初期比率案:

```text
symmetry/exact tactic 10%
opening/expert move    20%
general result         25%
self-search rank       30%
self-play result/TD    15%
```

SWAは既存trainerの仕組みを使えるが、通常best checkpointとの親子対局で勝った場合だけ採用する。
最終modelをfreezeした後は、testを見てparameter、比率、epochを変更しない。

## 11. CPU・メモリ・時間計画

### 11.1 実測からの基準

このPCでは以下が既に観測されている。

- 1M棋譜局面の単純WDL学習: 約23.7秒/epoch
- 約1.13M search record、8 epochのmini-batch学習: 約12分
- 約1M規模のdepth 4 search teacher一式: およそ1日
- HalfKP-64 model: 約27 MB
- 現在の`data/`使用量: 約62 GB
- 空きdisk: 約282 GB（2026-07-21調査時）

teacher fileはcandidate childを含むため、1M rootで数GBになる。新pipelineへ100～150 GBを上限として
割り当て、stale teacherとrejected checkpointをmanifest/reportを残して順次圧縮または削除する。

### 11.2 概算

| 作業 | 規模 | wall time概算 |
|---|---:|---:|
| dataset build | 1M局面 | 5～20分 |
| outcome/rank training | 1M、5～10 epoch | 10～60分 |
| depth 2～3 teacher | 250k root | 1～4時間 |
| depth 4 teacher | 1M root | 20～30時間 |
| exact mate corpus | 100k証明 | 数時間～数日 |
| self-play pilot | 10k局、0.25～1秒/手、8並列 | 理想13～52時間、実運用2～4日 |
| self-play full | 100k局 | 約2～4週間 |
| paired match | 200局、1秒/手、jobs 1 | 約6～12時間 |

詰み証明時間はheavy-tailなので、件数より総node budgetで管理する。難問1件がrun全体を止めないよう、
per-problem node/time capを必須にする。

### 11.3 運用設定

- 長時間探索は物理core数に合わせ、まず`jobs=8`を測る。
- SMT 16 threadが8 threadより速いとは限らない。1000 rootのpilotで決める。
- trainerとteacher generatorを同時に全coreで動かさない。
- laptopの温度、clock、load average、RSS、swap、diskを30～60秒間隔でlogする。
- 調査時はswap使用量が多かったため、長時間run開始前に他processを閉じ、available memoryを確認する。
- checkpointはepochだけでなく一定record/nodeごとに原子的保存する。
- 停電・suspend後のresumeでdataset順、RNG、optimizer stepが一致するtestを持つ。

## 12. 評価gate

### 12.1 correctness

- trainer forward、engine full refresh、incremental evalが一致する。
- black/white、mirror、持ち駒slot、成り、捕獲、駒打ちで一致する。
- save/load前後がbit exactである。
- finite-difference gradient checkを通す。
- thread数1/8で許容誤差内に再現する。
- resume runとuninterrupted runが一致する。
- dataset manifestの親hashまたはlineage不一致を拒否する。

### 12.2 coverage

毎stageで次を記録する。

- 全105,480 rowの出現数と更新数
- king bucket別の未観測率
- piece kind、所有者、盤上/持ち駒別の未観測率
- trainとtestのfeature frequency divergence
- direct/factor gradient norm

未観測rowを乱数のままfinal exportしない。十分な実局面が得られないrowはfactorをfoldした値を使い、
その割合をreportする。

### 12.3 offline quality

- exact mate/tacticの発見率とpair accuracy
- 棋譜結果のBrier、log loss、calibration error
- expert move top-1、top-k、pair accuracy
- opening/middle/endgame別、opening stratum別の指標
- material差、王手、合法手数、終局種別ごとのslice
- score分布、output norm、activation飽和

confidence intervalは局面単位でなくgame/problem family単位bootstrapで計算する。

### 12.4 search and speed

重みだけを変える段階では推論architectureを変えない。factorはexport時にfoldし、runtime costを
増やさない。

- fixed trace eval/sが同じmodel形式の乱数baselineから3%以上低下しない。
- fixed node searchのnode数、best move、mate発見率を記録する。
- fixed time searchはNPSだけでなくdepth、qsearch、TT hitも比較する。
- portable/AVX2のscore一致を維持する。

異なる重みは探索木を変えるため、NPS単独では推論速度の回帰と判定しない。

### 12.5 新系列のpromotion match

各世代は直前のaccepted parentとのみ比較する。

1. 20局 crash/illegal/extreme-score smoke
2. 100局 screening
3. 200局 paired opening、先後交換
4. 僅差なら400、800、最大1600局へ延長

初期のpromotion条件:

- 点推定score rateが50%以上
- practical non-inferiority下限47.5%を95%区間で下回らない
- exact/offline主要指標に重大な退行がない
- 反則、panic、NaN、異常投了が0

「改善した」と主張するには95%区間下限が50%を超える必要がある。そこまで達しなくても、offline改善
が明確ならcandidateとして保存するが、次世代parentへ昇格させるかは別判断にする。

## 13. 停止・rollback条件

次のいずれかでrunまたは系列を停止する。

- 現行または外部modelのlineage混入を検出した。
- NaN/Inf、model file破損、resume不一致が発生した。
- output norm、score p99、activation飽和が安全上限を超えた。
- exact metricだけ改善し、general holdoutとparent matchが2世代連続で悪化した。
- 3世代連続でparentへ昇格できない。
- datasetを4倍にしてもgame-family bootstrap区間で改善が確認できない。
- teacher depthを上げてもbest-move安定率が改善しない。
- disk残量50 GB未満、available memory 2 GB未満、継続的thermal throttlingが発生した。

失敗modelから継続学習しない。最後のaccepted parent、dataset hash、失敗理由をreportに残す。

## 14. 実装順序とGit運用

推奨branchは`training/halfkp-scratch-v1`とし、生成dataや重みはcommitしない。

| 順序 | 変更 | 完了条件 | 状態 |
|---:|---|---|---|
| 1 | scratch lineage/manifestとinitializer | genesis再現、外部model拒否 | 完了（2026-07-21） |
| 2 | HKCL codecとgolden fixture | round trip、corrupt拒否 | 完了（2026-07-21） |
| 3 | 汎用mini-batch trainer | gradient/resume/overfit test | 実装完了、clean rank 128件pilot待ち |
| 4 | feature factorization | fold前後forward一致 | mapping/fold完了、trainer統合待ち |
| 5 | bounded exact mate solver | proof再検証、規則fixture | 1/3手reference完了、全件再検証とdf-pn待ち |
| 6 | exact curriculum generator | split/coverage manifest | v1棄却、family-safe v2再設計中 |
| 7 | opening stratum classifier | deterministic分類、report | 未着手 |
| 8 | 棋譜rank/result pipeline | no-score provenance | 未着手 |
| 9 | scratch-only search teacher | lineage強制、immutable teacher | 未着手 |
| 10 | self-play league | result semantics、resume | 未着手 |
| 11 | promotion/report automation | CI外ローカルgate再現 | 未着手 |

commitは次のように小さく分ける。

```text
training: add scratch model lineage and initializer
training: add curriculum dataset codec
training: add scratch HalfKP trainer
training: add HalfKP feature factorization
training: add bounded mate proof generator
training: add opening-stratified record pipeline
training: enforce scratch lineage in search teachers
training: add scratch self-play league
docs: record scratch training experiments
```

各実験は`runs/halfkp64-scratch-v1/<run-id>/`へ置き、追跡する結論だけを`report/`へ追加する。Phaseごとに
この文書の進捗表、実測時間、採否、次の仮説を更新する。

### 14.1 Phase 0実施記録（2026-07-21）

`halfkp_scratch_init`を追加し、外部modelを受け取る引数を持たないgeneration 0専用initializerとした。
manifestは`lineage_id=halfkp64-scratch-v1`、`generation=0`、`parent=null`を記録する。featureと
STM/NSTM outputはseed固定のChaCha20正規乱数、hidden biasは0.5、material係数とoutput biasは0である。
出力weightは各perspective内で中心化し、有限乱数標本による任意の定数offsetを除いた。

`runs/halfkp64-scratch-v1/phase0-final-pilot/`で各10,000合法局面を検査した結果は次の通りである。

| seed | model SHA-256 | mean cp | stddev cp | abs p99 cp | 飽和率 | reload最大差 |
|---:|---|---:|---:|---:|---:|---:|
| 20260721 | `a953b54da0d0c2f891ef94ebda81237ac4263b52f7addbc9bf3f383e540534c7` | -0.137 | 4.444 | 11.161 | 0% | 0 cp |
| 20260722 | `18631b5b894152226b3680c516ac8a2866bcb884f89f96f8a01c4ee841ad52ee` | 0.395 | 4.194 | 10.769 | 0% | 0 cp |

seed 20260721を別pathへ再生成したSHA-256も完全一致した。生成物はGit管理外の`runs/`に置き、現行重み、
現行重み由来dataset、親checkpointは一切openしていない。Phase 0のGateを満たしたため採用し、次はHKCL
codecの実装へ進む。

### 14.2 HKCL0001 codec実施記録（2026-07-21）

汎用curriculum datasetの`HKCL0001` writer/readerを追加した。headerはHalfKP次元、宣言record数、scratch
lineage、feature profile、producer revision、seed、入力SHA-256、generator設定、任意の教師model hashを
保持する。recordはlabel source、family/split、局面、WDL、proof、candidate preferenceを別fieldで保持する。

proof statusは`unknown`、`not_proven`、`proven_mate`、`proven_no_mate`、`resource_limit`を別codeにした。
exact labelに教師modelが付くこと、self-search labelに同一lineageの教師modelがないこと、非有限scalar、
範囲外feature、record数不一致、trailing byteを拒否する。固定header golden prefix、record round trip、marker
破損、provenance違反、file-level count patch/trailing byteの4 testを通過した。

### 14.3 scratch trainer実施記録（2026-07-21）

`halfkp_scratch_train`を追加した。HKCL以外を入力できず、初期modelのSHA-256とmanifest、dataset headerの
`halfkp64-scratch-v1` lineageを照合する。WDL BCE、証明済み詰み、candidate順位、対称一致を別lossとして
扱い、mini-batch sparse-row AdaGradで更新する。`proven_no_mate`は敗北labelへ変換せず、明示WDLまたは
証明済みcandidate pairだけを学習する。

optimizer/model checkpointはhash付きで原子的に保存し、run fingerprintが異なる再開を拒否する。value
gradientの有限差分、極端入力の有限性、単一target overfit、optimizer破損拒否、中断再開と連続更新の
bit一致testを通過した。計画上の128件overfit Gateは、bounded solverが生成するexact fixtureで実施するため、
この時点では項目3を暫定状態とする。

### 14.4 feature factorization実施記録（2026-07-21）

training-onlyの`E_piece`と`E_relative`のindex規約、accumulation、runtime HalfKP direct rowへのfoldを実装した。
持ち駒がpiece factorだけを持つこと、king bucketが異なっても同じ相対配置を共有すること、fold前後の
hidden accumulationがbit一致することをtestした。trainerのoptimizer/checkpointへfactor stateを統合し、
factor有無のexact pilotを実行するまでは項目4を暫定状態とする。

### 14.5 bounded exact solver・初回pilot（2026-07-21）

評価関数を一切呼ばないAND/OR完全列挙solverを追加した。攻め方王手限定modeと全合法手mode、ply/node上限、
proof tree、別走査でのproof再検証を持つ。node上限到達は`resource_limit`であり、不詰みへ変換しない。
既存の合法手生成を使うため、打ち歩詰めを含む規則判定を重複実装していない。既知1手詰みfixtureの証明・
再検証と、node capが`proven_no_mate`にならないtestを通過した。5手以上の大規模生成用df-pnは未実装である。

合法なrandom playout局面だけを対象とする`halfkp_exact_generate`を追加し、3手以内の完全証明からHKCLを
生成した。`exact-pilot-800-v3`は1手詰み200、3手詰み200、3手以内不詰み400で、train/valid/testは
622/84/94、node cap到達は0だった。詰みrootでは、同じ上限内で証明済み不詰みの別王手がある場合だけ
candidate pairを作った。bounded不詰みを敗北WDLへ変換していない。

初回hyperparameter pilotのtest結果は次の通りである。全候補は同じscratch genesisから学習した。

| lr | mate target | rank weight | pair accuracy | 1手pair | 3手pair | score abs p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.8 | 0.5 | 96.23% | 93.55% | 100.00% | 1,789 cp |
| 0.02 | 0.8 | 0.5 | 90.57% | 90.32% | 90.91% | 1,648 cp |
| 0.005 | 0.8 | 1.0 | 96.23% | 93.55% | 100.00% | 2,082 cp |

`lr=0.01, mate_target=0.8, rank_weight=0.5`を次のexact拡大pilotの暫定候補とする。1手pairが初期Gateの
100%へ届かないため、weight完成扱いにはせず、factor統合、coverage拡大、探索発見率測定へ進む。

### 14.6 exact 12k拡大pilot（2026-07-21）

独立seed 20260731で1手詰み4,000、3手詰み4,000、3手以内不詰み4,000を生成した。184,659局面を調べ、
node cap到達0、family split後はtrain 9,636、valid 1,227、test 1,137となった。3 epochのlr比較では
valid lossが0.005/0.01/0.02でそれぞれ0.3763/0.3647/0.3653だったが、独立testのpair accuracyは
88.69%/87.21%/90.04%だった。後の監査でsplit独立性とtest運用に不備が判明したため、これを小規模pilot
との差やlearning rate選択の根拠として扱わない。

安定性を優先してlr 0.005を10 epochまで延長した結果、test pair accuracy 91.25%、1手93.78%、3手
88.74%、score abs p99 1,963 cpだった。exact split内の暫定score閾値は満たしたが、一般合法局面Gateを
測ったものではなく、pair accuracy 95%と1手100%にも未達だった。後述の監査により、このweightは棄却した。

### 14.7 中間監査と計画修正（2026-07-21）

#### 判定

計画修正が必要である。800件と12kの結果はcodec、solver、trainerが端から端まで動くことを示す
engineering smokeとしては有効だが、model選択の統計としては無効とする。生成weightを次世代の親へ昇格
させない。

| 成果物 | 判定 | 今後の扱い |
|---|---|---|
| scratch genesis 2 seed | 有効 | 新しいclean pilotの唯一の初期modelとして再利用 |
| model codec/forward・optimizer unit test | 有効 | 実装基盤として維持 |
| HKCL0001 | 汎用codecのengineering smokeとして有効 | exact corpusはHKCL0002で再生成 |
| exact v1 800/12k corpus | model選択には無効 | 再学習・混合・評価へ使用禁止 |
| v1から学習した全weight | 棄却 | lineage parent、探索教師、比較baselineへ使用禁止 |
| 800/12kのloss・pair精度 | 診断値のみ | Gate達成の証拠として引用禁止 |

理由は次の通りである。

1. v1 generatorは各局面SFENをfamily hashにしたため、同じrandom playoutの隣接局面がsplitを跨ぎ得る。
2. Phase 1のsymmetry/canonical split Gateを完了する前にPhase 2を実行した。
3. bounded mate正例への任意BCE targetが絶対scoreを押し上げた。0.9 pilotではtest exact局面のp99が
   2,947 cpだった。`kappa=600, target=0.8`の定数最適値は`600 * logit(0.8) = 831.8 cp`であり、
   実測test平均855.9 cpとほぼ一致した。これは戦術知識でなく目的関数由来のbiasである。
4. 1手詰みpairはterminal childを評価していたが、実際の探索はterminalで評価関数を呼ばないため、学習と
   runtimeのleaf semanticsが一致しない。
5. pair accuracyはpair生成に成功したrootだけを分母とし、計画のsearch discovery Gateを測っていない。
6. 報告したp99はexact局面上であり、計画が要求する独立した一般合法局面のhealth checkではない。
7. generatorはproofを返すが、書き出し前の全件再検証、canonical重複監査、カテゴリcoverage manifestがない。
8. 現trainerはlineage一致だけを確認し、Gate未通過modelでも親に指定できる。計画したpromotion allowlistが
   まだ強制されていない。

12kのpair accuracyはtrain 6,070/6,131 = 99.01%、valid 694/787 = 88.18%、test 678/743 = 91.25%だった。
trainをfitできているため、現時点でmodel容量やoptimizerの学習不能を主因とはしない。優先仮説はsparse rowの
共有不足、source分布、split、教師/loss設計である。800件の51/53 = 96.23%はWilson 95%区間が
87.25～98.96%と広く、12kとの統計的矛盾や「拡大で悪化」を主張しない。

#### 修正後の直近手順

1. immutable promotion registryを実装し、accepted model hash、Gate report hash、承認時刻を記録する。
   genesis 2個だけを初期allowlistへ登録し、v1 weightを`diagnostic/rejected`として明示的にrejectする。
2. `HKCL0002`へhorizon semanticsとgenerator family idを追加する。
3. split別seed/stream、game-family固定、canonical重複0を保証するgenerator v2とaudit toolを作る。
4. Phase 1 symmetry 10kを生成し、clean 128 pairの100% overfitを通す。
5. exact lossを非終端childのrank-onlyへ変更し、Phase 1～2はoutput bias/material係数を凍結する。
6. 全合法候補と攻守両側のproof nodeから、カテゴリを均したexact 12k v2を作る。
7. direct/factorizedを同seed、3 learning rate、同じ3 epochで比較する。
8. static top-1、実探索発見率、一般合法局面healthの全Gateを通った場合だけ50k/200kへ進む。
9. df-pn、opening、棋譜導入は上記Gate後まで延期する。

12k v2は開発Gateとしてtrain pair 99%以上、dev point estimate 90%以上、family bootstrap 95%下限88%以上を
要求する。これはfactorization等の比較を続けるためのGateであり、Phase 2本Gateの95%以上を緩和しない。
50k以上で95%を再要求し、3つのscratch seedで改善方向が一致してから次段階へ進む。

この修正は「現行weightを使わず、exact -> opening -> game -> scratch search」の大枠を変えない。変更するのは
Phase 1を飛ばさないこと、dataset独立性、exact label semantics、評価Gateの厳密化である。

## 15. 最初に行うpilot

中間監査後の最小実行単位を次へ改訂する。v1 exact corpusを追加学習してこの手順を代替してはならない。

1. 作成済みの2つの独立genesis seedを再検証する。
2. split別streamのsymmetry 10,000 familyを作り、canonical衝突0を確認する。
3. `HKCL0002`で非終端exact rank pair 128 familyを作り、100% overfitする。
4. 1/3手各4,000、horizon内非詰み4,000を別seedで生成し、全proofを再検証する。
5. 12kで3 learning rate、direct/factorizedを各3 epoch学習する。
6. 全root static top-1、実探索発見率、独立legal health set、activation、feature coverageで候補を選ぶ。
7. 成功した場合だけ50k/200k exact corpusとdf-pn実装へ進む。
8. self-play smokeはopening/general game Gate後へ延期する。

このpilotでは棋譜、現行重み、自己探索scoreを一切使わない。ここで学習が成立しなければ、長時間run
には進まず、initializer、loss、factorization、model容量を調査する。

### 15.1 修正版計画の実施記録

#### promotion registry（2026-07-21）

trackedな`training/halfkp64-scratch-v1/promotion_registry.json`を追加した。初期allowlistは検証済みscratch
genesis 2 seedだけであり、v1 exactから生成した10個のmodel hashを`exact_v1_diagnostic`として明示的に
rejectした。trainerはparent model、parent manifest、Gate report、registry、generationの組を照合し、単に
同じlineageであるだけのmodelを親として受け付けない。registry hashとGate report hashは新しいtraining
manifestとrun fingerprintへ入る。hash不一致拒否testを通過した。

#### HKCL0002とsplit isolation（2026-07-21）

codecを`HKCL0002`へ更新し、`HKCL0001`をheaderで拒否するようにした。各recordはgenerator family hashと、
実際にnetworkが見る`PackedHalfKP + side-to-move + material`のSHA-256を別々に必須保持する。reader/writerは
network-input hashをroot featuresから再計算し、不一致を拒否する。bounded proofには`horizon_plies`と
`solver_mode`を必須化し、旧`ProvenNoMate`を`NoForcedMateWithinHorizon`へ改名した。mate distanceがhorizonを
超えるrecord、proof metadataのないexact recordも拒否する。

`halfkp_exact_generate`はv2で1実行1 splitへ変更した。train/valid/testは異なるseedで別々に生成し、random
gameの開始時にfamily IDを作って、そのgameから採用した全局面へ同じIDを付ける。局面ごとのhashでsplitを
決める旧経路は削除した。`halfkp_curriculum_audit`は複数HKCLを横断してfamilyとnetwork-input canonical hashの
split衝突を検査し、衝突が1件でもあれば非zero終了する。これによりv2 corpusは学習前に衝突0を証明できる。

左右反転anchor generatorも追加し、seed `2026072101/02/03`の独立streamでtrain 8,000、valid 1,000、
test 1,000の計10,000 familyを`runs/halfkp64-scratch-v1/symmetry-10k-v2`へ生成した。audit結果はfamily split
衝突0、network-input split衝突0である。preference内のgood/bad childもaudit対象に含めており、rootだけを
検査して派生入力の漏洩を見逃すことはない。

trainer v2ではexact mateの正例BCEを削除し、Phase 1～2の`out_b`とmaterial bypass係数を凍結した。
`margin_class`はsource別のcp marginへ写像し、family IDでrecordを束ねてfamily内objective総数を正規化する。
familyはhash順へ一度固定してから`seed ^ epoch`でshuffleするため、実行ごとのHashMap乱数に依存しない。
gradientの分母はrecord数でなく有効family数とし、objectiveを持たないbounded non-mate recordがlearning rateを
薄めない。1手詰みのterminal childもrank pairから除外した。

Phase 1の確定run `phase1-symmetry-v2c`はgenesis
`a953b54da0d0c2f891ef94ebda81237ac4263b52f7addbc9bf3f383e540534c7`から5 epoch学習した。独立test
1,000 familyの左右対称score RMSEはgenesisの2.7360 cpから0.6228 cpへ改善し、score absolute p99は
3.228 cp、`out_b=0`、material coefficient `=0`を維持した。model hash
`caf2e9559028a674d762e4e2c587801cb5c907571e647adf8f5b7c666e27bd6b`をPhase 1 Gate合格としてregistryへ
追加した。これ以前のdirty pilot `phase1-symmetry-v2`は昇格対象にしない。

## 16. 参考資料と本計画への反映

1. Yoshua Bengio et al., [Curriculum Learning (ICML 2009)](https://icml.cc/2009/papers/119.pdf)
   簡単な課題から段階的に難しくする考え方を、exact -> opening -> general -> self-playの順序へ反映。
2. Kunihito Hoki, [局面評価の学習を目指した探索結果の最適制御](https://ipsj.ixsq.nii.ac.jp/records/97627)
   棋譜手とminimax結果を一致させる目的、ゼロ解回避、過学習制約を順位lossとscale監視へ反映。
3. Kunihito Hoki and Tomoyuki Kaneko,
   [Large-Scale Optimization for Evaluation Functions with Minimax Search](https://doi.org/10.1613/jair.4217)
   大規模将棋評価関数を少数hyperparameterで学ぶMMTOを、Phase 3～5の中核方針に採用。
4. Joel Veness et al.,
   [Bootstrapping from Game Tree Search](https://proceedings.neurips.cc/paper/2009/hash/389bc7bb1e1c2a5e7e147703232a88f6-Abstract.html)
   小さい乱数weightから深いalpha-beta探索を教師にする反復を、scratch lineage内のPhase 5へ反映。
5. Jonathan Baxter et al.,
   [KnightCap: TD(λ) with Game-Tree Search](https://arxiv.org/abs/cs/9901002)
   alpha-betaとTDを結合する代替案を、Phase 6の限定A/B候補とした。
6. Thomas Anthony et al.,
   [Thinking Fast and Slow with Deep Learning and Tree Search](https://proceedings.neurips.cc/paper/2017/file/d8e1344e27a5b08cdfd5d027d9b8d6de-Paper.pdf)
   planningとgeneralizationを世代ごとに分離し、教師modelを凍結する運用へ反映。
7. David Silver et al.,
   [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815)
   規則だけからの学習可能性は確認したが、計算資源とarchitecture差から本計画の直接方式には不採用。
8. Stockfish project,
   [NNUE technical documentation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)
   sparse batch、WDL空間、result混合、HalfKP factorizationをtrainer設計へ反映。
9. Yu Nasu,
   [Efficiently Updatable Neural-Network-based Evaluation Functions for Computer Shogi](https://www.apply.computer-shogi.org/wcsc28/appeal/the_end_of_genesis_T.N.K.evolution_turbo_type_D/nnue.pdf)
   CPU向け差分更新評価というHalfKP/NNUEの前提を維持し、学習のために推論costを増やさない。
10. Ayumu Nagai,
    [df-pnアルゴリズムの詰将棋を解くプログラムへの応用](https://ipsj.ixsq.nii.ac.jp/records/32069)
    詰み・不詰みをproof/disproof numberで解く方法を、Phase 2のbounded exact solverへ反映。

## 17. 最終判断

提案された「型から始める」方針は採用する。ただし、型を人手の絶対評価値として注入するのではなく、
ルールで証明できる局面と順位、戦型別sampling、固定testとして使う。

強い重みを作る本体は、次の3信号の組合せになる。

1. 証明可能な局所知識
2. 棋譜手と最終結果
3. 新系列自身の深い探索と自己対局

初期modelから最終世代まで現行重みとの数値的な接続を完全に断つ。これにより、旧KPP模倣の上限を
引き継がず、失敗時にも原因を各stageへ切り分けられる。
