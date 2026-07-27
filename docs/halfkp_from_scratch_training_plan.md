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

Phase 1を通過し、Phase 2のexact dataset integrityとtrainer capacityを確認してから、広い棋譜分布へ進む。
Phase 2のexact-only modelが単独でpair精度95%へ達することは、棋譜pipelineの実装開始条件にはしない。
ただしPhase 2で棄却したmodelを親にせず、accepted Phase 1 modelからexact replayを混合して学習する。
combined candidateを昇格する時点では、Phase 2 exact、Phase 3 opening、一般局面health、実探索の全Gateを
同時に満たさなければならない。

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
- output normまたはscore p99が安全上限を超えた、あるいはactivation診断で非有限値・全channel崩壊を検出した。
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
| 3 | 汎用mini-batch trainer | gradient/resume/overfit test | 完了、HKCL0003 clean rank 128件を100% overfit |
| 4 | feature factorization | fold前後forward一致 | factor-onlyを3 seedで選定、尺度付き64k Gate通過 |
| 5 | bounded exact mate solver | proof再検証、規則fixture | 1/3手referenceと完全検証器完了、df-pn待ち |
| 6 | exact curriculum generator | split/coverage manifest | HKCL0004 rule v3、攻守64k/全proof監査完了 |
| 7 | opening stratum classifier | deterministic分類、report | bitfield v1とHKCL分布report完了 |
| 8 | 棋譜rank/result pipeline | no-score provenance | wdoor pilot完了、confidence/loss分離と250k拡張待ち |
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

以下は2026-07-21時点の手順であり、実施結果を受けて15.2で再改訂した。今後は15.2を優先する。

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

## 15. 最初に行うpilot（完了済みの旧手順）

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

Phase 2のcapacity Gateとして、terminal childを含まない3手詰みrank pairを128独立family生成した。初回は
128 record/22 familyだったため棄却し、pair採用ごとにrandom gameを終了するdiagnostic modeで128/128 familyへ
再生成した。Phase 1 modelからのoverfitはbest epoch 2でtrain pair 128/128（100%）に到達し、`out_b=0`、
material coefficient `=0`を維持した。このrunはcapacity確認専用で、trainをvalidにも指定しているため昇格modelや
12kの親には使用しない。

training-only factorizationをscratch trainerへ統合した。`direct`と`piece-relative`をCLIで選択でき、後者は
direct rowに加えてpiece state共有factorと相対王位置factorへ同じgradientを流す。評価・export時にはfactorを
direct HalfKP rowへfoldし、runtime model形式や探索時costを変更しない。初回pilotではcheckpoint resumeを
factorized runに許可しない。factor状態を欠落させた不完全resumeを防ぐためであり、各runは短い3 epochを一度に
完了させる。

exact-12k初回v2は12,000 record生成後のauditでfamily split衝突0、network-input split衝突2となった。
学習には一度も渡さず、audit-failed diagnosticとして固定する。再生成は固定優先順位`test > valid > train`とし、
上位splitのroot/good/bad全network-input hashを下位split generatorのexclusionへ渡す。v2bでは1 recordを
1 random-game familyへ固定するため、衝突recordの棄却が必ずfamily全体の棄却になる。同一split内の既採用
root/childとの重複もgenerator内で拒否し、最後に全file横断auditを再実行する。これは中間監査用サブエージェント
の助言とも一致する。

v2bは12,000 record/12,000 family、rootとchildを合わせた20,000 unique network inputでsplit衝突0となった。
初回6 pilotではfactorizedがdirectを上回ったが、最良でもtrain 93.1%、valid 75.5%で90% Gateに未達だった。
さらにPhase 1 symmetry test RMSEが0.6228 cpから最大246 cpへ退行した。調査するとrank lossはcp/temperatureで
正規化している一方、symmetry MSEだけがraw network unitのままで勾配尺度が不整合だった。Gateは緩和せず、
symmetry differenceも`difference_cp / symmetry_temperature_cp`へ正規化するよう修正し、pilotを再実行する。

cp正規化後も20% anchor混合ではexact objectiveとのtrade-offが残り、symmetry RMSEをPhase 1水準へ保てなかった。
左右反転は本来厳密な不変量なので、training時に中央筋の王bucketのmirror対応direct rowを平均し、relative factorの
左右delta対応rowも平均する射影を各update後に適用する。中央筋以外は既存feature extractorが王位置でmirrorを
canonical化済みである。export modelは通常HalfKP rowへfold済みなのでruntime形式・推論命令数・探索速度は変わらず、
対称性をdataset weightとの綱引きにせず構造的に保証する。

factor側の初回射影ではrelative factorだけを左右tieし、piece factorのpiece stateに盤上squareが含まれる点を
見落としていたため、factorized modelだけ対称性が崩れた。piece factorのmirror square rowもtieする修正と
回帰testを追加した。修正後pilotはsymmetry RMSE 0.0001 cp、train 97.41%、valid 78.75%となり、表現制約を
守りながら旧pilotの75.5%を上回った。ただし90% Gateには未達なのでGateを緩めない。train-valid gapから
pair family数不足を主因と仮定し、testを開かずにvalid 2,000/train 20,000 pairの中間learning-curve runを追加する。
ここでも90%に届かなければ、50kへ盲目的に拡大せずlabel設計とmodel容量を再検討する。

20k train/2k validではvalid 81.55%まで改善したが90%未達で、score absolute p99も3,617 cpへ拡大したため
Gate不合格とした。さらにHKCL0002が元局面を復元できず、proof再走と実探索Gateを独立実行できないことが判明した。
sidecarによる救済は採用せず、codecを`HKCL0003`へbumpしてHKCL0002をfail-closedで拒否する。

HKCL0003はcanonical root SFEN、preferenceごとのstable USI good/bad move、proofの`max_nodes`を必須情報へ
追加する。readerはSFENから局面を再構築してposition key、PackedHalfKP、network-input hashを照合し、exact moveが
相異なる合法手で、適用後のpacked childと完全一致することも検証する。auditの`--revalidate-proofs`はroot proofを
同一horizon/mode/node boundで再走し、proof treeを検証したうえで、good childが同じattackerのmate、bad childが
horizon内非詰みであることを全件再証明する。今回のexact GateはSFEN snapshotをstateless rootとして開始する。
千日手履歴を使う将来のgame/search corpusではposition commandとmove historyを別途必須化する。

生成中のSFEN round-trip監査で、incremental Zobrist keyがsnapshot再構築時のkeyと一致しない局面が見つかった。
内部探索keyをdataset identityに流用せず、HKCLの`position_hash`はcanonical network-input SHA-256の先頭64 bitと
定義し直した。完全な256 bit hashも別fieldに保持してsplit監査へ使うため、64 bit値は索引用途に限定する。

HKCL0002 exact corpusとその派生modelはすべてdiagnostic-only/rejectとする。test metricは未開封だが旧schemaのtestは
freezeせず破棄し、HKCL0003の独立seedで再生成する。Phase 1も安価にHKCL0003で再生成・再学習して旧Phase 1を
supersedeし、trainerの標準経路へ旧schema例外を残さない。この方針は計画外事象に対するサブエージェント監査の
推奨と一致する。

HKCL0003 Phase 1は`symmetry-10k-v3d`（train 8,000/valid 1,000/test 1,000、split衝突0）で再実行した。
左右row射影により全splitのsymmetry lossとtest RMSEは表示精度で0、score p99 10.697 cp、`out_b=0`、material
coefficient `=0`だった。新Phase 1 model hash
`1d773b4f41f5b485cda8fc94dd4b7131328fde332e33e5c6e7382fff33840ea2`を昇格し、HKCL0002 Phase 1 modelと
確認済みHKCL0002 exact派生modelをregistryで明示rejectした。

HKCL0003 clean-128は128 record/128 familyを生成し、root proofとgood/bad childのsolver再走を含む128/128件の
revalidationに成功した。新Phase 1 modelからfactorized trainerで100%（128/128）overfitし、capacity Gateを
再通過した。このmodelはtrain=validの診断専用で昇格させない。

#### HKCL0003 rule v2 exact 12k（2026-07-22）

最初の`exact-12k-v3`はfamily/input split衝突0だったが、旧検証器がhorizon末端の
`ProvenNoMate + empty edges`を通常のAND反証として扱い、health 102件を誤って拒否した。独立監査で
train 83、valid 12、test 7件がすべて同じ原因で、rank pairの失敗は0件と確認した。検証器をsolverと同じ
`resource -> repetition -> legal terminal -> horizon`順に直し、horizon leafと探索路上反復だけに空edgeを
許した。さらにdefender-to-moveから始まるgood/bad child proofもtree全体を検証するようにした。回帰test、
clean-128、旧12kの再検証はすべて成功した。

旧成果物をそのまま昇格せず、generatorの`rule_version=2`、新規seed、固定優先順位
`test > valid > train`で`exact-12k-v3b`を再生成した。内訳はtrain 9,600、valid 1,200、test 1,200、合計
12,000 record/12,000 family/23,920 unique network inputである。rootおよび全childのinput split衝突0、
family split衝突0、proof再検証12,000/12,000を満たした。凍結test artifactは次であり、構成確定まで
model metricを開かない。

- `test/pairs.hkcl`: `cd032a633566c1b5f6486c205c95664d68a68dfb024465438410da5692080355`
- `test/health.hkcl`: `5cc2136e293822d7255a582c15e62b6c40f6b425eeb2d5a82fd554a6a18a04d6`
- `audit-final.json`: `357215521a42d30dd3b29321cbedf5fb6c27604d9b0ea538983924d774bb849e`

新Phase 1 modelを親に、train/valid pairだけでdirectとpiece-relativeを3 learning rate、各3 epoch比較した。
最良piece-relative (`lr=0.005`) はtrain 95.84%、valid 77.25%、valid score absolute p99 1,777 cpで、
開発Gateのtrain 99%/valid 90%に未達だった。左右対称RMSEは全候補で0.0001 cp以下を維持した。

巨大なdirect rowの局所記憶と共有factorの競合を分離するため、Phase 1 direct rowを固定してpiece/relative
factorだけを学習し、最後に通常HalfKP rowへfoldする`factor-only`対照を追加した。`lr=0.01, l2=1e-4`で
validは80.00%へ改善したが、train 90.38%、p99 3,342 cpでGate未達かつ尺度が大きい。`l2=0.01`では
p99 1,786 cpへ抑制できたがvalid 79.00%だった。したがって12k候補はすべてdiagnostic/rejectedとし、
registryへ昇格しない。testは未開封のまま維持する。

この時点では50kへ拡大しない。次節の中間監査で、record総数12kに対して実際のtrain rank familyが3.2kしか
なく、generatorも計画した全合法候補・攻守両側を実装していないことが判明したためである。90%開発Gateと
95%本Gateは緩和せず、dataset objectiveとscale制御を直してからlearning curveを取り直す。

### 15.2 12k途中結果による計画改訂（2026-07-22）

#### 判断

大枠の`rule anchor -> exact tactics -> opening/game -> scratch search`と、現行weightを一切使わない方針は
維持する。一方、Phase 2を現generatorのまま50kへ増やす直近手順と、exact-only Gateを棋譜pipeline実装の
前提にする順序は修正する。

根拠は次のとおりである。

1. `exact-12k-v3b`の12,000 record中、順位学習へ渡したのはtrain 3,200 family、dev 400 familyだけである。
   残りはmate/no-mate health recordであり、`12k学習`という呼び方は有効objective数を過大に見せる。
2. 現generatorは攻め側の3手詰みrootから、proof moveと「最初に見つかった別の王手」1手だけをpairにする。
   全合法候補、non-check negative、複数の正解手、防御側の詰み回避、moveカテゴリ均衡は未実装であり、
   計画書のdataset仕様を満たしていない。
3. 同じdev 400 familyでdirect 72.50%（Wilson 95%区間67.93～76.65%）、piece-relative 77.25%
   （72.89～81.09%）、factor-only 80.00%（75.80～83.63%）だった。共有factorの方向は有望だが、同一familyの
   paired差をまだ保存していないため、point estimateだけで方式を確定できない。
4. piece-relativeはtrain 95.84% / dev 77.25%でsparse direct rowの記憶が強い。factor-onlyは
   train 90.38% / dev 80.00%でgapを縮めたが、score p99が3,342 cpへ膨張した。`l2=0.01`ではp99を
   1,786 cpへ抑えた一方dev 79.00%であり、表現共有と尺度制御を別々に解く必要がある。
5. rank-only logistic objectiveは正しい順序に有限の絶対cpを定めず、現在の200 cp marginは分離可能なpairで
   score normを押し上げる。通常局面の尺度を持たない段階で、exact-only lossへ任意のcp marginを課すべきでない。
6. 現healthはmate statusで選別したrandom-walk局面であり、将来の一般探索分布から独立したlegal health setの
   代用にはならない。

#### 改訂後の直近手順

1. 現在のPhase 2 model hashをregistryで`diagnostic/rejected`に固定し、accepted Phase 1 modelだけを親にする。
2. evaluatorへfamilyごとの予測、move種別、king bucket、ply、feature coverageを出力するdev専用診断を追加する。
   同じ400 family上の方式比較はpaired family bootstrapまたはMcNemar検定で行い、testは開かない。
3. exact generatorをrule v3へ更新する。各rootの全合法childを同一horizon/node limitで再証明し、
   `ProvenMate / NoForcedMateWithinHorizon / ResourceLimit`を分離する。複数goodと、checking/capture/promotion/
   drop/quietから決定的に層化した複数badを1 familyへ保持する。resource-limit childを教師にしない。
4. 攻め側だけでなく、王手後のdefender-to-move nodeから「詰みを外す応手 > 詰む応手」も生成する。
   変換・派生nodeは同じfamily/splitに固定し、1 familyのobjective総weightを1へ正規化する。
5. root sourceは、標準初期局面からの制約付き合法playoutと、capture/promotion/dropを均した手続き的mate templateを
   別stratumにする。極端なrandom walkだけへ依存せず、template IDと変換元をfamily provenanceへ含める。
6. mate statusで条件付けない独立legal health corpusを別seedで作る。score mean/std/p99、`|score| > 2,000`、
   ClippedReLU飽和、手番・色bias、feature coverageを測る。
7. exact rankは0 cp marginをbaselineにし、現在のAdaGrad勾配へ混ぜたL2とは別にdecoupled weight decayを実装する。
   output head固定と、legal health上のscore RMS anchorをA/Bし、順位精度と尺度を同じ仕組みで調整しない。
8. rule v3でtrain 4,000 root、dev 1,000 rootのpilotを3 scratch seedで行う。指標は全合法手中の
   `any-proven-good top-1`、category別pair accuracy、paired 95%区間、p99、対称性である。
9. 4kから16k rank rootへ増やし、paired区間で改善方向が一致し、一般healthが悪化しない場合だけ64kへ進む。
   4倍化で2 point未満の改善または区間が重なる状態が2回続けばexact単独拡大を停止する。
10. Phase 3のopening classifierと棋譜ingestは手順2～7と並行して実装してよい。棋譜はlicense/provenanceを固定し、
    指し手と最終結果だけを用い、評価値・PV・現行weight由来datasetを禁止する。最初のcombined trainingは
    accepted Phase 1から開始し、rule v3 exactをreplayとして混合する。
11. exact-only modelを単独で昇格しない。combined candidateがexact regression、opening/game holdout、独立health、
    fixed-node searchの全Gateを通った時点で初めてgeneration 2へ昇格する。exact 90%/95%閾値は最終昇格時まで
    維持し、失敗を理由に閾値を下げない。
12. df-pnと5手以上への拡張は、rule v3の3手learning curveが上向いた後に行う。現段階ではlabel数より
    generator semanticsと一般化診断を優先する。

この改訂により、現在の失敗modelから継続せず、Phase 1の安全なscratch parentを保持したまま、exact-onlyの
局所課題で全体計画が停止することも避ける。棋譜pipelineを前倒しするのは旧weightへ戻ることではなく、
ゼロ系列へ不足している広域分布と結果信号を追加するためである。

### 15.3 HKCL0004 rule v3 learning curve（2026-07-22）

防御側rootではroot手番と詰ませる側が異なるが、HKCL0003はproofのattacker色を保存していなかった。このまま
同一schemaへ防御recordを追加すると旧readerが黙って誤解釈するため、HKCL0004へfail-closedでbumpし、
`ProofLabel.attacker`を必須化した。auditはroot/good/badをstored attackerで再探索し、攻撃rootでは
`good=mate, bad=no-mate`、防御rootでは逆を要求する。64件smokeと正式corpusの全proofが再検証に成功した。

rule v3は全合法childを分類し、checking/capture/promotion/drop/quietの組合せごとに複数negativeを保持する。
攻撃・防御を同数にし、train 4k、追加12k、追加48kを非重複streamで生成した。最終監査はtrain 64,000、
dev 1,000、合計65,000 family、415,042 unique network input、split衝突0、proof成功65,000/65,000である。
testはgeneratorと学習設定を固定するまで生成もmodel評価も行っていない。

rank-only proofは絶対cpを定めないためmarginを0 cpへ変更した。factor-only、decoupled weight decay、legal-health
score anchor、output head低速更新を分離してA/Bした。directに対するfactor候補のpaired改善は4k devで
+13.33 point、95%区間+11.32～+15.39 pointだった。4kから16kで約4 point改善し、3 seed同方向かつhealth非悪化
だったため64kへ進んだ。

64kではexact:health比に合わせてanchor weightを2.0へ補正し、best epoch 1を採用した。3 seedのdev結果は次の
とおりで、開発Gate `point >= 90%`、family bootstrap下限`>= 88%`、一般health p99 `< 2,000 cp`、左右対称性を
すべて同時に満たした。

| seed | model SHA-256 | family精度 | 95%下限 | health abs p99 | symmetry RMSE |
|---:|---|---:|---:|---:|---:|
| 20260721 | `93e81b2b...30deeb` | 90.61% | 89.42% | 1,834 cp | 0.000062 cp |
| 20260722 | `6d219213...75f9f9` | 90.47% | 89.25% | 1,807 cp | 0.000066 cp |
| 20260723 | `db1563dc...b25fcd` | 90.41% | 89.21% | 1,820 cp | 0.000055 cp |

これは12k開発Gateの通過であり、Phase 2本Gateやgeneration昇格ではない。3候補をregistryへまだaccepted登録せず、
次にdev全合法手static top-1、fixed-node search、activation/feature coverageを実施する。構成を固定してHKCL0004
testを独立生成・SHA freezeした後、一度だけtestを開く。combined opening/game Gate前にexact-only modelを
探索教師や次世代parentへ使わない。

全合法手static gateを実装し、dev 1,000 rootの全43,952合法childをstored attackerと同じsolver条件で再証明した。
全rootに少なくとも1つのproven-goodがあり、resource limitは0、coverageは100%だった。seed 20260721/22/23の
`any-proven-good top-1`はそれぞれ68.6%/68.4%/68.1%、内訳は攻撃root 39.4%/39.6%/38.8%、防御root
97.8%/97.2%/97.4%である。3 seedで安定しているため偶然の失敗ではなく、sampled pair精度90%と全合法手選択能力の
差を示す。最終exact 90%/95% Gateを緩和せず、このexact-only候補をaccepted/promotedにしない。次は同じdevで
固定ノード探索発見率を測り、静的表現不足と探索による補完を分離する。独立testは未生成・未評価のまま維持する。

既存探索はdepth/time制限だけだったため、`SearchLimits.max_nodes`を追加した。node制限はalpha-betaとquiescenceの
各node進入前に判定し、time/stopの1,024 node間隔とは分離する。未完のiteration/aspiration再探索は採用せず、最後の
完了depthだけを返す。node制限付き並列探索は全体予算が非決定的になるためfail-closedで拒否し、Gateは単一threadとした。
上限、再現性、未完反復、quiescence、並列拒否のtestを追加した。

seed 20260721で10k/50k/100k/200k nodeを開発用に校正した。全root成功率は55.4%/82.6%/89.9%/95.0%、
depth 0は437/156/84/37だったため、閾値を変更せず200k nodeを固定した。200kでのseed 20260721/22/23の結果は、
全root成功率95.0%/95.6%/95.3%、判定可能root内98.65%/98.56%/98.55%、depth 0は37/30/33、探索失敗は全seed 0。
攻撃root成功は457/462/461（91.4%/92.4%/92.2%）、防御root成功は493/494/492（98.6%/98.8%/98.4%）だった。
探索は静的top-1の攻撃側約39%を大きく補完したが、全攻撃root基準では初期3手目標98%に未達である。したがって
exact-only候補は引き続き昇格不可とし、activation/feature coverageを確認後、all-legal hard negative/listwise改善と
Phase 3 combined pipelineへ進む。独立testはまだ生成・評価しない。

一般合法validation 1,000 rootでroot accumulatorの128,000 activationを測定した。seed 20260721/22/23の
0未満率は黒1.21%/2.31%/1.16%、白2.06%/2.65%/1.91%、1超過率は黒0.30%/0.51%/0.78%、
白0.42%/0.85%/1.31%だった。各側1%未満Gateに対し、下側は全seed、上側はseed 20260723の白側が未達である。
さらにvalidationで観測したfeature rowは3,780/105,480（3.58%）、king bucketは黒13/45、白18/45に留まった。
これはmodel飽和だけでなく、random legal health corpusが標準初期局面近傍へ偏り、coverage Gateとして不足していることを
示す。よって3候補はhealth Gateでも棄却し、独立testを開かない。次の学習前にhealth generatorを45 king bucket、
盤上/持駒、game phaseで層化して拡張し、activation penaltyまたはanchorのA/Bは拡張validation上で行う。
exact-only失敗modelから継続せず、combined候補はaccepted Phase 1 parentから開始する原則を維持する。

上記1,000 root corpusのcoverage不足を解消するため、weightを一切読まない`legal_health_stratified_v1`を追加した。
king moveとquiet moveを決定的にoversampleし、黒白king bucketとopening/middlegame/endgameに最低quotaを設ける。
初回v5は独立seedだけでsplitを分けたためfamily 441件、network input 760件がsplitを跨ぎ、監査で棄却した。
v5のtestは一度もmodel評価せず無効artifactとした。v6はtrainを確定し、validからtrain、testからtrain+validの
root/mirror/family全hashを除外して順に生成した。auditは200,000 record/family、cross-record重複0、family/input
split衝突0、黒白45/45を確認した。固定SHA-256はtrain `67b221a8...14e26`、valid `b62686e3...aa9e7`、
未開封test `2f529828...4c641`である。

v6 valid 50,000 rootは33,644/105,480 feature row（31.90%）と黒白45/45 bucketを観測した。seed 20260721/22/23の
0未満activation率は黒1.26%/1.69%/0.74%、白1.28%/1.76%/0.78%、1超過率は全側0.55%以下だった。
したがって旧1,000 rootだけによる「3 seedすべてhealth棄却」は過度に狭い分布の影響を含む。seed 20260723は
拡張health activation Gateを通るが、全合法手static攻撃38.8%、200k search攻撃92.2%で最終Gate未達のため、
結論（exact-onlyを昇格せず、失敗modelから継続しない）は変えない。今後のhealth判断はv6 validを正本とし、testは
combined構成固定後まで開かない。

Phase 3のsampling keyとして`opening bitfield v1`を実装した。先後を各々の視点へ向け、rookを
`none/home/ranging/shifted`、king fileを`left/center/right`、初期rankから進んだ歩数を`0/1-2/3+`へ分類し、
12 bitの安定IDへ符号化する。表示aliasはreport専用で、戦型名を教師labelにはしない。初期局面IDのgolden testと
HKCL root用の決定的分布reportを追加した。v6 valid 50,000 rootでは638 stratumを観測した。これはhealth random
playoutの多様性診断であり、棋譜splitの代用ではない。次はlicense/provenanceを固定した棋譜ingestでgame単位splitと
opening-family holdoutを構築し、このIDをsampling/上限管理にだけ使用する。

既存CSA dataset builderをv4へ更新し、`license-id`と`source-uri`を必須化した。両値はstage fingerprintとmanifest、
各JSONL recordへ保存し、空値を拒否する。各局面には棋譜手を指す前のpositionからopening stratum ID/aliasを付与する。
入力内容hashによるgame単位split、棋譜手、既知終局、rate/filter情報だけを扱い、評価値・PV・既存weightは入力schemaに
存在しない。現workspaceには出典とlicenseを確認できるCSA棋譜がないため、無出典の既存SFENを棋譜教師へ転用しない。
licensed CSA sourceを固定するまではPhase 3学習を開始せず、pipeline実装とfixture検証だけを進める。

ユーザー指定によりPhase 3の棋譜sourceをwdoor/floodgate公式archiveへ固定した。workspaceには2023～2026年の
421,453 CSA局が既に展開済みであり、2023/2024/2025 archive SHA-256は公式掲載値
`d10ba4a2...fd5b9` / `4847d98c...f4e3e9` / `42350490...d09baa`と一致した。2026 archiveは公式ページに
現在SHAが掲載されていないためout-of-time候補として隔離し、学習・開発には使わない。provenance IDは
`user-authorized:wdoor-public-archive`、source URIは`https://wdoor.c.u-tokyo.ac.jp/shogi/`とする。

game contentだけでなく最初の16 plyをcanonical USI列へしたopening-family SHA-256を追加し、同じprefix familyを
必ず同一splitへ送るようbuilder v4を拡張した。2023年から2,000局を処理したpilotは1,881局、42,748 record、
opening/middle/late=15,048/21,765/5,935、train/valid/test=34,394/4,420/3,934だった。全recordは既知の決着、
非王手、合法手2以上で、1局24局面（phase別8/12/4）を上限とした。このpilotの`test`はpipeline内部splitであり、
最終out-of-time testではない。builder変更をclean commit後に同設定を再生成してmanifestのdirty flagを除去する。

game/family/provenance/教師手合法性/network inputを独立検査する`game_dataset_audit`を追加した。clean pilot v2は
game split衝突0、opening-family split衝突0、違法教師手0、unknown/non-decisive 0だったが、異なるopening prefixから
同一局面へ合流するtranspositionが403件splitを跨いだため棄却した。builderはselected recordのHalfKP network-input
SHA-256を全gameで追跡し、先着splitと異なる候補を学習前に除外する。重複許容や閾値緩和は行わず、auditでcross-split
network collision 0を必須とする。

修正後pilot v3は41,913 record、game/family/network split衝突0、違法手0、provenance不一致0でauditに合格した。
ただし同一split内に同一network inputが903件あり、同じ局面を複数棋譜由来で過重に数えるため、最終builderでは
split内外を問わず先着network inputだけを保持する。cross-split除外とsame-split重複除外を別counterでmanifestへ残し、
最終pilotは両方0をauditで確認する。

pilot v4は39,712 unique network input、game/opening-family/network split衝突0、split内重複0、教師手合法性・
provenance全件一致でaudit合格した。ここから評価器を呼ばない`game_soft_expert_v1` generatorを追加した。棋譜手を
good、棋譜手以外のquiet legal moveをroot hashとseedで1手決定しbadとするが、棋譜手が常に最善とは仮定せず
confidence 0.6、sample weight 0.1のsoft preferenceに限定する。既知勝敗はroot手番視点の0/1補助targetとして保持する。
初回HKCLはtrain/validの派生child 197件がsplitを跨いだため棄却した。train HKCL確定後、valid生成時にtrainの
root/good/bad全network hashを除外し、派生入力を含むsplit衝突0を必須とする。pilot内部testは変換しない。

### 15.4 Phase 3中間監査とGate再校正（2026-07-23）

#### 結論

scratch学習そのものは成立している。accepted Phase 1 parentに対し、exact family精度は47.42%から
約90.0～90.4%、wdoor soft pair精度は50.29%から60.65～62.24%へ改善した。一方、wdoor勝敗BCEは
0.69329から0.739前後へ悪化しており、棋譜の順位信号と勝敗信号を一括して「成功」とは判定できない。

従来の「pre-ClippedReLU値が0未満または1超となる割合を、黒白各々1%未満」というGateは撤回する。
ClippedReLUは非線形性を作る層であり、0側のclipは疎性でもある。全値を`[0.02, 0.98]`へ押し込むと、このrepoの
1 hidden layer構成はほぼ線形化する。実測でもanchorはclip率だけを改善したが、offline品質の明確な改善を示して
いない。これは安全Gateが厳しいというより、測っている量と品質リスクの対応付けが誤っていた。

activationは次のように扱い直す。

1. 全体clip率はhard Gateでなくdiagnosticとし、低側と高側を別に報告する。
2. channel別のpreactivation分位点、分散、interior出現率、常時0側/1側にclipされるchannel数を追加する。
3. health Gate v2をtest開封前に固定する。hard failureはNaN/Inf、forward不一致、量子化範囲違反、channel別崩壊、
   異常score分布で判定し、全体率だけの閾値は使わない。旧上側1%は暫定alertとして残し、探索品質との相関を検証する。
4. activation anchorは実験用optionとして残すが既定値0とし、clip率をGateへ合わせる目的では使わない。
5. v4（2 epoch、anchorなし）とv5（1 epoch、anchorなし）を、v6/v7（anchorあり）と同じstatic/fixed-node
   searchで比較し、clip率と実探索品質の関係を確認するまでは候補を昇格しない。

exact開発Gateの`point >= 90%`は厳しすぎないため維持する。v4/v5/v7は同じvalidで既に通過し、v6の
89.993%だけを丸めて救済する必要はない。family bootstrap下限88%以上、3 seedの方向一致、static/search非退行も
同時に要求する。Phase 2本目標95%も維持するが、小数点境界を跨ぐためのhyperparameter探索は行わない。
testを開いた後に基準を変えない。
dataset integrity、proof再検証、split衝突0、lineage、NaN/InfのGateは従来どおりzero toleranceとする。

#### 学習器・評価器で先に直す事項

1. HKCLの`confidence`が現在trainerで使われていない。確実な最終結果用`result_weight`と、不確実な棋譜手用
   `preference_weight * confidence`をschema/損失上で分離し、finite-differenceとweight 0/1のtestを追加する。
   修正前のv4～v7候補はすべてdiagnostic/rejectedに限定する。
2. 現在のbest epochはexact+gameの平均validation lossで選ばれ、healthを含まない。epochごとのmodelを保存し、
   integrity/safety -> exact非劣性 -> game rank/BCEのPareto順で選ぶ。validationはtrainと異なりfamily正規化を
   行っておらず、activation penaltyもloss分子だけに入るため、現scalar lossを候補間比較に使わない。
3. game評価へresult prevalence baseline、Brier、ECE、phase/opening別BCEを追加する。BCEは少なくとも定数予測と
   accepted parentを悪化させないことをpromotion条件にし、親との差をpaired game-family bootstrapで出す。
4. 棋譜手対quiet 1手は証明済みの良悪ではないため、`bad`でなく`sampled alternative`と表記する。
   rank-only、result-only、rank+resultを同じsplitでA/Bし、confidenceとsource weightを別々に校正する。
5. 2,000局pilotを最終品質の判定に使わない。上記修正後、2023～2025 wdoorからopening/year/ratingを層化した
   約250k positionへ拡張し、game-family bootstrapで学習曲線を測る。2026と最終testは引き続き開かない。
6. 現在のfamily正規化後の名目量は概算でexact 64,000、health 5,000、game約152で、gameは約0.2%にすぎない。
   objective-stratified weightingを導入し、実効weight、objective数、gradient norm、更新回数を種類別にmanifestへ
   記録する。20/40/40は初期A/B比率であり、raw record数で達成したと見なさない。
7. black/white集計だけでなくSTM/NSTM、hidden channel、king bucket、phase別activationを出し、親との差もpairedで
   評価する。clip率を改善しただけのmodelを品質改善と呼ばない。

#### 改訂後の直近手順

1. activationのchannel別診断、weight/confidence分離、objective-stratified weighting、目的別validation/epoch保存を実装してtestする。
2. 実効source weightを事前固定し、同じpilotでrank-only/result-only/rank+resultを1 epoch、3 scratch seedで比較する。
3. anchorなしv4/v5相当とanchorあり候補へstatic all-legalと200k fixed-node searchを実施する。
4. exact point 90%以上かつbootstrap下限88%以上、health p99 2,000 cp未満、game rankがparentより改善、
   game BCEが定数予測に非劣性、search非退行を同時に満たす構成だけを250kへ進める。
5. 250kで改善区間が再現した後にのみgeneration 2 promotion matchへ進む。未達候補を教師にしない。

#### 実施記録（2026-07-23）

- trainerで棋譜の`preference_weight * confidence`と`result_weight`を分離し、各々を0にしたA/Bを可能にした。
- exact/symmetry、棋譜手preference、最終resultの実効nominal lossを事前指定比へ正規化するobjective mixを追加した。
  family数でなく、family内objective正規化、sample weight、confidence、score anchorを反映して倍率を計算し、
  run logとmanifestへ残す。
- 異種scalar validation lossで親modelが選ばれるのを避けるため、診断run用のfinal epoch保存を追加した。
- evaluatorへ勝率prevalence、定数予測BCE、Brier、10-bin ECEと、black/white/STM/NSTM別の64 channel診断を追加した。
- wdoor validの勝率は52.25%、定数予測BCEは0.69213、accepted Phase 1 parentは0.69329だった。
- scratch training関連12 testと全lib 85 testが合格した。次は同一pilotの3 objective構成・3 seed A/Bである。

#### objective A/Bとwdoor 250k結果（2026-07-23）

2,000局pilotでrank-only/result-only/combinedを分離した。20/40/40 combinedはwdoor順位80.28%、BCE 0.63496へ
改善したがexact 87.06%へ退行した。比率を限定A/Bし、exact/symmetry 70%、棋譜手15%、result 15%、2 epochで
3 seedすべてがpilot Gateを通った。exact 90.01～90.32%、wdoor順位78.66～79.25%、BCE 0.62367～0.62832、
health p99 1,485～1,534 cpだった。200k searchは全体98.2～98.7%、攻撃96.4～97.4%、防御100%で、旧exact-onlyを上回った。

wdoor 2023～2025から12,237局、241,177 unique position（train 193,070 / valid 25,057 / 未開封test 23,050）へ
拡張した。game/opening-family/network split衝突、split内重複、違法教師手、provenance不一致はすべて0だった。
HKCL変換後はvalidからtrainおよびexact/healthのroot/good/badを除外し、合同432,124 recordでもfamily/network
split衝突0、proof再検証失敗0を確認した。

250k、70/15/15、2 epochの3 seed結果は次のとおりである。

| seed | model SHA-256 | exact family | 95%下限 | wdoor family | BCE | health p99 | static | search 200k |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 20260721 | `9a1bfc9b...b3db6` | 90.81% | 89.62% | 80.56% | 0.59728 | 1,673 cp | 70.0% | 98.6% |
| 20260722 | `74d4cc6c...6d93c` | 90.41% | 89.20% | 81.09% | 0.58840 | 1,539 cp | 70.1% | 98.6% |
| 20260723 | `f30a9d85...fdd3c` | 90.86% | 89.69% | 81.02% | 0.59202 | 1,607 cp | 69.5% | 98.6% |

全seedでexact point 90%以上、bootstrap下限88%以上、wdoor定数BCE 0.69265より改善、health p99 2,000 cp未満、
search非退行を満たした。activationは全channelがinteriorを使用しNaN/Inf・channel崩壊はないが、一部channelの
上側clip集中が進むため診断値として継続監視する。1 epochは集中が少ない一方seed 20260722のexactが89.76%で、
固定3 seed Gateを満たさなかった。したがって2 epoch構成をpromotion match候補として固定するが、match完了前に
generation 2へaccepted登録せず、探索教師にも使わない。

#### generation 2 promotion match（2026-07-23）

事前固定したseed 20260721候補を、同seedのaccepted Phase 1 parentと比較した。対局器はmodel manifestの
scratch lineageと実ファイルSHA-256を照合し、parentがimmutable promotion registryにaccepted登録済みであることを
必須とする。openingは未開封testを使わず、wdoor validからply 8～40のunique opening familyを入力順に選んだ。
各openingは同一SFENでcandidateの先後を入れ替え、双方を20,000 fixed nodes/手、最大256手で探索させた。
千日手・連続王手、合法手、探索失敗、root欠落、非有限scoreを対局器内で判定・記録する。

20局smokeはcandidate 19勝 / parent 1勝、100局screenは95勝 / 5勝だった。最終100 opening pair・200局は
candidate 192勝 / parent 8勝 / 引分0、score 96.0%、Wilson 95%下限92.31%だった。candidate先手97勝、後手95勝で、
探索失敗、root欠落、違法手、NaN/Inf、最大手数引分、千日手はいずれも0だった。事前Gateのpoint score 50%以上、
実用非劣性下限47.5%以上、安全異常0をすべて満たした。

全200局を含む最終report SHA-256は`9557bfc78e9470689c93a96b480b9144bca7421ce83d0ef947741a71bb76a2b7`、
Git管理するdecision report SHA-256は`d006cc7a8d43a6ba22e4e8b8ede179707d43436f1296651e8bf3f7874ad3b18a`である。
model `9a1bfc9b7b317f451fa9a6efb350fed0a50e3020659686cefccc98646b9b3db6`をgeneration 2としてregistryへ
`accepted_promotion_match`登録した。以後の探索教師生成は、このaccepted modelからのみ開始できる。残る2 seedは
再現性確認用diagnosticであり、個別のpromotion matchなしにacceptedとはしない。

#### Phase 5生成性能監査と段階規模の修正（2026-07-23）

generation 2からdepth 2 search teacherを100局面生成するスモークは5分超で完了せず、CPU使用率も約1 coreに
低下したため中断し、不完全な一時artifactを破棄した。Rayon自体の無効化ではなく、1 rootにつき全合法手の独立
full-window探索と最大12候補の再探索を行い、各回TTを消去する構造、およびchunk barrier末尾のheavy-tail rootが原因である。
このthroughputで250kを直接開始しない。

Phase 5は次の順へ修正する。

1. 各探索へdeterministic fixed-node上限を設け、cap到達rootは部分scoreを教師にせず全体を除外して集計する。
2. rootごとのsearch call、nodes/qnodes、wall timeのp50/p95/maxを記録し、固定size shardをatomic保存・resume可能にする。
3. 64 rootのphase・opening family・legal-move数層化profile、128 rootの候補recall比較を先に行う。
4. 初期候補はselection depth 1、teacher depth 2、candidate top 4とし、game/tactical/random候補を保持する。
5. jobs 1/2/4で同一seedのbyte一致を確認後、train 1,024 / valid 256、4k/1k、16k/2kのlearning curveへ進む。
6. 64kで改善と全Gateを再現した場合だけ250kを再検討する。規模拡大を成功条件にしない。
7. 初回学習はparentのabsolute search cp回帰を無効化し、search順位signalだけを既存70/15/15 replayへ追加する。

正式生成前に、search teacherとsearch trainerはaccepted model・Gate report・generationのregistry照合に加え、実model
manifestのlineage、model SHA、generation、parent chainをparse検証する。clean commitからrelease binaryを再buildし、
engine binary SHAも固定する。候補は従来のexact、game rank/BCE、health、static、200k searchをすべて再通過し、
parentとのpromotion matchに勝った場合だけ次の教師にできる。

実装後のprofileでは、train全域をstride samplingした64 rootをselection depth 1 / teacher depth 2 /
top 4 / 5,000 nodesで約1.2秒処理し、50件を書き、cap到達14件を教師から除外した。128 rootの全合法手比較では
top 4＋補助候補のdeep-best recallが61.45%、top 8で75.90%、top 16でも86.75%に留まったため、depth 1
preselectionは採用しない。fixed-node化後はselection depth 2とteacher depth 2を同一にして全合法手のscoreを
再利用でき、128 rootを約1.7秒で処理できた。これをPhase 5初期構成とする。jobs 1/4で同じ32 rootから生成した
HKST SHA-256はともに`7e84edb78f2918b21693e09ca64a875e16344c830b3a6aa1dd8c86dbc2dbdf50`で、
dataset byte一致を確認した。正式pilotではcandidate top 8、5,000 nodes/探索、cap root全体除外を用いる。

最初のrank-only pipeline pilotはtrain 1,001 / valid 247 recordを生成した。探索失敗は0で、5,000-node capへ
到達したrootはtrain 599、valid 153を丸ごと除外した。accepted generation 2から`value_weight=0`、search rank
のみ、低学習率、1 epochで学習した結果、search valid pairは67.20%から69.30%、top-1は16.19%から17.81%へ
改善した。exactは90.50%（95%下限89.28%）、BCEは0.59718、health p99は1,093 cpだった。一方、wdoor順位は
80.56%から79.58%、staticは70.0%から68.6%、200k searchは98.6%から97.9%へ退行した。このmodel
`b61484ecf2a5caf7d1763f50ab30087ccd5b23a266007e2390bd52e6cd3909d8`はGate失敗としてregistryへrejected登録し、
教師にも初期値にも使わない。次のpilotはaccepted generation 2へ戻り、search rankを単独fine-tuneせず、
70/15/15 replay objectiveと同一更新内で混合できるtrainerへ統合してから行う。

### 15.5 現行v2.5.0重みを上回るための短縮計画（2026-07-26）

#### 判断

計画を修正する。直前の「search rankと70/15/15 replayを統合するtrainerを先に実装する」という方針は停止する。
実装可能性ではなく、`policy_weights_halfkp64_kpp_distilled_v2.5.0.binary`を実対局で上回るまでの距離を
最優先する。

accepted generation 2はPhase 1 parentには200局で192勝したが、これは目標重みとの比較ではなかった。
同一engine、同一opening、先後交換、100 ms/手、Threads 1でv2.5.0重みと20局screenを行った結果は7勝13敗、
score 35.0%（95%区間14.1～55.9%）だった。20局だけで差を確定はできないが、現時点で「現行重みより強い」
という証拠はなく、少なくとも昇格親との勝率を目標達成の代理にはできない。

Phase 5のrank-only pilotもsearch教師上のpair/top-1だけを改善し、wdoor順位、static、200k searchを悪化させた。
弱い世代自身のdepth 2順位を増やす前に、人間が設計した追加基盤なしで利用できる高品質棋譜を使い切る。

2023～2025 WDoorには両対局者のrateが判明した282,492局があり、両者3,500以上は107,951局、4,000以上は
51,791局ある。一方、generation 2に使った241k datasetにはrate下限がなかった。したがって、次の一手は
データ量の無条件拡大でもsearch bootstrapでもなく、既存`dataset_build`による高rate棋譜への選別である。

#### 最短実行順

1. 現行重みは初期値にも教師にも使わず、比較相手としてだけ固定する。SHA-256は
   `91784d6f03f70580468a1509f968cbaaf598fbce084c22a739cb870344fb7c00`である。
2. 2023～2025 WDoorから両対局者rate 4,000以上、既知の決着局、1局最大24局面、opening/middle/lateを
   8/12/4、ply 8～160、合法手2以上、非王手局面を条件に約1M局面を作る。敗者のply 80以降は除外する。
   content-hash 80/10/10 splitを維持し、2026とtestは開かない。
3. accepted generation 2から、既に3 seedで検証済みの70/15/15、2 epoch、seed 20260721を1候補だけ学習する。
   exact/symmetry replayを残し、KPP由来model・score・探索結果は一切入力しない。
4. integrity、NaN/Inf、exact 90%かつ95%下限88%、health p99 2,000 cp未満、static/200k searchの親非退行だけを
   安全Gateとする。offline値だけで棋力改善とは判定しない。
5. 安全Gateを通った候補を、v2.5.0重みと20局・100 ms/手でscreenする。score 45%未満なら直ちに棄却する。
6. screen通過時だけ100局・300 ms/手へ進み、score 48%未満なら棄却する。通過時だけ1,000 ms/手の
   200局batchを行う。最大800局まで事前条件を変えずに追加し、scoreの95%下限が50%を超えた時点で
   「現行重みより強い」を達成とする。95%上限が50%未満なら停止する。
7. 最初の候補が安全Gateを通るが対局screenに失敗した場合だけ、同じdatasetでexpert比率を増やす
   60/25/15を1回だけ試す。これも失敗した場合に限り、search rankとreplayのtrainer統合を再検討する。

#### 当面実施しないもの

- depth 3～5 search teacherの大量生成
- search/replay統合trainerの先行実装
- self-play、TD、league、SWA
- 3M dataset、複数seedの全組合せ
- 新しいschema、dashboard、追加Gate、activation閾値調整

これらは1M高rate棋譜の最大2候補が直接対局で失敗するまで保留する。最終目標に寄与する証拠がない工程を、
完全性や将来拡張性だけを理由に追加しない。

#### 実行結果（2026-07-26）

WDoor 2023～2025から両対局者rate 4,000以上の44,228局を採用し、723,867局面を生成した。
content-hash splitはtrain 570,875、valid 90,759、未開封test 62,233である。既存exact、health、
旧WDoor train/validを除外してHKCLへ変換した結果はtrain 537,723、valid 80,807だった。合同1,050,654
recordの監査でopening family/network inputのsplit衝突は0、proof再検証失敗も0だった。最初に異なる
split seedで作ったdiagnostic datasetには旧集合とのnetwork input衝突3,790件があったため使用せず、
既存splitと同じseedで再構築した上でHKCL変換時のcontent exclusionにより衝突を解消した。testは開いていない。

事前計画の2候補を同じaccepted generation 2から学習した。結果は次のとおりである。

| 配分 exact/expert/result | model SHA-256 | exact family | 95%下限 | high-rate family | broad WDoor family | health p99 | static | search 200k | v2.5.0 20局 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 70/15/15 | `23c2a650...a4319` | 91.01% | 89.85% | 85.24% | 83.22% | 1,754 cp | 71.5% | 98.7% | 4勝16敗、20% |
| 60/25/15 | `1db4dc2b...76bc5` | 90.84% | 89.65% | 87.68% | 85.10% | 1,943 cp | 70.6% | 98.8% | 6勝14敗、30% |

両候補ともintegrity、exact、health、static、searchの安全Gateを通過した。expert比率増加は未使用WDoorの
順位精度と直接対局を改善したが、事前固定したscreen条件45%（9/20）には達しなかった。両modelをregistryへ
rejected登録し、教師・初期値・accepted generationには使用しない。現行v2.5.0重みは学習入力に一切使用せず、
SHA固定した比較相手としてのみ使った。

この結果により15.5の高rate棋譜最大2候補は完了した。棋譜の選別と目的配分だけをさらに微調整する経路は停止する。
次に再検討する対象は、既に保留していたsearch rankと70/15/15 replayの同時更新である。ただし、rank-only pilotの
退行を避ける最小実装と小規模A/Bを先に行い、直接対局へ届かない大規模教師生成や追加Gateへは広げない。

#### search rank/replay同時更新の実装（2026-07-26）

`halfkp_scratch_train`へ`--search-train`、`--search-valid`と第4目的`source_mix_search`を追加した。
HKSTのroot絶対値とgame resultは使用せず、探索候補のscore差が15 cp以上の最善手対劣後手を最大4 pairだけ
rank lossへ変換する。exact/expert/result/searchは同一family shuffleと同一batch更新へ混在し、指定した
4要素の実効目的比率へ校正する。search教師manifestはHKST0002の現行意味論に加え、教師model SHA-256が
宣言されたaccepted scratch parentと一致しなければ拒否する。このため現行KPP由来重みや別系列の探索教師は
混入できない。

単体10件とlibrary全89件はすべて合格した。既存search pilot train 1,001件とreplay 1,000件を使った
統合スモークは1 epochを完走し、合同valid lossは0.32571746から0.32431975へ低下した。同一seed・条件の
2実行はmodel SHA-256 `5332508daec1dd131f288c53e841a2f5b1e7b54203503fd0627d29e8fbdf5f81`で
byte一致した。

search 0%の70/15/15 controlと、replay比を保ったままsearchを5%にした66.5/14.25/14.25/5の小規模A/Bでは、
exact familyは90.51%対90.55%、health p99は1,688対1,690 cpで維持した。一方、広域WDoor familyは
80.53%対80.06%でsearch混合側が0.47 point低かった。したがって統合実装そのものは採用するが、5%配分modelは
昇格候補にせず教師にも使わない。次は既存1k/247 search教師のまま、search比率を1～2%へ下げた1 epoch pilotを
1候補だけ行い、search holdout改善とWDoor非退行を両立した場合だけ安全Gateと直接対局へ進める。

#### search 1%単一候補の結果（2026-07-26）

accepted generation 2のmanifestに記録された5 train / 2 valid replay集合をそのまま使い、既存search教師
train 1,001 / valid 247件を追加した。replay内70/15/15を保つ69.3/14.85/14.85/1、1 epoch、learning rate
0.001、seed 20260721の1候補だけを学習した。合同valid lossは0.07221626から0.07148750へ改善した。

model `06d88946384926be72d4f9917bd56fbf0f4db6f07140e56c15ed8bcddc3b8442`はexact family
90.68%（95%下限89.48%）、WDoor family 81.00% / BCE 0.59452、health p99 1,738 cp、static 70.4%で、
親のWDoor 80.56% / BCE 0.59728、static 70.0%を改善した。しかし200k searchは98.5%で親98.6%を
1 root下回り、事前固定した親非退行Gateを満たさなかった。

この候補はregistryへrejected登録し、v2.5.0との直接対局、教師生成、次世代の初期値には使わない。
5%小規模A/Bと1%正式pilotの双方で探索順位signalによる明確な総合改善を示せなかったため、同じ1k教師で
2%などの比率探索は行わない。次の改善には比率微調整ではなく、探索教師の候補recall・深さ・対象局面の
情報量を上げる必要がある。ただしCPU制約下で大規模生成を始める前に、既存教師の失敗rootを分析して
小さな再生成条件を決める。

### 15.6 学習量監査と長時間direct学習への修正（2026-07-26）

#### 結論

計画を修正する。15.5末尾の「次は探索教師の情報量を上げる」という判断は保留し、既存データによる
長時間direct学習を先に行う。短時間pilotは実装、目的関数、scale、安全性を検証する役割は果たしたが、
現行v2.5.0を上回るための学習量比較が欠けていた。

`data/halfkp64_kpp_training.csv`によると、v2.5.0は1 epochあたり32,117,302 sampleを12 epoch、
合計385,407,624 sample提示で学習している。これに対し、accepted scratch generation 2へ至る主要な
Phase 2～3は68,000 + 357,070 x 2 = 782,140 record提示にすぎず、約493分の1である。高rate Phase 4を
含む最長の不採用系列でも2,185,586 recordで約176分の1しかない。小規模exact等を加えても桁は変わらない。

さらに、これまでの正式scratch候補は`factor-only`だった。runtimeのdirect feature embeddingは
6,750,720 parameterだが、piece/relative factorは667,904 parameterで約10分の1である。factorizationは
sparse featureへ初期知識を広げる土台として有効だったが、最終性能を目指す全容量学習ではない。

したがって「数分～十数分、1～2 epochのfactor-only候補が50時間超のdirect modelへ届かなかった」ことから、
棋譜教師やscratch方針そのものを否定することはできない。単純な学習量不足は可能性の一つではなく、
現時点で最初に解消すべき主要因である。

#### Gate運用の修正

これまでGateを早期実験の打ち切り条件として使いすぎた。1000 root中986対985という1 root差には、
棋力非退行を断定する分解能がない。20局screenも信頼区間が広く、未成熟checkpointを棄却する根拠には弱い。

以後、Gateを次の2種類に分ける。

1. 学習継続を止める安全異常はNaN/Inf、違法出力、明確なscale崩壊、health p99 3,000 cp以上などに限定する。
2. 対局候補にできるmilestoneはexact 90%以上かつ95%下限88%以上、health p99 2,000 cp未満、
   static 68%以上、200k search 98.0%以上とする。親との差1 rootだけでは棄却しない。

offline Gateは棋力の代理ではない。最終判断はv2.5.0との同一engine・先後交換対局で行う。

#### 最短の長時間学習計画

新しいtrainer、schema、探索教師、self-playは作らず、既に監査済みのデータとcheckpoint機能を使う。

1. 初期値はaccepted generation 2
   `9a1bfc9b7b317f451fa9a6efb350fed0a50e3020659686cefccc98646b9b3db6`へ戻す。
   rejected Phase 4/5 modelとv2.5.0重みは初期値・教師・targetに使わない。
2. trainは既存high-rate 4,000 WDoor 537,723件とexact/symmetry/health replayを使う。
   2候補screenで相対的に良かったexact/expert/result 60/25/15を固定し、search rankは混ぜない。
3. `factorization=direct`、learning rate 0.001、既存weight decay 0.01、output LR scale 0.4、
   score anchor 2を固定する。hyperparameter sweepは行わない。
4. 同じoptimizerを保持して最大256 epochまでresumeする。milestoneは4、8、16、32、64、128、256 epochとし、
   発散しない限り最低32 epochまでは短い対局結果だけで停止しない。
5. 1 epochは701,723 record、約1,541,548 objectiveである。32 epochでrecord約22.5M /
   objective約49.3M、128 epochで約89.8M / 197.3M、256 epochで約179.6M / 394.6Mとなる。
   256 epochのobjective提示量はv2.5.0のsample提示量と同程度であり、この上限で初めて学習量について
   比較可能になる。現在の約8分/epochを基準にすると上限は約34時間だが、時間そのものではなく提示量と
   learning curveを管理する。
6. 各milestoneでtrain/valid loss、exact、WDoor順位/BCE、health、static、200k searchを保存する。
   epoch 4～16の20局・100 ms対局は傾向診断にだけ使い、棄却判定には使わない。
7. epoch 32以降で安全Gateを通るcheckpointを40局・100 msで比較する。scoreが前milestoneから改善中なら
   次へ進む。validationと対局scoreが2 milestone連続で改善せず、かつv2.5.0との差が残る場合だけ停止する。
8. 成熟checkpointが40局screenで40%以上なら100局・300 msへ進み、48%以上なら1,000 ms・200局batchへ
   進む。最大800局で95%下限が50%を超えた場合のみ「v2.5.0より強い」とする。
9. 32～64 epochで明確にplateauし、direct容量を使っても40%screenへ近付かない場合に限り、
   WDoor rate 3,500以上へ2～3M局面まで拡張する。探索教師再設計はその後であり、先に実施しない。

#### 当面実施しないもの

- search rank比率1～5%の追加探索
- depth 3～5教師の大量生成
- self-play、TD、league、SWA
- 新しいdataset schema、dashboard、追加Gate
- 複数seed・複数learning rateの総当たり

最短経路は、既に動作確認済みのscratch目的をHalfKP全容量へ十分な回数適用し、learning curveと直接対局を
観測することである。教師方式の追加は、この単一長時間runで学習量不足を除外してから判断する。

### 15.7 長時間direct学習 epoch 4～16 評価記録（2026-07-27）

accepted generation 2を初期値とする`phase6-long-direct-v1-seed20260721`はepoch 16まで正常完了した。
validation lossはepoch 4の0.04662774、epoch 8の0.04613198、epoch 16の0.04574677と一貫して改善し、
NaN/Inf、checkpoint破損、探索失敗は発生していない。epoch 16 model SHA-256は
`6e5d3727c414409dd189717669067e943cabd3b978b272f57e1990fb325a0bee`である。

同一holdoutによるmilestone比較は次のとおり。

| 指標 | epoch 4 | epoch 8 | epoch 16 | 判定 |
|---|---:|---:|---:|---|
| exact family accuracy | 90.76% | 91.12% | 91.17% | 改善、候補基準通過 |
| exact 95% CI lower | 89.57% | 89.96% | 90.01% | 改善、候補基準通過 |
| WDoor high-rate family accuracy | 84.68% | 85.25% | 85.72% | 改善 |
| WDoor broad family accuracy | 83.03% | 83.42% | 83.83% | 改善 |
| health score abs p99 | 1999.65 cp | 2068.66 cp | 2130.83 cp | 候補基準外、継続基準内 |
| static exact top-1 | 70.9% | 71.4% | 70.8% | 候補基準通過、横ばい |
| search 200k | 98.8% | 98.9% | 99.0% | 改善、候補基準通過 |
| v2.5.0対20局score | 10% | 25% | 10% | 診断のみ、未成熟 |

WDoor broad BCEはepoch 8の0.67099からepoch 16の0.67852へ悪化しており、順位正解率の改善と確率校正が
一致していない。health p99も上昇しているため監視を続ける。一方、停止基準の3000 cpには余裕があり、
exact、high-rate順位、broad順位、searchは改善している。20局screenはepoch 4と16が10%、epoch 8が25%で
信頼区間が広く、学習曲線の単調性を判定できない。

したがって計画は変更せず、同じoptimizerとhyperparameterでepoch 32まで継続する。epoch 32では全offline
評価に加えて40局・100 ms screenを行う。health p99が3000 cpへ到達する、数値異常が出る、または探索が
明確に崩壊する場合を除き、それ以前に短い対局結果だけで停止しない。

epoch 32は正常完了し、valid lossは`0.04543033`まで改善した。exact familyは90.63%（95%下限89.42%）へ
低下した一方、WDoor high-rateは86.10%、broadは84.13%へ改善した。health p99は2155.18 cp、staticは
69.7%、200k searchは98.9%であり、継続停止基準には該当しない。40局・100 ms診断は9勝31敗（22.5%、
95% CI 9.56～35.44%）で、v2.5.0超えは確認できなかった。ただしこの診断は中間checkpointの傾向観測用
であり、offline指標の一部が改善中でhealthも安全範囲内のため、計画を変更せずepoch 64へ進める。

epoch 64も正常完了し、valid lossは`0.04533218`となった。exact familyは90.09%（95%下限88.84%）へ
さらに低下したが、WDoor high-rateは86.40%、broadは84.53%へ改善し、health p99は2086.05 cpへ低下した。
staticは69.1%、200k searchは98.9%である。40局・100 ms診断は6勝34敗（15.0%、95% CI 3.93～26.07%）で、
短時間対局の改善は確認できない。offlineの一部改善と停止基準未該当を理由にepoch 128まで進めるが、
epoch 128でexact/static/対局が改善しない場合は、単純な追加epochだけでの打開を再検討する。

epoch 128でこの長時間direct系列を終了した。valid lossは`0.04590280`へ悪化し、exact familyは89.29%
（95%下限87.98%）、staticは67.6%、200k searchは98.8%へ低下した。high-rate familyだけは86.53%まで
改善したが、broadは84.50%で頭打ちだった。health p99は1898.51 cpまで改善した。40局・100 ms診断は
3勝37敗（7.5%、95% CI 0～15.66%）であり、v2.5.0を上回る根拠はない。

この結果から、単純な同一データ・同一目的関数のepoch延長は採用しない。学習系列内の比較対象はexactと
staticが最良のepoch 16、40局診断が最良のepoch 32であり、いずれもリリース候補ではない。次の作業は
新しい長時間runを無条件に開始することではなく、(1) epoch 16/32/64/128の重みスケールと目的別損失の
乖離を解析し、(2) direct評価と実対局の不一致を解消するデータ・目的関数の変更を小規模に検証し、
(3) その変更が確認できてから新しい長時間学習へ進むこととする。

### 15.8 長期学習後の戦略改訂: WDoor MMTO-lite（2026-07-28）

#### 結論

同じHKCLをepoch 256まで反復する計画は中止する。次は、WDoor棋譜手とscratch探索が実際に競合する手を
探索木のleafで直接比較するMMTO-liteを実施する。現行v2.5.0は引き続き対局比較だけに使い、初期値、
教師score、negative mining、anchorには一切使わない。

次の初期値はpromotion済みgeneration 2
`9a1bfc9b7b317f451fa9a6efb350fed0a50e3020659686cefccc98646b9b3db6`へ戻す。epoch 16/32/64/128は
原因解析用に保存するが、未昇格であり次世代parentにはしない。

#### 失敗原因の優先順位

1. 現行`game_soft_expert_v1`は、棋譜手に対して「決定論的に選んだ静かな合法手」1手だけをalternativeに
   している。quiet手が存在すると捕獲、王手、駒打ち、成りをnegative候補に含めない。そのためWDoor pair
   accuracyは上げられても、探索中にモデルが過大評価して実際に選ぶhard negativeを矯正できない。
2. epoch 16から128でhigh-rate familyは85.72%から86.53%へ改善した一方、static attacker top-1は
   44.2%から38.6%、40局scoreは10%から7.5%へ悪化した。代理課題と棋力の不一致は統計不足ではなく、
   方向の不一致である。
3. 128 epochで約89.8M record / 約197.3M objectiveを提示した。v2.5.0の総提示量には届かないが、
   valid lossはepoch 64以降悪化し、exact/static/対局も同時に悪化した。同じ537,723局面の反復を増やす
   だけでは、新しい意思決定情報は増えない。
4. result BCEは改善しても対局へ転移しなかった。最終勝敗の絶対値回帰は補助信号としては使えるが、
   次の主目的にはしない。

Bonanza/MMTOの要点は、静的なランダム手pairではなく、現在の探索が選ぶ手と棋譜手の不一致を探索leaf上で
最小化することにある。本repoの`halfkp_search_teacher`は既に全合法手の浅いpreselection、上位候補、
tactical候補、棋譜手、深いPV leaf、`CANDIDATE_GAME_MOVE`と`CANDIDATE_SEARCH_BEST`をHKSTへ保存できる。
不足しているのは、この2候補を直接比較する損失だけである。

#### 実装範囲

新しいbinary、dataset schema、optimizer、dashboardは作らない。`halfkp_search_train`へ次だけ追加する。

1. `--game-vs-search-weight`（既定0）を追加する。
2. 棋譜手とscratch search-bestが異なり、parent探索でのregretが上限内のrecordについて、棋譜手のPV leafを
   good、search-bestのPV leafをbadとするrank lossを1 pair追加する。
3. このlossが有効なrecordでは、通常のsearch-best対劣後手lossから棋譜手を除外し、同じpairを逆向きにも
   学ぶ相殺を防ぐ。その他のhard negativeには従来のsearch rankを残す。
4. manifest/logへgame/search不一致数、regret 50/150/300/600 cp以下の件数、適用pair数、pair accuracyを
   記録する。
5. weight 0互換、gradient符号、regret上限、game move除外、決定性、resumeをunit testする。

これは既存HKST codecと候補探索をそのまま使う局所変更であり、過剰な基盤追加は行わない。

#### 実行計画

##### Step 1: 10k root pilot

- source: 既存のrate 4000以上WDoor train/valid JSONL
- parent/teacher: accepted generation 2のみ
- train 10,000 root、valid 2,000 rootを固定strideで抽出
- selection depth 2、teacher depth 4、1探索最大20,000 node
- candidate top 8、tactical candidate 2、棋譜手を必ずcandidateへ追加
- 別の1,000 rootだけall-legal recallを監査し、depth 4 bestの候補recall 95%以上を要求
- test splitと2026年out-of-time splitは開かない

同じHKSTで次の2候補だけを1 epoch比較する。

1. control: search rankのみ（既存実装）
2. treatment: search rank + game-vs-search。value/result lossは0、旧game-rank lossも0

初期値、seed、shuffle、learning rate `1e-5`、output learning rate `1e-6`を固定する。regret上限は
300 cpを基本とし、事前監査で適用recordが20%未満の場合だけ600 cpへ上げる。weight sweep、複数seed、
epoch sweepは行わない。

##### Step 2: pilot Gate

treatmentを次へ進めるには、以下をすべて満たす。

- NaN/Inf、違法手、provenance/split衝突0
- exact family point 90%以上、95%下限88%以上
- health p99 2,500 cp未満
- static top-1 69%以上、200k search 98.5%以上
- 固定HKST validのgame-move top-1がcontrolより3 point以上改善、またはgame/search mismatch regretが
  10%以上減少
- accepted generation 2との100局・100 ms先後交換対局でscore 50%以上
- v2.5.0との同一100局で、generation 2を同じ条件で再測定したscoreより5 point以上改善

ここで失敗した場合、比率やepochを微調整せず、このloss仮説を棄却する。

##### Step 3: 50k、250kへの段階拡張

10k pilot通過時だけ、同じ設定で50k train / 5k validへ拡張する。50k候補がv2.5.0に40%以上なら250kへ
進め、40%未満なら停止する。250kでも学習は最大2 epoch、各epochを保存して早い方をGateで選ぶ。
100 msで45%以上になった場合だけ300 ms・200局へ進み、48%以上で1,000 ms評価へ進む。

候補がgeneration 2と全安全Gateを通過し、親子対局で明確に非退行となった場合だけpromotion registryへ
登録する。昇格後に限り、その新parentからHKSTを1回再生成して第2世代MMTO-liteを行う。最大2世代で
v2.5.0との差が縮まらなければ停止し、self-play/TDではなく表現能力と探索leaf教師の再監査へ戻る。

#### 当面実施しないもの

- epoch 256への継続、同じHKCLの追加反復
- rate 3,500 WDoorを2～3M局面へ先に拡大すること
- exact/expert/result比率の細かな再探索
- 現行v2.5.0やepoch 128を教師・初期値に使うこと
- self-play、TDLeaf、SWA、league
- 新しいpolicy head、dataset schema、trainerの新設

最短経路は、既にある探索教師pipelineで「scratch engineが選びたがる手」をhard negativeとしてWDoor棋譜手へ
直接合わせ、その変更が実対局へ転移するかを10k rootで先に確認することである。

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

### 2026-07-22 Phase 3 combined pilot: activation anchor

- 以下は当日の実験記録であり、Gate判断は翌日の15.4監査で改訂した。
- anchorなし2 epoch候補はhealth activation範囲外率が下側約5.5%、上側約1.3%で不採用。
- legal-health rootのpre-ClippedReLU値を`[0.02, 0.98]`へhinge anchorする実装を追加した。
- weight 10・1 epochはhealthで下側0.42%以下、上側0.82%以下となりGate通過。exact family 89.99%、wdoor family 60.78%。
- 2 epochはexact 90.33%、wdoor 62.24%へ改善したがhealthが下側約1.9%、上側約1.2%へ再悪化し不採用。
- 当時は追加anchor A/Bを予定したが中止した。15.4のchannel別診断と目的別A/Bへ切り替え、testは未開封を維持する。

## 17. 最終判断

提案された「型から始める」方針は採用する。ただし、型を人手の絶対評価値として注入するのではなく、
ルールで証明できる局面と順位、戦型別sampling、固定testとして使う。

強い重みを作る本体は、次の3信号の組合せになる。

1. 証明可能な局所知識
2. 棋譜手と最終結果
3. 新系列自身の深い探索と自己対局

初期modelから最終世代まで現行重みとの数値的な接続を完全に断つ。これにより、旧KPP模倣の上限を
引き継がず、失敗時にも原因を各stageへ切り分けられる。
