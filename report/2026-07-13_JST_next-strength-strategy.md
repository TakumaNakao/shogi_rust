# 将棋AIを次に強くするための技術判断

- 作成日: 2026-07-13 JST
- 対象: `v2.4.4` (`f79ec42`) と `policy_weights_v2.1.0.binary`
- 目的: ノートPC CPUで、現行KPP重みを超える実戦棋力を得るための次の行動を決める

> 2026-07-13 方針更新: 外部将棋エンジンの評価値は学習に使用しない。外部教師を推奨した記述は撤回し、棋譜、自己対局、現行エンジン自身の探索、勝敗、厳密な詰み証明だけを使用する。探索改善の具体計画は `2026-07-13_JST_search-correctness-implementation-plan.md` を参照。

## 結論

次に行うべきことは、現行の密KPPを別のlossでさらに長く学習することではない。

優先順位は次の通りとする。

1. 評価基準を現行HEADと証明付き未使用棋譜局面へ更新し、paired opening + SPRT相当の採否へ変える。
2. 学習を再開する場合も、棋譜結果、自己対局WDL、現行エンジン自身の探索安定性だけを使う。
3. 短期の棋力向上は探索で狙う。最初に王手中qsearchを正しく高速化し、次にPVS、慎重なLMRへ進む。
4. 評価関数の中期本命は、差分Accumulatorと量子化を前提にした小型HalfKP系NNUEとする。
5. ノートPC適性を確認する安価な対照実験として、低ランクKPP（factorization machine）を同じ教師データで学習する。

現行TinyNNUEの棄却結果は「NNUEはこのPCに不向き」という証拠ではない。現在の実装は盤面を毎回走査し、差分Accumulatorも量子化もないため、NNUEの速度上の要点を実装していない。一方、密KPPは約890MB、2億2242万重みで、8MB L3のこのPCには大きすぎる。

## 現状から確実に言えること

### エンジンは重み以外で大幅に強くなった

現行探索は同じ `policy_weights_v2.1.0.binary` を使いながら、v2.1.0エンジンに100局で `97-2-1` を記録している。したがって、少なくともこのプロジェクトでは探索・合法手生成・時間管理の改善が、重み更新より安定して実戦力へ変換されている。

ただし、この97.5%は「プロ級」の証拠ではない。古い自分自身、同じ36局面集合、depth 5 / 100ms条件への勝率であり、絶対棋力を測っていない。

### 密KPP再学習は目的関数を変えても実戦へ移っていない

- 2023-2026高レート棋譜のguarded CE: 100局で `42-54-4`。
- MMTO-lite top128: offline regretは `76.99 -> 64.78` まで改善したが、40局で得点率51.25%。
- 100K stream、hard negative、PV sibling、feedback、DAgger: 局所指標は改善しても広いrerankまたは対局で再現しない。
- 40局で67.5%だった候補が追加100局で44%になっており、現行の短ベンチ採用は分散が大きい。

これは単なるepoch不足ではない。主因は次の組み合わせである。

1. 教師が主に同じKPP評価の1-2 ply深い探索で、評価知識の上限を超えにくい。
2. 2億超の疎パラメータに対し、100K程度の探索教師では被覆が不足する。
3. 棋譜手一致、静的rerank、浅い探索regretと、時間制限付き実戦勝率の相関が弱い。
4. 候補が壊した少数局面を直す更新が、共有KPP特徴を通して別局面を動かす。

### 現行TinyNNUE実験は方式評価として不十分

現在の `TinyNnueModel` は、駒特徴embedding、両玉bucket embedding、駒得を加算してclipped ReLUへ入れる小型モデルである。標準的なHalfKPの「自玉位置 x 各駒」入力ではなく、差分Accumulatorも持たない。

実測ではKPP単体が約28万 eval/s、KPP + H64 residualが約15万 eval/sで、探索速度が約37%落ちた。しかし、これは毎回81マスを走査しfloat演算する現在の実装コストである。NNUE本来の、少数の入力差分だけで先頭層を更新する条件ではない。

また、学習データが512局面、教師depth 4では、探索葉・終盤・王手局面の分布を支えられない。2勝17敗1分という結果は、このモデルとデータを棄却する根拠にはなるが、差分NNUE全体を棄却する根拠にはならない。

### 再現可能性がまだ不足している

現ワークスペースにはWdoor CSA原本約7.7GBと報告書はあるが、報告書が参照するMMTO JSONL、学習CSV、run manifest、候補重みは残っていない。READMEと報告書には別環境の `/home/nami_ride_trade` や `/tmp` も残る。

重みもファイル名は `v2.1.0`、依頼上の呼称はv2.1.1で一致していない。現ファイルのSHA-256は次である。

```text
8d2ad6ebd65afd9bdd921f7c03205582421f00cbe2c83ccbda984fbbe13747b3
```

今後はバージョン名ではなく、engine commitとweight SHA-256の組をbaseline IDにする必要がある。

## 方式比較

| 方式 | 棋力上限 | CPU学習適性 | CPU推論適性 | 現状からの距離 | 判断 |
| --- | --- | --- | --- | --- | --- |
| 現行密KPPの追加学習 | 中 | 低 | 中 | 近い | 本線から外す |
| 現行TinyNNUE/residual | 低-中 | 中 | 低 | 実装済み | 現形は終了 |
| 差分HalfKP小型NNUE | 高 | 中 | 高 | 中 | 中期本命 |
| 低ランクKPP/FM | 中 | 高 | 高 | 中 | 安価な対照実験 |
| 手作り評価項追加 | 低-中 | 高 | 高 | 近い | 敗因が明確な時だけ |
| full dlshogi/AlphaZero | 非常に高 | 非常に低 | 低 | 遠い | この資源では不採用 |

### 差分HalfKP小型NNUE

初回は次の程度に抑える。

- 入力: 各視点の `自玉81 x 駒状態約2300`。両視点のAccumulatorを保持する。
- feature transformer: H=64。共有重みなら約1200万パラメータ、float32で約50MB、int16で約25MB。
- 後段: `2x64 -> 32 -> 1` 程度。
- 推論: int16 accumulator、int8/int16重み、AVX2。玉移動時だけ全再計算し、通常手は駒差分だけ更新する。
- 教師: 棋譜結果、自己対局WDL、現行エンジン自身の探索安定局面の混合。棋譜手one-hotを主目的にしない。

このサイズなら15GB RAMに収まり、現KPPの890MBよりキャッシュ適性が高い。学習速度はGPU版より遅いが、10万局面pilotから始め、通過後だけ100万局面へ増やせる。

### 低ランクKPP / factorization machine

玉bucketごとの駒embedding `v[k,p]` を持ち、駒ペアの重みを内積で表す。

```text
score = KP線形項 + 1/2 * (||sum(v[k,p])||^2 - sum(||v[k,p]||^2)) + material
```

rank 16なら概算約300万float、約12MBである。未出現の三駒関係にもembedding共有で一般化し、計算量は駒数の二乗ではなく `駒数 x rank` になる。差分更新も容易で、ノートPC上の学習・推論対照として有用である。

ただし、強豪将棋エンジンでの実績はHalfKP NNUEより弱い。これを本命に固定せず、同一教師・同一gateでH64 NNUEと比較し、短期間で切る。

## 探索側の優先課題

現行の主探索は、全ての子をfull-window alpha-betaで読む。PVS、LMR、futility pruningは実装されていない。null moveは一度大きくノードを減らしたが40局で悪化したため、すぐ再試行しない。

### 1. qsearchの正しさと速度を分離して直す

現在はqsearchノードが全体の約90%を占める。また王手中でもstand-patを許し、候補を捕獲または王手に絞るため、quietな王手回避を読まない経路が残る。bench failure miningでも大regret局面の約6割が王手中で、合法手4-5手の巨大regret例がある。過去の「全王手回避」は速度で棄却されたが、これは正しさの問題を消したことにはならない。

まず `in_check` qsearchを別関数にし、全合法回避を読む正しい参照版を作る。その後、専用evasion生成、TT move、king/capture/block orderingで速度を戻す。採否は全体平均だけでなく、既存の `loss_in_check_low_reply.sfen` とbench failureのin-check集合を必須gateにする。

### 2. PVS

最初の手だけfull window、後続手をzero windowで確認し、fail-high時だけ再探索する。評価の意味を変えずにノードを減らせるため、現在の良いorderingと相性がよい。固定depthで手/PV一致、mate/repetition固定セット、複数seed対局の順に検証する。

### 3. LMR

PVS通過後に、非王手・非捕獲・非TT・後順位の手だけを1 ply reductionする。fail-highは必ずfull depthで再探索する。いきなりStockfish級の複雑な式を移植せず、depth >= 3、move index >= 4程度から始める。

### 4. 本格SEEと詰み

現在のSEEは `victim - attacker` だけで、qsearchでは負なら手を捨てる。過去の再帰SEEはdo/undoや合法手生成が重かった。次に試すなら、Position内部bitboardを使い、局面cloneなしのswap-list SEEとして実装する。

詰みについては、root近傍のnode制限付きDFPNまたは短手数詰みを独立moduleとして測る。評価学習で詰み級外れ値を吸収しようとしない。

## オリジナルな学習データの作り方

### 教師信号

他将棋エンジンの評価値、指し手分布、PVは使用しない。用途ごとに次を使う。

- 公開棋譜の勝敗と指し手: 実戦に現れた局面分布とWDL。外部エンジンの評価値は使用しない。
- 現行HEADの自己対局: 現在の探索が実際に到達する局面分布とWDL。
- 現行HEAD自身の深さ違い探索: 深さを上げても符号と最善手が安定する局面だけを補助value/rankingへ使う。
- ルールベース詰み証明: mate局面の終端ラベル。KPP評価値は使わない。

自己探索値は新しい知識源ではなく、探索しやすい静的評価へ圧縮する補助信号として扱う。棋力の主な新情報は棋譜結果と自己対局結果から得る。

### データ

1. 2023-2025をtrain、2026を完全holdoutにする。棋譜単位で分割し、同一SFENをdedupeする。
2. 序盤・中盤・終盤、王手中、合法手少数、駒得不均衡を明示bucket化する。
3. 初回は10万局面。自前探索ラベルは固定depthではなく固定nodesで動かす。
4. 低nodesと高nodesで符号または最善手が安定する局面を通常value学習へ使い、不安定局面はpolicy/ranking用へ分ける。
5. targetはclipped scoreだけでなくWDLを混ぜる。詰み値は別ラベルにする。

100Kでholdout、探索内部葉、対局の全gateを通らない方式を、100万へ拡張しない。

## ベンチマークを先に直す理由

現在の `taya36.sfen` paired benchは回帰検出には有用だが、同じ集合で多数の候補を選び続けたため、selection biasを持つ。今後は次の三層に分ける。

1. `micro`: 速度、node数、PV、固定の戦術/王手回避セット。
2. `regression`: 非公開holdout opening 500以上、先後入替、同一time control、SPRT相当。
3. `strength`: 固定baselineとの短時間・長時間自己対局。人間対局はリリース後確認とする。

候補重みの採用は最低でも、現行HEAD相手に複数seed合計400局またはSPRT終端、別opening holdoutで非悪化とする。100局の点推定だけで「改善」としない。

## 実行順序

### 最初の3日

1. baseline IDを `engine commit + weight SHA-256` で固定する。
2. Wdoorから学習に使わないopening holdoutを500-1000局面作る。
3. benchmarkへElo、paired結果、SPRT、run manifestを追加する。
4. 公開棋譜と自己対局から、詰み証明・合法回避・資源劣化手順を持つ固定corpusを自動生成する。

### 1-2週間

1. 王手中qsearchの正しい参照版と高速な専用経路を作り、失敗局面gateを通す。
2. PVSを実装し、固定depth tree/PV gateとSPRTを通す。
3. 棋譜結果、自己対局WDL、自前探索安定性を持つ10万局面dumpを作る。
4. 低ランクKPP rank 8/16と、非差分H64 HalfKPのoffline学習を同じデータで比較する。
5. H64がKPPより有望なら、学習拡大より先にAccumulatorを実装してNPSを測る。

### 3-6週間

1. int16/int8量子化とAVX2推論を実装する。
2. H64 NNUEを100万局面まで段階拡大する。
3. PVS通過後の探索へ保守的LMRを追加する。
4. NNUE単体、KPP+小residual、低ランクKPPを同一time/SPRTで最終比較する。

## 中止条件

- 密KPP: 探索改善と症状別gateが整うまでCE/MMTOの追加調整は行わない。
- 低ランクKPP: 100K holdoutで現KPPのWDL/valueとroot rankingを両方超えなければ終了。
- H64 NNUE: 差分・量子化後もNPSがKPP比20%以上落ち、400局相当で補えなければ構造を拡大しない。
- 学習拡大: 100Kで別opening対局へ改善が移らなければ1Mへ進めない。
- 探索: 20/40局の上振れでは採用せず、SPRTまたは事前に固定した局数まで走らせる。

## 最終判断

このプロジェクトの次の一手は「KPPを捨てて大きなニューラルネットへ移る」ことではない。KPPを強いincumbent兼fallbackとして固定し、探索を先に強化しながら、棋譜・自己対局・自前探索だけで学習した小型NNUEを段階的に競わせることである。

計算資源に最も合う研究上の賭けは低ランクKPP、実績と最終棋力を重視した本命は差分HalfKP NNUEである。両者を同じ10万局面・同じholdout・同じ対局gateで比較すれば、数日から2週間で次の大規模投資先を判断できる。

## 参考資料

- Hoki and Kaneko, Large-Scale Optimization for Evaluation Functions with Minimax Search: https://doi.org/10.1613/jair.4217
- Yu Nasu, NNUE reference implementation: https://github.com/ynasu87/nnue
- Stockfish NNUE technical documentation: https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md
- Cute Chess CLI SPRT / paired openings: https://github.com/cutechess/cutechess/blob/master/docs/cutechess-cli.6
