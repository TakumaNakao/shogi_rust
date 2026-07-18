# v2.1.0 KPPからHalfKPへの評価値近似設計

## 1. 目的

v2.5.0では評価関数の独自改善を行わず、`policy_weights_v2.1.0.binary`を使ったKPPの静的評価値を、現在の高速なHalfKPで可能な限り忠実に再現する。狙いは評価品質をv2.1.0相当に保ったまま、HalfKPの評価速度によって探索ノード数を増やすことである。

教師はこのプロジェクト自身のv2.1.0 KPPだけとする。他エンジンの評価値、探索PV、勝敗ラベルはv2.5.0用の近似学習には混ぜない。棋譜は局面供給源としてのみ使う。

KPPは`王 + 2駒`、HalfKPは`王 + 1駒`を基本とするため、全局面での完全一致は構造上保証できない。ここでいう一致は、独立局面、合法手の子局面、探索結果の3段階で誤差を規定値以内に収めることを意味する。

## 2. 現行実験からの必須修正

勝敗ラベルで学習した2M候補は、2026 holdout BCEを改善した一方、v2.1.0 KPPとの36局で2勝34敗だった。候補のoutput weightはstatic HalfKPの約`1e-3`から約`1e-1`へ増大し、初期局面の探索scoreも深さごとに数百cp単位で符号反転した。

したがって新学習器は次を必須とする。

- WDL BCEを使わず、KPPのcentipawn評価値を直接教師にする。
- trainer forwardとengine forwardを共通実装にし、保存後のscore一致を自動テストする。
- `target_scale=1000`を固定し、bias、material、output headの膨張を監視・制約する。
- 親局面だけでなく合法手後の子局面を学習し、手の順位もKPPへ合わせる。
- validation BCEではなく、cp誤差、score分布、合法手順位でearly stoppingする。

## 3. プログラム構成

### 3.1 `halfkp_kpp_dump`

KPP教師データの生成専用バイナリを追加する。

```text
halfkp_kpp_dump
  --input <dataset_build JSONL>...
  --kpp-weights policy_weights_v2.1.0.binary
  --train-output train.hkpd
  --valid-output valid.hkpd
  --test-output test.hkpd
  --children-per-root 8
  --seed 20260716
```

教師値は`SparseModel::predict_from_position`で計算する。探索scoreは使用しない。KPPファイルはプロセス開始時に1回だけ読み込み、局面評価を並列実行する。

各rootでは次の局面を保存する。

- 棋譜本譜局面
- KPP評価で最善の合法手後局面
- KPP評価で最悪の合法手後局面
- 最善との差が小さい候補手後局面
- seed固定で選んだランダム合法手後局面

子局面の指し手側utilityは、着手後に手番が相手へ移るため`-KPP(child)`とする。rootとchildrenを同じgroup idで保持する。

データはJSONではなくbounded streaming可能なpacked binaryとする。

```text
magic/version/schema hash
game hash, position hash, group id, split, ply, phase, flags
side to move, KPP teacher score f32
black: king bucket u8, len u8, piece states [u16; len], material f32
white: king bucket u8, len u8, piece states [u16; len], material f32
```

splitは棋譜単位で行い、position hashでsplit間重複を拒否する。2023--2025をtrain/valid、2026をout-of-time testに固定する。左右反転と180度回転・駒色交換で得られる対称局面もKPPで再評価して保存し、教師に存在する小さな非対称性まで含めて近似する。

### 3.2 `halfkp_kpp_fit`

KPP回帰専用trainerを追加する。既存の勝敗用`halfkp_train`とは分離する。

```text
halfkp_kpp_fit
  --train train.hkpd
  --valid valid.hkpd
  --test test.hkpd
  --init stable_halfkp.bin
  --output halfkp_weights_v2.5.0.binary
  --checkpoint-dir checkpoints/
  --batch-size 2048
  --epochs 20
  --early-stop-patience 3
```

trainer用のパラメータ構造とforwardは`evaluation.rs`のHalfKP実装から共有する。SFENから再抽出したfeature、packed feature、engineの増分accumulatorの3経路が同一scoreを返すことを保証する。

## 4. 初期化と評価尺度

KPP scoreを次のように分解する。

```text
teacher = kpp_bias + kpp_material_coeff * material + kpp_relation
```

HalfKPのmaterial経路は`target_scale=1000`のとき`material * out_w_material`になる。そのため、初期値を次のように設定する。

- `out_b = kpp_bias / 1000`
- `out_w_material = kpp_material_coeff`
- 最初の1 epochはbiasとmaterialを固定する。
- relation部分だけをembeddingとhidden/output headへ学習させる。
- 2 epoch目以降にbias/materialを低いlearning rateで解放する。

既存の安定したHalfKP初期値がある場合はembeddingとhiddenを引き継ぐ。ただしoutput headはnormを記録し、初期normの4倍を超えたcheckpointを無効とする。output headのlearning rateはembeddingの1/10とする。

## 5. 損失関数

評価値は`[-6000, 6000] cp`へclipし、1000で正規化する。詰み値や異常な外れ値が通常局面を支配しないよう、絶対評価値にはHuber lossを使う。

```text
L_abs = Huber((HalfKP(P) - KPP(P)) / 1000)

u_teacher(m) = -KPP(child_m)
u_model(m)   = -HalfKP(child_m)

L_delta = Huber(((u_model(a) - u_model(b))
               - (u_teacher(a) - u_teacher(b))) / 1000)

L_rank = pairwise logistic loss for teacher gaps >= 30 cp

L = L_abs + 0.5 * L_delta + 0.1 * L_rank
```

最初のpilotでは`L_abs`だけのbaselineを必ず作る。`L_delta`と`L_rank`は、絶対誤差を悪化させず合法手順位を改善した場合だけ採用する。

optimizerはmini-batch AdamWまたはmini-batch AdaGradとし、sampleごとの更新は禁止する。batch内gradientを平均後、global norm 1.0でclipする。embeddingとoutputでlearning rateを分け、decoupled weight decayを使用する。Rayonでsampleのforward/backwardを並列化し、batch bufferとprefetch queueはメモリ上限内に固定する。

checkpointにはmodel、optimizer、epoch/batch、RNG、dataset hash、KPP weight SHA-256、best metricsを保存し、中断再開でoptimizerを初期化しない。

## 6. 正当性テスト

本学習前に次を自動テスト化する。

1. dumpされたteacher scoreと`SparseModel::predict_from_position`が`1e-4 cp`以内で一致する。
2. packed featureとSFEN再抽出featureが一致する。
3. trainer forward、engine full forward、engine incremental forwardが`1e-4 cp`以内で一致する。
4. 親子局面で手番が反転し、指し手utilityが`-child_score`になる。
5. finite-difference gradient checkを通す。
6. batch size 1がscalar参照実装と一致する。
7. thread数を変えても許容誤差内で再現する。
8. save/load前後の全scoreが一致する。
9. 128局面を意図的に過学習し、MAEをほぼ0へ下げられる。
10. NaN、ClippedReLU飽和率、output norm、score p01/p50/p99を毎epoch記録する。

## 7. 学習段階

### Phase A: 100k correctness pilot

- root 100kとそのchildren
- `L_abs`のみ
- learning rate 3候補
- engine forward一致とscore分布を確認

### Phase B: 1M selection

- `L_abs` baseline
- `L_abs + L_delta`
- 必要な場合だけ`L_rank`を追加
- 固定validと2026 testで比較

### Phase C: 5M final fit

- 最良1方式だけを学習
- patience 3でearly stopping
- best checkpointを`HKP00001`形式へexport

データを増やしてtest MAEまたは合法手順位が改善しない場合は、長時間反復せず停止する。

## 8. v2.5.0採用ゲート

### Offline gate

2026 testと探索用子局面testの両方で、暫定的に次を要求する。

- mean errorの絶対値: 5 cp以下
- MAE: 50 cp以下
- median absolute error: 25 cp以下
- p95 absolute error: 150 cp以下
- KPPとのPearson correlation: 0.995以上
- 回帰slope: 0.98--1.02、intercept: ±10 cp以内
- `|KPP| >= 100 cp`での符号一致率: 97%以上
- teacher差50 cp以上の合法手pair順位一致率: 95%以上
- 合法手top-1一致率: 85%以上

この値を満たせない場合、対局結果だけを理由に閾値を下げない。HalfKP hidden 32の表現力不足を疑い、hidden 64候補の速度を別途測る。

### Search parity gate

- `taya36.sfen`と固定holdoutでdepth 1--8のscoreを比較する。
- startposで見られた数百cpの深さ交互反転を失格条件とする。
- 固定depthでのbestmove一致率90%以上を要求する。
- HalfKP探索NPSがKPPの1.3倍以上であることをLinux/Windows release buildで確認する。

### Match gate

検索コードを同一commitに固定し、評価ファイルだけを変更する。

1. 500 ms、200局のscreening。
2. 2秒、最低60局のrelease gate。
3. 開始局面と色をpair化し、全棋譜と終局理由を保存する。
4. 2秒gateでHalfKP score rate 50%以上を目標とし、95%信頼区間下限45%以上を最低条件とする。
5. 違法手、評価値飽和、同一手反復、深さparityによるscore振動があれば不採用とする。

## 9. 成果物

- `halfkp_kpp_dump`
- `halfkp_kpp_fit`
- packed dataset manifest
- full training checkpoint
- `halfkp_weights_v2.5.0.binary`
- offline/search/match gate report
- KPP重み、dataset、git revision、hyperparameterを結び付ける再現用manifest

v2.5.0ではこのprojection重みだけを採用し、勝敗学習や棋譜rankingによるKPP超えはv2.5.1以降の別実験に分離する。
