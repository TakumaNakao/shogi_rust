# HalfKP高速化・学習設計

## 1. 目的と制約

この設計の目的は、次の2点を同時に満たすHalfKP評価関数を作ることである。

1. GitHub Actionsが生成するWindows x86_64 MSVC版で、v2.4.4のKPPより探索全体が高速であること。
2. v2.1.0 KPPより正確な評価値を、ノートPCのCPUと14 GiB程度のメモリで学習できること。

学習に利用してよい教師情報は次に限定する。

- 公開または自前の対局棋譜に含まれる指し手と勝敗
- このプロジェクト自身のv2.1.0 KPP評価値
- 将来生成する、このプロジェクト自身の自己対局結果

他の将棋エンジンが出力した評価値や探索PVは使用しない。公開棋譜の対局者が別エンジンであっても、棋譜に含まれる指し手と最終結果だけを使う。

## 2. 現状の判断

現行の `HalfKP[105480 -> 32] x 2 -> 1` は維持する。入力の左右対称化、32要素のaccumulator、増分更新という構成は、目標強度とCPU速度の釣り合いがよい。強い重みを一度も作れていない段階でhidden幅や層数を増やすと、学習器とモデル容量の問題を切り分けられない。

2026-07-15に同じLinux releaseバイナリで4局面を各500 ms探索した結果は次の通りだった。

| 評価関数 | NPS |
| --- | ---: |
| v2.1.0 KPP | 495,429 |
| 増分HalfKP | 855,157 |

HalfKPは約1.73倍だった。別途、同一指し手列を評価するtrace計測では、KPP 326,502 eval/s、増分HalfKP 862,225 eval/sで約2.64倍だった。ただし、KPPとHalfKPでは評価値が異なるため探索木も変わる。1.73倍は実戦的な参考値であり、評価器単体の厳密な倍率ではない。

現行の100kモデルはv2.1.0 KPPを模倣するpilotであり、強さの候補ではない。静的評価回帰は改善した一方、そのモデルを勝敗へfine-tuneするとvalidation BCEが悪化した。これはHalfKP方式の限界ではなく、現在の逐次学習器、固定されたWDL尺度、データの相関と終局label設計に原因がある可能性が高い。

## 3. 推論実装の調査結果

### 3.1 最優先: 特徴行の固定長化

`HalfKpModel.feature_emb` は平坦な `Vec<f32>` であり、モデルファイルの `hidden` も実行時の `usize` として保持されている。モデルロード時には32であることを検証しているが、hot loopではこの事実をコンパイラが十分利用できない。

現行releaseの逆アセンブルでは、1特徴行32要素の加減算が32個のscalar `subss` とほぼ32回の境界チェックに展開されていた。AVX2以前に、データ型が最適化を妨げている。

実装方針:

```rust
#[repr(C, align(64))]
struct FeatureRowF32([f32; HALFKP_HIDDEN]);

struct HalfKpModel {
    feature_rows: Box<[FeatureRowF32]>,
    hidden_b: [f32; HALFKP_HIDDEN],
    out_w_stm: [f32; HALFKP_HIDDEN],
    out_w_nstm: [f32; HALFKP_HIDDEN],
    // material weight and bias follow
}
```

- `hidden` フィールドを推論モデルから除去し、ロード時だけファイル値を検証する。
- 行番号を一度検査して `&FeatureRowF32` を得る。要素ごとの境界検査を発生させない。
- 64 byte alignmentを型で保証し、LinuxとWindowsのallocator差に依存しない。
- portable実装でも固定長配列同士のloopにし、LLVMのSSE2自動vectorizeを可能にする。

### 3.2 最優先: 差分行を一括適用する

現在は、移動、捕獲、成り、持ち駒化の各行を別々にaccumulatorへ適用する。捕獲を伴う手では同じ128 byteのaccumulatorを何度もload/storeする。

`MoveDelta` に削除行と追加行を先に確定し、1 accumulatorにつき1回のkernel呼び出しで処理する。

```text
acc = acc - removed[0] - removed[1] + added[0] + added[1]
```

- `prepare` 時に移動後の駒種、捕獲駒の生駒種、持ち駒index、material差分を一度だけ求める。
- `after.piece_at(to)` と `piece_kind_value` を黒・白accumulatorごとに繰り返さない。
- add/subの符号付き乗算は使わず、加算kernelと減算kernelを明示する。
- 32要素をf32 AVX2なら8要素x4、将来のi16なら16要素x2として処理する。

### 3.3 最優先: ply stackへ変更する

現行 `history.push((black, white))` は、make時に約288 byteを保存し、undo時に同量をcurrent accumulatorへコピーし直す。探索の各edgeで往復コピーが起きる。

次の形式に変更する。

```text
frames[MAX_PLY]
current_ply

make: frames[ply + 1] = frames[ply]; childへ差分適用; ply += 1
undo: ply -= 1
```

undoではコピーを行わない。frameはblack/white hidden、king bucket、mirror、materialをまとめて64 byte alignmentする。`perspective` は配列indexで決まるためframeごとに保持しない。materialは白黒で符号が反対なので1値だけ保持する。

### 3.4 高優先: 王移動refreshを1回の盤面走査にする

`accumulator_for_position` は現在、`fast_accumulator`、`extract_halfkp_features_for`、`extract_halfkp_fixed_mirror` を順に呼ぶ。その結果、王位置、mirror、material、active featureを得るため同じ盤面を複数回走査し、一時 `Vec` も作る。

- `extract_halfkp_features_fixed` の1回だけでbucket、mirror、material、最大64 featureを得る。
- その固定長結果から直接refreshする。
- 探索用refresh経路ではsort、deduplicate、`Vec` allocationを行わない。合法局面の駒・持ち駒slotから生成されるfeatureは構造上重複しないことをtestで保証する。

王移動は通常手より少ないため、通常行更新の改善後に測る。ただし実装は局所的で、確実に無駄を除ける。

### 3.5 高優先: 型消去contextを探索hot pathから外す

`ShogiAI` は各make/evaluate/undoで `Box<dyn Any>` とdowncastを通り、`EngineEvaluator` でもenum matchを行う。さらに、この境界のためHalfKP固有処理がinlineされにくい。

長期形は `Evaluator` にassociated contextを持たせる設計とする。

```rust
trait SearchEvaluator {
    type Context;
    type MoveDelta;
    fn begin(&self, pos: &Position) -> Self::Context;
    fn prepare(&self, ctx: &Self::Context, before: &Position, mv: Move) -> Self::MoveDelta;
    fn push(&self, ctx: &mut Self::Context, delta: Self::MoveDelta, after: &Position);
    fn undo(&self, ctx: &mut Self::Context);
    fn evaluate(&self, pos: &Position, ctx: &Self::Context) -> f32;
}
```

探索側は `prepare`、`Position::do_move`、`push` の順に呼ぶ。変更前後の `Position` を同時に保持したりcloneしたりしない。

KPPとの同一バイナリ内互換は要件ではないため、production HalfKP engineを型付きでmonomorphizeしてよい。解析ツール用のruntime enumは別adapterに残す。

### 3.6 中優先: 量子化

f32版で強い重みを確定した後、feature transformerを `i16` にする。現モデルは約13.5 MBで、このPCの8 MB L3より大きい。i16化で約6.75 MBになり、ランダムな特徴行accessのcache missと加減算量を同時に減らせる。

初期量子化案:

- feature rowとbias: i16
- accumulator: i16。ただし全合法feature組合せのoverflow上限を検証できない間はi32版も比較する。
- ClippedReLU出力: 0..127
- 最終64要素dot: 最初はf32へ変換してもよい。transformer更新の効果を先に切り分ける。
- 次段階でactivation u8、output weight i8、accumulation i32へ移す。

量子化は学習中の範囲制約とexport時の誤差検査を必要とする。未学習の重みを基準にscaleを決めない。

### 3.7 build設定

`Cargo.toml` にはrelease向けLTO設定がない。最初に以下をA/B計測する。

```toml
[profile.release]
lto = "thin"
codegen-units = 1
```

`panic = "abort"` は速度よりbinary sizeへの効果が中心なので任意とする。PGOは再現可能な代表探索workloadが固まるまで導入しない。

## 4. Windows CIとSIMD設計

release workflowは `windows-latest` 上で `x86_64-pc-windows-msvc` をbuildしている。CI runnerに対する `-C target-cpu=native` は使用しない。runner固有命令を含むbinaryになり、利用者PCでillegal instructionになる可能性があるためである。

単一のWindows配布binaryに次の2 backendを含める。

1. `portable`: x86_64で保証されるSSE2を上限とする安全な実装
2. `avx2`: `#[target_feature(enable = "avx2")]` と `core::arch::x86_64` intrinsicによる実装

起動時、モデルロード時に `is_x86_feature_detected!("avx2")` を一度だけ評価し、`KernelKind` をcontextへ保存する。特徴行ごとにCPUIDを呼ばない。hot pathでは予測可能な1分岐で、手全体のdelta kernelを選ぶ。

```rust
enum KernelKind {
    Portable,
    Avx2,
}
```

Windows固有assemblyは書かない。Rust intrinsicを使えばMSVC ABIとLinux System V ABIの差はcompilerが処理する。`#[repr(align(64))]` によるalignmentも両OSで同じ型契約になる。

code generationの変化を追跡できるようRust toolchainをrepositoryでpinし、artifact metadataへ `rustc -Vv`、git revision、選択可能backendを記録する。

検証用に `auto | portable | avx2` を明示選択できる非USI内部optionを設ける。Windows CIでは以下を実行する。

- `cargo test --release --target x86_64-pc-windows-msvc`
- portableとAVX2の評価値・増分更新がf32許容誤差内で一致するtest。量子化整数経路はbit exactとする
- 王移動、成り、捕獲、駒打ち、連続make/undoのrandom trace test
- `usi_engine.exe` の起動、`isready`、短い `go` のsmoke test
- release workflowと同じprofileで `usi_engine.exe` をbuild

GitHub hosted runnerのNPSはノイズが大きいため速度の合否には使わない。workflowが生成した実物の `usi_engine.exe` をWindowsノートPCへ取得し、固定局面・固定設定で計測した値を採用する。Linuxローカルbinaryだけで速度ゲートを完了扱いにしない。

## 5. 学習モデル

### 5.1 forward

各perspectiveについて次を計算する。

```text
z_p = b + sum(E[f] for f in active_features_p)
h_p = clamp(z_p, 0, 1)
q   = b_out + w_stm*h_stm + w_nstm*h_nstm + w_material*(material_stm/1000)
e   = target_scale * q
```

`e` が探索へ返す評価値である。baselineでは `target_scale=1000`、hidden=32を固定する。追加dense layer、phase別head、hidden拡大は行わない。

### 5.2 学習時だけのfactorization

HalfKPのrare featureを少量データでも初期化するため、学習時だけ次を足す。

```text
E_effective(king, state)
  = E_direct(king, state)
  + E_piece(state)
  + E_relative(relative_king_piece)  // 盤上駒のみ
```

- `E_piece` はking bucketをまたいで同じpiece stateを共有する。
- `E_relative` は王から見た相対file/rank、駒種、色を共有する。
- 持ち駒は `E_piece` だけをfactorとする。
- 2から3 epoch後、factorを全direct rowへ加算してfoldし、factorを無効化する。
- exportされる推論モデルは従来と同じ1個のHalfKP tableであり、推論costは増えない。

factorizationなしのcandidateも必ず同じseed・データで作り、効果をvalidationと対局で比較する。

## 6. データ設計

### 6.1 splitと重複排除

既存 `dataset_build` は棋譜path単位でtrain/valid/testを分けており、同一棋譜内のposition leakは防いでいる。この性質は維持する。

追加要件:

- splitを決めてから局面を抽出する。
- position hashで重複を除き、同一局面を複数splitへ入れない。
- 棋譜ごとに局面を決定的reservoir samplingし、先頭からN件だけを取らない。
- 2から4 ply間隔を基本とし、1棋譜24から32局面を上限にする。
- opening/middle/lateをほぼ均等にする。
- drawを0.5として含める。
- 明示された終局理由を保存し、`Sennichite`、`Jishogi`、`Hikiwake` など規定上の引き分けだけを0.5にする。
- `Chudan`、`Matta`、`Fuzumi`、`Error`、終局actionなしを不明結果として除外する。
- 最終数手、王手局面、合法手が極端に少ない局面を全削除せず、それぞれ比率を制限する。
- 高品質棋譜用には両対局者のrate下限を指定できるようにする。
- game hash splitとは別に、後年データを固定したout-of-time testとして保存する。

validationの誤差区間は局面単位ではなく棋譜単位bootstrapで求める。連続局面を独立標本として数えない。

現在の `dataset_build` は明示的な引き分けと異常・不明終局をどちらも `winner=None` にし、`halfkp_feature_dump` は `winner=None` を一律0.5にしている。このままdrawを含めてはならない。`winner` とは別に `termination` と `result_known` を持たせ、packerが不明結果を拒否する。

### 6.2 packed binary

JSONLをPythonの `list[dict]` に全件ロードする方式は廃止する。学習用ファイルは次の情報だけを持つ。

```text
magic/version/schema hash
record count and offsets
game hash, position hash, ply, phase, flags
side to move, result {loss, draw, win}
material_black
black: king_bucket u8, len u8, piece_states [u16; len]
white: king_bucket u8, len u8, piece_states [u16; len]
optional own_kpp_eval i16
```

full feature idは105480未満だがu16には入らない。一方、全active featureは同じking bucketを共有し、piece stateは2344未満である。そのためbucketを1回、stateをu16で持つ方がu32 feature id列より小さい。

train/valid/testを別ファイルにし、mmapまたはbounded buffered readerで読む。3M局面でもデータ本体をPython objectへ展開しない。manifestには入力棋譜、git revision、抽出条件、schema、seed、件数、hashを保存する。

## 7. 教師信号と損失

### 7.1 WDL尺度を先にfitする

モデル評価 `e` を勝率へ変換する。

```text
p_model = sigmoid(e / kappa)
```

`kappa=600` を固定しない。train内のv2.1.0 KPP評価と棋譜結果から、切片0のlogistic slopeをfitし、validで校正を確認する。色・手番の対称性を壊す自由な切片は入れない。

### 7.2 own-KPP bootstrapと勝敗の混合

v2.1.0 KPP評価を `e_kpp`、手番側の最終結果を `r in {0, 0.5, 1}` とする。

```text
t_kpp = sigmoid(clamp(e_kpp) / kappa_kpp)
y = lambda * t_kpp + (1 - lambda) * r
L_value = BCE(p_model, y)
```

これは他エンジン蒸留ではなく、このプロジェクト自身の旧評価関数を初期値として使う処理である。KPPだけを模倣してもKPPを超えられないため、最終結果の比率を必ず持たせる。

最初のpilotでは `lambda in {0.0, 0.25, 0.5}` を比較する。固定scheduleを先に決めず、frozen validationのBCE/Brierと対局screeningで選ぶ。別案として最初の1から2 epochをKPP soft targetだけでpretrainし、その後混合lossへ移るcandidateも作る。

raw評価値のMSEよりWDL空間のsoft-label BCEをbaselineとする。極端な評価値がlossを支配せず、勝敗と同じ尺度で扱えるためである。

### 7.3 棋譜手rankingは第2実験

価値学習が安定した後、棋譜の指し手を直接使うcandidateを追加する。他エンジン評価値は不要である。

- 棋譜手後の子局面をpositiveとする。
- 合法手から一様に選んだ手を基本negativeとする。
- baseline確立後だけ、現モデルが高く見積もるhard negativeを一部混ぜる。
- 子局面では手番が相手へ移るため、指し手側utilityは `u = -e_child` とする。
- sampled softmaxまたはpairwise logistic lossを使用する。
- 終盤の敗者手、ほぼforcedな局面、低rate棋譜の重みを下げる。

```text
L_total = L_value + mu * L_rank
```

`mu=0` の価値専用baselineを必ず残す。棋譜手一致率だけ良く、対局が弱くなるcandidateは採用しない。

### 7.4 将来の自己対局

最初の採用可能なHalfKPができた後に限り、互角局面集から自己対局を生成する。教師は最終結果を主とし、自身のsearch scoreは補助実験に分離する。自己評価だけを教師にすると既存の誤りを再生産するため、公開棋譜結果を完全には置き換えない。

## 8. CPU学習器

新しい学習器はRustで実装する。現在のNumPy trainerは、recordごと・active featureごとのPython loopが中心で、全JSON recordをobjectとして保持する。100k件で約468 MBを使い、CPU 1 core相当しか利用できなかった。

### 8.1 batch処理

- batch sizeは1024を初期値とし、512/2048を比較する。
- Rayonの既定thread poolを使い、利用可能な全logical CPUを使用する。
- thread数は制限しない。代わりにbatch bufferと先読みqueueをboundedにする。
- `--memory-limit-mib` の既定を6144程度とし、見積り超過時はthreadではなくbatch sizeとprefetch数を下げる。
- model、optimizer state、mmapを共有し、recordを複製しない。

forward/backwardを各sampleで並列計算した後、active row eventを `(row_id, sample_id, perspective)` として集め、row_idでsort/coalesceする。同じrowのgradientを決定的順序で合算してから1回だけoptimizerへ渡す。feature rowへのatomic f32加算や巨大なlockは使わない。

### 8.2 gradient

ClippedReLUのgradientは `0 < z < 1` の要素だけを通す。hidden gradientを計算するときは、更新前のoutput weightを使う。現NumPy実装はoutput weightを先に更新し、その更新後weightでhidden gradientを計算しており、修正が必要である。

gradientはbatch平均してからclipする。sampleごとにparameterを更新しない。最小構成ではembedding、bias、outputの全てにmini-batch AdaGradを使う。

```text
G_i <- G_i + g_i^2
theta_i <- theta_i - lr * g_i / (sqrt(G_i) + eps)
```

feature model約13.5 MBと同サイズのAdaGrad stateを足しても約27 MBであり、メモリ上の問題はない。weight decayをactive rowだけへ毎回足す実装は真のglobal L2ではないため、baselineでは無効またはepoch単位のdecoupled decayとする。

### 8.3 checkpoint

checkpointには次を含める。

- 全model parameter
- optimizer accumulator
- epoch、batch、RNG state
- dataset manifest hash
- factorizationのfold状態
- kappa、lambda、target scale
- best validation metricとgit revision

epochごとに保存し、validation BCEが最良のcheckpointだけをbinary exportする。中断再開でoptimizerを初期化しない。

## 9. 学習器の正当性検証

本格学習前に次を自動test化する。

1. toy modelのfinite-difference gradient check
2. batch size 1と参照scalar実装の1 update一致
3. thread数を変えたときの許容誤差内一致
4. 同じseed・checkpointからの再現性
5. 32から128局面だけを過学習できること
6. save/load後のforward一致
7. Rust trainerのforwardとengine f32 forwardの一致
8. factor fold前後の全validation予測一致
9. packed featureとSFENから再抽出したfeatureの一致
10. NaN、ClippedReLU飽和率、weight範囲の監視

## 10. pilotと採用ゲート

### Phase A: 学習器smoke

- 100k train、固定valid/test
- 5から10 epoch
- batch 512/1024/2048
- learning rate 3候補
- lambda 0/0.25/0.5
- factorization有無

全組合せを総当たりせず、まずbatch/lrをlambda 0.25で決め、その後lambdaとfactorizationを比較する。

### Phase B: 中規模学習

- 500k局面
- 最良2 candidate
- early stopping patience 2から3 epoch
- 棋譜単位bootstrap付きoffline比較

### Phase C: 本学習

- 1Mから3M局面
- RSS、positions/s、epoch時間を記録
- best f32 checkpointを確定
- その後に量子化candidateをexport

### Offline gate

固定testで次を比較する。

- result priorだけの定数予測
- sigmoid尺度をfitしたv2.1.0 KPP
- KPP bootstrap直後HalfKP
- 勝敗fine-tune後HalfKP

主要指標はBCE、Brier score、calibration errorとし、opening/middle/late、material差、王手有無、勝/分/負ごとにも出す。評価値符号一致率とKPP RMSEは補助指標である。KPPに対するBCE/Brier改善の95%区間が0をまたぐ場合、offlineでは改善未確認と扱う。

### Correctness and speed gate

- 増分評価とfull refreshのrandom trace一致
- portableとAVX2の一致
- f32とquantizedの誤差分布
- 固定move trace eval/s
- 固定HalfKP重み・同一探索木で変更前後のNPS比較
- Windows CI artifactでのNPS比較

速度値は7回以上計測したmedianを使う。構造最適化で3%以上遅くなる変更は、明確な強度上昇がない限り戻す。

### Match gate

検索ロジックを揃え、次の順に進める。

1. 20局の短時間smokeでcrash、反則、極端な評価崩壊を検出
2. 100局の1秒screening
3. 互角局面集から色を入れ替えた200局の3秒対局
4. 200局が僅差なら400から800局へ延長

最終比較対象はv2.4.4 Windows binary + v2.1.0 KPPである。HalfKP側もGitHub CIが生成したWindows binaryを使う。点推定の勝率だけでなく、paired opening単位の区間を報告する。

## 11. 実装順序

1. 固定長・aligned feature rowへ変更し、境界検査とscalar展開を除去する。
2. `MoveDelta` 一括適用とply stack化を行う。
3. 王移動refreshを1走査にし、型付きHalfKP search contextへ移行する。
4. portable/AVX2 runtime dispatchを実装し、Windows CI testを追加する。
5. packed dataset writer/readerとmanifestを実装する。
6. Rust mini-batch trainer、gradient test、checkpointを実装する。
7. WDL scale fitting、mixed target、offline reportを実装する。
8. 100k pilot、500k選抜、1Mから3M本学習を行う。
9. f32重みが採用候補になってからi16量子化を実装する。
10. GitHub CIのWindows artifactで速度・対局ゲートを実行する。

hidden拡大、追加dense layer、複雑な新featureは、この手順でH32がKPPをoffline・対局の両方で超えられなかった場合だけ再検討する。

## 12. 参考資料

- [Stockfish NNUE technical documentation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)
- [Rust runtime x86 feature detection](https://doc.rust-lang.org/std/macro.is_x86_feature_detected.html)
- [Rust `target_feature` attribute](https://doc.rust-lang.org/reference/attributes/codegen.html#the-target_feature-attribute)
- [Cargo release profile and LTO](https://doc.rust-lang.org/cargo/reference/profiles.html#lto)
