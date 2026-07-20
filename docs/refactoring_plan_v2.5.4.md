# v2.5.4以降のアーキテクチャ・リファクタリング実行計画

| 項目 | 内容 |
|---|---|
| 状態 | Active planning document |
| 最終更新 | 2026-07-20 |
| 調査基準revision | `5bc22c740389` (`v2.5.4-5-g5bc22c7`) |
| リリース基準 | `v2.5.4` |
| 上位契約 | [`refactoring_handoff_v2.5.4.md`](refactoring_handoff_v2.5.4.md) |
| 対象 | エンジン、探索、評価、USI、教師生成、学習、ツール、CI、文書、生成物管理 |

この文書は、v2.5.4以降のコードベースを、棋力と実行性能を不用意に低下させずに整理するための実行計画である。
`refactoring_handoff_v2.5.4.md` が「壊してはいけない既存契約」を記録するのに対し、本書は次を定義する。

- 現行実装を調査して判明した構造上・正しさ上・性能上の課題
- 目標とするcrate、module、実行時責務の境界
- 機械的変更、正しさ修正、性能変更、学習変更を分離する規則
- PR単位の作業順序、依存関係、検証方法、完了条件
- ベンチマーク、対局、教師データ、学習再開の非退行ゲート
- 文書、実験レポート、バイナリ、生成物を長期的に管理する方法

本書は大規模な一括書き換えを指示するものではない。現在動いている実装を参照系として保存し、観測可能性を先に作ったうえで、責務境界を一つずつ移動する。

---

## 1. 結論と基本方針

採用する方針は、次の順序で進める段階的リファクタリングである。

1. 現行動作と性能を固定する。
2. Cargo workspace、CI、fixture、characterization testを整える。
3. アルゴリズムを変えずに巨大moduleを分割する。
4. HalfKPと教師データの形式を単一実装へ統合する。
5. 探索coreとUSI lifecycleを分離する。
6. 千日手を独立した正しさ変更として修正する。
7. データ、教師生成、trainerを再現可能かつbounded memoryな構造へ移す。
8. バイナリ、文書、生成物を整理する。
9. 最後に、計測で裏付けられた性能最適化を一件ずつ行う。

重要な判断は次のとおり。

- 「きれいな設計にすること」と「速くなること」は別の成果として測る。
- 構造変更PRでは、探索パラメータ、評価式、重み、手順順序、学習目的を変更しない。
- 現行の巨大ファイルを、最初から多数のcrateへ移動しない。まず同一crate内のprivate moduleへ分割する。
- HalfKPのwire formatは共有するが、runtimeのaligned layout、AVX2 kernel、hot loopは専用実装として残す。
- qsearchは最大の性能候補だが、既存の失敗実験を根拠なく再試行しない。
- 千日手修正後は探索結果と教師labelの意味が変わり得る。旧教師データと新教師データをversionなしで混在させない。
- NPSだけで採用を決めない。探索結果、node fingerprint、実行時間、対局結果を分けて評価する。

---

## 2. 文書間の優先順位

設計判断が競合した場合は、次の順で扱う。

1. リリース済み形式と、固定fixtureで確認された実際の互換性
2. [`refactoring_handoff_v2.5.4.md`](refactoring_handoff_v2.5.4.md) の非交渉条件
3. 本書のアーキテクチャと作業順序
4. 個別module文書
5. 過去の実験レポート

ただし、既知の不具合については「現行実装が動いている」ことを正しさの根拠にしない。現行挙動をcharacterization testで保存した後、正しさ変更として別PRで修正する。

本書の基準revisionから実装が進んだ場合は、冒頭のrevision、Phaseの状態、基準値、未解決事項を更新する。完了済みPhaseを削除せず、完了revisionと結果への参照を追記する。

---

## 3. 調査時点のコードベース

### 3.1 リポジトリ構成

2026-07-20、revision `5bc22c740389`で確認した状態は次のとおり。

| 項目 | 状態 |
|---|---:|
| `src/bin/*.rs` | 54ファイル |
| `src/bin`概算規模 | 約28,000行 |
| `docs/*.md` | 19ファイル |
| `report/*.md` | 130ファイル |
| `tools/`配下 | 28ファイル |
| `scripts/`配下 | 5ファイル |
| `src/ai.rs` | 1,525行 |
| `src/evaluation.rs` | 2,420行 |
| `src/bin/mmto_tree_train.rs` | 5,409行 |

ルートは単一Cargo packageで、実行エンジン、学習、ベンチマーク、可視化、研究用ツールが同じdependency集合と自動binary検出を共有している。
`shogi_lib`はpath dependencyだがworkspace memberではなく、独自の`Cargo.toml`、`Cargo.lock`、profileを持つ。

この構成には次の問題がある。

- ルートの通常テストが`shogi_lib`のテストを実行しない。
- production engineの変更と、研究用binaryのdependency・warningが分離されない。
- どのbinaryが正式サポート対象かCargoから判別できない。
- crate境界とpublic APIが、実際の責務境界を表していない。
- rootのrelease profileが、path dependency側へ期待どおり適用されるか分かりにくい。
- 54 binaryに同種のloader、parser、JSONL出力、対局統計処理が重複している。

### 3.2 ビルドとfeature

現行`Cargo.toml`は`default = []`、`halfkp64 = []`である。
一方、v2.5.xの配布エンジンと配布weightはHalfKP-64を前提とする。

したがって、次の2つは現在同義ではない。

```bash
cargo build --release --bin usi_engine
cargo build --release --features halfkp64 --bin usi_engine
```

リファクタリング中に`halfkp64`を不用意にdefaultへ変更すると、既存利用者やHalfKP-32 toolの意味を変える。
最終的にはproduction app側が必要featureを明示的に選ぶ構成にし、単純な配布用buildが誤ったhidden widthを生成しないようにする。

### 3.3 テストと静的検査

調査時に確認した基準は次のとおり。

| 検証 | 結果 |
|---|---|
| root library、release、HalfKP-64 | 31 tests passed |
| `shogi_lib`単独、release | 33 tests passed |
| `halfkp_search_train` binary test | 1 test passed |
| handoff記載の主要release check | passed |
| `cargo check --features halfkp64 --all-targets` | passed |
| `cargo fmt --all -- --check` | passed |
| `git diff --check` | passed |
| clippy、HalfKP-64、all targets | build passed、warning診断97件 |

97件は同一warningのtarget間重複を含むため、97種類の独立問題を意味しない。
既存warningを一括自動修正するとhot loopや実験コードに大きな差分が生じるため、初期段階では「production codeに新しいwarningを増やさない」ratchet方式を採る。

### 3.4 現行探索の性能基準

以下は履歴比較ではなく、今後の変更と比較するための調査時点の基準である。

実行環境:

```text
OS: Linux x86_64
CPU: AMD Ryzen 7 PRO 4750U, 8 cores / 16 threads
CPU feature: AVX2 available
rustc: 1.95.0
cargo: 1.95.0
revision: 5bc22c740389
```

入力:

```text
weight:
  policy_weights_halfkp64_kpp_distilled_v2.5.0.binary
  sha256=91784d6f03f70580468a1509f968cbaaf598fbce084c22a739cb870344fb7c00

positions:
  taya36.sfen
  sha256=22950f42a2d65292fa689a8c84e3af7dc160c571e9aa559699b0beaf1bc9adce
```

コマンド:

```bash
target/release/search_profile \
  --halfkp-weights policy_weights_halfkp64_kpp_distilled_v2.5.0.binary \
  --positions taya36.sfen \
  --samples 16 \
  --depth 5 \
  --seed 9501 \
  --threads 1
```

3回とも一致した意味的カウンタ:

| 指標 | 値 |
|---|---:|
| completed depth | 5 |
| total nodes | 6,208,693 |
| qsearch nodes | 5,763,074 |
| qsearch比率 | 92.82% |
| qsearch generated moves | 87,300,534 |
| qsearch discarded moves | 76,862,168 |
| qsearch discard率 | 88.04% |
| considered moves | 10,438,366 |
| searched moves | 3,134,345 |

実行時間は`6075.50 ms`、`6086.42 ms`、`6133.18 ms`で、中央値は`6086.42 ms`だった。
対応するNPS中央値は約`1.020 million`である。

この計測から、qsearchが重要な性能領域であることは分かる。
しかし、過去のreportには直接qsearch生成、ArrayVec化、手順順序変更、評価table変更などで全体性能が低下した結果がある。
したがって、discard率だけを理由にmove generatorを書き換えてはならない。

---

## 4. 現行実装で確認した主要課題

### 4.1 探索

`src/ai.rs`は次の責務を同時に持つ。

- mate scoreとTT正規化
- local/shared TT
- worker-local探索状態
- alpha-beta
- qsearch
- 反復深化
- root結果選択
- parallel workerの生成と調停
- timeout、stop、panic recovery
- 探索統計
- USI `info`文字列のstdout出力

`ShogiAI`は評価器、千日手検出器、TT、killer、history、timer、統計、型消去された評価contextを一つの構造体に保持する。

現状の重要な性質:

- worker 0の結果だけが最終root resultになる。
- helper workerはshared TTとshared stopを通じてのみ寄与する。
- `Threads=1`にはserial pathがある。
- `Threads=0`は論理CPU数を使い、256で上限になる。
- root結果がない場合でも合法手fallbackを選ぶ経路がある。
- checked qsearchではstand-patを使わず、quiet evasionも含める。

これらは構造上は整理対象だが、動作としては保存対象である。

### 4.2 USI lifecycle

`src/usi_shogi.rs`はPosition、option、共有AI、stop signalを保持し、`go`ごとにthreadを起動する。
探索threadのhandleをエンジン状態へ保持せず、完了、stop、次のgo、quitの同期関係が明示的なstate machineになっていない。

主なリスク:

- 直前の探索が完全に終了する前に次の`go`を受ける。
- 古い探索が遅れて出力する。
- stop-before-startが初期化で消える。
- panic、no evaluator、timeoutで複数またはゼロの`bestmove`になる。
- Windowsでthread終了とprocess終了の順序が変わる。

探索coreはUSI文字列を出力せず、構造化された進捗と結果を返すべきである。
プロトコル上の「必ず一つのbestmove」はUSI adapterが所有する。

### 4.3 千日手

現在の`SennichiteDetector`は最大256件のposition keyを保持し、同一keyが4回現れた時点の`position.in_check()`を使って連続王手負けを判定する。

この情報だけでは次を判定できない。

- 反復区間の全該当着手が王手だったか
- 王手を継続していた側はどちらか
- 現在王手だが途中に非王手を含む反復か

さらに、USI `position ... moves`で再生された探索前の全履歴が、AI内の検出器へseedされない。
benchmark側も同じ判定に依存する。

`Position`は既にstate stackを持つため、長期的には次のどちらかへ一本化する。

1. `Position`のstate stackから読み出すread-only `HistoryView`
2. `Position`と常に同期する単一の`GameHistory`

position keyだけを別のcircular bufferへ複製する構造は廃止する。

### 4.4 評価

`src/evaluation.rs`には以下が同居する。

- 駒価値とmaterial評価
- KPP featureとsparse model
- Tiny NNUE
- HalfKP feature
- HalfKP wire format
- runtime aligned weight row
- AVX2とportable kernel
- incremental accumulator
- engine evaluator dispatcher
- KPP debug、index説明、SFEN生成補助

HalfKP runtimeに必要な責務は、次の3層に分ける。

```text
format:
  header、version、dimension、flat tensor、little-endian codec

semantics:
  perspective、king mirror、piece ownership、hand slot、
  side-to-move、material sign、feature order

runtime:
  aligned row、accumulator、portable kernel、AVX2 kernel、
  make/undo差分更新
```

formatとsemanticsはruntime、trainer、分析toolで共有する。
runtimeだけに必要なaligned layoutやSIMD実装はengine内へ残す。

現行`Evaluator`は`Box<dyn Any + Send>`のcontextをmake、undo、evaluate時にdowncastする。
これは性能候補だが、typed contextの最終形は以下を比較して決める。

- backend enumと対応するcontext enum
- associated contextを持つgeneric search
- modelごとのconcrete search session

設計上きれいでも、nodeごとのmatchやcode size増加で遅くなる可能性があるため、Phase 8まで変更しない。

### 4.5 HalfKP形式の重複

HalfKPのmagic、version、target scale、flat weight load/save、forward passがruntimeと複数trainerに重複する。

この状態では次が起こり得る。

- trainerが出力できるがengineが読めない。
- dimension、tensor order、endiannessの片側だけが変わる。
- runtimeとtrainerでforwardの加算順またはperspectiveがずれる。
- HalfKP-32とHalfKP-64が分かりにくいエラーで混ざる。

移行は一括で行わず、shared codecを追加した後にruntimeと各trainerを一つずつ切り替える。
切替前後の出力byte、読み戻しscore、fixture hashを比較する。

### 4.6 教師データ

HKST v2はmagic `HKST0002`、version、HalfKP dimension、候補flag、root utilityを持つ。
現状はround-tripテストが中心で、固定byte fixture、truncated header/record、未知flag、dimension mismatchの網羅が不足している。

候補flagについては、shallow search bestとdeep score sort後のbestが異なる場合に、複数候補へ`search best`が付く可能性がある。
trainerは最大scoreを選ぶため現在直ちに破綻しないが、flagの意味が曖昧である。

この挙動を機械的format抽出と同時に修正してはならない。
まずfixtureで現状を記録し、その後に次のどちらかを決める。

- flagを「いずれかの探索段階でbestだった候補」として文書化する。
- schemaまたはteacher semantics versionを上げ、最終search bestだけに修正する。

### 4.7 教師生成

候補手の評価ごとに新しい`ShogiAI`を構築するため、以下を繰り返し確保する。

- TTまたはTT wrapper
- move ordering state
- evaluation context
- history、killer、statistics
- worker setup

Rayon workerごとに再利用可能な`TeacherSearchSession`を持たせる。
ただし、候補間で残してよい状態とresetすべき状態を明示する。

| 状態 | 方針 |
|---|---|
| immutable model | 共有 |
| bounded scratch buffer | worker内で再利用 |
| evaluation context | 局面ごとに正しく初期化 |
| node counters | 候補ごとにreset |
| killer/history | まず完全resetして現状同値を優先 |
| TT | 現状同値を確認するまでは候補間共有しない |
| stop state | 必ず候補ごとにreset。ただし外部stopは保持 |

TTやhistoryの候補間共有は教師scoreを変える可能性があるため、性能変更として別に評価する。

### 4.8 trainer

現行trainerはvalidation/testを全件保持し、training件数を得るために全shardをdecodeした後、epoch中に再度読み込む。
レコード勾配はactive featureごとに`[f32; 64]`を複製し、batch集約では新しいHashMapを使う。

長期形は次のとおり。

```text
manifest:
  shard path、record count、sha256、format version、split、
  engine revision、weight hash、search config hash

reader:
  bounded chunk reader

shuffle:
  deterministic shard order + bounded in-chunk shuffle

gradient:
  hidden gradientを局面単位で1つ保持
  + feature ID列

reducer:
  worker-local reusable accumulator
  + 決められた順序でmerge
```

浮動小数点のreduction順が変わるとbitwise一致しない。
最初は既存のfeature順とbatch merge順を保存する表現変更だけを行い、並列reduction変更は別PRにする。

### 4.9 datasetと実験再現性

phase境界、CSA metadata解析、結果変換が複数binaryで重複する。
game splitはpath文字列hashに依存するため、同じ棋譜を移動するとsplitが変わり、同じ棋譜を別pathへ複製するとtrain/test leakageが起こり得る。

新しいsplit v2は次のいずれかの安定IDを用いる。

- 正規化した棋譜内容hash
- sourceが保証する一意game ID
- 両者を含むcanonical game identity

split方式の変更はdataset semantics変更であり、旧datasetとの混在を禁止する。

shell scriptの「出力が非空」「`.complete`が存在する」という判定は、入力や設定が変わった場合の再利用条件として不十分である。
各stageは入力fingerprintと出力hashを持つmanifestで再利用可否を判定する。

### 4.10 文書、binary、生成物

`docs/binaries.md`が3 binaryだけを説明する一方、実際には54 binaryがある。
`report/`には失敗を含む重要な実験結果が多数あるが、状態、対象revision、後継実験を横断して探す索引がない。

ローカルには巨大なdatasetやtargetが存在し得るが、それらの多くはGit管理外である。
したがって「Git履歴を削る」ことより、次を優先する。

- 正式binaryと研究binaryの区分
- 生成物のcanonical directory
- tracked fixtureとuntracked artifactの明示
- report metadataと索引
- 古い文書を削除せず、current、historical、supersededを表示

---

## 5. 絶対に保存する契約

詳細はhandoffを正本とし、以下を自動テストへ変換する。

### 5.1 探索とUSI

- 合法手がある非終端局面の`go`は、全経路で合法な`bestmove`をちょうど1回出す。
- timeout、外部stop、開始前stop、worker panic、forced mate、全候補負けでも上記を守る。
- worker 0だけが報告root resultを所有する。
- helper workerは部分root結果を最終結果として公開しない。
- `Threads=0`と`Threads=1`の意味を保存する。
- position、history、accumulator、history heuristic、killer、node countersはworker-localとする。
- shared stateはTTとstop coordinationに限定する。
- search開始処理は既に立っているstopを消さない。
- TTのmate scoreはply正規化して保存・復元する。
- checked qsearchはstand-patせず、quiet moveを含む全合法evasionを探索する。
- qsearch maximum plyを維持する。

### 5.2 HalfKP

- HalfKP-32とHalfKP-64は互換扱いしない。
- magic、version、dimension、tensor order、little-endian `f32`をversionなしで変えない。
- full refreshとincremental accumulatorを、全着手種別で数値的に一致させる。
- AVX2とportable fallbackを一致させる。
- WindowsやAVX2非対応CPUでportable pathを利用できるようにする。
- perspective、king mirror、ownership、hand slot、side-to-move、material signを保存する。
- side-to-move accumulator、opponent accumulator、materialの結合順とscore視点を保存する。

### 5.3 教師と学習

- HKST v2のmagic、version、dimension検証を保存する。
- child scoreをroot utilityへ変換するとき、negateはちょうど1回とする。
- randomized descendantは元棋譜の結果を継承しない。
- splitはgame単位とし、同一gameの隣接局面を複数splitへ分けない。
- checkpointにはoptimizer stateを含める。
- resumeはoptimizerを再初期化せず、次epochを再現する。
- validationだけでbest checkpointを選ぶ。
- test setをhyperparameter、early stopping、kappa、checkpoint選択に使わない。

---

## 6. 変更の分類とPR規則

すべてのPRは、原則として次のいずれか一つへ分類する。

| 分類 | 目的 | 許される変化 | 必須ゲート |
|---|---|---|---|
| M: Mechanical | 移動、分割、rename、重複除去 | 公開される動作は不変 | exact fingerprint、format byte一致 |
| C: Correctness | 千日手など規則上の不具合修正 | allowlistされた探索差分 | 新規規則test、差分説明、teacher version |
| P: Performance | allocation、SIMD、探索効率改善 | 意図した性能差。原則意味不変 | exact/allowlisted結果、中央値、対局 |
| D: Data/Training | split、教師意味、optimizer、学習器変更 | version化されたdata/model差分 | manifest、resume、metrics、対局 |
| O: Operations | CI、script、文書、artifact管理 | product動作不変 | CI、dry-run、リンク検証 |

### 6.1 一つのPRで混ぜないもの

- module移動と探索parameter tuning
- format抽出とformat version変更
- USI分離とfallback policy変更
- 千日手修正とqsearch最適化
- trainer streaming化とoptimizer変更
- data split変更と教師search depth変更
- clippy一括修正とhot loop変更
- crate移動とLTO/profile変更

### 6.2 Mechanical PRの禁止事項

- comparison operatorやscore境界を「整理」しない。
- move sortのstable/unstableを変えない。
- iterator化で評価順、allocation、short-circuit順を変えない。
- floatの加算順を変えない。
- `unwrap`、panic、fallbackを別policyへ変えない。
- `HashMap`を別collectionへ置き換えない。
- public fieldを消すために複数利用箇所の意味を同時変更しない。

---

## 7. 目標アーキテクチャ

### 7.1 最終的な物理構成

目標形は以下とする。名称は移行時のADRで最終確定する。

```text
Cargo.toml
Cargo.lock

crates/
  shogi_position/
    src/
      position.rs
      state.rs
      movegen/
      attack.rs
      bitboard.rs
      zobrist.rs

  shogi_formats/
    src/
      halfkp.rs
      search_teacher.rs
      manifest.rs

  shogi_engine/
    src/
      search/
      evaluation/
      rules/
      notation/

  shogi_training/
    src/
      data/
      teacher/
      halfkp/
      metrics/

apps/
  usi_engine/
    src/
      command.rs
      options.rs
      session.rs
      search_job.rs
      output.rs
      main.rs

tools/
  training/
  benchmark/
  research/

tests/
  fixtures/
    search/
    halfkp/
    teacher/
    usi/

benchmarks/
  baselines/
  configs/

docs/
  adr/
  architecture/
  guides/
  experiments/
```

最終形へ一度に移動しない。Phase 2では現行crate内に同じ論理moduleを作り、Phase 3以降で安定した境界だけをcrateへ昇格させる。

### 7.2 依存方向

```text
shogi_position <--------- shogi_engine <--------- usi_engine
      ^                        ^
      |                        |
      +----- shogi_training ---+

shogi_formats <---------- shogi_engine
      ^
      |
      +------------------ shogi_training

benchmark tools --------> adapters / public APIs
research tools ---------> 必要な全crate。ただしproductionから逆依存しない
```

規則:

- `shogi_position`はengine、USI、trainingを知らない。
- `shogi_formats`はsearch algorithmを知らない。
- `shogi_engine`はstdin、stdout、USI command文字列を知らない。
- `usi_engine`は学習器を知らない。
- `shogi_training`はUSIを経由せず、必要なsearch APIを直接利用する。
- production crateはresearch toolへ依存しない。
- binaryの`main`はargument parse、構成、library call、終了codeだけを担当する。

### 7.3 `shogi_position`

現在の`shogi_lib`を基礎とする。

責務:

- board、hand、side to move
- `Move`とUSI move変換に必要なprimitive
- do/undoとstate stack
- attack、check、pin
- pseudolegal/legal move generation
- position key
- historyを読むための安全なview

持たせないもの:

- alpha-beta
- evaluator
- TT
- USI process lifecycle
- trainer
- match統計

`Position::switch_turn`のようにhashだけを変え、attack/state invariantを満たさない可能性があるAPIは、利用箇所を評価視点APIへ置き換えた後にprivate化または削除する。

### 7.4 `shogi_formats`

責務:

- HalfKP headerとflat tensor codec
- HKST teacher header、record、flag codec
- schema version
- dimensionとendianness検証
- manifestのserialize/deserialize
- checksum対象となるcanonical representation

持たせないもの:

- AVX2
- aligned runtime row
- alpha-beta
- optimizer
- filesystem上の実験directory policy

readerは次を区別して返す。

- unsupported version
- wrong hidden/input dimension
- truncated header
- truncated record
- invalid count/length
- non-finite weightまたはscore
- checksum mismatch

### 7.5 `shogi_engine::search`

論理module:

```text
search/
  score.rs
  limits.rs
  outcome.rs
  stats.rs
  tt.rs
  ordering.rs
  qsearch.rs
  alphabeta.rs
  iterative.rs
  worker.rs
  parallel.rs
  observer.rs
```

概念APIは次の責務を持つ。これは設計例であり、Phase 4でcharacterization testを基に確定する。

```rust
pub struct SearchRequest<'a> {
    pub position: &'a Position,
    pub history: HistoryView<'a>,
    pub limits: SearchLimits,
}

pub enum SearchOutcome {
    Terminal,
    Completed(RootResult),
    Stopped { best_so_far: Option<RootResult> },
    Failed(SearchFailure),
}

pub struct RootResult {
    pub best_move: Move,
    pub score: Score,
    pub pv: Vec<Move>,
    pub completed_depth: u32,
    pub stats: SearchStats,
}

pub trait SearchObserver {
    fn on_info(&mut self, info: &SearchInfo);
}
```

search coreはstdoutへ書き込まない。
`SearchOutcome::Failed`やroot move未確定の`Stopped`から合法fallbackを選び、USIとして一つだけ出力する責務はUSI sessionに置く。

### 7.6 worker-localとshared state

| worker-local | shared |
|---|---|
| Position | transposition table |
| history view / repetition state | cancellation token |
| evaluation accumulator | optional global hard-stop reason |
| killer moves | read-only evaluator model |
| history heuristic | read-only search configuration |
| node counters |  |
| temporary move buffers |  |
| current PV |  |

root resultをshared mutable slotへ各workerが書く設計にはしない。
worker 0の結果だけをcoordinatorが受け取り、helperはTTとstop requestで支援する現行契約を維持する。

### 7.7 USI state machine

目標状態:

```text
Idle
  |
  | go
  v
Starting ---- stop/quit/new go ----+
  |                                |
  v                                v
Running ----------------------> Cancelling
  |                                |
  | completed/panic                | worker joined
  v                                v
Finishing ----------------------> Idle
  |
  | exactly one bestmove for a legal non-terminal go
  v
Idle
```

`SearchJob`は最低限、次を持つ。

```text
generation ID
cancellation token
JoinHandle
position snapshot
legal fallback
bestmove-emitted flagまたは単一output owner
```

規則:

- 新しい`go`は以前のjobをcancelし、必要な同期を終えてから開始する。
- 古いgenerationからのeventは出力しない。
- `stop`は存在するjobのtokenを立てる。開始前stopも保持する。
- `quit`はcancel後にjoinする。
- workerは直接`bestmove`を出力しない。
- sessionの一箇所だけが最終outcomeをUSIへ変換する。
- `info`eventと`bestmove`は同じgenerationへ関連付ける。

### 7.8 `shogi_engine::evaluation`

```text
evaluation/
  mod.rs
  piece_values.rs
  sparse/
  tiny_nnue/
  halfkp/
    features.rs
    model.rs
    accumulator.rs
    kernels/
      portable.rs
      x86_64_avx2.rs
```

HalfKPの処理:

```text
shogi_formats::HalfKpParameters
             |
             | validate + load
             v
shogi_engine::HalfKpRuntimeModel
             |
             | build aligned rows
             v
portable / AVX2 evaluation
```

trainerも同じ`HalfKpParameters`を使うが、runtime aligned representationへ依存しない。

### 7.9 `shogi_training`

```text
data/
  csa.rs
  game_id.rs
  phase.rs
  result.rs
  split.rs
  manifest.rs

teacher/
  candidates.rs
  search_session.rs
  generator.rs

halfkp/
  parameters.rs
  forward.rs
  loss.rs
  gradient.rs
  optimizer/
  checkpoint.rs
  trainer.rs

metrics/
  classification.rs
  ranking.rs
  calibration.rs
```

dataset builder、teacher generator、trainer、scriptでphase、split、resultの意味を再定義しない。
CLIは共通library APIを呼び出し、manifestへ完全な設定を記録する。

---

## 8. 実行Phase

### Phase 0: ベースラインと変更契約の固定

目的:

- 以降の差分が「同値」「意図した変更」「退行」のどれか判定できる状態にする。

作業:

1. revision `5bc22c740389`を基準として記録する。
2. 小さな固定局面集合を作る。
3. 各局面についてThreads=1のsearch snapshotを保存する。
4. TT mate score、qsearch checked node、forced mate、全敗rootのfixtureを追加する。
5. released HalfKP weightはcommitせず、pathとSHA-256をbaseline manifestへ記録する。
6. HalfKP model、HKST teacherの小さなgolden binary fixtureを作る。
7. benchmark結果へdirty flag、rustc、target、CPU feature、binary hash、weight hash、corpus hashを記録する。

推奨配置:

```text
tests/fixtures/search/
tests/fixtures/halfkp/
tests/fixtures/teacher/
benchmarks/baselines/v2.5.4-plus-master.json
```

変更しないもの:

- 探索式
- move ordering
- 評価
- format
- 千日手挙動

完了条件:

- 同じbinaryと入力で意味的カウンタが再現する。
- 外部weightがない場合、テストは明確にskipまたはfixture modelを使う。
- baseline manifestだけで計測条件を復元できる。

### Phase 1: Workspace、CI、characterization test

目的:

- すべてのproduction componentを一つの標準コマンドで検証する。

作業:

1. root workspaceを作り、現`shogi_lib`をmemberへ追加する。
2. Cargo.lockを一本化する。
3. resolverとrelease profileをworkspace rootで管理する。editionは当面両packageで
   Rust 2021を明示し、検証していない`rust-version`は宣言しない。
4. 初期段階ではcrate名を変更しない。
5. LinuxとWindowsのPR CIを追加する。
6. HalfKP-32、HalfKP-64を必要なtargetだけでmatrix化する。
7. `cargo test --workspace`でroot 31件と`shogi_lib` 33件が実行されることを確認する。
8. USI transcript test harnessを作る。
9. clippy ratchetをproductionとresearchで分ける。

USI fixture:

- `usi`、`isready`、`usinewgame`
- startposから通常`go`
- forced mate
- 合法手がすべて低scoreのroot
- `go`直後の`stop`
- search中の`stop`
- stop-before-start
- worker panicを注入した場合
- evaluator未設定
- 連続する`go`
- `quit`
- Windows subprocess起動と終了

完了条件:

- PRでLinux/Windows buildが動く。
- `cargo test --workspace`がposition/movegen testsを取りこぼさない。
- 合法非終端`go`ごとにbestmoveが一つであることをprocess外から検証できる。
- workspace化前後のrelease profile benchmarkが基準内である。

### Phase 2: 巨大moduleの機械的分割

目的:

- 依存関係を変える前に、現行algorithmの責務をソース上で分離する。

`ai.rs`の移行順:

1. score定数とmate score変換
2. TT entry、local/shared TT
3. search limits、statistics
4. move ordering state
5. qsearch
6. alpha-beta
7. iterative deepening
8. worker setupとparallel coordinator

`evaluation.rs`の移行順:

1. constantsとpiece value
2. HalfKP feature semantics
3. HalfKP model codec wrapper
4. accumulator
5. portable/AVX2 kernels
6. KPP sparse model
7. Tiny NNUE
8. debug、SFEN helper

各移行は一つのPRまたはレビュー可能な小単位にする。
`ShogiAI`と`EngineEvaluator`は互換facadeとして残し、54 binaryを同時変更しない。

検証:

- search fingerprint完全一致
- model byte完全一致
- incremental/full error上限維持
- portable/AVX2一致
- 7回中央値で3%以上悪化しない

完了条件:

- `ai.rs`と`evaluation.rs`がfacadeまたはmodule rootとして読める規模になる。
- search coreの主要責務がprivate module境界で分離される。
- algorithm、定数、探索順の意図しない差分がない。

### Phase 3: 形式とモデル表現の統合

目的:

- runtime、trainer、toolが同じ形式定義を使う。

作業順:

1. 現行byteをそのまま扱う`HalfKpHeader`とcodecを追加する。
2. golden fixtureで旧loader/writerと一致させる。
3. runtime loaderをshared codecへ移行する。
4. `halfkp_search_train`を移行する。
5. `halfkp_train`を移行する。
6. `halfkp_kpp_train`を移行する。
7. 重複magic、version、flat forwardを削除する。
8. HKST v2にも同じ手順を適用する。

検証:

- 旧writerと新writerのSHA-256一致
- 旧loaderと新loaderの全fixture score一致
- wrong width、wrong version、truncationを拒否
- runtime aligned変換後のscore一致
- teacher recordのcandidate順とflag byte一致

完了条件:

- HalfKP magic/versionの定義箇所が一つ。
- HKST header/recordの定義箇所が一つ。
- runtimeのSIMD hot loopは不必要なgeneric abstractionを通らない。

### Phase 4: Search APIとUSI lifecycleの分離

目的:

- search algorithmをプロトコル、thread lifecycle、stdoutから独立させる。

作業:

1. `SearchLimits`をUSI private typeからengine typeへ移す。
2. `SearchInfo`、`SearchStats`、`RootResult`、`SearchOutcome`を導入する。
3. stdout出力をobserver eventへ置き換える。
4. serial pathを新APIの参照実装として通す。
5. parallel coordinatorを新APIへ移す。
6. USI側にgeneration付き`SearchJob`を追加する。
7. cancel、join、fallback、bestmove出力をsessionへ集約する。
8. teacherとbenchmarkをUSIではなくsearch APIへ接続する。

変更しないもの:

- worker 0 ownership
- helper workerの役割
- Threads semantics
- TT replacement
- fallbackの合法手選択policy
- `info`に含まれる既存情報

検証:

- Phase 1の全USI transcript
- serial exact fingerprint
- parallel legalityとbestmove一意性
- stop-before-start
- panic injection
- Windows process shutdown

完了条件:

- search moduleに`println!`、USI token、stdin処理がない。
- job threadはUSI sessionが所有し、quit時に未回収threadを残さない。
- 一つの`go`に対する最終出力箇所が一つ。

### Phase 5: 千日手・連続王手の正しさ修正

目的:

- search、USI、benchmark、teacherで同一の正しい裁定を使う。

このPhaseは分類Cであり、探索snapshotの変化を許可する。

作業:

1. 公式規則を基に必要な履歴情報と判定区間を仕様化する。
2. 現行挙動のcharacterization testを追加する。
3. `HistoryView`または統一`GameHistory`を実装する。
4. 各plyのkey、side、gave-check情報を判定可能にする。
5. searchのdo/undoと履歴を同期する。
6. USI `position ... moves`の全履歴を探索へ渡す。
7. in-process benchmarkとUSI benchmarkを同じadjudicatorへ移す。
8. teacher semantics versionを更新する。
9. 影響する教師データを再生成する。

最低限のtest:

- 王手を伴わない通常千日手
- 一方の真の連続王手
- 現在王手だが途中に非王手を含む反復
- 両者の王手が混在する履歴
- USIで探索開始前に3回出現している局面
- search subtree内で4回目になる局面
- undo後の履歴復元
- 256plyを超える履歴
- benchmarkとengineの裁定一致

完了条件:

- position keyだけの独立circular bufferへ依存しない。
- 同一fixtureをsearch、USI、benchmarkが同じ結果に裁定する。
- 新旧教師データの意味versionがmanifestで区別される。

### Phase 6: データ、教師、trainerの再構成

目的:

- 大規模学習を再現可能、検証可能、bounded memoryにする。

#### Phase 6A: 規約の一本化

- CSA parser、game metadata、result、phaseをlibrary化
- phase境界を一箇所で定義
- content/game-ID based split v2を導入
- split algorithm、seed、input IDをmanifestへ記録

#### Phase 6B: 実験manifest

最低限のfield:

```text
schema_version
stage
created_at
git_revision
git_dirty
rustc_version
target
input files + sha256 + record counts
model sha256
engine binary sha256
feature profile
search limits
threads/jobs
random seeds
phase/split policy versions
teacher semantics version
optimizer and hyperparameters
output files + sha256 + record counts
parent manifest hashes
```

stage再利用はmanifest fingerprint一致時だけ許可する。
directory名や`.complete`だけを根拠にしない。

#### Phase 6C: teacher session再利用

- Rayon workerごとにsessionを構築
- scratchを再利用
- 候補ごとのreset契約をtest
- 旧実装とteacher file SHA-256を比較
- 同値確認後にのみ、TT共有等を別の性能実験として検討

#### Phase 6D: streaming trainer

1. manifestから件数を読み、count目的の全decodeを廃止する。
2. validation/testをbounded chunk評価へ変更する。
3. active featureごとのhidden gradient複製をやめる。
4. batch accumulatorを再利用する。
5. deterministic reductionを維持する。
6. peak RSS、epoch time、weight hash、metricを記録する。

#### Phase 6E: optimizer

- AdaGradを参照pathとして維持
- schedule-freeは実験featureまたはresearch packageへ分離
- optimizer state round-trip
- uninterrupted 2 epochsと1 epoch + resumeの一致
- best checkpoint選択とtest isolationを検証

完了条件:

- 同一manifestから同一splitとstage構成を再現できる。
- stale artifactを誤って再利用しない。
- trainerのpeak RSSがdataset総量に比例しない。
- semantics不変の変更ではteacher/model hashまたは許容誤差が一致する。

### Phase 7: Binary、文書、生成物の整理

目的:

- 利用者と開発者が、正式機能と研究用実験を区別できる状態にする。

作業:

1. `autobins = false`へ移行する。
2. binary inventoryを作り、次の分類を付ける。
3. production/training/benchmark/researchへpackageまたはdirectoryを分ける。
4. 旧CLI名は一リリース以上thin shimとして維持する。
5. `main`を原則としてargument parseとlibrary callへ縮小する。
6. READMEを利用者向け入口にする。
7. `docs/README.md`を開発文書の索引にする。
8. reportにstatus、revision、input、結論、後継へのmetadataを追加する。
9. 重要な採否判断をADRへまとめる。
10. artifact directoryとignore policyを統一する。

binary分類:

| 分類 | 例 | default build |
|---|---|---|
| production | `usi_engine` | 含む |
| supported training | dataset、teacher、trainer | 明示build |
| benchmark/profile | `usi_benchmark`、`search_profile` | CI smokeのみ |
| research | MMTO、probe、dump、gate | defaultから除外 |
| deprecated | 互換shim | 期限を明記 |

生成物policy:

```text
tests/fixtures/       小さく、固定され、Git追跡するもの
data/raw/             入力dataset。Git非追跡
data/derived/         teacher shard等。Git非追跡
runs/                 checkpoint、log、model、match。Git非追跡
report/               人がレビューした要約。Git追跡
```

既存pathからの移行コストを確認してから最終名称をADRで決める。
`*.binary`や`*.hkst`を無条件に全ignoreするとfixtureまで隠れるため、directory単位のpolicyを優先する。

完了条件:

- Cargoと文書のbinary一覧が一致する。
- production buildがresearch dependencyを不要にcompileしない。
- READMEからarchitecture、refactoring plan、build、trainingへ到達できる。
- rejected experimentを検索でき、同じ失敗を理由なく繰り返さない。

### Phase 8: 性能最適化とcutover

目的:

- 整理後の境界を使い、計測で裏付けられた改善だけを採用する。

候補の優先順:

1. teacherでのsearch object再構築
2. trainerのgradient allocationとHashMap
3. evaluator contextの`Any` downcast
4. qsearch内のcheck判定とAttackInfo再利用
5. shared TTのlock、hash、replacement cost
6. crate境界を越えるhot callのinline/LTO

qsearch move generationは数字上有望でも、過去の直接生成実験が悪化しているため独立研究とする。

一件ごとの手順:

1. profilerで対象costを確認
2. 仮説を文書化
3. exact semanticsを保存するtestを追加
4. 一つの実装変更
5. 7回benchmark
6. 2〜5%のノイズ域なら15回再測定
7. 必要ならpaired games
8. 採用またはrejected report化

cutover条件:

- Linux/Windows E2E
- fixed search snapshots
- format fixtures
- teacher/trainer smoke
- resume test
- dedicated benchmark
- paired matchまたはSPRT
- release candidateをShogiHomeで起動

---

## 9. 推奨PR列

以下は、実際に着手する単位の推奨順である。

| PR | 分類 | 内容 | 依存 | 主な合否 |
|---|---|---|---|---|
| 001 | O | baseline manifestと固定局面 | なし | 再現性 |
| 002 | O | search fingerprint harness | 001 | exact counters |
| 003 | O | workspace化、単一lockfile | 001 | workspace tests、perf |
| 004 | O | Linux/Windows PR CI | 003 | CI green |
| 005 | O | USI transcript harness | 003 | one bestmove |
| 006 | O | HalfKP/HKST golden fixtures | 003 | exact bytes |
| 007 | M | search score、TT分割 | 002 | exact fingerprint |
| 008 | M | qsearch、alpha-beta分割 | 007 | exact fingerprint |
| 009 | M | iterative、parallel分割 | 008 | exact fingerprint |
| 010 | M | evaluation feature/model分割 | 006 | exact score/bytes |
| 011 | M | accumulator/kernel分割 | 010 | incremental/SIMD |
| 012 | M | shared HalfKP codec | 006, 010 | exact bytes |
| 013 | M | runtimeをshared codecへ移行 | 012 | exact score |
| 014 | M | trainerを一つずつ移行 | 012 | exact output |
| 015 | M | HKST codec統合 | 006 | exact records |
| 016 | M | structured Search API | 009 | serial exact |
| 017 | M | USI SearchJob state machine | 005, 016 | transcript/E2E |
| 018 | C | historyと千日手修正 | 017 | rule fixtures |
| 019 | D | teacher semantics version | 018 | manifest separation |
| 020 | M | CSA/phase/result共有化 | 003 | fixture一致 |
| 021 | D | content-based split v2 | 020 | stable split |
| 022 | O | experiment manifest | 021 | stale reuse拒否 |
| 023 | P | teacher session再利用 | 016, 022 | teacher hash/time |
| 024 | P | streaming validation/test | 022 | metrics/RSS |
| 025 | P | gradient表現圧縮 | 024 | weight hash/RSS |
| 026 | O | binary inventory/autobins | 003 | supported builds |
| 027 | O | docs/report/artifact整理 | 026 | link/inventory |
| 028+ | P | 一件ずつ性能実験 | すべて | perf/strength |

PR番号は実際のGitHub番号ではなく、依存順を示す論理番号である。
一つの論理PRが大きくなった場合は、合否条件を維持したままさらに分割する。

---

## 10. 検証設計

### 10.1 Search fingerprint

Threads=1では、機械的変更に対して少なくとも次を局面ごとに比較する。

```json
{
  "sfen": "...",
  "limits": {
    "depth": 5,
    "threads": 1
  },
  "bestmove": "7g7f",
  "score": 123,
  "pv": ["7g7f", "3c3d"],
  "completed_depth": 5,
  "nodes": 1000,
  "qnodes": 700,
  "q_generated": 5000,
  "q_discarded": 4200
}
```

TT hit数、cutoff数など、実装移動で一致すべき統計も利用可能なら保存する。
elapsed timeとNPSは同じJSONのexact fieldにせず、別のperformance reportへ記録する。

### 10.2 Format gate

- header byte
- tensor byte order
- total length
- writer output SHA-256
- loader後のdimension
- 固定局面score
- malformed input error kind

機械的format移行ではbyte-for-byte一致を要求する。

### 10.3 Incremental evaluation gate

対象:

- quiet move
- capture
- promotion
- capture + promotion
- drop
- king move、king bucket change
- hand count増減
- do/undo連続
- portable/AVX2

full refreshとの差の既存上限を維持し、最大誤差と発生局面を出力する。

### 10.4 Teacher gate

- record count
- candidate countと順序
- candidate move
- score
- flags
- root result
- file SHA-256

正しさ変更後に差分が生じる場合は、旧versionと新versionを別fixtureにする。

### 10.5 Trainer gate

- 1 batchのloss
- gradient checksum
- update後weight checksum
- optimizer state checksum
- epoch metrics
- best checkpoint
- uninterrupted 2 epochs
- 1 epoch + save + resume + 1 epoch

浮動小数点並列化を意図的に変える場合はbitwise一致の代わりに許容誤差を定義するが、そのPRで理由と上限を明記する。

### 10.6 性能gate

専用または安定した同一machineで実施する。

標準手順:

1. release、正しいfeature、clean treeでbuild
2. binary、weight、corpusのhashを記録
3. warm-up
4. 7回実行
5. 中央値と分散を記録
6. 3%以上の悪化で原則停止
7. 2〜5%範囲は15回再実行してノイズを確認

hosted CIのNPSはCPU steal、周波数、共有負荷の影響が大きいためreport-onlyとする。
node、score、bestmoveなど決定的な値はCIでもhard gateにする。

### 10.7 棋力gate

200局の単純な勝敗はsmokeとして使えるが、強さの証明には不足する。
openingを先後入替でpair化し、pair単位の結果を保存する。

推奨:

- 小変更: 200〜400局のpaired smoke
- 探索・評価の採用判断: 非劣性SPRT
- 初期仮説例: `H0 <= -10 Elo`、`H1 >= 0 Elo`
- `alpha = 0.05`、`beta = 0.05`
- 最大局数と中止条件を事前にmanifestへ記録

opening setを少数局面だけで繰り返さず、release判断では多様化する。
`Threads=0`のengineを使うときは原則`JOBS=1`とし、複数対局のCPU競合を避ける。

---

## 11. CI構成

### 11.1 PR必須

Linux:

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo test --workspace --features halfkp64
cargo check --workspace --all-targets --features halfkp64
cargo clippy <production packages> --all-targets --features halfkp64
git diff --check
```

Windows:

```text
production build
portable fallback build
USI transcript subprocess tests
start/stop/quit lifecycle
```

### 11.2 Nightlyまたは専用runner

- fixed search profile
- HalfKP full/incremental trace
- portable/AVX2 comparison
- teacher generation smoke
- trainer one-epoch + resume smoke
- in-process/USI benchmark adjudication一致
- paired self-match smoke

### 11.3 Release

- clean revision
- production feature profile
- Linux/Windows artifact
- artifact SHA-256
- weight compatibility check
- ShogiHome manual smoke
- release notesにformat、semantics、CLI互換性を記載

---

## 12. Git、branch、rollback方針

### 12.1 branch

- 長期間の巨大な`refactor-all` branchを作らない。
- `refactor/workspace`、`refactor/search-modules`のように一つの境界へ限定する。
- performance実験は`perf/<hypothesis>`に分ける。
- correctness変更は`fix/repetition-history`のように独立させる。

### 12.2 commit

- file移動と内容変更を可能なら別commitにする。
- mechanical commitには「no behavior change」と検証結果を記録する。
- generated weight、teacher shard、checkpoint、match recordをcommitしない。
- fixtureは小さく、生成方法とhashを記録する。

### 12.3 rollback

各PRは次を満たす。

- 直前PRへ単純revertできる。
- format migration中は旧readerとの比較pathを一時的に残す。
- CLI移行中は旧entry pointをshimとして残す。
- data semantics変更では旧manifest readerを直ちに削除しない。
- performance変更が不採用なら、仮説と計測をreportへ残して実装だけrevertする。

---

## 13. 文書管理方針

### 13.1 文書区分

| 区分 | 内容 | 更新方針 |
|---|---|---|
| architecture | 現在の責務、依存、API | 実装変更と同じPR |
| guide | build、実行、学習、release手順 | CLI変更と同じPR |
| format | binary schema、version、compatibility | format変更前に更新 |
| ADR | 採用した重要判断と代替案 | 判断時に追加、原則改変しない |
| experiment/report | 仮説、条件、結果、採否 | 実験終了時に追加 |
| handoff | release時点の契約 | 次の大規模移行時に作成 |

### 13.2 report metadata

新規reportには少なくとも次を含める。

```text
status: accepted / rejected / inconclusive / superseded
date
revision
hypothesis
input hashes
config
machine
result
decision
follow-up
```

過去reportは削除しない。
同じ失敗を繰り返さないため、rejectedも一級の成果として索引へ載せる。

### 13.3 更新責任

実装PRで次が変わった場合、そのPR内で対応文書も更新する。

- public command
- feature
- file format
- crate/module境界
- search contract
- data split/teacher semantics
- benchmark command
- artifact path

文書だけが先行して未実装の内容は「目標」「提案」「未決定」と明示する。

---

## 14. 既知の危険な進め方

次を禁止または強く避ける。

1. `ai.rs`と`evaluation.rs`を一つのPRで全面書き換えする。
2. module分割のついでに探索parameterを調整する。
3. discard率だけを見てqsearch direct generatorを再実装する。
4. SIMD hot loopを汎用traitやiteratorへ抽象化する。
5. 全warningを`cargo clippy --fix`で一括修正する。
6. `Position::switch_turn`を用途確認なしに削除する。
7. `Any` downcastをbenchmarkなしでgenericへ変更する。
8. TT、history、killerを教師候補間で共有して「再利用」と呼ぶ。
9. path-based splitを無告知でcontent-basedへ置き換える。
10. 千日手修正後も旧教師datasetを同じversionとして使う。
11. `.complete`または非空fileだけでstageを再利用する。
12. 200局の独立二項区間だけでweightをrelease昇格する。
13. reportを「古いから」という理由で削除する。
14. 54 binaryを一括renameして利用者のscriptを壊す。

---

## 15. リスク登録

| リスク | 影響 | 検出 | 対策 |
|---|---|---|---|
| crate分割でinlineが失われる | NPS低下 | dedicated benchmark | module分割を先行、ThinLTO維持 |
| move orderが変わる | node/棋力変化 | exact fingerprint | stable/unstable sortを保存 |
| float加算順が変わる | model差分 | weight/gradient hash | reduction変更を別PR |
| USI旧threadが出力する | 複数bestmove | generation transcript | job ownership、join |
| stopが初期化で消える | 応答停止 | stop-before-start test | monotonic token |
| format片側だけ移行 | model非互換 | golden bytes | consumerを一つずつ移行 |
| 千日手修正でteacher混在 | label汚染 | manifest version | regenerate、version拒否 |
| split移行でleakage | 評価過大 | game ID audit | content-based split |
| streamingでshuffle意味変更 | 学習結果変化 | record order/hash | version化、段階移行 |
| clippy整理がhot pathを変える | 性能低下 | perf gate | ratchet方式 |
| research codeがCIを不安定化 | 開発速度低下 | package別CI | default-members分離 |
| artifact誤commit | repository肥大 | CI size/path check | directory policy |

---

## 16. 各PRの標準作業手順

### 16.1 着手前

- [ ] PR分類をM/C/P/D/Oから一つ選んだ。
- [ ] 対象責務と対象外を説明した。
- [ ] 基準revisionと作業tree状態を記録した。
- [ ] 必要なfixtureとbaselineが存在する。
- [ ] hot pathの場合、変更前benchmarkを実行した。
- [ ] data/modelを使う場合、SHA-256を記録した。

### 16.2 実装中

- [ ] 一つの責務境界だけを変更している。
- [ ] 無関係なfmt、rename、warning修正を混ぜていない。
- [ ] fallback、panic、stop経路も移行した。
- [ ] Windows/portable pathを削っていない。
- [ ] generated artifactをGitへ追加していない。

### 16.3 検証

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo test --workspace`
- [ ] HalfKP-64 tests/checks
- [ ] `git diff --check`
- [ ] search fingerprint
- [ ] format/teacher hash
- [ ] 必要なperformance benchmark
- [ ] 必要なpaired games
- [ ] `git status`で意図した差分だけを確認

### 16.4 レビュー説明

- [ ] 変更前の責務と問題
- [ ] 変更後の依存方向
- [ ] 意図的に変わるもの
- [ ] 変わらないことを確認した値
- [ ] 計測環境とhash
- [ ] rollback方法
- [ ] 後続PR

---

## 17. 標準検証コマンド

現行構成で利用できる基準コマンド:

```bash
RUST_FONTCONFIG_DLOPEN=1 cargo check --release --features halfkp64 \
  --bin dataset_build \
  --bin halfkp_search_teacher \
  --bin halfkp_search_train \
  --bin usi_engine \
  --bin usi_benchmark

RUST_FONTCONFIG_DLOPEN=1 cargo test --release --features halfkp64 --lib

RUST_FONTCONFIG_DLOPEN=1 cargo test --release --features halfkp64 \
  --bin halfkp_search_train

cargo test --release --manifest-path shogi_lib/Cargo.toml

RUST_FONTCONFIG_DLOPEN=1 cargo check --features halfkp64 --all-targets

cargo fmt --all -- --check
git diff --check
git status --short
```

workspace化後は、独立した`shogi_lib`コマンドを原則として次へ統合する。

```bash
cargo test --workspace
cargo test --workspace --features halfkp64
```

performance基準:

```bash
target/release/search_profile \
  --halfkp-weights policy_weights_halfkp64_kpp_distilled_v2.5.0.binary \
  --positions taya36.sfen \
  --samples 16 \
  --depth 5 \
  --seed 9501 \
  --threads 1
```

大規模学習や対局を実行する場合は、handoffに記録されたコマンドと注意事項も参照する。

---

## 18. 完了定義

リファクタリング全体は、単にファイルが小さくなった時点では完了としない。
次をすべて満たした時点を完了とする。

### 構造

- [ ] workspaceとlockfileが一つ。
- [ ] production、training、benchmark、researchの境界がCargo上で明確。
- [ ] search coreがUSI/stdoutへ依存しない。
- [ ] HalfKP/HKST format定義が一つ。
- [ ] phase、result、splitの定義が一つ。
- [ ] supported binaryの`main`が薄い。
- [ ] productionからresearch codeへ逆依存しない。

### 正しさ

- [ ] 非終端`go`が全経路で合法bestmoveを一つ返す。
- [ ] full USI historyがsearchへ渡る。
- [ ] 千日手と連続王手が正しく区別される。
- [ ] search、benchmark、teacherが同じ裁定を使う。
- [ ] checked qsearch、TT mate normalization、worker 0 ownershipがtestされる。

### 評価と形式

- [ ] HalfKP-32/64 mismatchを明確に拒否する。
- [ ] runtime/trainer/toolがshared codecを使う。
- [ ] incremental/full refreshとportable/AVX2が許容誤差内。
- [ ] リリース済みweightをversionなしで壊していない。

### 学習

- [ ] dataset、teacher、trainingの全stageにmanifestがある。
- [ ] splitがpath移動で変わらない。
- [ ] stale artifactを自動再利用しない。
- [ ] trainer memoryがdataset全量に比例しない。
- [ ] checkpoint resumeが次epochを再現する。
- [ ] validationとtestの役割が分離される。

### 性能と棋力

- [ ] mechanical変更はexact fingerprintを維持。
- [ ] dedicated benchmarkで未説明の3%以上の中央値低下がない。
- [ ] 意味変更は差分と理由が記録される。
- [ ] 探索・評価変更はpaired gamesまたはSPRTを通る。
- [ ] Linux/Windows release candidateがShogiHomeで起動する。

### 保守性

- [ ] READMEから本書と文書索引へ到達できる。
- [ ] binary inventoryが実装と一致する。
- [ ] reportに採否と後継が記録される。
- [ ] generated artifactとtracked fixtureの境界が明確。
- [ ] 新しい開発者が本書から次の未完了PRを判断できる。

---

## 19. 未決定事項

以下は、実測または移行コストを確認してからADRで決める。

1. `shogi_lib`を`shogi_position`へrenameするか。
2. production `usi_engine`を独立packageにする時点。
3. HalfKP-64をworkspace defaultにするか、production appだけで強制するか。
4. evaluator typed contextをenum、generic、session objectのどれにするか。
5. 生成物rootを`data/derived`、`artifacts`、`runs`のどれへ統一するか。
6. dedicated performance runnerをローカル固定機とself-hosted CIのどちらにするか。
7. release用SPRTのElo境界と最大局数。
8. teacher candidate flagを現行意味で文書化するか、schemaを更新するか。
9. streaming shuffleの互換性をどこまで維持するか。

未決定事項は、機械的リファクタリングを止める理由にはしない。
ただし、その選択によってformat、探索意味、学習意味、外部CLIが変わる直前までにはADRで確定する。

---

## 20. 進捗記録

Phaseの着手・完了時に以下へ追記する。

| Phase | 状態 | 開始revision | 完了revision | 結果・参照 |
|---|---|---|---|---|
| 0 Baseline | 完了 | `9926430` | `5aa5250` | [`benchmarks/baselines/`](../benchmarks/baselines/)、fingerprint/format fixture、性能差+1.81% |
| 1 Workspace/CI | 進行中（local gate整備済み） | `52cbd84` |  | 単一lockfile、USI process test、warning ratchet、workspace全test成功、性能差は原始基準比+1.21% |
| 2 Module split | 完了 | `a71cc3b` | `4da6f3e` | search/evaluation責務をprivate moduleへ分離、fingerprint一致、原始性能差+1.18% |
| 3 Format consolidation | 完了 | `f1ecb8b` | `5941250` | HalfKP/HKST codecを単一定義化、golden byte一致、paired性能差+0.12% |
| 4 Search/USI separation | 完了 | `edf04b0` | `edf04b0` | typed outcome/observer、generation job、single bestmove emitter、性能差+0.38% |
| 5 Repetition correctness | 完了 | `8a86417` | `7cc7413` | 完全履歴と王手区間による公式裁定、教師意味論v3、paired性能差+1.69% |
| 6 Data/Training | 完了 | `be5897c` | `fd2c5cd` | 規約一本化、content split v2、全stage manifest、bounded-memory trainer、resume完全一致 |
| 7 Repository/Documents | 完了 | `7f8251d` | `4d267e4` | 57 targetを明示分類、production依存分離、README/文書索引/ADR/artifact・report policy、paired性能差-0.77% |
| 8 Performance/Cutover | 未着手 |  |  |  |

### 2026-07-20 Phase 0途中経過

- `b72a717`: 本計画書と文書索引を追加。
- `9926430`: 外部HalfKP性能基準manifestと固定7局面を追加。
- `96c59e5`: Threads=1の決定的search fingerprint gateを追加。
- `a5becf4`: HalfKP-32/64 v1 headerとHKST v2のgolden fixture、破損入力testを追加。
- `5aa5250`: revision、toolchain、CPU、artifact hash、commandを取得するmetadata toolを追加。
- fingerprintを2回実行し、bestmove、root score bit、PV、node/qsearch/aspiration統計の完全一致を確認。
- HalfKP性能基準と同一条件で3回再測定し、全決定的カウンタの一致を確認。
- 実行時間中央値は`6086.42 ms`から`6196.43 ms`で、差は`+1.81%`。3%の停止基準内。
- HalfKP-64とHalfKP-32のlibrary test各35件、`shogi_lib` 33件、trainer 1件、HalfKP-64 all-target checkが成功。

Phase 0は完了した。

### 2026-07-20 Phase 1途中経過

- Phase 0の`refactor/phase0-baseline`を保存し、stacked branch `refactor/phase1-workspace`を作成。
- `52cbd84`: root packageと`shogi_lib`をresolver 2のworkspaceへ統合。
- root `Cargo.lock`へ一本化し、`shogi_lib/Cargo.lock`と無効になるmember側profileを削除。
- `cargo test --workspace --release --features halfkp64`で、root library 35件、trainer 1件、`shogi_lib` 33件、全binary test targetが成功。
- search fingerprintは完全一致。
- HalfKP探索の全決定的カウンタは一致。実行時間中央値は`6160.15 ms`で、Phase 0再測定比`-0.59%`、原始基準比`+1.21%`。
- `8233f65`: Linux/WindowsのPR CIを追加。HalfKP-32 library、HalfKP-64 workspace、all-target、clippy baseline、fingerprintを検証対象化。
- `318fcd2`: 実際の`usi_engine` subprocessへstdinでcommandを送り、stdoutを検証する
  transcript testを追加。handshake、evaluator未設定、通常局面、合法手なし局面、
  `go infinite`直後の`stop`、`quit`を対象とし、各`go`の`bestmove`が一件であることを確認。
- transcript testは15秒の応答timeout、5秒の終了timeout、失敗時のprocess killを持ち、
  外部weightへ依存せずtest内で生成したTiny NNUE fixtureを使用する。
- 初期transcript test 3件を3回反復して全件成功。続けてHalfKP-64 workspace全testと
  search fingerprintの完全一致を確認。
- `d96dbd6`: production library用Clippy ratchetを追加。`shogi_lib`はwarningゼロ、
  `shogi_ai` libraryは既存lint classだけを明示的に隔離し、新しいclassのwarningを失敗させる。
- 同commitでRust stable運用、単一lockfile、release metadata、MSRV宣言条件を
  [`toolchain_policy.md`](toolchain_policy.md)へ記録。未検証のMSRV値は宣言しない。
- warning解消に伴うZobrist初期化のiterator化後、`shogi_lib` 33件とsearch fingerprintが成功。
  RNG呼出順とposition hashは維持された。
- `e5ff34e`: transcriptを5件へ拡張。既知の強制詰みで`S*5g`を維持し、
  stop-before-start、`info depth`受信後のsearch中stop、`usinewgame`後の再探索を追加した。
- 拡張後の5件を3回反復し、全件成功。各runは約3.2秒で完了した。

Phase 1の残作業は、remote CIでLinux/Windows jobを実行して結果を記録することである。
worker panic注入やoverlapする`go`の所有権検証など、内部へのfault injectionまたは
search所有権の変更が必要なcaseは、Phase 4のSearch/USI分離と同時に実装する。

### 2026-07-20 Phase 2途中経過

- stacked branch `refactor/phase2-module-split`を作成。
- `a71cc3b`: mate score変換、USI表示score、履歴依存score判定を`ai/score.rs`へ、
  local/shared transposition tableとreplacement policyを`ai/transposition.rs`へ移動。
- `7676115`: piece value、KPP/NNUE/HalfKPの公開次元定数、盤上square tableを
  `evaluation/constants.rs`へ移動。既存の`evaluation::*`公開pathはre-exportで維持。
- `0f30666`: KPP pair featureとTiny NNUE feature抽出を`evaluation/features.rs`へ移動。
- `432e275`: HalfKPの視点変換、左右mirror、piece state、固定長feature抽出を
  同じfeature semantics moduleへ移動。公開する`HalfKpFeatures`と抽出関数はre-exportで維持。
- `2131c92`: little-endian scalar/array読込とaligned HalfKP row構築を
  `evaluation/codec.rs`へ移動。golden headerとmodel load testでbyte契約を確認。
- `81b9697`: AVX2 availability判定、accumulate、差分適用kernelを
  `evaluation/kernels.rs`へ移動。portable/AVX2一致testと差分/全refresh一致testが成功。
- `b305ecf`: KPP index decodeと調査用SFEN生成を`evaluation/debug.rs`へ分離。
  公開pathはre-exportで維持し、workspace all-target checkで利用binaryを確認。
- `87b1ed4`: `Evaluator` traitと`Arc<T>`委譲実装を`evaluation/evaluator.rs`へ分離。
- `c8382ba`: Tiny NNUEのmodel表現、strict loader、forwardを
  `evaluation/tiny_nnue.rs`へ分離。model fixture、USI process test、fingerprintが成功。
- `cad5c7e`: `EngineEvaluator`、sparse adapter、hybrid adapterを
  `evaluation/facade.rs`へ分離。既存constructor、variant、context委譲を維持。
- `1e7d0d4`: sparse KPP model、strict loader、pair評価を`evaluation/sparse.rs`へ分離。
- `1d9487b`: HalfKP model、strict loader、accumulator、差分更新contextを
  `evaluation/halfkp.rs`へ分離。既存のrefresh条件とportable/AVX2 dispatchを維持。
- `5b25c89`: quiescence searchを`ai/qsearch.rs`へ分離。
- `6872874`: alpha-beta本体を`ai/alpha_beta.rs`へ分離。
- `9edc4f8`: aspiration windowを含むiterative deepeningを`ai/iterative.rs`へ分離。
- `4da6f3e`: worker生成、panic回収、root result集約を含むparallel coordinatorを
  `ai/parallel.rs`へ分離。
- 各移動後にHalfKP-64 library test 35件とsearch fingerprintの完全一致を確認。
  評価constants移動後はHalfKP-32 library test 35件も成功。
- algorithm、定数値、TT shard数、replacement順、RNG、公開APIは変更していない。
- 最終状態で`cargo test --workspace --release --features halfkp64`を実行し、
  `shogi_ai` 35件、trainer 1件、USI transcript 5件、`shogi_lib` 33件が成功。
  全binary test targetのbuild、Clippy ratchet、`cargo fmt --check`も成功した。
- 性能基準と同一条件で7回再測定し、全決定的カウンタが基準と一致した。
  実行時間中央値は`6158.48 ms`で、原始基準比`+1.184%`、Phase 1比`-0.027%`。
  3%の停止基準内である。

Phase 2は完了した。

### 2026-07-20 Phase 3実施結果

- Phase 2の`refactor/phase2-module-split`を保存し、stacked branch
  `refactor/phase3-format-codec`を作成。
- `f1ecb8b`: 現行32 byteをcanonical representationとする`HalfKpHeader`を追加。
  HalfKP-32/64 golden fixtureのdecode/encodeが完全一致し、wrong magic、version、
  dimension、scale、truncationを拒否する。
- `494dcc2`: runtime loaderをshared header codecへ移行。aligned rowへの変換と
  portable/AVX2 hot loopは変更していない。
- `20b9ee5`、`6083be7`、`92c4821`: `halfkp_search_train`、`halfkp_train`、
  `halfkp_kpp_train`の順にshared header codecへ移行。
- `b54cada`、`af17a69`: feature tensor、hidden bias、output weight/biasを扱う
  `HalfKpFlatModel` codecを追加し、runtimeと3 trainerの重複loader/writerを統合。
  旧test writerのbyte列をdecodeして再encodeした結果は完全一致した。
- `1a2b89e`: HKST v2のheader、record marker、position、result、candidate codecを
  `halfkp_training/codec.rs`へ集約。既存のpublic reader/writerはfacadeとして維持。
  HalfKP-32/64 golden fixtureでheader、candidate順、flag byteを完全一致させ、
  truncation、wrong version、不正marker/result/side/featureを拒否した。
- `5941250`: 3 trainerに重複していたportable accumulatorとflat forwardを共有化。
  feature加算順、clamp、material項、STM/NSTM順は維持し、finite-difference testが成功。
- 最終状態でHalfKP-64 library 35件、trainer 1件、all-target check、
  Clippy ratchet、format check、search fingerprintが成功した。Phase途中の全workspace
  release testではUSI transcript 5件と`shogi_lib` 33件を含む全targetが成功した。
- 初回の非paired性能再測定はhost全体の速度低下により3%基準を超えたため進行を停止し、
  Phase 2完了revision `89245bd`を別worktreeで同一sessionに再buildして5回ずつ交互測定した。
  Phase 2中央値`6275.83 ms`、Phase 3中央値`6283.06 ms`、差は`+0.115%`だった。
  全決定的カウンタは両binaryで一致し、3%の停止基準内である。

Phase 3は完了した。

### 2026-07-20 Phase 4実施結果

- Phase 3の`refactor/phase3-format-codec`を保存し、stacked branch
  `refactor/phase4-search-api`を作成。
- `edf04b0`: engine側へ`SearchLimits`、`SearchInfo`、`SearchStats`、`RootResult`、
  `SearchOutcome`、`SearchObserver`を追加。既存`find_best_move*`は互換wrapperとして残した。
- iterative deepeningから`println!`、stdout flush、USI score変換、move文字列化を除去し、
  raw score、elapsed、nodes、PVをobserver eventで通知するようにした。
- serial/parallel searchはtyped outcomeを返すAPIを共有し、worker 0 ownership、
  helper worker、shared TT、fallback、Threads解決規則は維持した。
- USI側へgeneration付き`SearchJob`を追加。新しい`go`、`stop`、`usinewgame`、`quit`は
  active jobをcancelしてjoinし、未回収threadとoverlapするsearchを残さない。
- 最終応答は`emit_search_response`一箇所へ集約。job thread内外のpanic recovery、
  evaluator未設定、thread spawn失敗も一つの`bestmove`へ収束する。
- search fingerprint、HalfKP-64 library 35件、USI transcript 6件、
  all-target check、Clippy ratchet、format checkが成功した。transcriptへ連続する二つの
  `go`が直列に各一件の合法`bestmove`を返し、active search中の`quit`がcleanに終了する
  caseを追加した。Windows process shutdownはremote CI未実行のため未確認。
- 性能基準を7回測定し、全決定的カウンタが一致した。中央値は`6182.05 ms`で、
  Phase 2の安定基準`6158.48 ms`比`+0.383%`、3%の停止基準内だった。

Phase 4はlocal gateについて完了した。

### 2026-07-21 Phase 5実施結果

- Phase 4の`refactor/phase4-search-api`を保存し、stacked branch
  `refactor/phase5-repetition`を作成した。
- 裁定仕様は日本将棋連盟の
  [対局規定（抄録）](https://www.shogi.or.jp/match/taikyoku_rules/)を一次資料とした。
  同一局面は盤面、持駒、手番が同じ状態の4回出現、連続王手は反復区間内で片側の
  全着手が王手である場合に王手側の負け、と定義した。
- `8a86417`: position keyだけを保持する固定長circular bufferを、各局面のkey、
  手番、直前の着手側、王手有無を保持する`GameHistory`へ置換した。履歴は
  256 plyで切り捨てず、USI `position ... moves`の初期局面から全着手を復元する。
  search、parallel worker、in-process benchmark、USI benchmark、対局、fingerprint、
  profileが同じ履歴表現を利用する。
- 通常千日手、一方の連続王手、反復区間途中の非王手、両者の王手混在、undo、
  256 ply超、合法な盤面遷移による4回反復を規則fixtureにした。USI入力から
  3回出現済みの履歴を渡し、search subtree内の4回目を裁定してundo後に履歴が
  復元されることも検証した。
- `88c9e13`: 裁定時の一時`Vec`を除き、現在局面から同手番の過去entryを逆走して
  4回目と反復区間を判定するallocation-free実装へ変更した。
- `7cc7413`: 新規HKST0002へ`.manifest.json` sidecarを必ず書き、
  `teacher_semantics_version = 3`、
  `teacher_semantics_id = "jsa-complete-check-interval-v1"`を記録する。
  trainerはversion、ID、formatの不一致とmanifest欠落を既定で拒否する。
  manifest導入前の教師データは意味が確定できないため、利用者が
  `--allow-legacy-teacher-semantics`を明示した場合だけ読み込める。
  したがって、旧教師データは新意味のdatasetへ暗黙には混在しない。
- このPhaseは分類Cの規則修正である。既存fingerprint fixtureは反復を含まないため
  完全一致し、意図した探索差分は上記の規則fixtureで固定した。
- 最終状態でHalfKP-32/64 library各41件、HalfKP search trainer 3件、
  USI transcript 6件、`shogi_lib` 33件、HalfKP-64 workspace release testと
  all-target checkが成功した。Clippy ratchet、format check、search fingerprintも成功した。
- 初回の非paired絶対性能値はhostの速度変動を含んだため、Phase 4完了revision
  `10992f4`とPhase 5のallocation-free revision `88c9e13`を同一sessionで5回ずつ
  交互測定した。Phase 4中央値`6367.12 ms`、Phase 5中央値`6474.86 ms`、
  差は`+1.692%`で、全決定的カウンタは一致し3%の停止基準内だった。
  教師manifest変更は探索runtimeへ入らない。

既存の教師shardは入力由来情報を欠くため自動変換せず、新しい規則で再生成する。
リポジトリ内に生成済み教師artifactは追跡していない。Phase 5は完了した。

### 2026-07-21 Phase 6実施結果

- Phase 5の`refactor/phase5-repetition`を保存し、stacked branch
  `refactor/phase6-data-training`を作成した。
- `be5897c`: CSAの色・駒・着手変換、終局理由、勝者、rate metadata、phase境界、
  CSA file収集を`training_data`へ集約した。`dataset_build`、`csa_policy_dump`、
  `csa_rate_stats`、`kpp_learn`、search teacherが同じ規約を参照する。
  phase policy v1はopening 40 ply以下、middlegame 41--90 ply、endgame 91 ply以上である。
- split policy v2はCSA内容のSHA-256をgame IDとし、domain separator、seed、game IDの
  SHA-256からtrain/valid/testを決める。file pathをsplit keyとrecord sourceから除き、
  同一内容の重複CSAを一件へまとめた。rename、root directory変更、列挙順によって
  splitが変わらない。dataset manifestはpolicy version、全入力ID、出力hash、件数を持つ。
- `9f46bdb`: trainerの件数確認をHKST manifest参照へ変更し、全record decodeを廃止した。
  trainは既定25,000 record、validation/testは既定4,096 recordのchunkで処理するため、
  peak memoryはdataset総量に比例しない。既存の25,000行shardは一chunkとなるため、
  従来と同じseedのshuffle順を維持する。
- `dab2531`: active featureごとに64要素hidden gradientを複製せず、一局面につき
  black/white各一組とfeature index列を保持する形へ変更した。dense/sparse batch
  accumulatorはclearしてcapacityを再利用する。AdaGradとschedule-freeのcheckpointは
  optimizer全状態をround-trip testで検証し、state commit fileへ各構成fileの
  SHA-256とrun fingerprintを記録する。旧checkpointは明示opt-inなしには再開しない。
- `a5c1f40`: teacher生成はRayon workerごとに`TeacherSession`を一度だけ作る。
  各候補の前にTT、move ordering、killer、履歴、評価context、統計を完全resetする。
  fresh session版`dab2531`と再利用版を8局面、depth 1/2、58候補で比較し、HKST SHA-256は
  両方とも`643ebb63...39d22b`で完全一致した。
- `a587663`、`dc4259b`、`bf397bd`: dataset、JSONL shard、teacher、trainerの各stageへ
  manifestを追加した。revision/dirty、rustc、target、入力・model・engine・出力のhashと
  件数、feature profile、limit、jobs/threads、seed、policy/semantics version、optimizer、
  hyperparameter、parent manifest hashを該当stageで記録する。stage再利用はdirectory名や
  `.complete`ではなく、入力と設定から得たfingerprint、および全出力hashの一致時だけ行う。
  `train_halfkp64_large.sh`から`.complete`とshell `split`を除き、manifest付き
  `jsonl_shard`へ移行した。HKST本体とmanifestは一時fileから確定する。
- `7d927c1`: 2 epoch連続実行と1 epoch + resumeのmodel hashを比較したところ、最初の
  characterizationではprocessごとにランダムな`HashMap`走査順がgradient normの加算順を
  変え、1 ULP程度の差を生むことを検出した。sparse feature index順の決定的加算へ修正後、
  両modelは`ef70ee56...6ef7ae`でbyte完全一致し、validation scoreも完全一致した。
  test setはbest model確定後だけ読み、checkpoint選択へ混入しない。
- HalfKP-64、2 record、train/eval chunk 1のtrainer smokeで最大RSSは`83,256 KiB`、
  elapsedは`0.27 s`だった。この値はrelease model自体を含む小規模測定であり、絶対的な
  大規模運用上限ではないが、入力件数に依存する全保持経路がないことを確認する測定である。
- 最終状態でHalfKP-32/64 library各45件、search trainer 6件、USI transcript 6件、
  `shogi_lib` 33件、HalfKP-64 workspace release test、all-target check、Clippy ratchet、
  format checkが成功した。search fingerprintは完全一致した。
- Phase 5完了revision `e7cffe0`とPhase 6 `fd2c5cd`を別binaryで5回ずつ交互測定した。
  control中央値`6578.11 ms`、candidate中央値`6685.45 ms`、差は`+1.632%`で、
  全決定的カウンタは一致し3%の停止基準内だった。

大規模生成物はGitへ追跡せず、smoke fixtureでmanifest chain、hash検証、stale拒否を確認した。
実データのfull runでも同じbinaryとmanifest契約を使用する。Phase 6は完了した。

### 2026-07-21 Phase 7実施結果

- Phase 6の`refactor/phase6-data-training`を保存し、stacked branch
  `refactor/phase7-repository-docs`を作成した。
- `7f8251d`: `autobins = false`へ移行し、57個のbinary targetをCargoへ明示した。
  `usi_engine`だけをdefault production targetとし、supported training、benchmark/profile、
  researchをそれぞれ`training-tools`、`benchmark-tools`、`research-tools`でopt-inにした。
  target名は変更していないため旧CLI名のshimは不要である。将来の改名時は最低1 releaseの
  thin shimを維持する契約をinventoryへ記録した。
- source fileの一括移動は既存scriptと外部運用への影響が大きいため行わず、Cargo manifestの
  target groupを実行境界の正本とした。repository内の全script、CI、release workflowへ
  対応featureを追加し、`tools/check_binary_inventory.py`で`src/bin`、Cargo、文書の不一致を
  CI失敗にする。
- production buildからtraining専用moduleを除外し、`csa`、`glob`、`plotters`、`sha2`を
  optional dependencyにした。未使用だった`circular-buffer`、`flate2`、`indicatif`、`libc`、
  `ndarray`の直接依存を削除した。default dependency graphにtool専用dependencyが存在しない
  ことと、`cargo build --release`がproduction engineだけを対象とすることを確認した。
- `4d267e4`: READMEを利用者向けのShogiHome、production build、test、training、開発文書への
  入口へ縮小した。`docs/README.md`を索引として更新し、全binary inventory、artifact配置、
  report metadata、重要な採否判断をADR 0001/0002へ記録した。
- 新規artifactの標準配置を`tests/fixtures/`、`benchmarks/baselines/`、`data/raw/`、
  `data/derived/`、`runs/`、`report/`に定義した。既存`data/` pathはscript互換のため一括移動せず、
  新規runから段階移行する。生成物の同一性はpath/mtimeではなくmanifestとcontent hashで判定する。
- 全featureのworkspace release testが成功し、HalfKP-64 library 45件、search trainer 6件、
  USI transcript 6件、`shogi_lib` 33件を含む57 targetをbuild/testした。all-target check、
  production Clippy ratchet、全feature Clippy可視化、format、shell syntax、inventory checkが成功した。
  search fingerprintは完全一致した。
- Phase 6完了revision `cd0f422`とPhase 7 `4d267e4`を別binaryで5回ずつ交互測定した。
  control中央値`6679.41 ms`、candidate中央値`6628.10 ms`、差は`-0.768%`で、全決定的counterは
  一致し3%の停止基準内だった。target/dependency整理による探索性能低下はない。

Phase 7は完了した。
