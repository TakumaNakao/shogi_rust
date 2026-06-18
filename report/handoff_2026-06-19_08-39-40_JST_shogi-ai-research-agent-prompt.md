# 将棋AI研究開発 引き継ぎプロンプト

あなたは `/home/kuro/shogi_rust` の将棋AI研究開発を引き継ぐコーディングエージェントです。目的は、現行のRust製将棋AIを継続的に改良し、まず `v2.1.0` baseline を安定して圧倒し、その後さらに強い探索・評価関数へ発展させることです。

## 最重要ゴール

1. `v2.1.0` baseline に対して、100局以上のベンチで明確に勝ち越す。
2. 実験は必ずベンチで検証し、短い10局結果だけで採用しない。
3. 自己対局学習は過去に評価関数を壊した経緯があるため、盲目的に回さない。
4. 今は「探索改善・検証基盤・敗局分析」を優先し、その後に探索蒸留や小型NNUEへ進む。

## 新PCで clone 後に必要なもの

`policy_weights_v2.1.0.binary` は `.gitignore` 対象でgit管理外です。旧PCからリポジトリ直下へコピーする必要があります。

```bash
git clone <repo-url>
cd shogi_rust
git checkout improve-self-play-learning

# 旧PCからコピー
# ./policy_weights_v2.1.0.binary
```

ビルド時は環境によって `pkg-config` がなくても通るよう、以下を使います。

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin usi_engine --bin usi_benchmark --bin record_analyze
```

`converted_records2016_10818.sfen`、`taya36.sfen`、`report/` はgit管理済みです。

## 現在のブランチと重要コミット

作業ブランチ:

```text
improve-self-play-learning
```

最近の採用コミット:

```text
640b0b2 Handle checked repetition as perpetual check loss
503a81c Respect USI stop during search
2a35afd Record USI benchmark end reasons
85aff64 Summarize end reasons in record analysis
c576d37 Summarize paired benchmark starts
d8e461c Prioritize checking moves in search ordering
8c5c23a Use SEE to order tactical captures
a8d9b8a Handle USI go time limits
17bf635 Add shogi AI research reports
885ea88 Ignore generated learning logs
898f058 Show tail evaluation swings in record analysis
```

## 現在までの代表ベンチ結果

現行HEADは `v2.1.0` baseline に対して有望です。

代表結果:

```text
seed 1401 / 20 games: 16-4
seed 1701 / 10 games: 9-1
seed 2201 / 40 games: 30-8-2
total score rate: 77.50%
decisive win rate: 78.95%
end reasons:
  Resign: 36
  MaxPliesAdjudication: 2
  RepetitionDraw: 2
paired starts:
  new sweeps: 10
  baseline sweeps: 0
  splits: 8
  draw/mixed pairs: 2
```

追加ベンチ:

```text
seed 2401 / 40 games
NewWin: 33
BaselineWin: 7
Draw: 0
new decisive win rate: 82.50%
new total score rate: 82.50%
95% CI decisive: 68.05%..91.25%
95% CI total: 70.72%..94.28%
end reasons:
  Resign: 37
  MaxPliesAdjudication: 3

new as black: 17-3
new as white: 16-4

20 start pairs:
  new sweeps: 14
  baseline sweeps: 1
  splits: 5
  draw/mixed: 0
```

この結果は、現行HEADが `v2.1.0` baseline に対して明確に優勢であることを補強しています。ただし、まだ100局以上の標準ベンチは未完了です。次はこれを最優先で実施してください。

## baseline比較の正しいやり方

`new` と `baseline` に同じ `target/release/usi_engine` を指定してはいけません。それではコード改善比較になりません。

正しい方法:

1. 現行HEADをビルド。
2. `v2.1.0` を別worktreeでビルド。
3. 両方に同じ `policy_weights_v2.1.0.binary` を読ませる。
4. `usi_benchmark` で対戦。

例:

```bash
# 現行HEAD
cd /home/kuro/shogi_rust
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin usi_engine --bin usi_benchmark --bin record_analyze

# v2.1.0 baseline
git worktree add /tmp/shogi_rust_v210_bench v2.1.0
cd /tmp/shogi_rust_v210_bench
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin usi_engine
```

ベンチ例:

```bash
cd /home/kuro/shogi_rust

rm -rf /tmp/shogi_bench_records_2401

env RUST_FONTCONFIG_DLOPEN=1 target/release/usi_benchmark \
  --new-engine /home/kuro/shogi_rust/target/release/usi_engine \
  --baseline-engine /tmp/shogi_rust_v210_bench/target/release/usi_engine \
  --new-weights /home/kuro/shogi_rust/policy_weights_v2.1.0.binary \
  --baseline-weights /home/kuro/shogi_rust/policy_weights_v2.1.0.binary \
  --positions /home/kuro/shogi_rust/taya36.sfen \
  --games 100 \
  --depth 5 \
  --time-limit-ms 100 \
  --max-plies 200 \
  --adjudicate-at-max-plies \
  --jobs 4 \
  --seed 2401 \
  --record-dir /tmp/shogi_bench_records_2401
```

その後:

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/record_analyze \
  --weights /home/kuro/shogi_rust/policy_weights_v2.1.0.binary \
  --record-dir /tmp/shogi_bench_records_2401 \
  --tail-plies 12
```

## サブエージェント運用ルール

ユーザーはサブエージェント活用を明示的に希望しています。

### 安価モデルに任せる作業

実験・検証・ログ集計・ベンチ実行など、単純で時間がかかる作業。

推奨モデル:

```text
gpt-5.4-mini
```

任せる例:

- 40局/100局ベンチ
- seed違いの再現確認
- `record_analyze` 出力の集計
- プロファイル実行
- 採用済み変更の回帰テスト

### GPT-5.5 xhighに任せる作業

重要な方針転換、高度な実験結果分析、アルゴリズム設計判断。

指定:

```text
model: gpt-5.5
reasoning_effort: xhigh
```

任せる例:

- 100局ベンチ結果の統計的解釈
- 敗局パターンから次の探索改善を選ぶ判断
- NNUE移行設計
- 自己対局学習再開の是非
- 大きな探索アルゴリズム変更の採否判断

## 今後の優先順位

### P0: 100局ベンチの確定

まず現行HEADが `v2.1.0` baseline に対して本当に安定して強いか確認します。

採用目安:

```text
100局で total score rate >= 70%
baseline sweeps が少ない
敗局理由に明確な実装バグがない
```

### P1: 敗局分類

`record_analyze` は終盤評価推移を出せるようになっています。

見るべき点:

- `BaselineWin` の最終評価
- `tail_scores`
- `worst_drop`
- 終局理由
- new_as black/white の偏り
- start_sfen ペアの sweep/split

分類軸:

```text
詰み逃し
王手回避ミス
捕獲順序ミス
駒損
千日手
入玉・長手数
序盤不利
評価関数の楽観/悲観
```

### P2: 次の探索改善

優先候補:

1. rootでの即詰み探索
2. 王手回避局面の指し手順序改善
3. SEEの精密化
4. 終盤の王手・詰めろ関連拡張
5. 序盤ブック

採用条件:

```text
最低20局で悪化しない
できれば40局で改善傾向
最終採用は100局または複数seedで確認
```

### P3: 学習の再開

自己対局学習は過去に壊れたため、いきなり大規模に回さないでください。

安全な方針:

- baseline重みを必ず保存
- 学習後重みは別名保存
- 10局だけで採用しない
- baselineと現行固定版の両方に対戦
- 悪化した重みは破棄

優先する学習方式:

```text
探索蒸留 > 教師あり再学習 > 小規模自己対局 > 大規模自己対局
```

### P4: 小型NNUEプロトタイプ

長期的な本命です。いきなり強豪級NNUEを作らないでください。

段階:

1. 既存KPP特徴を入力にした小型MLPを設計。
2. Rustでfloat推論を実装。
3. 既存SparseModelと切り替え可能にする。
4. Python/PyTorchで学習。
5. 効果が見えたら量子化・差分Accumulatorへ進む。

## これまでに棄却した実験

以下は再試行しないでください。必要なら条件を変えて明確な仮説を立ててください。

```text
record_finetune weights:
  短い10局では良く見えたが20局で不採用。

qsearch all check evasions:
  悪化。

qsearch capped check evasions:
  悪化。

static eval ordering for evasion moves:
  悪化。

TT repetition-sensitive probe/store:
  seed 1401で14-6、従来16-4より悪化かつ遅い。

finite mate score:
  seed 1401で13-7、悪化。

qsearch ordering with SEE/check bonus:
  速度は少し良いが20局15-5、現行16-4未満。

recapture ordering:
  10局8-2、20局14-6で不採用。

quiet-history/killer-only quiet updates:
  10局9-1、20局15-5で不採用。

draw contempt:
  10局6-4まで悪化。

internal PVS
LMR
static eval cache
score-only full search
qsearch score-only split
check extension
古いSEE bonus ordering
qsearch drop-check removal
qsearch MVV-only
qsearch pre-SEE
TT capacity tweaks
identity hasher
TT depth-preferred replacement
root-only check bonus
split score_move branch
```

## 重要レポート

以下を必ず読んでください。

```text
report/2026-06-19_07-57-58_JST_shogi-ai-improvement_search-ordering_usi-report.md
report/research_2026-06-19_08-19-45_JST_shogi-ai-algorithms.md
```

前者はこれまでの実装・実験履歴です。  
後者は将棋AIアルゴリズム調査と今後の方針です。

## コーディング方針

- 既存設計に寄せる。
- 大きな探索変更は小さく実装し、必ずベンチで確認。
- dirty worktreeではユーザーの変更を戻さない。
- 手動編集は `apply_patch` を使う。
- `rg` を優先して調査する。
- 変更は適度にコミットする。
- ログや重みファイルはgit管理しない。

## 現在のgitignore上の注意

`.binary`、`log_*.txt`、`*.png` は無視されます。

そのため、実験重みやログは必要なら明示的にパスを報告してください。重要な結果は `report/` にmarkdownで保存してコミットしてください。

## 次にやるべきこと

最初の作業はこれです。

1. `policy_weights_v2.1.0.binary` が存在するか確認。
2. 現行HEADとv2.1.0 worktreeをビルド。
3. 100局ベンチを安価サブエージェントに依頼。
4. 完了後、`record_analyze` で敗局を解析。
5. 結果が単純なら次の小改善へ進む。
6. 解釈が難しい場合は GPT-5.5 xhigh サブエージェントに分析依頼。

最初の安価サブエージェントへの依頼例:

```text
作業ディレクトリは /home/kuro/shogi_rust です。
コード編集・コミットはしないでください。
現行HEADの usi_engine と v2.1.0 worktree の usi_engine を使って、v2.1.0 baseline比較を100局実行してください。
weights は両方 /home/kuro/shogi_rust/policy_weights_v2.1.0.binary を使ってください。
seed は 2401、positions は taya36.sfen、depth 5、time-limit-ms 100、max-plies 200、adjudicate-at-max-plies、record-dir は /tmp/shogi_bench_records_2401。
完了後、勝敗、勝率、信頼区間、終局理由、先後偏り、paired starts、record_analyze の要約を報告してください。
```

重要分析が必要な場合のGPT-5.5 xhigh依頼例:

```text
以下の100局ベンチ結果とrecord_analyze出力を分析してください。
目的は、次に実装すべき探索・評価改善を1つか2つに絞ることです。
統計的に有意な傾向、敗局の共通原因、過去に棄却済みの実験との重複を考慮してください。
大きな方針転換が必要か、現行AlphaBeta+KPP路線を続けるべきかも判断してください。
```

