# 将棋AI 研究開発報告

作成日時: 2026-06-19 07:57:58 JST  
対象ブランチ: `improve-self-play-learning`  
目的: `v2.1.0 baseline` を上回る将棋AIに向けた自己対局学習、ベンチマーク、探索・USI改良の反復

## 概要

本日の開発では、`v2.1.0` の重みを基準にしながら、最新コード側の探索・USI実装・検証基盤を改良した。現在の主要な採用済み改善は、探索時の手順ordering強化とUSI/ベンチ分析基盤の整備である。

現行HEADは `v2.1.0 baseline` に対して明確に優勢で、代表的なUSIベンチでは以下の結果を得た。

- 20局 seed 1401: `16-4`
- 10局 seed 1701: `9-1`
- 40局 seed 2201: `30-8-2`
- 40局 total score rate: `77.50%`
- 40局 decisive win rate: `78.95%`

## 背景分析

過去バージョン調査により、`v2.1.0` は教師あり学習による `KPP + material` 評価のバランスが良く、以降の自己対局学習では駒得係数の勾配計算不備や過大な学習率により重みが崩れた可能性が高いと判断した。

特に自己対局学習では、material係数の勾配が本来 `error * material` であるべきところ、単に `error` として扱われていた。このため、自己対局学習を進めると評価関数の駒得バランスが壊れるリスクがあった。

## 採用済みの主な改良

### 1. 自己対局学習ロジックの正常化

`kpp_self_learn` で material 勾配を局面の駒得値込みで計算するよう修正した。

目的:
- 自己対局学習で material 係数が誤った方向に更新される問題を防ぐ
- 今後の学習実験の前提を正常化する

ただし、現時点では新しい自己対局重みの採用には至っていない。短期ベンチで良く見える候補も、20局以上では現行探索改善版より弱くなる傾向があった。

### 2. USI stop 対応

USIの `stop` で探索停止フラグを立て、探索中でも `bestmove` を返せるようにした。

確認:
- 手動USI確認で `go` 後の `stop` に対し `bestmove` が返ることを確認
- その後のUSIベンチでも破綻なし

効果:
- 実GUI、対局サーバ、外部ベンチ環境での信頼性が向上
- 長時間探索や時間切れ時の不安定挙動を減らす

### 3. 連続王手千日手の扱い修正

通常の千日手と、連続王手による千日手負けを区別した。連続王手千日手では、直前に王手をかけていた側を負けとして扱う。

採用理由:
- 将棋ルール上、連続王手千日手は通常引き分けではない
- 探索上も、王手の繰り返しを安全な引き分けとして誤評価しないようにする必要がある

確認:
- 既存テストに連続王手千日手判定を追加
- `cargo test --all-targets` 成功
- USIベンチで悪化なし

### 4. USIベンチと棋譜記録の整備

`usi_benchmark` に棋譜保存機能を追加し、対局ごとの結果を `.usi` として保存できるようにした。

さらに、保存棋譜に終局理由を記録するようにした。

終局理由:
- `Resign`
- `IllegalMove`
- `RepetitionDraw`
- `PerpetualCheckLoss`
- `MaxPliesAdjudication`
- `MaxPliesDraw`

効果:
- 勝敗だけでなく、なぜ終局したかを分析できる
- 千日手、最大手数裁定、投了負けを分離できる
- 改良候補の原因分析がしやすくなった

### 5. `record_analyze` の強化

保存したUSI棋譜を解析する `record_analyze` を追加・拡張した。

主な出力:
- 勝敗数
- 平均最終評価
- 終局理由の集計
- 評価符号と実勝敗の不一致数
- 先後入れ替えペア単位の集計

ペア集計:
- new sweeps
- baseline sweeps
- splits
- draw/mixed pairs

これにより、「特定初期局面で両色とも負けるのか」「先後で割れるだけなのか」を把握できるようになった。

### 6. 王手手の探索ordering優先

主探索とroot探索の手順orderingで、王手になる手に固定ボーナスを加えた。

目的:
- 終盤の王手連打、攻め合いでの読み負けを減らす
- qsearchの探索幅を増やさず、手順順序だけ改善する

結果:
- seed 1401 / 20局: `15-5`
- seed 1701 / 10局: `8-2`

小さいが正方向の改善として採用した。

### 7. SEEによる戦術的捕獲手のordering改善

主探索/root探索の手順スコアに簡易SEE補正を追加した。

内容:
- MVV-LVAだけでは、大駒で安い駒を取るような損な捕獲も前に出やすい
- `victim_value - attacker_value` を用いて捕獲手の順序を補正
- 探索する手の数は増やさず、順序だけ改善

結果:
- seed 1401 / 20局: `16-4`
- seed 1701 / 10局: `9-1`

王手ordering単独より良好だったため採用した。

## 代表的なベンチ結果

### 現行HEAD vs v2.1.0 baseline

条件:
- USI engine 同士の直接対局
- baseline engine: `/tmp/shogi_rust_v210_worktree/target/release/usi_engine`
- new weights: `policy_weights_v2.1.0.binary`
- baseline weights: `policy_weights_v2.1.0.binary`
- depth: 5
- time limit: 100ms
- max plies: 160
- max plies adjudication enabled

結果:

| 条件 | 結果 | 得点率 |
| --- | ---: | ---: |
| seed 1401 / 20局 | `16-4-0` | 80.0% |
| seed 1701 / 10局 | `9-1-0` | 90.0% |
| seed 2201 / 40局 | `30-8-2` | 77.5% |

40局の詳細:
- `NewWin`: 30
- `BaselineWin`: 8
- `Draw`: 2
- decisive win rate: 78.95%
- total score rate: 77.50%
- end reasons: `Resign 36`, `MaxPliesAdjudication 2`, `RepetitionDraw 2`
- paired starts: `new sweeps 10`, `baseline sweeps 0`, `splits 8`, `draw/mixed pairs 2`

この結果から、現行HEADは `v2.1.0 baseline` をかなり安定して上回っていると判断できる。

## 試したが不採用にした実験

### record_finetune による重み微調整

保存済みUSI棋譜から、終盤局面を使って重みを微調整した。

試した候補:
- `tail_plies=16`, `epochs=1`, `lr=0.0001`, `freeze_material=true`
- `tail_plies=32`, `epochs=1`, `lr=0.00005`, `freeze_material=true`
- `tail_plies=16`, `epochs=2`, `lr=0.00005`, `freeze_material=true`
- より保守的な `lr=0.00001` 系

短期10局では強く見える候補もあったが、20局以上では現行探索改善版より弱くなった。

結論:
- 現状の棋譜fine-tuneは過学習気味
- 重み更新より探索改善の方が安定して強くなっている

### qsearch 全王手拡張

静止探索で全王手を広げる案を試した。

結果:
- 悪化したため不採用

理由:
- 探索幅が増え、限られた100ms条件では重要手を読み切れなくなる可能性が高い

### qsearch ordering 改善

静止探索内の候補手を、主探索と同様にSEEや王手ボーナスで並べる案を試した。

結果:
- profileではわずかに速度改善
- しかしUSI 20局では `15-5` となり、現行HEADの `16-4` を下回った

結論:
- 探索速度だけでは採用不可
- 実勝率基準で戻した

### 有限mate score

詰みや連続王手千日手を `f32::INFINITY` ではなく有限のmate scoreに置き換え、詰み距離を反映する案を試した。

結果:
- 20局で `13-7` まで悪化

結論:
- 理論上は妥当な面があるが、現行探索との相性が悪い
- 不採用

### TT履歴依存回避

千日手履歴に依存する局面でTTを使わない案を試した。

結果:
- 勝率悪化とオーバーヘッドが大きかった

結論:
- 現行条件では不採用

### recapture ordering

直前に取られた地点を取り返す捕獲手にボーナスを与える案を試した。

結果:
- 10局では `8-2`
- 20局では `14-6`

結論:
- 短期では良く見えたが20局で悪化
- 不採用

### quiet-history 限定

killer/history更新を静かな手だけに限定する案を試した。

結果:
- 10局では `9-1`
- 20局では `15-5`

結論:
- 現行HEADの `16-4` を下回ったため不採用

### draw contempt

通常千日手を少し嫌うように、通常引き分けスコアに小さな補正を与えた。

背景:
- 40局記録で `RepetitionDraw` が2局あり、どちらも新エンジン視点で評価が悪くなかった

結果:
- 10局スモークで `6-4` まで悪化

結論:
- 方針としてはあり得るが、現行の単純補正は通常勝率を削る
- 不採用

## 現在作業中のUSI時間管理対応

現在、未コミットで `src/usi_shogi.rs` と `src/ai.rs` にUSI時間管理関連の変更がある。

目的:
- `go depth`
- `go movetime`
- `go btime`
- `go wtime`
- `go byoyomi`
- `go infinite`
- `stop`

を解釈し、実GUIや対局サーバで正しく時間制御できるようにする。

確認済み:
- `go movetime 100` は `bestmove` を返す
- `go infinite` 後の `stop` でも `bestmove` を返す
- `go depth 1 movetime 100` も、USI info出力を無効化することで `bestmove` を返すようになった
- `cargo test --all-targets` 成功

注意:
- サブエージェントによる短いUSIベンチでは、未コミットUSI変更込みで `6-4` とやや弱い結果が出た
- これは10局のみで揺れが大きいが、採用前に追加確認が必要
- 実装面では、USI経由の探索時に `set_emit_info(false)` を入れている

## 現在の状態

採用済み最新コミット:

- `8c5c23a Use SEE to order tactical captures`
- `d8e461c Prioritize checking moves in search ordering`
- `c576d37 Summarize paired benchmark starts`
- `85aff64 Summarize end reasons in record analysis`
- `2a35afd Record USI benchmark end reasons`
- `503a81c Respect USI stop during search`
- `640b0b2 Handle checked repetition as perpetual check loss`

未コミット状態:
- `Cargo.lock` に既存の変更あり
- 複数の未追跡 `log_*.txt`
- 現在作業中の `src/ai.rs`, `src/usi_shogi.rs` 変更あり

`Cargo.lock` と未追跡ログは既存の汚れとして扱い、今回の作業では触っていない。

## 今後の方針案

### 方針A: USI時間管理を完成させる

優先度: 高

理由:
- 実戦環境では `go` 引数を正しく解釈できないと、探索性能以前に時間制御で不利になる
- GUIや対局サーバで使えるAIにするには必須

次にやること:
- 未コミットUSI変更の差分整理
- `go depth`, `go movetime`, `go btime/wtime/byoyomi`, `go infinite/stop` の手動確認
- 20局程度のUSIベンチで強さ回帰がないか確認
- 問題なければコミット

### 方針B: 中盤のBaselineWin 8局を詳細分析する

優先度: 高

理由:
- 40局で負けた8局はすべて `Resign`
- 手数は `76, 78, 84, 85, 87, 107, 110, 137`
- 短中手数の戦術崩れが主因

次にやること:
- BaselineWin 8局の中盤テーマを分類
- 飛角交換、駒損、玉薄、端攻め、打ち込み筋などに分ける
- 共通テーマが見つかれば、評価またはorderingに小さく反映

### 方針C: 自己対局学習を再開する前に、学習データ設計を見直す

優先度: 中

理由:
- record_finetuneは短期で過学習しやすかった
- 自己対局学習も、浅い探索結果をそのまま教師にすると崩れる可能性がある

次にやること:
- 負け棋譜の終盤だけでなく、中盤の分岐点を抽出
- 勝敗ラベルだけでなく、探索評価差やPV変化も使う
- material係数は当面freezeする
- 20局未満の短期ベンチで採用判断しない

### 方針D: 探索高速化

優先度: 中

理由:
- profileでは静止探索ノードが全体の約90%を占める
- 100ms条件では速度改善がそのまま読める深さに効く

ただし注意:
- qsearch候補を増やす変更は過去に悪化
- qsearch orderingも実勝率では悪化
- 速度だけでなくベンチ勝率で採否判断する必要がある

## 推奨

直近は以下の順で進めるのがよい。

1. 未コミットのUSI時間管理対応を完成させる
2. 40局のBaselineWin 8局を詳細分析する
3. 共通する中盤テーマがあれば小さな評価・ordering改善を試す
4. 学習は、過学習対策と採否ベンチ基準を決めてから再開する

現時点では、探索ordering改善により `v2.1.0 baseline` を明確に上回る状態には到達している。一方で、さらに強くするには、短期ベンチで良く見える変更を採用せず、40局以上の確認と負け筋分析に基づいて進めるべきである。
