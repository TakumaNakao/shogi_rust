# wdoor教師あり重み更新スモーク結果

- 作成日時: 2026-06-19 16:05 UTC
- ブランチ: `exp/wdoor-supervised-weight-smoke`
- 目的: wdoor/floodgate 2026 公式CSA棋譜から作った教師データで、現行KPP重みを安全に更新できる見込みがあるか確認する。

## 前提

使用データ:

```text
/tmp/shogi_external_kifu/datasets/wdoor2026_policy_train_20k.jsonl   18000 samples
/tmp/shogi_external_kifu/datasets/wdoor2026_policy_valid_20k.jsonl    2000 samples
/tmp/shogi_external_kifu/datasets/wdoor2026_policy_train_200k.jsonl 180000 samples
/tmp/shogi_external_kifu/datasets/wdoor2026_policy_valid_200k.jsonl  20000 samples
```

元重み:

```text
policy_weights_v2.1.0.binary
size: 889702712 bytes
sha256: 8d2ad6ebd65afd9bdd921f7c03205582421f00cbe2c83ccbda984fbbe13747b3
```

安全条件:

- material coefficientは固定。
- 学習後重みは別名で `/tmp` に保存。
- 20局スモークだけでは採用しない。
- offline gateを通らない候補は対局ベンチへ進めない。

## 20k版学習

コマンド概要:

```bash
target/release/distill_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/shogi_external_kifu/datasets/wdoor2026_policy_train_20k.jsonl \
  --valid /tmp/shogi_external_kifu/datasets/wdoor2026_policy_valid_20k.jsonl \
  --output /tmp/shogi_wdoor_train_20k_lr005/policy_weights_wdoor2026_20k_lr005_t100_e3.binary \
  --epochs 3 \
  --batch-size 512 \
  --learning-rate 0.05 \
  --softmax-temperature 100
```

結果:

```text
baseline train samples=18000 ce=3.857033 top1=0.2191
baseline valid samples=2000 ce=3.894128 top1=0.2185

epoch 1 train_ce=3.857014 train_top1=0.2201 valid_ce=3.894115 valid_top1=0.2170
epoch 2 train_ce=3.856997 train_top1=0.2207 valid_ce=3.894097 valid_top1=0.2175
epoch 3 train_ce=3.856983 train_top1=0.2209 valid_ce=3.894080 valid_top1=0.2165
```

20局スモーク:

```text
seed: 9801
games: 20
depth: 5
time-limit-ms: 100
positions: taya36.sfen
new weights: /tmp/shogi_wdoor_train_20k_lr005/policy_weights_wdoor2026_20k_lr005_t100_e3.binary
baseline weights: policy_weights_v2.1.0.binary

new wins: 15
baseline wins: 5
draws: 0
new decisive win rate: 75.00%
95% CI: 53.13%..88.81%
paired starts:
  new sweeps: 5
  baseline sweeps: 0
  splits: 5
  draw/mixed pairs: 0
record_analyze:
  non-terminal score/result sign mismatches: 6
```

20局では良く見えるが、valid top1低下とCIの広さから採用不可。あくまで「即破壊ではなさそう」という足切り結果に留める。

候補重み:

```text
/tmp/shogi_wdoor_train_20k_lr005/policy_weights_wdoor2026_20k_lr005_t100_e3.binary
size: 889702712 bytes
sha256: bc7bd007f407e1c6722125592fe8a167e9c284a247a7dd419a8644bbbcc75282
```

## GPT-5.5 xhigh分析

GPT-5.5 xhighサブエージェントに、20k結果の解釈と次実験の選定を依頼した。

要旨:

- 20局15-5は有望だが、CIが広く採用根拠として不足。
- 20kのCE改善はほぼフラットで、valid top1低下もあり学習成否を判断しにくい。
- 自己対局型value更新へ進む根拠にはならない。
- 次は200k版1 epochだけ実施。
- 200k版がoffline gateを通らないなら、重み更新はいったん止めてqsearch直接候補生成へ戻る。

提案されたoffline gate:

```text
valid CEがbaselineより少なくとも0.001改善
valid top1がbaseline未満にならない
```

## 200k版学習

コマンド概要:

```bash
target/release/distill_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/shogi_external_kifu/datasets/wdoor2026_policy_train_200k.jsonl \
  --valid /tmp/shogi_external_kifu/datasets/wdoor2026_policy_valid_200k.jsonl \
  --output /tmp/shogi_wdoor_train_200k_lr005/policy_weights_wdoor2026_200k_lr005_t100_e1.binary \
  --epochs 1 \
  --batch-size 512 \
  --learning-rate 0.05 \
  --softmax-temperature 100
```

結果:

```text
baseline train samples=180000 ce=3.859106 top1=0.2224
baseline valid samples=20000 ce=3.880646 top1=0.2204

epoch 1 train_ce=3.858959 train_top1=0.2232 valid_ce=3.880484 valid_top1=0.2195
material_coeff=0.145648
```

差分:

```text
valid CE:   3.880646 -> 3.880484  改善 +0.000162
valid top1: 0.2204   -> 0.2195    悪化 -0.0009
```

候補重み:

```text
/tmp/shogi_wdoor_train_200k_lr005/policy_weights_wdoor2026_200k_lr005_t100_e1.binary
size: 889702712 bytes
sha256: 9e5efdc0a9ca6293f99bcfd8cff6ee33c92142ec1a4deb0f788976e30b4ec1f9
```

## 判断

200k版はoffline gateを通らなかったため、対局ベンチへ進めず棄却する。

理由:

- valid CE改善が `0.000162` と小さすぎる。
- valid top1がbaseline未満になった。
- 20k版の15-5は、浅い探索・短い持ち時間での偶然または微小な枝刈り変化の可能性を排除できない。
- `non-terminal score/result sign mismatches` もあり、評価校正の改善とは言いにくい。

## 次の方針

重み更新は一旦停止し、探索・高速化へ戻る。

次候補:

```text
qsearch直接候補生成
```

理由:

- 既存プロファイルではqsearchが総ノードの大半を占める。
- qsearchでは全合法手を生成してから捕獲・王手系だけをフィルタしている可能性が高く、候補を直接生成できればNPS改善余地がある。
- 強さに寄与するかは、まず候補集合一致テストとsearch_profileで検証する。

採用条件:

```text
1. qsearch候補集合が現行フィルタと一致すること。
2. search_profileで少なくとも8%以上の速度改善があること。
3. 20局で悪化しないこと。
4. 最終採用は40局以上、できれば複数seedで確認すること。
```

## qsearch直接候補生成の試行

上記方針に従い、`Position::quiescence_candidate_moves()` を追加し、qsearchで全合法手を生成してから候補を絞る処理を、捕獲・直接王手・王手駒打ちの候補生成へ置き換える実験を行った。

正当性確認:

```text
cargo test --manifest-path shogi_lib/Cargo.toml quiescence_candidate_moves_match_legal_filter -- --nocapture
cargo test --manifest-path shogi_lib/Cargo.toml
cargo test --all-targets
```

結果はいずれも通過した。

profile比較:

```text
old search_profile samples=72 depth=5 seed=9501:
  total nodes: 19079358
  qnodes: 17282476
  elapsed ms: 62600.79
  nodes/sec: 304778.24

old rerun:
  total nodes: 19079358
  qnodes: 17282476
  elapsed ms: 63096.49
  nodes/sec: 302383.80

new direct candidates:
  total nodes: 19078516
  qnodes: 17281634
  elapsed ms: 59126.48
  nodes/sec: 322672.95
```

速度は約5.5%から6.3%改善したが、事前採用目安の8%以上には届かなかった。

20局スモーク:

```text
seed: 9901
games: 20
new engine: qsearch direct candidates
baseline engine: previous master binary
weights: policy_weights_v2.1.0.binary for both

new wins: 8
baseline wins: 12
draws: 0
new decisive win rate: 40.00%
95% CI: 21.88%..61.34%
paired starts:
  new sweeps: 1
  baseline sweeps: 3
  splits: 6
  draw/mixed pairs: 0
record_analyze:
  non-terminal score/result sign mismatches: 4
```

判断:

- 速度改善は実在するが小さい。
- qsearch候補の順序変化により探索木が微妙に変わり、短期対局では悪化した。
- 事前ゲートの「8%以上の速度改善」「20局で悪化しない」を両方満たさない。

結論:

```text
qsearch直接候補生成は棄却。
コード変更は戻し、結果のみ報告書に残す。
```
