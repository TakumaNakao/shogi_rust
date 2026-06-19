# 探索蒸留パイプライン導入レポート

- 日時: 2026-06-19 15:05:38 UTC
- ブランチ: `tooling/search-distillation-pipeline`
- 目的: `v2.4.1` に対して95%級の勝率を目指すため、探索改善と並行して安全な重み更新を開始できる状態にする。

## 背景

`v2.4.2` の境界王手応手延長は有効だったが、その後の探索・高速化実験は不採用が続いた。

不採用になった主な実験:

- 単応手チェーン延長
- 境界王手応手ノードの合法手再利用
- 探索側ArrayVecソート
- root前回depth全候補スコア順序
- `legal_moves_count_up_to`
- 単一合法手ノードのソート省略
- precise SEE rescue
- TT extension budget guard
- 応手4手以下への拡張

一方で、評価重みはまだ `policy_weights_v2.1.0.binary` のままである。探索だけで95%へ伸ばすより、固定データによる探索蒸留を安全に試す段階に入ったと判断した。

## 方針

自己対局学習はいきなり再開しない。過去に評価関数を壊した経緯があるため、最初の重み更新は以下に限定する。

- 固定SFEN入力のみ
- teacherは現行 `v2.4.3` 探索
- policy-only
- material係数固定
- 学習後重みは必ず別名保存
- baseline重みを上書きしない
- offline指標、profile、対局ゲートを通らない重みは破棄

## 追加したツール

### `distill_dump`

固定SFENからteacher bestmoveを生成し、JSONLに保存する。

主な引数:

```bash
target/release/distill_dump \
  --weights policy_weights_v2.1.0.binary \
  --input converted_records2016_10818.sfen \
  --input taya36.sfen \
  --train-output data/distill/v243_teacher_d6_seed9601_train.jsonl \
  --valid-output data/distill/v243_teacher_d6_seed9601_valid.jsonl \
  --depth 6 \
  --seed 9601 \
  --valid-percent 10
```

JSONL形式:

```json
{"sfen":"...","teacher_move":"7g7f","depth":6,"legal_moves":30}
```

### `distill_train`

`distill_dump` のJSONLを読み、KPP child-position scoreがteacher bestmoveを選ぶようにcross entropyで更新する。

主な引数:

```bash
target/release/distill_train \
  --weights policy_weights_v2.1.0.binary \
  --train data/distill/v243_teacher_d6_seed9601_train.jsonl \
  --valid data/distill/v243_teacher_d6_seed9601_valid.jsonl \
  --output policy_weights_v2.4.4_distill_policy_d6_seed9601_lr002_freezemat.binary \
  --epochs 1 \
  --batch-size 256 \
  --learning-rate 0.02
```

`freeze_material` はデフォルトで有効。material係数は更新前の値へ戻す。

## smoke test

コマンド:

```bash
rm -rf /tmp/shogi_distill_smoke
mkdir -p /tmp/shogi_distill_smoke

env RUST_FONTCONFIG_DLOPEN=1 target/release/distill_dump \
  --weights policy_weights_v2.1.0.binary \
  --input taya36.sfen \
  --train-output /tmp/shogi_distill_smoke/train.jsonl \
  --valid-output /tmp/shogi_distill_smoke/valid.jsonl \
  --depth 1 \
  --seed 9601 \
  --valid-percent 20 \
  --max-positions 20

env RUST_FONTCONFIG_DLOPEN=1 target/release/distill_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/shogi_distill_smoke/train.jsonl \
  --valid /tmp/shogi_distill_smoke/valid.jsonl \
  --output /tmp/shogi_distill_smoke/candidate.binary \
  --epochs 1 \
  --batch-size 8 \
  --learning-rate 0.02
```

結果:

```text
train records: 16
valid records: 4
skipped positions: 0
baseline train samples=16 ce=4.156351 top1=0.2500
baseline valid samples=4 ce=4.058296 top1=0.0000
epoch 1 train_ce=4.156350 train_top1=0.2500 valid_ce=4.058296 valid_top1=0.0000 material_coeff=0.145648
saved /tmp/shogi_distill_smoke/candidate.binary
```

出力重みサイズは元重みと同じ849MBだった。

## 次のゲート

本格実験では以下を満たすまで採用しない。

1. offline:
   - valid CEがbaselineより改善
   - valid top1 agreementが改善
   - material係数が変化しない
   - NaN/infなし
   - ファイルサイズがbaselineと一致
2. profile:
   - `search_profile --weights <candidate> --samples 36 --depth 5 --seed 9501`
   - elapsed +10%超、qnodes大幅増なら破棄
3. 対局:
   - `v2.4.3` code old weight vs candidate weight 20局で45%未満なら破棄
   - 100局で55%未満なら破棄
   - `v2.4.1` baselineへの100局で既存 `v2.4.2` 水準を明確に下回るなら破棄

## 今後

次は `depth 6` teacherで小規模データを生成し、`learning_rate=0.02` と `0.05` を比較する。10局や20局だけで採用しない。採用候補になった場合のみ100局以上へ進める。

探索側の残候補としては、qsearchで全合法手を生成せず `captures + quiet checks + drop checks` を直接生成する大きめの高速化が残る。ただしmovegen本体へ踏み込むため、重み更新の初回ゲート後に扱う。
