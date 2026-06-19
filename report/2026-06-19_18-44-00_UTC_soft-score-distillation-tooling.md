# 探索スコア分布蒸留基盤の追加

- 作成日時: 2026-06-19 18:44:00 UTC
- ブランチ: `exp/soft-score-distillation`
- 目的: hard bestmove 蒸留だけでなく、探索スコア分布を教師にした小規模な評価関数更新実験を可能にする。

## 背景

現行masterは `v2.4.1` に対して seed 10201 / 40局で 28-11-1、総合 71.25% だった。敗局tailを `root_rescue_probe` で確認したところ、tail 12手 / depth 5 では `strong root-rescuable candidates: 0` だった。

このため、root救済型の追加探索よりも、以下を優先する判断にした。

1. 探索木不変の基礎高速化
2. 探索スコア分布を使う安全な評価関数更新基盤

## 棄却した高速化実験

`Position::do_move` に、事前計算済みの王手判定を渡す `do_move_with_check_hint` を試した。root/full search と qsearch の一部で `is_check_move` の二重計算を削る狙いだった。

結果:

- `cargo test --all-targets`: 成功
- `loss_in_check_low_reply.sfen`: summary不変
- `taildrop_root_rescue.sfen`: summary不変
- `taya36.sfen` / 72 samples / depth 5: total nodes、qnodes、terminal mates、check evasion extensions は完全一致
- elapsed中央値は改善せず、むしろ不利な反復が出た

判断:

探索木は不変だったが、速度ゲートを通らなかったため不採用。変更は戻した。

## 追加した基盤

`distill_dump` に、任意指定時だけ候補手ごとの探索スコアをJSONLへ出力する機能を追加した。

主なオプション:

```bash
--teacher-score-top N
--teacher-score-depth D
```

`teacher_score_top > 0` のとき、教師bestmoveを必ず含めた候補手集合に対して探索スコアを保存する。

JSONL例:

```json
{
  "sfen": "...",
  "teacher_move": "8h9g",
  "depth": 3,
  "legal_moves": 87,
  "teacher_scores": [
    {"move_usi":"8h9g","score":0.15084249},
    {"move_usi":"2f2c+","score":-261.76096}
  ]
}
```

`distill_train` は後方互換を維持し、`teacher_scores` がない既存JSONLは従来通りhard labelとして扱う。`teacher_scores` がある場合は、`--teacher-temperature` でsoft target分布を作り、legal move全体のmodel softmaxに対するcross entropyで更新する。

## smoke test

実行した確認:

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin distill_dump --bin distill_train
```

hard形式:

```bash
target/release/distill_dump \
  --weights policy_weights_v2.1.0.binary \
  --input taya36.sfen \
  --train-output /tmp/shogi_soft_distill_smoke/hard_train.jsonl \
  --valid-output /tmp/shogi_soft_distill_smoke/hard_valid.jsonl \
  --depth 3 \
  --max-positions 20 \
  --valid-percent 20 \
  --jobs 2

target/release/distill_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/shogi_soft_distill_smoke/hard_train.jsonl \
  --valid /tmp/shogi_soft_distill_smoke/hard_valid.jsonl \
  --output /tmp/shogi_soft_distill_smoke/hard_candidate.binary \
  --epochs 1 \
  --batch-size 8 \
  --learning-rate 0.001 \
  --dry-run
```

soft形式:

```bash
target/release/distill_dump \
  --weights policy_weights_v2.1.0.binary \
  --input taya36.sfen \
  --train-output /tmp/shogi_soft_distill_smoke_soft/soft_train.jsonl \
  --valid-output /tmp/shogi_soft_distill_smoke_soft/soft_valid.jsonl \
  --depth 3 \
  --teacher-score-top 4 \
  --teacher-score-depth 3 \
  --max-positions 20 \
  --valid-percent 20 \
  --jobs 2

target/release/distill_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/shogi_soft_distill_smoke_soft/soft_train.jsonl \
  --valid /tmp/shogi_soft_distill_smoke_soft/soft_valid.jsonl \
  --output /tmp/shogi_soft_distill_smoke_soft/soft_candidate.binary \
  --epochs 1 \
  --batch-size 8 \
  --learning-rate 0.001
```

結果:

- hard dump: train 16 / valid 4
- hard dry-run: baseline train CE 4.160075, valid CE 4.071232
- soft dump: train 16 / valid 4
- soft train 1 epoch: baseline train CE 4.154179, valid CE 4.048867
- soft candidate weightは `/tmp/shogi_soft_distill_smoke_soft/soft_candidate.binary` にのみ保存
- 基準重み `policy_weights_v2.1.0.binary` は未変更

## 次の検証ゲート

この基盤は採用済み重みを作ったわけではない。次に進める場合は、以下の順で確認する。

1. 小規模soft dumpを数百から数千局面に増やす。
2. valid CE と teacher top1 agreement がhard蒸留より悪化しないか確認する。
3. 候補重みは必ず別名保存する。
4. 現行固定重みと20局 smoke。
5. 悪化しなければ40局、さらに100局または複数seedへ進む。

採用条件:

- 20局で45%未満なら即破棄
- 40局で55%未満なら原則不採用
- 100局で55%以上かつ固定セットが悪化しない場合のみ採用候補

自己対局学習の失敗履歴があるため、この基盤でも短い10局だけでは絶対に採用しない。
