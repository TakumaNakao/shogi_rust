# soft探索蒸留 1000局面 smoke 結果

- 作成日時: 2026-06-19 18:53:19 UTC
- ブランチ: `exp/soft-distill-weight-smoke`
- 目的: 追加したsoft探索蒸留基盤で、小規模候補重みが現行固定重みに勝てる兆候を持つか確認する。

## 実験条件

教師局面:

```text
converted_records2016_10818.sfen から seed 11001 で 1000局面
train: 900
valid: 100
```

教師生成:

```bash
target/release/distill_dump \
  --weights policy_weights_v2.1.0.binary \
  --input converted_records2016_10818.sfen \
  --train-output /tmp/shogi_soft_distill_1000_d3_top4/train.jsonl \
  --valid-output /tmp/shogi_soft_distill_1000_d3_top4/valid.jsonl \
  --depth 3 \
  --teacher-score-top 4 \
  --teacher-score-depth 3 \
  --max-positions 1000 \
  --valid-percent 10 \
  --seed 11001 \
  --jobs 4
```

複数候補を試したが、低学習率候補はCEがほぼ動かなかった。対局に回した候補は以下。

```bash
target/release/distill_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/shogi_soft_distill_1000_d3_top4/train.jsonl \
  --valid /tmp/shogi_soft_distill_1000_d3_top4/valid.jsonl \
  --output /tmp/shogi_soft_distill_1000_d3_top4/candidate_lr1_st200_tt200_e5.binary \
  --epochs 5 \
  --batch-size 64 \
  --learning-rate 1.0 \
  --softmax-temperature 200 \
  --teacher-temperature 200
```

offline:

```text
baseline train samples=900 ce=3.622036 top1=0.0900
baseline valid samples=100 ce=3.675939 top1=0.0700
epoch 5 train_ce=3.621083 train_top1=0.1422 valid_ce=3.675125 valid_top1=0.1600
```

valid top1は上がったが、CE改善は小さい。

## 固定セット確認

候補重みで `position_probe` を実行した。

`loss_in_check_low_reply.sfen`:

```text
total: 27
in_check: 27
low_legal_in_check: 21
terminal: 4
search_win: 0
search_loss: 26
legal_without_bestmove: 0
bestmove_gives_check: 0
bestmove_limits_reply: 0
```

`taildrop_root_rescue.sfen`:

```text
total: 12
in_check: 1
low_legal_in_check: 1
terminal: 0
search_win: 3
search_loss: 1
legal_without_bestmove: 0
bestmove_gives_check: 6
bestmove_limits_reply: 4
```

固定セットのsummaryは基準重みと同等で、ここでは致命的な悪化は見えなかった。

## 対局結果

比較条件:

```text
engine: 現行masterの同一 usi_engine
new weights: candidate_lr1_st200_tt200_e5.binary
baseline weights: policy_weights_v2.1.0.binary
positions: taya36.sfen
depth: 5
time-limit-ms: 100
max-plies: 200
adjudicate-at-max-plies
jobs: 4
```

20局 smoke / seed 12001:

```text
new wins: 11
baseline wins: 9
draws: 0
new total score rate: 55.00%
```

40局 gate / seed 12021:

```text
new wins: 18
baseline wins: 21
draws: 1
new decisive win rate: 46.15%
new total score rate: 46.25%
95% CI total: 30.99%..61.51%
end reasons:
  RepetitionDraw: 1
  Resign: 39
paired starts:
  new sweeps: 4
  baseline sweeps: 6
  splits: 9
  draw/mixed pairs: 1
```

## 判断

不採用。

20局の 11-9 はノイズだった。40局では総合46.25%に落ち、paired startsでも baseline sweeps が new sweeps を上回った。

この結果から、1000局面・depth3・top4のsoft探索蒸留では、現行固定重みを超える兆候は弱い。今後soft蒸留を続けるなら、以下の条件を変える必要がある。

- 1000局面ではなく、より多い局面を使う。
- depth3教師では浅すぎる可能性があるため、少数局面でもdepth4以上を試す。
- `converted_records2016` だけでなく、現行エンジンの敗局・互角局面を混ぜる。
- 先にoffline validationを強化し、CEだけでなく候補手score gapやtop-k agreementを見る。

今回生成した候補重みは不採用のため削除対象。
