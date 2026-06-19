# CSA policy dump 品質フィルタ追加とwdoor高レート勝者データ確認

- 作成日時: 2026-06-19 16:45 UTC
- ブランチ: `tooling/csa-policy-quality-filters`
- 目的: wdoor/floodgate棋譜から「トップAIらしい手」をより安全に抽出するため、CSA教師データ変換にレート・勝者フィルタを追加する。

## 実装

`csa_policy_dump` に以下のオプションを追加した。

```text
--min-player-rate <N>
  その手を指した側のwdoorレートがN以上の手だけを教師化する。

--winner-only
  勝者側の手だけを教師化する。

--decisive-only
  勝敗推定できる棋譜だけを使う。
```

既存のデフォルト挙動は変えない。オプション未指定なら従来通り全手を教師化する。

実装方針:

- wdoor CSAコメントの `'black_rate:` / `'white_rate:` からレートを抽出。
- CSA終局アクションから勝者を推定。
- `%TORYO` / `%TIME_UP` / `%ILLEGAL_MOVE` は直前に指した側を勝者とする。
- `%+ILLEGAL_ACTION` / `%-ILLEGAL_ACTION` は違反した側の反対を勝者とする。
- 千日手・持将棋・引き分け・中断などは勝者なしとして扱う。

## データ生成確認

条件:

```bash
target/release/csa_policy_dump \
  --input /tmp/shogi_external_kifu/wdoor/extract/2026 \
  --train-output /tmp/shogi_external_kifu/datasets_quality/wdoor2026_policy_train_20k_r4000_winner.jsonl \
  --valid-output /tmp/shogi_external_kifu/datasets_quality/wdoor2026_policy_valid_20k_r4000_winner.jsonl \
  --seed 20260619 \
  --valid-percent 10 \
  --max-records 20000 \
  --min-ply 8 \
  --max-ply 160 \
  --min-player-rate 4000 \
  --winner-only \
  --decisive-only
```

結果:

```text
games used: 332
games skipped: 7
games filtered: 4733
records filtered: 449598
train records: 18000
valid records: 2000
```

SHA256:

```text
0af583841780da13ccc7e3902936725920e2fd0b772357dffc1dbe9c4033a0be  wdoor2026_policy_train_20k_r4000_winner.jsonl
aa7080a0cdcc0bae86dd17adc609f72346e11c30a8598f3f8af50416d7b17ea2  wdoor2026_policy_valid_20k_r4000_winner.jsonl
```

## 既存重みの基準値

```text
baseline train samples=18000 ce=3.997822 top1=0.2172
baseline valid samples=2000 ce=4.008378 top1=0.2095
```

通常20kデータよりvalid top1が低く、教師として難しいデータになっている。

## 小規模学習

条件:

```bash
target/release/distill_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/shogi_external_kifu/datasets_quality/wdoor2026_policy_train_20k_r4000_winner.jsonl \
  --valid /tmp/shogi_external_kifu/datasets_quality/wdoor2026_policy_valid_20k_r4000_winner.jsonl \
  --output /tmp/shogi_wdoor_quality_20k_lr005/policy_weights_wdoor2026_20k_r4000_winner_lr005_t100_e3.binary \
  --epochs 3 \
  --batch-size 512 \
  --learning-rate 0.05 \
  --softmax-temperature 100
```

結果:

```text
baseline train samples=18000 ce=3.997822 top1=0.2172
baseline valid samples=2000 ce=4.008378 top1=0.2095
epoch 1 train_ce=3.997797 train_top1=0.2208 valid_ce=4.008359 valid_top1=0.2105
epoch 2 train_ce=3.997773 train_top1=0.2234 valid_ce=4.008338 valid_top1=0.2115
epoch 3 train_ce=3.997749 train_top1=0.2233 valid_ce=4.008315 valid_top1=0.2115
```

候補重み:

```text
/tmp/shogi_wdoor_quality_20k_lr005/policy_weights_wdoor2026_20k_r4000_winner_lr005_t100_e3.binary
sha256: c4819958d56a229241a2c0d3a5b45238400f922cb9648f45078f20e5ee6ca42c
```

通常wdoor 20k学習ではvalid top1が悪化したが、この品質フィルタ版ではvalid top1が `0.2095 -> 0.2115` に改善した。

## 20局スモーク

条件:

```text
new weights: /tmp/shogi_wdoor_quality_20k_lr005/policy_weights_wdoor2026_20k_r4000_winner_lr005_t100_e3.binary
baseline weights: policy_weights_v2.1.0.binary
engine: same current usi_engine
positions: taya36.sfen
games: 20
depth: 5
time-limit-ms: 100
seed: 9811
record-dir: /tmp/shogi_bench_wdoor_quality20k_9811
```

結果:

```text
new wins: 9
baseline wins: 9
draws: 2
new decisive win rate: 50.00%
new total score rate: 50.00%
95% CI total: 29.21%..70.79%

end reasons:
  RepetitionDraw: 2
  Resign: 18

paired starts:
  new sweeps: 2
  baseline sweeps: 2
  splits: 5
  draw/mixed pairs: 1

record_analyze:
  non-terminal score/result sign mismatches: 3
```

## 判断

- ツール改善として採用する。
- この20k品質フィルタ重みは採用しない。
- 通常wdoor 20kよりオフライン指標は良い方向に動いたため、次に試すなら200k品質フィルタ版を作り、offline gateを先に見る。
- 重み採用判断は引き続き最低40局、できれば100局または複数seedで行う。

次の候補:

```text
--min-player-rate 4000 --winner-only --decisive-only で200k版を生成
1 epochだけ学習
valid CE/top1が十分改善する場合だけ20局スモークへ進む
```
