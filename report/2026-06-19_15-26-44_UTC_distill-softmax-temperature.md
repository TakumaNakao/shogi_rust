# 探索蒸留 softmax温度 実験レポート

- 日時: 2026-06-19 15:26:44 UTC
- ブランチ: `tooling/distill-temperature`
- 目的: policy蒸留/教師あり学習で、softmax温度を固定600から調整可能にする。

## 背景

`depth6 / 300局面` の探索蒸留では、`learning_rate=0.02/0.05`、softmax温度600固定だとoffline指標がほとんど動かなかった。

これは重み更新の勾配が弱すぎる可能性がある。プロ棋譜・強AI棋譜を使う場合でも、温度を調整できないと学習速度の比較が難しい。

## 変更内容

- `SparseModel::update_batch_with_cross_entropy_temperature(batch, temperature)` を追加。
- 既存 `update_batch_with_cross_entropy(batch)` は互換性維持のため温度600で呼ぶ。
- `distill_train` に `--softmax-temperature` を追加。
- offline評価のCE計算にも同じ温度を使う。

## depth6 / 300局面での比較

データ:

```text
train records: 270
valid records: 30
teacher depth: 6
seed: 9601
```

温度600、`learning_rate=0.05`, `epochs=5`:

```text
baseline train ce=3.751236 top1=0.1148
baseline valid ce=3.807892 top1=0.3333
epoch 5 train_ce=3.751229 train_top1=0.1222 valid_ce=3.807889 valid_top1=0.3333
```

温度200、`learning_rate=0.05`, `epochs=5`:

```text
baseline train ce=3.739866 top1=0.1148
baseline valid ce=3.740978 top1=0.3333
epoch 5 train_ce=3.739806 train_top1=0.1481 valid_ce=3.740945 valid_top1=0.3333
```

温度100、`learning_rate=0.05`, `epochs=5`:

```text
baseline train ce=3.732837 top1=0.1148
baseline valid ce=3.658641 top1=0.3333
epoch 5 train_ce=3.732606 train_top1=0.1778 valid_ce=3.658508 valid_top1=0.3333
```

温度を下げると学習信号は明確に強くなる。valid CEはわずかに改善したが、valid top1はまだ改善していない。

## candidate profile

候補:

```text
/tmp/shogi_distill_d6_300/policy_weights_d6_300_lr005_t100_e5.binary
```

profile:

```text
samples: 36
total nodes: 8380231
quiescence nodes: 7556069
quiescence moves considered: 3985593
quiescence moves searched: 1494395
quiescence see skips: 1426887
check evasion extensions: 11532
elapsed ms: 27570.36
```

baseline系の代表値は total nodes 8,638,168、qnodes 7,802,474、elapsed 約28秒なので、profile上は破綻していない。

## 判断

温度引数化は採用する。小規模探索蒸留候補そのものは、valid top1が改善していないため採用候補とはまだ言えない。ただしprofile破綻はないため、20局smokeで悪化幅を確認する価値はある。

プロ棋譜・強AI棋譜を使う場合も、まず温度100/200/600をofflineで比較し、valid CEとtop1の両方を見る。
