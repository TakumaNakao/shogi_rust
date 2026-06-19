# 探索蒸留と教師あり棋譜学習の優先度整理

- 日時: 2026-06-19 15:23:25 UTC
- ブランチ: `tooling/csa-policy-jsonl-dump`
- 目的: 重み更新方針を「現行探索蒸留だけ」に寄せすぎず、プロ棋譜・強AI棋譜を使える形へ広げる。

## 結論

今の重みだけを教師にする学習は効果が薄い。探索蒸留は「現行重みそのもの」ではなく「現行重み + v2.4.3探索」を教師にするため一定の意味はあるが、同じ評価関数に依存する以上、評価関数の偏りを固定化するリスクがある。

したがって今後の重み更新は以下の順で扱う。

1. プロ棋譜・強AI棋譜など外部の良質な指し手ラベルを優先して教師あり学習する。
2. その後、現行探索で探索蒸留し、エンジンの探索癖に合わせて微調整する。
3. 自己対局学習は、上記2つの検証ゲートが整ってから小規模に限定して扱う。

## 現状のデータ確認

`converted_records2016_10818.sfen` は局面のみで、正解指し手ラベルは含まれていない。

```text
ln1g1k1nl/1r3sg2/p2psp1pp/2p3p2/1p2b4/2PP3P1/PPSG1PP1P/1BG2S1R1/LN1K3NL b Pp 25
...
```

既存の `kpp_learn` は `csa_files/<year>/*.csa` から指し手教師あり学習を行う設計だが、このPCのリポジトリにはCSA本体は存在しなかった。

## depth6探索蒸留の小規模結果

teacher:

- code: `v2.4.3` 相当
- weights: `policy_weights_v2.1.0.binary`
- SHA256: `8d2ad6ebd65afd9bdd921f7c03205582421f00cbe2c83ccbda984fbbe13747b3`
- input: `converted_records2016_10818.sfen`, `taya36.sfen`
- depth: 6
- max positions: 300
- jobs: 4

dump結果:

```text
train records: 270
valid records: 30
skipped positions: 0
```

`learning_rate=0.02`, `epochs=5`, `batch_size=64`:

```text
baseline train samples=270 ce=3.751236 top1=0.1148
baseline valid samples=30 ce=3.807892 top1=0.3333
epoch 5 train_ce=3.751234 train_top1=0.1185 valid_ce=3.807891 valid_top1=0.3333
```

`learning_rate=0.05`, `epochs=5`, `batch_size=64`:

```text
baseline train samples=270 ce=3.751236 top1=0.1148
baseline valid samples=30 ce=3.807892 top1=0.3333
epoch 5 train_ce=3.751229 train_top1=0.1222 valid_ce=3.807889 valid_top1=0.3333
```

offline指標はほとんど動かず、valid top1も改善しなかった。小規模データなので決定的ではないが、探索蒸留だけを小さく回してもすぐ強い候補は出にくい。

## 追加したツール

### `csa_policy_dump`

CSA棋譜を `distill_train` と同じJSONL形式へ変換する。

使用例:

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/csa_policy_dump \
  --input /path/to/csa_games \
  --train-output data/policy/pro_train.jsonl \
  --valid-output data/policy/pro_valid.jsonl \
  --seed 9601 \
  --valid-percent 10 \
  --min-ply 8 \
  --max-ply 160
```

出力形式:

```json
{"sfen":"...","teacher_move":"7g7f"}
```

この形式は `distill_train` がそのまま読める。つまり、プロ棋譜・強AI棋譜・探索蒸留を同じpolicy-only学習パイプラインで比較できる。

## 今後の優先順位

1. 外部CSA棋譜が利用できる場合は、まず `csa_policy_dump` で教師ありデータを作る。
2. `distill_train` で `learning_rate=0.02/0.05`、material固定、1 epochから検証する。
3. offline valid CE/top1が改善した候補だけprofileへ進める。
4. profileで破綻しなければ、old weight vs candidate weightの20局smokeへ進める。
5. 100局で55%以上を満たさない重みは破棄する。

探索側では、qsearch候補手の直接生成がまだ残る有望候補だが、movegen本体に踏み込むため、重み学習データの入口を整えた後に扱う。
