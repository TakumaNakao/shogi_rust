# wdoor 2026 棋譜収集と教師データ化メモ

- 作成日時: 2026-06-19 15:40 UTC
- ブランチ: `data/wdoor-2026-kifu-collection`
- 目的: プロ棋士・トップAI棋譜学習に進む前段として、公開条件を確認できるトップAI対局データを収集し、既存の教師あり学習パイプラインに流せるか確認する。

## 入手元

- wdoor / floodgate 公式ページ: https://wdoor.c.u-tokyo.ac.jp/shogi/
- 使用アーカイブ: https://wdoor.c.u-tokyo.ac.jp/shogi/archive/wdoor2026.7z
- ページ上では年別アーカイブとして `wdoor2026.7z` から `wdoor2008.7z` までが配布されている。

無断スクレイピングが必要なサイトではなく、公式にアーカイブとして公開されているデータを対象にした。

## ローカル保存先

大容量データはgit管理外として `/tmp` に置いた。

```text
/tmp/shogi_external_kifu/wdoor/archive/wdoor2026.7z
/tmp/shogi_external_kifu/wdoor/extract/2026
/tmp/shogi_external_kifu/datasets/wdoor2026_policy_train_20k.jsonl
/tmp/shogi_external_kifu/datasets/wdoor2026_policy_valid_20k.jsonl
/tmp/shogi_external_kifu/datasets/wdoor2026_policy_train_200k.jsonl
/tmp/shogi_external_kifu/datasets/wdoor2026_policy_valid_200k.jsonl
```

展開には `p7zip-full` を使用した。Dockerの未使用ビルドキャッシュと未使用イメージを削除し、空き容量を約18GBまで回復させてから展開した。

## 取得結果

```text
wdoor2026.7z:
  size: 138M
  sha256: d7d7108af3a0f280be8685bf0acd6759f756bb8ddaa8fa7266889ebea02b46bf

展開後:
  CSA files: 60485
  extracted size: 946M
```

サンプルファイル名には以下のようなトップAI・強豪エンジン名が含まれていた。

```text
AobaZero_w4708_n_p800
Suisho5_750_473stb_1000k
62KIN_Suisho11_1c
gikou2_1c
tanuki_wcsc33_473stb_1000k
```

## JSONL教師データ化

既存の `csa_policy_dump` を使い、CSA棋譜を `distill_train` と同じJSONL形式に変換した。

20k版:

```bash
target/release/csa_policy_dump \
  --input /tmp/shogi_external_kifu/wdoor/extract/2026 \
  --train-output /tmp/shogi_external_kifu/datasets/wdoor2026_policy_train_20k.jsonl \
  --valid-output /tmp/shogi_external_kifu/datasets/wdoor2026_policy_valid_20k.jsonl \
  --seed 20260619 \
  --valid-percent 10 \
  --max-records 20000 \
  --min-ply 8 \
  --max-ply 160
```

結果:

```text
games used: 197
games skipped: 4
train records: 18000
valid records: 2000
```

200k版:

```bash
target/release/csa_policy_dump \
  --input /tmp/shogi_external_kifu/wdoor/extract/2026 \
  --train-output /tmp/shogi_external_kifu/datasets/wdoor2026_policy_train_200k.jsonl \
  --valid-output /tmp/shogi_external_kifu/datasets/wdoor2026_policy_valid_200k.jsonl \
  --seed 20260619 \
  --valid-percent 10 \
  --max-records 200000 \
  --min-ply 8 \
  --max-ply 160
```

結果:

```text
games used: 1987
games skipped: 56
train records: 180000
valid records: 20000
```

SHA256:

```text
0701525017b2939eb11cd0b549027dc6c0a1cfbdba1d172402a107d2e593db24  wdoor2026_policy_train_200k.jsonl
e5bb15ea5d2a96b3f8a48254cdb7f9e699fedbf1d49c9d0a9cf9a68692e29c0e  wdoor2026_policy_valid_200k.jsonl
4f9663d9bb6f187976f8da6ba866248ebdc34d857c36f64f1a047bdb076df353  wdoor2026_policy_train_20k.jsonl
6ee2ef50abc58642b14354e8d45fe956cfd94cf90e6cb5ac58d2c9038e1cb3f0  wdoor2026_policy_valid_20k.jsonl
```

## 既存重みの基準値

`policy_weights_v2.1.0.binary` を使い、200k版に対する教師手一致率を `distill_train --dry-run` で測定した。

```text
baseline train samples=180000 ce=3.859106 top1=0.2224
baseline valid samples=20000 ce=3.880646 top1=0.2204
```

作業中に、`dry-run` が基準値を出力した後も学習ループへ進む挙動を確認した。これは同じブランチで修正し、基準値出力後に即終了するようにした。

修正後の20k版確認:

```text
baseline train samples=18000 ce=3.857033 top1=0.2191
baseline valid samples=2000 ce=3.894128 top1=0.2185
real 0m8.249s
```

## 判断

- wdoor 2026だけで、既存の教師あり学習パイプラインに直接入れられるCSA対局を6万局以上確保できた。
- 20万局面の教師データでもJSONLは約21MBで、初期実験に十分扱いやすい。
- 現行重みのトップAI棋譜手一致率はvalid top1で約22%なので、教師あり微調整で改善余地はある。
- ただし、強さに直結するかは別問題であり、学習後重みは必ず固定版エンジンとの対戦ゲートで判定する。

## 次の候補

1. 20k版で小さな教師あり学習を行い、CE/top1が正常に改善するか確認する。
2. 改善する場合、200k版で1 epochだけ学習し、旧重みとの20局スモークを行う。
3. 採用判断は最低でも40局、できれば100局または複数seedで行う。
