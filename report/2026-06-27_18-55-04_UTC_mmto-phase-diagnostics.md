# MMTO学習のカテゴリ別診断ログ

- 作成日時: 2026-06-27 18:55:04 UTC
- 対象ブランチ: `training/strong-weight-learning-infra`
- 目的: 長時間学習の前に、序盤・中盤・終盤・王手局面・合法手少数局面のどこが改善/悪化しているかを見える化する。

## 背景

ベンチ敗局feedbackを通常lossへ混ぜる小実験では、学習中のvalid指標やfeedback violationは改善したが、探索込みのrerank gateでは悪化した。

この結果から、単純にepochや実行時間を増やすより先に、以下を検出できる基盤が必要と判断した。

- 序盤だけ改善して終盤を壊す。
- 王手局面だけ悪化する。
- 合法手が少ない強制局面だけ悪化する。
- 静的なvalid指標は改善しているのに、探索後の手選択が悪化する。

GPT-5.5 xhigh分析でも、支配的な問題は量不足ではなく、目的関数と探索後の手選択のズレ、少数hard feedbackの混ぜ方、局面構成の偏りであると判断された。

## 実装内容

`mmto_tree_dump` のJSONLに以下のメタ情報を追加した。

```text
ply
phase: opening / middle / late
in_check
```

`mmto_tree_train` は、新旧どちらのdumpでも動くようにした。

- 新dumpでは `ply`, `in_check`, `legal_moves` を読む。
- 古いdumpではSFEN/局面から同じ情報を復元する。

`mmto_tree_train` の標準出力に、train/validそれぞれのカテゴリ別指標を追加した。

カテゴリ:

```text
phase:
  opening: ply <= 40
  middle:  41 <= ply <= 90
  late:    ply > 90

in_check:
  手番側が王手されている局面

low_legal:
  legal_moves <= 3
```

各カテゴリで出す値:

```text
samples
selected_regret mean
p95
teacher_match
bad50
bad100
```

## 使い方

通常の `mmto_tree_train` 実行で追加ログが出る。

例:

```text
baseline valid buckets phase=[opening: samples=..., mean=..., p95=..., match=..., bad50=..., bad100=...; ...] in_check=[...] low_legal=[...]
epoch 1 valid buckets phase=[opening: ...] in_check=[...] low_legal=[...]
```

## 判断

この変更は学習更新式や評価関数には影響しない。次の長時間学習候補は、カテゴリ別ログで以下を満たすものだけに絞る。

- valid全体が非悪化。
- opening/middle/late のいずれかが大きく悪化しない。
- in_check と low_legal が悪化しない。
- そのうえで探索込みrerank gateを通す。

この診断を入れずに長時間学習を回すと、局所的な悪化に気づかないまま時間を消費する可能性が高い。
