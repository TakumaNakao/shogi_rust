# qsearch capture prepass 実験棄却レポート

- 作成日時: 2026-06-29 05:05:55 UTC
- 対象ブランチ: `training/strong-weight-learning-infra`
- 比較対象: `current` 実験差分 vs `e936dd4 Add capture move generation profile`

## 目的

`legal_capture_moves_with_generated_count()` で捕獲手だけを先に直接生成し、qsearch で beta cutoff できる場合に高コストな従来の `legal_quiescence_moves_with_generated_count()` を省略する。

狙いは、qsearch の大量の全合法手生成と破棄を減らし、探索速度を上げること。

## 実装内容

通常の非王手 qsearch 局面で、以下の二段処理を試した。

1. 直接生成した合法捕獲手だけを先に探索する。
2. beta cutoff できなかった場合は従来の qsearch 候補を生成し、すでに探索した捕獲手を除いて残りを探索する。

このため、候補生成数は減るが、従来の「捕獲手と王手をまとめて生成して同じスコアで並べる」順序とは変わる。

## 検証結果

GPT-5.3-codex-spark サブエージェントで検証した。

### search_profile

条件:

- weights: `policy_weights_v2.1.0.binary`
- depth: 5
- samples: 32
- 各条件3 seed

`taya36.sfen`:

- NPS 中央値: `+3.64%`
- qsearch generated 中央値: `-25.3%`
- qsearch discarded 中央値: `-29.2%`
- nodes / qsearch nodes / qsearch searched はほぼ同等

`converted_records2016_10818.sfen`:

- NPS 中央値: `+1.60%`
- NPS 平均: わずかに低下
- elapsed 中央値: `+5.9%` で悪化
- qsearch generated 平均/中央値: `-23.2% / -20.9%`
- qsearch discarded 平均/中央値: `-26.7% / -24.6%`
- nodes / qsearch nodes / qsearch searched はほぼ同等

### 20局ベンチ

条件:

- current(new) vs `e936dd4` baseline
- games: 20
- depth: 5
- time-limit-ms: 100
- max-plies: 200
- seed: 9707
- positions: `taya36.sfen`

結果:

- NewWin: 9
- BaselineWin: 11
- Draw: 0
- new total score rate: 45.00%
- end reasons: `Resign` 19, `MaxPliesAdjudication` 1
- paired starts: new sweeps 0, baseline sweeps 1, splits 9
- terminal result mismatches: 0
- non-terminal score/result sign mismatches: 0

## 判断

棄却する。

理由:

- qsearch の生成・破棄数削減は明確だが、全体の NPS / elapsed は局面依存で揺れ、安定した速度改善とは言えない。
- 20局ベンチは 9-11 で、破綻はないが採用を後押しする結果でもない。
- 捕獲手を先に探索することで、従来の捕獲手・王手混在の move ordering が変わる。これは強さへの影響が読みにくい。

## 今後の扱い

この形の qsearch capture prepass は再試行しない。

`legal_capture_moves_with_generated_count()` 自体は単体プロファイルで有望なため、探索順序を壊さない別用途や、capture-only の局所解析・プロファイル基盤として残す。
