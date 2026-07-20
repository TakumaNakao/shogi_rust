# Artifact and repository policy

大規模な棋譜、教師data、checkpoint、weight、対局結果をsource treeから分離し、入力と
生成物の取り違えを防ぐための規約である。

| path | 内容 | Git | 再現性に必要な情報 |
|---|---|---|---|
| `tests/fixtures/` | 小さなgolden fixture | 追跡 | schema、意図、固定hash |
| `benchmarks/baselines/` | 承認済み非退行baseline | 追跡 | revision、環境、binary/input hash |
| `data/raw/` | 取得した原棋譜等 | 非追跡 | source URLまたは由来、content hash |
| `data/derived/` | dataset、teacher shard | 非追跡 | producer manifest、parent hash |
| `runs/` | checkpoint、log、候補model、対局 | 非追跡 | run manifest、全引数、seed、親artifact |
| `report/` | 人がレビューした結論 | 追跡 | status、revision、input、結論、後継 |

既存の`data/wdoor/`等は互換pathとして維持する。新規pipelineは上記layoutを優先し、
一括移動で既存scriptを壊さない。大きな生成物をGitへ追加せず、必要なfixtureだけを
`tests/fixtures/`へ明示的に追加する。

## Manifest

再利用可能な生成物は少なくともschema version、producer、source revision、完全な引数、
seed、入力content hash、出力content hashを持つ。pathやmtimeだけを同一性判定に使わない。
Phase 6のdataset/teacher/trainerはこの規約に従い、fingerprint不一致時はstale artifactを
再利用しない。

## Ignore rule

`.gitignore`はdirectory単位を基本とする。`*.binary`のような拡張子ruleは既存weightの
誤commit防止として残すが、追跡するgolden fixtureは`git add -f`で意図を明示する。
新規の可変出力は`data/raw/`、`data/derived/`、`runs/`の外へ置かない。
