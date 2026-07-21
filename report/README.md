# Experiment report index and metadata

`report/`は実験の生logではなく、人がレビューした採否判断を保存する。過去reportはfile名と
本文を維持し、新規または更新するreportは先頭に次のmetadataを置く。

```yaml
---
status: accepted | rejected | inconclusive | superseded | informational
revision: <git commit>
inputs:
  - <manifest path or SHA-256>
conclusion: <one sentence>
superseded_by: <report path or null>
---
```

検索例:

```bash
# 明示metadataを持つ不採用実験
rg -l '^status: rejected$' report

# metadata導入前の不採用実験（file名規約）
rg --files report | rg 'rejected\.md$'

# 特定の仮説や実装を過去に試したか
rg -n -i 'qsearch|transposition|halfkp' report
```

既存reportはhistorical recordとして一括書換えしない。後続作業で参照・再評価した時点で
metadataを追記し、元の測定値と結論は改変しない。重要な横断的判断は`docs/adr/`へ要約する。
