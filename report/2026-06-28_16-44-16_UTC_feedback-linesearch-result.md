# Feedback-only delta line-search 結果

- 作成日時: 2026-06-28 16:44:16 UTC
- ブランチ: `training/strong-weight-learning-infra`
- 実験ディレクトリ: `data/mmto/runs/tree_feedback_linesearch_20260628_162722`

## 目的

tree feedback の feedback-only 学習で得られた方向が「正しいが小さすぎる」のかを確認するため、baseline -> candidate の差分を alpha 1, 2, 4, 8, 16 に拡大して gate した。

## データ

source dump:

```text
data/mmto/runs/mmto_rerank_long_20260624_140151
```

feedback抽出:

```text
input lines: 18513
feedback samples: 750
train samples: 562
guard samples: 188
filtered: 17763
```

feedback-only 学習:

```text
best_epoch: 8
best feedback loss: 103.267410
final feedback loss: 103.267410
final violation: 0.5000
```

## line-search

`adjust_weights` の `--blend-ratio` は 1, 2, 4, 8, 16 を受理し、alpha > 1 の外挿に使えた。

| alpha | score p95 cp | score max cp | score mean cp | score gate | rerank mean/p90/p95 | bad50/bad100 | match | rerank |
|---:|---:|---:|---:|---|---|---|---:|---|
| 1 | 0.022642 | 0.032103 | 0.006403 | PASS | 8.363155 / 29.151466 / 43.52255 | 0.0380 / 0.0030 | 49.0% | FAIL |
| 2 | 0.045284 | 0.064207 | 0.012807 | PASS | not run | not run | - | not run |
| 4 | 0.090568 | 0.128414 | 0.025614 | PASS | not run | not run | - | not run |
| 8 | 0.181144 | 0.256830 | 0.051227 | PASS | not run | not run | - | not run |
| 16 | 0.362289 | 0.513662 | 0.102455 | PASS | not run | not run | - | not run |

alpha=1 の rerank gate は、平均regretが微小悪化したため失敗した。

```text
mean regret worsened: 8.363155 > baseline + 0
```

20局ベンチは、rerank gate を通る alpha がなかったため未実施。

## 判断

今回のfeedback抽出方向は「良いが小さい」ではなく、少なくとも rerank のroot選択には安定して移らない方向だった。

これ以上同じ tree-feedback-only を長時間回しても、強い重みへ進む見込みは低い。

## 次の方針

1. root feedback の単純拡大はいったん停止する。
2. PV sibling などで信号密度を増やす場合も、通常 tree/listwise loss として流すのではなく、hard feedback / rerank gate 前提で使う。
3. 次の実装候補は、PV sibling dump から「teacher/PV側の良手 vs student/current側の悪手」を直接 feedback JSON に変換する経路。
4. 生成件数、held-out violation、rerank非悪化を通らない限り対局ベンチへ進めない。

## 容量管理

実験中に alpha 重みを複数生成して空き容量が約1GBまで低下した。

最終的に `RUN_DIR` 内の `.binary` は全削除済み。
