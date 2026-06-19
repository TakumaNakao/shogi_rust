# ArrayVec探索手順ソート高速化の棄却

- 作成日時: 2026-06-19 22:16:52 UTC
- ブランチ: `perf/arrayvec-search-ordering`
- 結論: 採用しない。ノード数は一致したが、探索速度が悪化した。

## 目的

探索内の手順ソートで発生している `Vec` 確保と詰め替えを減らし、100ms条件で探索できる局面数を増やす。

試した変更:

- `quiescence_search` の `Vec<(Move, i32)>` を `ArrayVec<(Move, i32), 593>` へ変更。
- full search/root search の `Vec<(Move, i32)>` + `Vec<Move>` を `ArrayVec` へ変更。
- ソートキー、killer move、TT move、探索順の意味は変えない。

## 検証

変更前のprofile:

```text
samples: 72
total nodes: 19067196
quiescence nodes: 17270535
elapsed ms: 64146.45
nodes/sec: 297244.76
quiescence node rate: 90.58%
```

変更後のprofile:

```text
samples: 72
total nodes: 19067196
quiescence nodes: 17270535
elapsed ms: 68273.57
nodes/sec: 279276.38
quiescence node rate: 90.58%
```

ノード数とqsearch統計は一致しており、挙動は実質的に同じだった。一方でelapsedは約6.4%悪化した。

## 判断

採用しない。

推定理由:

- `ArrayVec` の最大合法手容量を再帰探索中にスタックへ確保するため、qsearchの深い再帰ではかえって重い。
- qsearchが全ノードの約90.6%を占めるため、小さなスタック/コピーコスト悪化が全体に強く出る。

## 次の候補

探索高速化は、単純なコンテナ置換ではなく、以下を優先する。

1. 手順リストを巨大固定容量で持たない形の確保削減。
2. qsearchでの追加枝削減は過去棄却が多いため慎重に扱う。
3. GPT-5.5 xhighの第二候補である `CHECK_EVASION_EXTENSION_MAX_REPLIES = 4` の単独実験へ進む。
