# TT best move delayed scoring 棄却

- 作成日時: 2026-06-27 17:46:31 UTC
- ブランチ: `perf/tt-bestmove-delayed-scoring`
- 結論: 採用しない。短い固定深さ profile で探索カウンタが一致しなかった。

## 目的

非rootの `alpha_beta_search_internal` では、TT best move が存在する場合でも、現行実装は全合法手を採点・sortしてからTT手を先頭へ移動している。

一時実験では、TT best move が合法手に含まれる場合にその1手を先に探索し、そこで cutoff した場合だけ残り手の採点・sortを省くことで、探索木を変えずに `search_ordering_score()` と `Position::is_check_move()` の呼び出しを減らすことを狙った。

## 事前懸念

TT手を先に探索すると、その子探索で history / killer が更新される。

cutoffしなかった場合、残り手の採点・killer並べ替えが「子探索後のhistory/killer」を使うため、元実装とは残り手orderingが変わる可能性がある。

このため、固定深さ profile の探索カウンタ完全一致を最初のゲートにした。

## 検証

実験は `GPT-5.3-codex-spark` サブエージェントに委任した。

比較対象:

- base: `origin/training/strong-weight-learning-infra`
- new: base + TT best move delayed scoring

短い確認条件:

```text
search_profile
positions: taya36.sfen
samples: 16
depth: 5
seed: 9501
```

結果:

```text
COUNTER_MATCH=NG

total nodes:
  new:  4147495
  base: 4159559

quiescence nodes:
  new:  3706969
  base: 3721395

quiescence moves considered:
  new:  2192316
  base: 2213311

quiescence moves generated:
  new:  32858623
  base: 32986052

quiescence moves discarded:
  new:  30666307
  base: 30772741

quiescence moves searched:
  new:  812833
  base: 824672

quiescence see skips:
  new:  798436
  base: 803229

quiescence terminal mates:
  new:  84
  base: 225

check evasion extensions:
  new:  4049
  base: 4156
```

指定ゲートに従い、長めのprofileや対局ベンチには進めなかった。

## 判断

この案は、単なる挙動不変の高速化ではなかった。

原因は事前懸念どおり、TT手を先に探索することでhistory/killer更新タイミングが変わり、cutoffしなかったノードの残り手順序が変化した可能性が高い。

TT周辺は過去にも replacement / capacity / hasher / repetition-sensitive probe-store が不採用になっている。今回の案も探索木を変えずに高速化する形にはできていないため採用しない。

## 次の方針

- TT手delayed scoringを再試行するなら、history/killerのスナップショットが必要になるが、コピーコストが高く本来の高速化目的と矛盾しやすい。
- TT周辺の小最適化は当面優先度を下げる。
- 次は、探索木不変のprofile改善ではなく、敗局分類に基づく小さな探索品質改善、または計測基盤の強化を優先する。
