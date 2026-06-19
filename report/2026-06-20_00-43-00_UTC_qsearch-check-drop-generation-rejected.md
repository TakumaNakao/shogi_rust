# qsearch check-drop 直接生成 棄却

- 日時: 2026-06-20 00:43 UTC
- ブランチ: `perf/qsearch-check-drop-generation`
- 目的: qsearch の `legal_moves() -> capture/check retain` で大量に捨てている非王手dropを生成しないことで高速化できるか確認する。

## 試した内容

`legal_quiescence_moves_with_generated_count()` の内部で、非王手局面だけ以下の生成に変える案を試した。

- 盤上駒の手は従来どおり全生成
- 駒打ちは `checkable(piece_kind, to)` が真になる王手dropだけ生成
- 生成後に `is_legal` と既存の capture/check filter を通す

この方針なら、盤上手の成り・開き王手判定は従来の `is_check_move` に任せつつ、非王手dropの大量生成だけを避けられる想定だった。

## 検証

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin search_profile
```

結果: 通過。

ただし `search_profile` の探索木が変わった。

```text
master/API足場:
total nodes: 19067196
quiescence nodes: 17270535
quiescence moves considered: 8841983
quiescence moves generated: 138833345
quiescence moves discarded: 129991362
quiescence moves searched: 3264998
quiescence see skips: 3096541
quiescence terminal mates: 424
check evasion extensions: 26959

check-drop直接生成案:
total nodes: 19067948
quiescence nodes: 17271287
quiescence moves considered: 8843199
quiescence moves generated: 92009169
quiescence moves discarded: 83165970
quiescence moves searched: 3265750
quiescence see skips: 3096549
quiescence terminal mates: 460
check evasion extensions: 26959
```

生成数は大きく減ったが、`total nodes`、`qnodes`、`moves considered/searched`、`terminal mates` が変わった。

## 判断

不採用。これは速度改善ではなく探索変更になっている。

原因候補は、drop生成を先に絞ったことで、合法性判定や候補順序の微妙な差が qsearch の探索木に波及したこと。qsearch は既に過去の小変更で悪化しやすい領域なので、nodes/counters 完全一致を満たさない案は採用しない。

次に進むなら、まず qsearch 局面ごとの候補集合・順序を比較する専用probeを作り、差分が出る最初の局面を特定してから再実装する。
