# root前回depth全候補スコア順序 実験レポート

- 日時: 2026-06-19 14:46:57 UTC
- ブランチ: `exp/root-previous-score-ordering`
- 比較対象: `v2.4.3`
- 重み: `policy_weights_v2.1.0.binary`
- 結論: 不採用

## 仮説

root反復深化では、前回depthの最善手だけを次depth先頭に移動している。

完走したdepthのroot全候補スコアを保存し、次depth開始前にスコア降順へ並べ替えれば、rootのPVSが効きやすくなり、短時間条件で有効手を早く読めると考えた。

## 実装内容

- rootの各depthで、探索完了した各root手の `current_eval` を `depth_root_scores` に保存。
- aspiration failでfull re-searchした場合は、最初のスコアを破棄し、re-search後のスコアだけを次depth用に採用。
- depth完走後、次depth用の `sorted_moves` を前回スコア降順に並べ替え。

触っていないもの:

- 内部alpha-beta
- qsearch
- 境界王手応手延長
- aspiration幅
- TT保存規則
- 評価関数

## profile結果

コマンド:

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/search_profile \
  --weights policy_weights_v2.1.0.binary \
  --positions taya36.sfen \
  --samples 36 \
  --depth 5 \
  --seed 9501
```

`v2.4.3` baseline 代表値:

```text
total nodes: 8638168
quiescence nodes: 7802474
check evasion extensions: 11652
elapsed ms: 約28000
```

実験版:

```text
samples: 36
total nodes: 9213768
quiescence nodes: 8270799
quiescence moves considered: 4660882
quiescence moves searched: 1747405
quiescence see skips: 1669823
check evasion extensions: 11712
elapsed ms: 31612.27
nodes/sec: 291461.79
```

total nodes は約6.66%増加し、elapsedも大きく悪化した。

## 判断

速度/ノードゲートを満たさないため不採用。20局ベンチには進めない。

原因として、root PVSのnull-window探索で返ったfail-lowの境界値を「全候補の正確なスコア」として扱ったことで、次depthのroot順序がむしろ悪化した可能性が高い。

この方向を再試行するなら、全候補を単純にスコア降順にするのではなく、以下のような制限が必要。

- full-windowで得たスコアだけを順序更新に使う。
- fail-low手は元順序を保ち、best/fail-high手だけを前へ出す。
- rootで追加のfull-window探索を行わない。追加探索は過去のroot rescue系と重なり、コストが高い。
