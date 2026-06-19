# qsearch direct generation order diff

- 作成日時: 2026-06-20 01:20:00 UTC
- ブランチ: `probe/qsearch-check-drop-diff`
- 目的: qsearch候補を直接生成する高速化案で、候補集合が同じでも探索木が変わる原因を調べる。

## 実験内容

以前棄却した `legal_quiescence_moves_with_generated_count()` の直接生成案を一時的に再適用した。

- 非王手局面では、全盤上手と王手になる駒打ちだけを生成。
- その後 `is_legal` で違法手を除外。
- 最後に捕獲手または王手手だけを残す。

`qsearch_candidate_probe` には、`--ordered` で集合は同じだが順序だけが違う場合に、最初の差分位置と前後の候補列を表示する診断出力を追加した。

## 結果

### taya36 全手順展開

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/qsearch_candidate_probe \
  --input taya36.sfen \
  --all-plies \
  --ordered \
  --max-positions 5000 \
  --show 5
```

```text
positions: 5000
mismatches: 4
ordered: true
all plies: true
reference candidates: 9089
current candidates: 9089
```

代表差分:

```text
first_order_diff_index=2
reference_window=8b8h,8b8h+,B*5i,B*7g,B*7i,B*8f,B*9e
current_window=8b8h,8b8h+,B*9e,B*5i,B*7g,B*7i,B*8f
```

### drop-window局面

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/qsearch_candidate_probe \
  --input /tmp/shogi_value_drop/drop_windows.sfen \
  --ordered \
  --show 5
```

```text
positions: 193
mismatches: 45
ordered: true
all plies: false
reference candidates: 1252
current candidates: 1252
```

代表差分:

```text
first_order_diff_index=7
reference_window=4c3b,4c3c,4c4d,G*1b,G*1c,G*3b,B*1c,B*3a
current_window=4c3b,4c3c,4c4d,B*3a,G*1b,G*1c,G*3b,B*1c
```

## 原因

候補集合の missing/extra は空なので、直接生成案は候補集合自体は一致している。

しかし旧実装は以下の順序で処理している。

1. 全疑似合法手を生成する。
2. `is_legal` で違法手を `swap_remove` する。
3. qsearch用に捕獲手・王手手だけを `retain` する。

`swap_remove` は、qsearchでは最終的に捨てられる静かな手や非王手駒打ちの存在にも依存して、残る候補手の順序を変える。直接生成案では最初から静かな手や非王手駒打ちを生成しないため、候補集合が一致しても最終順序が一致しない。

qsearchは候補順序により alpha 更新、SEE skip、枝刈りの発生順が変わるため、探索木と評価結果が変わる。

## 判断

この直接生成案は現時点では採用しない。

順序完全互換を保つには、旧実装の `swap_remove` 副作用まで再現する必要がある。そこまで行うと、全疑似合法手生成を避けるという高速化メリットが小さくなる可能性が高い。

次に試すなら、以下のどちらかに分ける。

- 探索木完全一致を諦め、強さベンチで採否する直接生成案。
- qsearch内部の独立した候補順序を設計し、順序変更込みの探索改善として検証する案。

現段階では、単純な qsearch直接生成は高速化単体の安全な変更として扱わない。
