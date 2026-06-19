# qsearch終端詰み検出 実験結果

- 作成日時: 2026-06-19 16:25 UTC
- ブランチ: `exp/qsearch-terminal-mate`
- 目的: qsearch中に王手で詰まされた局面へ到達した場合、静的評価ではなく終端負けとして扱う修正が有効か検証する。

## 仮説

現行qsearchは `stand_pat` を評価してから合法手を生成する。そのため、qsearch中に「手番側が王手されていて合法手ゼロ」の局面へ到達した場合でも、詰み負けとして `-inf` を返さず、静的評価で返す可能性がある。

これは探索拡張ではなく終端判定の整合性修正なので、過去に棄却した `qsearch all check evasions` や `check extension` とは別の小実験として試した。

## 実装内容

一時実装として以下のみを変更した。

- `quiescence_search` 冒頭で `position.in_check()` のときだけ `legal_moves()` を先に生成。
- 合法手ゼロなら `Some((-f32::INFINITY, Vec::new()))` を返す。
- 合法手がある場合は、その手リストを後続の既存qsearch候補フィルタに使い回す。
- qsearchの候補範囲、SEE skip、指し手順序、通常探索、評価重みは変更しない。
- profile確認用に `quiescence_terminal_mates` カウンタを追加。

## 検証

テスト:

```text
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

結果:

```text
passed
```

profile条件:

```text
weights: policy_weights_v2.1.0.binary
positions: taya36.sfen
samples: 36
depth: 5
seed: 9501
```

旧master:

```text
samples: 36
total nodes: 8638168
quiescence nodes: 7802474
quiescence moves considered: 4271670
quiescence moves searched: 1625940
quiescence see skips: 1475088
check evasion extensions: 11652
elapsed ms: 28048.33
nodes/sec: 307974.41
avg nodes/sample: 239949.11
quiescence node rate: 90.33%
quiescence moves/node: 0.55
quiescence see skip rate: 34.53%
```

qsearch終端詰み検出:

```text
samples: 36
total nodes: 8627638
quiescence nodes: 7792404
quiescence moves considered: 4256963
quiescence moves searched: 1620576
quiescence see skips: 1470364
quiescence terminal mates: 380
check evasion extensions: 11671
elapsed ms: 30590.93
nodes/sec: 282032.51
avg nodes/sample: 239656.61
quiescence node rate: 90.32%
quiescence moves/node: 0.55
quiescence see skip rate: 34.54%
```

速度差:

```text
elapsed: 28048.33ms -> 30590.93ms
悪化: +9.07%
```

## 判断

事前ゲートでは、`search_profile depth5 samples36 seed9501` でelapsed悪化が `+8%超` なら20局スモークへ進めず棄却とした。

今回の悪化は `+9.07%` でゲート超過のため、v2.4.1相手の20局スモークは実施しない。

結論:

```text
qsearch終端詰み検出は現形では棄却。
コード変更は戻し、報告書のみ残す。
```

## 補足

`quiescence terminal mates: 380` と発火は確認できたため、終端詰みを見落としている局面は実際にある。ただし、王手中qsearchノードで毎回 `legal_moves()` を先行生成するコストが大きく、現行の浅い100msベンチ条件では速度悪化が勝る可能性が高い。

将来再試行するなら、以下のような「合法手ゼロだけを安価に判定する専用API」を先に作る必要がある。

```text
Position::has_legal_evasion()
Position::legal_moves_up_to(1) for in-check positions only
```

ただし、過去に `legal-move-count-up-to` は単体で速度改善が出なかったため、再試行する場合は「qsearch終端判定専用」という明確な仮説で行う。
