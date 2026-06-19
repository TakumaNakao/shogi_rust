# 精密SEE/捕獲救済実験メモ

- 作成日時: 2026-06-19 13:22:42 UTC
- 対象ブランチ: `exp/precise-see`
- 結論: 棄却。通常の再帰SEEも軽量な捕獲後被攻撃判定も、現状の実装形では速度劣化が大きい。

## 背景

`see_probe` で、旧SEEが負と判定する捕獲の中に、再帰SEEでは非負になる例が確認できた。

代表的な事前プローブ:

```text
/tmp/v241_taildrop_positions.sfen
positions: 10
qsearch candidates: 118
capture candidates: 44
see sign flips: 21
old negative -> new nonnegative: 19
old nonnegative -> new negative: 1

/tmp/root_rescue_strong3.sfen
positions: 3
qsearch candidates: 27
capture candidates: 13
see sign flips: 4
old negative -> new nonnegative: 3
old nonnegative -> new negative: 0
```

このため、qsearchで捨てている捕獲を一部救済できる可能性があると判断した。

## 試した内容

1. 全量再帰SEE
   - `see()` を通常捕獲の再帰交換評価に差し替え。
   - 探索順序付けとqsearch skipの両方に効く。

2. 旧SEE負候補のみ再帰確認
   - 探索順序付けは旧SEEのまま。
   - qsearchで旧SEEが負の捕獲だけ再帰SEEで再確認。

3. 捕獲後被攻撃判定による軽量救済
   - `Position::is_attacked_by()` を追加。
   - 旧SEEが負の捕獲について、捕獲後に移動先が相手から攻撃されていなければ救済。
   - clone版とdo/undo版を試した。

## 速度結果

比較条件:

```bash
target/release/search_profile \
  --weights policy_weights_v2.1.0.binary \
  --positions taya36.sfen \
  --samples 36 \
  --depth 5 \
  --seed 9501
```

master基準:

```text
elapsed ms: 約28104
total nodes: 8511421
```

実験結果:

```text
全量再帰SEE:
elapsed ms: 62017.27
total nodes: 10442316

旧SEE負候補のみ再帰確認:
elapsed ms: 57564.76
total nodes: 11183229

捕獲後被攻撃判定 clone版:
elapsed ms: 41710.31
total nodes: 10900169

捕獲後被攻撃判定 do/undo版:
elapsed ms: 38263.17
total nodes: 10900169
```

最も軽いdo/undo版でも、master比で約36%遅い。事前ゲートの「速度悪化は概ね8%以内」を大きく超えた。

## 判断

この方向は、現状のqsearch構造にそのまま入れるには高コストすぎる。救済によって探索ノードも増えるため、局面あたりの計算時間が大きく増えた。

今後この系統を再試行する場合は、以下のどちらかに限定する。

- `Position` 内部のbitboardを使った本格SEEを、clone/do/undoなしで実装する。
- qsearch候補生成そのものを絞り、SEE救済の呼び出し回数を大幅に減らす。

単純な再帰合法手生成SEE、またはqsearch負候補ごとのdo/undo救済は再試行しない。
