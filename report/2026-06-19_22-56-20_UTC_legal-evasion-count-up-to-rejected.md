# legal_evasion_count_up_to による王手応手延長判定高速化の棄却

- 作成日時: 2026-06-19 22:56:20 UTC
- ブランチ: `perf/count-legal-evasions-up-to`
- 結論: 採用しない。探索ノード数と延長回数は一致したが、search profileが悪化した。

## 目的

採用済みの低分岐王手応手延長は、以下の判定で合法手を全生成していた。

```rust
position.legal_moves().len() <= CHECK_EVASION_EXTENSION_MAX_REPLIES
```

これを上限付き合法応手カウントに置き換え、4手以上見つかったら早期終了することで高速化を狙った。

## 試した変更

`shogi_lib::Position` に以下を追加した。

```rust
pub fn legal_evasion_count_up_to(&self, limit: usize) -> usize
```

探索側では以下へ置換した。

```rust
position.legal_evasion_count_up_to(CHECK_EVASION_EXTENSION_MAX_REPLIES + 1)
    <= CHECK_EVASION_EXTENSION_MAX_REPLIES
```

## 検証

変更前:

```text
samples: 72
total nodes: 19067196
quiescence nodes: 17270535
check evasion extensions: 26959
elapsed ms: 62230.49
nodes/sec: 306396.37
```

変更後:

```text
samples: 72
total nodes: 19067196
quiescence nodes: 17270535
check evasion extensions: 26959
elapsed ms: 64726.09
nodes/sec: 294582.83
```

探索ノード数・qsearchノード数・延長回数は一致した。挙動は不変と見られるが、elapsedは悪化した。

## 判断

採用しない。

理由:

- 実装では `generate_evasions` 自体は全擬似応手を生成しており、合法性filterだけ早期終了している。
- 低分岐王手応手延長の判定箇所では、早期終了の利益より追加メソッド/分岐のコストが上回った。

## 次の方針

この方向を続けるなら、擬似応手生成自体を上限付きで止める必要がある。ただしmovegenの各生成関数へ上限伝播が必要になり、実装リスクが高い。現時点では別候補を優先する。
