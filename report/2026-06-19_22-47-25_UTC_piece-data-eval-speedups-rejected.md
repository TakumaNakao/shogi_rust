# PieceKindテーブル化とpiece data直読み高速化の棄却

- 作成日時: 2026-06-19 22:47:25 UTC
- ブランチ: `perf/piece-kind-tables`
- 結論: 採用しない。eval単体は改善したが、search profileが悪化した。

## 試した内容

評価関数内の以下の処理を高速化しようとした。

1. `piece_kind_value` のmatchを `PieceKind as usize` のテーブル参照へ変更。
2. `board_kind_to_index` / `hand_kind_to_offset` をテーブル化。
3. `predict_from_position` の盤上走査で `Piece::piece_kind()` / `Piece::color()` 呼び出しを減らし、`Piece::as_u8()` からdisc/colorを直接読む。

途中で、`PieceKind` の識別子順がKPPのboard kind順と異なることを確認した。

```text
King = 8
ProPawn = 9
...
```

そのため、単純な `kind as usize - 1` はKPP順と一致せず、評価値が壊れる。修正後は明示テーブルを使い、score sum一致を確認した。

## 結果

### テーブル化

修正後のeval profile:

```text
evals: 819200
score sum: 754071.6
elapsed ms: 2808.47
```

採用済みmasterの `2894.85ms` よりeval単体は改善した。

しかしsearch profile:

```text
samples: 72
total nodes: 19067196
elapsed ms: 62572.40
```

採用済みmasterの `62230.49ms` よりわずかに悪化した。

### `predict_from_position` のpiece data直読み

eval profile:

```text
evals: 819200
score sum: 754071.6
elapsed ms: 2569.90
```

eval単体は大きく改善した。

しかしsearch profile:

```text
samples: 72
total nodes: 19067196
elapsed ms: 71364.14
```

探索全体では大きく悪化した。

## 判断

採用しない。

理由:

- 目的は探索速度改善であり、eval単体が速くてもsearch profileが悪化する変更は採用できない。
- テーブル化やpiece data直読みは、関数サイズ・インライン展開・分岐予測・キャッシュ挙動のいずれかで探索全体に悪影響を与えている可能性がある。
- `PieceKind` の識別子順とKPP順が異なるため、今後この領域を触る際は明示テーブルが必須。

## 次の方針

評価関数ホットパスは、今回採用済みの `final_index < MAX_FEATURES` 分岐削除に留める。さらなる高速化は、piece kind変換のテーブル化ではなく、探索中の評価呼び出し回数削減や局面生成側の改善を優先する。
