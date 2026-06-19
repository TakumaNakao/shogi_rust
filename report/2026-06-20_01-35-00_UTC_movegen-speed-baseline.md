# movegen speed baseline

- 作成日時: 2026-06-20 01:35:00 UTC
- 対象: `master` 519d8e3
- 目的: 合法手生成・do/undo高速化実験の比較基準を固定する。

## コマンド

```text
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin movegen_profile --bin search_profile
```

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/movegen_profile \
  --positions taya36.sfen \
  --samples 3600 \
  --repeat 100 \
  --seed 12001
```

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/movegen_profile \
  --positions taya36.sfen \
  --samples 3600 \
  --repeat 100 \
  --seed 12001 \
  --do-undo
```

## 結果

### legal_movesのみ

```text
positions: 360000
legal moves: 22521400
max moves: 143
do/undo moves: 0
elapsed ms: 1143.46
positions/sec: 314832.82
moves/sec: 19695766.31
avg moves/position: 62.56
```

### legal_moves + do/undo

```text
positions: 360000
legal moves: 22521400
max moves: 143
do/undo moves: 22521400
elapsed ms: 4349.45
positions/sec: 82769.13
moves/sec: 5177990.81
do-undo/sec: 5177990.81
avg moves/position: 62.56
```

## 使い方

合法手生成・do/undo・局面更新まわりを変更した場合は、同じコマンドで比較する。

時間の揺らぎがあるため、小さい差だけで採用しない。速度改善だけでなく、`search_profile` のノード数・qsearchカウンタが意図せず変わっていないかも確認する。
