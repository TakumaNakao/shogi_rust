# root境界王手回避延長 実験結果

- 作成日時: 2026-06-19 17:30 UTC
- ブランチ: `exp/root-boundary-check-evasion`
- 目的: 採用済みの境界王手回避延長がroot depth1経路だけ抜けている可能性を検証する。

## 仮説

v2.4.2で採用した境界王手回避延長は、通常探索の `alpha_beta_search_internal()` で `depth == 1` かつ子局面が王手中、合法応手が3手以下のときに1手だけ延長する。

一方、root探索は `find_best_move()` から直接 `alpha_beta_search(position, depth - 1, ...)` を呼ぶため、root depth1では子局面がそのままqsearchへ落ちる。このため「root手が王手で、相手応手が3手以下」の局面だけ採用済み改善が効いていない可能性がある。

## 実装内容

一時実装として以下を追加した。

- `find_best_move()` のroot子探索で、`depth == 1 && position.in_check() && position.legal_moves().len() <= 3` のとき、子探索深さを `0` ではなく `1` にする。
- PVS narrow search側とaspiration fail後のfull search側の両方で同じ判定を使う。
- `root_check_evasion_extensions` カウンタを追加し、`search_profile` に表示する。
- qsearch、通常の境界王手回避延長、評価関数、指し手順序は変更しない。

## テスト

```text
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

結果:

```text
passed
```

## profile

条件:

```text
weights: policy_weights_v2.1.0.binary
positions: taya36.sfen
samples: 72
depth: 5
seed: 9501
```

旧master相当:

```text
samples: 72
total nodes: 19079358
quiescence nodes: 17282476
quiescence moves considered: 8857066
quiescence moves searched: 3270548
quiescence see skips: 3101047
check evasion extensions: 26968
elapsed ms: 77718.62
nodes/sec: 245492.75
```

root境界王手回避延長:

```text
samples: 72
total nodes: 19079358
quiescence nodes: 17282476
quiescence moves considered: 8857066
quiescence moves searched: 3270548
quiescence see skips: 3101047
check evasion extensions: 26968
root check evasion extensions: 0
elapsed ms: 62145.44
nodes/sec: 307011.41
```

taya36 depth5 profileでは発火ゼロだった。マシン負荷により旧profileのelapsedが大きくぶれているため、速度差は判断材料にしない。

発火確認:

```text
converted_records2016_10818.sfen samples=200 depth=1 seed=10031:
  root check evasion extensions: 4

converted_records2016_10818.sfen samples=200 depth=2 seed=10031:
  root check evasion extensions: 4

taya36.sfen samples=72 depth=1 seed=10031:
  root check evasion extensions: 4
```

depth1/2では発火するが、主ベンチ条件のdepth5では発火ゼロだった。

## 20局スモーク

条件:

```text
new engine: root boundary check evasion extension
baseline engine: old master-equivalent binary
weights: policy_weights_v2.1.0.binary for both
positions: taya36.sfen
games: 20
depth: 5
time-limit-ms: 100
seed: 10041
record-dir: /tmp/shogi_bench_root_boundary_master_10041_20
```

結果:

```text
new wins: 9
baseline wins: 11
draws: 0
new decisive win rate: 45.00%
new total score rate: 45.00%
95% CI total: 23.20%..66.80%

end reasons:
  MaxPliesAdjudication: 1
  Resign: 19

paired starts:
  new sweeps: 1
  baseline sweeps: 2
  splits: 7
  draw/mixed pairs: 0

record_analyze:
  non-terminal score/result sign mismatches: 3
```

## 判断

20局で負け越し、paired startsでもbaseline sweepsが多い。主ベンチ条件ではroot拡張の発火がほぼない可能性も高い。

結論:

```text
root境界王手回避延長は棄却。
コード変更は戻し、報告書のみ残す。
```
