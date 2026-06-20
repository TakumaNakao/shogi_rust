# negative SEE check rescue rejected

- 作成日時: 2026-06-20 02:18:00 UTC
- 実験ブランチ: `experiment/qsearch-negative-see-check-rescue`
- 実験コミット: `e74b6e2`
- 判断: 不採用

## 背景

GPT-5.5 xhigh の次点案として、qsearch内で `SEE < 0` の手を即skipする処理を、王手になる手だけ救済する実験を行った。

通常捕獲の負SEE skipは維持し、負SEEでも王手になる手だけ探索する。目的は、粗いSEEが犠牲王手や終盤の強制手を落としている可能性を検証すること。

## 実装

`src/ai.rs`:

- `self.see(position, mv) < 0` のとき、
  - `position.is_check_move(mv)` なら skip せず探索する。
  - それ以外は従来どおり skip する。
- `quiescence_negative_see_check_rescues` カウンタを追加。

`src/bin/search_profile.rs`:

- 救済発火数を出力する。

## 検証

### tests

```text
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

結果: pass

### search_profile

```text
env RUST_FONTCONFIG_DLOPEN=1 target/release/search_profile \
  --weights policy_weights_v2.1.0.binary \
  --positions taya36.sfen \
  --samples 72 \
  --depth 5 \
  --seed 9501
```

```text
samples: 72
total nodes: 18717204
quiescence nodes: 17021813
quiescence moves considered: 9068609
quiescence moves generated: 137904523
quiescence moves discarded: 128835914
quiescence moves searched: 3498220
quiescence see skips: 2881797
quiescence negative see check rescues: 209572
quiescence terminal mates: 240
check evasion extensions: 26728
aspiration fail lows: 0
aspiration fail highs: 0
aspiration researches: 0
elapsed ms: 62504.26
nodes/sec: 299454.84
```

master基準と比較すると、qsearch searched は約7%増えたが、総ノードはやや減少した。profile上は即撤退するほどのノード爆発ではなかった。

### same-engine 20局

new: `experiment/qsearch-negative-see-check-rescue`  
baseline: `master`  
seed: `13101`

```text
new wins: 11
baseline wins: 9
draws: 0
new decisive win rate: 55.00%
new total score rate: 55.00%
```

20局では最低ゲートを通過したため、40局へ拡大した。

### same-engine 40局

new: `experiment/qsearch-negative-see-check-rescue`  
baseline: `master`  
seed: `13101`  
record-dir: `/tmp/shogi_bench_negsee_check_40_seed13101`

```text
new wins: 20
baseline wins: 20
draws: 0
new decisive win rate: 50.00%
new total score rate: 50.00%
```

`record_analyze` 要約:

```text
end reasons:
  Resign: 40
paired starts:
  new sweeps: 4
  baseline sweeps: 4
  splits: 12
  draw/mixed pairs: 0
average final score for new: -58.3
average final score for NewWin: 460.2
average final score for BaselineWin: -576.7
terminal result mismatches: 0
non-terminal score/result sign mismatches: 0
```

## 判断

40局で 20-20、paired starts でも new sweeps と baseline sweeps が同数だった。

profile上は許容範囲だったが、強さ改善としては確認できないため採用しない。qsearchで負SEE王手を一律救済する方針は、現行評価・探索では有効な改善ではなかった。

次は、qsearchに踏み込むよりも、敗局や高regret局面に限定した root/終盤系の小さな救済、または高速化基盤のより安全な改善に戻る。
