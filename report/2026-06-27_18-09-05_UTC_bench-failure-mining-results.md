# bench_failure_miner 60局解析結果

- 作成日時: 2026-06-27 18:09:05 UTC
- 対象: 既存60局ベンチ棋譜2本
- 目的: 次にroot救済を実装するべきか、bench由来counterexample学習へ進むべきか判断する。

## 実行条件

実験は `GPT-5.3-codex-spark` サブエージェントに委任した。

共通条件:

```text
tail-plies: 10
timed-depth: 4
teacher-depth: 5
time-limit-ms: 100
bad-regret-cp: 200
root-rescue-good-regret-cp: 80
root-rescue-min-improvement-cp: 200
```

## Run A

対象:

```text
/tmp/shogi_weight_bench_pv_sibling_3k_ultrasafe_11101
```

出力:

```text
/tmp/bench_failure_miner_60_20260627_175746
```

主要結果:

```text
records loaded: 60
positions probed: 125
samples mined: 125
actual teacher mismatches: 95 (76.00%)
timed teacher mismatches: 96 (76.80%)
actual bad_regret_gt_200: 13 (10.40%)
timed bad_regret_gt_200: 12 (9.60%)
both actual_and_timed_bad: 12
actual_bad_timed_not_bad: 1
root_rescuable: 1
in_check: 75 (60.00%)
legal_moves_le_3: 57 (45.60%)
baseline sweep starts: 1
baseline sweep samples: 10 (8.00%)
```

代表:

```text
top timed-bad:
  game_045_new_black_BaselineWin.usi ply=104 timed_regret=99867.5 actual_minus_timed=0.0
  game_048_new_white_BaselineWin.usi ply=95 timed_regret=99692.0 actual_minus_timed=0.0
  game_012_new_white_BaselineWin.usi ply=139 timed_regret=99663.3 actual_minus_timed=0.0

top root-rescuable:
  game_021_new_black_BaselineWin.usi ply=118 improvement=99075.1 in_check=true legal=4
```

## Run B

対象:

```text
/tmp/shogi_weight_bench_mmto_replay100k_hard_10001
```

出力:

```text
/tmp/bench_failure_miner_60_replay100k_20260627_180224
```

主要結果:

```text
records loaded: 60
positions probed: 180
samples mined: 180
actual teacher mismatches: 137 (76.11%)
timed teacher mismatches: 135 (75.00%)
actual bad_regret_gt_200: 19 (10.56%)
timed bad_regret_gt_200: 19 (10.56%)
both actual_and_timed_bad: 18
actual_bad_timed_not_bad: 1
root_rescuable: 1
in_check: 107 (59.44%)
legal_moves_le_3: 78 (43.33%)
baseline sweep starts: 10
baseline sweep samples: 100 (55.56%)
```

代表:

```text
top timed-bad:
  game_055_new_black_BaselineWin.usi ply=134 timed_regret=100407.7 actual_minus_timed=0.0
  game_001_new_black_BaselineWin.usi ply=76 timed_regret=100174.9 actual_minus_timed=0.0
  game_022_new_white_BaselineWin.usi ply=141 timed_regret=100088.3 actual_minus_timed=0.0

top root-rescuable:
  game_051_new_black_BaselineWin.usi ply=104 improvement=99701.7 in_check=true legal=5
```

## 解釈

2本の60局で傾向はほぼ同じだった。

- bad regret は約10%。
- `actual_bad` と `timed_bad` がほぼ重なる。
- `actual_minus_timed_regret_cp >= 200` は各runで1件だけ。
- root-rescuable も各runで1件だけ。
- bad局面の約半分は王手中または合法手少数に偏る。

これは「実戦手だけが悪く、同じ局面で現行root探索なら救える」ケースが少ないことを意味する。過去のroot verificationが不発だった理由とも整合する。

一方で、`actual==timed` の巨大regretが多い。これは現行depth/timeのroot探索がteacher depthの安全手を選べていない局面であり、単発root救済よりも以下の使い方が自然である。

- teacher手をbench由来counterexampleとして集める。
- timed rootが同じ悪手を選ぶ局面だけを、学習またはrerank gateのhard setにする。
- 王手中・合法手少数の割合を見て、将来の王手回避専用改善を検討する。

## 判断

現時点ではroot救済の新実装へ進まない。

理由:

- root-rescuable は120局相当で2件だけ。
- 過去のroot verify/root ordering系は不発または悪化。
- bad局面の多くは `actual` と `timed` が同じ悪手で、root探索そのものがteacher depthに負けている。

次は、既存bench群から `timed_bad` / teacher手のcounterexampleをさらに集め、50件以上の重複除去済みhard setを作る。

## 次の手順

1. 既存の20局/60局ベンチ群へ `bench_failure_miner` をまとめて実行する。
2. `timed_bad.sfen` と `failures.jsonl` を結合し、重複SFENを除去する。
3. `teacher_move`, `actual_move`, `timed_move`, `regret` を持つJSONLを学習/検証用hard setとして保存する。
4. `timed_bad` が50件以上集まれば、direct counterexample学習またはroot decision rerank実験へ進む。
5. 集まらなければ、探索改善より序盤・評価関数更新へ戻る。
