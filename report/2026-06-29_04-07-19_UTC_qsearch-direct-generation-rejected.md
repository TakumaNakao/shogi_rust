# 静止探索候補手の直接生成実験

- 作成日時: 2026-06-29 04:07:19 UTC
- ブランチ: `training/strong-weight-learning-infra`
- 目的: 静止探索で全合法手を生成してから捕獲・王手へ絞る処理を、候補手の直接生成へ置き換えて高速化できるか確認する。

## 実装概要

試験実装では、非王手局面の `legal_quiescence_moves_with_generated_count()` について、以下を直接生成する方針を試した。

- 捕獲手
- 直接王手になり得る移動手
- 王手になり得る駒打ち
- 開き王手候補

候補生成後は既存の合法性判定と静止探索条件で再フィルタした。候補集合の安全確認として、既存の全合法手フィルタ版と一致するテストも追加して検証した。

## 検証結果

`GPT-5.3-codex-spark` サブエージェントに速度測定を委任した。

### movegen profile

条件:

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/movegen_profile \
  --positions converted_records2016_10818.sfen \
  --samples 4096 \
  --repeat 20 \
  --seed 9104 \
  --quiescence
```

結果:

- 変更前: `621k` から `653k positions/sec`
- 直接生成版: `137k` から `140k positions/sec`
- 約78%悪化

### search profile

条件:

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/search_profile \
  --weights policy_weights_v2.1.0.binary \
  --positions taya36.sfen \
  --samples 64 \
  --depth 5 \
  --seed 9101
```

結果:

- 変更前: `387k` から `401k nodes/sec`
- 直接生成版: 約`309k nodes/sec`
- 約22%悪化

`converted_records2016_10818.sfen` でも約38%悪化した。

## 判断

この実装は不採用。

候補手数は大きく減ったが、各駒種の王手可能マスを毎回全マス走査で作る設計が重く、全合法手生成後フィルタより遅かった。探索ノード数と王手回避拡張回数はほぼ同一だったため、棋力差ではなく純粋なオーバーヘッド増加と判断する。

## 採用した副産物

実装本体は戻した。ただし、今後の静止探索高速化で候補集合を壊さないため、`shogi_lib` 単体テストのSFEN入力修正と、`converted_records2016_10818.sfen` 先頭256局面での参照一致テストを追加した。

## 次の候補

静止探索高速化を再試行する場合は、直接生成ではなく以下を優先する。

- 王手可能マスの事前計算または `AttackInfo` の再利用
- 全合法手生成は維持し、`is_check_move()` 呼び出し回数を削る
- 捕獲手のみの軽量生成を先に分離し、王手生成は別段階で評価する
