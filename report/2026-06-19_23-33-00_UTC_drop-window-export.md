# worst drop window export 追加

- 日時: 2026-06-19 23:33 UTC
- ブランチ: `tooling/export-drop-windows`
- 目的: 終局 tail そのものではなく、評価が崩れる直前・直後の window 局面を value regression / regret gate 用に抽出できるようにする。

## 追加したもの

`record_analyze` に以下の引数を追加した。

```text
--export-drop-windows <path>
--drop-window-before <N>  # default 3
--drop-window-after <N>   # default 1
```

既存の `--top-drops` で大きい評価急落を選び、その各 drop ply について `before..after` の合法手あり局面を SFEN として出力する。重複は除去する。

## 検証

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin record_analyze
```

結果: 通過。

## スモーク

```bash
mkdir -p /tmp/shogi_value_drop

env RUST_FONTCONFIG_DLOPEN=1 target/release/record_analyze \
  --weights policy_weights_v2.1.0.binary \
  --record-dir /tmp/shogi_bench_master_vs_v241_40_seed10101 \
  --tail-plies 16 \
  --top-drops 20 \
  --export-drop-windows /tmp/shogi_value_drop/drop_windows.sfen \
  --drop-window-before 3 \
  --drop-window-after 1
```

結果:

```text
exported tail drop window positions: 95 to /tmp/shogi_value_drop/drop_windows.sfen
```

## 判断

GPT-5.5 xhigh の提案どおり、次の重み更新実験は hard tail だけではなく、この drop window と通常局面 guard を混ぜて行う。採否は offline valid、regret gate、same-engine weight 対戦、v2.4.1 比較の順で判定する。

この変更は解析基盤のみで、探索本体・評価関数・重みファイルには影響しない。
