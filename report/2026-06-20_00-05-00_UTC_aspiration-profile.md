# aspiration fail 計測

- 日時: 2026-06-20 00:05 UTC
- ブランチ: `tooling/aspiration-profile`
- 目的: root aspiration window の再探索が速度ボトルネックになっているか確認する。

## 追加したもの

`ShogiAI` に以下の計測カウンタを追加し、`search_profile` に表示した。

```text
aspiration fail lows
aspiration fail highs
aspiration researches
```

探索の分岐や評価値は変更していない。

## 検証

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin search_profile --bin root_decision_probe
```

結果: 通過。

## profile

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/search_profile \
  --weights policy_weights_v2.1.0.binary \
  --positions taya36.sfen \
  --samples 72 \
  --depth 5 \
  --seed 9501
```

結果:

```text
total nodes: 19067196
quiescence nodes: 17270535
quiescence moves considered: 8841983
quiescence moves searched: 3264998
quiescence see skips: 3096541
check evasion extensions: 26959
aspiration fail lows: 0
aspiration fail highs: 0
aspiration researches: 0
elapsed ms: 62238.81
```

従来基準と nodes/qnodes/extensions は一致した。

## 判断

この条件では aspiration full re-search が発生していない。`ASPIRATION_WINDOW = 300` の調整による速度改善余地は確認できないため、window 500/800 の A/B 実験は行わない。

この変更は今後の profile 用計測として残す価値があるが、強さ向上ではないためタグ対象ではない。
