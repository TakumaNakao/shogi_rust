# legal_quiescence_moves API 追加

- 日時: 2026-06-20 00:32 UTC
- ブランチ: `tooling/legal-quiescence-api`
- 目的: qsearch 直接生成の前段として、旧 `legal_moves() -> capture/check retain` と同じ候補集合を返すAPIと一致テストを追加する。

## 追加したもの

`shogi_lib::Position` に以下を追加した。

```text
legal_quiescence_moves()
legal_quiescence_moves_with_generated_count()
```

現時点では内部実装は旧方式と同じで、速度改善は狙っていない。`ai.rs` の qsearch はこのAPI経由に差し替えた。

`shogi_lib/src/movegen.rs` に、複数SFENで旧方式の参照フィルタと新APIの集合一致を確認するテストを追加した。

## 検証

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin search_profile
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
quiescence moves generated: 138833345
quiescence moves discarded: 129991362
quiescence moves searched: 3264998
quiescence see skips: 3096541
check evasion extensions: 26959
elapsed ms: 62414.08
```

前回計測と qsearch カウンタは完全一致した。

## 判断

この変更は探索挙動を変えず、qsearch 直接生成の安全な比較足場になるため採用する。次は `legal_quiescence_moves_with_generated_count` の内部だけを最適化し、同一集合テストと `search_profile` の nodes/counters 完全一致を採否条件にする。
