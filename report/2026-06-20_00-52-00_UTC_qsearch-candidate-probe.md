# qsearch candidate probe 追加

- 日時: 2026-06-20 00:52 UTC
- ブランチ: `tooling/qsearch-candidate-probe`
- 目的: qsearch 直接生成の実装時に、参照フィルタ方式との候補集合差分をすぐ検出できるようにする。

## 追加したもの

`src/bin/qsearch_candidate_probe.rs` を追加した。

入力局面ごとに以下を比較する。

- 参照方式: `legal_moves() -> capture/check retain`
- 現行API: `legal_quiescence_moves()`

差分がある場合は、missing / extra とSFENを出力して非ゼロ終了する。

## 検証

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin qsearch_candidate_probe
```

結果: 通過。

## probe

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/qsearch_candidate_probe \
  --input taya36.sfen \
  --input /tmp/shogi_value_drop/drop_windows.sfen \
  --show 5
```

結果:

```text
positions: 2150
mismatches: 0
reference candidates: 6605
current candidates: 6605
```

## 判断

現行APIは参照方式と一致している。次に qsearch 直接生成を試す場合は、このprobeを最初のゲートにし、mismatchが出る案は `search_profile` に進めず棄却または修正する。
