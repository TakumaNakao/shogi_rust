# Benchmark baselines

このディレクトリは、リファクタリング前後の比較条件と結果を保存する。
生成されたweight、dataset、実行binaryそのものは置かず、再現に必要なrevision、command、hash、意味的カウンタを記録する。

## Baselineの種類

### Search fingerprint

`search_fingerprint`は、小さなtracked fixtureをThreads=1で探索し、次をJSONとして出力する。

- bestmove
- root scoreとその`f32` bit pattern
- PV
- completed depth
- total/qsearch node counters
- qsearch生成、破棄、探索、SEE skip
- aspirationとcheck evasionの統計

時間とNPSは含めない。機械的リファクタリングでは、expected JSONとの完全一致を要求する。

```bash
RUST_FONTCONFIG_DLOPEN=1 cargo run --release --features halfkp64 \
  --bin search_fingerprint -- \
  --expected benchmarks/baselines/search_fingerprint_v2.5.4_plus_master.json
```

expectedを更新してよいのは、基準revisionを意図的に変更した場合、または正しさ・探索意味変更として差分がレビューされた場合だけである。

### Performance profile

性能基準は`v2.5.4_plus_master.json`に記録する。
elapsed timeとNPSはmachine依存のため、同一machine、同一toolchain、同一binary/model/input hashで比較する。
hosted CIでは時間をhard gateにしない。

## 更新規則

1. Git worktreeがcleanか確認する。
2. revision、dirty flag、rustc、target、CPU featureを記録する。
3. binary、model、inputのSHA-256を記録する。
4. warm-up後に7回測定する。初期baselineの3回値は履歴として維持する。
5. 決定的カウンタとelapsed timeを分離して保存する。
6. 変更理由と対応するPRまたはADRを記録する。
