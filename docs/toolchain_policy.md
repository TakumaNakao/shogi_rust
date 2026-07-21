# Rust toolchain・MSRV方針

## 現在の方針

- 開発とPR CIはRust stableを使用する。
- editionはworkspace内の両packageでRust 2021を維持する。
- `Cargo.lock`はworkspace rootの一冊をcommitし、applicationとして同じ依存解決を共有する。
- release benchmarkと探索fingerprintには、実行時の正確な`rustc`、`cargo`、target、CPU、revisionを記録する。
- 現時点ではMSRV（Minimum Supported Rust Version）を保証せず、`package.rust-version`も宣言しない。

MSRVを未宣言とする理由は、過去の最小toolchainでworkspace全targetを検証した証拠がまだないためである。
推測した版をmanifestへ書くと、利用者へ誤った互換性を約束することになる。

## CIの役割

Linux CIはformat、HalfKP-32 library test、HalfKP-64 workspace test、全featureのall-target check、
production libraryのClippy ratchet、全targetのClippy可視化、探索fingerprintを実行する。
Windows CIはHalfKP-64 workspace testと探索fingerprintを実行する。

Clippy ratchetのproduction範囲は次の二つである。

- `shogi_lib`のlibrary target: warningを一件も許可しない。
- `shogi_ai`のlibrary target: 2026-07-20時点で存在するlint classだけを
  [`tools/check_clippy_ratchet.sh`](../tools/check_clippy_ratchet.sh)で明示的に許可し、
  それ以外のwarningを失敗させる。

多数の研究・学習binaryとtest targetは既存warningを含むため、現段階ではreport-onlyとする。
一括自動修正は探索・評価のhot pathを変え得るため行わず、担当moduleを変更するPRで小さく解消する。

## 再現可能なrelease

性能比較では「stable」という名前だけを比較条件にしない。
次のようにmetadataを保存し、記録された正確なtoolchainで比較対象を再測定する。

```bash
tools/capture_benchmark_metadata.sh \
  --artifact target/release/search_fingerprint \
  --command 'target/release/search_fingerprint --depth 3'
```

toolchain更新後に生成コードや実行時間が変わった場合は、コード変更と混ぜず、
専用branchでfingerprint、format fixture、release benchmarkを再検証する。

## MSRVを宣言する条件

将来MSRVを設定する場合は、次を同じ変更で満たす。

1. 候補toolchainを固定したCI jobで`cargo test --workspace --release --features halfkp64,training-tools`を通す。
2. `cargo check --workspace --all-targets --all-features`を通す。
3. 全直接依存とlockfileの依存解決が候補版をサポートすることを確認する。
4. rootとmemberの`package.rust-version`を同じworkspace値から継承させる。
5. 本文書とbenchmark metadataへ、検証日と正確なversionを記録する。

宣言後は、MSRV CIとstable CIを分離する。MSRVは互換性、stableは新しいcompilerでの
warning・将来非互換を検出する役割とする。
