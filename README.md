# shogi_ai

Rust製のUSI将棋エンジンと、評価関数・教師dataを構築する学習tool群です。通常利用する
正式binaryは`usi_engine`です。研究用のprobeや過去の実験toolはdefault buildから分離して
います。

## ShogiHomeで使う

1. [Releases](https://github.com/TakumaNakao/shogi_rust/releases)からOSに合う
   `usi_engine`または`usi_engine.exe`を取得する。
2. ShogiHomeの「エンジン設定」からbinaryを追加する。
3. エンジン設定の「評価関数ファイル」に対応するweight fileを指定する。
4. 設定を保存し、対局または検討でengineを選ぶ。

weight形式はHalfKP-32とHalfKP-64で異なるため、配布物とweightの組合せを確認してください。

## Build

Rust stableを使用します。Linuxでplot機能を含む学習・研究toolをbuildする場合はfontconfigが
必要です。

```bash
# production engine（defaultはHalfKP-32）
cargo build --release

# 配布構成のHalfKP-64 engine
cargo build --release --features halfkp64 --bin usi_engine

# supported training tool
RUST_FONTCONFIG_DLOPEN=1 cargo build --release \
  --features halfkp64,training-tools \
  --bin halfkp_search_teacher --bin halfkp_search_train

# benchmark/profile tool
cargo build --release --features halfkp64,benchmark-tools --bin search_fingerprint
```

全binaryの分類とfeatureは[`docs/binaries.md`](docs/binaries.md)を参照してください。

## Test

```bash
cargo test --workspace --release --features halfkp64,training-tools
cargo check --workspace --all-targets --all-features
tools/check_clippy_ratchet.sh
```

探索の意味的fingerprint、format golden fixture、USI transcript testを含みます。性能変更は
固定条件のpaired benchmarkを行い、棋力や探索意味を推測だけで変更しません。

## Training

- HalfKP-64の学習と検証: [`docs/halfkp64_training.md`](docs/halfkp64_training.md)
- search teacher/trainer設計: [`docs/halfkp_performance_training_design.md`](docs/halfkp_performance_training_design.md)
- KPP教師あり学習: [`docs/kpp_learn.md`](docs/kpp_learn.md)
- artifactとmanifest: [`docs/artifact_policy.md`](docs/artifact_policy.md)

大規模pipelineは`scripts/train_halfkp64_large.sh`、小規模なsearch teacher学習は
`scripts/train_halfkp64_search.sh`が入口です。生成物を再利用するときはsidecar manifestの
fingerprintとhashが一致した場合だけ許可されます。

## Development

- 開発文書索引: [`docs/README.md`](docs/README.md)
- architecture・全Phase計画・進捗: [`docs/refactoring_plan_v2.5.4.md`](docs/refactoring_plan_v2.5.4.md)
- 保存すべき探索/format/USI契約: [`docs/refactoring_handoff_v2.5.4.md`](docs/refactoring_handoff_v2.5.4.md)
- toolchainとCI policy: [`docs/toolchain_policy.md`](docs/toolchain_policy.md)
- 実験reportの検索とmetadata: [`report/README.md`](report/README.md)

変更はformat互換、探索意味、determinism、性能の各gateを通してから採用します。新しいbinary、
artifact、設計判断を加える場合は、それぞれinventory、artifact policy、ADRも同じ変更で更新して
ください。
