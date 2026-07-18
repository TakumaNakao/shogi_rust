# HalfKP-64 KPP蒸留

`halfkp64` featureでコンパイルすると、HalfKPのhidden幅が32から64になり、
重みマジックも`HKP00064`へ切り替わります。通常のビルドは従来の32次元重みを
そのまま扱います。

## 学習

リポジトリのルートで次を実行します。

```bash
./scripts/train_halfkp64_kpp.sh
```

デフォルトでは、探索・ランダム・本譜をシャード単位で混合し、実効比率を
約54%/26%/20%にして学習します。各エポックでシャード順とバッチ内局面を
変更します。さらに約47万の合法手後局面を学習へ追加します。独立した2026年本譜、
探索、ランダムのholdoutでMAE/RMSE/bias/p95/p99/符号不一致を検証し、
1,957ルートの全合法手についてKPP最善候補一致率、ペア順位一致率、選択後悔値も
測定します。出力先は
`data/policy_weights_halfkp64_kpp_distilled.binary`、ログは
`data/halfkp64_kpp_training.csv`です。既定の最大エポック数は12で、複合検証指標が
3エポック改善しなければ最良重みを残して停止します。初期値・エポック数を変更する場合は
環境変数で指定します。

```bash
EPOCHS=12 \
./scripts/train_halfkp64_kpp.sh
```

各エポック終了時に`data/halfkp64_kpp_checkpoint`へlatest重みとAdaGrad状態を
原子的に保存します。中断後は次で再開します。

```bash
RESUME=1 ./scripts/train_halfkp64_kpp.sh
```

別の64次元重みを初期値にする場合は`INIT=/path/to/weights`を追加できます。
32次元重みは初期値にできません。

`THREADS=0`はRayonの既定値を使い、利用可能なCPUスレッドを制限しません。
メモリを増減する場合は`BATCH_SIZE`を変更できますが、通常は256のまま使います。

## 64次元エンジンでの検証

学習後、同じfeatureを付けてエンジンをビルドします。

```bash
RUST_FONTCONFIG_DLOPEN=1 cargo build --release --features halfkp64 --bin usi_engine
RUST_FONTCONFIG_DLOPEN=1 cargo run --release --features halfkp64 --bin search_profile -- \
  --halfkp-weights data/policy_weights_halfkp64_kpp_distilled.binary \
  --positions taya36.sfen --samples 32 --depth 10 --time-limit-ms 1000
```

32次元重みを64次元featureのエンジンで読み込むことはできません。学習・検証時は
必ず`--features halfkp64`を付けてください。
