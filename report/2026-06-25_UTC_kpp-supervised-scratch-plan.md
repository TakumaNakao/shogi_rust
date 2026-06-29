# KPP教師ありscratch学習への方針転換

- 作成日時: 2026-06-25 UTC
- ブランチ: `improve-mmto-training-density`
- 目的: MMTO系fine-tuneが採用可能な重みを作れていないため、棋譜教師あり学習でscratch重みとwarm-start重みを比較できる基盤を作る。

## 背景

直近のMMTO実験では、lossは改善するがrerank regretの改善が小さく、100局ベンチへ進める候補を作れなかった。

代表例:

- rerank baseline mean regret: `8.36`
- candidate mean regret: `8.26`
- 改善量: `0.10cp`
- gate要求: `0.50cp`
- 判定: 不採用

この結果から、現行重み自身の探索をteacherにしたfine-tuneでは新しい棋力情報が不足していると判断した。

## 方針

当面は以下を主軸にする。

1. Wdoor/floodgate高レート棋譜を使った教師ありCE学習。
2. `scratch` と `warm-start` を同じ条件で比較。
3. offline精度だけで採用せず、重み検査、探索プロファイル、20局smoke、100局ベンチを通す。

MMTOは完全に捨てず、棋譜教師ありで作った候補重みの補助fine-tune候補に下げる。

## 実装

`kpp_learn` に明示的な初期化モードを追加した。

- `--init-mode auto`
  - 従来互換。
  - `--weights` があればload、なければ新規保存。
- `--init-mode load`
  - `--weights` が存在しなければエラー。
  - warm-start比較用。
- `--init-mode scratch`
  - `--weights` を読まず、`SparseModel::new` から開始。
  - `--scratch-material-coeff` でmaterial係数を指定。
  - baseline誤上書きを避けるため、`--output` と `--weights` が同じ場合はエラー。

また `tools/run_kpp_supervised_compare.sh` を追加し、同じ条件でscratch/warm-startを順に実行できるようにした。

## 検証

実行済み:

```bash
bash -n tools/run_kpp_supervised_compare.sh
env RUST_FONTCONFIG_DLOPEN=1 cargo fmt --check
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin kpp_learn
```

1棋譜を使ったscratch smoke:

```text
init mode: scratch
scratch mode: 重みファイルを読み込まず、初期値から学習します
material coeff: 0.145648
学習完了
```

安全確認:

```text
--init-mode load requires --weights to exist
--init-mode scratch requires --output to differ from --weights
```

warm-start smokeも同じ1棋譜で正常終了した。

## 次の実験

まずは同じデータ・同じハイパーパラメータで比較する。

```bash
RUN_ROOT="data/wdoor/runs/kpp_supervised_compare_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"

nohup env RUST_FONTCONFIG_DLOPEN=1 \
  RUN_ROOT="$RUN_ROOT" \
  YEARS="2023 2024 2025 2026" \
  RUN_KIND=both \
  EPOCHS=4 \
  MIN_PLAYER_RATE=4000 \
  bash tools/run_kpp_supervised_compare.sh \
  > "$RUN_ROOT/compare_stdout.log" 2>&1 &
```

長時間学習後は、`best.binary` と最終重みの両方を検査対象にする。scratchが明らかに弱い場合でも、その結果は重要であり、以後はwarm-startまたは別目的関数へ集中する。
