# KPP学習安全制御の追加

- 作成日時: 2026-06-21 12:16:25 UTC
- ブランチ: `training/kpp-learn-safety-controls`
- 目的: 長時間KPP学習で既存重みを壊しにくくし、検証精度の悪化を早期に検出する。

## 背景

WCSC系CSA棋譜を使った長時間CE学習では、対局ベンチで明確な改善が確認できなかった。

- 200局合算: 103勝92敗5分
- total score rate: 52.75%
- 95% CI total: 約45.84%..59.55%
- 検証精度: baseline 22.83% から epoch 4 で 21.00% へ低下

この結果から、現状のhard-label CEをそのまま長時間回すだけでは不安定であり、重み更新を制御する仕組みを先に入れるべきと判断した。

## 追加した機能

`kpp_learn` に以下の安全オプションを追加した。

- `--anchor-l2 X`
  - 各バッチ後に学習開始時の重みへ引き戻す。
  - 式: `w := w + X*(w0 - w)`
  - 評価関数の大きなドリフトを抑える目的。
- `--max-weight-delta X`
  - 各重みについて `|w - w0| <= X` にクリップする。
  - 長時間学習で一部特徴だけが暴走するのを防ぐ目的。
- `--early-stop-min-accuracy-drop X`
  - 検証精度がbaselineからXポイント以上低下したら学習を停止する。
  - 悪化した学習を長時間継続しないための停止条件。
- `--best-checkpoint-path path`
  - epochごとの検証精度がこれまでで最良なら重みを保存する。
  - 最終epochではなく、最も検証精度の良い重みを後から評価できるようにする。

`--freeze-material` との併用を前提にし、今回のanchor/clampはKPP重みだけを対象にしている。

## 次の実験方針

まず小規模な1 epoch実験で、以下を確認する。

- 検証精度がbaselineから大きく落ちないこと。
- `max|w-w0|` と `clamped_weights` がログで監視できること。
- best checkpointが保存されること。
- 40局以上のベンチで現行固定版に対して明確に悪化しないこと。

候補設定:

```bash
--anchor-l2 0.0005 \
--max-weight-delta 0.05 \
--early-stop-min-accuracy-drop 0.5 \
--best-checkpoint-path data/wdoor/runs/<run-name>/best.binary
```

この設定で改善傾向が出ない場合は、hard-label CE単独ではなく、regret-weighted soft CEやroot候補手のpairwise/listwise rankingへ進む。

## 検証

以下を確認した。

```bash
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

結果: 成功。
