# KPP学習の勝敗awareサンプリング

- 作成日時: 2026-06-20 02:36:07 UTC
- ブランチ: `training/kpp-result-aware-sampling`
- 対象: `src/kpp_learn.rs`, `README.md`

## 背景

従来の `kpp_learn` は、CSA棋譜の全指し手を教師手として扱っていた。そのため、最終的に負けたプレイヤーの手も、終盤の敗着も同じ重みで正解手として学習していた。

Bonanza/MMTO系の評価関数学習は、強いプレイヤーの棋譜手や探索結果へ評価関数を合わせる教師あり最適化が基本である。棋譜全体を使うこと自体は合理的だが、負け側の終盤手を無条件に正解扱いするのはノイズになりやすい。

参照:

- Hoki and Kaneko, "Large-Scale Optimization for Evaluation Functions with Minimax Search", JAIR 2014  
  https://www.jair.org/index.php/jair/article/view/10871
- Bonanza / Bonanza Method / KPP の概要  
  https://www.chessprogramming.org/Bonanza
- Tomoyuki Kaneko publications, shogi evaluation-function learning references  
  https://www.graco.c.u-tokyo.ac.jp/~kaneko/papers/

## 実装

`kpp_learn` に以下を追加した。

```text
--decisive-only
--winner-only
--exclude-loser-after-ply <N>
--loser-sample-rate <R>
```

意味:

- `--decisive-only`: 勝者を推定できる棋譜だけ使う。
- `--winner-only`: 勝った側の手だけ使う。
- `--exclude-loser-after-ply N`: 負けた側のN手目以降を除外する。
- `--loser-sample-rate R`: 負けた側の手を確率Rで残す。

デフォルトは従来互換で、負け側の手もすべて使う。

勝者推定はCSAの終局理由から行う。

- `%TORYO`, `%TIME_UP`, `%ILLEGAL_MOVE`: 最後に指した側を勝者とみなす。
- `%+ILLEGAL_ACTION` / `%-ILLEGAL_ACTION`: 反則した側の反対を勝者とみなす。
- `%TSUMI`, `%KACHI`: 最後に指した側を勝者とみなす。
- 千日手、持将棋、中断などは勝者なし。

## スモーク検証

`/tmp/shogi_kpp_smoke_csa` の最小CSA 2局で確認した。

全手:

```text
train_total: 12
```

勝者手のみ:

```text
--winner-only --decisive-only
train_total: 6
```

負け側の4手目以降を除外:

```text
--decisive-only --exclude-loser-after-ply 4
train_total: 10
```

負け側を全除外:

```text
--decisive-only --loser-sample-rate 0
train_total: 6
```

期待通りサンプル数が変化した。一時重みは削除済み。

## 推奨設定

初回の長時間学習では、勝者手だけに絞りすぎない。負け側の序中盤には有用な手も多いため、まずは敗着が混ざりやすい終盤だけ除外する。

推奨:

```text
--decisive-only --exclude-loser-after-ply 100
```

比較候補:

```text
--decisive-only --winner-only
--decisive-only --loser-sample-rate 0.5
```

採用は、valid CE/top1 と40局/100局ベンチで判断する。短い10局だけでは採用しない。
