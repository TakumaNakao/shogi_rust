# HalfKP学習開始ゲート

## 実装

`dataset_build` に棋譜単位のsplit、`--max-records-per-game`、`--sample-every`、終局理由、結果既知フラグを追加した。不明終局は `--decisive-only` で除外でき、同一棋譜の局面がtrain/valid/testをまたがらない。`halfkp_feature_dump` は新しい結果情報を保持する。

`halfkp_train` は棋譜の勝敗だけを教師にするHalfKP WDL学習器である。SFENをストリーム処理し、ClippedReLU、WDL BCE、active row AdaGrad、L2、初期HalfKP重み、validation BCEによる早期停止を備える。評価時の温度は学習時の `--temperature` と一致させる。外部エンジンの評価値は使わない。

## データ

2023--2025年の `data/wdoor/extract` から次の条件で生成した。

```text
--shuffle-games --seed 20260716 --max-records-per-game 32 --sample-every 3
--min-player-rate 4000 --decisive-only --valid-percent 10 --test-percent 10
--min-ply 8 --max-ply 160
```

生成済みマニフェストは `/tmp/halfkp_scale100k/manifest.json`、`/tmp/halfkp_scale500k/manifest.json`、`/tmp/halfkp_scale1m/manifest.json` にある。1M版は39,923棋譜、train 798,146、valid 101,766、test 100,088局面である。2026年は別年の独立ホールドアウト `/tmp/halfkp_holdout_fair/train.jsonl`（120,000局面）に固定した。

## 開始ゲート実験

初期値は同じHalfKP静的モデル、learning rate `0.003`、temperature `600`、L2 `1e-6`、3 epochで揃えた。評価は2026年ホールドアウトで行った。

| model | holdout BCE | accuracy |
| --- | ---: | ---: |
| static | 0.689235 | 0.6081 |
| 100k | 0.601362 | 0.6786 |
| 500k | 0.593675 | 0.6852 |
| 1M | 0.587966 | 0.6866 |

validation BCEも1Mでは epoch 1/2/3 が `0.567925/0.566053/0.566394` で、発散やNaNはない。学習時間は1M trainで約23.7秒/epoch（このLinux CPU）だった。
10 epoch上限・patience 2の再実験では epoch 4 の validation BCE `0.567410` の後に停止し、最良は epoch 2 の `0.566053` だった。

## 判定と運用

1Mまで独立ホールドアウトBCEが単調に改善し、accuracyも改善したため、データを増やす方針とこのWDL教師で**段階的な長時間学習を開始してよい**と判定する。ただし、固定48時間を無監視で回すことは承認しない。まず1M--3M局面を対象に、epochごとにvalidationを測定し、patience 2--3の早期停止と最良重み保存を必須にする。最良重みが500k/1M候補を独立ホールドアウトで上回らなければ、その時点で停止する。

このtrainerは現在、bounded streamingのオンラインAdaGradであり、Rayon mini-batch、optimizer再開checkpoint、factorizationは未実装である。48時間級の本学習を定常運用する前にこれらを追加する。現段階の実験は、長時間学習に進む価値があることを確認する採用ゲートであり、v2.1.0 KPPを置き換える強さの証明ではない。

## 2M中間フェーズの対局ゲート

2M局面（train 1,600,118 / valid 200,866 / test 199,016）で学習率を比較した。

| lr | 2026 holdout BCE | accuracy | validation best |
| ---: | ---: | ---: | ---: |
| 0.001 | 0.582227 | 0.6888 | 0.560923 |
| 0.003 | 0.584190 | 0.6892 | 0.559084 |
| 0.006 | 0.583494 | 0.6926 | 0.557274 |

各候補をv2.1.0 KPPと12局、両色6局ずつ、1手2秒、最大120手、jobs 1で対局させた。全36局が投了で終了し、最大手数引き分けはなかった。

| 候補 | HalfKP勝 | KPP勝 | 引分 | HalfKP score rate |
| ---: | ---: | ---: | ---: | ---: |
| lr 0.001 | 2 | 10 | 0 | 16.67% |
| lr 0.003 | 0 | 12 | 0 | 0.00% |
| lr 0.006 | 0 | 12 | 0 | 0.00% |

したがって、2Mへの拡張と長時間学習への移行は採用しない。独立ホールドアウトの改善は、実戦強度の改善を意味しなかった。候補の出力層を確認すると、static HalfKPの出力重みがおおむね `1e-3` なのに対し、候補は `1e-1` 程度まで増大していた。WDL BCEのオンラインAdaGradが評価値の絶対尺度を壊し、探索で過大な確信を与えた可能性が高い。次は出力スケールを固定・校正し、更新前後の評価値分布とengine forward一致を検証してから再学習する。
