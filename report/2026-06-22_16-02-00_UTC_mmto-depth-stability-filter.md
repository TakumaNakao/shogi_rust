# MMTO-lite depth安定性フィルター実装と小規模検証

- 作成日時: 2026-06-22 16:02:00 UTC
- 対象ブランチ: `feature/mmto-depth-stable-filter`
- 目的: depth3教師の浅さ由来ノイズを減らすため、depth3/depth4で安定したroot教師局面だけを抽出できるか検証する。

## 結論

`mmto_stability_filter` の実装は成功したが、今回の小規模データでは学習へ進めるだけのstable局面数は得られなかった。

strict条件では以下だった。

```text
train: 4 / 108 = 3.70%
valid: 0 / 12 = 0.00%
```

soft条件やbest mismatch許容を加えても最大は以下に留まった。

```text
train: 19 / 108 = 17.59%
valid: 1 / 12 = 8.33%
```

このため、現時点ではdepth3/depth4安定フィルタ済みデータでの学習へ進まない。

## 実装内容

追加:

```text
src/bin/mmto_stability_filter.rs
```

機能:

- depth3/depth4の `kpp_rank_v1` JSONLをSFENで照合する。
- depth4 recordを教師本体として、stable/unstableへ振り分ける。
- 出力JSONLは元のdepth4行をそのまま保持する。
- reject理由、ヒストグラム、分布統計をJSONで出力する。

主なCLI:

```text
--depth3
--depth4
--output-stable
--output-unstable
--stats-output
--key-mode canonical|full-sfen
--keep-duplicates
--require-best-match
--max-d4-best-rank-in-d3
--max-d3-best-regret-in-d4-cp
--allow-best-mismatch
--min-d4-gap-cp
--min-d3-gap-cp
--min-d4-topk-span-cp
--min-common-candidates
--min-pairwise-agreement
```

既定値ではstrictに近い挙動を維持する。`--allow-best-mismatch` は診断用であり、学習用stableデータを作る主設定にはしない。

## 検証データ

run directory:

```text
data/mmto/runs/depth_stability_smoke_20260622_154700
```

dump条件:

```text
input: converted_records2016_10818.sfen
max-positions: 120
teacher-score-top: 64
teacher-score-source: searched
valid-percent: 10
score-all-legal-for-valid
exclude-in-check
max-abs-root-score: 3000
seed: 9101
```

dump件数:

```text
d3 train: 108
d3 valid: 12
d4 train: 108
d4 valid: 12
```

## Strict結果

| split | stable | d4 records | pass rate | 主なreject |
|---|---:|---:|---:|---|
| train | 4 | 108 | 3.70% | best_mismatch 83, topk_span_low 19 |
| valid | 0 | 12 | 0.00% | best_mismatch 11, topk_span_low 1 |

depth4 bestのdepth3 rank:

```text
train: rank0=25, rank1-3=18, rank4-8=20, rank>8=45
valid: rank0=1, rank1-3=0, rank4-8=1, rank>8=10
```

depth4 bestがdepth3 bestと一致する局面は少なく、validではほぼ一致しなかった。

## Soft Sweep

既存runに対してフィルタ条件だけを変えた。

| config | train stable | train pass | valid stable | valid pass | 主な観察 |
|---|---:|---:|---:|---:|---|
| A strict | 4/108 | 3.70% | 0/12 | 0.00% | best_mismatchが支配的 |
| B soft_best | 4/108 | 3.70% | 0/12 | 0.00% | best条件緩和だけでは増えない |
| C soft_span0 | 9/108 | 8.33% | 1/12 | 8.33% | span条件も強く効いている |
| D soft_gap0_span0 | 16/108 | 14.81% | 1/12 | 8.33% | trainは増えるがvalidは弱い |
| E top8_regret50_span0 | 9/108 | 8.33% | 1/12 | 8.33% | top8/regret緩和でも不十分 |
| F allow_best_mismatch | 19/108 | 17.59% | 1/12 | 8.33% | best不一致を許しても20%未満 |

`--allow-best-mismatch` でも20%に届かないため、best一致だけが問題ではない。

## 解釈

1. depth3とdepth4のroot bestは、このデータではかなり入れ替わる。
2. depth3を「安定性判定」として使うには、strict条件ではデータがほぼ残らない。
3. soft条件でもvalid pass rateが低く、学習用データを作るには不足している。
4. root_deltaは比較的小さいため、評価値全体は近くても候補手順位は不安定である。
5. topK spanやgapが低い局面が多く、浅いroot探索scoreは同等候補の揺れを多く含む可能性がある。

## 判断

今回のdepth安定フィルタは、診断基盤として残す。

ただし、このrunからstable学習データを作って `mmto_train` へ進めることはしない。データ数とvalid再現性が足りない。

次に試すなら、以下の順で行う。

1. depth4/depth5の小規模比較にする。depth3は浅すぎる可能性が高い。
2. best一致ではなく、depth4 bestのdepth5 regretが小さいかで見る。
3. rootの単一bestではなく、soft teacher分布のKLやtopK overlapを安定性指標にする。
4. 対局分布からhard validを別に作り、train dump由来validだけで判断しない。

## 検証コマンド

通過:

```text
env RUST_FONTCONFIG_DLOPEN=1 cargo fmt --check
env RUST_FONTCONFIG_DLOPEN=1 cargo check --bin mmto_stability_filter
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

