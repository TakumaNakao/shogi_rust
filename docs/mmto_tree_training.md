# MMTO tree訓練運用手順

この文書は `mmto_tree_dump` / `mmto_tree_train` / `mmto_score_gate` の想定運用を整理する。
目標は「学習後に現実の局面評価値が壊れていないか」を検査し、安定してから対局gateへ進めること。

## 1. 全体設計

1. `mmto_tree_dump` で木データを収集する  
   - 根局面、候補手、教師深さ、スコア、選択情報をJSONL化。
2. `mmto_tree_train` で重み更新する  
   - Trackごとに別の初期化方式で学習を回す。
3. `mmto_score_gate` でscore-space deltaを検査する  
   - baseline/candidateを同一局面集合で評価し、`abs(candidate-baseline)` 分布を集計。
4. gate通過後、対局で実性能を確認する  
   - まずsmoke、次に拡大ゲートへ。

## 2. Trackの扱い

### Track A（v2.1.0初期値）
- 基準重みとして `policy_weights_v2.1.0.binary` を使用する。  
- 新規実験は基本的にこの重みを出発点にする。  
- `mmto_score_gate` の比較はこのTrackを基準に取る。

### Track B（ゼロ/駒価値prior）
- 学習初期化を変えて崩れにくさを検証する。  
- KPP重みをゼロ寄りにしつつ、`material_coeff` は有効ならpriorとして扱う。  
- Track A/Bを同時に同条件で回し、どちらが安定性と実戦成果を満たすかを採点する。

## 3. `mmto_score_gate` の設計思想

`mmto_score_gate` は以下を行う。

- `--baseline-weights`, `--candidate-weights`
- `--input`（複数ファイル）
  - SFEN / USI 1行形式を読み込む
- `--seed` でshuffle
- `--max-positions` でtruncate
- `SparseModel` で両者の評価を比較
- `abs_delta` 分布を計算  
  `mean / p50 / p90 / p95 / p99 / max`

失敗条件:

- `p95` > `--p95-limit-cp`（既定50）
- `max` > `--max-limit-cp`（既定200）

どちらかを超えると終了コード2で失敗する。
`--mean-limit-cp` と `--fail-on-material-drift-cp` は任意追加条件として扱う。
`--json-output` を指定した場合、以下をJSONで保存する。

- summary
- worst positions（絶対差が大きい上位）

### material係数差の扱い

`SparseModel.material_coeff` が利用可能なら、baseline/candidate差分を要約に含める。  
材料項は `material_term_delta_cp` の形式で参考統計（mean/p95/max）を持つ。

## 4. 最小smoke手順（最小）

以下は、まだCargo登録が未反映でも名前上のコマンドとしての想定である。

```bash
# 1) 木データ収集（例）
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin mmto_tree_dump
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_dump \
  --student-weights policy_weights_v2.1.0.binary \
  --input taya36.sfen \
  --train-output /tmp/mmto_tree/train.rank.jsonl \
  --valid-output /tmp/mmto_tree/valid.rank.jsonl \
  --max-positions 200 \
  --seed 7101

# 2) 学習
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin mmto_tree_train
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_train \
  --weights policy_weights_v2.1.0.binary \
  --train /tmp/mmto_tree/train.rank.jsonl \
  --valid /tmp/mmto_tree/valid.rank.jsonl \
  --output /tmp/mmto_tree/candidate.binary \
  --epochs 1

# 3) score-space deltaゲート
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin mmto_score_gate
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_score_gate \
  --baseline-weights policy_weights_v2.1.0.binary \
  --candidate-weights /tmp/mmto_tree/candidate.binary \
  --input taya36.sfen \
  --max-positions 200 \
  --seed 7201 \
  --p95-limit-cp 50 \
  --max-limit-cp 200 \
  --json-output /tmp/mmto_tree/mmto_score_gate.json
```

## 5. Offline gate

- まずは `mmto_score_gate` が通過していることを最優先で確認する。
- 基本条件:
  - `p95<=50`
  - `max<=200`
- 追加条件（任意）:
  - `mean` が許容値以下（`--mean-limit-cp`）
  - `material_coeff` ドリフトが許容値以下（`--fail-on-material-drift-cp`）

`--json-output` には worst上位の局面を残し、どの局面で崩れが出ているか追える状態にする。

## 6. 対局gate

offline gateを通過しても、すぐ採用せず対局検証で再確認する。

- まず20局程度のsmoke
- その後に拡張（例: 100局）

重要:

- **20局だけで採用しない**  
- 100局では必ず seedを複数回回して再現性を確認する

推奨:

- 20局: 粗い破綻検知（明らかな不安定性・バグを除外）
- 100局: 複数seedで同傾向が出るか確認（1 seed依存を避ける）

## 7. 採用判断の最低線

- offline gate: `mmto_score_gate` を通過
- `mmto_score_gate` が通過していても、20局で良好でも採用扱いしない
- 100局で複数seedで有意に悪くならないこと
- 追跡対象は Track A/Track B それぞれで比較し、偶発崩れ（seed依存）を避ける
