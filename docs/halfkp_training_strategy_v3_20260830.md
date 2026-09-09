# HalfKP scratch 学習方針 v3

- 制定日: 2026-08-30
- 対象lineage: `halfkp64-scratch-v1`
- 現在のaccepted head: generation 5、SHA-256
  `61c05fe287cb0c0f9b434194802e07c835476b737a37bb9d4063143c63e28637`
- 目的: v2.5.0/KPP由来重みを学習へ一切使わず、CPU-onlyでv2.5.0を上回るHalfKP重みを作る
- 前計画: [`halfkp_v250_breakthrough_plan_20260823.md`](halfkp_v250_breakthrough_plan_20260823.md)

## 1. 結論

次の一手は、accepted generation 5から、現在動作しているPV再生成MMTOを**過去の成功実験に近い実効量で一度だけ**
実行する。その成否にかかわらず、その次はgeneration型のscratch-only自己対局へ移る。

順序は次のとおりとする。

1. generation 5から500k unique root・4 outerのPV再生成MMTOを行う。
2. 安全なcheckpointをoffline指標だけで1個に凍結し、親対局を1回だけ行う。
3. Gate合格ならgeneration 6、不合格ならgeneration 5を親として、世代型自己対局学習を開始する。
4. 自己対局は1M fresh root/世代を基本とし、acceptedモデルだけで次世代教師を作る。

この順序にした理由は、10k・1 outerの最新MMTOが候補としては不合格だった一方、fresh-validのpair lossと安全性は
改善しており、過去に同じMMTO系列がscratch親を約80 Elo改善した実績があるためである。新規実装がほぼ不要で、
最短の再現性ある賭けになる。ただし固定棋譜分布を無期限に反復せず、この十分量の試行を最後に自己対局へ切り替える。

## 2. 実験から確定したこと

### 2.1 正本と主要結果

| 実験 | offline | 親対局 | 判断 |
|---|---:|---:|---|
| preserved 25k | Brier 0.012328→0.011882 | 未実施 | semantics smoke合格 |
| preserved 100k | 0.012325→0.011459 | 50.8%、paired CI 46.6～55.0% | 小規模では不確定 |
| preserved 500k | 0.012341→0.011109 | 56.2%、CI 54.025～58.375%（2,000局） | strict合格、gen4 |
| preserved 1.18M | 0.009943→0.009478 | 56.45%、CI 54.275～58.625%（2,000局） | strict合格、gen5 |
| 通常2M | valid 0.009667 | gen5へ50.125%、CI 48.0～52.275% | 不採用 |
| rate>=4000 2M | epoch 10で停止 | gen5へ49.875%、CI 47.775～51.975% | 不採用 |
| game-rank 22k | Brier +6.73%、objective +1.21%悪化 | 未実施 | 現実装経路を停止 |
| PV-MMTO 10k・1 outer | pair loss差 -0.001997、CI全域<0 | gen3へ49.8%、CI 45.6～54.0% | 候補不採用、量の判定は未了 |
| gen5 vs v2.5.0 | — | 14-36、28%（taya36、5秒、50局） | release不可 |

現在の正本はgen5 `61c05f...`である。通常2M `521d35...`、高rate 2M `6510e7...`、
game-rank候補、PV-MMTO 10k `2170cc...`はinit、teacher、自己対局親に使わない。

### 2.2 支持された仮説

- 正しいcompleted root search値をbest qsearch leafへ保持し、終局結果を混合するvalue経路は実際に強くなる。
  500kと1.18Mは別々の2,000局Gateで、95% CI下限54%以上の親改善を示した。
- 学習量は必要だが、同じ親・同じ固定分布では1.18M付近から行数だけの増加が飽和した。
  rateを下げた通常2Mだけでなく、rate>=4000を維持した2Mもgen5と互角だった。
- offline lossは必要な診断だが、単独では棋力の採用指標にならない。2M、game-rank、PV-MMTO 10kが反例である。
- 学習率上昇は情報不足を解決しない。過去のMMTO `3e-4`ではfeature deltaが安全上限を超えており、次の主変数にしない。

### 2.3 棄却していない仮説

- **十分量のPV再生成MMTO**: 最新10kは937,849 pair・1 outerだった。過去に成功したfull MMTOは
  約2.28M root提示・224.1M pairであり、最新pilotはroot、pairとも約0.4%にすぎない。
  10k候補の不採用は正しいが、これでMMTO全体を棄却するのは早い。
- **on-policy自己対局**: 既存200局は同一モデル対局の安全smokeで、局面・探索値・終局結果を教師へ保存していない。
  HalfKPの世代型自己対局学習はまだ一度も行われていない。
- **新しい独立棋譜**: 2M化は主に同一棋譜内の抽出数を増やした。古いWDoorを新しいgame/openingとして使う効果は別仮説である。
- **表現容量**: 同一HalfKP推論系でv2.5.0が強いため、現段階では容量不足より教師と分布の問題を優先する。

## 3. コード監査の反映

value経路に、現在の結果を無効にする致命的な問題は見つからなかった。
`halfkp_search_teacher`は最後に完了したroot search値を保持し、trainerはroot/leafの手番に応じて探索値と勝敗を反転する。
HKST v4のmanifest、accepted親確認、train/validのnetwork-input非重複、checkpoint/resumeも利用できる。

一方、rank経路には潜在的な符号不具合がある。utility計算はroot/leaf手番を考慮するが、`add_rank_pair`の勾配符号は
全candidateが1手後のchildである旧前提に固定されている。同手番qsearch leafでは逆向きの勾配になり得る。
したがってgame-rank pilotは「現在の実装・設定が不合格」という証拠であって、順位学習一般を棄却する証拠にはしない。
ただし最短経路ではないため、rank weightはすべて0へ固定し、順位経路を再開する時まで修正を延期する。

HalfKP用の自己対局教師生成器は存在しない。`halfkp_promotion_match --self-play`は勝敗要約しか保存せず、
旧`kpp_self_learn`はSparse KPP専用である。自己対局段階では、既存HKST/trainerを変えず、生成器1本だけを追加する。

## 4. Phase A: meaningful-dose PV再生成MMTO

### 4.1 固定条件

- init/search snapshot: accepted gen5 `61c05f...`
- data: 2018～2025 WDoor、両者rate 4,000以上、game-disjoint train/valid
- train: 500,000 unique root、valid: 50,000 root
- 同一gameへの過密化を避け、既存のphase-balanced集合からgame順をshuffleした固定subsetをmanifest化する
- outer: 4、各outerで現モデルから全合法手のqsearch PV leafを再生成する
- LR: feature `1e-4`、output比 `0.1`
- objective: MMTO-only。WDL追加、LR grid、rank loss、architecture変更を同時に行わない
- optimizer: この新目的用のzero-state AdaGradを1回だけ作り、4 outer間とresumeで継承する
- 50k accepted rootごとにatomic checkpoint。既存run directoryは上書きしない

最新pilotからの単純外挿では約188M pair、約15～20日を見込む。これは保証値ではなく、進捗率と実測ETAをmanifest/logへ
更新する。過去full MMTOの約84%のpair量になり、10kのsemantic pilotとは異なり、棋力転移を判定できる量を狙う。

2026-08-30 22:11 JSTに、`scripts/run_halfkp_mmto_phaseA.sh`（commit `261c67f`）から
`runs/halfkp64-scratch-v1/phaseA-mmto-500k-gen5-20260830`を起動した。dataset manifest、gen5 model、
gen5 Gate reportのSHAを固定値と照合し、train 500,000 / valid 50,000 prefixの重複検査（各splitで完全一致root 0件）を
通過している。outer単位のresumeとfresh-valid loss 2回連続悪化時の停止を有効にしている。長期runの途中で候補を
採用せず、`metrics/outer_XXXX.json`と`checkpoint.json`が揃ったouterだけを監査対象とする。
実行中プロセスは起動時指定の8スレッドを維持する。次回のresumeまたは後続処理は、commit `e33a7e0`で既定値を
12スレッドへ変更して起動する。

2026-09-09、outer 1～4のうちfresh-valid lossが最小で安全違反0だったouter 4を
`candidate-outer4/model.bin`（SHA `9a1d477b…`）として凍結した。`scripts/run_halfkp_phaseA_gate.sh`
（commit `6ce960b`）が候補manifestとselectionを再生成不能な形で保存し、gen5親との250 opening-pair / 500局
Gateを起動している。Gateはpair単位で再開でき、候補はGate完了までaccepted registryへ追加しない。

### 4.2 監査・停止・Gate

- 各outerで同じfresh game-equal validのpair loss、agreement、regret、PV遷移、parameter healthを保存する。
- feature delta `>0.02`、output delta `>0.004`、non-finite、bound異常、artifact不一致で即停止する。
- fresh-valid lossが2 outer連続で悪化した場合も停止する。
- 全safe checkpointからfresh-valid pair loss最小の1個だけを、対局前に候補として凍結する。対局後の選び直しはしない。
- 親Gateは20k nodes/手・250 opening-pair（500局）。点推定52.5%以上でCIだけ不確定なら、未使用openingで
  合計1,000 pair（2,000局）まで延長する。
- 昇格は`score >= 52.5% && paired bootstrap 95%下限 > 50% && safety違反0`をすべて要求する。

不合格なら、同じ固定WDoor MMTOのroot数、outer、LRを後付け変更しない。候補を破棄してgen5から自己対局へ移る。
合格ならgeneration 6として凍結し、そのモデルから自己対局へ移る。いずれの場合もこのMMTOを連続世代化しない。

## 5. Phase B: 世代型scratch-only自己対局

### 5.1 最小実装

`halfkp_selfplay_teacher`を1本だけ追加する。league、policy head、MCTS、TDLeaf、探索揺らぎは実装しない。

- accepted scratch 1モデルを両者に使い、WDoor由来のdisjoint opening 1件につき1局だけ指す。
- 各手の最後に完了したroot search値、best move、exact qsearch leaf、leaf手番、completed depthをゲーム中にbufferする。
- 真の終局後にroot手番視点の0/0.5/1を付け、HKST v4へ保存する。
- max-ply終了は偽drawにせず`result=None`、search failure・illegal moveはゲーム全体を破棄する。
- source棋譜の指し手、winner、評価値は学習へ使わない。opening ballotとしてだけ使う。
- 25k root単位のatomic shard、sidecar game report、global network-input dedup、決定的resumeを持たせる。
- candidateはbest qsearch leaf 1件だけ、rank/game-rank/game-vs-search weightはすべて0へ固定する。

まず200 unique opening・200局で約20k rootのsemantic smokeを行う。これは符号、勝敗、千日手、重複、shard resume、
manifestを確認するだけで、対局による棋力判定は行わない。

### 5.2 最初の実効世代

親はPhase A合格時のgen6、不合格時は現在のgen5とする。

- 1,000,000 unique root/世代
- 内訳: fresh on-policy 80%、同じ親で再ラベルしたWDoor/search replay 20%
- actor/teacher: 100k nodes/手を基準とする。固定5k rootで20k/50k/100kのWDL差とbest-move安定性を一度比較し、
  十分なら小さいbudgetへ下げて総探索量を一定に保つ
- search/result=`0.8/0.2`、`kappa=1352.7871`、loss power `2.5`
- AdaGrad、feature LR `3e-4`、output LR `3e-5`
- 4/8 epochを保存し、validがまだ改善しparameter healthが正常な場合だけ16 epochまで進める
- game-disjoint self-play validとWDoor validを併記し、対局前に候補1個だけを凍結する

親対局のpromotion GateはPhase Aと同じにする。ただし**昇格条件**と**方式を一度だけ増量する条件**を分ける。

- score <50%、offline悪化、または安全違反: 候補不採用。その親での同方式を停止する。
- score 50～52.5%、offline改善、安全正常: 親は昇格させず、同じ親・同じ設定で1M fresh rootを一度だけ追加し、
  合計2M候補を再試験する。
- score >=52.5%でCIだけ不確定: 2,000局へ延長する。
- strict Gate合格: accepted次世代とし、新しいモデルで教師を全面更新する。

最大4 accepted世代を最初の区切りとし、同じaccepted親から十分量の候補が2回連続不合格なら自己対局を停止する。
同じ教師をepochだけ増やすより、accepted親からfresh rootを再生成することを優先する。

## 6. 古い棋譜と次のfallback

2008～2017等の古いWDoorを追加する場合は、次の用途に限定する。

1. 自己対局のopening ballotを増やす。
2. 20% replay用の独立gameを増やし、accepted scratch親で探索値を付け直す。

元棋譜の重み、評価値、v2.5.0/KPP由来artifactは使わない。同一game内の行数を4M/10Mへ増やすだけの実験は再開しない。

meaningful-dose MMTOと十分量の自己対局がともにaccepted親を2回改善できなかった場合だけ、次の順で進む。

1. rank parityの符号不具合を直し、100k rootの順位pilotをやり直す。
2. TTのexact/lower/upperを区別する真のTreeStrapを100k rootで比較する。
3. HalfKP feature coverage、幅、量子化、score scale、NPSをv2.5.0と同じ入力集合で監査する。

## 7. v2.5.0評価と禁止事項

- v2.5.0/KPP由来重みはinit、teacher、target、anchor、optimizer、自己対局相手、候補選択に使わない。
- v2.5.0はaccepted scratch候補を凍結した後の外部評価にだけ使い、結果を学習へ戻さない。
- taya36・両者5秒・50局screenで50%以下ならreleaseしない。より大きい最終対局は別途決定する。
- failed候補同士のブレンド、LR grid、同一openingの反復選択、対局結果を見てからのcheckpoint変更を行わない。
- 現時点でpolicy head、MCTS、大規模league、真TreeStrap、architecture拡大を同時導入しない。

## 8. 進捗表

| 作業 | 状態 | 完了条件 |
|---|---|---|
| 全実験・lineage再集計 | 完了 | accepted head=gen5、失敗候補と比較条件を確定 |
| value/rank/self-playコード監査 | 完了 | value経路健全、rank符号問題、自己対局生成器不足を確定 |
| 一次資料再調査 | 完了 | NNUE、MMTO、世代型自己対局、TreeStrapを照合 |
| PV-MMTO 10k semantic pilot | 完了・候補不採用 | offline改善、安全正常、親Gate 49.8% |
| Phase A 500k×4 MMTO | 完了（outer 4、loss最小） | safe候補1個を凍結し、親Gateまで完了 |
| Phase A候補凍結・親Gate | 実行中（250 pair） | strict Gate判定・必要ならgen6登録 |
| Phase B自己対局生成器 | 未着手 | 200 unique-game smoke、再開・重複・符号test合格 |
| Phase B最初の1M世代 | Phase A後 | 学習・valid・strict親Gate完了 |
| 世代更新 | 保留 | 最大4 accepted世代、または停止条件到達 |
| v2.5.0再評価・release | 保留 | accepted候補を外部評価しrelease条件を判定 |

## 参考一次資料

- [Hoki & Kaneko, Large-Scale Optimization for Evaluation Functions with Minimax Search](https://www.jair.org/index.php/jair/article/download/10871/25935)
- [Stockfish NNUE: loss functions and WDL/result mix](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#loss-functions-and-how-to-apply-them)
- [やねうら王: 評価関数の学習](https://github.com/yaneurao/YaneuraOu/wiki/%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%81%AE%E5%AD%A6%E7%BF%92)
- [Veness et al., Bootstrapping from Game Tree Search](https://proceedings.neurips.cc/paper_files/paper/2009/file/389bc7bb1e1c2a5e7e147703232a88f6-Paper.pdf)
