# HalfKP scratch 学習方針 v4 — 教師更新と独立棋譜による価値学習

- 決定日: 2026-09-09
- 状態: T01–T12を完了。T12ではv2.5.0超えを確認できず、T13の総train 2M拡張を自動実行中
- 調査対象commit: `afe766b`、lineage: `halfkp64-scratch-v1`
- accepted親: gen6、SHA `162e17bacea3de80ecdbbdae765ca1157246c4febb6fa6a15f8d36b900d011dc`
- 本書が[旧v3](halfkp_training_strategy_v3_20260830.md)の後続作業を置き換える。
- 実行順・完了条件・進捗の詳細: [v4作業タスク](halfkp_training_v4_tasks.md)。次の作業はT13（gen6教師による総train 2M拡張）。

## 1. 決定

次の本線は、**accepted scratch親で棋譜局面を再解析し、深い探索値と実際の終局結果を混合する価値学習**とする。
gen5で教師を更新し、未使用の独立棋譜を加えて長時間学習する。改善した親で再び教師を作り直す。
同じ古い教師の行数だけを増やす実験、MMTO-onlyの追加outer、無条件の学習率引き上げは本線にしない。

飛躍のために優先するのは、良い勾配が実際に流れる実装、更新される教師、独立した局面分布、十分な更新量の組合せである。
新しいoptimizerへ交換すれば必ず勝てる、時間を増やせば必ずv2.5.0を超える、とは現在の証拠から言えない。
ただし正しい価値学習には二段階の親改善実績があり、ここを伸ばす方が、未検証の大規模な探索・network刷新より根拠が強い。

自作探索を教師にするのは、自分の静的評価をそのまま複写することとは異なる。先の手順・駒の取り合い・終局を探索した結果を
短時間の評価へ取り込めるため、外部評価なしでも改善する余地がある。ただし探索にも親の偏りが残るので、
実際の棋譜結果と独立データを併用し、対局で改善を確認した親だけを次の教師にする。

外部AIの重み・評価値・解析PV・探索visit分布を教師として使わない。WDoorの実際の指し手・盤面・終局結果・出典・rateは利用する。
CSA中の評価コメント等は学習targetにも品質選別にも使わず、解析は本repoのaccepted scratchモデルだけで行う。
v2.5.0/KPP由来artifactは初期値、教師、anchor、optimizer、自己対局生成に一切使わず、凍結後の比較対局専用とする。
自己対局は、ユーザーの希望どおり棋譜利用の拡張を試した後の最終手段に据え置く。

## 2. 実験結果を読み直す

### 判断に使える主要結果

今回は下表の対局reportを実際に読み、候補・親SHA、局数、score、paired CIを確認した。gen4/gen5はregistryのreport SHAとも一致した。
番号を再利用した旧「gen4～6」と混同せず、モデルはSHAで識別する。

| 実験 | 親に対する結果 | 解釈 |
|---|---|---|
| preserved-target 500k、`a26bb5…` | 2,000局、56.20%、paired 95% CI 54.025～58.375% | accepted gen4。価値学習の改善実績 |
| 通称1.18M、`61c05f…` | 2,000局、56.45%、CI 54.275～58.625% | accepted gen5。さらに親を改善 |
| 通常2M、`521d35…` vs gen5 | 2,000局、50.125%、CI 48.0～52.275% | 優位性未確認。rate等も変わり純粋な量の比較ではない |
| 高rate 2M、`6510e7…` vs gen5 | 2,000局、49.875%、CI 47.775～51.975% | 同じ旧親からの増量で優位性未確認 |
| Phase A 500k×4、`9a1d47…` vs gen5 | 500局、246勝245敗9分、50.1%、CI 45.7～54.5%、安全違反0 | 今回の候補は不採用。小改善も劣化も否定できず、「同等の証明」ではない |
| gen5 vs v2.5.0 | taya36・両者5秒・50局、14勝36敗、28% | この条件で大差が残る。releaseしない |

対局の正本:

- [500k / gen4 Gate](../runs/halfkp64-scratch-v1/phase3-500k-preserved-target-qleaf-20260824/match-20k-1000g-finalreserve-20260824/report.json)
- [1.18M / gen5 Gate](../runs/halfkp64-scratch-v1/phase3-2m-preserved-target-20260825/match-20k-1000g-gen5-reserve-20260825/report.json)
- [通常2M Gate](../runs/halfkp64-scratch-v1/phase3-2m-wide-20260826/match-20k-1000g-vs-1p18m-20260827/report.json)
- [高rate 2M Gate](../runs/halfkp64-scratch-v1/phase3-2m-highrate-20260829/match-20k-1000g-vs-1p18m-20260829/report.json)
- [今回のPhase A Gate](../runs/halfkp64-scratch-v1/phaseA-mmto-500k-gen5-20260830/candidate-outer4/match-20k-500g/report.json)、report SHA `e582cac325b6dd2fc677a0deec425258cc41f2ef3ff67ee828c7c55a004d8785`
- [gen5対v2.5.0](../runs/halfkp64-scratch-v1/phase3-1p18m-v250-taya36-5s-50g-20260828/report.json)

### 「学習量」と「loss」の補正

1.18Mは生データ全split合計の呼称で、実学習は1,038,880レコード×16 epochだった。
高rate 2Mの実学習は1,727,656レコード×10 epoch。約16.62M対17.28M提示であり、総更新量は大幅には増えていない。
生trainの棋譜数も58,396→54,699で、局面の多様性が増えたとは言い切れない。これらの数値は各runの
`raw/manifest.json`と`model-2m-100k.bin.manifest.json`による。レコード数と重複除去後の入力数も今後は区別する。

両者とも初期値・探索教師はgen4 `a26bb5…`だった。したがって高rate 2Mの不採用は、
**gen5を教師にして次世代を作る方式の失敗ではない**。また、splitや学習停止epochの違いがあり、
「1.18Mで棋譜学習は飽和した」という一般化は撤回する。

今回のPhase Aは12,460 train棋譜の500k rootを4回使い、194,166,132 pairを更新した。
各outerの`elapsed_ms`合計は約234.05時間（約9.75日）。outer4単独の約59時間を全run時間と混同しない。

| 今回の指標 | 初期またはouter1 | outer4 |
|---|---:|---:|
| fresh-valid pair loss | outer1: 0.14815655 | 0.14327998 |
| fresh-valid棋譜手一致率 | outer1: 23.086% | 23.840% |
| 同じvalid rootの勝敗Brier、固定温度 | 学習前: 0.21706476 | 0.21846035（悪化） |
| 勾配が一定以上あるpairの割合 | outer1: 30.976% | 29.571% |

根拠は[outer metrics](../runs/halfkp64-scratch-v1/phaseA-mmto-500k-gen5-20260830/metrics/outer_0004.json)と
[subset記録](../runs/halfkp64-scratch-v1/phaseA-mmto-500k-gen5-20260830/subset-policy.json)。
pair lossは合法手pair数による平均であり、旧v3にある「game-equal valid」という説明は実装と一致しない。
勝敗Brierの悪化はこのsubset・固定温度での観測であり、独立棋譜全体での有意な悪化や棋力低下を証明したものではない。
一方、順位目的だけを改善しても絶対価値が良くなる保証はないことを示す診断材料にはなる。

過去の`0.009478`等は**探索値と勝敗を混ぜたsoft targetに対する二乗誤差**である。
真の終局結果に対するBrierとは別物で、gen5のselected validでは後者は約0.21135。
学習時のloss powerは2.5、検証のこの指標は2乗である。意図した設計差として記録し、数値を混同しない。
候補1個だけのvalueデータでtop-1=100%でも、合法手全体での指し手精度を意味しない。

## 3. 実装不具合と、方式の限界を分ける

| 記録・経路 | 今回の扱い |
|---|---|
| 古い無勾配output clamp、誤datasetのall-legal CE | 交絡・条件違反。方式を棄却する根拠から除外 |
| 旧WDL auditのcpをさらに600倍した処理 | 当時のWDL棄却は無効。修正済みで、今回scale testも合格 |
| 旧qsearch-leaf / TreeStrap-lite | leaf未使用、root値置換、候補weight=0等があり、名前どおりの方式を検証していない |
| 旧gen4～6の緩い昇格判定 | 当時の対局scoreは観測として残すが、昇格を棋力改善の証明に使わない |
| `halfkp_search_train::add_rank_pair` | **現在も符号不具合を再現**。可変長leafで使ったrank実験から一般的な否定結論を出さない |
| 現行preserved-target value / 専用MMTO | 関連テスト合格。今回のrank不具合を理由に、別経路の実験まで無効化しない |

実行した検証（学習・コード修正はしていない）:

```bash
RUST_FONTCONFIG_DLOPEN=1 cargo test --locked --features halfkp64,training-tools --lib training_tools::halfkp_search_train::tests
RUST_FONTCONFIG_DLOPEN=1 cargo test --locked --features halfkp64,training-tools --lib training_tools::halfkp_mmto_train::tests
```

前者は21合格・1失敗。`game_rank_gradient_matches_finite_difference_for_both_sides`で
解析勾配`+0.05523079`に対して数値微分`-0.055283304`となった。後者は8件すべて合格。
原因はutilityの計算がroot/leaf手番に対応する一方、`add_rank_pair`の勾配が一律`utility=-child_eval`のままであること。
次作業で、good側は`-derivative * good_sign`、bad側は`+derivative * bad_sign`をcp/raw換算して適用する。
good/badそれぞれの手番、偶数・奇数leafを独立にテストする。単に既存testを変更して通してはならない。

value経路の手番反転、有限差分、ゼロ更新、checkpoint再開は合格した。専用MMTOは各leafに`root_sign`を持つ別実装である。
これは全探索・全データに不具合がないという保証ではない。修正後も価値学習では全rank weight=0を維持する。
過去の不成立理由の詳細は[7/29監査](halfkp_prior_art_recovery_plan_20260729.md)、
[8/20監査](halfkp_v250_breakthrough_plan_20260820.md)、[8/23監査](halfkp_v250_breakthrough_plan_20260823.md)を参照。

## 4. 外部調査から採用すること

一次資料を2026-09-09に確認した。以下は方法の参考であり、重み・評価値データを輸入する根拠ではない。

| 一次資料 | 確認できること | このrepoへの判断 |
|---|---|---|
| [Hoki・Kaneko / MMTO、2014](https://www.jair.org/index.php/jair/article/download/10871/25935/) | 棋譜手と代替手の探索末端を比較。約447万局面、PV再生成・再利用、尺度制約・正則化を併用。強い棋譜の効果も検証 | 棋譜だけで学ぶ方式は有効。ただし原法の32回のPV再利用・多数反復と今回の4 passは同等でなく、CPU時間だけで「十分学習した」とは言えない |
| [Stockfish NNUE公式解説](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html#loss-functions-and-how-to-apply-them) | 探索値を確率へ変換し、実結果と混合する。尺度はengine/data依存 | 自作scratch探索値＋棋譜結果を使う。混合係数と確率校正を明示する |
| [やねうら王・評価関数の学習](https://github.com/yaneurao/YaneuraOu/wiki/%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%81%AE%E5%AD%A6%E7%BF%92) | 棋譜順の強い相関を避けるshuffle、qsearch末端への入力置換 | 学習入力と探索時の分布を揃える。ただし現在の「deep best手の後をqsearchする」変換は本repoの近似で、深い探索PVそのものではない |
| [Stockfish・feature factorization](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html#feature-factorization) | 学習時に共通特徴を持ち、疎な特徴間で更新を共有。出力時に重みへ合成できる。特に初期学習で有効 | 稀なHalfKP特徴の学習不足が測定された場合の第一候補。推論幅を増やさず試せるが効果は未検証 |
| [RubiChess開発者の学習記録](https://github.com/Matthies/NN) | 自エンジン生成の大量局面・継続学習を記録。幅拡大のfixed-node改善が実時間では小さくなった例もある | 大量学習は現実の成功例にある。ただし棋譜数・局面数・提示数・hardwareを揃えず時間だけを比較しない |
| [Venessほか / TreeStrap、2009](https://proceedings.neurips.cc/paper_files/paper/2009/file/389bc7bb1e1c2a5e7e147703232a88f6-Paper.pdf) | 自分の探索結果で価値を更新。alpha-betaのboundには片側lossを使う | 自作探索から新しい教師を得る方向を支持。内部node全取得は実装が大きく、次の本線にはしない |
| [Silverほか / AlphaZero、2017](https://arxiv.org/abs/1712.01815) | 外部教師なしの自己対局による棋力獲得 | 自己対局は可能性のある後段。CPUノートPCで同じ速度・成果を期待する根拠にはならない |

MMTOのsigmoidは大差の誤りでも勾配が小さくなるが、容易な正解pairを弱め、棋譜の誤りの影響を抑える面もある。
active pair約30%だけを見て「lossが致命的に悪い」「softplusにすれば解決する」とは判断しない。
原法を踏まえ、学習量不足の可能性は残す。それでも今回約234時間で親改善が見えず、価値学習に成功実績があるため、次の計算配分を変える。

## 5. 原因の優先順位

| 仮説 | 現時点の判断 | 次に区別する観測 |
|---|---|---|
| 実装のミス | rankは確定。旧実験にも複数あり、失敗回数の水増し要因 | 正負・手番・scale・resumeの関連testを学習前に通す |
| 教師・目的の限界 | 最優先。古いscratch探索値の固定と、順位lossだけの改善では新情報が弱い | gen5での再解析、実結果Brier、深い探索と浅い探索の差 |
| データ不足・偏り | 未棄却。高rate 2Mは棋譜数も提示総数も大幅増ではない | 新規game数、unique入力、局面のphase・手番・勝敗・特徴頻度 |
| 学習時間不足 | あり得るが、探索時間とoptimizer更新時間を分ける必要あり | 4/8/16/32 epochの固定valid曲線、更新norm、実際の提示数 |
| LR・AdaGrad | 二次要因。Phase Aはzero optimizer開始でも停滞。古い履歴だけでは説明できない | 更新前後のnorm・clip・活性・train/valid差。必要なら同条件でLRだけ比較 |
| HalfKP表現・疎性 | 容量不足の断定は早い。同一推論系のv2.5.0はより強い | validで使う未学習・低頻度feature、活性飽和と誤差の関係 |
| 評価の弱さ・探索相性 | gate不合格と方式無効は別。少数局・同じopeningの反復には限界 | 未使用opening、親以外のscratch参照、固定nodeと実時間評価 |

## 6. 次の作業手順

### A. 修正・短い診断

1. 上記rank符号を修正し、関連testを通す。外部評価コメントを書き換えても生成targetが変わらない回帰確認を加える。
2. 既存gen4教師に対応する固定5k rootを、gen5・同じ100k nodes/手、depth上限8で再解析する。
   同じroot手番で、completed depth、手の変化率、探索値の差、実結果Brier、生成時間を比較する。
   深い探索値は自己申告の正解ではなく、改善の可能性がある教師と扱う。
3. 同じ5kの一部でroot→qsearch入力置換の手番・score保持・exact種別を確認する。
   変換後leafは実棋譜から分岐する場合があり、実棋譜結果をそのleafへ付けるのは近似であることも記録する。
4. まず既存形式の意味を維持する。新教師の失敗・大きな分布異常が出た場合だけ原因を直す。
   5kで棋力や方式全体の合否は決めず、長期runの時間・保存容量を実測から見積もる。

### B. 次の本実験：gen5教師＋独立棋譜

- 親・初期値・探索教師はgen5。今回のMMTO候補は使わない。
- **新規棋譜から受理済み・重複除去後のtrain入力1M**を確保し、既存train由来の250k replayを加える（合計1.25M）。
  replayもgen5で再ラベルし、旧valid/test/Gateのgame・opening familyをtrainへ移さない。
- 未使用の2018～2026の収録済み棋譜を棚卸しし、不足分を2017から過去へ年単位で追加する。
  現在`data/wdoor/extract`にある年は2018～2026。過去年の在庫・高品質棋譜があるとは仮定しない。
  出典は[WDoor](https://wdoor.c.u-tokyo.ac.jp/shogi/)。一度に全期間をdownloadせず、受理数を見ながら進める。
- 当初は両者rate>=4000を維持。古い年に十分な棋譜がなければ、その不足を記録する。
  年代の異なるrateを同じ絶対棋力とはみなさず、低rate追加は別のデータ比較として扱う。件数達成のため黙って条件を変えない。
- 原則1棋譜24 root以下、両手番・序中終盤を含め、敗者の終盤を消さない。局面数だけでなく新規game数を保存する。
  phase比の不足で大量の利用可能データを捨てず、偏りを記録してshuffleする。
- 独立gameからvalid 50k、test 50kを別に凍結する。過去のholdout保護に加え、生成後のnetwork入力重複も除去する。
- 教師は既存`halfkp_search_teacher --value-only --qsearch-leaf`のsemantics v4を使い、
  completed root scoreを保存する。100k nodes/root、depth上限8、最低completed depth1を基準にする。
  depthの低い教師やmate・千日手等の除外率を報告し、未完了探索値を教師にしない。
- 学習は既存value trainer、search/result=`0.8/0.2`、kappa=`1352.7871`、loss power=`2.5`、
  batch128、AdaGrad、feature LR=`3e-4`、output LR=`3e-5`、全rank weight=0。
  新runのoptimizerを1回初期化し、その後は再開もepochも履歴を継承する。
- 4/8/16 epochを保存。16時点でも固定validの混合target誤差が改善し、実結果指標・parameter healthも悪化していなければ32まで続ける。
  2 epochの小さな一致率差で打ち切らない。明確なtrain/valid乖離、連続した悪化ならbestを保存して停止する。
- 親と同じ固定valid・同じ温度で混合target誤差、真の勝敗Brier/log-loss、phase別誤差を併記する。
  対局前にbest checkpointを1個だけ凍結し、testはその後一度だけ確認する。

これは「教師更新＋独立データ」という実用上の組合せを試す本実験である。成功しても、どちらの寄与かを個別に証明したとは書かない。
原因を分離する必要が生じた時だけ、同じroot・初期値・更新量で旧教師/新教師のA/Bを追加する。

### C. 改善したら拡張、しなければ原因に応じて一つだけ変える

合格した候補を新しい親にして教師を更新し、新規棋譜を加える。初回は新規1M＋replay 250k＝総train 1.25M。
次の規模は総train 2M→4M→10Mとし、各数字はreplayを含む受理済みunique train入力数とする。
epoch数・提示数・新規game数・replay内訳も必ず併記する。初回の候補が統計的に不確定でも、offline改善と安全性が揃えば、
**同じ親・設定で総train 2M（新規1.6M＋replay 0.4M）まで一度拡張**してよい。
これを昇格とは区別する。4M/10Mへの自動拡張は、未使用openingで親改善を確認した後にする。

十分な独立データでも改善しなければ、次の表から観測に合う一つを試す。全部を同時実装しない。

| 観測 | 次の変更 |
|---|---|
| train/validとも改善中、更新が有効 | まず学習継続・独立データ追加。新algorithmは不要 |
| trainだけ改善、valid悪化 | epoch追加を止め、gameの多様性・重複・局面分布を修正 |
| 新教師への回帰だけ改善、実結果への改善がない | 同じデータでsearch/resultを0.8/0.2と0.5/0.5で一度比較。係数以外は固定 |
| trainも停滞、勾配は正常だが更新が極小 | 固定subsetでLRを基準の1/3・1・3倍まで一度校正。optimizerとLRを同時変更しない |
| 稀な特徴への誤差集中・coverage不足 | 学習時だけP共通特徴を加えるfactorizationを一種類試す。合成前後の評価一致を必須とし、推論形式・幅は維持 |

順位学習の再挑戦は符号修正だけで自動開始しない。再開するなら、棋譜手と代替手の両leafに正しい勾配を流し、
棋譜手がscratch探索のbestと違うという理由だけで教師から除外しない。正規化した価値lossとの寄与を測る単一実験として別途記録する。
policy head、MCTS、network幅拡大、真TreeStrapは現段階では追加しない。

### D. 自己対局へ進む条件

独立棋譜を増やした価値学習と、診断に対応する上記の単一変更でも改善できず、固定validも停滞した場合に検討する。
単発500局の不合格や、古い教師の2M化だけでは「棋譜が飽和した」としない。
ここでいう飽和は、この資源・データ・方式での実務上の判断であり、棋譜学習一般の限界ではない。

その場合もaccepted scratchだけで対局し、局面・自作探索値・本当の終局結果を保存して既存value trainerへ渡す。
目的は、自分が実戦で到達する局面と失敗を学ぶことである。棋譜replayを残し、外部AIを対戦教師にしない。

## 7. 評価・長時間実行・進捗

- 開発screenは20k nodes/手・未使用250 opening pair＝500局。既存の繰返し使用済みMMTO dev suiteは最終確認に流用しない。
- 有望候補だけ、別の未使用250 pairで確認する。昇格は確認側単独でscore>=52.5%、paired 95%下限>50%、安全違反0。
  screenを見て選んだ後に両者を合算したCIを、無条件の95%保証として扱わない。4,000局を既定にしない。
- 不合格は候補不採用であって、方式の完全否定ではない。CIが50%をまたぐ結果を「下回らなかった」とも表現しない。
- accepted候補だけをtaya36・両者5秒・50局でv2.5.0と比較する。このscreenだけで小幅な優位性やreleaseを確定しない。
  固定nodeの親改善と実時間のv2.5.0超えは別に評価する。
- 次runは12 worker。12 worker＝12物理coreではなく、CPU 8 core / 16 thread上の並列数であり、1.5倍の速度は保証しない。
- 既存の50k root shard、checkpoint、モデル/optimizer/RNG保存、manifest SHA、端末切断に耐える起動を再利用する。
  新しい総合管理frameworkは作らない。生成・学習・検証・対局の所要時間を分けて保存する。
- non-finite、非合法、壊れたartifactは即停止。feature/output deltaの旧目安`0.02/0.004`は数学的な棋力保証ではない。
  超えた候補を自動採用せず診断するが、それだけでalgorithmを棄却しない。結果に合わせた黙示の上限変更はしない。

| 作業 | 状態 | 次の完了条件 |
|---|---|---|
| Phase A集計・旧実験再分類・一次資料調査 | 完了 | 本書、report SHA、テスト結果を記録 |
| rank符号修正・関連test | 完了（2026-09-10） | 同手番/逆手番の有限差分とvalue無変更を確認済み。T02のcommit記録を追記 |
| 5k教師更新・意味/時間診断 | 完了（2026-09-10） | gen4/gen5比較、target出典・符号・所要時間を固定。再ラベルを継続 |
| v4再開runner・checkpoint・勝敗指標ログ | 実装完了（2026-09-10） | T07データ確定後にfixture、本番4/8/16 epochを実行 |
| WDoor在庫・旧split除外capacity | 完了（2026-09-10） | 2018–2026は独立80,369 root。2017高rateは1,362、2016高rateは0。2017 rateメタデータなし分岐は独立1Mを確認 |
| rateメタデータなし分岐の小規模診断 | 完了（2026-09-10） | gen5教師の結果較正・失敗率・phase/勝敗分布を記録し、条件付き採用 |
| 独立1M＋replay 250k構築・gen5再解析 | 完了（2026-09-10） | dedup後train1,425,255（新規1,152,203/replay273,052）、valid54,997、test55,275。final-manifest固定 |
| 16 epoch、条件付き32 epoch | 完了（2026-09-10、epoch24停止） | valid改善・health正常、report固定 |
| 候補選定・凍結・test | 完了（2026-09-10、epoch24） | 候補SHA `162e17ba…`、test report固定 |
| 親screen・独立確認 | 完了（T10/T11） | screen 55.7%、確認 56.2%、安全合格、gen6昇格 |
| v2.5.0比較・結果記録 | 完了（T12） | 候補17勝33敗、34.0%。release保留。report固定 |
| generation6教師・総train 2M拡張 | 実行中（T13） | 新規1.8M＋replay450k、dedup後2M以上、4/8/16/32 epoch checkpoint |
| 4M/10M・次世代教師更新 | 条件付き | T13で親改善を独立確認した場合のみ規模を再決定 |
| 自己対局 | 最終手段・保留 | 上記Dの診断後に判断 |

2026-09-11時点でT10親screen（55.7%）とT11独立確認（56.2%、paired CI 52.0–60.4%、安全違反0）を完了し、候補をgeneration 6としてregistryへ昇格した。T12のv2.5.0比較（taya36・両者5秒・50局）は候補17勝33敗で、v2.5.0超えを確認できずreleaseを保留した。現在はT13としてgeneration6を唯一の教師にした総train 2M拡張を、shard/checkpoint単位で再開可能なスクリプトにより自動実行している。T12比較はrelease判断専用であり、学習教師へは使わない。
