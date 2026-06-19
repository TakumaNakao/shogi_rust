# root rescue 分類と高速化プロファイル結果

## 目的

`v2.4.1` に対して勝率を伸ばすため、終盤の敗局が root で救えるものかを分類し、並行して基礎アルゴリズム高速化の計測基盤を整備した。

## 追加した分析基盤

`root_rescue_probe` を追加した。

このツールは、new が負けた棋譜の終盤で new 手番の局面を走査し、実戦手と同じ局面での root 探索最善手を比較する。

strong root-rescuable 条件:

```text
best_move が存在する
actual_score <= -1000 または -inf
best_score > -1000
best_score - actual_score >= 300
```

これにより、単なる静的評価 mismatch ではなく、「実戦手は強制負けだが root には救済手があった」局面を抽出できる。

## root rescue 結果

### qdelta 100局

対象:

```text
/tmp/shogi_bench_records_qdelta_vs_v241_5201_100
tail-plies: 12
depth: 4
```

結果:

```text
probed positions: 306
same as root best: 182
candidate positions: 306
actual forced-loss moves: 55
strong root-rescuable candidates: 3
```

代表例:

```text
game_032 ply189 actual=6g6b+ score=-inf
best=N*5e best_score=342.1

game_050 ply83 actual=3b2b score=-inf
best=P*5b best_score=-414.4

game_068 ply121 actual=6a7b score=-inf
best=P*2d best_score=-778.9
```

### nullmove 40局

対象:

```text
/tmp/shogi_bench_records_nullmove_vs_v241_5401_40
tail-plies: 12
depth: 4
```

結果:

```text
probed positions: 126
strong root-rescuable candidates: 1
```

代表例:

```text
game_040 ply195 actual=P*1f score=-inf
best=L*1d best_score=-43.8
```

### forced-bestmove 20局

対象:

```text
/tmp/shogi_bench_records_forced_bestmove_vs_v241_5501_20
tail-plies: 12
depth: 4
```

結果:

```text
probed positions: 72
strong root-rescuable candidates: 2
```

代表例:

```text
game_006 ply105 actual=G*1c score=-inf
best=P*2b best_score=-757.6

game_007 ply74 actual=B*1b score=-inf
best=P*6g best_score=408.5
```

## root verification 実験

GPT-5.5 xhigh 相当の判断では、3系統すべてで strong root-rescuable が出ているため root 限定の上位手再評価へ進む価値がある、という結論だった。

そのため `exp/root-tail-verify-top4-d1` を実装した。

初期仕様:

```text
root の最終depthのみ
best_score <= -700 で発動
root上位4手を depth+1 で再評価
候補が現在best + 200以上、かつ > -1000 なら差し替え
```

結果:

```text
taya36 search_profile depth5 seed9501:
  root verify attempts: 0

strong root-rescue 3局面:
  root verify attempts: 1
  root verify candidates: 4
  root verify switches: 0
```

判断:

```text
採用しない。
```

理由:

- 通常ベンチ局面では発動しない。
- strong候補局面でも差し替えが起きなかった。
- `root_rescue_probe` の strong 候補は「過去の実戦手」と「現行root best」の差であり、現行rootが既に救済手を選べている局面が多い。

次に root verification を再試行するなら、通常bestの危険検出ではなく「root探索中に上位候補の verified score を比較する」別設計が必要。

## 高速化プロファイル

`movegen_profile` を追加した。

現行 `master` 基準:

```text
positions: 81,920
legal moves: 5,136,700
elapsed: 242.35 ms
positions/sec: 338,017
moves/sec: 21,194,973

do/undo込み:
positions: 10,240
legal moves: 638,740
elapsed: 133.68 ms
positions/sec: 76,602
do-undo/sec: 4,778,208
```

`perf/cache-movegen-occupied` では、滑り駒生成中の `occupied_bitboard()` をローカルにキャッシュする実験を行った。

結果:

```text
perft depth4: 719731 で一致
movegen単体: ノイズ範囲、通常生成はやや悪化
search_profile depth5 seed9501: 33.7s まで悪化
```

判断:

```text
採用しない。
```

見た目には再計算削減でも、レジスタ圧や最適化の都合で探索全体では悪化した可能性が高い。

## 次の方針

短期の探索変更は、これ以上 root verification を雑に入れない。

次に進める候補:

1. `root_rescue_probe` の分類を使い、実戦手がなぜ root best から外れたかをさらに追う。
   - 時間切れ
   - aspiration fail
   - TT影響
   - PVS narrow search
   - 探索中断

2. 基礎高速化では、見た目の小最適化を直接採用せず、`movegen_profile` と `search_profile` の両方で改善するものだけ採用する。

3. 調査計画に沿って、次は「詰み探索・王手回避・SEE精密化」のうち、root rescue で現れた強制負け局面に一番近い候補を GPT-5.5 xhigh に再選定させる。

現時点では、速度改善だけでも探索変更だけでも単独では伸びにくい。分類基盤で敗因を絞りつつ、低リスクな高速化だけを積み上げる方針を継続する。
