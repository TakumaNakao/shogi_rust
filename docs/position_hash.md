# 局面ハッシュモジュール (`position_hash.rs`)

このモジュールは、各局面に一意のハッシュ値を生成する手法であるZobristハッシュを実装しています。これは、以前に分析された局面を検索するためにハッシュ値を使用するトランスポジションテーブルにとって不可欠です。

## 主な機能

*   **Zobristハッシュ:** 局面ハッシュを生成するための非常に効率的な方法です。ハッシュは、指し手が実行されたり取り消されたりするたびに増分更新されるため、毎回ハッシュを最初から再計算するよりもはるかに高速です。
*   **事前計算されたランダムキー:** 起動時にランダムな64ビットキーのセットが生成されます。これらのキーはハッシュ値の計算に使用されます。

## 主要な構造体

*   `ZobristKeys`: 事前計算されたランダムキーを保持する構造体です。
*   `PositionHasher`: `calculate_hash` メソッドを提供する構造体です。

## 主要な関数

*   `calculate_hash`: 特定の局面のZobristハッシュを計算します。