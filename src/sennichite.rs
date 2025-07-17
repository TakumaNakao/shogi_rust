use circular_buffer::CircularBuffer;
use shogi_core::{Color, Position};

// --- 千日手検出ロジック ---

/// 千日手の判定に使用する、手数を含まない局面情報。
/// shogi_core::PartialPosition には手数(ply)が含まれるため、
/// 同一局面の判定にはこの構造体を使用する。
#[derive(PartialEq, Eq, Clone)]
pub struct SennichiteKey {
    board: shogi_core::Bitboard,
    hands: [shogi_core::Hand; 2],
    side_to_move: shogi_core::Color,
}

impl From<&shogi_core::PartialPosition> for SennichiteKey {
    fn from(pos: &shogi_core::PartialPosition) -> Self {
        SennichiteKey {
            // PartialPositionから手数(ply)以外のフィールドをクローンする
            board: pos.occupied_bitboard(),
            hands: [pos.hand_of_a_player(Color::Black), pos.hand_of_a_player(Color::White)],
            side_to_move: pos.side_to_move(),
        }
    }
}


/// 千日手の状態を表す列挙型
#[derive(PartialEq, Debug, Eq, Clone, Copy)]
pub enum SennichiteStatus {
    /// 千日手ではない
    None,
    /// 千日手による引き分け（連続王手ではない場合）
    Draw,
    /// 連続王手による負け
    PerpetualCheckLoss,
}

/// 千日手検出器
/// 固定サイズのリングバッファを使用して、過去の局面ハッシュを管理します。
pub struct SennichiteDetector<const CAPACITY: usize> {
    /// 過去の局面ハッシュの履歴（固定サイズのリングバッファ）
    /// circular-bufferは容量に達すると古い要素を自動的に上書きします。
    history: CircularBuffer<CAPACITY, SennichiteKey>,
}

impl<const CAPACITY: usize> SennichiteDetector<CAPACITY> {
    /// 新しい千日手検出器を作成します。
    ///
    /// `CAPACITY`は、履歴に保持する局面の最大数です。
    pub fn new() -> Self {
        SennichiteDetector {
            history: CircularBuffer::new(), // const generics でサイズが指定される
        }
    }

    /// 現在の局面を履歴に記録し、出現回数を更新します。
    ///
    /// `shogi_core::Position`のハッシュ機能を使用します。
    pub fn record_position(&mut self, position: &Position) {
        // PositionからSennichiteKeyを生成して履歴に追加
        let key = SennichiteKey::from(position.inner());
        self.history.push_back(key);
    }

    /// 履歴から最も新しい局面ハッシュを削除し、出現回数を更新します。
    /// 探索中の仮の指し手を元に戻す際に使用します。
    pub fn unrecord_last_position(&mut self) {
        self.history.pop_back(); // 最も新しい要素を削除
    }

    /// 特定の局面の出現回数を取得します。
    pub fn get_position_count(&self, position: &Position) -> u32 {
        // PositionからSennichiteKeyを生成して比較
        let target_key = SennichiteKey::from(position.inner());
        self.history.iter().filter(|key| *key == &target_key).count() as u32
    }

    /// 特定の局面が千日手であるか（4回出現したか）をチェックします。
    pub fn check_sennichite(&self, position: &Position) -> SennichiteStatus {
        let count = self.get_position_count(position);
        if count >= 4 {
            if self.is_perpetual_check_placeholder() {
                SennichiteStatus::PerpetualCheckLoss
            } else {
                SennichiteStatus::Draw
            }
        } else {
            SennichiteStatus::None
        }
    }

    /// 連続王手チェックのプレースホルダー関数。
    ///
    /// 実際のアプリケーションでは、この関数は、繰り返し発生した局面の
    /// 指し手履歴を分析し、それらがすべて王手であったかを判断する
    /// 複雑なロジックを含む必要があります。
    /// `shogi_core`は合法手判定を提供しないため、このロジックは
    /// 別のクレートや自作のエンジンで実装される必要があります [3, 4]。
    pub fn is_perpetual_check_placeholder(&self) -> bool {
        // ここに実際の連続王手チェックロジックを実装します。
        // 現時点では常にfalseを返します。
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_core::{Move, Position, Square};

    #[test]
    fn test_is_sennichite_detection() {
        // テスト用に小さな容量のリングバッファを使用
        const TEST_CAPACITY: usize = 10;
        let mut detector = SennichiteDetector::<TEST_CAPACITY>::new();

        // 繰り返す局面を作成
        let mut repeating_pos = Position::default();
        // 最初の局面を記録
        detector.record_position(&repeating_pos);

        // 4回同じ局面を繰り返す
        for _ in 0..3 {
            // 局面を2手進めて元に戻す
            let mv1 = Move::Normal { from: Square::new(7, 7).unwrap(), to: Square::new(7, 6).unwrap(), promote: false };
            let mv2 = Move::Normal { from: Square::new(3, 3).unwrap(), to: Square::new(3, 4).unwrap(), promote: false };
            let mv3 = Move::Normal { from: Square::new(7, 6).unwrap(), to: Square::new(7, 7).unwrap(), promote: false };
            let mv4 = Move::Normal { from: Square::new(3, 4).unwrap(), to: Square::new(3, 3).unwrap(), promote: false };
            
            repeating_pos.make_move(mv1).unwrap();
            repeating_pos.make_move(mv2).unwrap();
            repeating_pos.make_move(mv3).unwrap();
            repeating_pos.make_move(mv4).unwrap();
            
            // この時点で初期局面に戻っている
            detector.record_position(&repeating_pos);
        }

        // 3回出現した時点では千日手ではないはず
        assert_eq!(detector.get_position_count(&Position::default()), 4);
        assert_eq!(detector.check_sennichite(&Position::default()), SennichiteStatus::Draw);
    }
}