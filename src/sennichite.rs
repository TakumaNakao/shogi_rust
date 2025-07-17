use circular_buffer::CircularBuffer;
use shogi_core::Position;

use crate::position_hash::PositionHasher;

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
    history: CircularBuffer<CAPACITY, u64>,
}

impl<const CAPACITY: usize> SennichiteDetector<CAPACITY> {
    /// 新しい千日手検出器を作成します。
    pub fn new() -> Self {
        SennichiteDetector {
            history: CircularBuffer::new(),
        }
    }

    /// 現在の局面のハッシュ値を履歴に記録します。
    pub fn record_position(&mut self, position: &Position) {
        let hash = PositionHasher::calculate_hash(position);
        self.history.push_back(hash);
    }

    /// 履歴から最も新しい局面ハッシュを削除します。
    pub fn unrecord_last_position(&mut self) {
        self.history.pop_back();
    }

    /// 特定の局面の出現回数を取得します。
    pub fn get_position_count(&self, position: &Position) -> u32 {
        let target_hash = PositionHasher::calculate_hash(position);
        self.history.iter().filter(|&&h| h == target_hash).count() as u32
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
    pub fn is_perpetual_check_placeholder(&self) -> bool {
        // ここに実際の連続王手チェックロジックを実装します。
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_core::{Move, Square};

    #[test]
    fn test_is_sennichite_detection_with_hash() {
        const TEST_CAPACITY: usize = 20;
        let mut detector = SennichiteDetector::<TEST_CAPACITY>::new();

        let mut pos = Position::default();
        detector.record_position(&pos);

        // 4回同じ局面を繰り返す
        for _ in 0..3 {
            let mv1 = Move::Normal { from: Square::new(2, 7).unwrap(), to: Square::new(2, 6).unwrap(), promote: false };
            let mv2 = Move::Normal { from: Square::new(8, 3).unwrap(), to: Square::new(8, 4).unwrap(), promote: false };
            pos.make_move(mv1).unwrap();
            pos.make_move(mv2).unwrap();
            detector.record_position(&pos);

            let mv3 = Move::Normal { from: Square::new(2, 6).unwrap(), to: Square::new(2, 7).unwrap(), promote: false };
            let mv4 = Move::Normal { from: Square::new(8, 4).unwrap(), to: Square::new(8, 3).unwrap(), promote: false };
            pos.make_move(mv3).unwrap();
            pos.make_move(mv4).unwrap();
            detector.record_position(&pos);
        }

        // 初期局面に戻っているはず
        let initial_pos = Position::default();
        assert_eq!(detector.get_position_count(&initial_pos), 4);
        assert_eq!(detector.check_sennichite(&initial_pos), SennichiteStatus::Draw);
    }
}
