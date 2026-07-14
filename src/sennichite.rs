use circular_buffer::CircularBuffer;
use shogi_core::{Color, PieceKind};
use shogi_lib::Position;

use crate::position_hash::PositionHasher;
use crate::utils::get_piece_value;

const HAND_KINDS: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

#[derive(Clone, Copy, Debug)]
struct HistoryEntry {
    hash: u64,
    board_key_with_side: u64,
    side_to_move: Color,
    in_check: bool,
    black_hand: [u8; 7],
    white_hand: [u8; 7],
}

impl HistoryEntry {
    fn new(position: &Position) -> Self {
        Self {
            hash: PositionHasher::calculate_hash(position),
            board_key_with_side: position.keys().0,
            side_to_move: position.side_to_move(),
            in_check: position.in_check(),
            black_hand: hand_counts(position, Color::Black),
            white_hand: hand_counts(position, Color::White),
        }
    }

    fn hand(&self, color: Color) -> &[u8; 7] {
        match color {
            Color::Black => &self.black_hand,
            Color::White => &self.white_hand,
        }
    }
}

fn hand_counts(position: &Position, color: Color) -> [u8; 7] {
    std::array::from_fn(|index| position.hand(color).count(HAND_KINDS[index]).unwrap_or(0))
}

fn opposite(color: Color) -> Color {
    match color {
        Color::Black => Color::White,
        Color::White => Color::Black,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceCycle {
    pub loser: Color,
    /// Material swing measured as the loser's decrease plus the opponent's increase.
    pub material_swing: i32,
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
    /// 手番側が連続王手を行っていたため、直前に着手した側の勝ち
    PerpetualCheckWin,
}

/// 千日手検出器
/// 固定サイズのリングバッファを使用して、過去の局面ハッシュを管理します。
pub struct SennichiteDetector<const CAPACITY: usize> {
    /// 過去の局面と持ち駒の履歴（固定サイズのリングバッファ）
    history: CircularBuffer<CAPACITY, HistoryEntry>,
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
        self.history.push_back(HistoryEntry::new(position));
    }

    /// 履歴から最も新しい局面ハッシュを削除します。
    pub fn unrecord_last_position(&mut self) {
        self.history.pop_back();
    }

    /// 特定の局面の出現回数を取得します。
    pub fn get_position_count(&self, position: &Position) -> u32 {
        let target_hash = PositionHasher::calculate_hash(position);
        self.history
            .iter()
            .filter(|entry| entry.hash == target_hash)
            .count() as u32
    }

    pub fn prior_position_hashes(&self, current: &Position) -> Vec<u64> {
        let mut hashes = self
            .history
            .iter()
            .map(|entry| entry.hash)
            .collect::<Vec<_>>();
        if hashes.last().copied() == Some(PositionHasher::calculate_hash(current)) {
            hashes.pop();
        }
        hashes
    }

    /// 特定の局面が千日手であるか（4回出現したか）をチェックします。
    pub fn check_sennichite(&self, position: &Position) -> SennichiteStatus {
        let count = self.get_position_count(position);
        if count >= 4 {
            if position.in_check() {
                SennichiteStatus::PerpetualCheckLoss
            } else {
                SennichiteStatus::Draw
            }
        } else {
            SennichiteStatus::None
        }
    }

    /// 探索中の交互着手履歴を前提に、同じ手番の局面だけを走査して千日手をチェックします。
    pub fn check_sennichite_assuming_alternating_history(
        &self,
        position: &Position,
    ) -> SennichiteStatus {
        let target_hash = PositionHasher::calculate_hash(position);
        let mut occurrence_count = 0;
        let mut cycle_start = None;
        for (index, entry) in self.history.iter().enumerate().rev() {
            if entry.hash == target_hash {
                occurrence_count += 1;
                if occurrence_count == 4 {
                    cycle_start = Some(index);
                    break;
                }
            }
        }
        let Some(cycle_start) = cycle_start else {
            return SennichiteStatus::None;
        };

        for checker in [Color::Black, Color::White] {
            let checked_side = opposite(checker);
            let mut checked_positions = self
                .history
                .iter()
                .skip(cycle_start)
                .filter(|entry| entry.side_to_move == checked_side)
                .peekable();
            if checked_positions.peek().is_some() && checked_positions.all(|entry| entry.in_check) {
                return if checker == position.side_to_move() {
                    SennichiteStatus::PerpetualCheckWin
                } else {
                    SennichiteStatus::PerpetualCheckLoss
                };
            }
        }
        SennichiteStatus::Draw
    }

    /// Detects a same-board path where one side has only lost hand material and
    /// the opponent has only gained it. Exact repetitions are handled separately.
    pub fn resource_cycle(&self, position: &Position) -> Option<ResourceCycle> {
        let target_hash = PositionHasher::calculate_hash(position);
        let recorded_current = self
            .history
            .back()
            .filter(|entry| entry.hash == target_hash)
            .copied();
        let current = recorded_current.unwrap_or_else(|| HistoryEntry::new(position));
        let skip_last = recorded_current.is_some();
        for ancestor in self.history.iter().rev().skip(usize::from(skip_last)) {
            if ancestor.board_key_with_side != current.board_key_with_side {
                continue;
            }
            for loser in [Color::Black, Color::White] {
                let old_own = ancestor.hand(loser);
                let new_own = current.hand(loser);
                let old_opp = ancestor.hand(opposite(loser));
                let new_opp = current.hand(opposite(loser));
                let own_not_improved = new_own.iter().zip(old_own).all(|(new, old)| new <= old);
                let opponent_not_worse = new_opp.iter().zip(old_opp).all(|(new, old)| new >= old);
                let own_strict = new_own.iter().zip(old_own).any(|(new, old)| new < old);
                let opponent_strict = new_opp.iter().zip(old_opp).any(|(new, old)| new > old);
                if !own_not_improved || !opponent_not_worse || !own_strict || !opponent_strict {
                    continue;
                }

                let own_loss = old_own
                    .iter()
                    .zip(new_own)
                    .zip(HAND_KINDS)
                    .map(|((&old, &new), kind)| i32::from(old - new) * get_piece_value(kind))
                    .sum::<i32>();
                let opponent_gain = new_opp
                    .iter()
                    .zip(old_opp)
                    .zip(HAND_KINDS)
                    .map(|((&new, &old), kind)| i32::from(new - old) * get_piece_value(kind))
                    .sum::<i32>();
                return Some(ResourceCycle {
                    loser,
                    material_swing: own_loss + opponent_gain,
                });
            }
        }
        None
    }

    /// 履歴をクリアします。
    pub fn clear(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_core::{Move, Square};
    use shogi_usi_parser::FromUsi;

    #[test]
    fn test_is_sennichite_detection_with_hash() {
        const TEST_CAPACITY: usize = 20;
        let mut detector = SennichiteDetector::<TEST_CAPACITY>::new();

        let mut pos = Position::default();
        detector.record_position(&pos);

        // 4回同じ局面を繰り返す
        for _ in 0..3 {
            let mv1 = Move::Normal {
                from: Square::new(2, 7).unwrap(),
                to: Square::new(2, 6).unwrap(),
                promote: false,
            };
            let mv2 = Move::Normal {
                from: Square::new(8, 3).unwrap(),
                to: Square::new(8, 4).unwrap(),
                promote: false,
            };
            pos.do_move(mv1);
            pos.do_move(mv2);
            detector.record_position(&pos);

            let mv3 = Move::Normal {
                from: Square::new(2, 6).unwrap(),
                to: Square::new(2, 7).unwrap(),
                promote: false,
            };
            let mv4 = Move::Normal {
                from: Square::new(8, 4).unwrap(),
                to: Square::new(8, 3).unwrap(),
                promote: false,
            };
            pos.do_move(mv3);
            pos.do_move(mv4);
            detector.record_position(&pos);
        }

        // 初期局面に戻っているはず
        let initial_pos = Position::default();
        assert_eq!(detector.get_position_count(&initial_pos), 4);
        assert_eq!(
            detector.check_sennichite(&initial_pos),
            SennichiteStatus::Draw
        );
        assert_eq!(
            detector.check_sennichite_assuming_alternating_history(&initial_pos),
            SennichiteStatus::Draw
        );
    }

    #[test]
    fn test_checked_repetition_is_perpetual_check_loss() {
        const TEST_CAPACITY: usize = 8;
        let mut detector = SennichiteDetector::<TEST_CAPACITY>::new();
        let partial =
            shogi_core::PartialPosition::from_usi("sfen 4r3k/9/9/9/9/9/9/9/4K4 b - 1").unwrap();
        let pos = Position::new(partial);
        assert!(pos.in_check());

        for _ in 0..4 {
            detector.record_position(&pos);
        }

        assert_eq!(
            detector.check_sennichite(&pos),
            SennichiteStatus::PerpetualCheckLoss
        );
    }

    #[test]
    fn alternating_history_scan_counts_same_side_positions() {
        const TEST_CAPACITY: usize = 20;
        let mut detector = SennichiteDetector::<TEST_CAPACITY>::new();

        let mut pos = Position::default();
        detector.record_position(&pos);

        for _ in 0..3 {
            let mv1 = Move::Normal {
                from: Square::new(2, 7).unwrap(),
                to: Square::new(2, 6).unwrap(),
                promote: false,
            };
            pos.do_move(mv1);
            detector.record_position(&pos);

            let mv2 = Move::Normal {
                from: Square::new(8, 3).unwrap(),
                to: Square::new(8, 4).unwrap(),
                promote: false,
            };
            pos.do_move(mv2);
            detector.record_position(&pos);

            let mv3 = Move::Normal {
                from: Square::new(2, 6).unwrap(),
                to: Square::new(2, 7).unwrap(),
                promote: false,
            };
            pos.do_move(mv3);
            detector.record_position(&pos);

            let mv4 = Move::Normal {
                from: Square::new(8, 4).unwrap(),
                to: Square::new(8, 3).unwrap(),
                promote: false,
            };
            pos.do_move(mv4);
            detector.record_position(&pos);
        }

        let initial_pos = Position::default();
        assert_eq!(
            detector.check_sennichite(&initial_pos),
            detector.check_sennichite_assuming_alternating_history(&initial_pos)
        );
        assert_eq!(
            detector.check_sennichite_assuming_alternating_history(&initial_pos),
            SennichiteStatus::Draw
        );
    }

    #[test]
    fn detects_suite_resource_cycle_without_exact_repetition() {
        let mut position = crate::utils::position_from_sfen_or_usi(
            "ln1g3nl/1k7/1spsprbpp/p2p2p2/1Pg1Pp1P1/P1PSG1P2/1KNP1P2P/2SGR4/L6NL w bp 74",
        )
        .expect("valid suite cycle");
        let mut detector = SennichiteDetector::<32>::new();
        detector.record_position(&position);
        for text in ["P*8h", "8g8h", "B*8g", "8h8g"] {
            let mv = crate::utils::parse_usi_move_for_color(text, position.side_to_move())
                .expect("valid proof move");
            assert!(position.legal_moves().contains(&mv), "illegal move {text}");
            position.do_move(mv);
            detector.record_position(&position);
        }

        assert_eq!(SennichiteStatus::None, detector.check_sennichite(&position));
        assert_eq!(
            Some(ResourceCycle {
                loser: Color::White,
                material_swing: 1_800,
            }),
            detector.resource_cycle(&position)
        );
    }

    #[test]
    fn resource_cycle_requires_monotone_hands_on_the_same_board() {
        let before = crate::utils::position_from_sfen_or_usi("4k4/9/9/9/9/9/9/9/4K4 b P 1")
            .expect("valid before");
        let different_board =
            crate::utils::position_from_sfen_or_usi("4k4/9/9/9/9/9/9/4P4/4K4 b p 3")
                .expect("valid different board");
        let mixed_hand = crate::utils::position_from_sfen_or_usi("4k4/9/9/9/9/9/9/9/4K4 b Lp 3")
            .expect("valid mixed hand");
        let mut detector = SennichiteDetector::<8>::new();
        detector.record_position(&before);

        assert_eq!(None, detector.resource_cycle(&different_board));
        assert_eq!(None, detector.resource_cycle(&mixed_hand));
    }
}
