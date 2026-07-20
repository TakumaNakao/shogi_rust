use shogi_core::Color;
use shogi_lib::Position;

use crate::position_hash::PositionHasher;

/// Repetition adjudication according to the Japan Shogi Association rule:
/// four occurrences draw unless one side gave check on every one of its moves
/// in the interval, in which case that checking side loses.
#[derive(PartialEq, Debug, Eq, Clone, Copy)]
pub enum SennichiteStatus {
    None,
    Draw,
    PerpetualCheckLoss { loser: Color },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoryEntry {
    pub key: u64,
    pub side_to_move: Color,
    pub moved_by: Option<Color>,
    pub gave_check: bool,
}

#[derive(Debug, Clone, Default)]
pub struct GameHistory {
    entries: Vec<HistoryEntry>,
}

impl GameHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_initial_position(&mut self, position: &Position) {
        self.entries.push(HistoryEntry {
            key: PositionHasher::calculate_hash(position),
            side_to_move: position.side_to_move(),
            moved_by: None,
            gave_check: false,
        });
    }

    /// Compatibility entry point. Prefer `record_position_after_move` when the
    /// caller knows that a legal move has just been made.
    pub fn record_position(&mut self, position: &Position) {
        let side_to_move = position.side_to_move();
        let moved_by = self
            .entries
            .last()
            .filter(|previous| previous.side_to_move != side_to_move)
            .map(|previous| previous.side_to_move);
        self.entries.push(HistoryEntry {
            key: PositionHasher::calculate_hash(position),
            side_to_move,
            moved_by,
            gave_check: moved_by.is_some() && position.in_check(),
        });
    }

    pub fn record_position_after_move(&mut self, position: &Position, moved_by: Color) {
        debug_assert_eq!(position.side_to_move(), moved_by.flip());
        self.entries.push(HistoryEntry {
            key: PositionHasher::calculate_hash(position),
            side_to_move: position.side_to_move(),
            moved_by: Some(moved_by),
            gave_check: position.in_check(),
        });
    }

    pub fn unrecord_last_position(&mut self) {
        self.entries.pop();
    }

    pub fn get_position_count(&self, position: &Position) -> u32 {
        let target_key = PositionHasher::calculate_hash(position);
        let side_to_move = position.side_to_move();
        self.entries
            .iter()
            .filter(|entry| entry.key == target_key && entry.side_to_move == side_to_move)
            .count() as u32
    }

    pub fn adjudicate(&self, position: &Position) -> SennichiteStatus {
        self.adjudicate_key(
            PositionHasher::calculate_hash(position),
            position.side_to_move(),
        )
    }

    pub fn check_sennichite(&self, position: &Position) -> SennichiteStatus {
        self.adjudicate(position)
    }

    pub fn check_sennichite_assuming_alternating_history(
        &self,
        position: &Position,
    ) -> SennichiteStatus {
        self.adjudicate(position)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[HistoryEntry] {
        &self.entries
    }

    fn adjudicate_key(&self, target_key: u64, side_to_move: Color) -> SennichiteStatus {
        let occurrences = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                (entry.key == target_key && entry.side_to_move == side_to_move).then_some(index)
            })
            .collect::<Vec<_>>();
        if occurrences.len() < 4 {
            return SennichiteStatus::None;
        }

        let start = occurrences[occurrences.len() - 4];
        let end = *occurrences.last().expect("four occurrences exist");
        let interval = &self.entries[start + 1..=end];
        let black_checks = all_moves_gave_check(interval, Color::Black);
        let white_checks = all_moves_gave_check(interval, Color::White);

        match (black_checks, white_checks) {
            (true, false) => SennichiteStatus::PerpetualCheckLoss {
                loser: Color::Black,
            },
            (false, true) => SennichiteStatus::PerpetualCheckLoss {
                loser: Color::White,
            },
            _ => SennichiteStatus::Draw,
        }
    }
}

fn all_moves_gave_check(interval: &[HistoryEntry], color: Color) -> bool {
    let mut moves = interval
        .iter()
        .filter(|entry| entry.moved_by == Some(color));
    let Some(first) = moves.next() else {
        return false;
    };
    first.gave_check && moves.all(|entry| entry.gave_check)
}

/// Compatibility wrapper for callers that still carry a compile-time history
/// capacity. The rich history is intentionally not truncated at that capacity.
#[derive(Debug, Clone, Default)]
pub struct SennichiteDetector<const CAPACITY: usize> {
    history: GameHistory,
}

impl<const CAPACITY: usize> SennichiteDetector<CAPACITY> {
    pub fn new() -> Self {
        Self {
            history: GameHistory::new(),
        }
    }
}

impl<const CAPACITY: usize> std::ops::Deref for SennichiteDetector<CAPACITY> {
    type Target = GameHistory;

    fn deref(&self) -> &Self::Target {
        &self.history
    }
}

impl<const CAPACITY: usize> std::ops::DerefMut for SennichiteDetector<CAPACITY> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_core::{Move, Square};

    fn push(
        history: &mut GameHistory,
        key: u64,
        side_to_move: Color,
        moved_by: Color,
        gave_check: bool,
    ) {
        history.entries.push(HistoryEntry {
            key,
            side_to_move,
            moved_by: Some(moved_by),
            gave_check,
        });
    }

    #[test]
    fn ordinary_fourfold_repetition_is_a_draw() {
        let mut history = GameHistory::new();
        let repeated_key = 7;
        history.entries.push(HistoryEntry {
            key: repeated_key,
            side_to_move: Color::Black,
            moved_by: None,
            gave_check: false,
        });
        for cycle in 0..3 {
            push(&mut history, 100 + cycle, Color::White, Color::Black, false);
            push(&mut history, 200 + cycle, Color::Black, Color::White, false);
            push(&mut history, 300 + cycle, Color::White, Color::Black, false);
            push(
                &mut history,
                repeated_key,
                Color::Black,
                Color::White,
                false,
            );
        }
        assert_eq!(
            SennichiteStatus::Draw,
            history.adjudicate_key(repeated_key, Color::Black)
        );
    }

    #[test]
    fn only_a_side_checking_on_every_move_loses() {
        let mut history = GameHistory::new();
        let repeated_key = 9;
        history.entries.push(HistoryEntry {
            key: repeated_key,
            side_to_move: Color::Black,
            moved_by: None,
            gave_check: false,
        });
        for cycle in 0..3 {
            push(&mut history, 100 + cycle, Color::White, Color::Black, true);
            push(&mut history, 200 + cycle, Color::Black, Color::White, false);
            push(&mut history, 300 + cycle, Color::White, Color::Black, true);
            push(
                &mut history,
                repeated_key,
                Color::Black,
                Color::White,
                false,
            );
        }
        assert_eq!(
            SennichiteStatus::PerpetualCheckLoss {
                loser: Color::Black
            },
            history.adjudicate_key(repeated_key, Color::Black)
        );
    }

    #[test]
    fn a_non_check_inside_the_repetition_interval_makes_it_a_draw() {
        let mut history = GameHistory::new();
        let repeated_key = 11;
        history.entries.push(HistoryEntry {
            key: repeated_key,
            side_to_move: Color::Black,
            moved_by: None,
            gave_check: false,
        });
        for cycle in 0..3 {
            push(
                &mut history,
                100 + cycle,
                Color::White,
                Color::Black,
                cycle != 1,
            );
            push(&mut history, 200 + cycle, Color::Black, Color::White, false);
            push(&mut history, 300 + cycle, Color::White, Color::Black, true);
            push(
                &mut history,
                repeated_key,
                Color::Black,
                Color::White,
                false,
            );
        }
        assert_eq!(
            SennichiteStatus::Draw,
            history.adjudicate_key(repeated_key, Color::Black)
        );
    }

    #[test]
    fn mixed_checks_by_both_sides_are_not_continuous_check() {
        let mut history = GameHistory::new();
        let repeated_key = 13;
        history.entries.push(HistoryEntry {
            key: repeated_key,
            side_to_move: Color::Black,
            moved_by: None,
            gave_check: false,
        });
        for cycle in 0..3 {
            push(&mut history, 100 + cycle, Color::White, Color::Black, true);
            push(
                &mut history,
                200 + cycle,
                Color::Black,
                Color::White,
                cycle == 1,
            );
            push(&mut history, 300 + cycle, Color::White, Color::Black, false);
            push(&mut history, repeated_key, Color::Black, Color::White, true);
        }
        assert_eq!(
            SennichiteStatus::Draw,
            history.adjudicate_key(repeated_key, Color::Black)
        );
    }

    #[test]
    fn undo_restores_the_previous_adjudication() {
        let mut history = GameHistory::new();
        let position = Position::default();
        for _ in 0..4 {
            history.record_position(&position);
        }
        assert_eq!(SennichiteStatus::Draw, history.adjudicate(&position));
        history.unrecord_last_position();
        assert_eq!(SennichiteStatus::None, history.adjudicate(&position));
    }

    #[test]
    fn history_is_not_truncated_at_legacy_capacity() {
        let mut detector = SennichiteDetector::<8>::new();
        for ply in 0..301 {
            let side_to_move = if ply % 2 == 0 {
                Color::Black
            } else {
                Color::White
            };
            detector.history.entries.push(HistoryEntry {
                key: ply,
                side_to_move,
                moved_by: (ply > 0).then_some(side_to_move.flip()),
                gave_check: false,
            });
        }
        assert_eq!(301, detector.len());
    }

    #[test]
    fn real_position_fourfold_cycle_is_a_draw() {
        let mut history = GameHistory::new();
        let mut position = crate::utils::position_from_sfen_or_usi("4k4/9/9/9/9/9/9/9/4K4 b - 1")
            .expect("valid kings-only position");
        history.record_initial_position(&position);
        let cycle = [
            Move::Normal {
                from: Square::new(5, 9).unwrap(),
                to: Square::new(6, 9).unwrap(),
                promote: false,
            },
            Move::Normal {
                from: Square::new(5, 1).unwrap(),
                to: Square::new(6, 1).unwrap(),
                promote: false,
            },
            Move::Normal {
                from: Square::new(6, 9).unwrap(),
                to: Square::new(5, 9).unwrap(),
                promote: false,
            },
            Move::Normal {
                from: Square::new(6, 1).unwrap(),
                to: Square::new(5, 1).unwrap(),
                promote: false,
            },
        ];
        for _ in 0..3 {
            for mv in cycle {
                let moved_by = position.side_to_move();
                position.do_move(mv);
                history.record_position_after_move(&position, moved_by);
            }
        }
        assert_eq!(SennichiteStatus::Draw, history.adjudicate(&position));
    }
}
