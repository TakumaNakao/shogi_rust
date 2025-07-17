use std::collections::HashMap;
use shogi_core::{Color, Move, PieceKind, Square};
use arrayvec::ArrayVec;

/// `PieceToHistory`テーブル
#[derive(Debug, Default)]
struct PieceToHistory {
    history: HashMap<(String, String), i32>,
}

impl PieceToHistory {
    fn get_score(&self, piece_kind: PieceKind, to_sq: Square) -> i32 {
        *self.history.get(&(format!("{:?}", piece_kind), format!("{:?}", to_sq))).unwrap_or(&0)
    }

    fn update_score(&mut self, piece_kind: PieceKind, to_sq: Square, delta: i32) {
        *self.history.entry((format!("{:?}", piece_kind), format!("{:?}", to_sq))).or_insert(0) += delta;
    }

    fn clear(&mut self) {
        self.history.clear();
    }
}

/// `ButterflyHistory`テーブル
#[derive(Debug, Default)]
struct ButterflyHistory {
    history: HashMap<(String, String, String), i32>,
}

impl ButterflyHistory {
    fn get_score(&self, from_sq: Square, to_sq: Square, color: Color) -> i32 {
        *self.history.get(&(format!("{:?}", from_sq), format!("{:?}", to_sq), format!("{:?}", color))).unwrap_or(&0)
    }

    fn update_score(&mut self, from_sq: Square, to_sq: Square, color: Color, delta: i32) {
        *self.history.entry((format!("{:?}", from_sq), format!("{:?}", to_sq), format!("{:?}", color))).or_insert(0) += delta;
    }

    fn clear(&mut self) {
        self.history.clear();
    }
}

/// `CounterMoveHistory`テーブル
#[derive(Debug, Default)]
struct CounterMoveHistory {
    history: HashMap<(String, String), i32>,
}

impl CounterMoveHistory {
    fn get_score(&self, opponent_move: Move, best_move: Move) -> i32 {
        *self.history.get(&(format!("{:?}", opponent_move), format!("{:?}", best_move))).unwrap_or(&0)
    }

    fn update_score(&mut self, opponent_move: Move, best_move: Move, delta: i32) {
        *self.history.entry((format!("{:?}", opponent_move), format!("{:?}", best_move))).or_insert(0) += delta;
    }

    fn clear(&mut self) {
        self.history.clear();
    }
}

/// 将棋の指し手オーダリングを管理する構造体
#[derive(Debug, Default)]
pub struct MoveOrdering {
    piece_to_history: PieceToHistory,
    butterfly_history: ButterflyHistory,
    counter_move_history: CounterMoveHistory,
}

impl MoveOrdering {
    pub fn new() -> Self {
        MoveOrdering::default()
    }

    /// 指し手にスコアを割り当てます。
    pub fn score_move(&self, current_move: &Move, position: &shogi_core::Position) -> i32 {
        let mut score = 0;

        let piece_kind = match current_move {
            Move::Normal { from, .. } => position.piece_at(*from).unwrap().piece_kind(),
            Move::Drop { piece, .. } => piece.piece_kind(),
        };

        score += self.piece_to_history.get_score(piece_kind, current_move.to());

        if let Some(from_sq) = current_move.from() {
            score += self.butterfly_history.get_score(from_sq, current_move.to(), position.side_to_move());
        }

        if let Some(op_move) = position.last_move() {
            score += self.counter_move_history.get_score(op_move, *current_move);
        }

        if let Move::Normal { to, .. } = current_move {
            if position.piece_at(*to).is_some() {
                score += 1000;
            }
        }
        if let Move::Normal { promote, .. } = current_move {
            if *promote {
                score += 500;
            }
        }
        if let Move::Drop { .. } = current_move {
            score += 100;
        }

        score
    }

    /// 指し手のリストをスコアに基づいてソートします。
    pub fn sort_moves(&self, moves: &mut ArrayVec<Move, 593>, position: &shogi_core::Position) {
        moves.sort_by_key(|m| -self.score_move(m, position));
    }

    /// 履歴テーブルを更新します。
    pub fn update_history(&mut self, good_move: &Move, position: &shogi_core::Position, delta: i32) {
        let piece_kind = match good_move {
            Move::Normal { from, .. } => position.piece_at(*from).unwrap().piece_kind(),
            Move::Drop { piece, .. } => piece.piece_kind(),
        };
        self.piece_to_history.update_score(piece_kind, good_move.to(), delta);

        if let Some(from_sq) = good_move.from() {
            self.butterfly_history.update_score(from_sq, good_move.to(), position.side_to_move(), delta);
        }

        if let Some(op_move) = position.last_move() {
            self.counter_move_history.update_score(op_move, *good_move, delta);
        }
    }

    /// すべての履歴テーブルをクリアします。
    pub fn clear_histories(&mut self) {
        self.piece_to_history.clear();
        self.butterfly_history.clear();
        self.counter_move_history.clear();
    }
}
