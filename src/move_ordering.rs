use std::collections::HashMap;
use shogi_core::{Color, Move, PieceKind, Square};
use arrayvec::ArrayVec;
use crate::position_hash::{color_to_index, ZOBRIST_KEYS};
use crate::utils::get_piece_value;
use shogi_lib::Position;

fn board_piece_kind_index(kind: PieceKind) -> usize {
    match kind {
        PieceKind::Pawn => 0,
        PieceKind::Lance => 1,
        PieceKind::Knight => 2,
        PieceKind::Silver => 3,
        PieceKind::Gold => 4,
        PieceKind::Bishop => 5,
        PieceKind::Rook => 6,
        PieceKind::King => 7,
        PieceKind::ProPawn => 8,
        PieceKind::ProLance => 9,
        PieceKind::ProKnight => 10,
        PieceKind::ProSilver => 11,
        PieceKind::ProBishop => 12,
        PieceKind::ProRook => 13,
    }
}

fn hand_piece_kind_index(kind: PieceKind) -> usize {
    match kind {
        PieceKind::Pawn => 0,
        PieceKind::Lance => 1,
        PieceKind::Knight => 2,
        PieceKind::Silver => 3,
        PieceKind::Gold => 4,
        PieceKind::Bishop => 5,
        PieceKind::Rook => 6,
        _ => 0,
    }
}

fn hash_move(mv: Move) -> u64 {
    match mv {
        Move::Normal { from, to, promote } => {
            let from_idx = (from.index() - 1) as usize;
            let to_idx = (to.index() - 1) as usize;
            let from_key = ZOBRIST_KEYS.board[0][from_idx][0];
            let to_key = ZOBRIST_KEYS.board[1][to_idx][1];
            let promote_key = if promote { ZOBRIST_KEYS.side_to_move } else { 0 };
            from_key ^ to_key ^ promote_key
        }
        Move::Drop { piece, to } => {
            let to_idx = (to.index() - 1) as usize;
            let piece_idx = hand_piece_kind_index(piece.piece_kind());
            let drop_key = ZOBRIST_KEYS.hand[piece_idx][1][color_to_index(piece.color())];
            let to_key = ZOBRIST_KEYS.board[1][to_idx][1];
            drop_key ^ to_key
        }
    }
}

/// `PieceToHistory`テーブル
#[derive(Debug)]
struct PieceToHistory {
    history: [[[i32; 2]; 81]; 14],
}

impl Default for PieceToHistory {
    fn default() -> Self {
        Self {
            history: [[[0; 2]; 81]; 14],
        }
    }
}

impl PieceToHistory {
    fn get_score(&self, piece_kind: PieceKind, to_sq: Square, color: Color) -> i32 {
        let to_idx = (to_sq.index() - 1) as usize;
        let piece_idx = board_piece_kind_index(piece_kind);
        self.history[piece_idx][to_idx][color_to_index(color)]
    }

    fn update_score(&mut self, piece_kind: PieceKind, to_sq: Square, color: Color, delta: i32) {
        let to_idx = (to_sq.index() - 1) as usize;
        let piece_idx = board_piece_kind_index(piece_kind);
        self.history[piece_idx][to_idx][color_to_index(color)] += delta;
    }

    fn clear(&mut self) {
        self.history = [[[0; 2]; 81]; 14];
    }

    fn decay(&mut self) {
        for piece_table in self.history.iter_mut() {
            for square_table in piece_table.iter_mut() {
                for score in square_table.iter_mut() {
                    *score /= 2;
                }
            }
        }
    }
}

/// `ButterflyHistory`テーブル
#[derive(Debug)]
struct ButterflyHistory {
    history: [[[i32; 2]; 81]; 81],
}

impl Default for ButterflyHistory {
    fn default() -> Self {
        Self {
            history: [[[0; 2]; 81]; 81],
        }
    }
}

impl ButterflyHistory {
    fn get_score(&self, from_sq: Square, to_sq: Square, color: Color) -> i32 {
        let from_idx = (from_sq.index() - 1) as usize;
        let to_idx = (to_sq.index() - 1) as usize;
        self.history[from_idx][to_idx][color_to_index(color)]
    }

    fn update_score(&mut self, from_sq: Square, to_sq: Square, color: Color, delta: i32) {
        let from_idx = (from_sq.index() - 1) as usize;
        let to_idx = (to_sq.index() - 1) as usize;
        self.history[from_idx][to_idx][color_to_index(color)] += delta;
    }

    fn clear(&mut self) {
        self.history = [[[0; 2]; 81]; 81];
    }

    fn decay(&mut self) {
        for from_table in self.history.iter_mut() {
            for to_table in from_table.iter_mut() {
                for score in to_table.iter_mut() {
                    *score /= 2;
                }
            }
        }
    }
}

/// `CounterMoveHistory`テーブル
#[derive(Debug, Default)]
struct CounterMoveHistory {
    history: HashMap<u64, i32>,
}

impl CounterMoveHistory {
    fn get_score(&self, opponent_move: Move, best_move: Move) -> i32 {
        let key = hash_move(opponent_move) ^ hash_move(best_move);
        *self.history.get(&key).unwrap_or(&0)
    }

    fn update_score(&mut self, opponent_move: Move, best_move: Move, delta: i32) {
        let key = hash_move(opponent_move) ^ hash_move(best_move);
        *self.history.entry(key).or_insert(0) += delta;
    }

    fn clear(&mut self) {
        self.history.clear();
    }

    fn decay(&mut self) {
        for score in self.history.values_mut() {
            *score /= 2;
        }
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
    pub fn score_move(&self, current_move: &Move, position: &Position) -> i32 {
        let mut score = 0;

        let (piece_kind, color) = match current_move {
            Move::Normal { from, .. } => {
                let piece = position.piece_at(*from).unwrap();
                (piece.piece_kind(), piece.color())
            },
            Move::Drop { piece, .. } => (piece.piece_kind(), piece.color()),
        };

        score += self.piece_to_history.get_score(piece_kind, current_move.to(), color);

        if let Some(from_sq) = current_move.from() {
            score += self.butterfly_history.get_score(from_sq, current_move.to(), position.side_to_move());
        }

        if let Some(op_move) = position.last_move() {
            score += self.counter_move_history.get_score(op_move, *current_move);
        }

        if let Move::Normal { from, to, .. } = current_move {
            // 駒を取る手の場合、MVV-LVAスコアを加算
            if let (Some(victim), Some(attacker)) = (position.piece_at(*to), position.piece_at(*from)) {
                score += get_piece_value(victim.piece_kind()) * 100 - get_piece_value(attacker.piece_kind());
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
    pub fn sort_moves(&self, moves: &mut ArrayVec<Move, 593>, position: &Position) {
        moves.sort_by_key(|m| -self.score_move(m, position));
    }

    /// 履歴テーブルを更新します。
    pub fn update_history(&mut self, good_move: &Move, position: &Position, delta: i32) {
        let (piece_kind, color) = match good_move {
            Move::Normal { from, .. } => {
                let piece = position.piece_at(*from).unwrap();
                (piece.piece_kind(), piece.color())
            },
            Move::Drop { piece, .. } => (piece.piece_kind(), piece.color()),
        };
        self.piece_to_history.update_score(piece_kind, good_move.to(), color, delta);

        if let Some(from_sq) = good_move.from() {
            self.butterfly_history.update_score(from_sq, good_move.to(), position.side_to_move(), delta);
        }

        if let Some(op_move) = position.last_move() {
            self.counter_move_history.update_score(op_move, *good_move, delta);
        }
    }

    /// すべての履歴テーブルをクリアします。
    pub fn clear(&mut self) {
        self.piece_to_history.clear();
        self.butterfly_history.clear();
        self.counter_move_history.clear();
    }

    /// すべての履歴テーブルのスコアを減衰させます。
    pub fn decay(&mut self) {
        self.piece_to_history.decay();
        self.butterfly_history.decay();
        self.counter_move_history.decay();
    }
}
