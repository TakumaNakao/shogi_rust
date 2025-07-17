use shogi_core::{Move, Position};

use crate::evaluation::{get_piece_value, Evaluator};
use crate::move_ordering::MoveOrdering;
use crate::sennichite::{SennichiteDetector, SennichiteStatus};

/// 将棋のアルファベータ探索を管理する構造体
pub struct ShogiAI<E: Evaluator, const HISTORY_CAPACITY: usize> {
    move_ordering: MoveOrdering,
    pub evaluator: E,
    pub sennichite_detector: SennichiteDetector<HISTORY_CAPACITY>,
}

impl<E: Evaluator, const HISTORY_CAPACITY: usize> ShogiAI<E, HISTORY_CAPACITY> {
    pub fn new(evaluator: E) -> Self {
        ShogiAI {
            move_ordering: MoveOrdering::new(),
            evaluator,
            sennichite_detector: SennichiteDetector::new(),
        }
    }

    /// 静的交換評価（SEE）を計算します。
    /// この実装では、指し手で発生する駒の交換の損得を単純に計算します。
    /// 正の値は得、負の値は損を意味します。
    fn see(&self, position: &Position, mv: Move) -> i32 {
        if let Move::Normal { from, to, .. } = mv {
            if let (Some(attacker), Some(victim)) = (position.piece_at(from), position.piece_at(to)) {
                //  victimの価値 - attackerの価値
                return get_piece_value(victim.piece_kind()) - get_piece_value(attacker.piece_kind());
            }
        }
        // 駒を取らない手や打ち込みはSEEの対象外
        0
    }

    pub fn quiescence_search( &mut self, position: &shogi_core::Position, mut alpha: f32, beta: f32, ) -> f32 {
        let stand_pat_score = self.evaluator.evaluate(position);

        if stand_pat_score >= beta {
            return beta;
        }
        alpha = alpha.max(stand_pat_score);

        let yasai_pos = yasai::Position::new(position.inner().clone());
        let mut moves = yasai_pos.legal_moves();
        
        // 駒を取る手のみにフィルタリング
        moves.retain(|m| {
            if let Move::Normal { to, .. } = m {
                position.piece_at(*to).is_some()
            } else {
                false
            }
        });

        if moves.is_empty() {
            return stand_pat_score;
        }

        self.move_ordering.sort_moves(&mut moves, position);

        let mut best_score = stand_pat_score;
        for mv in moves {
            // SEEがマイナス（損）になる手は枝刈り
            if self.see(position, mv) < 0 {
                continue;
            }

            let mut next_position = position.clone();
            if next_position.make_move(mv).is_none() { continue; }

            self.sennichite_detector.record_position(&next_position);
            let sennichite_status = self.is_sennichite_internal(&next_position);
            let score = match sennichite_status {
                SennichiteStatus::Draw => 0.0,
                SennichiteStatus::PerpetualCheckLoss => -f32::INFINITY,
                SennichiteStatus::None => {
                    -self.quiescence_search(&next_position, -beta, -alpha)
                }
            };
            self.sennichite_detector.unrecord_last_position();

            if score > best_score {
                best_score = score;
            }
            alpha = alpha.max(score);

            if alpha >= beta {
                break;
            }
        }
        best_score
    }

    pub fn alpha_beta_search( &mut self, position: &shogi_core::Position, depth: u8, mut alpha: f32, beta: f32, ) -> f32 {
        if depth == 0 {
            return self.quiescence_search(position, alpha, beta);
        }

        let yasai_pos = yasai::Position::new(position.inner().clone());
        let mut moves = yasai_pos.legal_moves();

        if moves.is_empty() {
            return -f32::INFINITY;
        }

        self.move_ordering.sort_moves(&mut moves, position);

        let mut best_score = -f32::INFINITY;
        for mv in moves {
            let mut next_position = position.clone();
            if next_position.make_move(mv).is_none() {
                continue;
            }

            self.sennichite_detector.record_position(&next_position);
            let sennichite_status = self.is_sennichite_internal(&next_position);

            let score = match sennichite_status {
                SennichiteStatus::Draw => 0.0,
                SennichiteStatus::PerpetualCheckLoss => -f32::INFINITY,
                SennichiteStatus::None => {
                    -self.alpha_beta_search(&next_position, depth - 1, -beta, -alpha)
                }
            };

            self.sennichite_detector.unrecord_last_position();

            if score > best_score {
                best_score = score;
            }
            alpha = alpha.max(score);

            if alpha >= beta {
                self.move_ordering.update_history(&mv, position, depth as i32 * 10);
                break;
            }
        }
        best_score
    }

    pub fn is_sennichite_internal(&self, position: &shogi_core::Position) -> SennichiteStatus {
        self.sennichite_detector.check_sennichite(position)
    }

    pub fn find_best_move(&mut self, position: &shogi_core::Position, depth: u8) -> Option<Move> {
        let mut best_move: Option<Move> = None;
        let mut best_eval = -f32::INFINITY;
        let mut alpha = -f32::INFINITY;
        let beta = f32::INFINITY;

        let yasai_pos = yasai::Position::new(position.inner().clone());
        let mut moves = yasai_pos.legal_moves();

        self.move_ordering.sort_moves(&mut moves, position);

        if moves.is_empty() {
            return None;
        }

        for mv in moves {
            let mut next_position = position.clone();
            if next_position.make_move(mv).is_none() {
                continue;
            }

            self.sennichite_detector.record_position(&next_position);
            let sennichite_status = self.is_sennichite_internal(&next_position);
            self.sennichite_detector.unrecord_last_position();

            let eval = match sennichite_status {
                SennichiteStatus::Draw => 0.0,
                SennichiteStatus::PerpetualCheckLoss => -f32::INFINITY,
                SennichiteStatus::None => {
                    -self.alpha_beta_search(&next_position, depth - 1, -beta, -alpha)
                }
            };

            if eval > best_eval {
                best_eval = eval;
                best_move = Some(mv);
            }
            alpha = alpha.max(eval);
        }

        if let Some(bm) = best_move {
            self.move_ordering.update_history(&bm, position, depth as i32 * 20);
        }

        best_move
    }
}