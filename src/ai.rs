use std::collections::HashMap;
use shogi_core::{Move, Position};
use crate::evaluation::{get_piece_value, Evaluator};
use crate::move_ordering::MoveOrdering;
use crate::position_hash::PositionHasher;
use crate::sennichite::{SennichiteDetector, SennichiteStatus};
use std::time::{Duration, Instant};

const MAX_DEPTH: usize = 64;

/// トランスポジションテーブルに格納する評価値の種類
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NodeType {
    /// 探索がすべての手を調べ、正確な評価値が確定したノード (score > alpha && score < beta)
    Exact,
    /// βカットが発生したノード。評価値は少なくともこの値以上である (score >= beta)
    LowerBound,
    /// すべての手を調べたがα値を更新できなかったノード。評価値はこの値以下である (score <= alpha)
    UpperBound,
}

/// トランスポジションテーブルのエントリ
#[derive(Clone, Copy, Debug)]
struct TranspositionEntry {
    score: f32,
    depth: u8,
    node_type: NodeType,
}

/// 将棋のアルファベータ探索を管理する構造体
pub struct ShogiAI<E: Evaluator, const HISTORY_CAPACITY: usize> {
    move_ordering: MoveOrdering,
    pub evaluator: E,
    pub sennichite_detector: SennichiteDetector<HISTORY_CAPACITY>,
    transposition_table: HashMap<u64, TranspositionEntry>,
    killer_moves: [[Option<Move>; 2]; MAX_DEPTH],
    start_time: Option<Instant>,
    time_limit: Option<Duration>,
}

impl<E: Evaluator, const HISTORY_CAPACITY: usize> ShogiAI<E, HISTORY_CAPACITY> {
    pub fn new(evaluator: E) -> Self {
        ShogiAI {
            move_ordering: MoveOrdering::new(),
            evaluator,
            sennichite_detector: SennichiteDetector::new(),
            transposition_table: HashMap::new(),
            killer_moves: [[None; 2]; MAX_DEPTH],
            start_time: None,
            time_limit: None,
        }
    }

    pub fn clear(&mut self) {
        self.move_ordering.clear();
        self.sennichite_detector.clear();
        self.transposition_table.clear();
        self.clear_killer_moves();
    }

    fn clear_killer_moves(&mut self) {
        self.killer_moves = [[None; 2]; MAX_DEPTH];
    }

    pub fn decay_history(&mut self) {
        self.move_ordering.decay();
    }

    fn update_killer_moves(&mut self, depth: u8, mv: Move) {
        let d = depth as usize;
        if d < MAX_DEPTH {
            // すでに登録されている場合は何もしない（重複を防ぐ）
            if self.killer_moves[d][0] == Some(mv) {
                return;
            }
            // 2番目を1番目にずらし、新しい手を1番目に登録
            self.killer_moves[d][1] = self.killer_moves[d][0];
            self.killer_moves[d][0] = Some(mv);
        }
    }

    /// 静的交換評価（SEE）を計算します。
    fn see(&self, position: &Position, mv: Move) -> i32 {
        if let Move::Normal { from, to, .. } = mv {
            if let (Some(attacker), Some(victim)) = (position.piece_at(from), position.piece_at(to)) {
                return get_piece_value(victim.piece_kind()) - get_piece_value(attacker.piece_kind());
            }
        }
        0
    }

    fn is_time_up(&self) -> bool {
        if let (Some(start), Some(limit)) = (self.start_time, self.time_limit) {
            start.elapsed() >= limit
        } else {
            false
        }
    }

    /// ある指し手が王手かどうかを判定します。
    fn is_check(&self, position: &Position, mv: Move) -> bool {
        // is_checkの判定のためにyasai::Positionを一時的に利用する
        let mut yasai_pos = yasai::Position::new(position.inner().clone());
        yasai_pos.do_move(mv);
        // do_move後、手番が相手に移るため、in_check()は
        // 相手玉が王手されているかを正しく判定します。
        yasai_pos.in_check()
    }

    pub fn quiescence_search( &mut self, position: &shogi_core::Position, mut alpha: f32, beta: f32, ) -> Option<f32> {
        if self.is_time_up() {
            return None;
        }

        let stand_pat_score = self.evaluator.evaluate(position);

        if stand_pat_score >= beta {
            return Some(beta);
        }
        alpha = alpha.max(stand_pat_score);

        let yasai_pos = yasai::Position::new(position.inner().clone());
        let mut moves = yasai_pos.legal_moves();
        
        moves.retain(|m| {
            // 駒を取る手か、王手をかける手のみを静止探索の対象とする
            let is_capture = if let Move::Normal { to, .. } = *m {
                position.piece_at(to).is_some()
            } else {
                false
            };
            is_capture || self.is_check(position, *m)
        });

        if moves.is_empty() {
            return Some(stand_pat_score);
        }

        self.move_ordering.sort_moves(&mut moves, position);

        let mut best_score = stand_pat_score;
        for mv in moves {
            if self.see(position, mv) < 0 {
                continue;
            }

            let mut next_position = position.clone();
            if next_position.make_move(mv).is_none() { continue; }

            self.sennichite_detector.record_position(&next_position);
            let sennichite_status = self.is_sennichite_internal(&next_position);
            let score = match sennichite_status {
                SennichiteStatus::Draw => Some(0.0),
                SennichiteStatus::PerpetualCheckLoss => Some(-f32::INFINITY),
                SennichiteStatus::None => {
                    self.quiescence_search(&next_position, -beta, -alpha).map(|s| -s)
                }
            };
            self.sennichite_detector.unrecord_last_position();

            if let Some(current_score) = score {
                if current_score > best_score {
                    best_score = current_score;
                }
                alpha = alpha.max(current_score);

                if alpha >= beta {
                    break;
                }
            } else {
                return None; // 時間切れを伝播
            }
        }
        Some(best_score)
    }

    pub fn alpha_beta_search( &mut self, position: &shogi_core::Position, depth: u8, mut alpha: f32, mut beta: f32, ) -> Option<f32> {
        if self.is_time_up() {
            return None;
        }

        if depth == 0 {
            return self.quiescence_search(position, alpha, beta);
        }

        let hash = PositionHasher::calculate_hash(position);

        // --- テーブル参照 ---
        if let Some(entry) = self.transposition_table.get(&hash) {
            if entry.depth >= depth {
                match entry.node_type {
                    NodeType::Exact => return Some(entry.score),
                    NodeType::LowerBound => alpha = alpha.max(entry.score),
                    NodeType::UpperBound => beta = beta.min(entry.score),
                }
                if alpha >= beta {
                    return Some(entry.score);
                }
            }
        }

        let yasai_pos = yasai::Position::new(position.inner().clone());
        let mut moves = yasai_pos.legal_moves();

        if moves.is_empty() {
            return Some(-f32::INFINITY);
        }

        self.move_ordering.sort_moves(&mut moves, position);

        // --- キラームーブを優先 ---
        if (depth as usize) < MAX_DEPTH {
            let killers = self.killer_moves[depth as usize];
            let mut killer_idx = 0;
            for i in 0..moves.len() {
                if killer_idx < killers.len() && killers[killer_idx] == Some(moves[i]) {
                    let killer_move = moves.remove(i);
                    moves.insert(killer_idx, killer_move);
                    killer_idx += 1;
                }
            }
        }


        let mut best_score = -f32::INFINITY;
        let mut node_type = NodeType::UpperBound;

        for mv in moves {
            let mut next_position = position.clone();
            if next_position.make_move(mv).is_none() {
                continue;
            }

            self.sennichite_detector.record_position(&next_position);
            let sennichite_status = self.is_sennichite_internal(&next_position);

            let score_result = match sennichite_status {
                SennichiteStatus::Draw => Some(0.0),
                SennichiteStatus::PerpetualCheckLoss => Some(-f32::INFINITY),
                SennichiteStatus::None => {
                    self.alpha_beta_search(&next_position, depth - 1, -beta, -alpha).map(|s| -s)
                }
            };

            self.sennichite_detector.unrecord_last_position();

            if let Some(score) = score_result {
                if score > best_score {
                    best_score = score;
                }
                if best_score > alpha {
                    alpha = best_score;
                    node_type = NodeType::Exact;
                }

                if alpha >= beta {
                    self.update_killer_moves(depth, mv);
                    self.move_ordering.update_history(&mv, position, depth as i32 * 10);
                    node_type = NodeType::LowerBound;
                    break;
                }
            } else {
                return None; // 時間切れを伝播
            }
        }

        // --- テーブル保存 ---
        // 時間切れで探索が不完全な場合は保存しない
        let entry = TranspositionEntry {
            score: best_score,
            depth,
            node_type,
        };
        self.transposition_table.insert(hash, entry);

        Some(best_score)
    }

    pub fn is_sennichite_internal(&self, position: &shogi_core::Position) -> SennichiteStatus {
        self.sennichite_detector.check_sennichite(position)
    }

    pub fn find_best_move(&mut self, position: &shogi_core::Position, max_depth: u8, time_limit_ms: Option<u64>) -> Option<Move> {
        self.transposition_table.clear();
        self.clear_killer_moves();
        self.start_time = Some(Instant::now());
        self.time_limit = time_limit_ms.map(Duration::from_millis);

        let mut best_move: Option<Move> = None;
        
        let yasai_pos = yasai::Position::new(position.inner().clone());
        let mut moves = yasai_pos.legal_moves();
        if moves.is_empty() {
            return None;
        }
        self.move_ordering.sort_moves(&mut moves, position);

        for depth in 1..=max_depth {
            let mut current_best_move_for_depth: Option<Move> = None;
            let mut best_eval_for_depth = -f32::INFINITY;
            let mut alpha = -f32::INFINITY;
            let beta = f32::INFINITY;

            let mut search_interrupted = false;

            for mv in &moves {
                if self.is_time_up() {
                    search_interrupted = true;
                    break;
                }

                let mut next_position = position.clone();
                if next_position.make_move(*mv).is_none() {
                    continue;
                }

                self.sennichite_detector.record_position(&next_position);
                let sennichite_status = self.is_sennichite_internal(&next_position);
                self.sennichite_detector.unrecord_last_position();

                let eval_result = match sennichite_status {
                    SennichiteStatus::Draw => Some(0.0),
                    SennichiteStatus::PerpetualCheckLoss => Some(-f32::INFINITY),
                    SennichiteStatus::None => {
                        self.alpha_beta_search(&next_position, depth - 1, -beta, -alpha).map(|s| -s)
                    }
                };

                if let Some(eval) = eval_result {
                    if eval > best_eval_for_depth {
                        best_eval_for_depth = eval;
                        current_best_move_for_depth = Some(*mv);
                    }
                    alpha = alpha.max(eval);
                } else {
                    search_interrupted = true;
                    break;
                }
            }

            if !search_interrupted {
                // 深さdの探索が完全に終わった場合のみ、最善手を更新
                best_move = current_best_move_for_depth;
                if let Some(bm) = best_move {
                     // 探索が完了した深さが大きいほど、履歴の重みを大きくする
                    self.move_ordering.update_history(&bm, position, depth as i32 * 20);
                }
            } else {
                // 時間切れでループを抜ける
                break;
            }
        }

        best_move
    }
}