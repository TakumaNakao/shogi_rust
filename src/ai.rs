use std::collections::HashMap;
use shogi_core::{Move};
use shogi_lib::Position;
use crate::evaluation::{Evaluator};
use crate::move_ordering::MoveOrdering;
use crate::position_hash::PositionHasher;
use crate::sennichite::{SennichiteDetector, SennichiteStatus};
use crate::utils::{format_move_usi, get_piece_value};
use std::time::{Duration, Instant};

const MAX_DEPTH: usize = 64;

/// トランスポジションテーブルに格納する評価値の種類
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NodeType {
    Exact,
    LowerBound,
    UpperBound,
}

/// トランスポジションテーブルのエントリ
#[derive(Clone, Copy, Debug)]
struct TranspositionEntry {
    score: f32,
    depth: u8,
    node_type: NodeType,
    best_move: Option<Move>,
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
    nodes_searched: u64,
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
            nodes_searched: 0,
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
            if self.killer_moves[d][0] == Some(mv) { return; }
            self.killer_moves[d][1] = self.killer_moves[d][0];
            self.killer_moves[d][0] = Some(mv);
        }
    }

    fn see(&self, position: &Position, mv: Move) -> i32 {
        if let Move::Normal { from, to, .. } = mv {
            if let (Some(attacker), Some(victim)) = (position.piece_at(from), position.piece_at(to)) {
                return get_piece_value(victim.piece_kind()) - get_piece_value(attacker.piece_kind());
            }
        }
        0
    }

    fn evaluate_move_safety(&self, position: &mut Position, mv: Move) -> i32 {
        let mut score = 0;

        // 1. 自分の手がタダ捨てになっていないかチェック
        position.do_move(mv);
        // 相手の手番にして合法手を生成
        let opponent_moves = position.legal_moves();
        for opponent_move in opponent_moves {
            if let Move::Normal { to, .. } = opponent_move {
                // mvで動かした駒が取り返されるか
                if to == mv.to() {
                    // 相手が取り返す手で駒損しないかチェック
                    if self.see(position, opponent_move) > 0 {
                        score -= 10000; // 大きなペナルティ
                        break;
                    }
                }
            }
        }
        position.undo_move(mv); // 盤面を元に戻す（自分の手番に戻る）


        // 2. 相手の直前の手による脅威に対応しているかチェック
        if let Some(last_move) = position.last_move() {
            if let Move::Normal { to: attacker_sq, .. } = last_move {
                // 相手の手番に切り替えて、脅威となる手（駒を取る手）を探す
                position.switch_turn();
                let threats = position.legal_moves();
                position.switch_turn(); // すぐに手番を元に戻す

                for threat in threats {
                    if let Move::Normal { from, to: victim_sq, .. } = threat {
                        // 相手が直前に動かした駒(attacker_sq)からの脅威か？
                        // そして、その手で駒を取れるか？
                        if from == attacker_sq && position.piece_at(victim_sq).is_some() {
                            // 脅威を発見！
                            // この脅威に対応する手（mv）にボーナスを与える
                            
                            // a. 脅威となっている相手の駒(attacker_sq)を取る手か？
                            if mv.to() == attacker_sq {
                                score += 10000;
                            } 
                            // b. 取られそうな自分の駒(victim_sq)を動かす手か？
                            else if mv.from() == Some(victim_sq) {
                                score += 8000;
                            }
                            // c. (発展) 間に駒を打って合駒する手か？
                            //    これは少し複雑なので、まずはa, bから実装するのがおすすめです。

                            // 重要な駒が取られる脅威ほど高く評価したい場合、
                            // victimの価値をスコアに加えることもできます。
                            // if let Some(victim_piece) = position.piece_at(victim_sq) {
                            //     score += get_piece_value(victim_piece.piece_kind()) as i32 * 10;
                            // }

                            // 脅威を一つ見つけたら、他の脅威は一旦無視してループを抜ける
                            break;
                        }
                    }
                }
            }
        }
        score
    }


    fn is_time_up(&self) -> bool {
        if let (Some(start), Some(limit)) = (self.start_time, self.time_limit) {
            start.elapsed() >= limit
        } else {
            false
        }
    }

    fn is_check(&self, position: &mut Position, mv: Move) -> bool {
        position.do_move(mv);
        let is_in_check = position.in_check();
        position.undo_move(mv);
        is_in_check
    }

    pub fn quiescence_search(&mut self, position: &mut Position, mut alpha: f32, beta: f32) -> Option<(f32, Vec<Move>)> {
        if self.is_time_up() { return None; }
        self.nodes_searched += 1;

        let stand_pat_score = self.evaluator.evaluate(position);
        if stand_pat_score >= beta { return Some((beta, Vec::new())); }
        alpha = alpha.max(stand_pat_score);

        let mut moves = position.legal_moves();
        
        moves.retain(|m| {
            if let Move::Normal { to, .. } = *m {
                position.piece_at(to).is_some() || self.is_check(position, *m)
            } else {
                self.is_check(position, *m)
            }
        });

        if moves.is_empty() { return Some((stand_pat_score, Vec::new())); }

        let mut scored_moves: Vec<(Move, i32)> = moves.iter().map(|&mv| (mv, self.move_ordering.score_move(&mv, position))).collect();
        scored_moves.sort_by_key(|a| -a.1);

        let mut best_score = stand_pat_score;
        for (mv, _) in scored_moves {
            if self.see(position, mv) < 0 {
                continue;
            }

            position.do_move(mv);
            self.sennichite_detector.record_position(position);
            let sennichite_status = self.is_sennichite_internal(position);
            let score_result = match sennichite_status {
                SennichiteStatus::Draw => Some((0.0, Vec::new())),
                SennichiteStatus::PerpetualCheckLoss => Some((-f32::INFINITY, Vec::new())),
                SennichiteStatus::None => self.quiescence_search(position, -beta, -alpha),
            };
            self.sennichite_detector.unrecord_last_position();
            position.undo_move(mv);

            if let Some((current_score, _)) = score_result {
                let negated_score = -current_score;
                if negated_score > best_score { best_score = negated_score; }
                alpha = alpha.max(negated_score);
                if alpha >= beta { break; }
            } else {
                return None;
            }
        }
        Some((best_score, Vec::new()))
    }

    pub fn alpha_beta_search(&mut self, position: &mut Position, depth: u8, mut alpha: f32, mut beta: f32) -> Option<(f32, Vec<Move>)> {
        if self.is_time_up() { return None; }
        if depth == 0 { return self.quiescence_search(position, alpha, beta); }
        self.nodes_searched += 1;

        let hash = PositionHasher::calculate_hash(position);
        if let Some(entry) = self.transposition_table.get(&hash) {
            if entry.depth >= depth {
                match entry.node_type {
                    NodeType::Exact => return Some((entry.score, entry.best_move.map_or(Vec::new(), |m| vec![m]))),
                    NodeType::LowerBound => alpha = alpha.max(entry.score),
                    NodeType::UpperBound => beta = beta.min(entry.score),
                }
                if alpha >= beta { return Some((entry.score, entry.best_move.map_or(Vec::new(), |m| vec![m]))); }
            }
        }

        let moves = position.legal_moves();
        if moves.is_empty() { return Some((-f32::INFINITY, Vec::new())); }

        let mut scored_moves: Vec<(Move, i32)> = moves.iter().map(|mv| {
            let mut score = self.move_ordering.score_move(mv, position);
            score += self.evaluate_move_safety(position, *mv);
            if let Move::Normal { promote, .. } = mv {
                if *promote {
                    score += 4000;
                }
            }
            (*mv, score)
        }).collect();
        scored_moves.sort_by_key(|a| -a.1);
        let mut sorted_moves: Vec<Move> = scored_moves.into_iter().map(|(mv, _)| mv).collect();

        if (depth as usize) < MAX_DEPTH {
            let killers = self.killer_moves[depth as usize];
            for &killer in killers.iter().flatten().rev() {
                if let Some(pos) = sorted_moves.iter().position(|&m| m == killer) {
                    let mv = sorted_moves.remove(pos);
                    sorted_moves.insert(0, mv);
                }
            }
        }

        let mut best_score = -f32::INFINITY;
        let mut best_move: Option<Move> = None;
        let mut best_pv = Vec::new();
        let mut node_type = NodeType::UpperBound;

        for mv in sorted_moves {
            position.do_move(mv);
            self.sennichite_detector.record_position(position);
            let sennichite_status = self.is_sennichite_internal(position);
            let search_result = match sennichite_status {
                SennichiteStatus::Draw => Some((0.0, Vec::new())),
                SennichiteStatus::PerpetualCheckLoss => Some((-f32::INFINITY, Vec::new())),
                SennichiteStatus::None => self.alpha_beta_search(position, depth - 1, -beta, -alpha),
            };
            self.sennichite_detector.unrecord_last_position();
            position.undo_move(mv);

            if let Some((score, pv)) = search_result {
                let current_score = -score;
                if current_score > best_score {
                    best_score = current_score;
                    best_move = Some(mv);
                    best_pv = pv;
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
                return None;
            }
        }
        
        let mut final_pv = Vec::new();
        if let Some(bm) = best_move {
            final_pv.push(bm);
            final_pv.extend(best_pv);
        }

        let entry = TranspositionEntry { score: best_score, depth, node_type, best_move };
        self.transposition_table.insert(hash, entry);
        Some((best_score, final_pv))
    }

    pub fn is_sennichite_internal(&self, position: &Position) -> SennichiteStatus {
        self.sennichite_detector.check_sennichite(position)
    }

    pub fn find_best_move(&mut self, position: &mut Position, max_depth: u8, time_limit_ms: Option<u64>) -> Option<Move> {
        self.transposition_table.clear();
        self.clear_killer_moves();
        self.start_time = Some(Instant::now());
        self.time_limit = time_limit_ms.map(Duration::from_millis);
        self.nodes_searched = 0;

        let mut best_move: Option<Move> = None;
        let moves = position.legal_moves();
        if moves.is_empty() { return None; }

        let mut scored_moves: Vec<(Move, i32)> = moves.iter().map(|mv| {
            let mut score = self.move_ordering.score_move(mv, position);
            score += self.evaluate_move_safety(position, *mv);
            if let Move::Normal { promote, .. } = mv {
                if *promote {
                    score += 4000;
                }
            }
            (*mv, score)
        }).collect();
        scored_moves.sort_by_key(|a| -a.1);
        let sorted_moves: Vec<Move> = scored_moves.into_iter().map(|(mv, _)| mv).collect();

        for depth in 1..=max_depth {
            let mut current_best_move_for_depth: Option<Move> = None;
            let mut best_eval_for_depth = -f32::INFINITY;
            let mut best_pv_for_depth: Vec<Move> = Vec::new();
            let mut alpha = -f32::INFINITY;
            let beta = f32::INFINITY;
            let mut search_interrupted = false;

            for mv in &sorted_moves {
                if self.is_time_up() {
                    search_interrupted = true;
                    break;
                }

                position.do_move(*mv);
                self.sennichite_detector.record_position(position);
                let sennichite_status = self.is_sennichite_internal(position);
                self.sennichite_detector.unrecord_last_position();

                let eval_result = match sennichite_status {
                    SennichiteStatus::Draw => Some((0.0, Vec::new())),
                    SennichiteStatus::PerpetualCheckLoss => Some((-f32::INFINITY, Vec::new())),
                    SennichiteStatus::None => self.alpha_beta_search(position, depth - 1, -beta, -alpha),
                };
                position.undo_move(*mv);

                if let Some((eval, pv)) = eval_result {
                    let current_eval = -eval;
                    if current_eval > best_eval_for_depth {
                        best_eval_for_depth = current_eval;
                        current_best_move_for_depth = Some(*mv);
                        let mut current_pv = vec![*mv];
                        current_pv.extend(pv);
                        best_pv_for_depth = current_pv;
                    }
                    alpha = alpha.max(current_eval);
                } else {
                    search_interrupted = true;
                    break;
                }
            }

            if !search_interrupted {
                best_move = current_best_move_for_depth;
                if let Some(bm) = best_move {
                    self.move_ordering.update_history(&bm, position, depth as i32 * 20);
                }
                
                // --- infoコマンド出力 ---
                let elapsed_time = self.start_time.unwrap().elapsed().as_millis();
                let pv_string = best_pv_for_depth.iter().map(|m| format_move_usi(*m)).collect::<Vec<_>>().join(" ");
                
                // 評価値は手番視点に変換する
                let score_cp = (best_eval_for_depth * 100.0) as i32;

                println!(
                    "info depth {} score cp {} time {} nodes {} pv {}",
                    depth,
                    score_cp,
                    elapsed_time,
                    self.nodes_searched,
                    pv_string
                );
                // --- ここまで ---

            } else {
                break;
            }
        }
        best_move
    }
}
