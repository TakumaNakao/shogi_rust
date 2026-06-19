use crate::evaluation::Evaluator;
use crate::move_ordering::MoveOrdering;
use crate::position_hash::PositionHasher;
use crate::sennichite::{SennichiteDetector, SennichiteStatus};
use crate::utils::{format_move_usi, get_piece_value};
use shogi_core::Move;
use shogi_lib::Position;
use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MAX_DEPTH: usize = 64;
const TRANSPOSITION_TABLE_MAX_ENTRIES: usize = 1_000_000;
const ASPIRATION_WINDOW: f32 = 300.0;
const CHECK_MOVE_BONUS: i32 = 2_000;
const SEE_ORDERING_SCALE: i32 = 20;
const CHECK_EVASION_EXTENSION_MAX_REPLIES: usize = 3;
const USI_SCORE_CP_LIMIT: i32 = 2_000;
const USI_SCORE_CP_SOFT_START: i32 = 1_000;

fn usi_display_score_cp(score: f32) -> i32 {
    if !score.is_finite() {
        return if score.is_sign_negative() {
            -USI_SCORE_CP_LIMIT
        } else {
            USI_SCORE_CP_LIMIT
        };
    }

    let sign = if score < 0.0 { -1 } else { 1 };
    let abs_score = score.abs();
    let soft_start = USI_SCORE_CP_SOFT_START as f32;
    let limit = USI_SCORE_CP_LIMIT as f32;
    let displayed = if abs_score <= soft_start {
        abs_score
    } else {
        let tail = limit - soft_start;
        soft_start + tail * (1.0 - (-(abs_score - soft_start) / tail).exp())
    };

    sign * (displayed.round() as i32).min(USI_SCORE_CP_LIMIT)
}

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
    generation: u32,
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
    quiescence_nodes_searched: u64,
    quiescence_moves_considered: u64,
    quiescence_moves_searched: u64,
    quiescence_see_skips: u64,
    check_evasion_extensions: u64,
    emit_info: bool,
    search_generation: u32,
    stop_signal: Option<Arc<AtomicBool>>,
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
            quiescence_nodes_searched: 0,
            quiescence_moves_considered: 0,
            quiescence_moves_searched: 0,
            quiescence_see_skips: 0,
            check_evasion_extensions: 0,
            emit_info: true,
            search_generation: 0,
            stop_signal: None,
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

    pub fn set_emit_info(&mut self, emit_info: bool) {
        self.emit_info = emit_info;
    }

    pub fn set_stop_signal(&mut self, stop_signal: Option<Arc<AtomicBool>>) {
        self.stop_signal = stop_signal;
    }

    pub fn nodes_searched(&self) -> u64 {
        self.nodes_searched
    }

    pub fn quiescence_nodes_searched(&self) -> u64 {
        self.quiescence_nodes_searched
    }

    pub fn quiescence_moves_considered(&self) -> u64 {
        self.quiescence_moves_considered
    }

    pub fn quiescence_moves_searched(&self) -> u64 {
        self.quiescence_moves_searched
    }

    pub fn quiescence_see_skips(&self) -> u64 {
        self.quiescence_see_skips
    }

    pub fn check_evasion_extensions(&self) -> u64 {
        self.check_evasion_extensions
    }

    fn update_killer_moves(&mut self, depth: u8, mv: Move) {
        let d = depth as usize;
        if d < MAX_DEPTH {
            if self.killer_moves[d][0] == Some(mv) {
                return;
            }
            self.killer_moves[d][1] = self.killer_moves[d][0];
            self.killer_moves[d][0] = Some(mv);
        }
    }

    fn see(&self, position: &Position, mv: Move) -> i32 {
        if let Move::Normal { from, to, .. } = mv {
            if let (Some(attacker), Some(victim)) = (position.piece_at(from), position.piece_at(to))
            {
                return get_piece_value(victim.piece_kind())
                    - get_piece_value(attacker.piece_kind());
            }
        }
        0
    }

    fn is_time_up(&self) -> bool {
        if self
            .stop_signal
            .as_ref()
            .is_some_and(|stop_signal| stop_signal.load(Ordering::SeqCst))
        {
            return true;
        }
        if let (Some(start), Some(limit)) = (self.start_time, self.time_limit) {
            start.elapsed() >= limit
        } else {
            false
        }
    }

    fn search_ordering_score(&mut self, position: &mut Position, mv: Move) -> i32 {
        let mut score = self.move_ordering.score_move(&mv, position);
        if let Move::Normal { promote, .. } = mv {
            if promote {
                score += 4000;
            }
        }
        score += self.see(position, mv) * SEE_ORDERING_SCALE;
        if position.is_check_move(mv) {
            score += CHECK_MOVE_BONUS;
        }
        score
    }

    pub fn quiescence_search(
        &mut self,
        position: &mut Position,
        mut alpha: f32,
        beta: f32,
    ) -> Option<(f32, Vec<Move>)> {
        if self.is_time_up() {
            return None;
        }
        self.nodes_searched += 1;
        self.quiescence_nodes_searched += 1;

        let stand_pat_score = self.evaluator.evaluate(position);
        if stand_pat_score >= beta {
            return Some((beta, Vec::new()));
        }
        alpha = alpha.max(stand_pat_score);

        let mut moves = position.legal_moves();

        moves.retain(|m| {
            if let Move::Normal { to, .. } = *m {
                position.piece_at(to).is_some() || position.is_check_move(*m)
            } else {
                position.is_check_move(*m)
            }
        });

        if moves.is_empty() {
            return Some((stand_pat_score, Vec::new()));
        }
        self.quiescence_moves_considered += moves.len() as u64;

        let mut scored_moves: Vec<(Move, i32)> = moves
            .iter()
            .map(|&mv| {
                (
                    mv,
                    self.move_ordering.score_move_without_counter(&mv, position),
                )
            })
            .collect();
        scored_moves.sort_unstable_by_key(|a| -a.1);

        let mut best_score = stand_pat_score;
        for (mv, _) in scored_moves {
            if self.see(position, mv) < 0 {
                self.quiescence_see_skips += 1;
                continue;
            }

            self.quiescence_moves_searched += 1;
            position.do_move(mv);
            self.sennichite_detector.record_position(position);
            let sennichite_status = self.is_sennichite_internal(position);
            let score_result = match sennichite_status {
                SennichiteStatus::Draw => Some((0.0, Vec::new())),
                SennichiteStatus::PerpetualCheckLoss => Some((f32::INFINITY, Vec::new())),
                SennichiteStatus::None => self.quiescence_search(position, -beta, -alpha),
            };
            self.sennichite_detector.unrecord_last_position();
            position.undo_move(mv);

            if let Some((current_score, _)) = score_result {
                let negated_score = -current_score;
                if negated_score > best_score {
                    best_score = negated_score;
                }
                alpha = alpha.max(negated_score);
                if alpha >= beta {
                    break;
                }
            } else {
                return None;
            }
        }
        Some((best_score, Vec::new()))
    }

    pub fn alpha_beta_search(
        &mut self,
        position: &mut Position,
        depth: u8,
        alpha: f32,
        beta: f32,
    ) -> Option<(f32, Vec<Move>)> {
        self.alpha_beta_search_internal(position, depth, alpha, beta, 1)
    }

    fn alpha_beta_search_internal(
        &mut self,
        position: &mut Position,
        depth: u8,
        mut alpha: f32,
        mut beta: f32,
        check_evasion_extension_budget: u8,
    ) -> Option<(f32, Vec<Move>)> {
        if self.is_time_up() {
            return None;
        }
        if depth == 0 {
            return self.quiescence_search(position, alpha, beta);
        }
        self.nodes_searched += 1;

        let hash = PositionHasher::calculate_hash(position);
        let tt_entry = self.transposition_table.get(&hash).copied();
        let tt_best_move = tt_entry.and_then(|entry| entry.best_move);
        if let Some(entry) = tt_entry {
            if entry.generation == self.search_generation && entry.depth >= depth {
                match entry.node_type {
                    NodeType::Exact => {
                        return Some((entry.score, entry.best_move.map_or(Vec::new(), |m| vec![m])))
                    }
                    NodeType::LowerBound => alpha = alpha.max(entry.score),
                    NodeType::UpperBound => beta = beta.min(entry.score),
                }
                if alpha >= beta {
                    return Some((entry.score, entry.best_move.map_or(Vec::new(), |m| vec![m])));
                }
            }
        }

        let moves = position.legal_moves();
        if moves.is_empty() {
            return Some((-f32::INFINITY, Vec::new()));
        }

        let mut scored_moves = Vec::with_capacity(moves.len());
        for mv in moves.iter() {
            scored_moves.push((*mv, self.search_ordering_score(position, *mv)));
        }
        scored_moves.sort_unstable_by_key(|a| -a.1);
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
        if let Some(tt_move) = tt_best_move {
            if let Some(pos) = sorted_moves.iter().position(|&m| m == tt_move) {
                let mv = sorted_moves.remove(pos);
                sorted_moves.insert(0, mv);
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
                SennichiteStatus::PerpetualCheckLoss => Some((f32::INFINITY, Vec::new())),
                SennichiteStatus::None => {
                    let mut child_depth = depth - 1;
                    let mut child_extension_budget = check_evasion_extension_budget;
                    if depth == 1
                        && check_evasion_extension_budget > 0
                        && position.in_check()
                        && position.legal_moves_count_up_to(CHECK_EVASION_EXTENSION_MAX_REPLIES + 1)
                            <= CHECK_EVASION_EXTENSION_MAX_REPLIES
                    {
                        child_depth = 1;
                        child_extension_budget -= 1;
                        self.check_evasion_extensions += 1;
                    }
                    self.alpha_beta_search_internal(
                        position,
                        child_depth,
                        -beta,
                        -alpha,
                        child_extension_budget,
                    )
                }
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
                    self.move_ordering
                        .update_history(&mv, position, depth as i32 * 10);
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

        let entry = TranspositionEntry {
            score: best_score,
            depth,
            node_type,
            best_move,
            generation: self.search_generation,
        };
        self.transposition_table.insert(hash, entry);
        Some((best_score, final_pv))
    }

    pub fn is_sennichite_internal(&self, position: &Position) -> SennichiteStatus {
        self.sennichite_detector.check_sennichite(position)
    }

    pub fn find_best_move(
        &mut self,
        position: &mut Position,
        max_depth: u8,
        time_limit_ms: Option<u64>,
    ) -> Option<Move> {
        if self.transposition_table.len() > TRANSPOSITION_TABLE_MAX_ENTRIES {
            self.transposition_table.clear();
        }
        self.search_generation = self.search_generation.wrapping_add(1);
        if self.search_generation == 0 {
            self.search_generation = 1;
            self.transposition_table.clear();
        }
        self.clear_killer_moves();
        self.start_time = Some(Instant::now());
        self.time_limit = time_limit_ms.map(Duration::from_millis);
        self.nodes_searched = 0;
        self.quiescence_nodes_searched = 0;
        self.quiescence_moves_considered = 0;
        self.quiescence_moves_searched = 0;
        self.quiescence_see_skips = 0;
        self.check_evasion_extensions = 0;

        let moves = position.legal_moves();
        if moves.is_empty() {
            return None;
        }

        let mut scored_moves = Vec::with_capacity(moves.len());
        for mv in moves.iter() {
            scored_moves.push((*mv, self.search_ordering_score(position, *mv)));
        }
        scored_moves.sort_unstable_by_key(|a| -a.1);
        let mut sorted_moves: Vec<Move> = scored_moves.into_iter().map(|(mv, _)| mv).collect();
        let root_hash = PositionHasher::calculate_hash(position);
        if let Some(tt_move) = self
            .transposition_table
            .get(&root_hash)
            .and_then(|entry| entry.best_move)
        {
            if let Some(pos) = sorted_moves.iter().position(|&m| m == tt_move) {
                let mv = sorted_moves.remove(pos);
                sorted_moves.insert(0, mv);
            }
        }
        let mut best_move: Option<Move> = sorted_moves.first().copied();
        let mut previous_eval: Option<f32> = None;

        for depth in 1..=max_depth {
            if let Some(previous_best) = best_move {
                if let Some(pos) = sorted_moves.iter().position(|&m| m == previous_best) {
                    let mv = sorted_moves.remove(pos);
                    sorted_moves.insert(0, mv);
                }
            }
            let mut current_best_move_for_depth: Option<Move> = None;
            let mut best_eval_for_depth = -f32::INFINITY;
            let mut best_pv_for_depth: Vec<Move> = Vec::new();
            let (mut alpha, mut beta) = previous_eval
                .map(|eval| (eval - ASPIRATION_WINDOW, eval + ASPIRATION_WINDOW))
                .unwrap_or((-f32::INFINITY, f32::INFINITY));
            let aspiration_alpha = alpha;
            let aspiration_beta = beta;
            let mut search_interrupted = false;

            for mv in &sorted_moves {
                if self.is_time_up() {
                    search_interrupted = true;
                    break;
                }

                position.do_move(*mv);
                self.sennichite_detector.record_position(position);
                let sennichite_status = self.is_sennichite_internal(position);

                let eval_result = match sennichite_status {
                    SennichiteStatus::Draw => Some((0.0, Vec::new())),
                    SennichiteStatus::PerpetualCheckLoss => Some((f32::INFINITY, Vec::new())),
                    SennichiteStatus::None => {
                        if current_best_move_for_depth.is_some() && alpha.is_finite() {
                            match self.alpha_beta_search(position, depth - 1, -alpha - 1.0, -alpha)
                            {
                                Some((score, _)) if -score > alpha => {
                                    self.alpha_beta_search(position, depth - 1, -beta, -alpha)
                                }
                                narrow_result => narrow_result,
                            }
                        } else {
                            self.alpha_beta_search(position, depth - 1, -beta, -alpha)
                        }
                    }
                };
                self.sennichite_detector.unrecord_last_position();
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
                if best_eval_for_depth <= aspiration_alpha || best_eval_for_depth >= aspiration_beta
                {
                    alpha = -f32::INFINITY;
                    beta = f32::INFINITY;
                    current_best_move_for_depth = None;
                    best_eval_for_depth = -f32::INFINITY;
                    best_pv_for_depth.clear();

                    for mv in &sorted_moves {
                        if self.is_time_up() {
                            search_interrupted = true;
                            break;
                        }

                        position.do_move(*mv);
                        self.sennichite_detector.record_position(position);
                        let sennichite_status = self.is_sennichite_internal(position);

                        let eval_result = match sennichite_status {
                            SennichiteStatus::Draw => Some((0.0, Vec::new())),
                            SennichiteStatus::PerpetualCheckLoss => {
                                Some((f32::INFINITY, Vec::new()))
                            }
                            SennichiteStatus::None => {
                                self.alpha_beta_search(position, depth - 1, -beta, -alpha)
                            }
                        };
                        self.sennichite_detector.unrecord_last_position();
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
                }
            }

            if !search_interrupted {
                best_move = current_best_move_for_depth;
                previous_eval = Some(best_eval_for_depth);
                if let Some(bm) = best_move {
                    self.move_ordering
                        .update_history(&bm, position, depth as i32 * 20);
                }

                // --- infoコマンド出力 ---
                let elapsed_time = self.start_time.unwrap().elapsed().as_millis();
                let pv_string = best_pv_for_depth
                    .iter()
                    .map(|m| format_move_usi(*m))
                    .collect::<Vec<_>>()
                    .join(" ");

                // 評価値は手番視点に変換する
                let score_cp = usi_display_score_cp(best_eval_for_depth);

                if self.emit_info {
                    println!(
                        "info depth {} score cp {} time {} nodes {} pv {}",
                        depth, score_cp, elapsed_time, self.nodes_searched, pv_string
                    );
                    let _ = io::stdout().flush();
                }
                // --- ここまで ---

                if depth == max_depth {
                    break;
                }
            } else {
                break;
            }
        }
        best_move
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usi_display_score_keeps_small_values() {
        assert_eq!(0, usi_display_score_cp(0.0));
        assert_eq!(500, usi_display_score_cp(500.0));
        assert_eq!(-500, usi_display_score_cp(-500.0));
        assert_eq!(1000, usi_display_score_cp(1000.0));
    }

    #[test]
    fn usi_display_score_soft_limits_large_values() {
        let two_thousand = usi_display_score_cp(2000.0);
        let four_thousand = usi_display_score_cp(4000.0);
        assert!(two_thousand > 1000);
        assert!(four_thousand > two_thousand);
        assert!(four_thousand < USI_SCORE_CP_LIMIT);
        assert_eq!(-four_thousand, usi_display_score_cp(-4000.0));
        assert_eq!(USI_SCORE_CP_LIMIT, usi_display_score_cp(f32::INFINITY));
        assert_eq!(-USI_SCORE_CP_LIMIT, usi_display_score_cp(f32::NEG_INFINITY));
    }
}
