use super::*;
use std::time::{Duration, Instant};

impl<E: Evaluator, const HISTORY_CAPACITY: usize> ShogiAI<E, HISTORY_CAPACITY> {
    pub fn search(&mut self, position: &mut Position, limits: SearchLimits) -> SearchOutcome {
        self.last_search_failed = false;
        let best_move = self.find_best_move_with_root_offset(
            position,
            limits.max_depth,
            limits.time_limit_ms(),
            0,
            true,
        );
        self.search_outcome(best_move)
    }

    pub fn find_best_move(
        &mut self,
        position: &mut Position,
        max_depth: u8,
        time_limit_ms: Option<u64>,
    ) -> Option<Move> {
        self.search(
            position,
            SearchLimits::from_millis(max_depth, time_limit_ms),
        )
        .best_move()
    }

    pub(super) fn find_best_move_with_root_offset(
        &mut self,
        position: &mut Position,
        max_depth: u8,
        time_limit_ms: Option<u64>,
        root_offset: usize,
        primary_worker: bool,
    ) -> Option<Move> {
        self.begin_eval_context(position);
        if primary_worker
            && self
                .transposition_table
                .local_len()
                .is_some_and(|len| len > TRANSPOSITION_TABLE_MAX_ENTRIES)
        {
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
        self.next_time_check_nodes = 0;
        self.nodes_searched = 0;
        self.quiescence_nodes_searched = 0;
        self.quiescence_moves_considered = 0;
        self.quiescence_moves_generated = 0;
        self.quiescence_moves_discarded = 0;
        self.quiescence_moves_searched = 0;
        self.quiescence_see_skips = 0;
        self.quiescence_terminal_mates = 0;
        self.check_evasion_extensions = 0;
        self.aspiration_fail_lows = 0;
        self.aspiration_fail_highs = 0;
        self.aspiration_researches = 0;
        self.last_completed_depth = 0;
        self.last_root_score = None;
        self.last_pv.clear();

        let moves = position.legal_moves();
        if moves.is_empty() {
            return None;
        }

        let mut scored_moves = Vec::with_capacity(moves.len());
        for mv in moves.iter() {
            scored_moves.push((*mv, self.search_ordering_score(position, *mv)));
        }
        scored_moves.sort_unstable_by_key(|a| -a.1);
        let mut sorted_moves = scored_moves;
        let root_hash = PositionHasher::calculate_hash(position);
        if let Some(tt_move) = self
            .transposition_table
            .get(root_hash)
            .and_then(|entry| entry.best_move)
        {
            if let Some(pos) = sorted_moves.iter().position(|&(m, _)| m == tt_move) {
                let mv = sorted_moves.remove(pos);
                sorted_moves.insert(0, mv);
            }
        }
        let mut best_move = sorted_moves.first().map(|&(mv, _)| mv);
        let mut previous_eval = None;

        for depth in 1..=max_depth {
            if let Some(previous_best) = best_move {
                if let Some(pos) = sorted_moves.iter().position(|&(m, _)| m == previous_best) {
                    let mv = sorted_moves.remove(pos);
                    sorted_moves.insert(0, mv);
                }
            }
            if root_offset > 0 && sorted_moves.len() > 1 {
                let offset = (root_offset + depth as usize - 1) % sorted_moves.len();
                sorted_moves.rotate_left(offset);
            }
            let mut current_best_move_for_depth = None;
            let mut best_eval_for_depth = -f32::INFINITY;
            let mut best_pv_for_depth = Vec::new();
            let (mut alpha, mut beta) = previous_eval
                .map(|eval| (eval - ASPIRATION_WINDOW, eval + ASPIRATION_WINDOW))
                .unwrap_or((-f32::INFINITY, f32::INFINITY));
            let aspiration_alpha = alpha;
            let aspiration_beta = beta;
            let mut search_interrupted = false;

            for &(mv, _) in &sorted_moves {
                if self.is_time_up() {
                    search_interrupted = true;
                    break;
                }

                let moved_by = position.side_to_move();
                self.make_move(position, mv);
                self.sennichite_detector
                    .record_position_after_move(position, moved_by);
                let eval_result = if let Some(score) = self.sennichite_score(position) {
                    Some((score, Vec::new()))
                } else if current_best_move_for_depth.is_some() && alpha.is_finite() {
                    match self.search_root_child(position, depth - 1, -alpha - 1.0, -alpha) {
                        Some((score, _)) if -score > alpha => {
                            self.search_root_child(position, depth - 1, -beta, -alpha)
                        }
                        narrow_result => narrow_result,
                    }
                } else {
                    self.search_root_child(position, depth - 1, -beta, -alpha)
                };
                self.sennichite_detector.unrecord_last_position();
                self.undo_move(position, mv);

                if let Some((eval, pv)) = eval_result {
                    let current_eval = -eval;
                    if current_eval > best_eval_for_depth {
                        best_eval_for_depth = current_eval;
                        current_best_move_for_depth = Some(mv);
                        let mut current_pv = vec![mv];
                        current_pv.extend(pv);
                        best_pv_for_depth = current_pv;
                    }
                    alpha = alpha.max(current_eval);
                    if alpha >= beta {
                        break;
                    }
                } else {
                    search_interrupted = true;
                    break;
                }
            }

            if !search_interrupted
                && (best_eval_for_depth <= aspiration_alpha
                    || best_eval_for_depth >= aspiration_beta)
            {
                if aspiration_alpha.is_finite() && aspiration_beta.is_finite() {
                    self.aspiration_researches += 1;
                    if best_eval_for_depth <= aspiration_alpha {
                        self.aspiration_fail_lows += 1;
                    } else {
                        self.aspiration_fail_highs += 1;
                    }
                }
                alpha = -f32::INFINITY;
                beta = f32::INFINITY;
                current_best_move_for_depth = None;
                best_eval_for_depth = -f32::INFINITY;
                best_pv_for_depth.clear();

                for &(mv, _) in &sorted_moves {
                    if self.is_time_up() {
                        search_interrupted = true;
                        break;
                    }

                    let moved_by = position.side_to_move();
                    self.make_move(position, mv);
                    self.sennichite_detector
                        .record_position_after_move(position, moved_by);
                    let eval_result = if let Some(score) = self.sennichite_score(position) {
                        Some((score, Vec::new()))
                    } else {
                        self.search_root_child(position, depth - 1, -beta, -alpha)
                    };
                    self.sennichite_detector.unrecord_last_position();
                    self.undo_move(position, mv);

                    if let Some((eval, pv)) = eval_result {
                        let current_eval = -eval;
                        if current_eval > best_eval_for_depth {
                            best_eval_for_depth = current_eval;
                            current_best_move_for_depth = Some(mv);
                            let mut current_pv = vec![mv];
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

            if !search_interrupted {
                let Some(current_best_move) = current_best_move_for_depth else {
                    break;
                };
                if best_pv_for_depth.is_empty() {
                    break;
                }
                best_move = Some(current_best_move);
                previous_eval = Some(best_eval_for_depth);
                self.last_completed_depth = depth;
                if let Some(best_move) = best_move {
                    self.move_ordering
                        .update_history(&best_move, position, depth as i32 * 20);
                }

                let elapsed_time = self.start_time.unwrap().elapsed().as_millis();
                self.last_root_score = Some(best_eval_for_depth);
                self.last_pv = best_pv_for_depth;

                if let Some(observer) = self.observer.clone() {
                    observer.on_info(&SearchInfo {
                        depth,
                        root_score: best_eval_for_depth,
                        elapsed: Duration::from_millis(elapsed_time.min(u64::MAX as u128) as u64),
                        stats: self.search_stats(),
                        pv: self.last_pv.clone(),
                    });
                }

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
