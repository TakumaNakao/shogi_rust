use super::*;

impl<E: Evaluator, const HISTORY_CAPACITY: usize> ShogiAI<E, HISTORY_CAPACITY> {
    pub fn alpha_beta_search(
        &mut self,
        position: &mut Position,
        depth: u8,
        alpha: f32,
        beta: f32,
    ) -> Option<(f32, Vec<Move>)> {
        if self.eval_context.is_none() {
            self.begin_eval_context(position);
        }
        self.alpha_beta_search_internal(position, depth, alpha, beta, 1, 0, None)
    }

    pub(super) fn search_root_child(
        &mut self,
        position: &mut Position,
        depth: u8,
        alpha: f32,
        beta: f32,
    ) -> Option<(f32, Vec<Move>)> {
        self.alpha_beta_search_internal(position, depth, alpha, beta, 1, 1, None)
    }

    fn alpha_beta_search_internal(
        &mut self,
        position: &mut Position,
        depth: u8,
        mut alpha: f32,
        mut beta: f32,
        check_evasion_extension_budget: u8,
        ply_from_root: u16,
        precomputed_moves: Option<LegalMoves>,
    ) -> Option<(f32, Vec<Move>)> {
        assert!(
            alpha < beta,
            "invalid alpha-beta window: alpha={alpha}, beta={beta}"
        );
        if self.is_time_up() {
            return None;
        }
        if depth == 0 {
            return self.quiescence_search_internal(
                position,
                alpha,
                beta,
                ply_from_root,
                0,
                precomputed_moves,
            );
        }
        self.nodes_searched += 1;

        let hash = PositionHasher::calculate_hash(position);
        let tt_entry = self.transposition_table.get(hash);
        let tt_best_move = tt_entry.and_then(|entry| entry.best_move);
        if let Some(entry) = tt_entry {
            if entry.generation == self.search_generation && entry.depth >= depth {
                let tt_score = score_from_tt(entry.score, ply_from_root);
                match entry.node_type {
                    NodeType::Exact => {
                        return Some((tt_score, entry.best_move.map_or(Vec::new(), |m| vec![m])))
                    }
                    NodeType::LowerBound => alpha = alpha.max(tt_score),
                    NodeType::UpperBound => beta = beta.min(tt_score),
                }
                if alpha >= beta {
                    return Some((tt_score, entry.best_move.map_or(Vec::new(), |m| vec![m])));
                }
            }
        }

        let moves = precomputed_moves.unwrap_or_else(|| position.legal_moves());
        if moves.is_empty() {
            return Some((mate_loss_score(ply_from_root), Vec::new()));
        }

        let mut scored_moves = Vec::with_capacity(moves.len());
        for mv in moves.iter() {
            scored_moves.push((*mv, self.search_ordering_score(position, *mv)));
        }
        scored_moves.sort_unstable_by_key(|a| -a.1);
        let mut sorted_moves = scored_moves;

        if (depth as usize) < MAX_DEPTH {
            let killers = self.killer_moves[depth as usize];
            for &killer in killers.iter().flatten().rev() {
                if let Some(pos) = sorted_moves.iter().position(|&(m, _)| m == killer) {
                    let mv = sorted_moves.remove(pos);
                    sorted_moves.insert(0, mv);
                }
            }
        }
        if let Some(tt_move) = tt_best_move {
            if let Some(pos) = sorted_moves.iter().position(|&(m, _)| m == tt_move) {
                let mv = sorted_moves.remove(pos);
                sorted_moves.insert(0, mv);
            }
        }

        let mut best_score = -f32::INFINITY;
        let mut best_move = None;
        let mut best_pv = Vec::new();
        let mut node_type = NodeType::UpperBound;

        for (mv, _) in sorted_moves {
            self.make_move(position, mv);
            self.sennichite_detector.record_position(position);
            let sennichite_status = self.is_sennichite_internal(position);
            let search_result = match sennichite_status {
                SennichiteStatus::Draw => Some((0.0, Vec::new())),
                SennichiteStatus::PerpetualCheckLoss => Some((REPETITION_WIN_SCORE, Vec::new())),
                SennichiteStatus::None => {
                    let mut child_depth = depth - 1;
                    let mut child_extension_budget = check_evasion_extension_budget;
                    let mut child_precomputed_moves = None;
                    if depth == 1 && check_evasion_extension_budget > 0 && position.in_check() {
                        let child_moves = position.legal_moves();
                        if child_moves.len() <= CHECK_EVASION_EXTENSION_MAX_REPLIES {
                            child_depth = 1;
                            child_extension_budget -= 1;
                            self.check_evasion_extensions += 1;
                        }
                        child_precomputed_moves = Some(child_moves);
                    }
                    self.alpha_beta_search_internal(
                        position,
                        child_depth,
                        -beta,
                        -alpha,
                        child_extension_budget,
                        ply_from_root.saturating_add(1),
                        child_precomputed_moves,
                    )
                }
            };
            self.sennichite_detector.unrecord_last_position();
            self.undo_move(position, mv);

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
        if let Some(best_move) = best_move {
            final_pv.push(best_move);
            final_pv.extend(best_pv);
        }

        if !is_history_dependent_score(best_score) {
            self.transposition_table.insert(
                hash,
                TranspositionEntry {
                    score: score_to_tt(best_score, ply_from_root),
                    depth,
                    node_type,
                    best_move,
                    generation: self.search_generation,
                },
            );
        }
        Some((best_score, final_pv))
    }
}
