use super::*;

impl<E: Evaluator, const HISTORY_CAPACITY: usize> ShogiAI<E, HISTORY_CAPACITY> {
    pub fn quiescence_search(
        &mut self,
        position: &mut Position,
        alpha: f32,
        beta: f32,
    ) -> Option<(f32, Vec<Move>)> {
        self.quiescence_search_internal(position, alpha, beta, 0, 0, None)
    }

    fn filter_quiescence_moves_from_legal(
        position: &Position,
        mut moves: LegalMoves,
    ) -> (LegalMoves, usize) {
        let generated = moves.len();
        moves.retain(|m| {
            if let Move::Normal { to, .. } = *m {
                position.piece_at(to).is_some() || position.is_check_move(*m)
            } else {
                position.is_check_move(*m)
            }
        });
        (moves, generated)
    }

    pub(super) fn quiescence_search_internal(
        &mut self,
        position: &mut Position,
        mut alpha: f32,
        beta: f32,
        ply_from_root: u16,
        quiescence_ply: u8,
        precomputed_moves: Option<LegalMoves>,
    ) -> Option<(f32, Vec<Move>)> {
        assert!(
            alpha < beta,
            "invalid quiescence window: alpha={alpha}, beta={beta}"
        );
        if self.is_time_up() {
            return None;
        }
        self.nodes_searched += 1;
        self.quiescence_nodes_searched += 1;

        let in_check = position.in_check();
        let stand_pat_score = if in_check {
            None
        } else {
            let score = self.evaluate_position(position);
            if score >= beta {
                return Some((beta, Vec::new()));
            }
            alpha = alpha.max(score);
            Some(score)
        };
        let (moves, generated_moves) = if in_check {
            let moves = precomputed_moves.unwrap_or_else(|| position.legal_moves());
            let generated = moves.len();
            (moves, generated)
        } else {
            precomputed_moves
                .map(|moves| Self::filter_quiescence_moves_from_legal(position, moves))
                .unwrap_or_else(|| position.legal_quiescence_moves_with_generated_count())
        };

        self.quiescence_moves_generated += generated_moves as u64;
        self.quiescence_moves_discarded += (generated_moves - moves.len()) as u64;
        if moves.is_empty() {
            if in_check {
                self.quiescence_terminal_mates += 1;
                return Some((mate_loss_score(ply_from_root), Vec::new()));
            }
            return Some((
                stand_pat_score.expect("non-check position has stand-pat"),
                Vec::new(),
            ));
        }
        self.quiescence_moves_considered += moves.len() as u64;

        let mut best_score = stand_pat_score.unwrap_or(-f32::INFINITY);
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

        for (mv, _) in scored_moves {
            if !in_check && self.see(position, mv) < 0 {
                self.quiescence_see_skips += 1;
                continue;
            }

            self.quiescence_moves_searched += 1;
            self.make_move(position, mv);
            self.sennichite_detector.record_position(position);
            let sennichite_status = self.is_sennichite_internal(position);
            let score_result = match sennichite_status {
                SennichiteStatus::Draw => Some((0.0, Vec::new())),
                SennichiteStatus::PerpetualCheckLoss => Some((REPETITION_WIN_SCORE, Vec::new())),
                SennichiteStatus::None => {
                    if quiescence_ply >= MAX_QUIESCENCE_PLY {
                        if position.in_check() && !position.has_legal_evasion() {
                            Some((mate_loss_score(ply_from_root.saturating_add(1)), Vec::new()))
                        } else {
                            Some((self.evaluate_position(position), Vec::new()))
                        }
                    } else {
                        self.quiescence_search_internal(
                            position,
                            -beta,
                            -alpha,
                            ply_from_root.saturating_add(1),
                            quiescence_ply.saturating_add(1),
                            None,
                        )
                    }
                }
            };
            self.sennichite_detector.unrecord_last_position();
            self.undo_move(position, mv);

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
}
