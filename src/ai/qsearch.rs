use super::*;

/// Describes why a training qsearch trace stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QsearchLeafKind {
    /// A normal stand-pat or no-tactical-move evaluation.
    StaticEvaluation,
    /// A raw evaluation taken at the deterministic qsearch ply cap.
    MaxPlyEvaluation,
    /// A checkmated terminal position. No raw evaluation gradient exists.
    Mate,
    /// A history-dependent repetition score. No raw evaluation gradient exists.
    Repetition,
    /// A fail-hard alpha-beta bound rather than an exact leaf value.
    FailHardBound,
}

/// Training-only qsearch result that retains the exact principal leaf.
///
/// Normal game search continues to expose the historical empty qsearch PV.
#[derive(Clone)]
pub struct QsearchLeafTrace {
    pub score: f32,
    pub pv: Vec<Move>,
    pub leaf: Option<Position>,
    pub kind: QsearchLeafKind,
}

impl QsearchLeafTrace {
    fn evaluated(score: f32, position: &Position, kind: QsearchLeafKind) -> Self {
        Self {
            score,
            pv: Vec::new(),
            leaf: Some(position.clone()),
            kind,
        }
    }

    fn terminal(score: f32, kind: QsearchLeafKind) -> Self {
        Self {
            score,
            pv: Vec::new(),
            leaf: None,
            kind,
        }
    }

    pub fn has_exact_evaluation_leaf(&self) -> bool {
        matches!(
            self.kind,
            QsearchLeafKind::StaticEvaluation | QsearchLeafKind::MaxPlyEvaluation
        ) && self.leaf.is_some()
    }
}

impl<E: Evaluator, const HISTORY_CAPACITY: usize> ShogiAI<E, HISTORY_CAPACITY> {
    pub fn quiescence_search(
        &mut self,
        position: &mut Position,
        alpha: f32,
        beta: f32,
    ) -> Option<(f32, Vec<Move>)> {
        self.quiescence_search_internal(position, alpha, beta, 0, 0, None)
    }

    /// Forces one legal root move and returns the qsearch principal leaf.
    ///
    /// This is a training API. It does not reset counters or game history, and
    /// always restores `position` and the incremental evaluation context.
    pub fn forced_move_qsearch_leaf_trace(
        &mut self,
        position: &mut Position,
        root_move: Move,
    ) -> Option<QsearchLeafTrace> {
        if !position.legal_moves().contains(&root_move) {
            return None;
        }
        self.begin_eval_context(position);
        // A preceding bounded alpha-beta search may have left its node/time
        // guard at the cap.  The leaf probe is an explicitly separate,
        // bounded training operation, so it must not inherit that guard.
        let previous_limits = (
            self.start_time,
            self.time_limit,
            self.node_limit,
            self.next_time_check_nodes,
        );
        self.start_time = None;
        self.time_limit = None;
        self.node_limit = None;
        self.next_time_check_nodes = 0;
        let moved_by = position.side_to_move();
        self.make_move(position, root_move);
        self.sennichite_detector
            .record_position_after_move(position, moved_by);
        let result = if let Some(score) = self.sennichite_score(position) {
            Some(QsearchLeafTrace::terminal(
                score,
                QsearchLeafKind::Repetition,
            ))
        } else {
            self.quiescence_search_trace_internal(
                position,
                f32::NEG_INFINITY,
                f32::INFINITY,
                1,
                0,
                None,
            )
        };
        self.sennichite_detector.unrecord_last_position();
        self.undo_move(position, root_move);

        (
            self.start_time,
            self.time_limit,
            self.node_limit,
            self.next_time_check_nodes,
        ) = previous_limits;
        result.map(|mut trace| {
            trace.score = -trace.score;
            trace.pv.insert(0, root_move);
            trace
        })
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
        alpha: f32,
        beta: f32,
        ply_from_root: u16,
        quiescence_ply: u8,
        precomputed_moves: Option<LegalMoves>,
    ) -> Option<(f32, Vec<Move>)> {
        self.quiescence_search_trace_internal(
            position,
            alpha,
            beta,
            ply_from_root,
            quiescence_ply,
            precomputed_moves,
        )
        // Preserve the normal search fingerprint: qsearch historically did not
        // contribute moves to the public PV.
        .map(|trace| (trace.score, Vec::new()))
    }

    fn quiescence_search_trace_internal(
        &mut self,
        position: &mut Position,
        mut alpha: f32,
        beta: f32,
        ply_from_root: u16,
        quiescence_ply: u8,
        precomputed_moves: Option<LegalMoves>,
    ) -> Option<QsearchLeafTrace> {
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
        let stand_pat = if in_check {
            None
        } else {
            let score = self.evaluate_position(position);
            if score >= beta {
                return Some(QsearchLeafTrace::terminal(
                    beta,
                    QsearchLeafKind::FailHardBound,
                ));
            }
            alpha = alpha.max(score);
            Some(QsearchLeafTrace::evaluated(
                score,
                position,
                QsearchLeafKind::StaticEvaluation,
            ))
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
                return Some(QsearchLeafTrace::terminal(
                    mate_loss_score(ply_from_root),
                    QsearchLeafKind::Mate,
                ));
            }
            return stand_pat;
        }
        self.quiescence_moves_considered += moves.len() as u64;

        let mut best = stand_pat;
        let mut best_score = best.as_ref().map_or(-f32::INFINITY, |trace| trace.score);
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
            let moved_by = position.side_to_move();
            self.make_move(position, mv);
            self.sennichite_detector
                .record_position_after_move(position, moved_by);
            let score_result = if let Some(score) = self.sennichite_score(position) {
                Some(QsearchLeafTrace::terminal(
                    score,
                    QsearchLeafKind::Repetition,
                ))
            } else if quiescence_ply >= MAX_QUIESCENCE_PLY {
                if position.in_check() && !position.has_legal_evasion() {
                    Some(QsearchLeafTrace::terminal(
                        mate_loss_score(ply_from_root.saturating_add(1)),
                        QsearchLeafKind::Mate,
                    ))
                } else {
                    Some(QsearchLeafTrace::evaluated(
                        self.evaluate_position(position),
                        position,
                        QsearchLeafKind::MaxPlyEvaluation,
                    ))
                }
            } else {
                self.quiescence_search_trace_internal(
                    position,
                    -beta,
                    -alpha,
                    ply_from_root.saturating_add(1),
                    quiescence_ply.saturating_add(1),
                    None,
                )
            };
            self.sennichite_detector.unrecord_last_position();
            self.undo_move(position, mv);

            let Some(mut child) = score_result else {
                return None;
            };
            let negated_score = -child.score;
            if negated_score > best_score {
                best_score = negated_score;
                child.score = negated_score;
                child.pv.insert(0, mv);
                best = Some(child);
            }
            alpha = alpha.max(negated_score);
            if alpha >= beta {
                if let Some(trace) = best.as_mut() {
                    trace.kind = QsearchLeafKind::FailHardBound;
                    trace.leaf = None;
                }
                break;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::position_from_sfen_or_usi;
    use shogi_core::Color;

    #[derive(Clone, Copy)]
    struct MaxPlyFixtureEvaluator;

    impl Evaluator for MaxPlyFixtureEvaluator {
        fn evaluate(&self, position: &Position) -> f32 {
            if position.side_to_move() == Color::Black {
                0.0
            } else {
                -100.0
            }
        }
    }

    #[test]
    fn qsearch_trace_classifies_the_deterministic_max_ply_leaf() {
        let mut position = position_from_sfen_or_usi("9/9/4k4/4r4/4P4/9/9/9/4K4 b - 1")
            .expect("valid capture fixture");
        let root_sfen = position.to_sfen_owned();
        let mut ai = ShogiAI::<_, 256>::new(MaxPlyFixtureEvaluator);
        ai.begin_eval_context(&position);

        let trace = ai
            .quiescence_search_trace_internal(
                &mut position,
                f32::NEG_INFINITY,
                f32::INFINITY,
                0,
                MAX_QUIESCENCE_PLY,
                None,
            )
            .expect("qsearch");

        assert_eq!(QsearchLeafKind::MaxPlyEvaluation, trace.kind);
        assert!(trace.has_exact_evaluation_leaf());
        assert!(!trace.pv.is_empty());
        assert_eq!(root_sfen, position.to_sfen_owned());
    }
}
