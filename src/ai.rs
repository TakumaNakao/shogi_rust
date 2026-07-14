use crate::evaluation::Evaluator;
use crate::mate_search::{
    MateSearchLimits, MateSearchResult, MateSearchStopReason, MateSearcher, MAX_MATE_HORIZON,
};
use crate::move_ordering::MoveOrdering;
use crate::position_hash::PositionHasher;
use crate::sennichite::{SennichiteDetector, SennichiteStatus};
use crate::utils::{format_move_usi, get_piece_value};
use arrayvec::ArrayVec;
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
pub const DEFAULT_ROOT_MATE_NODE_BUDGET: u64 = 8_192;
pub const DEFAULT_REPLY_MATE_NODE_BUDGET: u64 = 128;
pub const DEFAULT_MATE_REPLY_CANDIDATES: usize = 1;
const ROOT_MATE_NODE_BUDGET_FRACTION: u64 = 2;
const ROOT_MATE_TIME_BUDGET_FRACTION: u64 = 10;
const ROOT_MATE_SOFT_TIME_MAX_MS: u64 = 10;
const ROOT_MATE_SHORT_TIME_MS: u64 = 50;
const ROOT_MATE_MEDIUM_TIME_MS: u64 = 200;
const ROOT_MATE_SHORT_NODE_CAP: u64 = 512;
const ROOT_MATE_MEDIUM_NODE_CAP: u64 = 2_048;
/// Production keeps non-check quiescence capture-only because complete checked-node
/// evasions substantially enlarge the tree. Checked nodes ignore this boundary and
/// always search every legal evasion. This Phase 1 change is correctness-oriented and
/// has a standalone dev-mate regression, so it is not sufficient for release by itself.
pub const DEFAULT_MAX_QUIESCENCE_CHECK_PLY: u16 = 0;
const TIME_CHECK_NODE_INTERVAL: u64 = 1024;
const USI_SCORE_CP_LIMIT: i32 = 2_000;
const USI_SCORE_CP_SOFT_START: i32 = 1_000;
const RESOURCE_CYCLE_BASE_PENALTY: f32 = 1_000.0;
type LegalMoves = ArrayVec<Move, 593>;

fn move_total_order_key(mv: Move) -> (u8, u8, u8, u8) {
    match mv {
        Move::Normal { from, to, promote } => (0, from.index(), to.index(), u8::from(promote)),
        Move::Drop { piece, to } => {
            let kind = match piece.piece_kind() {
                shogi_core::PieceKind::Pawn => 0,
                shogi_core::PieceKind::Lance => 1,
                shogi_core::PieceKind::Knight => 2,
                shogi_core::PieceKind::Silver => 3,
                shogi_core::PieceKind::Gold => 4,
                shogi_core::PieceKind::Bishop => 5,
                shogi_core::PieceKind::Rook => 6,
                _ => 7,
            };
            (1, kind, to.index(), 0)
        }
    }
}

#[derive(Clone, Copy)]
enum EvasionMoveSource {
    Dedicated,
    #[cfg(test)]
    Reference,
}

#[derive(Clone, Copy)]
enum EvasionKind {
    Capture = 0,
    KingMove = 1,
    MoveInterposition = 2,
    DropInterposition = 3,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MateProbeStopScope {
    Local,
    Global,
}

const EVASION_KIND_COUNT: usize = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchLimits {
    pub max_depth: u8,
    pub time_limit_ms: Option<u64>,
    pub node_limit: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SearchStopReason {
    #[default]
    Completed,
    NodeLimit,
    TimeLimit,
    ExternalStop,
    NoLegalMove,
}

#[derive(Clone, Debug, Default)]
pub struct SearchReport {
    pub best_move: Option<Move>,
    pub score: Option<f32>,
    pub pv: Vec<Move>,
    pub completed_depth: u8,
    pub nodes: u64,
    pub qnodes: u64,
    pub terminal_mates: u64,
    pub in_check_qnodes: u64,
    pub negative_see_checks_considered: u64,
    pub negative_see_check_searches: u64,
    pub repetition_hits: u64,
    pub resource_cycle_hits: u64,
    pub max_quiescence_ply: u16,
    pub quiescence_evasion_cutoffs: [u64; EVASION_KIND_COUNT],
    pub quiescence_tt_evasion_cutoffs: u64,
    pub mate_nodes: u64,
    pub mate_probes: u64,
    pub mate_proven: u64,
    pub mate_unknown: u64,
    pub mate_rejected: u64,
    pub stop_reason: SearchStopReason,
}

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
    node_limit: Option<u64>,
    stop_reason: SearchStopReason,
    next_time_check_nodes: u64,
    nodes_searched: u64,
    quiescence_nodes_searched: u64,
    max_quiescence_ply: u16,
    max_quiescence_check_ply: u16,
    quiescence_moves_considered: u64,
    quiescence_moves_generated: u64,
    quiescence_moves_discarded: u64,
    quiescence_moves_searched: u64,
    quiescence_see_skips: u64,
    quiescence_terminal_mates: u64,
    terminal_mates: u64,
    in_check_quiescence_nodes: u64,
    negative_see_checks_considered: u64,
    negative_see_check_searches: u64,
    repetition_hits: u64,
    resource_cycle_hits: u64,
    check_evasion_extensions: u64,
    aspiration_fail_lows: u64,
    aspiration_fail_highs: u64,
    aspiration_researches: u64,
    quiescence_evasion_cutoffs: [u64; EVASION_KIND_COUNT],
    quiescence_tt_evasion_cutoffs: u64,
    root_mate_node_budget: u64,
    reply_mate_node_budget: u64,
    mate_reply_candidates: usize,
    mate_nodes: u64,
    mate_probes: u64,
    mate_proven: u64,
    mate_unknown: u64,
    mate_rejected: u64,
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
            node_limit: None,
            stop_reason: SearchStopReason::Completed,
            next_time_check_nodes: 0,
            nodes_searched: 0,
            quiescence_nodes_searched: 0,
            max_quiescence_ply: 0,
            max_quiescence_check_ply: DEFAULT_MAX_QUIESCENCE_CHECK_PLY,
            quiescence_moves_considered: 0,
            quiescence_moves_generated: 0,
            quiescence_moves_discarded: 0,
            quiescence_moves_searched: 0,
            quiescence_see_skips: 0,
            quiescence_terminal_mates: 0,
            terminal_mates: 0,
            in_check_quiescence_nodes: 0,
            negative_see_checks_considered: 0,
            negative_see_check_searches: 0,
            repetition_hits: 0,
            resource_cycle_hits: 0,
            check_evasion_extensions: 0,
            aspiration_fail_lows: 0,
            aspiration_fail_highs: 0,
            aspiration_researches: 0,
            quiescence_evasion_cutoffs: [0; EVASION_KIND_COUNT],
            quiescence_tt_evasion_cutoffs: 0,
            root_mate_node_budget: DEFAULT_ROOT_MATE_NODE_BUDGET,
            reply_mate_node_budget: DEFAULT_REPLY_MATE_NODE_BUDGET,
            mate_reply_candidates: DEFAULT_MATE_REPLY_CANDIDATES,
            mate_nodes: 0,
            mate_probes: 0,
            mate_proven: 0,
            mate_unknown: 0,
            mate_rejected: 0,
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

    /// Replaces only the game-path state. Search heuristics and the evaluator are kept.
    pub fn set_game_history(&mut self, positions: &[Position]) {
        self.sennichite_detector.clear();
        for position in positions {
            self.sennichite_detector.record_position(position);
        }
    }

    pub fn nodes_searched(&self) -> u64 {
        self.nodes_searched
    }

    pub fn quiescence_nodes_searched(&self) -> u64 {
        self.quiescence_nodes_searched
    }

    pub fn max_quiescence_ply(&self) -> u16 {
        self.max_quiescence_ply
    }

    pub fn set_max_quiescence_check_ply(&mut self, max_ply: u16) {
        self.max_quiescence_check_ply = max_ply;
    }

    pub fn max_quiescence_check_ply(&self) -> u16 {
        self.max_quiescence_check_ply
    }

    pub fn set_mate_search_budgets(&mut self, root_nodes: u64, reply_nodes: u64) {
        self.root_mate_node_budget = root_nodes;
        self.reply_mate_node_budget = reply_nodes;
    }

    pub fn set_mate_reply_candidates(&mut self, candidates: usize) {
        self.mate_reply_candidates = candidates.min(DEFAULT_MATE_REPLY_CANDIDATES);
    }

    pub fn quiescence_moves_considered(&self) -> u64 {
        self.quiescence_moves_considered
    }

    pub fn quiescence_moves_generated(&self) -> u64 {
        self.quiescence_moves_generated
    }

    pub fn quiescence_moves_discarded(&self) -> u64 {
        self.quiescence_moves_discarded
    }

    pub fn quiescence_moves_searched(&self) -> u64 {
        self.quiescence_moves_searched
    }

    pub fn quiescence_see_skips(&self) -> u64 {
        self.quiescence_see_skips
    }

    pub fn quiescence_terminal_mates(&self) -> u64 {
        self.quiescence_terminal_mates
    }

    pub fn check_evasion_extensions(&self) -> u64 {
        self.check_evasion_extensions
    }

    pub fn aspiration_fail_lows(&self) -> u64 {
        self.aspiration_fail_lows
    }

    pub fn aspiration_fail_highs(&self) -> u64 {
        self.aspiration_fail_highs
    }

    pub fn aspiration_researches(&self) -> u64 {
        self.aspiration_researches
    }

    pub fn quiescence_evasion_cutoffs(&self) -> [u64; EVASION_KIND_COUNT] {
        self.quiescence_evasion_cutoffs
    }

    pub fn quiescence_tt_evasion_cutoffs(&self) -> u64 {
        self.quiescence_tt_evasion_cutoffs
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

    fn should_stop(&mut self) -> bool {
        if self
            .node_limit
            .is_some_and(|limit| self.nodes_searched >= limit)
        {
            self.stop_reason = SearchStopReason::NodeLimit;
            return true;
        }
        if self.nodes_searched < self.next_time_check_nodes {
            return false;
        }
        self.next_time_check_nodes = self.nodes_searched + TIME_CHECK_NODE_INTERVAL;

        if self
            .stop_signal
            .as_ref()
            .is_some_and(|stop_signal| stop_signal.load(Ordering::Relaxed))
        {
            self.stop_reason = SearchStopReason::ExternalStop;
            return true;
        }

        if let (Some(start), Some(limit)) = (self.start_time, self.time_limit) {
            if start.elapsed() >= limit {
                self.stop_reason = SearchStopReason::TimeLimit;
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    fn root_mate_soft_limits(
        &self,
        hard_node_limit: Option<u64>,
        hard_time_limit: Option<Duration>,
    ) -> (u64, Option<Instant>) {
        let mut node_limit = self.root_mate_node_budget;
        if let Some(hard_limit) = hard_node_limit {
            node_limit = node_limit.min(hard_limit / ROOT_MATE_NODE_BUDGET_FRACTION);
        }

        let soft_deadline = hard_time_limit.and_then(|hard_limit| {
            let hard_ms = hard_limit.as_millis().min(u128::from(u64::MAX)) as u64;
            let soft_ms =
                (hard_ms / ROOT_MATE_TIME_BUDGET_FRACTION).min(ROOT_MATE_SOFT_TIME_MAX_MS);
            if soft_ms == 0 {
                node_limit = 0;
                return None;
            }
            node_limit = node_limit.min(if hard_ms <= ROOT_MATE_SHORT_TIME_MS {
                ROOT_MATE_SHORT_NODE_CAP
            } else if hard_ms <= ROOT_MATE_MEDIUM_TIME_MS {
                ROOT_MATE_MEDIUM_NODE_CAP
            } else {
                self.root_mate_node_budget
            });
            self.start_time?.checked_add(Duration::from_millis(soft_ms))
        });

        (node_limit, soft_deadline)
    }

    /// Returns a score from the current side-to-move's perspective when the
    /// just-recorded path is terminal for repetition or resource dominance.
    fn recorded_path_score(&mut self, position: &Position) -> Option<f32> {
        match self.is_sennichite_internal(position) {
            SennichiteStatus::Draw => {
                self.repetition_hits += 1;
                return Some(0.0);
            }
            SennichiteStatus::PerpetualCheckLoss => {
                self.repetition_hits += 1;
                return Some(f32::INFINITY);
            }
            SennichiteStatus::PerpetualCheckWin => {
                self.repetition_hits += 1;
                return Some(f32::NEG_INFINITY);
            }
            SennichiteStatus::None => {}
        }

        let cycle = self.sennichite_detector.resource_cycle(position)?;
        self.resource_cycle_hits += 1;
        let penalty = RESOURCE_CYCLE_BASE_PENALTY + cycle.material_swing as f32;
        let static_score = self.evaluator.evaluate(position);
        Some(if cycle.loser == position.side_to_move() {
            static_score - penalty
        } else {
            static_score + penalty
        })
    }

    fn run_mate_probe(
        &mut self,
        position: &mut Position,
        attacker: shogi_core::Color,
        requested_nodes: u64,
        hard_node_limit: Option<u64>,
        deadline: Option<Instant>,
        stop_scope: MateProbeStopScope,
    ) -> MateSearchResult {
        self.mate_probes += 1;
        let available_nodes = hard_node_limit
            .map(|limit| limit.saturating_sub(self.nodes_searched))
            .unwrap_or(requested_nodes);
        let node_limit = requested_nodes.min(available_nodes);
        if node_limit == 0 {
            self.mate_unknown += 1;
            if stop_scope == MateProbeStopScope::Global && hard_node_limit.is_some() {
                self.stop_reason = SearchStopReason::NodeLimit;
            }
            return MateSearchResult::Unknown;
        }
        if self
            .stop_signal
            .as_ref()
            .is_some_and(|signal| signal.load(Ordering::Relaxed))
        {
            self.mate_unknown += 1;
            self.stop_reason = SearchStopReason::ExternalStop;
            return MateSearchResult::Unknown;
        }
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            self.mate_unknown += 1;
            if stop_scope == MateProbeStopScope::Global {
                self.stop_reason = SearchStopReason::TimeLimit;
            }
            return MateSearchResult::Unknown;
        }

        let mut searcher = MateSearcher::new(MateSearchLimits {
            node_limit,
            deadline,
            stop_signal: self.stop_signal.clone(),
            prior_position_hashes: self.sennichite_detector.prior_position_hashes(position),
        });
        let result = searcher.search_shortest(position, attacker, MAX_MATE_HORIZON);
        let nodes = searcher.nodes();
        self.mate_nodes += nodes;
        self.nodes_searched += nodes;
        match result {
            MateSearchResult::ProvenMate { .. } => self.mate_proven += 1,
            MateSearchResult::Unknown => {
                self.mate_unknown += 1;
                match searcher.stop_reason() {
                    Some(MateSearchStopReason::TimeLimit) => {
                        if stop_scope == MateProbeStopScope::Global {
                            self.stop_reason = SearchStopReason::TimeLimit;
                        }
                    }
                    Some(MateSearchStopReason::ExternalStop) => {
                        self.stop_reason = SearchStopReason::ExternalStop;
                    }
                    Some(MateSearchStopReason::NodeLimit)
                        if stop_scope == MateProbeStopScope::Global
                            && hard_node_limit.is_some()
                            && self.nodes_searched >= hard_node_limit.unwrap_or(u64::MAX) =>
                    {
                        self.stop_reason = SearchStopReason::NodeLimit;
                    }
                    _ => {}
                }
            }
            MateSearchResult::ProvenNoMateWithinHorizon => {}
        }
        result
    }

    fn reject_mated_root_candidates(
        &mut self,
        position: &mut Position,
        root_lines: &[(Move, f32, Vec<Move>)],
        hard_node_limit: Option<u64>,
        hard_deadline: Option<Instant>,
    ) -> Vec<Move> {
        let mut rejected = Vec::new();
        for &(candidate, _, _) in root_lines.iter().take(self.mate_reply_candidates) {
            if self.stop_reason != SearchStopReason::Completed {
                break;
            }
            if self.reply_mate_node_budget == 0 {
                break;
            }
            position.do_move(candidate);
            let opponent = position.side_to_move();
            let result = self.run_mate_probe(
                position,
                opponent,
                self.reply_mate_node_budget,
                hard_node_limit,
                hard_deadline,
                MateProbeStopScope::Global,
            );
            position.undo_move(candidate);
            if matches!(result, MateSearchResult::ProvenMate { .. }) {
                rejected.push(candidate);
                self.mate_rejected += 1;
            }
        }
        rejected
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
        alpha: f32,
        beta: f32,
    ) -> Option<(f32, Vec<Move>)> {
        self.quiescence_search_internal(
            position,
            alpha,
            beta,
            None,
            EvasionMoveSource::Dedicated,
            0,
        )
    }

    #[cfg(test)]
    fn quiescence_search_reference(
        &mut self,
        position: &mut Position,
        alpha: f32,
        beta: f32,
    ) -> Option<(f32, Vec<Move>)> {
        self.quiescence_search_internal(
            position,
            alpha,
            beta,
            None,
            EvasionMoveSource::Reference,
            0,
        )
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

    fn filter_capture_moves_from_legal(
        position: &Position,
        mut moves: LegalMoves,
    ) -> (LegalMoves, usize) {
        let generated = moves.len();
        moves.retain(|mv| match *mv {
            Move::Normal { to, .. } => position.piece_at(to).is_some(),
            Move::Drop { .. } => false,
        });
        (moves, generated)
    }

    fn quiescence_search_internal(
        &mut self,
        position: &mut Position,
        mut alpha: f32,
        beta: f32,
        precomputed_moves: Option<LegalMoves>,
        evasion_source: EvasionMoveSource,
        qply: u16,
    ) -> Option<(f32, Vec<Move>)> {
        if self.should_stop() {
            return None;
        }
        self.nodes_searched += 1;
        self.quiescence_nodes_searched += 1;
        self.max_quiescence_ply = self.max_quiescence_ply.max(qply);

        if position.in_check() {
            self.in_check_quiescence_nodes += 1;
            let (moves, generated_moves) = match precomputed_moves {
                Some(moves) => {
                    let generated = moves.len();
                    (moves, generated)
                }
                None => match evasion_source {
                    EvasionMoveSource::Dedicated => {
                        position.legal_evasion_moves_with_generated_count()
                    }
                    #[cfg(test)]
                    EvasionMoveSource::Reference => {
                        let moves = position.legal_moves();
                        let generated = moves.len();
                        (moves, generated)
                    }
                },
            };
            return self.quiescence_search_in_check(
                position,
                alpha,
                beta,
                moves,
                generated_moves,
                evasion_source,
                qply,
            );
        }

        let stand_pat_score = self.evaluator.evaluate(position);
        if stand_pat_score >= beta {
            return Some((beta, Vec::new()));
        }
        alpha = alpha.max(stand_pat_score);

        let (moves, generated_moves) = if qply >= self.max_quiescence_check_ply {
            precomputed_moves
                .map(|moves| Self::filter_capture_moves_from_legal(position, moves))
                .unwrap_or_else(|| position.legal_capture_moves_with_generated_count())
        } else {
            precomputed_moves
                .map(|moves| Self::filter_quiescence_moves_from_legal(position, moves))
                .unwrap_or_else(|| position.legal_quiescence_moves_with_generated_count())
        };

        self.quiescence_moves_generated += generated_moves as u64;
        self.quiescence_moves_discarded += (generated_moves - moves.len()) as u64;
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
        scored_moves.sort_unstable_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| move_total_order_key(left.0).cmp(&move_total_order_key(right.0)))
        });

        let mut best_score = stand_pat_score;
        for (mv, _) in scored_moves {
            let negative_see = self.see(position, mv) < 0;
            let negative_see_check = negative_see && position.is_check_move(mv);
            if negative_see_check {
                self.negative_see_checks_considered += 1;
            }
            if negative_see {
                self.quiescence_see_skips += 1;
                continue;
            }

            if negative_see_check {
                self.negative_see_check_searches += 1;
            }

            self.quiescence_moves_searched += 1;
            position.do_move(mv);
            self.sennichite_detector.record_position(position);
            let score_result = match self.recorded_path_score(position) {
                Some(score) => Some((score, Vec::new())),
                None => self.quiescence_search_internal(
                    position,
                    -beta,
                    -alpha,
                    None,
                    evasion_source,
                    qply.saturating_add(1),
                ),
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

    fn ordered_evasions(
        &mut self,
        position: &Position,
        moves: &LegalMoves,
    ) -> Vec<(Move, i32, EvasionKind)> {
        let mut scored_moves = moves
            .iter()
            .map(|&mv| {
                let kind = if position.piece_at(mv.to()).is_some() {
                    EvasionKind::Capture
                } else {
                    match mv {
                        Move::Normal { from, .. }
                            if position.piece_at(from).is_some_and(|piece| {
                                piece.piece_kind() == shogi_core::PieceKind::King
                            }) =>
                        {
                            EvasionKind::KingMove
                        }
                        Move::Normal { .. } => EvasionKind::MoveInterposition,
                        Move::Drop { .. } => EvasionKind::DropInterposition,
                    }
                };
                (
                    mv,
                    self.move_ordering.score_move_without_counter(&mv, position),
                    kind,
                )
            })
            .collect::<Vec<_>>();
        scored_moves.sort_unstable_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| move_total_order_key(left.0).cmp(&move_total_order_key(right.0)))
        });
        let tt_move = self
            .transposition_table
            .get(&PositionHasher::calculate_hash(position))
            .and_then(|entry| entry.best_move);
        if let Some(tt_move) = tt_move {
            if let Some(index) = scored_moves.iter().position(|&(mv, _, _)| mv == tt_move) {
                let entry = scored_moves.remove(index);
                scored_moves.insert(0, entry);
            }
        }
        scored_moves
    }

    fn quiescence_search_in_check(
        &mut self,
        position: &mut Position,
        mut alpha: f32,
        beta: f32,
        moves: LegalMoves,
        generated_moves: usize,
        evasion_source: EvasionMoveSource,
        qply: u16,
    ) -> Option<(f32, Vec<Move>)> {
        self.quiescence_moves_generated += generated_moves as u64;
        self.quiescence_moves_discarded += (generated_moves - moves.len()) as u64;
        if moves.is_empty() {
            self.quiescence_terminal_mates += 1;
            self.terminal_mates += 1;
            return Some((-f32::INFINITY, Vec::new()));
        }
        self.quiescence_moves_considered += moves.len() as u64;

        let tt_move = self
            .transposition_table
            .get(&PositionHasher::calculate_hash(position))
            .and_then(|entry| entry.best_move);
        let scored_moves = self.ordered_evasions(position, &moves);

        let mut best_score = -f32::INFINITY;
        let mut best_move = None;
        let mut best_child_pv = Vec::new();
        for (mv, _, kind) in scored_moves {
            self.quiescence_moves_searched += 1;
            position.do_move(mv);
            self.sennichite_detector.record_position(position);
            let score_result = match self.recorded_path_score(position) {
                Some(score) => Some((score, Vec::new())),
                None => self.quiescence_search_internal(
                    position,
                    -beta,
                    -alpha,
                    None,
                    evasion_source,
                    qply.saturating_add(1),
                ),
            };
            self.sennichite_detector.unrecord_last_position();
            position.undo_move(mv);

            let Some((child_score, child_pv)) = score_result else {
                return None;
            };
            let score = -child_score;
            if score > best_score {
                best_score = score;
                best_move = Some(mv);
                best_child_pv = child_pv;
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                self.quiescence_evasion_cutoffs[kind as usize] += 1;
                if Some(mv) == tt_move {
                    self.quiescence_tt_evasion_cutoffs += 1;
                }
                break;
            }
        }

        let mut pv = Vec::new();
        if let Some(best_move) = best_move {
            pv.push(best_move);
            pv.extend(best_child_pv);
        }
        Some((best_score, pv))
    }

    pub fn alpha_beta_search(
        &mut self,
        position: &mut Position,
        depth: u8,
        alpha: f32,
        beta: f32,
    ) -> Option<(f32, Vec<Move>)> {
        self.alpha_beta_search_internal(position, depth, alpha, beta, 1, None)
    }

    fn alpha_beta_search_internal(
        &mut self,
        position: &mut Position,
        depth: u8,
        mut alpha: f32,
        mut beta: f32,
        check_evasion_extension_budget: u8,
        precomputed_moves: Option<LegalMoves>,
    ) -> Option<(f32, Vec<Move>)> {
        if self.should_stop() {
            return None;
        }
        if depth == 0 {
            return self.quiescence_search_internal(
                position,
                alpha,
                beta,
                precomputed_moves,
                EvasionMoveSource::Dedicated,
                0,
            );
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

        let moves = precomputed_moves.unwrap_or_else(|| position.legal_moves());
        if moves.is_empty() {
            self.terminal_mates += 1;
            return Some((-f32::INFINITY, Vec::new()));
        }

        let mut scored_moves = Vec::with_capacity(moves.len());
        for mv in moves.iter() {
            scored_moves.push((*mv, self.search_ordering_score(position, *mv)));
        }
        scored_moves.sort_unstable_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| move_total_order_key(left.0).cmp(&move_total_order_key(right.0)))
        });
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
        let mut best_move: Option<Move> = None;
        let mut best_pv = Vec::new();
        let mut node_type = NodeType::UpperBound;

        for (mv, _) in sorted_moves {
            position.do_move(mv);
            self.sennichite_detector.record_position(position);
            let search_result = match self.recorded_path_score(position) {
                Some(score) => Some((score, Vec::new())),
                None => {
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
                        child_precomputed_moves,
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
        self.sennichite_detector
            .check_sennichite_assuming_alternating_history(position)
    }

    pub fn find_best_move(
        &mut self,
        position: &mut Position,
        max_depth: u8,
        time_limit_ms: Option<u64>,
    ) -> Option<Move> {
        self.find_best_move_with_limits(
            position,
            SearchLimits {
                max_depth,
                time_limit_ms,
                node_limit: None,
            },
        )
        .best_move
    }

    pub fn find_best_move_with_limits(
        &mut self,
        position: &mut Position,
        limits: SearchLimits,
    ) -> SearchReport {
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
        self.time_limit = limits.time_limit_ms.map(Duration::from_millis);
        self.node_limit = limits.node_limit;
        self.stop_reason = SearchStopReason::Completed;
        self.next_time_check_nodes = 0;
        self.nodes_searched = 0;
        self.quiescence_nodes_searched = 0;
        self.max_quiescence_ply = 0;
        self.quiescence_moves_considered = 0;
        self.quiescence_moves_generated = 0;
        self.quiescence_moves_discarded = 0;
        self.quiescence_moves_searched = 0;
        self.quiescence_see_skips = 0;
        self.quiescence_terminal_mates = 0;
        self.terminal_mates = 0;
        self.in_check_quiescence_nodes = 0;
        self.negative_see_checks_considered = 0;
        self.negative_see_check_searches = 0;
        self.repetition_hits = 0;
        self.resource_cycle_hits = 0;
        self.check_evasion_extensions = 0;
        self.aspiration_fail_lows = 0;
        self.aspiration_fail_highs = 0;
        self.aspiration_researches = 0;
        self.quiescence_evasion_cutoffs = [0; EVASION_KIND_COUNT];
        self.quiescence_tt_evasion_cutoffs = 0;
        self.mate_nodes = 0;
        self.mate_probes = 0;
        self.mate_proven = 0;
        self.mate_unknown = 0;
        self.mate_rejected = 0;

        let moves = position.legal_moves();
        if moves.is_empty() {
            self.terminal_mates += 1;
            self.stop_reason = SearchStopReason::NoLegalMove;
            return self.build_search_report(None, None, Vec::new(), 0);
        }

        let hard_node_limit = limits.node_limit;
        let hard_time_limit = limits.time_limit_ms.map(Duration::from_millis);
        let hard_deadline = hard_time_limit.and_then(|limit| self.start_time?.checked_add(limit));
        let attacker = position.side_to_move();
        let (root_mate_nodes, root_mate_deadline) =
            self.root_mate_soft_limits(hard_node_limit, hard_time_limit);
        if root_mate_nodes > 0 {
            let root_probe = self.run_mate_probe(
                position,
                attacker,
                root_mate_nodes,
                hard_node_limit,
                root_mate_deadline,
                MateProbeStopScope::Local,
            );
            if let MateSearchResult::ProvenMate {
                first_move: Some(first_move),
                proof,
                ..
            } = root_probe
            {
                return self.build_search_report(Some(first_move), Some(f32::INFINITY), proof, 0);
            }
        }
        if limits.max_depth == 0 {
            let _ = self.should_stop();
        }

        self.node_limit = hard_node_limit;
        self.time_limit = hard_time_limit;

        let mut scored_moves = Vec::with_capacity(moves.len());
        for mv in moves.iter() {
            scored_moves.push((*mv, self.search_ordering_score(position, *mv)));
        }
        scored_moves.sort_unstable_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| move_total_order_key(left.0).cmp(&move_total_order_key(right.0)))
        });
        let mut sorted_moves = scored_moves;
        let root_hash = PositionHasher::calculate_hash(position);
        if let Some(tt_move) = self
            .transposition_table
            .get(&root_hash)
            .and_then(|entry| entry.best_move)
        {
            if let Some(pos) = sorted_moves.iter().position(|&(m, _)| m == tt_move) {
                let mv = sorted_moves.remove(pos);
                sorted_moves.insert(0, mv);
            }
        }
        let mut best_move: Option<Move> = sorted_moves.first().map(|&(mv, _)| mv);
        let mut previous_eval: Option<f32> = None;

        let mut completed_depth = 0;
        let mut completed_pv = Vec::new();
        let mut completed_root_lines: Vec<(Move, f32, Vec<Move>)> = Vec::new();

        for depth in 1..=limits.max_depth {
            if let Some(previous_best) = best_move {
                if let Some(pos) = sorted_moves.iter().position(|&(m, _)| m == previous_best) {
                    let mv = sorted_moves.remove(pos);
                    sorted_moves.insert(0, mv);
                }
            }
            let mut current_best_move_for_depth: Option<Move> = None;
            let mut best_eval_for_depth = -f32::INFINITY;
            let mut best_pv_for_depth: Vec<Move> = Vec::new();
            let mut root_lines_for_depth: Vec<(Move, f32, Vec<Move>)> = Vec::new();
            let (mut alpha, mut beta) = previous_eval
                .map(|eval| (eval - ASPIRATION_WINDOW, eval + ASPIRATION_WINDOW))
                .unwrap_or((-f32::INFINITY, f32::INFINITY));
            let aspiration_alpha = alpha;
            let aspiration_beta = beta;
            let mut search_interrupted = false;

            for &(mv, _) in &sorted_moves {
                if self.should_stop() {
                    search_interrupted = true;
                    break;
                }

                position.do_move(mv);
                self.sennichite_detector.record_position(position);
                let eval_result = match self.recorded_path_score(position) {
                    Some(score) => Some((score, Vec::new())),
                    None => {
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
                position.undo_move(mv);

                if let Some((eval, pv)) = eval_result {
                    let current_eval = -eval;
                    let mut candidate_pv = vec![mv];
                    candidate_pv.extend(pv.iter().copied());
                    root_lines_for_depth.push((mv, current_eval, candidate_pv));
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

            if !search_interrupted {
                if best_eval_for_depth <= aspiration_alpha || best_eval_for_depth >= aspiration_beta
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
                    root_lines_for_depth.clear();

                    for &(mv, _) in &sorted_moves {
                        if self.should_stop() {
                            search_interrupted = true;
                            break;
                        }

                        position.do_move(mv);
                        self.sennichite_detector.record_position(position);
                        let eval_result = match self.recorded_path_score(position) {
                            Some(score) => Some((score, Vec::new())),
                            None => self.alpha_beta_search(position, depth - 1, -beta, -alpha),
                        };
                        self.sennichite_detector.unrecord_last_position();
                        position.undo_move(mv);

                        if let Some((eval, pv)) = eval_result {
                            let current_eval = -eval;
                            let mut candidate_pv = vec![mv];
                            candidate_pv.extend(pv.iter().copied());
                            root_lines_for_depth.push((mv, current_eval, candidate_pv));
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
            }

            if !search_interrupted {
                if let Some(current_best_move) = current_best_move_for_depth {
                    best_move = Some(current_best_move);
                    previous_eval = Some(best_eval_for_depth);
                    completed_depth = depth;
                    completed_pv = best_pv_for_depth.clone();
                    completed_root_lines = root_lines_for_depth;
                }
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

                if depth == limits.max_depth {
                    break;
                }
            } else {
                break;
            }
        }

        completed_root_lines.sort_unstable_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| move_total_order_key(left.0).cmp(&move_total_order_key(right.0)))
        });
        let rejected = completed_root_lines
            .first()
            .map_or_else(Vec::new, |best_line| {
                self.reject_mated_root_candidates(
                    position,
                    std::slice::from_ref(best_line),
                    hard_node_limit,
                    hard_deadline,
                )
            });
        if let Some((candidate, score, pv)) = completed_root_lines
            .iter()
            .find(|(candidate, _, _)| !rejected.contains(candidate))
        {
            best_move = Some(*candidate);
            previous_eval = Some(*score);
            completed_pv = pv.clone();
        }
        self.build_search_report(best_move, previous_eval, completed_pv, completed_depth)
    }

    fn build_search_report(
        &self,
        best_move: Option<Move>,
        score: Option<f32>,
        pv: Vec<Move>,
        completed_depth: u8,
    ) -> SearchReport {
        SearchReport {
            best_move,
            score,
            pv,
            completed_depth,
            nodes: self.nodes_searched,
            qnodes: self.quiescence_nodes_searched,
            terminal_mates: self.terminal_mates,
            in_check_qnodes: self.in_check_quiescence_nodes,
            negative_see_checks_considered: self.negative_see_checks_considered,
            negative_see_check_searches: self.negative_see_check_searches,
            repetition_hits: self.repetition_hits,
            resource_cycle_hits: self.resource_cycle_hits,
            max_quiescence_ply: self.max_quiescence_ply,
            quiescence_evasion_cutoffs: self.quiescence_evasion_cutoffs,
            quiescence_tt_evasion_cutoffs: self.quiescence_tt_evasion_cutoffs,
            mate_nodes: self.mate_nodes,
            mate_probes: self.mate_probes,
            mate_proven: self.mate_proven,
            mate_unknown: self.mate_unknown,
            mate_rejected: self.mate_rejected,
            stop_reason: self.stop_reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{parse_usi_move_for_color, position_from_sfen_or_usi};
    use serde_json::Value;

    struct ZeroEvaluator;

    impl Evaluator for ZeroEvaluator {
        fn evaluate(&self, _position: &Position) -> f32 {
            0.0
        }
    }

    struct RejectCheckedEvaluator;

    impl Evaluator for RejectCheckedEvaluator {
        fn evaluate(&self, position: &Position) -> f32 {
            assert!(
                !position.in_check(),
                "stand-pat evaluation must not run while in check"
            );
            0.0
        }
    }

    fn checked_position(sfen: &str) -> Position {
        let position = position_from_sfen_or_usi(sfen).expect("valid checked position");
        assert!(position.in_check(), "fixture must be in check: {sfen}");
        position
    }

    fn sorted_usi(moves: impl IntoIterator<Item = Move>) -> Vec<String> {
        let mut moves = moves.into_iter().map(format_move_usi).collect::<Vec<_>>();
        moves.sort_unstable();
        moves
    }

    fn resource_cycle_before_final_move() -> (Position, Vec<Position>, Move) {
        let mut position = position_from_sfen_or_usi(
            "ln1g3nl/1k7/1spsprbpp/p2p2p2/1Pg1Pp1P1/P1PSG1P2/1KNP1P2P/2SGR4/L6NL w bp 74",
        )
        .expect("valid suite cycle");
        let mut history = vec![position.clone()];
        for text in ["P*8h", "8g8h", "B*8g"] {
            let mv =
                parse_usi_move_for_color(text, position.side_to_move()).expect("valid proof move");
            assert!(
                position.legal_moves().contains(&mv),
                "illegal move {text}; legal={:?}",
                sorted_usi(position.legal_moves())
            );
            position.do_move(mv);
            history.push(position.clone());
        }
        let final_move =
            parse_usi_move_for_color("8h8g", position.side_to_move()).expect("valid final move");
        assert!(position.legal_moves().contains(&final_move));
        (position, history, final_move)
    }

    fn qsearch_resource_cycle_before_final_move() -> (Position, Vec<Position>, Move) {
        let mut position =
            position_from_sfen_or_usi("k8/9/9/4G4/9/9/9/9/K8 w r 1").expect("valid qsearch cycle");
        let mut history = vec![position.clone()];
        for text in ["R*5e", "5d4d", "5e5d"] {
            let mv =
                parse_usi_move_for_color(text, position.side_to_move()).expect("valid proof move");
            assert!(
                position.legal_moves().contains(&mv),
                "illegal move {text}; legal={:?}",
                sorted_usi(position.legal_moves())
            );
            position.do_move(mv);
            history.push(position.clone());
        }
        let final_move =
            parse_usi_move_for_color("4d5d", position.side_to_move()).expect("valid final move");
        assert!(position.legal_moves().contains(&final_move));
        (position, history, final_move)
    }

    const EVASION_CLASS_SFENS: [&str; 5] = [
        // Unique quiet king escape.
        "l5r1l/1S1g2g2/2n1pkn2/p1pp2BRp/1p2S1b2/P1PP4P/1PS1P1+p2/2G1G4/LNK5L w sn5p 72",
        // Move interposition.
        "k3r4/9/9/9/9/9/9/9/4KG3 b - 1",
        // Drop interpositions.
        "k3r4/9/9/9/9/9/9/9/4K4 b G 1",
        // Negative-SEE capture evasion.
        "4k4/9/9/9/9/9/4Rg3/3PpP3/3PKP3 b - 1",
        // Checkmate.
        "9/7pp/8k/7PP/7G1/9/9/9/K8 w 2r2b3g4s4n4l14p 2",
    ];

    fn run_qsearch_with_fresh_state(
        sfen: &str,
        source: EvasionMoveSource,
        alpha: f32,
        beta: f32,
    ) -> (f32, Vec<Move>) {
        let mut position = checked_position(sfen);
        let original_sfen = position.to_sfen_owned();
        let mut ai = ShogiAI::<_, 256>::new(RejectCheckedEvaluator);
        ai.set_max_quiescence_check_ply(0);
        ai.sennichite_detector.record_position(&position);
        let result = match source {
            EvasionMoveSource::Dedicated => ai.quiescence_search(&mut position, alpha, beta),
            EvasionMoveSource::Reference => {
                ai.quiescence_search_reference(&mut position, alpha, beta)
            }
        }
        .expect("unlimited qsearch must complete");
        assert_eq!(original_sfen, position.to_sfen_owned(), "must undo moves");
        result
    }

    fn assert_legal_pv(sfen: &str, pv: &[Move]) {
        let mut position = checked_position(sfen);
        for &mv in pv {
            assert!(
                position.legal_moves().contains(&mv),
                "illegal PV move in {sfen}"
            );
            position.do_move(mv);
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    enum TestWindowBound {
        Upper,
        Exact,
        Lower,
    }

    fn test_window_bound(score: f32, alpha: f32, beta: f32) -> TestWindowBound {
        if score >= beta {
            TestWindowBound::Lower
        } else if score <= alpha {
            TestWindowBound::Upper
        } else {
            TestWindowBound::Exact
        }
    }

    #[test]
    fn dedicated_evasions_match_shared_legality_oracle_and_order_on_dev_suite() {
        // This is a behavioral oracle, not an implementation-independent one:
        // both paths ultimately share Position's attack tables and legality rules.
        for (index, line) in
            include_str!("../data/search_quality/generated/dev_quiet_evasion.jsonl")
                .lines()
                .enumerate()
        {
            let record: Value = serde_json::from_str(line).expect("valid fixed-suite JSONL");
            let sfen = record["sfen"].as_str().expect("record has sfen");
            let position = checked_position(sfen);
            let reference_moves = position.legal_moves();
            let dedicated_moves = position.legal_evasion_moves_with_generated_count().0;
            let reference = sorted_usi(reference_moves.clone());
            let dedicated = sorted_usi(dedicated_moves.clone());
            let mut expected = record["legal_evasions"]
                .as_array()
                .expect("record has legal evasions")
                .iter()
                .map(|value| value.as_str().expect("evasion is text").to_owned())
                .collect::<Vec<_>>();
            expected.sort_unstable();
            assert_eq!(expected, reference, "record line {}: {sfen}", index + 1);
            assert_eq!(reference, dedicated, "line {}: {sfen}", index + 1);

            let mut reference_ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
            let reference_order = reference_ai
                .ordered_evasions(&position, &reference_moves)
                .into_iter()
                .map(|(mv, _, _)| mv)
                .collect::<Vec<_>>();
            let mut dedicated_ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
            let dedicated_order = dedicated_ai
                .ordered_evasions(&position, &dedicated_moves)
                .into_iter()
                .map(|(mv, _, _)| mv)
                .collect::<Vec<_>>();
            assert_eq!(reference_order, dedicated_order, "order line {}", index + 1);

            let mover = position.side_to_move();
            for mv in dedicated_moves {
                let mut child = position.clone();
                child.do_move(mv);
                assert!(
                    !child.is_king_in_check(mover),
                    "line {} evasion leaves mover king checked: {}",
                    index + 1,
                    format_move_usi(mv)
                );
            }
        }
    }

    #[test]
    fn checked_qsearch_full_window_exact_score_and_best_match_reference() {
        for sfen in EVASION_CLASS_SFENS {
            let dedicated = run_qsearch_with_fresh_state(
                sfen,
                EvasionMoveSource::Dedicated,
                f32::NEG_INFINITY,
                f32::INFINITY,
            );
            let reference = run_qsearch_with_fresh_state(
                sfen,
                EvasionMoveSource::Reference,
                f32::NEG_INFINITY,
                f32::INFINITY,
            );
            assert_eq!(reference.0, dedicated.0, "exact score: {sfen}");
            assert_eq!(reference.1.first(), dedicated.1.first(), "best: {sfen}");
            assert_legal_pv(sfen, &dedicated.1);
            assert_legal_pv(sfen, &reference.1);
        }
    }

    #[test]
    fn checked_qsearch_narrow_window_bound_and_legal_pv_match_reference() {
        const ALPHA: f32 = -1.0;
        const BETA: f32 = 0.0;
        for sfen in EVASION_CLASS_SFENS {
            let dedicated =
                run_qsearch_with_fresh_state(sfen, EvasionMoveSource::Dedicated, ALPHA, BETA);
            let reference =
                run_qsearch_with_fresh_state(sfen, EvasionMoveSource::Reference, ALPHA, BETA);
            assert_eq!(
                test_window_bound(reference.0, ALPHA, BETA),
                test_window_bound(dedicated.0, ALPHA, BETA),
                "bound: {sfen}"
            );
            assert_eq!(reference.1.first(), dedicated.1.first(), "PV head: {sfen}");
            assert_legal_pv(sfen, &dedicated.1);
            assert_legal_pv(sfen, &reference.1);
        }
    }

    #[test]
    fn checked_qsearch_searches_unique_quiet_king_escape_without_stand_pat() {
        let sfen = "l5r1l/1S1g2g2/2n1pkn2/p1pp2BRp/1p2S1b2/P1PP4P/1PS1P1+p2/2G1G4/LNK5L w sn5p 72";
        let mut position = checked_position(sfen);
        assert_eq!(vec!["4c4b"], sorted_usi(position.legal_moves()));
        let mut ai = ShogiAI::<_, 256>::new(RejectCheckedEvaluator);
        ai.sennichite_detector.record_position(&position);

        let (_, pv) = ai
            .quiescence_search(&mut position, f32::NEG_INFINITY, f32::INFINITY)
            .expect("qsearch must complete");

        assert_eq!(
            Some("4c4b"),
            pv.first().copied().map(format_move_usi).as_deref()
        );
        assert!(ai.quiescence_moves_searched() >= 1);
    }

    #[test]
    fn checked_qsearch_searches_all_evasions_without_see_pruning() {
        let sfen = "4k4/9/9/9/9/9/4Rg3/3PpP3/3PKP3 b - 1";
        let mut position = checked_position(sfen);
        let root_evasions = position.legal_moves().len() as u64;
        assert_eq!(vec!["5g5h"], sorted_usi(position.legal_moves()));
        let mut ai = ShogiAI::<_, 256>::new(RejectCheckedEvaluator);
        let evasion = position.legal_moves()[0];
        assert!(
            ai.see(&position, evasion) < 0,
            "fixture must have negative SEE"
        );
        ai.sennichite_detector.record_position(&position);

        let (_, pv) = ai
            .quiescence_search(&mut position, f32::NEG_INFINITY, f32::INFINITY)
            .expect("qsearch must complete");

        assert!(ai.quiescence_moves_searched() >= root_evasions);
        assert_eq!(Some(&evasion), pv.first());
    }

    #[test]
    fn checked_qsearch_completes_deep_tactical_tree_without_stack_overflow() {
        let mut position = checked_position(
            "+Bn1g4l/4k1s2/1p1p+Np2n/6p1p/p5Pp1/5G3/PP1PPP1RP/2+r+s1G3/L4K2L w BSNL2Pgs2p 86",
        );
        let mut ai = ShogiAI::<_, 256>::new(RejectCheckedEvaluator);
        ai.set_max_quiescence_check_ply(8);
        ai.sennichite_detector.record_position(&position);

        ai.quiescence_search(&mut position, f32::NEG_INFINITY, f32::INFINITY)
            .expect("deep tactical qsearch must complete");
        assert!(ai.max_quiescence_ply() >= 8);
    }

    #[test]
    fn non_capture_checks_stop_at_configured_qply_boundary() {
        const BOUNDARY: u16 = 4;
        let position = position_from_sfen_or_usi("4k4/9/9/9/9/9/9/9/K4R3 b - 1")
            .expect("valid boundary fixture");
        assert!(!position.in_check());
        let tactical_moves = position.legal_quiescence_moves_with_generated_count().0;
        assert!(!tactical_moves.is_empty());
        assert!(tactical_moves.iter().all(|mv| match *mv {
            Move::Normal { to, .. } => position.piece_at(to).is_none(),
            Move::Drop { .. } => true,
        }));

        let mut before_position = position.clone();
        let mut before = ShogiAI::<_, 256>::new(RejectCheckedEvaluator);
        before.set_max_quiescence_check_ply(BOUNDARY);
        before.sennichite_detector.record_position(&before_position);
        before
            .quiescence_search_internal(
                &mut before_position,
                f32::NEG_INFINITY,
                f32::INFINITY,
                None,
                EvasionMoveSource::Dedicated,
                BOUNDARY - 1,
            )
            .expect("qsearch before boundary must complete");
        assert!(before.quiescence_moves_searched() > 0);

        let mut at_position = position.clone();
        let mut at = ShogiAI::<_, 256>::new(RejectCheckedEvaluator);
        at.set_max_quiescence_check_ply(BOUNDARY);
        at.sennichite_detector.record_position(&at_position);
        at.quiescence_search_internal(
            &mut at_position,
            f32::NEG_INFINITY,
            f32::INFINITY,
            None,
            EvasionMoveSource::Dedicated,
            BOUNDARY,
        )
        .expect("qsearch at boundary must complete");
        assert_eq!(0, at.quiescence_moves_considered());
        assert_eq!(0, at.quiescence_moves_searched());
    }

    #[test]
    fn all_dev_mate_sacrifice_proofs_remain_legal_and_terminal() {
        let mut horizons = [0usize; 4];
        for (index, line) in
            include_str!("../data/search_quality/generated/dev_mate_sacrifice.jsonl")
                .lines()
                .enumerate()
        {
            let record: Value = serde_json::from_str(line).expect("valid mate JSONL");
            let horizon = record["mate_horizon"].as_u64().expect("mate horizon") as u16;
            horizons[((horizon - 1) / 2) as usize] += 1;

            let sfen = record["sfen"].as_str().expect("record has sfen");
            let mut position = position_from_sfen_or_usi(sfen).expect("valid mate position");
            for move_text in record["proof_line"].as_array().expect("proof line") {
                let move_text = move_text.as_str().expect("proof move is text");
                let mv = parse_usi_move_for_color(move_text, position.side_to_move())
                    .expect("parse proof move");
                assert!(
                    position.legal_moves().contains(&mv),
                    "line {} illegal proof move {move_text}",
                    index + 1
                );
                position.do_move(mv);
            }
            assert!(position.in_check(), "line {} must end in check", index + 1);
            assert!(
                position.legal_moves().is_empty(),
                "line {} must end in mate",
                index + 1
            );
        }
        assert_eq!([35, 58, 62, 45], horizons);
    }

    #[test]
    fn checked_qsearch_reports_mate_without_stand_pat() {
        let mut position = checked_position("9/7pp/8k/7PP/7G1/9/9/9/K8 w 2r2b3g4s4n4l14p 2");
        let mut ai = ShogiAI::<_, 256>::new(RejectCheckedEvaluator);
        ai.sennichite_detector.record_position(&position);

        let result = ai
            .quiescence_search(&mut position, f32::NEG_INFINITY, f32::INFINITY)
            .expect("mate qsearch must complete");

        assert_eq!(f32::NEG_INFINITY, result.0);
        assert!(result.1.is_empty());
        assert_eq!(1, ai.quiescence_terminal_mates());
    }

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

    #[test]
    fn find_best_move_keeps_legal_fallback_when_all_lines_lose() {
        let mut position = position_from_sfen_or_usi(
            "l6nl/7+R1/4k4/p1P+bp1n1p/3K1p3/P3P1S1P/ls2BP3/s3G4/9 b R2GS2NL9Pg 131",
        )
        .expect("valid sfen");
        assert!(!position.legal_moves().is_empty());

        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        ai.set_emit_info(false);
        ai.sennichite_detector.record_position(&position);

        assert!(ai.find_best_move(&mut position, 5, None).is_some());
    }

    #[test]
    fn node_limit_is_strict_and_reported() {
        let mut position = Position::default();
        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        ai.set_emit_info(false);
        ai.sennichite_detector.record_position(&position);
        let report = ai.find_best_move_with_limits(
            &mut position,
            SearchLimits {
                max_depth: 8,
                time_limit_ms: None,
                node_limit: Some(1),
            },
        );
        assert_eq!(1, report.nodes);
        assert_eq!(SearchStopReason::NodeLimit, report.stop_reason);
        assert!(report.best_move.is_some(), "a legal fallback must remain");
        assert_eq!(0, report.mate_nodes);
        assert_eq!(0, report.mate_unknown);
    }

    #[test]
    fn node_limit_has_priority_when_limits_fire_together() {
        let mut position = Position::default();
        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        ai.set_emit_info(false);
        let report = ai.find_best_move_with_limits(
            &mut position,
            SearchLimits {
                max_depth: 8,
                time_limit_ms: Some(0),
                node_limit: Some(0),
            },
        );
        assert_eq!(0, report.nodes);
        assert_eq!(SearchStopReason::NodeLimit, report.stop_reason);
    }

    #[test]
    fn mate_probe_global_deadline_and_external_stop_reach_report() {
        let mut deadline_position = Position::default();
        let mut deadline_ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        deadline_ai.set_emit_info(false);
        let deadline_report = deadline_ai.find_best_move_with_limits(
            &mut deadline_position,
            SearchLimits {
                max_depth: 0,
                time_limit_ms: Some(0),
                node_limit: None,
            },
        );
        assert_eq!(SearchStopReason::TimeLimit, deadline_report.stop_reason);
        assert_eq!(0, deadline_report.mate_probes);
        assert_eq!(0, deadline_report.mate_unknown);

        let signal = Arc::new(AtomicBool::new(true));
        let mut external_position = Position::default();
        let mut external_ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        external_ai.set_emit_info(false);
        external_ai.set_stop_signal(Some(signal));
        let external_report = external_ai.find_best_move_with_limits(
            &mut external_position,
            SearchLimits {
                max_depth: 0,
                time_limit_ms: None,
                node_limit: None,
            },
        );
        assert_eq!(SearchStopReason::ExternalStop, external_report.stop_reason);
        assert_eq!(1, external_report.mate_probes);
        assert_eq!(1, external_report.mate_unknown);
    }

    #[test]
    fn zero_depth_local_mate_budget_unknown_does_not_stop_search() {
        let mut position = Position::default();
        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        ai.set_emit_info(false);
        ai.set_mate_search_budgets(1, 0);
        let report = ai.find_best_move_with_limits(
            &mut position,
            SearchLimits {
                max_depth: 0,
                time_limit_ms: None,
                node_limit: None,
            },
        );
        assert_eq!(SearchStopReason::Completed, report.stop_reason);
        assert_eq!(1, report.mate_probes);
        assert_eq!(1, report.mate_unknown);
        assert_eq!(1, report.mate_nodes);
        assert_eq!(report.mate_nodes, report.nodes);
    }

    #[test]
    fn local_mate_budget_does_not_claim_global_node_stop_with_room_left() {
        let record: Value = serde_json::from_str(
            include_str!("../data/search_quality/generated/dev_mate_sacrifice.jsonl")
                .lines()
                .find(|line| line.contains("\"mate_horizon\":7"))
                .expect("depth-seven dev fixture"),
        )
        .expect("valid mate record");
        let mut position = position_from_sfen_or_usi(record["sfen"].as_str().unwrap()).unwrap();
        let attacker = position.side_to_move();
        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        let result = ai.run_mate_probe(
            &mut position,
            attacker,
            8,
            Some(16),
            None,
            MateProbeStopScope::Local,
        );
        assert!(matches!(result, MateSearchResult::Unknown));
        assert_eq!(SearchStopReason::Completed, ai.stop_reason);
        assert_eq!(8, ai.nodes_searched);
    }

    #[test]
    fn local_mate_deadline_does_not_stop_normal_search() {
        let mut position = Position::default();
        let attacker = position.side_to_move();
        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        let result = ai.run_mate_probe(
            &mut position,
            attacker,
            128,
            None,
            Some(Instant::now()),
            MateProbeStopScope::Local,
        );
        assert_eq!(MateSearchResult::Unknown, result);
        assert_eq!(SearchStopReason::Completed, ai.stop_reason);
    }

    #[test]
    fn root_mate_budget_preserves_main_search_budget() {
        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        ai.start_time = Some(Instant::now());
        assert_eq!(8_192, ai.root_mate_soft_limits(Some(20_000), None).0);
        assert_eq!(
            ROOT_MATE_SHORT_NODE_CAP,
            ai.root_mate_soft_limits(None, Some(Duration::from_millis(30)))
                .0
        );
        assert_eq!(0, ai.root_mate_soft_limits(None, Some(Duration::ZERO)).0);
    }

    #[test]
    fn reply_probe_global_external_stop_is_reported() {
        let mut position = checked_position(
            "l2l4l/3ks1G2/2p1pgn1p/3p1ppp1/p1P5P/3P3P1/P1N+bPPPs1/1r3S3/+p3GKSNL b RBNgp 77",
        );
        let bad = parse_usi_move_for_color("5i5h", position.side_to_move()).unwrap();
        let signal = Arc::new(AtomicBool::new(true));
        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        ai.set_stop_signal(Some(signal));
        ai.set_mate_search_budgets(0, 128);
        ai.set_mate_reply_candidates(1);
        let rejected =
            ai.reject_mated_root_candidates(&mut position, &[(bad, 1.0, vec![bad])], None, None);
        assert!(rejected.is_empty());
        assert_eq!(SearchStopReason::ExternalStop, ai.stop_reason);
        assert_eq!(1, ai.mate_unknown);
    }

    #[test]
    fn fixed_node_search_report_is_reproducible() {
        fn search() -> SearchReport {
            let mut position = Position::default();
            let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
            ai.set_emit_info(false);
            ai.find_best_move_with_limits(
                &mut position,
                SearchLimits {
                    max_depth: 5,
                    time_limit_ms: None,
                    node_limit: Some(500),
                },
            )
        }
        let first = search();
        let second = search();
        assert_eq!(first.best_move, second.best_move);
        assert_eq!(first.score, second.score);
        assert_eq!(first.pv, second.pv);
        assert_eq!(first.completed_depth, second.completed_depth);
        assert_eq!(first.nodes, second.nodes);
        assert_eq!(first.qnodes, second.qnodes);
        assert_eq!(first.stop_reason, second.stop_reason);
    }

    #[test]
    fn reply_probe_budget_does_not_reduce_main_node_limit() {
        let mut position = Position::default();
        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        ai.set_emit_info(false);
        ai.set_mate_search_budgets(0, 128);
        let report = ai.find_best_move_with_limits(
            &mut position,
            SearchLimits {
                max_depth: 8,
                time_limit_ms: None,
                node_limit: Some(500),
            },
        );
        assert_eq!(500, report.nodes);
        assert_eq!(SearchStopReason::NodeLimit, report.stop_reason);
        assert_eq!(0, report.mate_probes);
    }

    #[test]
    fn reply_mate_probe_rejects_only_proven_mated_candidate_and_restores_position() {
        let mut position = checked_position(
            "l2l4l/3ks1G2/2p1pgn1p/3p1ppp1/p1P5P/3P3P1/P1N+bPPPs1/1r3S3/+p3GKSNL b RBNgp 77",
        );
        let original = position.to_sfen_owned();
        let bad = parse_usi_move_for_color("5i5h", position.side_to_move()).unwrap();
        let good = parse_usi_move_for_color("R*5h", position.side_to_move()).unwrap();
        assert!(position.legal_moves().contains(&bad));
        assert!(position.legal_moves().contains(&good));
        let root_lines = vec![(bad, 100.0, vec![bad]), (good, 0.0, vec![good])];

        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        ai.set_mate_search_budgets(0, 128);
        ai.set_mate_reply_candidates(2);
        let rejected = ai.reject_mated_root_candidates(&mut position, &root_lines, None, None);

        assert_eq!(vec![bad], rejected);
        assert_eq!(1, ai.mate_rejected);
        assert_eq!(1, ai.mate_proven);
        assert_eq!(original, position.to_sfen_owned());
    }

    #[test]
    fn reply_mate_probe_does_not_reject_unknown() {
        let mut position = checked_position(
            "l2l4l/3ks1G2/2p1pgn1p/3p1ppp1/p1P5P/3P3P1/P1N+bPPPs1/1r3S3/+p3GKSNL b RBNgp 77",
        );
        let original = position.to_sfen_owned();
        let bad = parse_usi_move_for_color("5i5h", position.side_to_move()).unwrap();
        let root_lines = vec![(bad, 100.0, vec![bad])];

        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        ai.set_mate_search_budgets(0, 1);
        let rejected = ai.reject_mated_root_candidates(&mut position, &root_lines, None, None);

        assert!(rejected.is_empty());
        assert_eq!(1, ai.mate_unknown);
        assert_eq!(0, ai.mate_rejected);
        assert_eq!(original, position.to_sfen_owned());
    }

    #[test]
    fn root_mate_probe_selects_proven_move_within_total_node_budget() {
        let mut position = position_from_sfen_or_usi(
            "7n1/4+R2s+L/p1p+Bppkp1/1p4p2/3g5/PP7/2P1PPPP1/1B2G2R1/LN1GK1SN1 b G2L2P2sn2p 55",
        )
        .unwrap();
        let original = position.to_sfen_owned();
        let expected = parse_usi_move_for_color("5b2b", position.side_to_move()).unwrap();
        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        ai.set_emit_info(false);
        ai.set_mate_search_budgets(8_192, 0);
        ai.sennichite_detector.record_position(&position);

        let report = ai.find_best_move_with_limits(
            &mut position,
            SearchLimits {
                max_depth: 7,
                time_limit_ms: None,
                node_limit: Some(20_000),
            },
        );

        assert_eq!(Some(expected), report.best_move);
        assert_eq!(Some(f32::INFINITY), report.score);
        assert_eq!(1, report.mate_proven);
        assert!(report.mate_nodes <= 8_192);
        assert!(report.nodes <= 20_000);
        assert_eq!(original, position.to_sfen_owned());
    }

    #[test]
    fn alpha_beta_scores_suite_resource_cycle_as_finite_material_loss() {
        let (mut position, history, final_move) = resource_cycle_before_final_move();
        let original = position.to_sfen_owned();
        let mut only_final = LegalMoves::new();
        only_final.push(final_move);
        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        ai.set_game_history(&history);

        let (score, pv) = ai
            .alpha_beta_search_internal(
                &mut position,
                1,
                f32::NEG_INFINITY,
                f32::INFINITY,
                1,
                Some(only_final),
            )
            .expect("search completes");

        assert!(score.is_finite());
        assert_eq!(2_800.0, score);
        assert_eq!(vec![final_move], pv);
        assert_eq!(1, ai.resource_cycle_hits);
        assert_eq!(original, position.to_sfen_owned());
    }

    #[test]
    fn qsearch_scores_resource_cycle_and_counts_the_hit() {
        let (mut position, history, final_move) = qsearch_resource_cycle_before_final_move();
        let original = position.to_sfen_owned();
        let mut only_final = LegalMoves::new();
        only_final.push(final_move);
        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        ai.set_game_history(&history);

        let (score, _) = ai
            .quiescence_search_internal(
                &mut position,
                f32::NEG_INFINITY,
                f32::INFINITY,
                Some(only_final),
                EvasionMoveSource::Dedicated,
                0,
            )
            .expect("qsearch completes");

        assert!(score.is_finite());
        assert_eq!(3_000.0, score);
        assert_eq!(1, ai.resource_cycle_hits);
        assert_eq!(original, position.to_sfen_owned());
    }

    #[test]
    fn report_counts_a_root_terminal_mate() {
        let mut position =
            position_from_sfen_or_usi("9/7pp/8k/7PP/7G1/9/9/9/K8 w 2r2b3g4s4n4l14p 2")
                .expect("valid checkmate");
        assert!(position.in_check());
        assert!(position.legal_moves().is_empty());
        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        ai.set_emit_info(false);
        let report = ai.find_best_move_with_limits(
            &mut position,
            SearchLimits {
                max_depth: 3,
                time_limit_ms: None,
                node_limit: None,
            },
        );
        assert_eq!(1, report.terminal_mates);
        assert_eq!(SearchStopReason::NoLegalMove, report.stop_reason);
    }
}
