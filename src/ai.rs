mod alpha_beta;
mod iterative;
mod outcome;
mod parallel;
mod qsearch;
mod score;
mod transposition;

use self::score::*;
use self::transposition::*;
use crate::evaluation::{EvaluationContext, Evaluator};
use crate::move_ordering::MoveOrdering;
use crate::position_hash::PositionHasher;
use crate::sennichite::{GameHistory, SennichiteStatus};
use crate::utils::get_piece_value;
use arrayvec::ArrayVec;
pub use outcome::{
    RootResult, SearchInfo, SearchLimits, SearchObserver, SearchOutcome, SearchStats,
    SharedSearchObserver,
};
pub use parallel::resolve_search_threads;
use shogi_core::Move;
use shogi_lib::Position;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const MAX_DEPTH: usize = 64;
const ASPIRATION_WINDOW: f32 = 300.0;
const CHECK_MOVE_BONUS: i32 = 2_000;
const SEE_ORDERING_SCALE: i32 = 20;
const CHECK_EVASION_EXTENSION_MAX_REPLIES: usize = 3;
const SEARCH_THREAD_STACK_BYTES: usize = 4 * 1024 * 1024;
// Checked qsearch nodes must examine every evasion, so cap checking sequences explicitly.
const MAX_QUIESCENCE_PLY: u8 = 8;
const TIME_CHECK_NODE_INTERVAL: u64 = 1024;
type LegalMoves = ArrayVec<Move, 593>;

/// 将棋のアルファベータ探索を管理する構造体
pub struct ShogiAI<E: Evaluator, const HISTORY_CAPACITY: usize> {
    move_ordering: MoveOrdering,
    pub evaluator: E,
    pub sennichite_detector: GameHistory,
    transposition_table: TranspositionTable,
    killer_moves: [[Option<Move>; 2]; MAX_DEPTH],
    start_time: Option<Instant>,
    time_limit: Option<Duration>,
    next_time_check_nodes: u64,
    nodes_searched: u64,
    quiescence_nodes_searched: u64,
    quiescence_moves_considered: u64,
    quiescence_moves_generated: u64,
    quiescence_moves_discarded: u64,
    quiescence_moves_searched: u64,
    quiescence_see_skips: u64,
    quiescence_terminal_mates: u64,
    check_evasion_extensions: u64,
    aspiration_fail_lows: u64,
    aspiration_fail_highs: u64,
    aspiration_researches: u64,
    observer: Option<SharedSearchObserver>,
    search_generation: u32,
    stop_signal: Option<Arc<AtomicBool>>,
    eval_context: Option<EvaluationContext>,
    last_completed_depth: u8,
    last_root_score: Option<f32>,
    last_pv: Vec<Move>,
    last_search_failed: bool,
}

impl<E: Evaluator, const HISTORY_CAPACITY: usize> ShogiAI<E, HISTORY_CAPACITY> {
    pub fn new(evaluator: E) -> Self {
        ShogiAI {
            move_ordering: MoveOrdering::new(),
            evaluator,
            sennichite_detector: GameHistory::new(),
            transposition_table: TranspositionTable::Local(HashMap::new()),
            killer_moves: [[None; 2]; MAX_DEPTH],
            start_time: None,
            time_limit: None,
            next_time_check_nodes: 0,
            nodes_searched: 0,
            quiescence_nodes_searched: 0,
            quiescence_moves_considered: 0,
            quiescence_moves_generated: 0,
            quiescence_moves_discarded: 0,
            quiescence_moves_searched: 0,
            quiescence_see_skips: 0,
            quiescence_terminal_mates: 0,
            check_evasion_extensions: 0,
            aspiration_fail_lows: 0,
            aspiration_fail_highs: 0,
            aspiration_researches: 0,
            observer: None,
            search_generation: 0,
            stop_signal: None,
            eval_context: None,
            last_completed_depth: 0,
            last_root_score: None,
            last_pv: Vec::new(),
            last_search_failed: false,
        }
    }

    fn new_with_shared_tt(
        evaluator: E,
        transposition_table: Arc<SharedTranspositionTable>,
        search_generation: u32,
    ) -> Self {
        let mut ai = Self::new(evaluator);
        ai.transposition_table = TranspositionTable::Shared(transposition_table);
        ai.search_generation = search_generation;
        ai
    }

    fn begin_eval_context(&mut self, position: &Position) {
        self.eval_context = self.evaluator.begin_context(position);
    }

    fn evaluate_position(&self, position: &Position) -> f32 {
        self.eval_context
            .as_ref()
            .and_then(|ctx| self.evaluator.evaluate_context(position, ctx))
            .unwrap_or_else(|| self.evaluator.evaluate(position))
    }

    fn make_move(&mut self, position: &mut Position, mv: Move) {
        if let Some(ctx) = self.eval_context.as_mut() {
            self.evaluator.prepare_context_move(ctx, position, mv);
        }
        position.do_move(mv);
        if let Some(ctx) = self.eval_context.as_mut() {
            self.evaluator.commit_context_move(ctx, position);
        }
    }

    fn undo_move(&mut self, position: &mut Position, mv: Move) {
        position.undo_move(mv);
        if let Some(ctx) = self.eval_context.as_mut() {
            self.evaluator.undo_context_move(ctx);
        }
    }

    pub fn clear(&mut self) {
        self.move_ordering.clear();
        self.sennichite_detector.clear();
        self.transposition_table.clear();
        self.clear_killer_moves();
        self.eval_context = None;
        self.last_root_score = None;
        self.last_pv.clear();
        self.last_search_failed = false;
    }

    /// Restores the state that can influence an independent search while
    /// retaining the evaluator allocation owned by this session.
    pub fn reset_for_independent_search(&mut self) {
        self.clear();
        self.start_time = None;
        self.time_limit = None;
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
    }

    fn clear_killer_moves(&mut self) {
        self.killer_moves = [[None; 2]; MAX_DEPTH];
    }

    pub fn decay_history(&mut self) {
        self.move_ordering.decay();
    }

    pub fn set_emit_info(&mut self, emit_info: bool) {
        if !emit_info {
            self.observer = None;
        }
    }

    pub fn set_search_observer(&mut self, observer: Option<SharedSearchObserver>) {
        self.observer = observer;
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

    pub fn last_completed_depth(&self) -> u8 {
        self.last_completed_depth
    }

    /// Returns the raw root score from the last fully completed iteration.
    pub fn last_root_score(&self) -> Option<f32> {
        self.last_root_score
    }

    /// Returns the principal variation from the last fully completed iteration.
    pub fn last_pv(&self) -> &[Move] {
        &self.last_pv
    }

    pub fn last_search_failed(&self) -> bool {
        self.last_search_failed
    }

    pub fn search_stats(&self) -> SearchStats {
        SearchStats {
            nodes: self.nodes_searched,
            quiescence_nodes: self.quiescence_nodes_searched,
            quiescence_moves_considered: self.quiescence_moves_considered,
            quiescence_moves_generated: self.quiescence_moves_generated,
            quiescence_moves_discarded: self.quiescence_moves_discarded,
            quiescence_moves_searched: self.quiescence_moves_searched,
            quiescence_see_skips: self.quiescence_see_skips,
            quiescence_terminal_mates: self.quiescence_terminal_mates,
            check_evasion_extensions: self.check_evasion_extensions,
            aspiration_fail_lows: self.aspiration_fail_lows,
            aspiration_fail_highs: self.aspiration_fail_highs,
            aspiration_researches: self.aspiration_researches,
        }
    }

    fn search_outcome(&self, best_move: Option<Move>) -> SearchOutcome {
        SearchOutcome {
            root: best_move.map(|best_move| RootResult {
                best_move,
                score: self.last_root_score,
                completed_depth: self.last_completed_depth,
                pv: self.last_pv.clone(),
            }),
            stats: self.search_stats(),
            failed: self.last_search_failed,
        }
    }

    pub fn recover_from_search_failure(&mut self, position: &Position) {
        self.move_ordering.clear();
        self.sennichite_detector.clear();
        self.transposition_table.clear();
        self.clear_killer_moves();
        self.begin_eval_context(position);
        self.start_time = None;
        self.time_limit = None;
        self.next_time_check_nodes = 0;
        self.last_completed_depth = 0;
        self.last_root_score = None;
        self.last_pv.clear();
        self.last_search_failed = true;
    }

    fn absorb_statistics(&mut self, other: &Self) {
        self.nodes_searched += other.nodes_searched;
        self.quiescence_nodes_searched += other.quiescence_nodes_searched;
        self.quiescence_moves_considered += other.quiescence_moves_considered;
        self.quiescence_moves_generated += other.quiescence_moves_generated;
        self.quiescence_moves_discarded += other.quiescence_moves_discarded;
        self.quiescence_moves_searched += other.quiescence_moves_searched;
        self.quiescence_see_skips += other.quiescence_see_skips;
        self.quiescence_terminal_mates += other.quiescence_terminal_mates;
        self.check_evasion_extensions += other.check_evasion_extensions;
        self.aspiration_fail_lows += other.aspiration_fail_lows;
        self.aspiration_fail_highs += other.aspiration_fail_highs;
        self.aspiration_researches += other.aspiration_researches;
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

    fn is_time_up(&mut self) -> bool {
        if self.nodes_searched < self.next_time_check_nodes {
            return false;
        }
        self.next_time_check_nodes = self.nodes_searched + TIME_CHECK_NODE_INTERVAL;

        if self
            .stop_signal
            .as_ref()
            .is_some_and(|stop_signal| stop_signal.load(Ordering::Relaxed))
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

    pub fn is_sennichite_internal(&self, position: &Position) -> SennichiteStatus {
        self.sennichite_detector.adjudicate(position)
    }

    fn sennichite_score(&self, position: &Position) -> Option<f32> {
        match self.is_sennichite_internal(position) {
            SennichiteStatus::None => None,
            SennichiteStatus::Draw => Some(0.0),
            SennichiteStatus::PerpetualCheckLoss { loser } => {
                Some(if loser == position.side_to_move() {
                    -REPETITION_WIN_SCORE
                } else {
                    REPETITION_WIN_SCORE
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{parse_usi_move_for_color, position_from_sfen_or_usi};

    const V252_INCIDENT_POSITION: &str = "startpos moves \
        7g7f 8c8d 2g2f 4a3b 2f2e 8d8e 8h7g 3c3d 7i6h 2b7g+ 6h7g 3a2b \
        3i4h 2b3c 6i7h 7a7b 4g4f 5a4b 4h4g 9c9d 9g9f 6c6d 3g3f 1c1d \
        1g1f 7c7d 2i3g 8a7c 6g6f 7b6c 4g5f 6a6b 2h2i 6c5d 4i4h 8b8a \
        3g4e 3c4d 2e2d 2c2d 2i2d P*2c 2d2h B*1c P*2d 1c2d 4h4g 2a1c \
        B*3g 4d5e 5f5e 5d5e 1f1e 1d1e 1i1e S*1g 2h2g 1g1h 2g2h 4c4d \
        2h1h 4d4e 4f4e P*4f 4g5f 5e5f 5g5f G*2g S*8b 8a8b S*7a 2g1h \
        7a8b R*3i 5i6h 4f4g+ 3f3e";
    const V253_NO_BESTMOVE_POSITION: &str =
        "+B4+R1n1/6+S1k/4+B1spp/plp3p2/1N5P1/P1P1S1P1+r/2G1PPN2/1p2KG3/L1G4+s1 w G2L5Pnp 116";

    #[derive(Clone, Copy)]
    struct ZeroEvaluator;

    impl Evaluator for ZeroEvaluator {
        fn evaluate(&self, _position: &Position) -> f32 {
            0.0
        }
    }

    #[derive(Clone, Copy)]
    struct HighEvaluator;

    impl Evaluator for HighEvaluator {
        fn evaluate(&self, _position: &Position) -> f32 {
            1_000.0
        }
    }

    #[derive(Clone, Copy)]
    struct HashEvaluator;

    impl Evaluator for HashEvaluator {
        fn evaluate(&self, position: &Position) -> f32 {
            (PositionHasher::calculate_hash(position) % 2_001) as f32 - 1_000.0
        }
    }

    #[derive(Clone, Copy)]
    struct PanicEvaluator;

    impl Evaluator for PanicEvaluator {
        fn evaluate(&self, _position: &Position) -> f32 {
            panic!("injected evaluator failure");
        }
    }

    #[test]
    fn mate_scores_round_trip_through_tt_at_different_ply() {
        let win_at_ply_seven = MATE_SCORE - 7.0;
        let loss_at_ply_nine = -MATE_SCORE + 9.0;
        assert_eq!(
            MATE_SCORE - 10.0,
            score_from_tt(score_to_tt(win_at_ply_seven, 3), 6)
        );
        assert_eq!(
            -MATE_SCORE + 12.0,
            score_from_tt(score_to_tt(loss_at_ply_nine, 3), 6)
        );
        assert!(!is_history_dependent_score(win_at_ply_seven));
        assert!(is_history_dependent_score(0.0));
        assert!(is_history_dependent_score(REPETITION_WIN_SCORE));
        assert!(is_history_dependent_score(-REPETITION_WIN_SCORE));
    }

    #[test]
    fn independent_search_reset_matches_fresh_sessions() {
        let position = Position::default();
        let moves = position.legal_moves();
        let mut reused = ShogiAI::<_, 256>::new(HashEvaluator);
        reused.set_emit_info(false);

        for &mv in moves.iter().take(4) {
            let mut child = position.clone();
            child.do_move(mv);
            let mut fresh = ShogiAI::<_, 256>::new(HashEvaluator);
            fresh.set_emit_info(false);
            fresh.sennichite_detector.record_initial_position(&child);
            let mut fresh_child = child.clone();
            let expected = fresh
                .alpha_beta_search(&mut fresh_child, 2, f32::NEG_INFINITY, f32::INFINITY)
                .expect("fresh score");

            reused.reset_for_independent_search();
            reused.sennichite_detector.record_initial_position(&child);
            let actual = reused
                .alpha_beta_search(&mut child, 2, f32::NEG_INFINITY, f32::INFINITY)
                .expect("reused score");
            assert_eq!(expected, actual);
        }
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
    fn quiescence_searches_all_quiet_evasions_without_stand_pat() {
        let mut position = position_from_sfen_or_usi("4k4/9/9/9/9/9/9/9/K3R4 w - 1")
            .expect("valid checked position");
        let legal_evasions = position.legal_moves();
        assert!(position.in_check());
        assert!(!legal_evasions.is_empty());
        assert!(position.legal_quiescence_moves().is_empty());

        let mut ai = ShogiAI::<_, 256>::new(HighEvaluator);
        ai.set_emit_info(false);
        let result = ai
            .quiescence_search(&mut position, -1.0, 1.0)
            .expect("quiescence search should complete");

        assert_eq!(-1.0, result.0);
        assert_eq!(
            legal_evasions.len() as u64,
            ai.quiescence_moves_considered()
        );
        assert_eq!(legal_evasions.len() as u64, ai.quiescence_moves_searched());
        assert_eq!(0, ai.quiescence_terminal_mates());
    }

    #[test]
    fn search_thread_count_resolves_auto_and_explicit_values() {
        let available = thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(256);

        assert_eq!(available, resolve_search_threads(0));
        assert_eq!(1, resolve_search_threads(1));
        assert_eq!(16, resolve_search_threads(16));
        assert_eq!(256, resolve_search_threads(usize::MAX));
    }

    #[test]
    fn one_thread_parallel_entry_matches_legacy_search() {
        let position = Position::default();
        let mut legacy_position = position.clone();
        let mut parallel_position = position;
        let mut legacy = ShogiAI::<_, 256>::new(ZeroEvaluator);
        let mut parallel = ShogiAI::<_, 256>::new(ZeroEvaluator);
        legacy.set_emit_info(false);
        parallel.set_emit_info(false);

        let legacy_move = legacy.find_best_move(&mut legacy_position, 4, None);
        let parallel_move = parallel.find_best_move_parallel(&mut parallel_position, 4, None, 1);

        assert_eq!(legacy_move, parallel_move);
        assert_eq!(legacy.nodes_searched(), parallel.nodes_searched());
        assert_eq!(
            legacy.last_completed_depth(),
            parallel.last_completed_depth()
        );
        assert_eq!(legacy.last_root_score(), parallel.last_root_score());
        assert_eq!(legacy.last_pv(), parallel.last_pv());
    }

    #[test]
    fn completed_search_exposes_root_score_and_pv() {
        let mut position = Position::default();
        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        ai.sennichite_detector.record_position(&position);
        let observed_depths = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observer_depths = observed_depths.clone();
        ai.set_search_observer(Some(Arc::new(move |info: &SearchInfo| {
            observer_depths.lock().unwrap().push(info.depth);
        })));

        let outcome = ai.search(&mut position, SearchLimits::from_millis(2, None));
        let best_move = outcome
            .best_move()
            .expect("start position has a legal move");

        assert_eq!(2, ai.last_completed_depth());
        assert_eq!(Some(0.0), ai.last_root_score());
        assert_eq!(Some(best_move), ai.last_pv().first().copied());
        assert_eq!(vec![1, 2], *observed_depths.lock().unwrap());
        assert_eq!(ai.search_stats(), outcome.stats);
        assert_eq!(Some(0.0), outcome.root.as_ref().and_then(|root| root.score));
        assert!(!outcome.failed);
    }

    #[test]
    fn subtree_history_detects_fourth_occurrence_and_restores_on_undo() {
        let mut position =
            position_from_sfen_or_usi("4k4/9/9/9/9/9/9/9/4K4 b - 1").expect("valid position");
        let cycle = ["5i6i", "5a6a", "6i5i", "6a5a"]
            .map(|text| parse_usi_move_for_color(text, position.side_to_move()));
        let cycle = cycle.map(|mv| mv.expect("valid cycle move"));
        let mut history = GameHistory::new();
        history.record_initial_position(&position);
        for _ in 0..2 {
            for mv in cycle {
                let moved_by = position.side_to_move();
                position.do_move(mv);
                history.record_position_after_move(&position, moved_by);
            }
        }
        assert_eq!(3, history.get_position_count(&position));

        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        ai.sennichite_detector = history;
        for mv in cycle {
            let moved_by = position.side_to_move();
            ai.make_move(&mut position, mv);
            ai.sennichite_detector
                .record_position_after_move(&position, moved_by);
        }
        assert_eq!(Some(0.0), ai.sennichite_score(&position));

        for mv in cycle.into_iter().rev() {
            ai.sennichite_detector.unrecord_last_position();
            ai.undo_move(&mut position, mv);
        }
        assert_eq!(3, ai.sennichite_detector.get_position_count(&position));
        assert_eq!(None, ai.sennichite_score(&position));
    }

    #[test]
    fn parallel_search_returns_a_legal_move_and_restores_position() {
        let mut position = Position::default();
        let original_sfen = position.to_sfen_owned();
        let legal_moves = position.legal_moves();
        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        ai.set_emit_info(false);

        let best_move = ai
            .find_best_move_parallel(&mut position, 4, None, 4)
            .expect("parallel search should find a move");

        assert!(legal_moves.contains(&best_move));
        assert_eq!(original_sfen, position.to_sfen_owned());
        assert_eq!(4, ai.last_completed_depth());
    }

    #[test]
    fn parallel_search_recovers_from_worker_and_main_panics() {
        let mut position = Position::default();
        let original_sfen = position.to_sfen_owned();
        let legal_moves = position.legal_moves();
        let mut ai = ShogiAI::<_, 256>::new(PanicEvaluator);
        ai.set_emit_info(false);

        let best_move = ai
            .find_best_move_parallel(&mut position, 2, None, 4)
            .expect("panic recovery should return a legal fallback");

        assert!(legal_moves.contains(&best_move));
        assert!(ai.last_search_failed());
        assert_eq!(original_sfen, position.to_sfen_owned());
    }

    #[test]
    fn parallel_search_preserves_stop_requested_before_start() {
        let mut position = Position::default();
        let legal_moves = position.legal_moves();
        let stop_signal = Arc::new(AtomicBool::new(true));
        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        ai.set_emit_info(false);
        ai.set_stop_signal(Some(stop_signal.clone()));

        let best_move = ai
            .find_best_move_parallel(&mut position, 64, None, 4)
            .expect("pre-stopped search should return its legal fallback");

        assert!(legal_moves.contains(&best_move));
        assert_eq!(0, ai.last_completed_depth());
        assert!(stop_signal.load(Ordering::Relaxed));
    }

    #[test]
    fn v252_incident_parallel_search_keeps_the_mating_move() {
        let position =
            position_from_sfen_or_usi(V252_INCIDENT_POSITION).expect("valid incident position");
        let expected =
            parse_usi_move_for_color("S*5g", position.side_to_move()).expect("valid mating move");
        assert!(position.legal_moves().contains(&expected));

        let mut single_position = position.clone();
        let mut parallel_position = position;
        let mut single = ShogiAI::<_, 256>::new(ZeroEvaluator);
        let mut parallel = ShogiAI::<_, 256>::new(ZeroEvaluator);
        single.set_emit_info(false);
        parallel.set_emit_info(false);

        let single_move = single.find_best_move(&mut single_position, 3, None);
        assert_eq!(Some(expected), single_move);
        for _ in 0..8 {
            let original_sfen = parallel_position.to_sfen_owned();
            let parallel_move =
                parallel.find_best_move_parallel(&mut parallel_position, 3, None, 4);
            assert_eq!(single_move, parallel_move);
            assert_eq!(original_sfen, parallel_position.to_sfen_owned());
        }
    }

    #[test]
    fn v253_no_bestmove_position_returns_legal_move_with_four_threads() {
        let mut position =
            position_from_sfen_or_usi(V253_NO_BESTMOVE_POSITION).expect("valid incident position");
        let original_sfen = position.to_sfen_owned();
        let legal_moves = position.legal_moves();
        let mut ai = ShogiAI::<_, 256>::new(ZeroEvaluator);
        ai.set_emit_info(false);

        for _ in 0..8 {
            let best_move = ai
                .find_best_move_parallel(&mut position, 4, None, 4)
                .expect("incident position has a legal move");
            assert!(legal_moves.contains(&best_move));
            assert!(!ai.last_search_failed());
            assert_eq!(original_sfen, position.to_sfen_owned());
        }
    }

    #[test]
    fn four_thread_search_stress_keeps_windows_and_positions_valid() {
        for (index, sfen) in include_str!("../converted_records2016_10818.sfen")
            .lines()
            .step_by(419)
            .take(24)
            .enumerate()
        {
            let mut position =
                position_from_sfen_or_usi(sfen).expect("record position should be valid");
            let original_sfen = position.to_sfen_owned();
            let legal_moves = position.legal_moves();
            if legal_moves.is_empty() {
                continue;
            }

            let mut ai = ShogiAI::<_, 256>::new(HashEvaluator);
            ai.set_emit_info(false);
            let best_move = ai
                .find_best_move_parallel(&mut position, 8, Some(25), 4)
                .unwrap_or_else(|| panic!("sample {index} should return a legal move"));

            assert!(legal_moves.contains(&best_move), "sample {index}");
            assert!(!ai.last_search_failed(), "sample {index}");
            assert_eq!(original_sfen, position.to_sfen_owned(), "sample {index}");
        }
    }

    #[test]
    fn shared_transposition_table_handles_concurrent_access() {
        let table = Arc::new(SharedTranspositionTable::new());
        thread::scope(|scope| {
            for worker in 0..8u64 {
                let table = table.clone();
                scope.spawn(move || {
                    for index in 0..10_000u64 {
                        let hash = index.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ worker;
                        let entry = TranspositionEntry {
                            score: worker as f32,
                            depth: (index % 32) as u8,
                            node_type: NodeType::Exact,
                            best_move: None,
                            generation: 7,
                        };
                        table.insert(hash, entry);
                        if let Some(read) = table.get(hash) {
                            assert_eq!(7, read.generation);
                            assert!(read.score.is_finite());
                        }
                    }
                });
            }
        });
    }
}
