use crate::ai::{SearchInfo, SearchLimits, SearchObserver, ShogiAI};
use crate::evaluation::{EngineEvaluator, HybridNnueEvaluator};
use crate::sennichite::GameHistory;
use shogi_core::{Color, Move};
use shogi_lib::Position;
use std::io::{self, BufRead, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::utils::{format_move_usi, position_and_history_from_sfen_or_usi};

// --- USI Engine Logic ---

const ENGINE_NAME: &str = "Shogi AI";
const ENGINE_AUTHOR: &str = "Gemini";
const HISTORY_CAPACITY: usize = 256;
const OVERWRITE_VALUE: f32 = 0.0;
const SEARCH_THREAD_STACK_BYTES: usize = 4 * 1024 * 1024;
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

fn emit_search_response(best_move: Option<Move>, message: Option<&str>) {
    if let Some(message) = message {
        println!("info string {message}");
    }
    if let Some(best_move) = best_move {
        println!("bestmove {}", format_move_usi(best_move));
    } else {
        println!("bestmove resign");
    }
    let _ = io::stdout().flush();
}

struct UsiSearchObserver;

impl SearchObserver for UsiSearchObserver {
    fn on_info(&self, info: &SearchInfo) {
        let pv = info
            .pv
            .iter()
            .copied()
            .map(format_move_usi)
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "info depth {} score cp {} time {} nodes {} pv {}",
            info.depth,
            usi_display_score_cp(info.root_score),
            info.elapsed.as_millis(),
            info.stats.nodes,
            pv
        );
        let _ = io::stdout().flush();
    }
}

struct SearchJob {
    generation: u64,
    stop_signal: Arc<AtomicBool>,
    handle: thread::JoinHandle<()>,
}

impl SearchJob {
    fn stop_and_join(self) {
        self.stop_signal.store(true, Ordering::Relaxed);
        if self.handle.join().is_err() {
            eprintln!(
                "info string Search job {} terminated without a response",
                self.generation
            );
        }
    }
}

struct UsiEngine {
    position: Position,
    game_history: GameHistory,
    search_job: Option<SearchJob>,
    next_search_generation: u64,
    eval_file_path: Option<PathBuf>,
    residual_eval_file_path: Option<PathBuf>,
    residual_scale: f32,
    eval_dirty: bool,
    max_depth: u8,
    search_time_limit: u64,
    threads: usize,
    ai: Arc<Mutex<Option<ShogiAI<Arc<EngineEvaluator>, HISTORY_CAPACITY>>>>,
}

impl UsiEngine {
    fn new() -> Self {
        let position = Position::default();
        let mut game_history = GameHistory::new();
        game_history.record_initial_position(&position);
        UsiEngine {
            position,
            game_history,
            search_job: None,
            next_search_generation: 1,
            eval_file_path: None,
            residual_eval_file_path: None,
            residual_scale: 1.0,
            eval_dirty: false,
            max_depth: 30,            // Default max depth
            search_time_limit: 10000, // Default time limit in ms
            threads: 0,
            ai: Arc::new(Mutex::new(None)),
        }
    }

    fn run(&mut self) {
        loop {
            self.reap_finished_search();
            let mut input = String::new();
            if io::stdin().lock().read_line(&mut input).is_err() {
                break;
            }

            let tokens: Vec<&str> = input.trim().split_whitespace().collect();

            if let Some(&command) = tokens.get(0) {
                match command {
                    "usi" => self.handle_usi(),
                    "isready" => self.handle_isready(),
                    "setoption" => self.handle_setoption(&tokens),
                    "usinewgame" => self.handle_usinewgame(),
                    "position" => self.handle_position(&tokens),
                    "go" => self.handle_go(&tokens),
                    "stop" => self.handle_stop(),
                    "quit" => {
                        self.handle_stop();
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    fn handle_usi(&self) {
        println!("id name {}", ENGINE_NAME);
        println!("id author {}", ENGINE_AUTHOR);
        println!("option name EvalFile type string default");
        println!("option name ResidualEvalFile type string default");
        println!("option name ResidualScale type string default 1.0");
        println!(
            "option name MaxDepth type spin default {} min 1 max 100",
            self.max_depth
        );
        println!(
            "option name SearchTimeLimit type spin default {} min 100 max 300000",
            self.search_time_limit
        );
        println!(
            "option name Threads type spin default {} min 0 max 256",
            self.threads
        );
        println!("usiok");
    }

    fn handle_isready(&mut self) {
        if self.eval_dirty {
            self.rebuild_ai();
            self.eval_dirty = false;
        }
        println!("readyok");
    }

    fn rebuild_ai(&mut self) {
        let Some(eval_path) = self.eval_file_path.clone() else {
            return;
        };
        let started_at = Instant::now();
        let evaluator_result = if let Some(residual_path) = &self.residual_eval_file_path {
            HybridNnueEvaluator::new(&eval_path, residual_path, self.residual_scale)
                .map(EngineEvaluator::HybridNnue)
        } else {
            EngineEvaluator::new(&eval_path, OVERWRITE_VALUE)
        };

        match evaluator_result {
            Ok(evaluator) => {
                let evaluator_name = evaluator.name();
                let mut new_ai = ShogiAI::new(Arc::new(evaluator));
                new_ai.sennichite_detector = self.game_history.clone();
                *self
                    .ai
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(new_ai);
                if let Some(residual_path) = &self.residual_eval_file_path {
                    eprintln!(
                        "info string Loaded {} evaluation files in {} ms: base={} residual={} residual_scale={}",
                        evaluator_name,
                        started_at.elapsed().as_millis(),
                        eval_path.display(),
                        residual_path.display(),
                        self.residual_scale
                    );
                } else {
                    eprintln!(
                        "info string Loaded {} evaluation file in {} ms: {}",
                        evaluator_name,
                        started_at.elapsed().as_millis(),
                        eval_path.display()
                    );
                }
            }
            Err(e) => {
                eprintln!(
                    "info string Error: Failed to load evaluation files. Error: {}",
                    e
                );
            }
        }
    }

    fn handle_setoption(&mut self, tokens: &[&str]) {
        if tokens.get(1) == Some(&"name") && tokens.get(3) == Some(&"value") {
            match tokens.get(2) {
                Some(&"EvalFile") => {
                    if let Some(path_str) = tokens.get(4) {
                        let path = PathBuf::from(path_str);
                        if self.eval_file_path.as_ref() != Some(&path) {
                            self.eval_file_path = Some(path);
                            self.eval_dirty = true;
                        }
                    }
                }
                Some(&"ResidualEvalFile") => {
                    if let Some(path_str) = tokens.get(4) {
                        let path = PathBuf::from(path_str);
                        if self.residual_eval_file_path.as_ref() != Some(&path) {
                            self.residual_eval_file_path = Some(path);
                            self.eval_dirty = true;
                        }
                    }
                }
                Some(&"ResidualScale") => {
                    if let Some(val_str) = tokens.get(4) {
                        if let Ok(val) = val_str.parse::<f32>() {
                            if self.residual_scale != val {
                                self.residual_scale = val;
                                self.eval_dirty |= self.residual_eval_file_path.is_some();
                            }
                        }
                    }
                }
                Some(&"MaxDepth") => {
                    if let Some(val_str) = tokens.get(4) {
                        if let Ok(val) = val_str.parse::<u8>() {
                            self.max_depth = val;
                        }
                    }
                }
                Some(&"SearchTimeLimit") => {
                    if let Some(val_str) = tokens.get(4) {
                        if let Ok(val) = val_str.parse::<u64>() {
                            self.search_time_limit = val;
                        }
                    }
                }
                Some(&"Threads") => {
                    if let Some(val_str) = tokens.get(4) {
                        if let Ok(value) = val_str.parse::<usize>() {
                            self.threads = value.min(256);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn handle_usinewgame(&mut self) {
        self.handle_stop();
        self.position = Position::default();
        self.game_history.clear();
        self.game_history.record_initial_position(&self.position);
        if let Some(ai_instance) = self
            .ai
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_mut()
        {
            ai_instance.clear();
        }
    }

    fn handle_position(&mut self, tokens: &[&str]) {
        let specification = tokens.get(1..).unwrap_or_default().join(" ");
        if let Some((position, history)) = position_and_history_from_sfen_or_usi(&specification) {
            self.position = position;
            self.game_history = history;
        }
    }

    fn parse_go_limits(&self, tokens: &[&str]) -> SearchLimits {
        let mut max_depth = self.max_depth;
        let mut movetime: Option<u64> = None;
        let mut byoyomi: Option<u64> = None;
        let mut black_time: Option<u64> = None;
        let mut white_time: Option<u64> = None;
        let mut infinite = false;

        let mut i = 1;
        while i < tokens.len() {
            match tokens[i] {
                "depth" => {
                    if let Some(value) = tokens.get(i + 1).and_then(|value| value.parse().ok()) {
                        max_depth = value;
                    }
                    i += 2;
                }
                "movetime" => {
                    movetime = tokens.get(i + 1).and_then(|value| value.parse().ok());
                    i += 2;
                }
                "byoyomi" => {
                    byoyomi = tokens.get(i + 1).and_then(|value| value.parse().ok());
                    i += 2;
                }
                "btime" => {
                    black_time = tokens.get(i + 1).and_then(|value| value.parse().ok());
                    i += 2;
                }
                "wtime" => {
                    white_time = tokens.get(i + 1).and_then(|value| value.parse().ok());
                    i += 2;
                }
                "infinite" => {
                    infinite = true;
                    i += 1;
                }
                _ => i += 1,
            }
        }

        let time_limit_ms = if infinite {
            None
        } else if let Some(movetime) = movetime {
            Some(movetime)
        } else {
            let side_time = match self.position.side_to_move() {
                Color::Black => black_time,
                Color::White => white_time,
            };
            match (side_time, byoyomi) {
                (Some(main_time), Some(byoyomi)) => {
                    let main_slice = main_time / 30;
                    let byoyomi_slice = byoyomi.saturating_mul(8) / 10;
                    Some((main_slice + byoyomi_slice).clamp(100, self.search_time_limit))
                }
                (Some(main_time), None) => {
                    Some((main_time / 30).clamp(100, self.search_time_limit))
                }
                (None, Some(byoyomi)) => Some(byoyomi.clamp(100, self.search_time_limit)),
                (None, None) => Some(self.search_time_limit),
            }
        };

        SearchLimits {
            max_depth,
            time_limit: time_limit_ms.map(Duration::from_millis),
        }
    }

    fn handle_go(&mut self, tokens: &[&str]) {
        self.stop_active_search();

        if self
            .ai
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_none()
        {
            emit_search_response(
                None,
                Some(
                    "Error: Evaluation file is not set. Use 'setoption name EvalFile value <path>'",
                ),
            );
            return;
        }

        // 思考開始前に履歴を減衰させる
        if let Some(ai_instance) = self
            .ai
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_mut()
        {
            ai_instance.sennichite_detector = self.game_history.clone();
            ai_instance.decay_history();
        }

        let limits = self.parse_go_limits(tokens);
        let mut position = self.position.clone();
        let stop_signal = Arc::new(AtomicBool::new(false));
        let thread_stop_signal = stop_signal.clone();
        let ai = self.ai.clone();
        let threads = self.threads;
        let spawn_fallback = position.legal_moves().first().copied();
        let recovery_position = position.clone();
        let generation = self.next_search_generation;
        self.next_search_generation = self.next_search_generation.wrapping_add(1).max(1);

        let spawn_result = thread::Builder::new()
            .name(format!("usi-search-{generation}"))
            .stack_size(SEARCH_THREAD_STACK_BYTES)
            .spawn(move || {
                let response = catch_unwind(AssertUnwindSafe(|| {
                    let mut ai_lock = ai.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    let Some(thinking_ai) = ai_lock.as_mut() else {
                        return (
                            spawn_fallback,
                            Some("Internal search error; evaluation is unavailable"),
                        );
                    };

                    thinking_ai.set_stop_signal(Some(thread_stop_signal.clone()));
                    thinking_ai.set_search_observer(Some(Arc::new(UsiSearchObserver)));
                    let search_result = catch_unwind(AssertUnwindSafe(|| {
                        thinking_ai.search_parallel(&mut position, limits, threads)
                    }));
                    let best_move = match search_result {
                        Ok(outcome) => outcome.best_move(),
                        Err(_) => {
                            thread_stop_signal.store(true, Ordering::Relaxed);
                            position = recovery_position.clone();
                            thinking_ai.recover_from_search_failure(&position);
                            spawn_fallback
                        }
                    };
                    let message = thinking_ai
                        .last_search_failed()
                        .then_some("Internal search error; returning a legal fallback move");
                    thinking_ai.set_stop_signal(None);
                    thinking_ai.set_search_observer(None);
                    (best_move, message)
                }));

                let (best_move, message) = match response {
                    Ok(response) => response,
                    Err(_) => {
                        thread_stop_signal.store(true, Ordering::Relaxed);
                        // State repair must not be able to suppress the protocol response.
                        let _ = catch_unwind(AssertUnwindSafe(|| {
                            let mut ai_lock =
                                ai.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                            if let Some(thinking_ai) = ai_lock.as_mut() {
                                thinking_ai.recover_from_search_failure(&recovery_position);
                                thinking_ai.set_stop_signal(None);
                                thinking_ai.set_search_observer(None);
                            }
                        }));
                        (
                            spawn_fallback,
                            Some("Internal search thread failure; returning a legal fallback move"),
                        )
                    }
                };
                emit_search_response(best_move, message);
            });

        match spawn_result {
            Ok(handle) => {
                self.search_job = Some(SearchJob {
                    generation,
                    stop_signal,
                    handle,
                });
            }
            Err(error) => {
                stop_signal.store(true, Ordering::Relaxed);
                emit_search_response(
                    spawn_fallback,
                    Some(&format!("Failed to start search thread: {error}")),
                );
            }
        }
    }

    fn reap_finished_search(&mut self) {
        if self
            .search_job
            .as_ref()
            .is_some_and(|job| job.handle.is_finished())
        {
            if let Some(job) = self.search_job.take() {
                job.stop_and_join();
            }
        }
    }

    fn stop_active_search(&mut self) {
        if let Some(job) = self.search_job.take() {
            job.stop_and_join();
        }
    }

    fn handle_stop(&mut self) {
        self.stop_active_search();
    }
}

pub fn run_usi() {
    let mut engine = UsiEngine::new();
    engine.run();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sennichite::SennichiteStatus;

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
    fn position_command_preserves_the_complete_move_history() {
        let mut engine = UsiEngine::new();
        let command = "position sfen 4k4/9/9/9/9/9/9/9/4K4 b - 1 moves \
            5i6i 5a6a 6i5i 6a5a \
            5i6i 5a6a 6i5i 6a5a";
        let tokens = command.split_whitespace().collect::<Vec<_>>();
        engine.handle_position(&tokens);

        assert_eq!(9, engine.game_history.len());
        assert_eq!(3, engine.game_history.get_position_count(&engine.position));
        assert_eq!(
            SennichiteStatus::None,
            engine.game_history.adjudicate(&engine.position)
        );
    }
}
