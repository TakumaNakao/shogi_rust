use crate::ai::ShogiAI;
use crate::evaluation::{EngineEvaluator, HybridNnueEvaluator};
use shogi_core::{Color, Move, Piece};
use shogi_lib::Position;
use std::io::{self, BufRead, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use crate::utils::{format_move_usi, parse_usi_move, position_from_sfen_or_usi};

// --- USI Engine Logic ---

const ENGINE_NAME: &str = "Shogi AI";
const ENGINE_AUTHOR: &str = "Gemini";
const HISTORY_CAPACITY: usize = 256;
const OVERWRITE_VALUE: f32 = 0.0;
const SEARCH_THREAD_STACK_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy)]
struct SearchLimits {
    max_depth: u8,
    time_limit_ms: Option<u64>,
}

fn print_bestmove(best_move: Option<Move>) {
    if let Some(best_move) = best_move {
        println!("bestmove {}", format_move_usi(best_move));
    } else {
        println!("bestmove resign");
    }
    let _ = io::stdout().flush();
}

struct UsiEngine {
    position: Position,
    stop_signal: Option<Arc<AtomicBool>>,
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
        UsiEngine {
            position: Position::default(),
            stop_signal: None,
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
                let new_ai = ShogiAI::new(Arc::new(evaluator));
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
        self.position = if tokens.get(1) == Some(&"sfen") {
            let moves_idx = tokens
                .iter()
                .position(|&s| s == "moves")
                .unwrap_or(tokens.len());
            let sfen = tokens[2..moves_idx].join(" ");
            position_from_sfen_or_usi(&sfen).unwrap_or_else(Position::default)
        } else {
            Position::default()
        };

        if let Some(moves_idx) = tokens.iter().position(|&s| s == "moves") {
            for move_str in &tokens[moves_idx + 1..] {
                if let Some(mut mv) = parse_usi_move(move_str) {
                    if let Move::Drop { piece, to } = mv {
                        let colored_piece =
                            Piece::new(piece.piece_kind(), self.position.side_to_move());
                        mv = Move::Drop {
                            piece: colored_piece,
                            to,
                        };
                    }
                    self.position.do_move(mv);
                }
            }
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
            time_limit_ms,
        }
    }

    fn handle_go(&mut self, tokens: &[&str]) {
        if let Some(active_search) = &self.stop_signal {
            active_search.store(true, Ordering::Relaxed);
        }

        if self
            .ai
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_none()
        {
            println!("info string Error: Evaluation file is not set. Use 'setoption name EvalFile value <path>'");
            print_bestmove(None);
            return;
        }

        // 思考開始前に履歴を減衰させる
        if let Some(ai_instance) = self
            .ai
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_mut()
        {
            ai_instance.decay_history();
        }

        let limits = self.parse_go_limits(tokens);
        let mut position = self.position.clone();
        let stop_signal = Arc::new(AtomicBool::new(false));
        self.stop_signal = Some(stop_signal.clone());
        let ai = self.ai.clone();
        let threads = self.threads;
        let spawn_fallback = position.legal_moves().first().copied();
        let recovery_position = position.clone();

        let spawn_result = thread::Builder::new()
            .name("usi-search".to_string())
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

                    thinking_ai.set_stop_signal(Some(stop_signal.clone()));
                    thinking_ai.set_emit_info(true);
                    let search_result = catch_unwind(AssertUnwindSafe(|| {
                        thinking_ai.find_best_move_parallel(
                            &mut position,
                            limits.max_depth,
                            limits.time_limit_ms,
                            threads,
                        )
                    }));
                    let best_move = match search_result {
                        Ok(best_move) => best_move,
                        Err(_) => {
                            stop_signal.store(true, Ordering::Relaxed);
                            position = recovery_position.clone();
                            thinking_ai.recover_from_search_failure(&position);
                            spawn_fallback
                        }
                    };
                    let message = thinking_ai
                        .last_search_failed()
                        .then_some("Internal search error; returning a legal fallback move");
                    thinking_ai.set_stop_signal(None);
                    (best_move, message)
                }));

                let (best_move, message) = match response {
                    Ok(response) => response,
                    Err(_) => {
                        stop_signal.store(true, Ordering::Relaxed);
                        // State repair must not be able to suppress the protocol response.
                        let _ = catch_unwind(AssertUnwindSafe(|| {
                            let mut ai_lock =
                                ai.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                            if let Some(thinking_ai) = ai_lock.as_mut() {
                                thinking_ai.recover_from_search_failure(&recovery_position);
                                thinking_ai.set_stop_signal(None);
                            }
                        }));
                        (
                            spawn_fallback,
                            Some("Internal search thread failure; returning a legal fallback move"),
                        )
                    }
                };
                if let Some(message) = message {
                    println!("info string {}", message);
                }
                print_bestmove(best_move);
            });

        if let Err(error) = spawn_result {
            if let Some(stop_signal) = &self.stop_signal {
                stop_signal.store(true, Ordering::Relaxed);
            }
            println!("info string Failed to start search thread: {}", error);
            print_bestmove(spawn_fallback);
        }
    }

    fn handle_stop(&self) {
        if let Some(stop_signal) = &self.stop_signal {
            stop_signal.store(true, Ordering::Relaxed);
        }
    }
}

pub fn run_usi() {
    let mut engine = UsiEngine::new();
    engine.run();
}
