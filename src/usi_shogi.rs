use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use shogi_core::{Move, Piece};
use crate::ai::ShogiAI;
use crate::evaluation::SparseModelEvaluator;
use std::path::PathBuf;
use shogi_lib::Position;

use crate::utils::{format_move_usi, parse_usi_move};


// --- USI Engine Logic ---

const ENGINE_NAME: &str = "Shogi AI";
const ENGINE_AUTHOR: &str = "Gemini";
const HISTORY_CAPACITY: usize = 256;

struct UsiEngine {
    position: Position,
    stop_signal: Arc<AtomicBool>,
    eval_file_path: Option<PathBuf>,
    max_depth: u8,
    search_time_limit: u64,
    ai: Arc<Mutex<Option<ShogiAI<SparseModelEvaluator, HISTORY_CAPACITY>>>>,
}

impl UsiEngine {
    fn new() -> Self {
        UsiEngine {
            position: Position::default(),
            stop_signal: Arc::new(AtomicBool::new(false)),
            eval_file_path: None,
            max_depth: 30, // Default max depth
            search_time_limit: 10000, // Default time limit in ms
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
                    "quit" => break,
                    _ => {}
                }
            }
        }
    }

    fn handle_usi(&self) {
        println!("id name {}", ENGINE_NAME);
        println!("id author {}", ENGINE_AUTHOR);
        println!("option name EvalFile type string default");
        println!("option name MaxDepth type spin default {} min 1 max 100", self.max_depth);
        println!("option name SearchTimeLimit type spin default {} min 100 max 300000", self.search_time_limit);
        println!("usiok");
    }

    fn handle_isready(&self) {
        println!("readyok");
    }

    fn handle_setoption(&mut self, tokens: &[&str]) {
        if tokens.get(1) == Some(&"name") && tokens.get(3) == Some(&"value") {
            match tokens.get(2) {
                Some(&"EvalFile") => {
                    if let Some(path_str) = tokens.get(4) {
                        let new_path = PathBuf::from(path_str);
                        match SparseModelEvaluator::new(&new_path) {
                            Ok(evaluator) => {
                                let new_ai = ShogiAI::new(evaluator);
                                *self.ai.lock().unwrap() = Some(new_ai);
                                self.eval_file_path = Some(new_path);
                                eprintln!("info string Loaded new evaluation file: {}", path_str);
                            }
                            Err(e) => {
                                eprintln!("info string Error: Failed to load new evaluation file: {}. Error: {}", path_str, e);
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
                _ => {}
            }
        }
    }

    fn handle_usinewgame(&mut self) {
        self.position = Position::default();
        if let Some(ai_instance) = self.ai.lock().unwrap().as_mut() {
            ai_instance.clear();
        }
    }

    fn handle_position(&mut self, tokens: &[&str]) {
        self.position = Position::default();

        if let Some(moves_idx) = tokens.iter().position(|&s| s == "moves") {
            for move_str in &tokens[moves_idx + 1..] {
                if let Some(mut mv) = parse_usi_move(move_str) {
                    if let Move::Drop { piece, to } = mv {
                        let colored_piece = Piece::new(piece.piece_kind(), self.position.side_to_move());
                        mv = Move::Drop { piece: colored_piece, to };
                    }
                    self.position.do_move(mv);
                }
            }
        }
    }

    fn handle_go(&mut self, _tokens: &[&str]) {
        if self.ai.lock().unwrap().is_none() {
            println!("info string Error: Evaluation file is not set. Use 'setoption name EvalFile value <path>'");
            return;
        }

        // 思考開始前に履歴を減衰させる
        if let Some(ai_instance) = self.ai.lock().unwrap().as_mut() {
            ai_instance.decay_history();
        }

        self.stop_signal.store(false, Ordering::SeqCst);

        let byoyomi = self.search_time_limit;
        let mut position = self.position.clone();
        let stop_signal = self.stop_signal.clone();
        let max_depth = self.max_depth;
        let ai = self.ai.clone();

        thread::spawn(move || {
            let mut ai_lock = ai.lock().unwrap();
            if let Some(thinking_ai) = ai_lock.as_mut() {
                if let Some(best_move) = thinking_ai.find_best_move(&mut position, max_depth, Some(byoyomi)) {
                    if !stop_signal.load(Ordering::SeqCst) {
                        println!("bestmove {}", format_move_usi(best_move));
                    }
                } else {
                    println!("bestmove resign");
                }
            }
        });
    }

    fn handle_stop(&self) {
        self.stop_signal.store(true, Ordering::SeqCst);
    }
}

pub fn run_usi() {
    let mut engine = UsiEngine::new();
    engine.run();
}
