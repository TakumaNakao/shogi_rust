use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use shogi_core::{Color, Move, Piece, PieceKind, Position, Square};
use crate::ai::ShogiAI;
use crate::evaluation::SparseModelEvaluator;
use std::path::PathBuf;

// --- USI Parsing/Formatting Helpers (omitted for brevity, no changes) ---
fn parse_usi_move(s: &str) -> Option<Move> {
    if s.len() < 4 || s.len() > 5 { return None; }
    if s.chars().nth(1) == Some('*') {
        let piece_char = s.chars().nth(0)?;
        let piece_kind = match piece_char {
            'P' => PieceKind::Pawn, 'L' => PieceKind::Lance, 'N' => PieceKind::Knight,
            'S' => PieceKind::Silver, 'G' => PieceKind::Gold, 'B' => PieceKind::Bishop,
            'R' => PieceKind::Rook,
            _ => return None,
        };
        let to_sq = parse_square(&s[2..4])?;
        return Some(Move::Drop { piece: Piece::new(piece_kind, Color::Black), to: to_sq });
    }
    let from_sq = parse_square(&s[0..2])?;
    let to_sq = parse_square(&s[2..4])?;
    let promote = s.len() == 5 && s.chars().nth(4) == Some('+');
    Some(Move::Normal { from: from_sq, to: to_sq, promote })
}
fn parse_square(s: &str) -> Option<Square> {
    let file = s.chars().nth(0)?.to_digit(10)? as u8;
    let rank_char = s.chars().nth(1)?;
    let rank = match rank_char {
        'a' => 1, 'b' => 2, 'c' => 3, 'd' => 4, 'e' => 5, 'f' => 6, 'g' => 7, 'h' => 8, 'i' => 9,
        _ => return None,
    };
    Square::new(file, rank)
}
fn format_move_usi(mv: Move) -> String {
    match mv {
        Move::Normal { from, to, promote } => {
            format!("{}{}{}", format_square(from), format_square(to), if promote { "+" } else { "" })
        }
        Move::Drop { piece, to } => {
            let piece_char = match piece.piece_kind() {
                PieceKind::Pawn => 'P', PieceKind::Lance => 'L', PieceKind::Knight => 'N',
                PieceKind::Silver => 'S', PieceKind::Gold => 'G', PieceKind::Bishop => 'B',
                PieceKind::Rook => 'R',
                _ => ' ',
            };
            format!("{}*{}", piece_char, format_square(to))
        }
    }
}
fn format_square(sq: Square) -> String {
    let file = sq.file();
    let rank = match sq.rank() {
        1 => 'a', 2 => 'b', 3 => 'c', 4 => 'd', 5 => 'e', 6 => 'f', 7 => 'g', 8 => 'h', 9 => 'i',
        _ => ' ',
    };
    format!("{}{}", file, rank)
}


// --- USI Engine Logic ---

const ENGINE_NAME: &str = "Shogi AI";
const ENGINE_AUTHOR: &str = "Gemini";
const HISTORY_CAPACITY: usize = 256;
const DEFAULT_WEIGHTS_PATH_NAME: &str = "weightsmany.binary";

struct UsiEngine {
    position: Position,
    stop_signal: Arc<AtomicBool>,
    eval_file_path: PathBuf,
    max_depth: u8,
    search_time_limit: u64,
    ai: Arc<Mutex<ShogiAI<SparseModelEvaluator, HISTORY_CAPACITY>>>,
}

impl UsiEngine {
    fn new() -> Self {
        let mut exe_path = std::env::current_exe().expect("Failed to find executable path");
        exe_path.pop();
        let default_weights_path = exe_path.join(DEFAULT_WEIGHTS_PATH_NAME);

        let evaluator = SparseModelEvaluator::new(&default_weights_path)
            .expect("Failed to load default evaluation file on startup");
        let ai = ShogiAI::<SparseModelEvaluator, HISTORY_CAPACITY>::new(evaluator);

        UsiEngine {
            position: Position::startpos(),
            stop_signal: Arc::new(AtomicBool::new(false)),
            eval_file_path: default_weights_path,
            max_depth: 30, // Default max depth
            search_time_limit: 10000, // Default time limit in ms
            ai: Arc::new(Mutex::new(ai)),
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
        println!("option name EvalFile type string default {}", self.eval_file_path.to_string_lossy());
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
                                *self.ai.lock().unwrap() = new_ai;
                                self.eval_file_path = new_path;
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
        self.position = Position::startpos();
        self.ai.lock().unwrap().clear();
    }

    fn handle_position(&mut self, tokens: &[&str]) {
        self.position = Position::startpos();

        if let Some(moves_idx) = tokens.iter().position(|&s| s == "moves") {
            for move_str in &tokens[moves_idx + 1..] {
                if let Some(mut mv) = parse_usi_move(move_str) {
                    if let Move::Drop { piece, to } = mv {
                        let colored_piece = Piece::new(piece.piece_kind(), self.position.side_to_move());
                        mv = Move::Drop { piece: colored_piece, to };
                    }
                    let _ = self.position.make_move(mv);
                }
            }
        }
    }

    fn handle_go(&mut self, _tokens: &[&str]) {
        self.stop_signal.store(false, Ordering::SeqCst);

        let byoyomi = self.search_time_limit;
        let position = self.position.clone();
        let stop_signal = self.stop_signal.clone();
        let max_depth = self.max_depth;
        let ai = self.ai.clone();

        thread::spawn(move || {
            let mut thinking_ai = ai.lock().unwrap();
            if let Some(best_move) = thinking_ai.find_best_move(&position, max_depth, Some(byoyomi)) {
                if !stop_signal.load(Ordering::SeqCst) {
                    println!("bestmove {}", format_move_usi(best_move));
                }
            } else {
                println!("bestmove resign");
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
