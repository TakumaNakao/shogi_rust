use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use shogi_ai::sennichite::{SennichiteDetector, SennichiteStatus};
use shogi_ai::utils::{parse_usi_move, position_from_sfen_or_usi};
use shogi_core::{Color, Move, Piece};
use shogi_lib::Position;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

const HISTORY_CAPACITY: usize = 256;

#[derive(Parser, Debug)]
#[command(about = "Benchmark two USI engine binaries by direct play")]
struct Args {
    #[arg(long)]
    new_engine: PathBuf,
    #[arg(long)]
    baseline_engine: PathBuf,
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    new_weights: PathBuf,
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    baseline_weights: PathBuf,
    #[arg(long, default_value = "./taya36.sfen")]
    positions: PathBuf,
    #[arg(long, default_value_t = 20)]
    games: usize,
    #[arg(long, default_value_t = 10)]
    depth: u8,
    #[arg(long, default_value_t = 100)]
    time_limit_ms: u64,
    #[arg(long, default_value_t = 200)]
    max_plies: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GameResult {
    NewWin,
    BaselineWin,
    Draw,
}

struct EngineProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl EngineProcess {
    fn start(engine_path: &Path, weights_path: &Path, depth: u8, time_limit_ms: u64) -> Result<Self> {
        let mut child = Command::new(engine_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("failed to spawn {}", engine_path.display()))?;

        let stdin = child.stdin.take().context("engine stdin is unavailable")?;
        let stdout = child.stdout.take().context("engine stdout is unavailable")?;
        let mut engine = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        };

        engine.send("usi")?;
        engine.read_until("usiok")?;
        engine.send(&format!(
            "setoption name EvalFile value {}",
            weights_path.display()
        ))?;
        engine.send(&format!("setoption name MaxDepth value {}", depth))?;
        engine.send(&format!(
            "setoption name SearchTimeLimit value {}",
            time_limit_ms
        ))?;
        engine.send("isready")?;
        engine.read_until("readyok")?;

        Ok(engine)
    }

    fn send(&mut self, command: &str) -> Result<()> {
        writeln!(self.stdin, "{}", command)?;
        self.stdin.flush()?;
        Ok(())
    }

    fn read_line(&mut self) -> Result<String> {
        let mut line = String::new();
        let bytes = self.stdout.read_line(&mut line)?;
        if bytes == 0 {
            return Err(anyhow!("engine exited before producing expected output"));
        }
        Ok(line.trim().to_string())
    }

    fn read_until(&mut self, expected: &str) -> Result<()> {
        loop {
            let line = self.read_line()?;
            if line == expected {
                return Ok(());
            }
        }
    }

    fn bestmove(&mut self, sfen: &str, moves: &[String]) -> Result<String> {
        let mut command = format!("position sfen {}", sfen);
        if !moves.is_empty() {
            command.push_str(" moves ");
            command.push_str(&moves.join(" "));
        }
        self.send(&command)?;
        self.send("go")?;

        loop {
            let line = self.read_line()?;
            if let Some(rest) = line.strip_prefix("bestmove ") {
                return Ok(rest.split_whitespace().next().unwrap_or("resign").to_string());
            }
        }
    }

    fn new_game(&mut self) -> Result<()> {
        self.send("usinewgame")
    }
}

impl Drop for EngineProcess {
    fn drop(&mut self) {
        let _ = self.send("quit");
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn load_positions(path: &Path) -> Result<Vec<String>> {
    let content = fs::read_to_string(path)?;
    let positions: Vec<_> = content
        .lines()
        .filter(|line| position_from_sfen_or_usi(line).is_some())
        .map(str::to_string)
        .collect();

    if positions.is_empty() {
        return Err(anyhow!("no valid positions loaded from {}", path.display()));
    }

    Ok(positions)
}

fn parse_engine_move(position: &Position, bestmove: &str) -> Option<Move> {
    if bestmove == "resign" {
        return None;
    }

    parse_usi_move(bestmove).map(|mv| match mv {
        Move::Drop { piece, to } => Move::Drop {
            piece: Piece::new(piece.piece_kind(), position.side_to_move()),
            to,
        },
        normal => normal,
    })
}

fn play_game(
    new_engine: &mut EngineProcess,
    baseline_engine: &mut EngineProcess,
    start_sfen: &str,
    new_is_black: bool,
    max_plies: usize,
) -> Result<GameResult> {
    let mut position = position_from_sfen_or_usi(start_sfen)
        .ok_or_else(|| anyhow!("invalid start sfen: {}", start_sfen))?;
    let mut moves = Vec::new();
    let mut detector = SennichiteDetector::<HISTORY_CAPACITY>::new();
    detector.record_position(&position);
    new_engine.new_game()?;
    baseline_engine.new_game()?;

    for _ in 0..max_plies {
        let new_to_move = match position.side_to_move() {
            Color::Black => new_is_black,
            Color::White => !new_is_black,
        };
        let engine = if new_to_move {
            &mut *new_engine
        } else {
            &mut *baseline_engine
        };

        let bestmove = engine.bestmove(start_sfen, &moves)?;
        let Some(mv) = parse_engine_move(&position, &bestmove) else {
            return Ok(if new_to_move {
                GameResult::BaselineWin
            } else {
                GameResult::NewWin
            });
        };

        if !position.legal_moves().contains(&mv) {
            return Ok(if new_to_move {
                GameResult::BaselineWin
            } else {
                GameResult::NewWin
            });
        }

        position.do_move(mv);
        moves.push(bestmove);
        detector.record_position(&position);

        match detector.check_sennichite(&position) {
            SennichiteStatus::Draw => return Ok(GameResult::Draw),
            SennichiteStatus::PerpetualCheckLoss => {
                return Ok(if new_to_move {
                    GameResult::BaselineWin
                } else {
                    GameResult::NewWin
                });
            }
            SennichiteStatus::None => {}
        }
    }

    Ok(GameResult::Draw)
}

fn wilson_interval(successes: usize, trials: usize, z: f64) -> Option<(f64, f64)> {
    if trials == 0 {
        return None;
    }

    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let margin = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denominator;
    Some((center - margin, center + margin))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.games == 0 {
        return Err(anyhow!("--games must be greater than zero"));
    }

    let mut positions = load_positions(&args.positions)?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    positions.shuffle(&mut rng);

    let mut new_engine = EngineProcess::start(
        &args.new_engine,
        &args.new_weights,
        args.depth,
        args.time_limit_ms,
    )?;
    let mut baseline_engine = EngineProcess::start(
        &args.baseline_engine,
        &args.baseline_weights,
        args.depth,
        args.time_limit_ms,
    )?;

    let mut new_wins = 0usize;
    let mut baseline_wins = 0usize;
    let mut draws = 0usize;

    for game_index in 0..args.games {
        let start_sfen = &positions[(game_index / 2) % positions.len()];
        let new_is_black = game_index % 2 == 0;
        let result = play_game(
            &mut new_engine,
            &mut baseline_engine,
            start_sfen,
            new_is_black,
            args.max_plies,
        )?;

        match result {
            GameResult::NewWin => new_wins += 1,
            GameResult::BaselineWin => baseline_wins += 1,
            GameResult::Draw => draws += 1,
        }

        println!(
            "game {:>3}: {:?} (new as {})",
            game_index + 1,
            result,
            if new_is_black { "black" } else { "white" }
        );
    }

    let decisive_games = new_wins + baseline_wins;
    let decisive_rate = if decisive_games == 0 {
        0.0
    } else {
        new_wins as f64 / decisive_games as f64
    };
    let total_score = new_wins as f64 + draws as f64 * 0.5;
    let total_rate = total_score / args.games as f64;

    println!("new wins: {}", new_wins);
    println!("baseline wins: {}", baseline_wins);
    println!("draws: {}", draws);
    println!("new decisive win rate: {:.2}%", decisive_rate * 100.0);
    println!("new total score rate: {:.2}%", total_rate * 100.0);
    if let Some((low, high)) = wilson_interval(new_wins, decisive_games, 1.96) {
        println!(
            "decisive win rate 95% CI: {:.2}%..{:.2}%",
            low * 100.0,
            high * 100.0
        );
    }

    Ok(())
}
