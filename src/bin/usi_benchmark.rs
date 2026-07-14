use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use shogi_ai::evaluation::SparseModel;
use shogi_ai::sennichite::{SennichiteDetector, SennichiteStatus};
use shogi_ai::utils::{parse_usi_move, parse_usi_move_for_color, position_from_sfen_or_usi};
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
    #[arg(long)]
    new_residual_weights: Option<PathBuf>,
    #[arg(long)]
    baseline_residual_weights: Option<PathBuf>,
    #[arg(long, default_value_t = 1.0)]
    new_residual_scale: f32,
    #[arg(long, default_value_t = 1.0)]
    baseline_residual_scale: f32,
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
    #[arg(long, default_value_t = false)]
    adjudicate_at_max_plies: bool,
    #[arg(long, default_value_t = 1)]
    jobs: usize,
    /// Keep one USI process per side for all games (low-memory selfplay mode).
    #[arg(long, default_value_t = false)]
    persistent_engines: bool,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    record_dir: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GameResult {
    NewWin,
    BaselineWin,
    Draw,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GameEndReason {
    Resign,
    IllegalMove,
    RepetitionDraw,
    PerpetualCheckLoss,
    MaxPliesAdjudication,
    MaxPliesDraw,
}

const GAME_END_REASONS: [GameEndReason; 6] = [
    GameEndReason::Resign,
    GameEndReason::IllegalMove,
    GameEndReason::RepetitionDraw,
    GameEndReason::PerpetualCheckLoss,
    GameEndReason::MaxPliesAdjudication,
    GameEndReason::MaxPliesDraw,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GameObservation {
    reason: GameEndReason,
    plies: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PlyStatistics {
    count: usize,
    average: f64,
    median: f64,
}

#[derive(Debug)]
struct PlayedGame {
    result: GameResult,
    reason: GameEndReason,
    moves: Vec<String>,
}

struct EngineProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl EngineProcess {
    fn start(
        engine_path: &Path,
        weights_path: &Path,
        residual_weights_path: Option<&Path>,
        residual_scale: f32,
        depth: u8,
        time_limit_ms: u64,
    ) -> Result<Self> {
        let mut child = Command::new(engine_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("failed to spawn {}", engine_path.display()))?;

        let stdin = child.stdin.take().context("engine stdin is unavailable")?;
        let stdout = child
            .stdout
            .take()
            .context("engine stdout is unavailable")?;
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
        if let Some(residual_weights_path) = residual_weights_path {
            engine.send(&format!(
                "setoption name ResidualEvalFile value {}",
                residual_weights_path.display()
            ))?;
            engine.send(&format!(
                "setoption name ResidualScale value {}",
                residual_scale
            ))?;
        }
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
        let command = build_position_command(sfen, moves);
        self.send(&command)?;
        self.send("go")?;

        loop {
            let line = self.read_line()?;
            if let Some(rest) = line.strip_prefix("bestmove ") {
                return Ok(rest
                    .split_whitespace()
                    .next()
                    .unwrap_or("resign")
                    .to_string());
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

fn build_position_command(start_position: &str, moves: &[String]) -> String {
    let start_position = start_position.trim();
    let (base, opening_moves) =
        if let Some(opening_moves) = start_position.strip_prefix("startpos moves ") {
            ("startpos".to_string(), Some(opening_moves))
        } else if start_position == "startpos" {
            ("startpos".to_string(), None)
        } else {
            let sfen = start_position
                .strip_prefix("sfen ")
                .unwrap_or(start_position);
            if let Some((sfen, opening_moves)) = sfen.split_once(" moves ") {
                (format!("sfen {}", sfen), Some(opening_moves))
            } else {
                (format!("sfen {}", sfen), None)
            }
        };

    let mut command = format!("position {}", base);
    if opening_moves.is_some() || !moves.is_empty() {
        command.push_str(" moves");
        if let Some(opening_moves) = opening_moves {
            command.push(' ');
            command.push_str(opening_moves);
        }
        if !moves.is_empty() {
            command.push(' ');
            command.push_str(&moves.join(" "));
        }
    }
    command
}

fn replay_start_position(start_position: &str) -> Option<(Position, Vec<Position>)> {
    let start_position = start_position.trim();
    let (base, opening_moves) =
        if let Some(opening_moves) = start_position.strip_prefix("startpos moves ") {
            ("startpos", Some(opening_moves))
        } else if start_position == "startpos" {
            ("startpos", None)
        } else {
            let sfen = start_position
                .strip_prefix("sfen ")
                .unwrap_or(start_position);
            if let Some((sfen, opening_moves)) = sfen.split_once(" moves ") {
                (sfen, Some(opening_moves))
            } else {
                (sfen, None)
            }
        };

    let mut position = position_from_sfen_or_usi(base)?;
    let mut history = vec![position.clone()];
    if let Some(opening_moves) = opening_moves {
        for token in opening_moves.split_whitespace() {
            let mv = parse_usi_move_for_color(token, position.side_to_move())?;
            if !position.legal_moves().contains(&mv) {
                return None;
            }
            position.do_move(mv);
            history.push(position.clone());
        }
    }
    Some((position, history))
}

fn repetition_status<const CAPACITY: usize>(
    detector: &SennichiteDetector<CAPACITY>,
    position: &Position,
) -> SennichiteStatus {
    detector.check_sennichite_assuming_alternating_history(position)
}

fn load_positions(path: &Path) -> Result<Vec<String>> {
    let content = fs::read_to_string(path)?;
    let positions: Vec<_> = content
        .lines()
        .filter(|line| replay_start_position(line).is_some())
        .map(str::to_string)
        .collect();

    if positions.is_empty() {
        return Err(anyhow!("no valid positions loaded from {}", path.display()));
    }

    Ok(positions)
}

fn load_model(path: &Path) -> Result<SparseModel> {
    let mut model = SparseModel::new(0.0, 0.0);
    model
        .load(path)
        .map_err(|e| anyhow!("failed to load {}: {}", path.display(), e))?;
    Ok(model)
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
    adjudication_model: Option<&SparseModel>,
    start_sfen: &str,
    new_is_black: bool,
    max_plies: usize,
) -> Result<PlayedGame> {
    let (mut position, start_history) = replay_start_position(start_sfen)
        .ok_or_else(|| anyhow!("invalid start sfen: {}", start_sfen))?;
    let mut moves = Vec::new();
    let mut detector = SennichiteDetector::<HISTORY_CAPACITY>::new();
    for historical_position in &start_history {
        detector.record_position(historical_position);
    }
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
            return Ok(PlayedGame {
                result: if new_to_move {
                    GameResult::BaselineWin
                } else {
                    GameResult::NewWin
                },
                reason: GameEndReason::Resign,
                moves,
            });
        };

        if !position.legal_moves().contains(&mv) {
            return Ok(PlayedGame {
                result: if new_to_move {
                    GameResult::BaselineWin
                } else {
                    GameResult::NewWin
                },
                reason: GameEndReason::IllegalMove,
                moves,
            });
        }

        position.do_move(mv);
        moves.push(bestmove);
        detector.record_position(&position);

        match repetition_status(&detector, &position) {
            SennichiteStatus::Draw => {
                return Ok(PlayedGame {
                    result: GameResult::Draw,
                    reason: GameEndReason::RepetitionDraw,
                    moves,
                });
            }
            SennichiteStatus::PerpetualCheckLoss => {
                return Ok(PlayedGame {
                    result: if new_to_move {
                        GameResult::BaselineWin
                    } else {
                        GameResult::NewWin
                    },
                    reason: GameEndReason::PerpetualCheckLoss,
                    moves,
                });
            }
            SennichiteStatus::PerpetualCheckWin => {
                return Ok(PlayedGame {
                    result: if new_to_move {
                        GameResult::NewWin
                    } else {
                        GameResult::BaselineWin
                    },
                    reason: GameEndReason::PerpetualCheckLoss,
                    moves,
                });
            }
            SennichiteStatus::None => {}
        }
    }

    if let Some(model) = adjudication_model {
        let score = model.predict_from_position(&position);
        let score_for_new = match position.side_to_move() {
            Color::Black if new_is_black => score,
            Color::White if !new_is_black => score,
            _ => -score,
        };

        if score_for_new > 0.0 {
            Ok(PlayedGame {
                result: GameResult::NewWin,
                reason: GameEndReason::MaxPliesAdjudication,
                moves,
            })
        } else if score_for_new < 0.0 {
            Ok(PlayedGame {
                result: GameResult::BaselineWin,
                reason: GameEndReason::MaxPliesAdjudication,
                moves,
            })
        } else {
            Ok(PlayedGame {
                result: GameResult::Draw,
                reason: GameEndReason::MaxPliesAdjudication,
                moves,
            })
        }
    } else {
        Ok(PlayedGame {
            result: GameResult::Draw,
            reason: GameEndReason::MaxPliesDraw,
            moves,
        })
    }
}

fn write_game_record(
    record_dir: &Path,
    game_index: usize,
    new_is_black: bool,
    start_sfen: &str,
    game: &PlayedGame,
) -> Result<()> {
    let side_label = if new_is_black { "black" } else { "white" };
    let result_label = format!("{:?}", game.result);
    let path = record_dir.join(format!(
        "game_{:03}_new_{}_{}.usi",
        game_index + 1,
        side_label,
        result_label
    ));

    let mut content = String::new();
    content.push_str(&format!("result {:?}\n", game.result));
    content.push_str(&format!("reason {:?}\n", game.reason));
    content.push_str(&format!("new_as {}\n", side_label));
    content.push_str(&format!("start_sfen {}\n", start_sfen));
    content.push_str(&build_position_command(start_sfen, &game.moves));
    content.push('\n');

    fs::write(path, content)?;
    Ok(())
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

fn total_score_interval(
    new_wins: usize,
    baseline_wins: usize,
    draws: usize,
    z: f64,
) -> Option<(f64, f64)> {
    let games = new_wins + baseline_wins + draws;
    if games == 0 {
        return None;
    }

    let n = games as f64;
    let mean = (new_wins as f64 + draws as f64 * 0.5) / n;
    let second_moment = (new_wins as f64 + draws as f64 * 0.25) / n;
    let variance = (second_moment - mean * mean).max(0.0);
    let margin = z * (variance / n).sqrt();
    Some(((mean - margin).max(0.0), (mean + margin).min(1.0)))
}

fn ply_statistics(plies: impl IntoIterator<Item = usize>) -> Option<PlyStatistics> {
    let mut plies = plies.into_iter().collect::<Vec<_>>();
    if plies.is_empty() {
        return None;
    }
    plies.sort_unstable();
    let count = plies.len();
    let average = plies.iter().map(|&plies| plies as f64).sum::<f64>() / count as f64;
    let median = if count % 2 == 0 {
        (plies[count / 2 - 1] as f64 + plies[count / 2] as f64) / 2.0
    } else {
        plies[count / 2] as f64
    };
    Some(PlyStatistics {
        count,
        average,
        median,
    })
}

fn end_reason_statistics(
    observations: &[GameObservation],
) -> Vec<(GameEndReason, Option<PlyStatistics>)> {
    GAME_END_REASONS
        .iter()
        .copied()
        .map(|reason| {
            let statistics = ply_statistics(
                observations
                    .iter()
                    .filter(|observation| observation.reason == reason)
                    .map(|observation| observation.plies),
            );
            (reason, statistics)
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.games == 0 {
        return Err(anyhow!("--games must be greater than zero"));
    }
    if args.jobs == 0 {
        return Err(anyhow!("--jobs must be greater than zero"));
    }
    if args.persistent_engines && args.jobs != 1 {
        return Err(anyhow!(
            "--persistent-engines requires --jobs 1 (one process per side)"
        ));
    }

    let mut positions = load_positions(&args.positions)?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    positions.shuffle(&mut rng);
    let adjudication_model = if args.adjudicate_at_max_plies {
        Some(load_model(&args.baseline_weights)?)
    } else {
        None
    };

    let mut new_wins = 0usize;
    let mut baseline_wins = 0usize;
    let mut draws = 0usize;
    let mut observations = Vec::with_capacity(args.games);

    if args.persistent_engines && args.record_dir.is_some() {
        return Err(anyhow!(
            "--record-dir cannot be combined with --persistent-engines"
        ));
    }
    if let Some(record_dir) = &args.record_dir {
        fs::create_dir_all(record_dir)?;
    }

    // Selfplay mode deliberately keeps exactly one engine process per side.
    // This avoids repeatedly loading the evaluator and bounds RSS for long runs.
    if args.persistent_engines {
        let mut new_engine = EngineProcess::start(
            &args.new_engine,
            &args.new_weights,
            args.new_residual_weights.as_deref(),
            args.new_residual_scale,
            args.depth,
            args.time_limit_ms,
        )?;
        let mut baseline_engine = EngineProcess::start(
            &args.baseline_engine,
            &args.baseline_weights,
            args.baseline_residual_weights.as_deref(),
            args.baseline_residual_scale,
            args.depth,
            args.time_limit_ms,
        )?;

        for game_index in 0..args.games {
            let start_sfen = positions[(game_index / 2) % positions.len()].clone();
            let new_is_black = game_index % 2 == 0;
            let game = play_game(
                &mut new_engine,
                &mut baseline_engine,
                adjudication_model.as_ref(),
                &start_sfen,
                new_is_black,
                args.max_plies,
            )?;
            match game.result {
                GameResult::NewWin => new_wins += 1,
                GameResult::BaselineWin => baseline_wins += 1,
                GameResult::Draw => draws += 1,
            }
            observations.push(GameObservation {
                reason: game.reason,
                plies: game.moves.len(),
            });
        }

        print_summary(args.games, new_wins, baseline_wins, draws, &observations);
        return Ok(());
    }

    let play_one = |game_index: usize| -> Result<(usize, bool, String, PlayedGame)> {
        let start_sfen = positions[(game_index / 2) % positions.len()].clone();
        let new_is_black = game_index % 2 == 0;
        let mut new_engine = EngineProcess::start(
            &args.new_engine,
            &args.new_weights,
            args.new_residual_weights.as_deref(),
            args.new_residual_scale,
            args.depth,
            args.time_limit_ms,
        )?;
        let mut baseline_engine = EngineProcess::start(
            &args.baseline_engine,
            &args.baseline_weights,
            args.baseline_residual_weights.as_deref(),
            args.baseline_residual_scale,
            args.depth,
            args.time_limit_ms,
        )?;
        let result = play_game(
            &mut new_engine,
            &mut baseline_engine,
            adjudication_model.as_ref(),
            &start_sfen,
            new_is_black,
            args.max_plies,
        )?;
        Ok((game_index, new_is_black, start_sfen, result))
    };

    let mut results: Vec<_> = if args.jobs == 1 {
        (0..args.games).map(play_one).collect::<Result<Vec<_>>>()?
    } else {
        ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build()?
            .install(|| {
                (0..args.games)
                    .into_par_iter()
                    .map(play_one)
                    .collect::<Result<Vec<_>>>()
            })?
    };
    results.sort_by_key(|(game_index, _, _, _)| *game_index);

    for (game_index, new_is_black, start_sfen, game) in results {
        match game.result {
            GameResult::NewWin => new_wins += 1,
            GameResult::BaselineWin => baseline_wins += 1,
            GameResult::Draw => draws += 1,
        }

        if let Some(record_dir) = &args.record_dir {
            write_game_record(record_dir, game_index, new_is_black, &start_sfen, &game)?;
        }

        println!(
            "game {:>3}: {:?} by {:?} (new as {})",
            game_index + 1,
            game.result,
            game.reason,
            if new_is_black { "black" } else { "white" }
        );
        observations.push(GameObservation {
            reason: game.reason,
            plies: game.moves.len(),
        });
    }

    print_summary(args.games, new_wins, baseline_wins, draws, &observations);

    Ok(())
}

fn print_summary(
    games: usize,
    new_wins: usize,
    baseline_wins: usize,
    draws: usize,
    observations: &[GameObservation],
) {
    let decisive_games = new_wins + baseline_wins;
    let decisive_rate = if decisive_games == 0 {
        0.0
    } else {
        new_wins as f64 / decisive_games as f64
    };
    let total_score = new_wins as f64 + draws as f64 * 0.5;
    let total_rate = total_score / games as f64;

    println!("games: {}", games);
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
    if let Some((low, high)) = total_score_interval(new_wins, baseline_wins, draws, 1.96) {
        println!(
            "total score rate 95% CI: {:.2}%..{:.2}%",
            low * 100.0,
            high * 100.0
        );
    }
    if let Some(statistics) = ply_statistics(observations.iter().map(|game| game.plies)) {
        println!("average plies: {:.2}", statistics.average);
        println!("median plies: {:.2}", statistics.median);
    }
    println!("end reason statistics:");
    for (reason, statistics) in end_reason_statistics(observations) {
        if let Some(statistics) = statistics {
            println!(
                "  {:?}: count={} average_plies={:.2} median_plies={:.2}",
                reason, statistics.count, statistics.average, statistics.median
            );
        } else {
            println!("  {:?}: count=0 average_plies=n/a median_plies=n/a", reason);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detector_from_history(history: &[Position]) -> SennichiteDetector<HISTORY_CAPACITY> {
        let mut detector = SennichiteDetector::new();
        for position in history {
            detector.record_position(position);
        }
        detector
    }

    #[test]
    fn final_checked_repetition_with_an_intermittent_check_is_a_draw() {
        let cycle = ["5a4a", "5b6b", "4a5a", "6b5b"];
        let start = format!(
            "sfen 4k4/4R4/9/9/9/9/9/9/K8 w - 1 moves {}",
            cycle.repeat(3).join(" ")
        );
        let (position, history) = replay_start_position(&start).expect("legal repetition");
        let detector = detector_from_history(&history);

        assert!(position.in_check());
        assert_eq!(
            SennichiteStatus::PerpetualCheckLoss,
            detector.check_sennichite(&position),
            "the legacy final-position check misclassifies this cycle"
        );
        assert_eq!(
            SennichiteStatus::Draw,
            repetition_status(&detector, &position)
        );
    }

    #[test]
    fn repetition_on_the_continuous_checkers_turn_is_the_checkers_loss() {
        let cycle = ["4b5b", "5a4a", "5b4b", "4a5a"];
        let start = format!(
            "sfen 4k4/5R3/9/9/9/9/9/9/K8 b - 1 moves {}",
            cycle.repeat(3).join(" ")
        );
        let (position, history) = replay_start_position(&start).expect("legal repetition");
        let detector = detector_from_history(&history);

        assert!(!position.in_check());
        assert_eq!(
            SennichiteStatus::Draw,
            detector.check_sennichite(&position),
            "the legacy final-position check misses this perpetual check"
        );
        assert_eq!(
            SennichiteStatus::PerpetualCheckWin,
            repetition_status(&detector, &position)
        );
    }

    #[test]
    fn opening_moves_restore_every_position_for_repetition_adjudication() {
        let cycle = ["5i5h", "5a5b", "5h5i", "5b5a"];
        let start = format!("startpos moves {}", cycle.repeat(2).join(" "));
        let (mut position, history) = replay_start_position(&start).expect("legal opening history");
        let mut detector = detector_from_history(&history);

        assert_eq!(9, history.len());
        assert_eq!(Position::default().keys(), position.keys());
        assert_eq!(
            SennichiteStatus::None,
            repetition_status(&detector, &position)
        );

        for token in cycle {
            let mv = parse_usi_move_for_color(token, position.side_to_move()).expect("valid move");
            assert!(position.legal_moves().contains(&mv));
            position.do_move(mv);
            detector.record_position(&position);
        }
        assert_eq!(
            SennichiteStatus::Draw,
            repetition_status(&detector, &position)
        );
    }

    #[test]
    fn end_reason_statistics_report_counts_and_ply_distribution() {
        let observations = [
            GameObservation {
                reason: GameEndReason::Resign,
                plies: 10,
            },
            GameObservation {
                reason: GameEndReason::Resign,
                plies: 20,
            },
            GameObservation {
                reason: GameEndReason::PerpetualCheckLoss,
                plies: 30,
            },
            GameObservation {
                reason: GameEndReason::MaxPliesDraw,
                plies: 200,
            },
            GameObservation {
                reason: GameEndReason::MaxPliesDraw,
                plies: 200,
            },
            GameObservation {
                reason: GameEndReason::MaxPliesDraw,
                plies: 200,
            },
        ];

        assert_eq!(
            Some(PlyStatistics {
                count: 6,
                average: 110.0,
                median: 115.0,
            }),
            ply_statistics(observations.iter().map(|game| game.plies))
        );

        let statistics = end_reason_statistics(&observations);
        assert_eq!(GAME_END_REASONS.len(), statistics.len());
        assert_eq!(
            Some(PlyStatistics {
                count: 2,
                average: 15.0,
                median: 15.0,
            }),
            statistics
                .iter()
                .find(|(reason, _)| *reason == GameEndReason::Resign)
                .and_then(|(_, statistics)| *statistics)
        );
        assert_eq!(
            Some(PlyStatistics {
                count: 3,
                average: 200.0,
                median: 200.0,
            }),
            statistics
                .iter()
                .find(|(reason, _)| *reason == GameEndReason::MaxPliesDraw)
                .and_then(|(_, statistics)| *statistics)
        );
        assert_eq!(
            None,
            statistics
                .iter()
                .find(|(reason, _)| *reason == GameEndReason::MaxPliesAdjudication)
                .and_then(|(_, statistics)| *statistics)
        );
    }
}
