use anyhow::{anyhow, Result};
use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{Evaluator, SparseModel};
use shogi_ai::sennichite::SennichiteStatus;
use shogi_ai::utils::{format_move_usi, position_from_sfen_or_usi};
use shogi_core::Color;
use shogi_lib::Position;
use std::fs;
use std::path::{Path, PathBuf};

const HISTORY_CAPACITY: usize = 256;

#[derive(Parser, Debug)]
#[command(about = "Benchmark two KPP weight files by direct play")]
struct Args {
    #[arg(long, default_value = "./policy_weights.binary")]
    new_weights: PathBuf,
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    baseline_weights: PathBuf,
    #[arg(long, default_value = "./taya36.sfen")]
    positions: PathBuf,
    #[arg(long, default_value_t = 50)]
    games: usize,
    #[arg(long, default_value_t = 5)]
    depth: u8,
    #[arg(long)]
    time_limit_ms: Option<u64>,
    #[arg(long, default_value_t = 200)]
    max_plies: usize,
    #[arg(long, default_value_t = false)]
    adjudicate_at_max_plies: bool,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = false)]
    serial: bool,
    #[arg(long)]
    new_material_override: Option<f32>,
    #[arg(long)]
    baseline_material_override: Option<f32>,
    #[arg(long, default_value_t = false)]
    new_stateless: bool,
    #[arg(long, default_value_t = false)]
    baseline_stateless: bool,
    #[arg(long)]
    record_dir: Option<PathBuf>,
}

struct SharedModelEvaluator<'a> {
    model: &'a SparseModel,
}

impl Evaluator for SharedModelEvaluator<'_> {
    fn evaluate(&self, position: &Position) -> f32 {
        self.model.predict_from_position(position)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GameResult {
    NewWin,
    BaselineWin,
    Draw,
}

#[derive(Debug)]
struct PlayedGame {
    result: GameResult,
    moves: Vec<String>,
}

fn load_model(path: &Path, material_override: Option<f32>) -> Result<SparseModel> {
    let mut model = SparseModel::new(0.0, 0.0);
    model
        .load(path)
        .map_err(|e| anyhow!("failed to load {}: {}", path.display(), e))?;
    if let Some(material_coeff) = material_override {
        model.material_coeff = material_coeff;
    }
    Ok(model)
}

fn load_positions(path: &Path) -> Result<Vec<Position>> {
    let content = fs::read_to_string(path)?;
    let positions: Vec<_> = content
        .lines()
        .filter_map(position_from_sfen_or_usi)
        .collect();

    if positions.is_empty() {
        return Err(anyhow!("no valid positions loaded from {}", path.display()));
    }

    Ok(positions)
}

fn choose_move_stateless(
    model: &SparseModel,
    position: &mut Position,
    history: &[Position],
    depth: u8,
    time_limit_ms: Option<u64>,
) -> Option<shogi_core::Move> {
    let evaluator = SharedModelEvaluator { model };
    let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
    ai.set_emit_info(false);
    for past_position in history {
        ai.sennichite_detector.record_position(past_position);
    }

    if let Some(time_limit_ms) = time_limit_ms {
        ai.find_best_move(position, depth, Some(time_limit_ms))
    } else {
        ai.alpha_beta_search(position, depth, -f32::INFINITY, f32::INFINITY)
            .and_then(|(_, pv)| pv.first().copied())
    }
}

fn play_game(
    new_model: &SparseModel,
    baseline_model: &SparseModel,
    start_position: &Position,
    new_is_black: bool,
    depth: u8,
    time_limit_ms: Option<u64>,
    max_plies: usize,
    adjudicate_at_max_plies: bool,
    new_stateless: bool,
    baseline_stateless: bool,
) -> PlayedGame {
    let mut position = start_position.clone();
    let mut history = vec![position.clone()];
    let mut moves = Vec::new();
    let mut new_ai = ShogiAI::<_, HISTORY_CAPACITY>::new(SharedModelEvaluator { model: new_model });
    let mut baseline_ai =
        ShogiAI::<_, HISTORY_CAPACITY>::new(SharedModelEvaluator { model: baseline_model });
    new_ai.set_emit_info(false);
    baseline_ai.set_emit_info(false);
    new_ai.sennichite_detector.record_position(&position);
    baseline_ai.sennichite_detector.record_position(&position);

    for _ in 0..max_plies {
        let new_to_move = match position.side_to_move() {
            Color::Black => new_is_black,
            Color::White => !new_is_black,
        };
        let stateless = if new_to_move {
            new_stateless
        } else {
            baseline_stateless
        };
        let best_move = if stateless {
            let model = if new_to_move {
                new_model
            } else {
                baseline_model
            };
            choose_move_stateless(model, &mut position, &history, depth, time_limit_ms)
        } else {
            let current_ai = if new_to_move {
                &mut new_ai
            } else {
                &mut baseline_ai
            };

            current_ai.decay_history();
            if let Some(time_limit_ms) = time_limit_ms {
                current_ai.find_best_move(&mut position, depth, Some(time_limit_ms))
            } else {
                current_ai
                    .alpha_beta_search(&mut position, depth, -f32::INFINITY, f32::INFINITY)
                    .and_then(|(_, pv)| pv.first().copied())
            }
        };

        let Some(best_move) = best_move.or_else(|| position.legal_moves().first().copied()) else {
            return PlayedGame {
                result: if new_to_move {
                    GameResult::BaselineWin
                } else {
                    GameResult::NewWin
                },
                moves,
            };
        };

        moves.push(format_move_usi(best_move));
        position.do_move(best_move);
        history.push(position.clone());
        new_ai.sennichite_detector.record_position(&position);
        baseline_ai.sennichite_detector.record_position(&position);

        match new_ai.is_sennichite_internal(&position) {
            SennichiteStatus::Draw => {
                return PlayedGame {
                    result: GameResult::Draw,
                    moves,
                };
            }
            SennichiteStatus::PerpetualCheckLoss => {
                return PlayedGame {
                    result: if new_to_move {
                        GameResult::BaselineWin
                    } else {
                        GameResult::NewWin
                    },
                    moves,
                };
            }
            SennichiteStatus::None => {}
        }
    }

    if adjudicate_at_max_plies {
        let baseline_score = baseline_model.predict_from_position(&position);
        let baseline_score_for_new = match position.side_to_move() {
            Color::Black if new_is_black => baseline_score,
            Color::White if !new_is_black => baseline_score,
            _ => -baseline_score,
        };

        if baseline_score_for_new > 0.0 {
            PlayedGame {
                result: GameResult::NewWin,
                moves,
            }
        } else if baseline_score_for_new < 0.0 {
            PlayedGame {
                result: GameResult::BaselineWin,
                moves,
            }
        } else {
            PlayedGame {
                result: GameResult::Draw,
                moves,
            }
        }
    } else {
        PlayedGame {
            result: GameResult::Draw,
            moves,
        }
    }
}

fn write_game_record(
    record_dir: &Path,
    game_index: usize,
    new_is_black: bool,
    start_position: &Position,
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
    let start_sfen = start_position.to_sfen_owned();
    content.push_str(&format!("result {:?}\n", game.result));
    content.push_str(&format!("new_as {}\n", side_label));
    content.push_str(&format!("start_sfen {}\n", start_sfen));
    content.push_str("position sfen ");
    content.push_str(&start_sfen);
    if !game.moves.is_empty() {
        content.push_str(" moves ");
        content.push_str(&game.moves.join(" "));
    }
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

fn total_score_interval(new_wins: usize, baseline_wins: usize, draws: usize, z: f64) -> Option<(f64, f64)> {
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

fn main() -> Result<()> {
    let args = Args::parse();
    if args.games == 0 {
        return Err(anyhow!("--games must be greater than zero"));
    }

    let new_model = load_model(&args.new_weights, args.new_material_override)?;
    let baseline_model = load_model(&args.baseline_weights, args.baseline_material_override)?;
    let mut positions = load_positions(&args.positions)?;

    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    positions.shuffle(&mut rng);

    let mut new_wins = 0usize;
    let mut baseline_wins = 0usize;
    let mut draws = 0usize;

    if let Some(record_dir) = &args.record_dir {
        fs::create_dir_all(record_dir)?;
    }

    let play_one = |game_index: usize| {
        let start_position = positions[(game_index / 2) % positions.len()].clone();
        let new_is_black = game_index % 2 == 0;
        let game = play_game(
            &new_model,
            &baseline_model,
            &start_position,
            new_is_black,
            args.depth,
            args.time_limit_ms,
            args.max_plies,
            args.adjudicate_at_max_plies,
            args.new_stateless,
            args.baseline_stateless,
        );
        (game_index, new_is_black, start_position, game)
    };

    let mut results: Vec<_> = if args.serial {
        (0..args.games).map(play_one).collect()
    } else {
        (0..args.games).into_par_iter().map(play_one).collect()
    };
    results.sort_by_key(|(game_index, _, _, _)| *game_index);

    for (game_index, new_is_black, start_position, game) in results {
        match game.result {
            GameResult::NewWin => new_wins += 1,
            GameResult::BaselineWin => baseline_wins += 1,
            GameResult::Draw => draws += 1,
        }

        if let Some(record_dir) = &args.record_dir {
            write_game_record(record_dir, game_index, new_is_black, &start_position, &game)?;
        }

        println!(
            "game {:>3}: {:?} (new as {})",
            game_index + 1,
            game.result,
            if new_is_black { "black" } else { "white" }
        );
    }

    let decisive_games = new_wins + baseline_wins;
    let win_rate = if decisive_games == 0 {
        0.0
    } else {
        new_wins as f32 / decisive_games as f32 * 100.0
    };
    let total_score = new_wins as f64 + draws as f64 * 0.5;
    let total_score_rate = total_score / args.games as f64 * 100.0;

    println!("new wins: {}", new_wins);
    println!("baseline wins: {}", baseline_wins);
    println!("draws: {}", draws);
    println!("new decisive win rate: {:.2}%", win_rate);
    println!("new total score rate: {:.2}%", total_score_rate);
    if let Some((lo, hi)) = wilson_interval(new_wins, decisive_games, 1.96) {
        println!(
            "decisive win rate 95% CI: {:.2}%..{:.2}%",
            lo * 100.0,
            hi * 100.0
        );
    } else {
        println!("decisive win rate 95% CI: n/a");
    }
    if let Some((lo, hi)) = total_score_interval(new_wins, baseline_wins, draws, 1.96) {
        println!(
            "total score rate 95% CI: {:.2}%..{:.2}%",
            lo * 100.0,
            hi * 100.0
        );
    } else {
        println!("total score rate 95% CI: n/a");
    }

    Ok(())
}
