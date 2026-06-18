use anyhow::{anyhow, Result};
use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{extract_kpp_features, Evaluator, SparseModel};
use shogi_ai::sennichite::SennichiteStatus;
use shogi_ai::utils::position_from_sfen_or_usi;
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
    #[arg(long, default_value_t = 200)]
    max_plies: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

struct SharedModelEvaluator<'a> {
    model: &'a SparseModel,
}

impl Evaluator for SharedModelEvaluator<'_> {
    fn evaluate(&self, position: &Position) -> f32 {
        let features = extract_kpp_features(position);
        self.model.predict(position, &features)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GameResult {
    NewWin,
    BaselineWin,
    Draw,
}

fn load_model(path: &Path) -> Result<SparseModel> {
    let mut model = SparseModel::new(0.0, 0.0);
    model
        .load(path)
        .map_err(|e| anyhow!("failed to load {}: {}", path.display(), e))?;
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

fn choose_move(
    model: &SparseModel,
    position: &mut Position,
    history: &[Position],
    depth: u8,
) -> Option<shogi_core::Move> {
    let evaluator = SharedModelEvaluator { model };
    let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
    for past_position in history {
        ai.sennichite_detector.record_position(past_position);
    }

    ai.alpha_beta_search(position, depth, -f32::INFINITY, f32::INFINITY)
        .and_then(|(_, pv)| pv.first().copied())
}

fn play_game(
    new_model: &SparseModel,
    baseline_model: &SparseModel,
    start_position: &Position,
    new_is_black: bool,
    depth: u8,
    max_plies: usize,
) -> GameResult {
    let mut position = start_position.clone();
    let mut history = vec![position.clone()];

    for _ in 0..max_plies {
        let new_to_move = match position.side_to_move() {
            Color::Black => new_is_black,
            Color::White => !new_is_black,
        };
        let model = if new_to_move {
            new_model
        } else {
            baseline_model
        };

        let Some(best_move) = choose_move(model, &mut position, &history, depth) else {
            return if new_to_move {
                GameResult::BaselineWin
            } else {
                GameResult::NewWin
            };
        };

        position.do_move(best_move);
        history.push(position.clone());

        let evaluator = SharedModelEvaluator { model };
        let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
        for past_position in &history {
            ai.sennichite_detector.record_position(past_position);
        }
        match ai.is_sennichite_internal(&position) {
            SennichiteStatus::Draw => return GameResult::Draw,
            SennichiteStatus::PerpetualCheckLoss => {
                return if new_to_move {
                    GameResult::BaselineWin
                } else {
                    GameResult::NewWin
                };
            }
            SennichiteStatus::None => {}
        }
    }

    GameResult::Draw
}

fn wilson_interval(successes: usize, trials: usize, z: f64) -> Option<(f64, f64)> {
    if trials == 0 {
        return None;
    }

    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let center = (p + z2 / (2.0 * n)) / (1.0 + z2 / n);
    let margin = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)) / (1.0 + z2 / n)).sqrt();
    Some((center - margin, center + margin))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.games == 0 {
        return Err(anyhow!("--games must be greater than zero"));
    }

    let new_model = load_model(&args.new_weights)?;
    let baseline_model = load_model(&args.baseline_weights)?;
    let mut positions = load_positions(&args.positions)?;

    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    positions.shuffle(&mut rng);

    let mut new_wins = 0usize;
    let mut baseline_wins = 0usize;
    let mut draws = 0usize;

    for game_index in 0..args.games {
        let start_position = &positions[(game_index / 2) % positions.len()];
        let new_is_black = game_index % 2 == 0;
        let result = play_game(
            &new_model,
            &baseline_model,
            start_position,
            new_is_black,
            args.depth,
            args.max_plies,
        );

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

    Ok(())
}
