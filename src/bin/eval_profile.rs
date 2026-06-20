use anyhow::{anyhow, Result};
use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use shogi_ai::evaluation::{HybridNnueEvaluator, SparseModel, TinyNnueModel};
use shogi_ai::utils::position_from_sfen_or_usi;
use shogi_lib::Position;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Profile KPP evaluation speed")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long)]
    nnue_weights: Option<PathBuf>,
    #[arg(long)]
    residual_nnue_weights: Option<PathBuf>,
    #[arg(long, default_value_t = 1.0)]
    residual_scale: f32,
    #[arg(long, default_value = "./taya36.sfen")]
    positions: PathBuf,
    #[arg(long, default_value_t = 4096)]
    samples: usize,
    #[arg(long, default_value_t = 100)]
    repeat: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn load_model(path: &Path) -> Result<SparseModel> {
    let mut model = SparseModel::new(0.0, 0.0);
    model
        .load(path)
        .map_err(|e| anyhow!("failed to load {}: {}", path.display(), e))?;
    Ok(model)
}

enum ProfileModel {
    Sparse(SparseModel),
    TinyNnue(TinyNnueModel),
    HybridNnue(HybridNnueEvaluator),
}

impl ProfileModel {
    fn predict_from_position(&self, position: &Position) -> f32 {
        match self {
            ProfileModel::Sparse(model) => model.predict_from_position(position),
            ProfileModel::TinyNnue(model) => model.predict_from_position(position),
            ProfileModel::HybridNnue(model) => model.predict_from_position(position),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            ProfileModel::Sparse(_) => "sparse",
            ProfileModel::TinyNnue(_) => "tiny-nnue",
            ProfileModel::HybridNnue(_) => "hybrid-nnue",
        }
    }
}

fn load_positions(path: &Path) -> Result<Vec<Position>> {
    let content = fs::read_to_string(path)?;
    let positions = content
        .lines()
        .filter_map(position_from_sfen_or_usi)
        .collect::<Vec<_>>();
    if positions.is_empty() {
        return Err(anyhow!("no valid positions loaded from {}", path.display()));
    }
    Ok(positions)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.samples == 0 {
        return Err(anyhow!("--samples must be greater than zero"));
    }
    if args.repeat == 0 {
        return Err(anyhow!("--repeat must be greater than zero"));
    }

    let model = if let Some(path) = &args.residual_nnue_weights {
        ProfileModel::HybridNnue(
            HybridNnueEvaluator::new(&args.weights, path, args.residual_scale)
                .map_err(|e| anyhow!("failed to load hybrid evaluator: {e}"))?,
        )
    } else if let Some(path) = &args.nnue_weights {
        ProfileModel::TinyNnue(
            TinyNnueModel::load(path)
                .map_err(|e| anyhow!("failed to load {}: {}", path.display(), e))?,
        )
    } else {
        ProfileModel::Sparse(load_model(&args.weights)?)
    };
    let mut positions = load_positions(&args.positions)?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    positions.shuffle(&mut rng);

    let start = Instant::now();
    let mut total_evals = 0usize;
    let mut score_sum = 0.0f64;
    let mut min_score = f32::INFINITY;
    let mut max_score = -f32::INFINITY;

    for _ in 0..args.repeat {
        for i in 0..args.samples {
            let position = &positions[i % positions.len()];
            let score = model.predict_from_position(position);
            total_evals += 1;
            score_sum += score as f64;
            min_score = min_score.min(score);
            max_score = max_score.max(score);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!("model: {}", model.name());
    println!("evals: {}", total_evals);
    println!("score sum: {:.1}", score_sum);
    println!("min score: {:.1}", min_score);
    println!("max score: {:.1}", max_score);
    println!("elapsed ms: {:.2}", elapsed * 1000.0);
    if elapsed > 0.0 {
        println!("evals/sec: {:.2}", total_evals as f64 / elapsed);
    }

    Ok(())
}
