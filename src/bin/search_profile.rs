use anyhow::{anyhow, Result};
use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{Evaluator, SparseModel};
use shogi_ai::utils::position_from_sfen_or_usi;
use shogi_lib::Position;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const HISTORY_CAPACITY: usize = 256;

#[derive(Parser, Debug)]
#[command(about = "Profile search speed on SFEN positions")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long, default_value = "./taya36.sfen")]
    positions: PathBuf,
    #[arg(long, default_value_t = 32)]
    samples: usize,
    #[arg(long, default_value_t = 10)]
    depth: u8,
    #[arg(long)]
    time_limit_ms: Option<u64>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

struct SharedModelEvaluator<'a> {
    model: &'a SparseModel,
}

impl Evaluator for SharedModelEvaluator<'_> {
    fn evaluate(&self, position: &Position) -> f32 {
        self.model.predict_from_position(position)
    }
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

fn main() -> Result<()> {
    let args = Args::parse();
    if args.samples == 0 {
        return Err(anyhow!("--samples must be greater than zero"));
    }

    let model = load_model(&args.weights)?;
    let mut positions = load_positions(&args.positions)?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    positions.shuffle(&mut rng);

    let mut total_nodes = 0u64;
    let mut total_quiescence_nodes = 0u64;
    let mut total_quiescence_moves_considered = 0u64;
    let mut total_quiescence_moves_searched = 0u64;
    let mut total_quiescence_see_skips = 0u64;
    let mut total_check_evasion_extensions = 0u64;
    let mut total_single_reply_chain_extensions = 0u64;
    let start = Instant::now();

    for i in 0..args.samples {
        let mut position = positions[i % positions.len()].clone();
        let evaluator = SharedModelEvaluator { model: &model };
        let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
        ai.set_emit_info(false);
        ai.sennichite_detector.record_position(&position);
        ai.find_best_move(&mut position, args.depth, args.time_limit_ms);
        total_nodes += ai.nodes_searched();
        total_quiescence_nodes += ai.quiescence_nodes_searched();
        total_quiescence_moves_considered += ai.quiescence_moves_considered();
        total_quiescence_moves_searched += ai.quiescence_moves_searched();
        total_quiescence_see_skips += ai.quiescence_see_skips();
        total_check_evasion_extensions += ai.check_evasion_extensions();
        total_single_reply_chain_extensions += ai.single_reply_chain_extensions();
    }

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let nps = if elapsed_secs > 0.0 {
        total_nodes as f64 / elapsed_secs
    } else {
        0.0
    };

    println!("samples: {}", args.samples);
    println!("total nodes: {}", total_nodes);
    println!("quiescence nodes: {}", total_quiescence_nodes);
    println!(
        "quiescence moves considered: {}",
        total_quiescence_moves_considered
    );
    println!(
        "quiescence moves searched: {}",
        total_quiescence_moves_searched
    );
    println!("quiescence see skips: {}", total_quiescence_see_skips);
    println!(
        "check evasion extensions: {}",
        total_check_evasion_extensions
    );
    println!(
        "single reply chain extensions: {}",
        total_single_reply_chain_extensions
    );
    println!("elapsed ms: {:.2}", elapsed_secs * 1000.0);
    println!("nodes/sec: {:.2}", nps);
    println!(
        "avg nodes/sample: {:.2}",
        total_nodes as f64 / args.samples as f64
    );
    println!(
        "quiescence node rate: {:.2}%",
        total_quiescence_nodes as f64 / total_nodes.max(1) as f64 * 100.0
    );
    println!(
        "quiescence moves/node: {:.2}",
        total_quiescence_moves_considered as f64 / total_quiescence_nodes.max(1) as f64
    );
    println!(
        "quiescence see skip rate: {:.2}%",
        total_quiescence_see_skips as f64 / total_quiescence_moves_considered.max(1) as f64 * 100.0
    );

    Ok(())
}
