use anyhow::{anyhow, Result};
use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{
    Evaluator, HalfKpModel, HalfKpSearchContext, HybridNnueEvaluator, SparseModel, TinyNnueModel,
};
use shogi_ai::utils::position_from_sfen_or_usi;
use shogi_core::Move;
use shogi_lib::Position;
use std::any::Any;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const HISTORY_CAPACITY: usize = 256;

#[derive(Parser, Debug)]
#[command(about = "Profile search speed on SFEN positions")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long)]
    nnue_weights: Option<PathBuf>,
    #[arg(long)]
    residual_nnue_weights: Option<PathBuf>,
    #[arg(long)]
    halfkp_weights: Option<PathBuf>,
    #[arg(long, default_value_t = 1.0)]
    residual_scale: f32,
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
    #[arg(long, default_value_t = 1)]
    threads: usize,
}

#[derive(Clone)]
struct SharedModelEvaluator {
    model: Arc<SparseModel>,
}

impl Evaluator for SharedModelEvaluator {
    fn evaluate(&self, position: &Position) -> f32 {
        self.model.predict_from_position(position)
    }
}

#[derive(Clone)]
struct SharedTinyNnueEvaluator {
    model: Arc<TinyNnueModel>,
}

impl Evaluator for SharedTinyNnueEvaluator {
    fn evaluate(&self, position: &Position) -> f32 {
        self.model.predict_from_position(position)
    }
}

#[derive(Clone)]
struct SharedHybridNnueEvaluator {
    model: Arc<HybridNnueEvaluator>,
}

impl Evaluator for SharedHybridNnueEvaluator {
    fn evaluate(&self, position: &Position) -> f32 {
        self.model.predict_from_position(position)
    }
}

#[derive(Clone)]
struct SharedHalfKpEvaluator {
    model: Arc<HalfKpModel>,
}

impl Evaluator for SharedHalfKpEvaluator {
    fn evaluate(&self, position: &Position) -> f32 {
        self.model.predict_from_position(position)
    }

    fn begin_context(&self, position: &Position) -> Option<Box<dyn Any + Send>> {
        self.model
            .begin_search_context(position)
            .map(|ctx| Box::new(ctx) as Box<dyn Any + Send>)
    }

    fn evaluate_context(&self, position: &Position, context: &(dyn Any + Send)) -> Option<f32> {
        context
            .downcast_ref::<HalfKpSearchContext>()
            .map(|ctx| self.model.evaluate_search_context(position, ctx))
    }

    fn prepare_context_move(&self, context: &mut (dyn Any + Send), position: &Position, mv: Move) {
        if let Some(ctx) = context.downcast_mut::<HalfKpSearchContext>() {
            self.model.prepare_search_context(ctx, position, mv);
        }
    }

    fn commit_context_move(&self, context: &mut (dyn Any + Send), position: &Position) {
        if let Some(ctx) = context.downcast_mut::<HalfKpSearchContext>() {
            self.model.commit_search_context_move(ctx, position);
        }
    }

    fn undo_context_move(&self, context: &mut (dyn Any + Send)) {
        if let Some(ctx) = context.downcast_mut::<HalfKpSearchContext>() {
            self.model.undo_search_context(ctx);
        }
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

fn run_profile<E, F>(
    args: &Args,
    positions: &[Position],
    model_name: &str,
    mut make_evaluator: F,
) -> Result<()>
where
    E: Evaluator + Clone + Send + Sync + 'static,
    F: FnMut() -> E,
{
    let mut total_nodes = 0u64;
    let mut total_quiescence_nodes = 0u64;
    let mut total_quiescence_moves_considered = 0u64;
    let mut total_quiescence_moves_generated = 0u64;
    let mut total_quiescence_moves_discarded = 0u64;
    let mut total_quiescence_moves_searched = 0u64;
    let mut total_quiescence_see_skips = 0u64;
    let mut total_quiescence_terminal_mates = 0u64;
    let mut total_check_evasion_extensions = 0u64;
    let mut total_aspiration_fail_lows = 0u64;
    let mut total_aspiration_fail_highs = 0u64;
    let mut total_aspiration_researches = 0u64;
    let mut total_completed_depth = 0u64;
    let start = Instant::now();

    for i in 0..args.samples {
        let mut position = positions[i % positions.len()].clone();
        let evaluator = make_evaluator();
        let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
        ai.set_emit_info(false);
        ai.sennichite_detector.record_position(&position);
        ai.find_best_move_parallel(&mut position, args.depth, args.time_limit_ms, args.threads);
        total_nodes += ai.nodes_searched();
        total_quiescence_nodes += ai.quiescence_nodes_searched();
        total_quiescence_moves_considered += ai.quiescence_moves_considered();
        total_quiescence_moves_generated += ai.quiescence_moves_generated();
        total_quiescence_moves_discarded += ai.quiescence_moves_discarded();
        total_quiescence_moves_searched += ai.quiescence_moves_searched();
        total_quiescence_see_skips += ai.quiescence_see_skips();
        total_quiescence_terminal_mates += ai.quiescence_terminal_mates();
        total_check_evasion_extensions += ai.check_evasion_extensions();
        total_aspiration_fail_lows += ai.aspiration_fail_lows();
        total_aspiration_fail_highs += ai.aspiration_fail_highs();
        total_aspiration_researches += ai.aspiration_researches();
        total_completed_depth += ai.last_completed_depth() as u64;
    }

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let nps = if elapsed_secs > 0.0 {
        total_nodes as f64 / elapsed_secs
    } else {
        0.0
    };

    println!("model: {}", model_name);
    println!("threads: {}", args.threads);
    println!("samples: {}", args.samples);
    println!("total nodes: {}", total_nodes);
    println!("quiescence nodes: {}", total_quiescence_nodes);
    println!(
        "quiescence moves considered: {}",
        total_quiescence_moves_considered
    );
    println!(
        "quiescence moves generated: {}",
        total_quiescence_moves_generated
    );
    println!(
        "quiescence moves discarded: {}",
        total_quiescence_moves_discarded
    );
    println!(
        "quiescence moves searched: {}",
        total_quiescence_moves_searched
    );
    println!("quiescence see skips: {}", total_quiescence_see_skips);
    println!(
        "quiescence terminal mates: {}",
        total_quiescence_terminal_mates
    );
    println!(
        "check evasion extensions: {}",
        total_check_evasion_extensions
    );
    println!("aspiration fail lows: {}", total_aspiration_fail_lows);
    println!("aspiration fail highs: {}", total_aspiration_fail_highs);
    println!("aspiration researches: {}", total_aspiration_researches);
    println!("elapsed ms: {:.2}", elapsed_secs * 1000.0);
    println!("nodes/sec: {:.2}", nps);
    println!(
        "avg nodes/sample: {:.2}",
        total_nodes as f64 / args.samples as f64
    );
    println!(
        "avg completed depth: {:.2}",
        total_completed_depth as f64 / args.samples as f64
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
        "quiescence discard rate: {:.2}%",
        total_quiescence_moves_discarded as f64 / total_quiescence_moves_generated.max(1) as f64
            * 100.0
    );
    println!(
        "quiescence see skip rate: {:.2}%",
        total_quiescence_see_skips as f64 / total_quiescence_moves_considered.max(1) as f64 * 100.0
    );

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.samples == 0 {
        return Err(anyhow!("--samples must be greater than zero"));
    }
    if args.threads == 0 {
        return Err(anyhow!("--threads must be greater than zero"));
    }

    let mut positions = load_positions(&args.positions)?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    positions.shuffle(&mut rng);

    if let Some(path) = &args.residual_nnue_weights {
        let model = Arc::new(
            HybridNnueEvaluator::new(&args.weights, path, args.residual_scale)
                .map_err(|e| anyhow!("failed to load hybrid evaluator: {e}"))?,
        );
        run_profile::<SharedHybridNnueEvaluator, _>(&args, &positions, "hybrid-nnue", || {
            SharedHybridNnueEvaluator {
                model: model.clone(),
            }
        })
    } else if let Some(path) = &args.halfkp_weights {
        let model = Arc::new(
            HalfKpModel::load(path)
                .map_err(|e| anyhow!("failed to load {}: {}", path.display(), e))?,
        );
        run_profile::<SharedHalfKpEvaluator, _>(&args, &positions, "halfkp", || {
            SharedHalfKpEvaluator {
                model: model.clone(),
            }
        })
    } else if let Some(path) = &args.nnue_weights {
        let model = Arc::new(
            TinyNnueModel::load(path)
                .map_err(|e| anyhow!("failed to load {}: {}", path.display(), e))?,
        );
        run_profile::<SharedTinyNnueEvaluator, _>(&args, &positions, "tiny-nnue", || {
            SharedTinyNnueEvaluator {
                model: model.clone(),
            }
        })
    } else {
        let model = Arc::new(load_model(&args.weights)?);
        run_profile::<SharedModelEvaluator, _>(&args, &positions, "sparse", || {
            SharedModelEvaluator {
                model: model.clone(),
            }
        })
    }
}
