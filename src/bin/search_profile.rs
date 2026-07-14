use anyhow::{anyhow, Result};
use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use shogi_ai::ai::{
    SearchLimits, SearchStopReason, ShogiAI, DEFAULT_MATE_REPLY_CANDIDATES,
    DEFAULT_MAX_QUIESCENCE_CHECK_PLY, DEFAULT_REPLY_MATE_NODE_BUDGET,
    DEFAULT_ROOT_MATE_NODE_BUDGET,
};
use shogi_ai::evaluation::{Evaluator, HybridNnueEvaluator, SparseModel, TinyNnueModel};
use shogi_ai::utils::position_from_sfen_or_usi;
use shogi_lib::Position;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const HISTORY_CAPACITY: usize = 256;

#[derive(Parser, Debug)]
#[command(about = "Profile search speed on SFEN positions (diagnostic settings only)")]
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
    #[arg(long, default_value_t = 32)]
    samples: usize,
    #[arg(long, default_value_t = 10)]
    depth: u8,
    #[arg(long)]
    time_limit_ms: Option<u64>,
    #[arg(long)]
    nodes: Option<u64>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(
        long,
        default_value_t = DEFAULT_MAX_QUIESCENCE_CHECK_PLY,
        help = "Diagnostic override for this profile run; does not change the production engine setting"
    )]
    qcheck_ply: u16,
    #[arg(long, default_value_t = DEFAULT_ROOT_MATE_NODE_BUDGET)]
    mate_root_nodes: u64,
    #[arg(long, default_value_t = DEFAULT_REPLY_MATE_NODE_BUDGET)]
    mate_reply_nodes: u64,
    #[arg(long, default_value_t = DEFAULT_MATE_REPLY_CANDIDATES)]
    mate_reply_candidates: usize,
}

struct SharedModelEvaluator<'a> {
    model: &'a SparseModel,
}

impl Evaluator for SharedModelEvaluator<'_> {
    fn evaluate(&self, position: &Position) -> f32 {
        self.model.predict_from_position(position)
    }
}

struct SharedTinyNnueEvaluator<'a> {
    model: &'a TinyNnueModel,
}

impl Evaluator for SharedTinyNnueEvaluator<'_> {
    fn evaluate(&self, position: &Position) -> f32 {
        self.model.predict_from_position(position)
    }
}

struct SharedHybridNnueEvaluator<'a> {
    model: &'a HybridNnueEvaluator,
}

impl Evaluator for SharedHybridNnueEvaluator<'_> {
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

fn run_profile<E, F>(
    args: &Args,
    positions: &[Position],
    model_name: &str,
    mut make_evaluator: F,
) -> Result<()>
where
    E: Evaluator,
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
    let mut total_selective_qchecks_considered = 0u64;
    let mut total_selective_qchecks_searched = 0u64;
    let mut total_selective_qchecks_reply_skipped = 0u64;
    let mut total_terminal_mates = 0u64;
    let mut total_check_evasion_extensions = 0u64;
    let mut total_aspiration_fail_lows = 0u64;
    let mut total_aspiration_fail_highs = 0u64;
    let mut total_aspiration_researches = 0u64;
    let mut total_evasion_cutoffs = [0u64; 4];
    let mut total_tt_evasion_cutoffs = 0u64;
    let mut total_in_check_qnodes = 0u64;
    let mut total_negative_see_checks = 0u64;
    let mut total_repetition_hits = 0u64;
    let mut node_limit_stops = 0u64;
    let mut time_limit_stops = 0u64;
    let mut total_completed_depth = 0u64;
    let mut min_completed_depth = u8::MAX;
    let mut max_completed_depth = 0u8;
    let mut max_quiescence_ply = 0u16;
    let mut total_mate_nodes = 0u64;
    let mut total_mate_probes = 0u64;
    let mut total_mate_proven = 0u64;
    let mut total_mate_unknown = 0u64;
    let mut total_mate_rejected = 0u64;
    let mut mate_nodes_per_sample = Vec::with_capacity(args.samples);
    let start = Instant::now();

    for i in 0..args.samples {
        let mut position = positions[i % positions.len()].clone();
        let evaluator = make_evaluator();
        let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
        ai.set_emit_info(false);
        ai.set_max_quiescence_check_ply(args.qcheck_ply);
        ai.set_mate_search_budgets(args.mate_root_nodes, args.mate_reply_nodes);
        ai.set_mate_reply_candidates(args.mate_reply_candidates);
        ai.sennichite_detector.record_position(&position);
        let report = ai.find_best_move_with_limits(
            &mut position,
            SearchLimits {
                max_depth: args.depth,
                time_limit_ms: args.time_limit_ms,
                node_limit: args.nodes,
            },
        );
        total_nodes += ai.nodes_searched();
        total_quiescence_nodes += ai.quiescence_nodes_searched();
        total_quiescence_moves_considered += ai.quiescence_moves_considered();
        total_quiescence_moves_generated += ai.quiescence_moves_generated();
        total_quiescence_moves_discarded += ai.quiescence_moves_discarded();
        total_quiescence_moves_searched += ai.quiescence_moves_searched();
        total_quiescence_see_skips += ai.quiescence_see_skips();
        total_quiescence_terminal_mates += ai.quiescence_terminal_mates();
        total_selective_qchecks_considered += ai.selective_qchecks_considered();
        total_selective_qchecks_searched += ai.selective_qchecks_searched();
        total_selective_qchecks_reply_skipped += ai.selective_qchecks_reply_skipped();
        total_terminal_mates += report.terminal_mates;
        total_check_evasion_extensions += ai.check_evasion_extensions();
        total_aspiration_fail_lows += ai.aspiration_fail_lows();
        total_aspiration_fail_highs += ai.aspiration_fail_highs();
        total_aspiration_researches += ai.aspiration_researches();
        for (total, value) in total_evasion_cutoffs
            .iter_mut()
            .zip(report.quiescence_evasion_cutoffs)
        {
            *total += value;
        }
        total_tt_evasion_cutoffs += report.quiescence_tt_evasion_cutoffs;
        max_quiescence_ply = max_quiescence_ply.max(report.max_quiescence_ply);
        total_in_check_qnodes += report.in_check_qnodes;
        total_negative_see_checks += report.negative_see_checks_considered;
        total_repetition_hits += report.repetition_hits;
        node_limit_stops += u64::from(report.stop_reason == SearchStopReason::NodeLimit);
        time_limit_stops += u64::from(report.stop_reason == SearchStopReason::TimeLimit);
        total_completed_depth += u64::from(report.completed_depth);
        min_completed_depth = min_completed_depth.min(report.completed_depth);
        max_completed_depth = max_completed_depth.max(report.completed_depth);
        total_mate_nodes += report.mate_nodes;
        total_mate_probes += report.mate_probes;
        total_mate_proven += report.mate_proven;
        total_mate_unknown += report.mate_unknown;
        total_mate_rejected += report.mate_rejected;
        mate_nodes_per_sample.push(report.mate_nodes);
    }

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let nps = if elapsed_secs > 0.0 {
        total_nodes as f64 / elapsed_secs
    } else {
        0.0
    };

    println!("model: {}", model_name);
    println!("samples: {}", args.samples);
    println!("qcheck ply: {}", args.qcheck_ply);
    println!(
        "mate root/reply nodes: {}/{}",
        args.mate_root_nodes, args.mate_reply_nodes
    );
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
        "selective qchecks considered/searched/reply-skipped: {}/{}/{}",
        total_selective_qchecks_considered,
        total_selective_qchecks_searched,
        total_selective_qchecks_reply_skipped
    );
    println!("terminal mates: {}", total_terminal_mates);
    println!(
        "check evasion extensions: {}",
        total_check_evasion_extensions
    );
    println!("aspiration fail lows: {}", total_aspiration_fail_lows);
    println!("aspiration fail highs: {}", total_aspiration_fail_highs);
    println!("aspiration researches: {}", total_aspiration_researches);
    println!(
        "q evasion cutoffs capture/king/move/drop: {}/{}/{}/{}",
        total_evasion_cutoffs[0],
        total_evasion_cutoffs[1],
        total_evasion_cutoffs[2],
        total_evasion_cutoffs[3]
    );
    println!("q TT evasion cutoffs: {}", total_tt_evasion_cutoffs);
    println!("max quiescence ply: {}", max_quiescence_ply);
    mate_nodes_per_sample.sort_unstable();
    let mate_p95 = mate_nodes_per_sample[(mate_nodes_per_sample.len() - 1) * 95 / 100];
    println!("mate nodes: {total_mate_nodes}");
    println!("mate nodes p95: {mate_p95}");
    println!("mate probes: {total_mate_probes}");
    println!("mate proven: {total_mate_proven}");
    println!("mate unknown: {total_mate_unknown}");
    println!("mate rejected: {total_mate_rejected}");
    println!("in-check qnodes: {}", total_in_check_qnodes);
    println!(
        "negative SEE checks considered: {}",
        total_negative_see_checks
    );
    println!("repetition hits: {}", total_repetition_hits);
    println!("node-limit stops: {}", node_limit_stops);
    println!("time-limit stops: {}", time_limit_stops);
    println!(
        "completed depth avg/min/max: {:.2}/{}/{}",
        total_completed_depth as f64 / args.samples as f64,
        min_completed_depth,
        max_completed_depth
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
    if args.nodes == Some(0) {
        return Err(anyhow!("--nodes must be greater than zero"));
    }

    let mut positions = load_positions(&args.positions)?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    positions.shuffle(&mut rng);

    if let Some(path) = &args.residual_nnue_weights {
        let model = HybridNnueEvaluator::new(&args.weights, path, args.residual_scale)
            .map_err(|e| anyhow!("failed to load hybrid evaluator: {e}"))?;
        run_profile::<SharedHybridNnueEvaluator<'_>, _>(&args, &positions, "hybrid-nnue", || {
            SharedHybridNnueEvaluator { model: &model }
        })
    } else if let Some(path) = &args.nnue_weights {
        let model = TinyNnueModel::load(path)
            .map_err(|e| anyhow!("failed to load {}: {}", path.display(), e))?;
        run_profile::<SharedTinyNnueEvaluator<'_>, _>(&args, &positions, "tiny-nnue", || {
            SharedTinyNnueEvaluator { model: &model }
        })
    } else {
        let model = load_model(&args.weights)?;
        run_profile::<SharedModelEvaluator<'_>, _>(&args, &positions, "sparse", || {
            SharedModelEvaluator { model: &model }
        })
    }
}
