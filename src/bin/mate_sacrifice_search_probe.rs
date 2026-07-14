use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::Deserialize;
use shogi_ai::ai::{
    SearchLimits, ShogiAI, DEFAULT_MATE_REPLY_CANDIDATES, DEFAULT_REPLY_MATE_NODE_BUDGET,
    DEFAULT_ROOT_MATE_NODE_BUDGET,
};
use shogi_ai::evaluation::{Evaluator, SparseModel};
use shogi_ai::search_quality::{MateOracle, MateProof};
use shogi_ai::utils::{format_move_usi, position_from_sfen_or_usi};
use shogi_lib::Position;
use std::fs;
use std::path::{Path, PathBuf};

const HISTORY_CAPACITY: usize = 256;

#[derive(Parser, Debug)]
#[command(about = "Measure root mate choices on a mate-sacrifice development suite")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(
        long,
        default_value = "data/search_quality/generated/dev_mate_sacrifice.jsonl",
        help = "Development-suite JSONL only; holdout input is prohibited"
    )]
    input: PathBuf,
    #[arg(long, default_value_t = 7)]
    depth: u8,
    #[arg(long)]
    nodes: Option<u64>,
    #[arg(long)]
    time_limit_ms: Option<u64>,
    #[arg(long, default_value_t = 2_000_000)]
    proof_node_limit: u64,
    #[arg(long, default_value_t = DEFAULT_ROOT_MATE_NODE_BUDGET)]
    mate_root_nodes: u64,
    #[arg(long, default_value_t = DEFAULT_REPLY_MATE_NODE_BUDGET)]
    mate_reply_nodes: u64,
    #[arg(long, default_value_t = DEFAULT_MATE_REPLY_CANDIDATES)]
    mate_reply_candidates: usize,
    #[arg(long, default_value_t = 0)]
    limit: usize,
}

#[derive(Deserialize)]
struct MateRecord {
    source_index: usize,
    sfen: String,
    first_move: String,
    mate_horizon: u8,
}

struct SharedEvaluator<'a> {
    model: &'a SparseModel,
}

impl Evaluator for SharedEvaluator<'_> {
    fn evaluate(&self, position: &Position) -> f32 {
        self.model.predict_from_position(position)
    }
}

fn load_model(path: &Path) -> Result<SparseModel> {
    let mut model = SparseModel::new(0.0, 0.0);
    model
        .load(path)
        .map_err(|error| anyhow!("failed to load {}: {error}", path.display()))?;
    Ok(model)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args
        .input
        .components()
        .any(|part| part.as_os_str().to_string_lossy().contains("holdout"))
    {
        return Err(anyhow!(
            "holdout input is prohibited for mate search diagnosis"
        ));
    }
    if args.nodes == Some(0) || args.proof_node_limit == 0 {
        return Err(anyhow!("node limits must be positive"));
    }
    let content = fs::read_to_string(&args.input)
        .with_context(|| format!("failed to read {}", args.input.display()))?;
    let records = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str::<MateRecord>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let sample_count = if args.limit == 0 {
        records.len()
    } else {
        args.limit.min(records.len())
    };
    if sample_count == 0 {
        return Err(anyhow!("mate suite is empty"));
    }

    let model = load_model(&args.weights)?;
    let mut first_move_matches = 0usize;
    let mut mate_acceptable = 0usize;
    let mut mate_scores = 0usize;
    let mut unknown_proofs = 0usize;
    let mut completed_depth_total = 0u64;
    let mut total_nodes = 0u64;
    let mut mate_acceptable_indices = Vec::new();
    let mut total_mate_nodes = 0u64;
    let mut total_mate_probes = 0u64;
    let mut total_mate_proven = 0u64;
    let mut total_mate_unknown = 0u64;
    let mut total_mate_rejected = 0u64;
    let mut mate_nodes_per_sample = Vec::with_capacity(sample_count);

    for record in records.iter().take(sample_count) {
        let mut position = position_from_sfen_or_usi(&record.sfen)
            .ok_or_else(|| anyhow!("invalid sfen: {}", record.sfen))?;
        let attacker = position.side_to_move();
        let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(SharedEvaluator { model: &model });
        ai.set_emit_info(false);
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
        completed_depth_total += u64::from(report.completed_depth);
        total_nodes += report.nodes;
        mate_scores += usize::from(report.score.is_some_and(|score| score == f32::INFINITY));
        total_mate_nodes += report.mate_nodes;
        total_mate_probes += report.mate_probes;
        total_mate_proven += report.mate_proven;
        total_mate_unknown += report.mate_unknown;
        total_mate_rejected += report.mate_rejected;
        mate_nodes_per_sample.push(report.mate_nodes);

        let Some(best_move) = report.best_move else {
            continue;
        };
        first_move_matches += usize::from(format_move_usi(best_move) == record.first_move);
        position.do_move(best_move);
        let mut oracle = MateOracle::new(args.proof_node_limit);
        match oracle.prove(
            &mut position,
            attacker,
            record.mate_horizon.saturating_sub(1),
        ) {
            MateProof::ProvenMate(_) => {
                mate_acceptable += 1;
                mate_acceptable_indices.push(record.source_index);
            }
            MateProof::Unknown => unknown_proofs += 1,
            MateProof::ProvenNoMateWithinHorizon => {}
        }
    }

    println!("samples: {sample_count}");
    println!("search depth: {}", args.depth);
    println!("total nodes: {total_nodes}");
    println!(
        "completed depth avg: {:.2}",
        completed_depth_total as f64 / sample_count as f64
    );
    println!(
        "expected first move: {}/{} ({:.2}%)",
        first_move_matches,
        sample_count,
        first_move_matches as f64 / sample_count as f64 * 100.0
    );
    println!(
        "mate acceptable: {}/{} ({:.2}%)",
        mate_acceptable,
        sample_count,
        mate_acceptable as f64 / sample_count as f64 * 100.0
    );
    println!("mate scores: {mate_scores}");
    println!("unknown mate proofs: {unknown_proofs}");
    mate_nodes_per_sample.sort_unstable();
    let mate_p95 = mate_nodes_per_sample[(mate_nodes_per_sample.len() - 1) * 95 / 100];
    println!("mate nodes: {total_mate_nodes}");
    println!("mate nodes p95: {mate_p95}");
    println!("mate probes/proven/unknown/rejected: {total_mate_probes}/{total_mate_proven}/{total_mate_unknown}/{total_mate_rejected}");
    println!(
        "mate acceptable source indices: {}",
        mate_acceptable_indices
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok(())
}
