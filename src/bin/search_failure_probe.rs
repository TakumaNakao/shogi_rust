use anyhow::{anyhow, Result};
use clap::Parser;
use serde::Serialize;
use shogi_ai::ai::{SearchLimits, ShogiAI};
use shogi_ai::evaluation::{Evaluator, SparseModel};
use shogi_ai::search_quality::{ensure_distinct_paths, load_input_positions, AtomicOutput};
use shogi_ai::utils::format_move_usi;
use shogi_lib::Position;
use std::io::Write;
use std::path::{Path, PathBuf};

const HISTORY_CAPACITY: usize = 256;

struct SharedEvaluator<'a>(&'a SparseModel);

impl Evaluator for SharedEvaluator<'_> {
    fn evaluate(&self, position: &Position) -> f32 {
        self.0.predict_from_position(position)
    }
}

#[derive(Parser, Debug)]
#[command(about = "Sweep deterministic depth and node limits over SFEN positions")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long)]
    positions: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_delimiter = ',', default_value = "3,4,5,6,7,8")]
    depths: Vec<u8>,
    #[arg(long, value_delimiter = ',', default_value = "10000,50000,200000")]
    nodes: Vec<u64>,
    #[arg(long, default_value_t = 0)]
    limit: usize,
}

#[derive(Serialize)]
struct ProbeRecord {
    source_index: usize,
    source_game_key: Option<String>,
    sfen: String,
    max_depth: u8,
    node_limit: u64,
    best_move: Option<String>,
    score: Option<String>,
    pv: Vec<String>,
    completed_depth: u8,
    nodes: u64,
    qnodes: u64,
    terminal_mates: u64,
    in_check_qnodes: u64,
    negative_see_checks_considered: u64,
    negative_see_check_searches: u64,
    repetition_hits: u64,
    resource_cycle_hits: u64,
    stop_reason: String,
}

fn load_model(path: &Path) -> Result<SparseModel> {
    let mut model = SparseModel::new(0.0, 0.0);
    model
        .load(path)
        .map_err(|e| anyhow!("failed to load {}: {e}", path.display()))?;
    Ok(model)
}

fn score_text(score: f32) -> String {
    if score == f32::INFINITY {
        "inf".to_string()
    } else if score == f32::NEG_INFINITY {
        "-inf".to_string()
    } else {
        format!("{score:.3}")
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.depths.is_empty()
        || args.depths.contains(&0)
        || args.nodes.is_empty()
        || args.nodes.contains(&0)
    {
        return Err(anyhow!("--depths and positive --nodes are required"));
    }
    ensure_distinct_paths(&[
        ("weights", &args.weights),
        ("positions", &args.positions),
        ("output", &args.output),
    ])?;
    let model = load_model(&args.weights)?;
    let (positions, _) = load_input_positions(&args.positions)?;
    let mut output = AtomicOutput::new(&args.output)?;
    let limit = if args.limit == 0 {
        usize::MAX
    } else {
        args.limit
    };

    for input in positions.into_iter().take(limit) {
        let source_index = input.source_line;
        let source_game_key = input.source_game_key;
        let position = input.position;
        for &depth in &args.depths {
            for &node_limit in &args.nodes {
                let mut search_position = position.clone();
                let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(SharedEvaluator(&model));
                ai.set_emit_info(false);
                ai.sennichite_detector.record_position(&search_position);
                let report = ai.find_best_move_with_limits(
                    &mut search_position,
                    SearchLimits {
                        max_depth: depth,
                        time_limit_ms: None,
                        node_limit: Some(node_limit),
                    },
                );
                let record = ProbeRecord {
                    source_index,
                    source_game_key: source_game_key.clone(),
                    sfen: position.to_sfen_owned(),
                    max_depth: depth,
                    node_limit,
                    best_move: report.best_move.map(format_move_usi),
                    score: report.score.map(score_text),
                    pv: report.pv.into_iter().map(format_move_usi).collect(),
                    completed_depth: report.completed_depth,
                    nodes: report.nodes,
                    qnodes: report.qnodes,
                    terminal_mates: report.terminal_mates,
                    in_check_qnodes: report.in_check_qnodes,
                    negative_see_checks_considered: report.negative_see_checks_considered,
                    negative_see_check_searches: report.negative_see_check_searches,
                    repetition_hits: report.repetition_hits,
                    resource_cycle_hits: report.resource_cycle_hits,
                    stop_reason: format!("{:?}", report.stop_reason),
                };
                serde_json::to_writer(&mut output, &record)?;
                writeln!(output)?;
            }
        }
    }
    output.commit()?;
    Ok(())
}
