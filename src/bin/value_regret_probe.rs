use anyhow::{anyhow, Result};
use clap::Parser;
use rayon::prelude::*;
use serde::Deserialize;
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{Evaluator, SparseModel};
use shogi_ai::utils::{format_move_usi, position_from_sfen_or_usi};
use shogi_core::Move;
use shogi_lib::Position;
use std::fs;
use std::path::PathBuf;

const HISTORY_CAPACITY: usize = 256;

#[derive(Parser, Debug)]
#[command(about = "Measure searched-move regret against a teacher search")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    teacher_weights: PathBuf,
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    candidate_weights: PathBuf,
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long, default_value_t = 4)]
    teacher_depth: u8,
    #[arg(long, default_value_t = 4)]
    candidate_depth: u8,
    #[arg(long)]
    max_positions: Option<usize>,
    #[arg(long, default_value_t = 300.0)]
    bad_regret_cp: f32,
    #[arg(long, default_value_t = 0)]
    jobs: usize,
    #[arg(long, default_value_t = 10)]
    show_worst: usize,
}

#[derive(Deserialize)]
struct SfenRecord {
    sfen: String,
}

struct SharedModelEvaluator<'a> {
    model: &'a SparseModel,
}

impl Evaluator for SharedModelEvaluator<'_> {
    fn evaluate(&self, position: &Position) -> f32 {
        self.model.predict_from_position(position)
    }
}

#[derive(Clone)]
struct ProbeResult {
    sfen: String,
    teacher_best_move: Option<String>,
    candidate_move: Option<String>,
    teacher_score: f32,
    candidate_score: f32,
    regret: f32,
    legal_moves: usize,
}

fn load_positions(paths: &[PathBuf]) -> Result<Vec<Position>> {
    let mut positions = Vec::new();
    for path in paths {
        let content = fs::read_to_string(path)
            .map_err(|e| anyhow!("failed to read {}: {}", path.display(), e))?;
        for (line_index, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let sfen = if line.starts_with('{') {
                let record: SfenRecord = serde_json::from_str(line).map_err(|e| {
                    anyhow!("{}:{} invalid json: {}", path.display(), line_index + 1, e)
                })?;
                record.sfen
            } else {
                line.to_string()
            };
            if let Some(position) = position_from_sfen_or_usi(&sfen) {
                positions.push(position);
            }
        }
    }
    if positions.is_empty() {
        return Err(anyhow!("no valid positions loaded"));
    }
    Ok(positions)
}

fn sanitize_score(score: f32) -> f32 {
    const LIMIT: f32 = 100_000.0;
    if score == f32::INFINITY {
        LIMIT
    } else if score == -f32::INFINITY {
        -LIMIT
    } else {
        score.clamp(-LIMIT, LIMIT)
    }
}

fn searched_root(model: &SparseModel, position: &Position, depth: u8) -> Option<(f32, Vec<Move>)> {
    let evaluator = SharedModelEvaluator { model };
    let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
    let mut root = position.clone();
    ai.set_emit_info(false);
    ai.sennichite_detector.record_position(&root);
    ai.alpha_beta_search(&mut root, depth, -f32::INFINITY, f32::INFINITY)
        .map(|(score, pv)| (sanitize_score(score), pv))
}

fn searched_move_score(
    model: &SparseModel,
    position: &Position,
    mv: Move,
    depth: u8,
) -> Option<f32> {
    let mut child = position.clone();
    child.do_move(mv);
    let evaluator = SharedModelEvaluator { model };
    let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
    ai.set_emit_info(false);
    ai.sennichite_detector.record_position(&child);
    let child_depth = depth.saturating_sub(1);
    ai.alpha_beta_search(&mut child, child_depth, -f32::INFINITY, f32::INFINITY)
        .map(|(score, _)| sanitize_score(-score))
}

fn probe_position(
    teacher: &SparseModel,
    candidate: &SparseModel,
    position: Position,
    teacher_depth: u8,
    candidate_depth: u8,
) -> Option<ProbeResult> {
    let legal_moves = position.legal_moves();
    if legal_moves.is_empty() {
        return None;
    }

    let (teacher_score, teacher_pv) = searched_root(teacher, &position, teacher_depth)?;
    let (_, candidate_pv) = searched_root(candidate, &position, candidate_depth)?;
    let candidate_move = candidate_pv.first().copied();
    let candidate_score = if let Some(mv) = candidate_move {
        searched_move_score(teacher, &position, mv, teacher_depth)?
    } else {
        teacher_score
    };
    let regret = (teacher_score - candidate_score).max(0.0);

    Some(ProbeResult {
        sfen: position.to_sfen_owned(),
        teacher_best_move: teacher_pv.first().copied().map(format_move_usi),
        candidate_move: candidate_move.map(format_move_usi),
        teacher_score,
        candidate_score,
        regret,
        legal_moves: legal_moves.len(),
    })
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn percentile(mut values: Vec<f32>, percentile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((values.len() - 1) as f32 * percentile).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.teacher_depth == 0 {
        return Err(anyhow!("--teacher-depth must be greater than zero"));
    }
    if args.candidate_depth == 0 {
        return Err(anyhow!("--candidate-depth must be greater than zero"));
    }
    if !args.bad_regret_cp.is_finite() || args.bad_regret_cp < 0.0 {
        return Err(anyhow!("--bad-regret-cp must be non-negative"));
    }
    if args.jobs > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build_global()
            .map_err(|e| anyhow!("failed to configure rayon thread pool: {e}"))?;
    }

    let mut teacher = SparseModel::new(0.0, 0.0);
    teacher.load(&args.teacher_weights)?;
    let mut candidate = SparseModel::new(0.0, 0.0);
    candidate.load(&args.candidate_weights)?;

    let mut positions = load_positions(&args.input)?;
    if let Some(max_positions) = args.max_positions {
        positions.truncate(max_positions);
    }

    let mut results = positions
        .into_par_iter()
        .filter_map(|position| {
            probe_position(
                &teacher,
                &candidate,
                position,
                args.teacher_depth,
                args.candidate_depth,
            )
        })
        .collect::<Vec<_>>();
    results.sort_by(|a, b| {
        b.regret
            .partial_cmp(&a.regret)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let regrets = results
        .iter()
        .map(|result| result.regret)
        .collect::<Vec<_>>();
    let bad_count = regrets
        .iter()
        .filter(|&&regret| regret > args.bad_regret_cp)
        .count();
    let exact_move_matches = results
        .iter()
        .filter(|result| result.teacher_best_move == result.candidate_move)
        .count();

    println!("samples: {}", results.len());
    println!("mean_regret_cp: {:.2}", mean(&regrets));
    println!("p50_regret_cp: {:.2}", percentile(regrets.clone(), 0.50));
    println!("p90_regret_cp: {:.2}", percentile(regrets.clone(), 0.90));
    println!("p95_regret_cp: {:.2}", percentile(regrets.clone(), 0.95));
    println!(
        "max_regret_cp: {:.2}",
        regrets.first().copied().unwrap_or(0.0)
    );
    println!(
        "bad_regret_count_gt_{:.0}: {} ({:.2}%)",
        args.bad_regret_cp,
        bad_count,
        if results.is_empty() {
            0.0
        } else {
            bad_count as f32 * 100.0 / results.len() as f32
        }
    );
    println!(
        "teacher_move_match: {} ({:.2}%)",
        exact_move_matches,
        if results.is_empty() {
            0.0
        } else {
            exact_move_matches as f32 * 100.0 / results.len() as f32
        }
    );

    for (idx, result) in results.iter().take(args.show_worst).enumerate() {
        println!(
            "worst[{}] regret={:.2} teacher_score={:.2} candidate_score={:.2} teacher_move={} candidate_move={} legal_moves={} sfen={}",
            idx + 1,
            result.regret,
            result.teacher_score,
            result.candidate_score,
            result.teacher_best_move.as_deref().unwrap_or("none"),
            result.candidate_move.as_deref().unwrap_or("none"),
            result.legal_moves,
            result.sfen
        );
    }

    Ok(())
}
