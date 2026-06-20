use anyhow::{anyhow, Result};
use clap::Parser;
use rayon::prelude::*;
use serde::Serialize;
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{extract_nnue_features, Evaluator, SparseModel};
use shogi_ai::utils::{format_move_usi, position_from_sfen_or_usi};
use shogi_core::Move;
use shogi_lib::Position;
use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const HISTORY_CAPACITY: usize = 256;

#[derive(Parser, Debug)]
#[command(about = "Dump NNUE child-position ranking records from searched legal root moves")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 4)]
    depth: u8,
    #[arg(long)]
    max_positions: Option<usize>,
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    #[arg(long, default_value_t = 0)]
    jobs: usize,
}

#[derive(Serialize)]
struct RankRecord {
    root_index: usize,
    root_sfen: String,
    child_sfen: String,
    move_usi: String,
    rank: usize,
    legal_moves: usize,
    teacher_score: f32,
    best_score: f32,
    regret: f32,
    depth: u8,
    king_bucket: usize,
    features: Vec<usize>,
    material: f32,
}

struct ScoredChild {
    child_sfen: String,
    mv: Move,
    teacher_score: f32,
    king_bucket: usize,
    features: Vec<usize>,
    material: f32,
}

struct DumpedRecord {
    root_index: usize,
    rank: usize,
    line: String,
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

fn load_positions(
    paths: &[PathBuf],
    max_positions: Option<usize>,
) -> Result<Vec<(usize, Position)>> {
    let mut positions = Vec::new();
    for path in paths {
        let content = fs::read_to_string(path)
            .map_err(|e| anyhow!("failed to read {}: {}", path.display(), e))?;
        for line in content.lines() {
            if max_positions.is_some_and(|limit| positions.len() >= limit) {
                break;
            }
            if let Some(position) = position_from_sfen_or_usi(line.trim()) {
                positions.push((positions.len(), position));
            }
        }
    }
    if positions.is_empty() {
        return Err(anyhow!("no valid positions loaded"));
    }
    Ok(positions)
}

fn create_writer(path: &Path) -> Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(BufWriter::new(File::create(path)?))
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

fn searched_child_score(model: &SparseModel, child: &mut Position, depth: u8) -> Option<f32> {
    let evaluator = SharedModelEvaluator { model };
    let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
    ai.set_emit_info(false);
    ai.sennichite_detector.record_position(child);
    let child_depth = depth.saturating_sub(1);
    ai.alpha_beta_search(child, child_depth, -f32::INFINITY, f32::INFINITY)
        .map(|(score, _)| sanitize_score(-score))
}

fn dump_root(
    model: &SparseModel,
    root_index: usize,
    position: Position,
    depth: u8,
    top_k: usize,
) -> Vec<DumpedRecord> {
    let root_sfen = position.to_sfen_owned();
    let legal_moves = position.legal_moves();
    if legal_moves.is_empty() {
        return Vec::new();
    }

    let mut scored = legal_moves
        .iter()
        .filter_map(|&mv| {
            let mut child = position.clone();
            child.do_move(mv);
            let teacher_score = searched_child_score(model, &mut child, depth)?;
            let nnue = extract_nnue_features(&child)?;
            Some(ScoredChild {
                child_sfen: child.to_sfen_owned(),
                mv,
                teacher_score,
                king_bucket: nnue.king_bucket,
                features: nnue.features,
                material: nnue.material,
            })
        })
        .collect::<Vec<_>>();

    scored.sort_by(|a, b| {
        b.teacher_score
            .partial_cmp(&a.teacher_score)
            .unwrap_or(Ordering::Equal)
    });
    if top_k > 0 && scored.len() > top_k {
        scored.truncate(top_k);
    }
    let best_score = scored
        .first()
        .map(|record| record.teacher_score)
        .unwrap_or(0.0);

    scored
        .into_iter()
        .enumerate()
        .filter_map(|(rank_index, child)| {
            let rank = rank_index + 1;
            let record = RankRecord {
                root_index,
                root_sfen: root_sfen.clone(),
                child_sfen: child.child_sfen,
                move_usi: format_move_usi(child.mv),
                rank,
                legal_moves: legal_moves.len(),
                teacher_score: child.teacher_score,
                best_score,
                regret: (best_score - child.teacher_score).max(0.0),
                depth,
                king_bucket: child.king_bucket,
                features: child.features,
                material: child.material,
            };
            serde_json::to_string(&record)
                .ok()
                .map(|line| DumpedRecord {
                    root_index,
                    rank,
                    line,
                })
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.depth == 0 {
        return Err(anyhow!("--depth must be greater than zero"));
    }
    if args.jobs > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build_global()
            .map_err(|e| anyhow!("failed to configure rayon thread pool: {e}"))?;
    }

    let model = load_model(&args.weights)?;
    let positions = load_positions(&args.input, args.max_positions)?;
    let total_roots = positions.len();

    let mut dumped = positions
        .into_par_iter()
        .flat_map(|(root_index, position)| {
            dump_root(&model, root_index, position, args.depth, args.top_k)
        })
        .collect::<Vec<_>>();
    dumped.sort_unstable_by_key(|record| (record.root_index, record.rank));

    let mut writer = create_writer(&args.output)?;
    for record in &dumped {
        writeln!(writer, "{}", record.line)?;
    }
    writer.flush()?;

    println!("roots: {total_roots}");
    println!("records: {}", dumped.len());
    println!(
        "avg records/root: {:.2}",
        dumped.len() as f64 / total_roots.max(1) as f64
    );
    Ok(())
}
