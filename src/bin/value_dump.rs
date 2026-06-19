use anyhow::{anyhow, Result};
use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Serialize;
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{Evaluator, SparseModel};
use shogi_ai::utils::{format_move_usi, position_from_sfen_or_usi};
use shogi_lib::Position;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const HISTORY_CAPACITY: usize = 256;

#[derive(Parser, Debug)]
#[command(about = "Dump searched root value targets for KPP value distillation")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long)]
    train_output: PathBuf,
    #[arg(long)]
    valid_output: PathBuf,
    #[arg(long, default_value_t = 5)]
    depth: u8,
    #[arg(long, default_value_t = 9801)]
    seed: u64,
    #[arg(long, default_value_t = 10)]
    valid_percent: u8,
    #[arg(long)]
    max_positions: Option<usize>,
    #[arg(long, default_value_t = 0)]
    jobs: usize,
}

#[derive(Serialize)]
struct ValueRecord {
    sfen: String,
    teacher_score: f32,
    depth: u8,
    legal_moves: usize,
    pv: Vec<String>,
}

struct DumpedRecord {
    index: usize,
    line: String,
    is_valid: bool,
}

struct SharedModelEvaluator<'a> {
    model: &'a SparseModel,
}

impl Evaluator for SharedModelEvaluator<'_> {
    fn evaluate(&self, position: &Position) -> f32 {
        self.model.predict_from_position(position)
    }
}

fn load_positions(paths: &[PathBuf]) -> Result<Vec<Position>> {
    let mut positions = Vec::new();
    for path in paths {
        let content = fs::read_to_string(path)
            .map_err(|e| anyhow!("failed to read {}: {}", path.display(), e))?;
        for line in content.lines() {
            if let Some(position) = position_from_sfen_or_usi(line) {
                positions.push(position);
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

fn sanitize_teacher_score(score: f32) -> f32 {
    const LIMIT: f32 = 100_000.0;
    if score == f32::INFINITY {
        LIMIT
    } else if score == -f32::INFINITY {
        -LIMIT
    } else {
        score.clamp(-LIMIT, LIMIT)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.depth == 0 {
        return Err(anyhow!("--depth must be greater than zero"));
    }
    if args.valid_percent > 90 {
        return Err(anyhow!("--valid-percent must be 0..=90"));
    }
    if args.jobs > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build_global()
            .map_err(|e| anyhow!("failed to configure rayon thread pool: {e}"))?;
    }

    let mut positions = load_positions(&args.input)?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    positions.shuffle(&mut rng);
    if let Some(max_positions) = args.max_positions {
        positions.truncate(max_positions);
    }
    let total_positions = positions.len();

    let mut model = SparseModel::new(0.0, 0.0);
    model.load(&args.weights)?;

    let valid_stride = if args.valid_percent == 0 {
        usize::MAX
    } else {
        (100 / args.valid_percent as usize).max(1)
    };

    let mut dumped = positions
        .into_par_iter()
        .enumerate()
        .filter_map(|(idx, mut position)| {
            let legal_moves = position.legal_moves();
            if legal_moves.is_empty() {
                return None;
            }

            let evaluator = SharedModelEvaluator { model: &model };
            let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
            ai.set_emit_info(false);
            ai.sennichite_detector.record_position(&position);
            let (score, pv) =
                ai.alpha_beta_search(&mut position, args.depth, -f32::INFINITY, f32::INFINITY)?;
            if !score.is_finite() && score != f32::INFINITY && score != -f32::INFINITY {
                return None;
            }

            let record = ValueRecord {
                sfen: position.to_sfen_owned(),
                teacher_score: sanitize_teacher_score(score),
                depth: args.depth,
                legal_moves: legal_moves.len(),
                pv: pv.into_iter().map(format_move_usi).collect(),
            };
            let line = serde_json::to_string(&record).ok()?;
            Some(DumpedRecord {
                index: idx,
                line,
                is_valid: valid_stride != usize::MAX && idx % valid_stride == 0,
            })
        })
        .collect::<Vec<_>>();
    dumped.sort_unstable_by_key(|record| record.index);

    let mut train_writer = create_writer(&args.train_output)?;
    let mut valid_writer = create_writer(&args.valid_output)?;
    let mut train_count = 0usize;
    let mut valid_count = 0usize;
    for record in &dumped {
        if record.is_valid {
            writeln!(valid_writer, "{}", record.line)?;
            valid_count += 1;
        } else {
            writeln!(train_writer, "{}", record.line)?;
            train_count += 1;
        }
    }

    train_writer.flush()?;
    valid_writer.flush()?;
    println!("train records: {train_count}");
    println!("valid records: {valid_count}");
    println!(
        "skipped positions: {}",
        total_positions.saturating_sub(train_count + valid_count)
    );
    Ok(())
}
