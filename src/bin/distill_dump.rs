use anyhow::{anyhow, Result};
use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
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
#[command(about = "Dump fixed-position teacher bestmove data for policy distillation")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long)]
    train_output: PathBuf,
    #[arg(long)]
    valid_output: PathBuf,
    #[arg(long, default_value_t = 6)]
    depth: u8,
    #[arg(long)]
    time_limit_ms: Option<u64>,
    #[arg(long, default_value_t = 9601)]
    seed: u64,
    #[arg(long, default_value_t = 10)]
    valid_percent: u8,
    #[arg(long)]
    max_positions: Option<usize>,
}

#[derive(Serialize)]
struct DistillRecord {
    sfen: String,
    teacher_move: String,
    depth: u8,
    legal_moves: usize,
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

fn main() -> Result<()> {
    let args = Args::parse();
    if args.depth == 0 {
        return Err(anyhow!("--depth must be greater than zero"));
    }
    if args.valid_percent > 90 {
        return Err(anyhow!("--valid-percent must be 0..=90"));
    }

    let mut positions = load_positions(&args.input)?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    positions.shuffle(&mut rng);
    if let Some(max_positions) = args.max_positions {
        positions.truncate(max_positions);
    }
    let mut model = SparseModel::new(0.0, 0.0);
    model.load(&args.weights)?;

    let mut train_writer = create_writer(&args.train_output)?;
    let mut valid_writer = create_writer(&args.valid_output)?;
    let valid_stride = if args.valid_percent == 0 {
        usize::MAX
    } else {
        (100 / args.valid_percent as usize).max(1)
    };

    let mut train_count = 0usize;
    let mut valid_count = 0usize;
    let mut skipped = 0usize;

    for (idx, mut position) in positions.into_iter().enumerate() {
        let legal_moves = position.legal_moves();
        if legal_moves.is_empty() {
            skipped += 1;
            continue;
        }

        let evaluator = SharedModelEvaluator { model: &model };
        let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
        ai.set_emit_info(false);
        ai.sennichite_detector.record_position(&position);
        let Some(best_move) = ai.find_best_move(&mut position, args.depth, args.time_limit_ms)
        else {
            skipped += 1;
            continue;
        };
        if !legal_moves.contains(&best_move) {
            skipped += 1;
            continue;
        }

        let record = DistillRecord {
            sfen: position.to_sfen_owned(),
            teacher_move: format_move_usi(best_move),
            depth: args.depth,
            legal_moves: legal_moves.len(),
        };
        let line = serde_json::to_string(&record)?;
        if valid_stride != usize::MAX && idx % valid_stride == 0 {
            writeln!(valid_writer, "{line}")?;
            valid_count += 1;
        } else {
            writeln!(train_writer, "{line}")?;
            train_count += 1;
        }
    }

    train_writer.flush()?;
    valid_writer.flush()?;
    println!("train records: {train_count}");
    println!("valid records: {valid_count}");
    println!("skipped positions: {skipped}");
    Ok(())
}
