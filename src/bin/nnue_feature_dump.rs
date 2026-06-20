use anyhow::{anyhow, Result};
use clap::Parser;
use rayon::prelude::*;
use serde::Serialize;
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{extract_nnue_features, Evaluator, SparseModel};
use shogi_ai::utils::{format_move_usi, position_from_sfen_or_usi};
use shogi_lib::Position;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const HISTORY_CAPACITY: usize = 256;

#[derive(Parser, Debug)]
#[command(about = "Dump side-to-move normalized sparse NNUE prototype features as JSONL")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    weights: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    depth: u8,
    #[arg(long)]
    max_positions: Option<usize>,
    #[arg(long, default_value_t = 0)]
    jobs: usize,
}

#[derive(Serialize)]
struct NnueFeatureRecord {
    sfen: String,
    king_bucket: usize,
    features: Vec<usize>,
    material: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    static_eval: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    teacher_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    depth: Option<u8>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pv: Vec<String>,
}

struct DumpedRecord {
    index: usize,
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
) -> Result<Vec<(usize, String, Position)>> {
    let mut positions = Vec::new();
    for path in paths {
        let content = fs::read_to_string(path)
            .map_err(|e| anyhow!("failed to read {}: {}", path.display(), e))?;
        for line in content.lines() {
            if max_positions.is_some_and(|limit| positions.len() >= limit) {
                break;
            }
            let text = line.trim();
            if text.is_empty() {
                continue;
            }
            if let Some(position) = position_from_sfen_or_usi(text) {
                positions.push((positions.len(), position.to_sfen_owned(), position));
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

fn main() -> Result<()> {
    let args = Args::parse();
    if args.depth > 0 && args.weights.is_none() {
        return Err(anyhow!(
            "--weights is required when --depth is greater than zero"
        ));
    }
    if args.jobs > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build_global()
            .map_err(|e| anyhow!("failed to configure rayon thread pool: {e}"))?;
    }

    let positions = load_positions(&args.input, args.max_positions)?;
    let total_positions = positions.len();
    let model = args.weights.as_deref().map(load_model).transpose()?;

    let mut dumped = positions
        .into_par_iter()
        .filter_map(|(index, sfen, position)| {
            let nnue = extract_nnue_features(&position)?;
            let static_eval = model
                .as_ref()
                .map(|model| sanitize_score(model.predict_from_position(&position)));

            let (teacher_score, pv) = if args.depth > 0 {
                let model = model.as_ref()?;
                let evaluator = SharedModelEvaluator { model };
                let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
                let mut search_position = position.clone();
                ai.set_emit_info(false);
                ai.sennichite_detector.record_position(&search_position);
                let (score, pv) = ai.alpha_beta_search(
                    &mut search_position,
                    args.depth,
                    -f32::INFINITY,
                    f32::INFINITY,
                )?;
                (
                    Some(sanitize_score(score)),
                    pv.into_iter().map(format_move_usi).collect(),
                )
            } else {
                (None, Vec::new())
            };

            let record = NnueFeatureRecord {
                sfen,
                king_bucket: nnue.king_bucket,
                features: nnue.features,
                material: nnue.material,
                static_eval,
                teacher_score,
                depth: teacher_score.map(|_| args.depth),
                pv,
            };
            serde_json::to_string(&record)
                .ok()
                .map(|line| DumpedRecord { index, line })
        })
        .collect::<Vec<_>>();
    dumped.sort_unstable_by_key(|record| record.index);

    let mut writer = create_writer(&args.output)?;
    for record in &dumped {
        writeln!(writer, "{}", record.line)?;
    }
    writer.flush()?;

    println!("records: {}", dumped.len());
    println!(
        "skipped positions: {}",
        total_positions.saturating_sub(dumped.len())
    );
    Ok(())
}
