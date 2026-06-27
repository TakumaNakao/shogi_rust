use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Convert bench_failure_miner JSONL into mmto_tree_train feedback JSON")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long, required = true)]
    output: PathBuf,
    #[arg(long, default_value_t = 150.0)]
    min_timed_regret_cp: f32,
    #[arg(long, default_value_t = 0)]
    limit: usize,
    #[arg(long, default_value_t = true)]
    dedupe_sfen: bool,
}

#[derive(Debug, Deserialize)]
struct FailureSample {
    sfen: String,
    #[serde(default)]
    teacher_move: Option<String>,
    #[serde(default)]
    timed_move: Option<String>,
    #[serde(default)]
    teacher_score_cp: f32,
    #[serde(default)]
    timed_score_cp: f32,
    #[serde(default)]
    timed_regret_cp: f32,
    #[serde(default)]
    legal_moves: Option<usize>,
}

#[derive(Serialize)]
struct FeedbackReport {
    hard_positions: Vec<FeedbackRecord>,
}

#[derive(Serialize)]
struct FeedbackRecord {
    index: usize,
    sfen: String,
    teacher_best_move: Option<String>,
    baseline_move: Option<String>,
    candidate_move: Option<String>,
    baseline_regret: f32,
    candidate_regret: f32,
    regret_delta: f32,
    baseline_score: f32,
    candidate_score: f32,
    teacher_score: f32,
    legal_moves: Option<usize>,
}

fn create_writer(path: &Path) -> Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    Ok(BufWriter::new(file))
}

fn clean_move(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty() && value != "none")
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.min_timed_regret_cp.is_finite() || args.min_timed_regret_cp < 0.0 {
        return Err(anyhow!(
            "--min-timed-regret-cp must be finite and non-negative"
        ));
    }

    let mut records = Vec::new();
    let mut seen_sfens = HashSet::new();
    let mut read_lines = 0usize;
    let mut filtered_regret = 0usize;
    let mut filtered_moves = 0usize;
    let mut filtered_dedupe = 0usize;

    'files: for path in &args.input {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let reader = BufReader::new(file);
        for (line_index, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            read_lines += 1;
            let sample: FailureSample = serde_json::from_str(&line).map_err(|e| {
                anyhow!("{}:{} invalid json: {}", path.display(), line_index + 1, e)
            })?;
            if sample.timed_regret_cp < args.min_timed_regret_cp {
                filtered_regret += 1;
                continue;
            }
            let teacher_move = clean_move(sample.teacher_move);
            let timed_move = clean_move(sample.timed_move);
            let (Some(teacher_move), Some(timed_move)) = (teacher_move, timed_move) else {
                filtered_moves += 1;
                continue;
            };
            if teacher_move == timed_move {
                filtered_moves += 1;
                continue;
            }
            if args.dedupe_sfen && !seen_sfens.insert(sample.sfen.clone()) {
                filtered_dedupe += 1;
                continue;
            }

            records.push(FeedbackRecord {
                index: records.len(),
                sfen: sample.sfen,
                teacher_best_move: Some(teacher_move.clone()),
                baseline_move: Some(teacher_move),
                candidate_move: Some(timed_move),
                baseline_regret: 0.0,
                candidate_regret: sample.timed_regret_cp,
                regret_delta: sample.timed_regret_cp,
                baseline_score: sample.teacher_score_cp,
                candidate_score: sample.timed_score_cp,
                teacher_score: sample.teacher_score_cp,
                legal_moves: sample.legal_moves,
            });
            if args.limit > 0 && records.len() >= args.limit {
                break 'files;
            }
        }
    }

    let report = FeedbackReport {
        hard_positions: records,
    };
    let mut writer = create_writer(&args.output)?;
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writeln!(writer)?;

    println!("input lines: {}", read_lines);
    println!("feedback samples: {}", report.hard_positions.len());
    println!("filtered by regret: {}", filtered_regret);
    println!("filtered by moves: {}", filtered_moves);
    println!("filtered by dedupe: {}", filtered_dedupe);
    println!("output: {}", args.output.display());

    Ok(())
}
