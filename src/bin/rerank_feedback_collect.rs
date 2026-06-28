use anyhow::{anyhow, Context, Result};
use clap::{ArgAction, Parser};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Collect mmto_rerank_gate hard positions into one feedback JSON")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long, required = true)]
    output: PathBuf,
    #[arg(long)]
    guard_output: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    guard_percent: usize,
    #[arg(long, default_value_t = 0x5eed)]
    seed: u64,
    #[arg(long, default_value_t = 0.0)]
    min_regret_delta_cp: f32,
    #[arg(long, default_value_t = 0.0)]
    min_candidate_regret_cp: f32,
    #[arg(long, default_value_t = 0)]
    limit: usize,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    dedupe_sfen: bool,
    #[arg(long, default_value_t = false)]
    include_worst_delta: bool,
    #[arg(long, default_value_t = false)]
    include_worst_candidate: bool,
}

#[derive(Debug, Deserialize)]
struct GateReport {
    #[serde(default)]
    hard_positions: Vec<FeedbackRecord>,
    #[serde(default)]
    worst_delta: Vec<FeedbackRecord>,
    #[serde(default)]
    worst_candidate: Vec<FeedbackRecord>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct FeedbackRecord {
    #[serde(default)]
    index: usize,
    sfen: String,
    #[serde(default)]
    teacher_best_move: Option<String>,
    #[serde(default)]
    baseline_move: Option<String>,
    #[serde(default)]
    candidate_move: Option<String>,
    #[serde(default)]
    baseline_regret: f32,
    #[serde(default)]
    candidate_regret: f32,
    #[serde(default)]
    regret_delta: f32,
    #[serde(default)]
    baseline_score: f32,
    #[serde(default)]
    candidate_score: f32,
    #[serde(default)]
    teacher_score: f32,
    #[serde(default)]
    legal_moves: Option<usize>,
}

#[derive(Serialize)]
struct FeedbackReport {
    hard_positions: Vec<FeedbackRecord>,
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

fn write_feedback_report(path: &Path, mut records: Vec<FeedbackRecord>) -> Result<()> {
    for (index, record) in records.iter_mut().enumerate() {
        record.index = index;
    }
    let report = FeedbackReport {
        hard_positions: records,
    };
    let mut writer = create_writer(path)?;
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writeln!(writer)?;
    Ok(())
}

fn clean_move(value: &mut Option<String>) {
    if let Some(text) = value {
        let trimmed = text.trim();
        if trimmed.is_empty() || trimmed == "none" {
            *value = None;
        } else if trimmed.len() != text.len() {
            *text = trimmed.to_string();
        }
    }
}

fn sanitize_record(mut record: FeedbackRecord) -> Option<FeedbackRecord> {
    record.sfen = record.sfen.trim().to_string();
    clean_move(&mut record.teacher_best_move);
    clean_move(&mut record.baseline_move);
    clean_move(&mut record.candidate_move);
    if record.sfen.is_empty() {
        return None;
    }
    if record.candidate_move.is_none() {
        return None;
    }
    if record.baseline_move.is_none() && record.teacher_best_move.is_none() {
        return None;
    }
    if !record.baseline_regret.is_finite()
        || !record.candidate_regret.is_finite()
        || !record.regret_delta.is_finite()
        || !record.baseline_score.is_finite()
        || !record.candidate_score.is_finite()
        || !record.teacher_score.is_finite()
    {
        return None;
    }
    if record
        .baseline_move
        .as_ref()
        .or(record.teacher_best_move.as_ref())
        == record.candidate_move.as_ref()
    {
        return None;
    }
    Some(record)
}

fn harder_order(a: &FeedbackRecord, b: &FeedbackRecord) -> Ordering {
    b.regret_delta
        .partial_cmp(&a.regret_delta)
        .unwrap_or(Ordering::Equal)
        .then_with(|| {
            b.candidate_regret
                .partial_cmp(&a.candidate_regret)
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| a.sfen.cmp(&b.sfen))
}

fn record_is_harder(a: &FeedbackRecord, b: &FeedbackRecord) -> bool {
    harder_order(a, b) == Ordering::Less
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.min_regret_delta_cp.is_finite() || args.min_regret_delta_cp < 0.0 {
        return Err(anyhow!(
            "--min-regret-delta-cp must be finite and non-negative"
        ));
    }
    if !args.min_candidate_regret_cp.is_finite() || args.min_candidate_regret_cp < 0.0 {
        return Err(anyhow!(
            "--min-candidate-regret-cp must be finite and non-negative"
        ));
    }
    if args.guard_output.is_some() && args.guard_percent == 0 {
        return Err(anyhow!(
            "--guard-percent must be greater than 0 when --guard-output is set"
        ));
    }
    if args.guard_percent >= 100 {
        return Err(anyhow!("--guard-percent must be less than 100"));
    }

    let mut records = Vec::new();
    let mut by_sfen: HashMap<String, FeedbackRecord> = HashMap::new();
    let mut read_records = 0usize;
    let mut filtered = 0usize;
    let mut deduped = 0usize;

    for path in &args.input {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let reader = BufReader::new(file);
        let report: GateReport = serde_json::from_reader(reader)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let mut groups = vec![report.hard_positions];
        if args.include_worst_delta {
            groups.push(report.worst_delta);
        }
        if args.include_worst_candidate {
            groups.push(report.worst_candidate);
        }
        for group in groups {
            for record in group {
                read_records += 1;
                let Some(record) = sanitize_record(record) else {
                    filtered += 1;
                    continue;
                };
                if record.regret_delta < args.min_regret_delta_cp
                    || record.candidate_regret < args.min_candidate_regret_cp
                {
                    filtered += 1;
                    continue;
                }
                if args.dedupe_sfen {
                    match by_sfen.get_mut(&record.sfen) {
                        Some(existing) => {
                            if record_is_harder(&record, existing) {
                                *existing = record;
                            }
                            deduped += 1;
                        }
                        None => {
                            by_sfen.insert(record.sfen.clone(), record);
                        }
                    }
                } else {
                    records.push(record);
                }
            }
        }
    }

    if args.dedupe_sfen {
        records.extend(by_sfen.into_values());
    }
    records.sort_by(harder_order);
    if args.limit > 0 && records.len() > args.limit {
        records.truncate(args.limit);
    }
    let total_samples = records.len();
    let guard_records = if let Some(path) = args.guard_output.as_ref() {
        let mut indices = (0..records.len()).collect::<Vec<_>>();
        indices.shuffle(&mut ChaCha8Rng::seed_from_u64(args.seed));
        let guard_count = ((records.len() * args.guard_percent) + 50) / 100;
        let guard_indices = indices
            .into_iter()
            .take(guard_count)
            .collect::<HashSet<usize>>();
        let mut train = Vec::new();
        let mut guard = Vec::new();
        for (idx, record) in records.into_iter().enumerate() {
            if guard_indices.contains(&idx) {
                guard.push(record);
            } else {
                train.push(record);
            }
        }
        write_feedback_report(&args.output, train.clone())?;
        write_feedback_report(path, guard.clone())?;
        records = train;
        guard
    } else {
        write_feedback_report(&args.output, records.clone())?;
        Vec::new()
    };

    println!("input files: {}", args.input.len());
    println!("read records: {}", read_records);
    println!("feedback samples: {}", total_samples);
    println!("train samples: {}", records.len());
    println!("guard samples: {}", guard_records.len());
    println!("filtered: {}", filtered);
    println!("deduped: {}", deduped);
    println!("output: {}", args.output.display());
    if let Some(path) = args.guard_output.as_ref() {
        println!("guard output: {}", path.display());
    }
    if total_samples == 0 {
        return Err(anyhow!("no feedback samples collected"));
    }
    Ok(())
}
