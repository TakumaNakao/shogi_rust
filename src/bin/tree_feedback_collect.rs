use anyhow::{anyhow, Context, Result};
use clap::{ArgAction, Parser};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Convert mmto_tree_v1 root choices into mmto feedback JSON")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long, required = true)]
    output: PathBuf,
    #[arg(long)]
    guard_output: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    guard_percent: usize,
    #[arg(long, default_value_t = 0x7eed)]
    seed: u64,
    #[arg(long, default_value_t = 15.0)]
    min_candidate_regret_cp: f32,
    #[arg(long, default_value_t = 0.0)]
    max_candidate_regret_cp: f32,
    #[arg(long, default_value_t = 5.0)]
    min_regret_delta_cp: f32,
    #[arg(long, default_value_t = 0.0)]
    max_regret_delta_cp: f32,
    #[arg(long, default_value_t = 30.0)]
    max_good_regret_cp: f32,
    #[arg(long, default_value_t = 0)]
    limit: usize,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    dedupe_sfen: bool,
}

#[derive(Debug, Deserialize)]
struct TreeRecord {
    #[serde(default)]
    schema: Option<String>,
    #[serde(default)]
    version: Option<u8>,
    #[serde(default)]
    sfen: Option<String>,
    #[serde(default)]
    legal_moves: Option<usize>,
    #[serde(default)]
    candidates: Vec<CandidateRecord>,
}

#[derive(Clone, Debug, Deserialize)]
struct CandidateRecord {
    #[serde(rename = "move")]
    #[serde(default)]
    move_usi: String,
    #[serde(default)]
    selected_by_student: bool,
    #[serde(default)]
    teacher_score: Option<f32>,
    #[serde(default)]
    teacher_rank: Option<usize>,
    #[serde(default)]
    regret: Option<f32>,
    #[serde(default)]
    student_rank: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
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

fn finite(value: Option<f32>) -> Option<f32> {
    value.filter(|value| value.is_finite())
}

fn candidate_regret(candidate: &CandidateRecord, teacher_best_score: f32) -> Option<f32> {
    finite(candidate.regret).or_else(|| {
        finite(candidate.teacher_score).map(|score| (teacher_best_score - score).max(0.0))
    })
}

fn best_teacher_candidate(record: &TreeRecord) -> Option<(usize, f32, f32)> {
    record
        .candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| !candidate.move_usi.trim().is_empty())
        .filter_map(|(idx, candidate)| {
            let score = finite(candidate.teacher_score)?;
            let rank = candidate.teacher_rank.unwrap_or(usize::MAX);
            Some((idx, rank, score))
        })
        .min_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal))
        })
        .map(|(idx, _, score)| {
            let regret = finite(record.candidates[idx].regret).unwrap_or(0.0);
            (idx, score, regret)
        })
}

fn selected_student_candidate(record: &TreeRecord) -> Option<usize> {
    record
        .candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| candidate.selected_by_student)
        .filter(|(_, candidate)| !candidate.move_usi.trim().is_empty())
        .min_by_key(|(_, candidate)| candidate.student_rank.unwrap_or(usize::MAX))
        .map(|(idx, _)| idx)
}

fn record_harder_order(a: &FeedbackRecord, b: &FeedbackRecord) -> Ordering {
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
    record_harder_order(a, b) == Ordering::Less
}

fn feedback_from_tree(
    record: TreeRecord,
    min_candidate_regret_cp: f32,
    max_candidate_regret_cp: f32,
    min_regret_delta_cp: f32,
    max_regret_delta_cp: f32,
    max_good_regret_cp: f32,
) -> Option<FeedbackRecord> {
    if record
        .schema
        .as_deref()
        .is_some_and(|schema| schema != "mmto_tree_v1")
        || record.version == Some(0)
    {
        return None;
    }
    let sfen = record.sfen.as_deref()?.trim();
    if sfen.is_empty() {
        return None;
    }
    let (good_idx, teacher_best_score, good_regret) = best_teacher_candidate(&record)?;
    let bad_idx = selected_student_candidate(&record)?;
    if good_idx == bad_idx || good_regret > max_good_regret_cp {
        return None;
    }
    let good = &record.candidates[good_idx];
    let bad = &record.candidates[bad_idx];
    let bad_regret = candidate_regret(bad, teacher_best_score)?;
    let regret_delta = bad_regret - good_regret;
    if bad_regret < min_candidate_regret_cp || regret_delta < min_regret_delta_cp {
        return None;
    }
    if max_candidate_regret_cp > 0.0 && bad_regret > max_candidate_regret_cp {
        return None;
    }
    if max_regret_delta_cp > 0.0 && regret_delta > max_regret_delta_cp {
        return None;
    }

    Some(FeedbackRecord {
        index: 0,
        sfen: sfen.to_string(),
        teacher_best_move: Some(good.move_usi.trim().to_string()),
        baseline_move: Some(good.move_usi.trim().to_string()),
        candidate_move: Some(bad.move_usi.trim().to_string()),
        baseline_regret: good_regret,
        candidate_regret: bad_regret,
        regret_delta,
        baseline_score: finite(good.teacher_score).unwrap_or(teacher_best_score),
        candidate_score: finite(bad.teacher_score).unwrap_or(teacher_best_score - bad_regret),
        teacher_score: teacher_best_score,
        legal_moves: record.legal_moves,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.min_candidate_regret_cp.is_finite() || args.min_candidate_regret_cp < 0.0 {
        return Err(anyhow!(
            "--min-candidate-regret-cp must be finite and non-negative"
        ));
    }
    if !args.max_candidate_regret_cp.is_finite() || args.max_candidate_regret_cp < 0.0 {
        return Err(anyhow!(
            "--max-candidate-regret-cp must be finite and non-negative; use 0 to disable"
        ));
    }
    if !args.min_regret_delta_cp.is_finite() || args.min_regret_delta_cp < 0.0 {
        return Err(anyhow!(
            "--min-regret-delta-cp must be finite and non-negative"
        ));
    }
    if !args.max_regret_delta_cp.is_finite() || args.max_regret_delta_cp < 0.0 {
        return Err(anyhow!(
            "--max-regret-delta-cp must be finite and non-negative; use 0 to disable"
        ));
    }
    if !args.max_good_regret_cp.is_finite() || args.max_good_regret_cp < 0.0 {
        return Err(anyhow!(
            "--max-good-regret-cp must be finite and non-negative"
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
    let mut input_lines = 0usize;
    let mut invalid_lines = 0usize;
    let mut filtered = 0usize;
    let mut deduped = 0usize;

    'files: for path in &args.input {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let reader = BufReader::new(file);
        for (line_index, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            input_lines += 1;
            let tree: TreeRecord = match serde_json::from_str(&line) {
                Ok(tree) => tree,
                Err(_) => {
                    invalid_lines += 1;
                    continue;
                }
            };
            let Some(record) = feedback_from_tree(
                tree,
                args.min_candidate_regret_cp,
                args.max_candidate_regret_cp,
                args.min_regret_delta_cp,
                args.max_regret_delta_cp,
                args.max_good_regret_cp,
            ) else {
                filtered += 1;
                continue;
            };
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
            if args.limit > 0 {
                let current_len = if args.dedupe_sfen {
                    by_sfen.len()
                } else {
                    records.len()
                };
                if current_len >= args.limit {
                    break 'files;
                }
            }
            let _ = line_index;
        }
    }

    if args.dedupe_sfen {
        records.extend(by_sfen.into_values());
    }
    records.sort_by(record_harder_order);
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
    println!("input lines: {}", input_lines);
    println!("invalid lines: {}", invalid_lines);
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
