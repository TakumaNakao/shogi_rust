use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Summarize mmto_tree_v1 JSONL rank/value datasets without loading them fully")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long)]
    json_output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct TreeRecord {
    #[serde(default)]
    legal_moves: Option<usize>,
    #[serde(default)]
    candidates: Vec<CandidateRecord>,
}

#[derive(Debug, Deserialize)]
struct CandidateRecord {
    #[serde(default)]
    teacher_score: Option<f32>,
    #[serde(default)]
    regret: Option<f32>,
    #[serde(default)]
    selected_by_student: Option<bool>,
    #[serde(default)]
    searched_by_teacher: Option<bool>,
}

#[derive(Debug, Default, Serialize)]
struct Stats {
    files: usize,
    lines: usize,
    invalid_lines: usize,
    samples: usize,
    total_candidates: usize,
    searched_candidates: usize,
    total_legal_moves: usize,
    samples_with_selected: usize,
    selected_regret_mean: f32,
    selected_regret_p50: f32,
    selected_regret_p90: f32,
    selected_regret_p95: f32,
    selected_regret_p99: f32,
    selected_regret_max: f32,
    bad50: f32,
    bad100: f32,
    bad200: f32,
    bad300: f32,
    score_span_mean: f32,
    candidate_count_mean: f32,
    legal_moves_mean: f32,
}

#[derive(Default)]
struct Accumulator {
    files: usize,
    lines: usize,
    invalid_lines: usize,
    samples: usize,
    total_candidates: usize,
    searched_candidates: usize,
    total_legal_moves: usize,
    selected_regrets: Vec<f32>,
    score_span_sum: f32,
}

fn percentile(sorted: &[f32], pct: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pct = pct.clamp(0.0, 1.0);
    let index = ((sorted.len() - 1) as f32 * pct).round() as usize;
    sorted[index]
}

impl Accumulator {
    fn add_record(&mut self, record: TreeRecord) {
        self.samples += 1;
        self.total_candidates += record.candidates.len();
        self.total_legal_moves += record.legal_moves.unwrap_or(0);
        self.searched_candidates += record
            .candidates
            .iter()
            .filter(|candidate| candidate.searched_by_teacher.unwrap_or(false))
            .count();

        if let Some(selected) = record
            .candidates
            .iter()
            .find(|candidate| candidate.selected_by_student.unwrap_or(false))
        {
            self.selected_regrets.push(selected.regret.unwrap_or(0.0));
        }

        let mut min_score = f32::INFINITY;
        let mut max_score = f32::NEG_INFINITY;
        for score in record
            .candidates
            .iter()
            .filter_map(|candidate| candidate.teacher_score)
            .filter(|score| score.is_finite())
        {
            min_score = min_score.min(score);
            max_score = max_score.max(score);
        }
        if min_score.is_finite() && max_score.is_finite() {
            self.score_span_sum += max_score - min_score;
        }
    }

    fn finish(mut self) -> Stats {
        self.selected_regrets.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
        let selected_count = self.selected_regrets.len();
        let selected_sum: f32 = self.selected_regrets.iter().sum();
        let bad_ratio = |limit: f32| -> f32 {
            if selected_count == 0 {
                0.0
            } else {
                self.selected_regrets
                    .iter()
                    .filter(|&&regret| regret >= limit)
                    .count() as f32
                    / selected_count as f32
            }
        };
        Stats {
            files: self.files,
            lines: self.lines,
            invalid_lines: self.invalid_lines,
            samples: self.samples,
            total_candidates: self.total_candidates,
            searched_candidates: self.searched_candidates,
            total_legal_moves: self.total_legal_moves,
            samples_with_selected: selected_count,
            selected_regret_mean: if selected_count == 0 {
                0.0
            } else {
                selected_sum / selected_count as f32
            },
            selected_regret_p50: percentile(&self.selected_regrets, 0.50),
            selected_regret_p90: percentile(&self.selected_regrets, 0.90),
            selected_regret_p95: percentile(&self.selected_regrets, 0.95),
            selected_regret_p99: percentile(&self.selected_regrets, 0.99),
            selected_regret_max: self.selected_regrets.last().copied().unwrap_or(0.0),
            bad50: bad_ratio(50.0),
            bad100: bad_ratio(100.0),
            bad200: bad_ratio(200.0),
            bad300: bad_ratio(300.0),
            score_span_mean: if self.samples == 0 {
                0.0
            } else {
                self.score_span_sum / self.samples as f32
            },
            candidate_count_mean: if self.samples == 0 {
                0.0
            } else {
                self.total_candidates as f32 / self.samples as f32
            },
            legal_moves_mean: if self.samples == 0 {
                0.0
            } else {
                self.total_legal_moves as f32 / self.samples as f32
            },
        }
    }
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    let mut acc = Accumulator::default();

    for path in &args.input {
        acc.files += 1;
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        for line in BufReader::new(file).lines() {
            acc.lines += 1;
            let line = line.with_context(|| format!("failed to read {}", path.display()))?;
            match serde_json::from_str::<TreeRecord>(&line) {
                Ok(record) => acc.add_record(record),
                Err(_) => acc.invalid_lines += 1,
            }
        }
    }

    let stats = acc.finish();
    println!("files={}", stats.files);
    println!("lines={} invalid={}", stats.lines, stats.invalid_lines);
    println!(
        "samples={} candidates={} searched_candidates={}",
        stats.samples, stats.total_candidates, stats.searched_candidates
    );
    println!(
        "means: candidates={:.2} legal_moves={:.2} score_span={:.2}",
        stats.candidate_count_mean, stats.legal_moves_mean, stats.score_span_mean
    );
    println!(
        "selected_regret: mean={:.2} p50={:.2} p90={:.2} p95={:.2} p99={:.2} max={:.2}",
        stats.selected_regret_mean,
        stats.selected_regret_p50,
        stats.selected_regret_p90,
        stats.selected_regret_p95,
        stats.selected_regret_p99,
        stats.selected_regret_max
    );
    println!(
        "bad ratios: bad50={:.4} bad100={:.4} bad200={:.4} bad300={:.4}",
        stats.bad50, stats.bad100, stats.bad200, stats.bad300
    );

    if let Some(path) = args.json_output {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file =
            File::create(&path).with_context(|| format!("failed to create {}", path.display()))?;
        serde_json::to_writer_pretty(BufWriter::new(file), &stats)?;
        println!("json-output: {}", path.display());
    }

    Ok(())
}
