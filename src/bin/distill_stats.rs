use anyhow::{anyhow, Result};
use clap::Parser;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Summarize teacher-score gaps in distillation JSONL files")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "5,10,20,40,80,1000,10000"
    )]
    buckets: Vec<f32>,
}

#[derive(Deserialize)]
struct DistillRecord {
    #[serde(default)]
    teacher_scores: Vec<TeacherScoreRecord>,
}

#[derive(Deserialize)]
struct TeacherScoreRecord {
    score: f32,
}

#[derive(Default)]
struct Stats {
    records: usize,
    with_teacher_scores: usize,
    without_teacher_scores: usize,
    with_gap: usize,
    one_score: usize,
    non_finite_scores: usize,
    gaps: Vec<f32>,
}

fn load_stats(path: &Path, stats: &mut Stats) -> Result<()> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: DistillRecord = serde_json::from_str(&line)
            .map_err(|e| anyhow!("{}:{} invalid json: {}", path.display(), line_index + 1, e))?;
        stats.records += 1;
        if record.teacher_scores.is_empty() {
            stats.without_teacher_scores += 1;
            continue;
        }
        stats.with_teacher_scores += 1;

        let mut scores = record
            .teacher_scores
            .iter()
            .filter_map(|teacher| {
                if teacher.score.is_finite() {
                    Some(teacher.score)
                } else {
                    stats.non_finite_scores += 1;
                    None
                }
            })
            .collect::<Vec<_>>();
        scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        match scores.as_slice() {
            [best, second, ..] => {
                stats.gaps.push(*best - *second);
                stats.with_gap += 1;
            }
            [_] => stats.one_score += 1,
            [] => {}
        }
    }
    Ok(())
}

fn percentile(sorted: &[f32], pct: f32) -> Option<f32> {
    if sorted.is_empty() {
        return None;
    }
    let pct = pct.clamp(0.0, 100.0);
    let idx = ((sorted.len() - 1) as f32 * pct / 100.0).round() as usize;
    sorted.get(idx).copied()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut stats = Stats::default();
    for path in &args.input {
        load_stats(path, &mut stats)?;
    }

    println!("records: {}", stats.records);
    println!("with_teacher_scores: {}", stats.with_teacher_scores);
    println!("without_teacher_scores: {}", stats.without_teacher_scores);
    println!("with_gap: {}", stats.with_gap);
    println!("one_score: {}", stats.one_score);
    println!("non_finite_scores: {}", stats.non_finite_scores);

    if stats.gaps.is_empty() {
        return Ok(());
    }

    stats
        .gaps
        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let sum = stats.gaps.iter().sum::<f32>();
    println!("gap_avg: {:.3}", sum / stats.gaps.len() as f32);
    for pct in [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 100.0] {
        if let Some(value) = percentile(&stats.gaps, pct) {
            println!("gap_p{:02}: {:.3}", pct as u8, value);
        }
    }

    let mut buckets = args.buckets;
    buckets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    buckets.dedup_by(|a, b| (*a - *b).abs() < f32::EPSILON);
    for bucket in buckets {
        if !bucket.is_finite() || bucket < 0.0 {
            return Err(anyhow!(
                "bucket must be a non-negative finite number: {bucket}"
            ));
        }
        let count = stats.gaps.iter().filter(|gap| **gap >= bucket).count();
        let pct = count as f32 * 100.0 / stats.gaps.len() as f32;
        println!("gap_ge_{}: {count} ({pct:.2}%)", format_bucket(bucket));
    }

    Ok(())
}

fn format_bucket(bucket: f32) -> String {
    if (bucket.fract()).abs() < f32::EPSILON {
        format!("{bucket:.0}")
    } else {
        format!("{bucket}")
    }
}
