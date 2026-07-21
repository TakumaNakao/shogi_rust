use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Paired family comparison for scratch HalfKP prediction JSONL files")]
struct Args {
    #[arg(long)]
    baseline: PathBuf,
    #[arg(long)]
    candidate: PathBuf,
    #[arg(long)]
    json: Option<PathBuf>,
    #[arg(long, default_value_t = 10_000)]
    bootstrap_iterations: usize,
    #[arg(long, default_value_t = 20260820)]
    seed: u64,
}

#[derive(Deserialize)]
struct FamilyPrediction {
    family_hash: String,
    preferences: Vec<PreferencePrediction>,
}

#[derive(Deserialize)]
struct PreferencePrediction {
    correct: bool,
}

#[derive(Serialize)]
struct Report {
    families: usize,
    preferences: usize,
    baseline_family_accuracy: f64,
    candidate_family_accuracy: f64,
    paired_difference: f64,
    paired_difference_ci95_lower: f64,
    paired_difference_ci95_upper: f64,
    candidate_only_correct: usize,
    baseline_only_correct: usize,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    if args.bootstrap_iterations == 0 {
        return Err(anyhow!("bootstrap-iterations must be non-zero"));
    }
    let baseline = read_predictions(&args.baseline)?;
    let candidate = read_predictions(&args.candidate)?;
    if baseline.keys().ne(candidate.keys()) {
        return Err(anyhow!("prediction files contain different families"));
    }
    let mut differences = Vec::with_capacity(baseline.len());
    let mut baseline_sum = 0.0;
    let mut candidate_sum = 0.0;
    let mut preferences = 0;
    let mut candidate_only_correct = 0;
    let mut baseline_only_correct = 0;
    for (family, baseline_values) in &baseline {
        let candidate_values = &candidate[family];
        if baseline_values.len() != candidate_values.len() || baseline_values.is_empty() {
            return Err(anyhow!("prediction objective mismatch for family {family}"));
        }
        let baseline_mean = accuracy(baseline_values);
        let candidate_mean = accuracy(candidate_values);
        baseline_sum += baseline_mean;
        candidate_sum += candidate_mean;
        differences.push(candidate_mean - baseline_mean);
        preferences += baseline_values.len();
        for (&a, &b) in baseline_values.iter().zip(candidate_values) {
            candidate_only_correct += usize::from(!a && b);
            baseline_only_correct += usize::from(a && !b);
        }
    }
    if differences.is_empty() {
        return Err(anyhow!("prediction files contain no comparable families"));
    }
    let mut rng = ChaCha20Rng::seed_from_u64(args.seed);
    let mut estimates = Vec::with_capacity(args.bootstrap_iterations);
    for _ in 0..args.bootstrap_iterations {
        estimates.push(
            (0..differences.len())
                .map(|_| differences[rng.gen_range(0..differences.len())])
                .sum::<f64>()
                / differences.len() as f64,
        );
    }
    estimates.sort_by(f64::total_cmp);
    let last = estimates.len() - 1;
    let report = Report {
        families: differences.len(),
        preferences,
        baseline_family_accuracy: baseline_sum / differences.len() as f64,
        candidate_family_accuracy: candidate_sum / differences.len() as f64,
        paired_difference: differences.iter().sum::<f64>() / differences.len() as f64,
        paired_difference_ci95_lower: estimates[last * 25 / 1000],
        paired_difference_ci95_upper: estimates[last * 975 / 1000],
        candidate_only_correct,
        baseline_only_correct,
    };
    let bytes = serde_json::to_vec_pretty(&report)?;
    println!("{}", String::from_utf8_lossy(&bytes));
    if let Some(path) = args.json {
        if path.exists() {
            return Err(anyhow!("refusing to overwrite {}", path.display()));
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, [&bytes[..], b"\n"].concat())?;
    }
    Ok(())
}

fn read_predictions(path: &Path) -> Result<BTreeMap<String, Vec<bool>>> {
    let reader = BufReader::new(File::open(path)?);
    let mut result = BTreeMap::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        let prediction: FamilyPrediction = serde_json::from_str(&line)
            .with_context(|| format!("parse {} line {}", path.display(), index + 1))?;
        let values = prediction
            .preferences
            .into_iter()
            .map(|preference| preference.correct)
            .collect::<Vec<_>>();
        if result.insert(prediction.family_hash, values).is_some() {
            return Err(anyhow!("duplicate family in {}", path.display()));
        }
    }
    Ok(result)
}

fn accuracy(values: &[bool]) -> f64 {
    values.iter().filter(|&&value| value).count() as f64 / values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accuracy_counts_boolean_predictions() {
        assert_eq!(accuracy(&[true, false, true]), 2.0 / 3.0);
    }
}
