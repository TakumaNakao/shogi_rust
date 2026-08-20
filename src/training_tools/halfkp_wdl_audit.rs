use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use shogi_ai::evaluation::{EngineEvaluator, Evaluator};
use shogi_ai::training_data::sha256_file;
use shogi_ai::utils::position_from_sfen_or_usi;
use shogi_core::Color;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Fit and audit terminal WDL calibration for HalfKP")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    FitKappa {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        input_manifest: PathBuf,
        #[arg(long, default_value_t = 50.0)]
        kappa_min_cp: f32,
        #[arg(long, default_value_t = 5000.0)]
        kappa_max_cp: f32,
        #[arg(long, default_value_t = 0.01)]
        tolerance_cp: f32,
        #[arg(long)]
        output: PathBuf,
    },
    Evaluate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        input_manifest: PathBuf,
        #[arg(long)]
        kappa_manifest: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Deserialize)]
struct Record {
    schema: String,
    split: String,
    sfen: String,
    side_to_move: String,
    winner: Option<String>,
}

#[derive(Clone, Copy)]
struct Sample {
    raw_cp: f32,
    target: f32,
}

#[derive(Serialize, Deserialize)]
struct KappaManifest {
    schema: String,
    algorithm: String,
    model_sha256: String,
    input_manifest_sha256: String,
    input_path: PathBuf,
    records: usize,
    kappa_min_cp: f32,
    kappa_max_cp: f32,
    tolerance_cp: f32,
    kappa_cp: f32,
    brier: f64,
    boundary: bool,
}

#[derive(Serialize)]
struct EvaluationRecord {
    raw_cp: f32,
    target: f32,
    probability: f32,
    brier: f32,
}

pub fn run() -> Result<()> {
    match Args::parse().command {
        Command::FitKappa {
            model,
            input,
            input_manifest,
            kappa_min_cp,
            kappa_max_cp,
            tolerance_cp,
            output,
        } => fit_kappa(
            &model,
            &input,
            &input_manifest,
            kappa_min_cp,
            kappa_max_cp,
            tolerance_cp,
            &output,
        ),
        Command::Evaluate {
            model,
            input,
            input_manifest,
            kappa_manifest,
            output,
        } => evaluate(&model, &input, &input_manifest, &kappa_manifest, &output),
    }
}

fn load_samples(model: &Path, input: &Path) -> Result<Vec<Sample>> {
    let evaluator = EngineEvaluator::new(model, 0.0)?;
    let mut samples = Vec::new();
    for (line_no, line) in BufReader::new(File::open(input)?).lines().enumerate() {
        let record: Record = serde_json::from_str(&line?)
            .with_context(|| format!("parse {}:{}", input.display(), line_no + 1))?;
        if record.schema != "rank_value_dataset_v3_position" || record.split != "valid" {
            return Err(anyhow!("invalid WDL audit record at line {}", line_no + 1));
        }
        let winner = match record.winner.as_deref() {
            Some("black") => Color::Black,
            Some("white") => Color::White,
            _ => continue,
        };
        let position = position_from_sfen_or_usi(&record.sfen)
            .ok_or_else(|| anyhow!("invalid position at line {}", line_no + 1))?;
        let expected_side = if position.side_to_move() == Color::Black {
            "black"
        } else {
            "white"
        };
        if record.side_to_move != expected_side {
            return Err(anyhow!("side-to-move mismatch at line {}", line_no + 1));
        }
        let target = f32::from(position.side_to_move() == winner);
        let raw_cp = evaluator.evaluate(&position);
        if !raw_cp.is_finite() {
            return Err(anyhow!("non-finite evaluation at line {}", line_no + 1));
        }
        samples.push(Sample { raw_cp, target });
    }
    if samples.is_empty() {
        return Err(anyhow!("no labeled WDL samples"));
    }
    Ok(samples)
}

fn probability(raw_cp: f32, kappa_cp: f32) -> f32 {
    let x = (raw_cp / kappa_cp).clamp(-80.0, 80.0);
    1.0 / (1.0 + (-x).exp())
}

fn brier(samples: &[Sample], kappa_cp: f32) -> f64 {
    samples
        .iter()
        .map(|s| {
            let d = f64::from(probability(s.raw_cp, kappa_cp) - s.target);
            d * d
        })
        .sum::<f64>()
        / samples.len() as f64
}

fn fit_kappa(
    model: &Path,
    input: &Path,
    input_manifest: &Path,
    min: f32,
    max: f32,
    tolerance: f32,
    output: &Path,
) -> Result<()> {
    if !(min.is_finite()
        && max.is_finite()
        && tolerance.is_finite()
        && min > 0.0
        && max >= min
        && tolerance > 0.0)
    {
        return Err(anyhow!("invalid kappa range"));
    }
    let samples = load_samples(model, input)?;
    let mut lo = min;
    let mut hi = max;
    while hi - lo > tolerance {
        let step = (hi - lo) / 32.0;
        let mut best = lo;
        let mut best_loss = f64::INFINITY;
        for i in 0..=32 {
            let k = lo + step * i as f32;
            let loss = brier(&samples, k);
            if loss < best_loss {
                best_loss = loss;
                best = k;
            }
        }
        lo = (best - step).max(min);
        hi = (best + step).min(max);
    }
    let kappa = (lo + hi) * 0.5;
    let manifest = KappaManifest {
        schema: "halfkp_wdl_kappa_manifest_v1".into(),
        algorithm: "brier_grid_refine_v1".into(),
        model_sha256: sha256_file(model)?,
        input_manifest_sha256: sha256_file(input_manifest)?,
        input_path: input.to_path_buf(),
        records: samples.len(),
        kappa_min_cp: min,
        kappa_max_cp: max,
        tolerance_cp: tolerance,
        kappa_cp: kappa,
        brier: brier(&samples, kappa),
        boundary: (kappa - min).abs() <= tolerance || (kappa - max).abs() <= tolerance,
    };
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, serde_json::to_vec_pretty(&manifest)?)?;
    Ok(())
}

fn evaluate(
    model: &Path,
    input: &Path,
    input_manifest: &Path,
    kappa_manifest: &Path,
    output: &Path,
) -> Result<()> {
    let manifest: KappaManifest = serde_json::from_slice(&fs::read(kappa_manifest)?)?;
    if manifest.schema != "halfkp_wdl_kappa_manifest_v1"
        || manifest.model_sha256 != sha256_file(model)?
        || manifest.input_manifest_sha256 != sha256_file(input_manifest)?
    {
        return Err(anyhow!("kappa manifest provenance mismatch"));
    }
    let samples = load_samples(model, input)?;
    let mut writer = BufWriter::new(File::create(output)?);
    for sample in &samples {
        let p = probability(sample.raw_cp, manifest.kappa_cp);
        serde_json::to_writer(
            &mut writer,
            &EvaluationRecord {
                raw_cp: sample.raw_cp,
                target: sample.target,
                probability: p,
                brier: (p - sample.target).powi(2),
            },
        )?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::probability;

    #[test]
    fn probability_uses_cp_and_kappa_on_the_same_scale() {
        let expected = 1.0_f32 / (1.0 + (-1.0_f32).exp());
        let actual = probability(600.0, 600.0);
        assert!((actual - expected).abs() < 1.0e-6);
    }
}
