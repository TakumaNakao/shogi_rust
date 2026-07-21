use anyhow::{anyhow, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use shogi_ai::evaluation::HalfKpFlatModel;
use shogi_ai::halfkp_training::{
    CurriculumReader, PackedHalfKpPosition, PreferenceKind, ProofStatus,
};
use shogi_ai::training_data::sha256_file;
use shogi_core::Color;
use std::fs;
use std::path::PathBuf;

const LINEAGE_ID: &str = "halfkp64-scratch-v1";

#[derive(Parser, Debug)]
#[command(about = "Evaluate one scratch-lineage HalfKP model on HKCL holdouts")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    model_manifest: PathBuf,
    #[arg(long, required = true)]
    dataset: Vec<PathBuf>,
    #[arg(long, default_value_t = 600.0)]
    kappa_cp: f32,
    #[arg(long)]
    json: Option<PathBuf>,
}

#[derive(Deserialize)]
struct ModelManifest {
    lineage_id: String,
    model: ModelArtifact,
}

#[derive(Deserialize)]
struct ModelArtifact {
    sha256: String,
}

#[derive(Default, Serialize)]
struct Report {
    model_sha256: String,
    records: usize,
    value_objectives: usize,
    value_bce: f64,
    pairs: usize,
    pair_correct: usize,
    pair_ties: usize,
    symmetry_pairs: usize,
    symmetry_rmse_cp: f64,
    mate1_records: usize,
    mate3_records: usize,
    mate1_pairs: usize,
    mate1_pair_correct: usize,
    mate3_pairs: usize,
    mate3_pair_correct: usize,
    root_score_mean_cp: f64,
    root_score_stddev_cp: f64,
    root_score_abs_p99_cp: f32,
    output_bias_cp: f32,
    material_coefficient: f32,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    if args.kappa_cp <= 0.0 {
        return Err(anyhow!("kappa-cp must be positive"));
    }
    let manifest: ModelManifest = serde_json::from_slice(&fs::read(&args.model_manifest)?)?;
    if manifest.lineage_id != LINEAGE_ID || manifest.model.sha256 != sha256_file(&args.model)? {
        return Err(anyhow!(
            "model is not the declared scratch-lineage artifact"
        ));
    }
    let model = HalfKpFlatModel::decode(&fs::read(&args.model)?)?;
    let mut report = Report {
        model_sha256: manifest.model.sha256,
        output_bias_cp: model.out_b * 1000.0,
        material_coefficient: model.out_w[model.out_w.len() - 1],
        ..Report::default()
    };
    let mut root_scores = Vec::new();
    for path in &args.dataset {
        let mut reader = CurriculumReader::open(path)?;
        if reader.header().lineage_id != LINEAGE_ID {
            return Err(anyhow!("dataset is outside scratch lineage"));
        }
        while let Some(record) = reader.read_record()? {
            report.records += 1;
            let root_score = score(&model, &record.root);
            root_scores.push(root_score);
            if let Some(target) = record.result {
                let probability = sigmoid(root_score / args.kappa_cp);
                report.value_bce += -(target as f64 * probability.max(1e-7).ln() as f64
                    + (1.0 - target) as f64 * (1.0 - probability).max(1e-7).ln() as f64);
                report.value_objectives += 1;
            }
            let distance = record
                .proof
                .as_ref()
                .filter(|proof| proof.status == ProofStatus::ProvenMate)
                .and_then(|proof| proof.mate_distance);
            match distance {
                Some(1) => report.mate1_records += 1,
                Some(3) => report.mate3_records += 1,
                _ => {}
            }
            for preference in &record.preferences {
                let good = score(&model, &preference.good_child);
                let bad = score(&model, &preference.bad_child);
                if preference.kind == PreferenceKind::SymmetryEqual {
                    report.symmetry_pairs += 1;
                    report.symmetry_rmse_cp += ((good - bad) as f64).powi(2);
                    continue;
                }
                // Children are opponent-to-move: lower is better for root.
                let correct = good < bad;
                report.pairs += 1;
                report.pair_correct += usize::from(correct);
                report.pair_ties += usize::from(good == bad);
                match distance {
                    Some(1) => {
                        report.mate1_pairs += 1;
                        report.mate1_pair_correct += usize::from(correct);
                    }
                    Some(3) => {
                        report.mate3_pairs += 1;
                        report.mate3_pair_correct += usize::from(correct);
                    }
                    _ => {}
                }
            }
        }
    }
    if report.value_objectives > 0 {
        report.value_bce /= report.value_objectives as f64;
    }
    if report.symmetry_pairs > 0 {
        report.symmetry_rmse_cp = (report.symmetry_rmse_cp / report.symmetry_pairs as f64).sqrt();
    }
    finish_score_stats(&mut report, &mut root_scores);
    println!(
        "records={} value_bce={:.8} pairs={} pair_accuracy={:.4} symmetry_pairs={} symmetry_rmse_cp={:.4} mate1={}/{} ({:.4}) mate3={}/{} ({:.4}) score_mean_cp={:.3} score_stddev_cp={:.3} score_abs_p99_cp={:.3} output_bias_cp={:.3} material_coefficient={:.6}",
        report.records,
        report.value_bce,
        report.pairs,
        ratio(report.pair_correct, report.pairs),
        report.symmetry_pairs,
        report.symmetry_rmse_cp,
        report.mate1_pair_correct,
        report.mate1_pairs,
        ratio(report.mate1_pair_correct, report.mate1_pairs),
        report.mate3_pair_correct,
        report.mate3_pairs,
        ratio(report.mate3_pair_correct, report.mate3_pairs),
        report.root_score_mean_cp,
        report.root_score_stddev_cp,
        report.root_score_abs_p99_cp,
        report.output_bias_cp,
        report.material_coefficient,
    );
    if let Some(path) = args.json {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    Ok(())
}

fn score(model: &HalfKpFlatModel, position: &PackedHalfKpPosition) -> f32 {
    HalfKpFlatModel::forward_parts(
        &model.feature_emb,
        &model.hidden_b,
        &model.out_w,
        model.out_b,
        position.features_black.iter().map(|&value| value as usize),
        position.features_white.iter().map(|&value| value as usize),
        position.material_black,
        position.material_white,
        position.side_to_move == Color::Black,
        model.header.target_scale,
    )
    .raw * model.header.target_scale
}

fn finish_score_stats(report: &mut Report, scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }
    let mean = scores.iter().map(|&value| value as f64).sum::<f64>() / scores.len() as f64;
    let variance = scores
        .iter()
        .map(|&value| (value as f64 - mean).powi(2))
        .sum::<f64>()
        / scores.len() as f64;
    report.root_score_mean_cp = mean;
    report.root_score_stddev_cp = variance.sqrt();
    scores.sort_by(|a, b| a.abs().total_cmp(&b.abs()));
    report.root_score_abs_p99_cp = scores[(scores.len() - 1) * 99 / 100].abs();
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    numerator as f64 / denominator.max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_stats_are_finite() {
        let mut report = Report::default();
        finish_score_stats(&mut report, &mut [-2.0, 0.0, 4.0]);
        assert!(report.root_score_mean_cp.is_finite());
        assert!(report.root_score_stddev_cp.is_finite());
        assert_eq!(report.root_score_abs_p99_cp, 2.0);
    }
}
