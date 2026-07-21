use anyhow::{anyhow, Result};
use clap::Parser;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use shogi_ai::evaluation::{HalfKpFlatModel, HALFKP_PIECE_STATES};
use shogi_ai::halfkp_training::{
    CurriculumReader, PackedHalfKpPosition, PreferenceKind, ProofStatus,
};
use shogi_ai::training_data::sha256_file;
use shogi_core::{Color, PartialPosition};
use shogi_lib::Position;
use shogi_usi_parser::FromUsi;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
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
    #[arg(long)]
    predictions_jsonl: Option<PathBuf>,
    #[arg(long, default_value_t = 10_000)]
    bootstrap_iterations: usize,
    #[arg(long, default_value_t = 20260820)]
    bootstrap_seed: u64,
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
    pair_families: usize,
    family_pair_accuracy_mean: f64,
    family_pair_accuracy_ci95_lower: f64,
    family_pair_accuracy_ci95_upper: f64,
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

#[derive(Serialize)]
struct FamilyPrediction {
    family_hash: String,
    group_id: u64,
    position_hash: u64,
    root_sfen: String,
    ply: u16,
    black_king_bucket: Option<usize>,
    white_king_bucket: Option<usize>,
    root_feature_count: usize,
    preferences: Vec<PreferencePrediction>,
}

#[derive(Serialize)]
struct PreferencePrediction {
    kind: String,
    good_move_usi: Option<String>,
    bad_move_usi: Option<String>,
    good_score_cp: f32,
    bad_score_cp: f32,
    margin_cp: f32,
    correct: bool,
    good_move: Option<MoveFacts>,
    bad_move: Option<MoveFacts>,
}

#[derive(Serialize)]
struct MoveFacts {
    checking: bool,
    capture: bool,
    promotion: bool,
    drop: bool,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    if args.kappa_cp <= 0.0 || args.bootstrap_iterations == 0 {
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
    let mut family_pairs = BTreeMap::<[u8; 32], (usize, usize)>::new();
    let mut predictions = match &args.predictions_jsonl {
        Some(path) => {
            if path.exists() {
                return Err(anyhow!("refusing to overwrite {}", path.display()));
            }
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            Some(BufWriter::new(File::create(path)?))
        }
        None => None,
    };
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
            let root_position = predictions
                .is_some()
                .then(|| parse_position(&record.root_sfen))
                .transpose()?;
            let mut preference_predictions = Vec::new();
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
                let family = family_pairs.entry(record.family_hash).or_default();
                family.0 += usize::from(correct);
                family.1 += 1;
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
                if let Some(position) = &root_position {
                    preference_predictions.push(PreferencePrediction {
                        kind: format!("{:?}", preference.kind).to_ascii_lowercase(),
                        good_move_usi: preference.good_move_usi.clone(),
                        bad_move_usi: preference.bad_move_usi.clone(),
                        good_score_cp: good,
                        bad_score_cp: bad,
                        margin_cp: bad - good,
                        correct,
                        good_move: move_facts(position, preference.good_move_usi.as_deref()),
                        bad_move: move_facts(position, preference.bad_move_usi.as_deref()),
                    });
                }
            }
            if let Some(writer) = &mut predictions {
                let prediction = FamilyPrediction {
                    family_hash: hex_hash(&record.family_hash),
                    group_id: record.group_id,
                    position_hash: record.position_hash,
                    root_sfen: record.root_sfen,
                    ply: record.ply,
                    black_king_bucket: king_bucket(&record.root.features_black),
                    white_king_bucket: king_bucket(&record.root.features_white),
                    root_feature_count: record.root.features_black.len()
                        + record.root.features_white.len(),
                    preferences: preference_predictions,
                };
                serde_json::to_writer(&mut *writer, &prediction)?;
                writer.write_all(b"\n")?;
            }
        }
    }
    if let Some(writer) = &mut predictions {
        writer.flush()?;
    }
    if report.value_objectives > 0 {
        report.value_bce /= report.value_objectives as f64;
    }
    if report.symmetry_pairs > 0 {
        report.symmetry_rmse_cp = (report.symmetry_rmse_cp / report.symmetry_pairs as f64).sqrt();
    }
    finish_score_stats(&mut report, &mut root_scores);
    finish_family_stats(&mut report, &family_pairs, &args);
    println!(
        "records={} value_bce={:.8} pairs={} pair_accuracy={:.4} family_accuracy={:.4} ci95=[{:.4},{:.4}] symmetry_pairs={} symmetry_rmse_cp={:.4} mate1={}/{} ({:.4}) mate3={}/{} ({:.4}) score_mean_cp={:.3} score_stddev_cp={:.3} score_abs_p99_cp={:.3} output_bias_cp={:.3} material_coefficient={:.6}",
        report.records,
        report.value_bce,
        report.pairs,
        ratio(report.pair_correct, report.pairs),
        report.family_pair_accuracy_mean,
        report.family_pair_accuracy_ci95_lower,
        report.family_pair_accuracy_ci95_upper,
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

fn finish_family_stats(
    report: &mut Report,
    families: &BTreeMap<[u8; 32], (usize, usize)>,
    args: &Args,
) {
    let values = families
        .values()
        .filter(|(_, total)| *total > 0)
        .map(|(correct, total)| *correct as f64 / *total as f64)
        .collect::<Vec<_>>();
    report.pair_families = values.len();
    if values.is_empty() {
        return;
    }
    report.family_pair_accuracy_mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut rng = ChaCha20Rng::seed_from_u64(args.bootstrap_seed);
    let mut estimates = Vec::with_capacity(args.bootstrap_iterations);
    for _ in 0..args.bootstrap_iterations {
        let mean = (0..values.len())
            .map(|_| values[rng.gen_range(0..values.len())])
            .sum::<f64>()
            / values.len() as f64;
        estimates.push(mean);
    }
    estimates.sort_by(f64::total_cmp);
    let last = estimates.len() - 1;
    report.family_pair_accuracy_ci95_lower = estimates[last * 25 / 1000];
    report.family_pair_accuracy_ci95_upper = estimates[last * 975 / 1000];
}

fn parse_position(root_sfen: &str) -> Result<Position> {
    Ok(Position::new(
        PartialPosition::from_usi(&format!("sfen {root_sfen}"))
            .map_err(|error| anyhow!("parse prediction SFEN: {error:?}"))?,
    ))
}

fn move_facts(position: &Position, text: Option<&str>) -> Option<MoveFacts> {
    let mv = shogi_ai::utils::parse_usi_move_for_color(text?, position.side_to_move())?;
    Some(MoveFacts {
        checking: position.is_check_move(mv),
        capture: position.piece_at(mv.to()).is_some(),
        promotion: mv.is_promoting(),
        drop: mv.is_drop(),
    })
}

fn king_bucket(features: &[u32]) -> Option<usize> {
    features
        .first()
        .map(|&feature| feature as usize / HALFKP_PIECE_STATES)
}

fn hex_hash(hash: &[u8; 32]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
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
