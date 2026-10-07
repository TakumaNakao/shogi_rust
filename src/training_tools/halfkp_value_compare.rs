use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::Serialize;
use shogi_ai::evaluation::HalfKpFlatModel;
use shogi_ai::halfkp_training::{
    read_search_teacher_manifest, PackedHalfKpPosition, SearchTeacherReader,
    CANDIDATE_QSEARCH_LEAF, CANDIDATE_SEARCH_BEST, SEARCH_TEACHER_SEMANTICS_ID,
    SEARCH_TEACHER_SEMANTICS_VERSION,
};
use shogi_ai::training_data::sha256_file;
use shogi_core::Color;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Paired value-target comparison for HalfKP HKST files")]
struct Args {
    #[arg(long, required = true)]
    dataset: Vec<PathBuf>,
    #[arg(long)]
    baseline_model: PathBuf,
    #[arg(long)]
    candidate_model: PathBuf,
    #[arg(long, default_value_t = 600.0)]
    kappa_cp: f32,
    #[arg(long, default_value_t = 0.8)]
    search_mix: f32,
    #[arg(long)]
    max_records: Option<usize>,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Default)]
struct Accumulator {
    records: usize,
    known_results: usize,
    sample_weight_sum: f64,
    value_loss: [f64; 2],
    log_loss: [f64; 2],
    search_abs_error: [f64; 2],
    result_brier: [f64; 2],
    result_log_loss: [f64; 2],
    result_weight_sum: f64,
    value_loss_improved: usize,
    value_loss_worsened: usize,
}

#[derive(Serialize)]
struct Artifact {
    path: String,
    sha256: String,
}

#[derive(Serialize)]
struct ModelSummary {
    value_brier: f64,
    value_log_loss: f64,
    search_abs_error_cp: f64,
    result_brier: Option<f64>,
    result_log_loss: Option<f64>,
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    metric: &'static str,
    datasets: Vec<Artifact>,
    baseline_model: Artifact,
    candidate_model: Artifact,
    records_read: usize,
    records: usize,
    known_results: usize,
    sample_weight_sum: f64,
    result_weight_sum: f64,
    kappa_cp: f32,
    search_mix: f32,
    baseline: ModelSummary,
    candidate: ModelSummary,
    candidate_minus_baseline: DeltaSummary,
    paired_counts: PairedCounts,
}

#[derive(Serialize)]
struct DeltaSummary {
    value_brier: f64,
    value_log_loss: f64,
    search_abs_error_cp: f64,
    result_brier: Option<f64>,
    result_log_loss: Option<f64>,
}

#[derive(Serialize)]
struct PairedCounts {
    value_loss_improved: usize,
    value_loss_worsened: usize,
    value_loss_unchanged: usize,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    if !args.kappa_cp.is_finite() || args.kappa_cp <= 0.0 {
        return Err(anyhow!("--kappa-cp must be finite and positive"));
    }
    if !args.search_mix.is_finite() || !(0.0..=1.0).contains(&args.search_mix) {
        return Err(anyhow!("--search-mix must be finite and in [0,1]"));
    }
    let baseline = load_model(&args.baseline_model)?;
    let candidate = load_model(&args.candidate_model)?;
    if baseline.header.target_scale != candidate.header.target_scale {
        return Err(anyhow!("baseline and candidate target scales differ"));
    }
    let mut accumulator = Accumulator::default();
    let mut datasets = Vec::new();
    let mut records_read = 0usize;
    for path in &args.dataset {
        validate_dataset(path)?;
        datasets.push(Artifact {
            path: path.display().to_string(),
            sha256: sha256_file(path)?,
        });
        let mut reader = SearchTeacherReader::open(path)?;
        while args.max_records.is_none_or(|limit| records_read < limit) {
            let Some(record) = reader.read_record()? else {
                break;
            };
            records_read += 1;
            let (position, target_score_cp, target_result) = value_target_for_record(&record);
            let target_probability = target_result
                .map(|result| {
                    let search_probability = sigmoid(target_score_cp / args.kappa_cp);
                    args.search_mix * search_probability + (1.0 - args.search_mix) * result
                })
                .unwrap_or_else(|| sigmoid(target_score_cp / args.kappa_cp));
            let scores = [score(&baseline, position), score(&candidate, position)];
            let probabilities = scores.map(|score_cp| sigmoid(score_cp / args.kappa_cp));
            let weight = f64::from(record.sample_weight);
            accumulator.records += 1;
            accumulator.known_results += usize::from(target_result.is_some());
            accumulator.sample_weight_sum += weight;
            for index in 0..2 {
                let error = f64::from(probabilities[index] - target_probability);
                accumulator.value_loss[index] += error * error * weight;
                accumulator.log_loss[index] +=
                    cross_entropy(target_probability, probabilities[index]) * weight;
                accumulator.search_abs_error[index] +=
                    f64::from((scores[index] - target_score_cp).abs()) * weight;
                if let Some(result) = target_result {
                    accumulator.result_brier[index] +=
                        f64::from((probabilities[index] - result).powi(2)) * weight;
                    accumulator.result_log_loss[index] +=
                        cross_entropy(result, probabilities[index]) * weight;
                }
            }
            if target_result.is_some() {
                accumulator.result_weight_sum += weight;
            }
            // Per-record paired counts are computed from the same unweighted quantity.
            let baseline_error = f64::from(probabilities[0] - target_probability).powi(2);
            let candidate_error = f64::from(probabilities[1] - target_probability).powi(2);
            if candidate_error + 1e-15 < baseline_error {
                accumulator.value_loss_improved += 1;
            } else if candidate_error > baseline_error + 1e-15 {
                accumulator.value_loss_worsened += 1;
            }
        }
        if args.max_records.is_some_and(|limit| records_read >= limit) {
            break;
        }
    }
    if accumulator.records == 0 || accumulator.sample_weight_sum <= 0.0 {
        return Err(anyhow!("no value records"));
    }
    let baseline_summary = summary(&accumulator, 0);
    let candidate_summary = summary(&accumulator, 1);
    let report = Report {
        schema_version: 1,
        metric: "hkst-value-target-paired-v1",
        datasets,
        baseline_model: artifact(&args.baseline_model)?,
        candidate_model: artifact(&args.candidate_model)?,
        records_read,
        records: accumulator.records,
        known_results: accumulator.known_results,
        sample_weight_sum: accumulator.sample_weight_sum,
        result_weight_sum: accumulator.result_weight_sum,
        kappa_cp: args.kappa_cp,
        search_mix: args.search_mix,
        candidate_minus_baseline: DeltaSummary {
            value_brier: candidate_summary.value_brier - baseline_summary.value_brier,
            value_log_loss: candidate_summary.value_log_loss - baseline_summary.value_log_loss,
            search_abs_error_cp: candidate_summary.search_abs_error_cp
                - baseline_summary.search_abs_error_cp,
            result_brier: match (
                baseline_summary.result_brier,
                candidate_summary.result_brier,
            ) {
                (Some(a), Some(b)) => Some(b - a),
                _ => None,
            },
            result_log_loss: match (
                baseline_summary.result_log_loss,
                candidate_summary.result_log_loss,
            ) {
                (Some(a), Some(b)) => Some(b - a),
                _ => None,
            },
        },
        paired_counts: PairedCounts {
            value_loss_improved: accumulator.value_loss_improved,
            value_loss_worsened: accumulator.value_loss_worsened,
            value_loss_unchanged: accumulator.records
                - accumulator.value_loss_improved
                - accumulator.value_loss_worsened,
        },
        baseline: baseline_summary,
        candidate: candidate_summary,
    };
    if args.output.exists() {
        return Err(anyhow!("refusing to overwrite {}", args.output.display()));
    }
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        &args.output,
        [serde_json::to_vec_pretty(&report)?.as_slice(), b"\n"].concat(),
    )
    .with_context(|| format!("write {}", args.output.display()))?;
    println!(
        "records={} value_brier={:.8}->{:.8} result_brier={:?}->{:?} improved={} worsened={}",
        report.records,
        report.baseline.value_brier,
        report.candidate.value_brier,
        report.baseline.result_brier,
        report.candidate.result_brier,
        report.paired_counts.value_loss_improved,
        report.paired_counts.value_loss_worsened,
    );
    Ok(())
}

fn load_model(path: &Path) -> Result<HalfKpFlatModel> {
    HalfKpFlatModel::decode(&fs::read(path).with_context(|| format!("read {}", path.display()))?)
}

fn artifact(path: &Path) -> Result<Artifact> {
    Ok(Artifact {
        path: path.display().to_string(),
        sha256: sha256_file(path)?,
    })
}

fn validate_dataset(path: &Path) -> Result<()> {
    let manifest = read_search_teacher_manifest(path)?
        .ok_or_else(|| anyhow!("{} has no HKST manifest", path.display()))?;
    if manifest.format != "HKST0002"
        || manifest.teacher_semantics_version != SEARCH_TEACHER_SEMANTICS_VERSION
        || manifest.teacher_semantics_id != SEARCH_TEACHER_SEMANTICS_ID
    {
        return Err(anyhow!(
            "{} uses unsupported teacher semantics",
            path.display()
        ));
    }
    let output = manifest
        .output
        .ok_or_else(|| anyhow!("{} manifest has no output artifact", path.display()))?;
    if output.sha256 != sha256_file(path)? {
        return Err(anyhow!("{} SHA mismatch", path.display()));
    }
    Ok(())
}

fn value_target_for_record(
    record: &shogi_ai::halfkp_training::SearchTeacherRecord,
) -> (&PackedHalfKpPosition, f32, Option<f32>) {
    let value_candidate = record.candidates.iter().find(|candidate| {
        candidate.flags & (CANDIDATE_SEARCH_BEST | CANDIDATE_QSEARCH_LEAF)
            == (CANDIDATE_SEARCH_BEST | CANDIDATE_QSEARCH_LEAF)
    });
    let position = value_candidate
        .map(|candidate| &candidate.child)
        .unwrap_or(&record.root);
    let target_score = value_candidate.map_or(record.root_search_score_cp, |candidate| {
        if candidate.child.side_to_move == record.root.side_to_move {
            record.root_search_score_cp
        } else {
            -record.root_search_score_cp
        }
    });
    let target_result = record.result.map(|result| {
        value_candidate.map_or(result, |candidate| {
            if candidate.child.side_to_move == record.root.side_to_move {
                result
            } else {
                1.0 - result
            }
        })
    });
    (position, target_score, target_result)
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

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn cross_entropy(target: f32, prediction: f32) -> f64 {
    let p = prediction.clamp(1e-7, 1.0 - 1e-7);
    -(f64::from(target) * f64::from(p).ln() + f64::from(1.0 - target) * f64::from(1.0 - p).ln())
}

fn summary(accumulator: &Accumulator, index: usize) -> ModelSummary {
    let result_brier = (accumulator.result_weight_sum > 0.0)
        .then(|| accumulator.result_brier[index] / accumulator.result_weight_sum);
    let result_log_loss = (accumulator.result_weight_sum > 0.0)
        .then(|| accumulator.result_log_loss[index] / accumulator.result_weight_sum);
    ModelSummary {
        value_brier: accumulator.value_loss[index] / accumulator.sample_weight_sum,
        value_log_loss: accumulator.log_loss[index] / accumulator.sample_weight_sum,
        search_abs_error_cp: accumulator.search_abs_error[index] / accumulator.sample_weight_sum,
        result_brier,
        result_log_loss,
    }
}
