use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::Serialize;
use shogi_ai::evaluation::HalfKpFlatModel;
use shogi_ai::halfkp_training::{
    read_search_teacher_manifest, PackedHalfKpPosition, SearchTeacherReader,
    SEARCH_TEACHER_SEMANTICS_ID, SEARCH_TEACHER_SEMANTICS_VERSION,
};
use shogi_ai::training_data::sha256_file;
use shogi_ai::training_tools::halfkp_search_objective::{
    eligible_game_vs_search_pair, observe_game_vs_search_pair, GameVsSearchObservation,
};
use shogi_core::Color;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Compare HalfKP models on the held-out game-vs-search training objective")]
struct Args {
    #[arg(long, required = true)]
    dataset: Vec<PathBuf>,
    #[arg(long)]
    baseline_model: PathBuf,
    #[arg(long)]
    candidate_model: PathBuf,
    #[arg(long, default_value_t = 300.0)]
    game_regret_cap_cp: f32,
    #[arg(long, default_value_t = 20.0)]
    rank_margin_cp: f32,
    #[arg(long, default_value_t = 100.0)]
    rank_temperature_cp: f32,
    #[arg(long)]
    max_records: Option<usize>,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Default)]
struct Accumulator {
    eligible: usize,
    sample_weight_sum: f64,
    baseline_correct: usize,
    candidate_correct: usize,
    baseline_margin_sum_cp: f64,
    candidate_margin_sum_cp: f64,
    baseline_weighted_loss_sum: f64,
    candidate_weighted_loss_sum: f64,
    margin_improved: usize,
    margin_worsened: usize,
    margin_unchanged: usize,
    accuracy_gained: usize,
    accuracy_lost: usize,
    margins_baseline: Vec<f32>,
    margins_candidate: Vec<f32>,
}

#[derive(Serialize)]
struct ModelSummary {
    accuracy: f64,
    mean_signed_margin_cp: f64,
    median_signed_margin_cp: f64,
    weighted_objective_loss: f64,
}

#[derive(Serialize)]
struct Artifact {
    path: String,
    sha256: String,
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    metric: &'static str,
    datasets: Vec<Artifact>,
    baseline_model: Artifact,
    candidate_model: Artifact,
    records_read: usize,
    eligible_records: usize,
    sample_weight_sum: f64,
    rank_margin_cp: f32,
    rank_temperature_cp: f32,
    game_regret_cap_cp: f32,
    baseline: ModelSummary,
    candidate: ModelSummary,
    candidate_minus_baseline: DeltaSummary,
    paired_counts: PairedCounts,
    objective_improved: bool,
}

#[derive(Serialize)]
struct DeltaSummary {
    accuracy: f64,
    mean_signed_margin_cp: f64,
    weighted_objective_loss: f64,
}

#[derive(Serialize)]
struct PairedCounts {
    margin_improved: usize,
    margin_worsened: usize,
    margin_unchanged: usize,
    accuracy_gained: usize,
    accuracy_lost: usize,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    if !args.game_regret_cap_cp.is_finite() || args.game_regret_cap_cp < 0.0 {
        return Err(anyhow!(
            "--game-regret-cap-cp must be finite and non-negative"
        ));
    }
    if !args.rank_margin_cp.is_finite() {
        return Err(anyhow!("--rank-margin-cp must be finite"));
    }
    if !args.rank_temperature_cp.is_finite() || args.rank_temperature_cp <= 0.0 {
        return Err(anyhow!("--rank-temperature-cp must be finite and positive"));
    }

    let baseline = load_model(&args.baseline_model)?;
    let candidate = load_model(&args.candidate_model)?;
    let mut datasets = Vec::new();
    let mut accumulator = Accumulator::default();
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
            let Some(pair) = eligible_game_vs_search_pair(&record, args.game_regret_cap_cp) else {
                continue;
            };
            let baseline_utilities = record
                .candidates
                .iter()
                .map(|entry| -score(&baseline, &entry.child))
                .collect::<Vec<_>>();
            let candidate_utilities = record
                .candidates
                .iter()
                .map(|entry| -score(&candidate, &entry.child))
                .collect::<Vec<_>>();
            let baseline_observation = observe_game_vs_search_pair(
                &baseline_utilities,
                pair,
                args.rank_margin_cp,
                args.rank_temperature_cp,
            );
            let candidate_observation = observe_game_vs_search_pair(
                &candidate_utilities,
                pair,
                args.rank_margin_cp,
                args.rank_temperature_cp,
            );
            accumulate(
                &mut accumulator,
                record.sample_weight,
                baseline_observation,
                candidate_observation,
            );
        }
        if args.max_records.is_some_and(|limit| records_read >= limit) {
            break;
        }
    }
    if accumulator.eligible == 0 || accumulator.sample_weight_sum <= 0.0 {
        return Err(anyhow!("no eligible game-vs-search records"));
    }

    let baseline_summary = summarize(
        accumulator.baseline_correct,
        accumulator.baseline_margin_sum_cp,
        accumulator.baseline_weighted_loss_sum,
        &mut accumulator.margins_baseline,
        accumulator.eligible,
        accumulator.sample_weight_sum,
    );
    let candidate_summary = summarize(
        accumulator.candidate_correct,
        accumulator.candidate_margin_sum_cp,
        accumulator.candidate_weighted_loss_sum,
        &mut accumulator.margins_candidate,
        accumulator.eligible,
        accumulator.sample_weight_sum,
    );
    let report = Report {
        schema_version: 1,
        metric: "game-vs-search-softplus-v1",
        datasets,
        baseline_model: Artifact {
            path: args.baseline_model.display().to_string(),
            sha256: sha256_file(&args.baseline_model)?,
        },
        candidate_model: Artifact {
            path: args.candidate_model.display().to_string(),
            sha256: sha256_file(&args.candidate_model)?,
        },
        records_read,
        eligible_records: accumulator.eligible,
        sample_weight_sum: accumulator.sample_weight_sum,
        rank_margin_cp: args.rank_margin_cp,
        rank_temperature_cp: args.rank_temperature_cp,
        game_regret_cap_cp: args.game_regret_cap_cp,
        candidate_minus_baseline: DeltaSummary {
            accuracy: candidate_summary.accuracy - baseline_summary.accuracy,
            mean_signed_margin_cp: candidate_summary.mean_signed_margin_cp
                - baseline_summary.mean_signed_margin_cp,
            weighted_objective_loss: candidate_summary.weighted_objective_loss
                - baseline_summary.weighted_objective_loss,
        },
        objective_improved: candidate_summary.weighted_objective_loss
            < baseline_summary.weighted_objective_loss,
        baseline: baseline_summary,
        candidate: candidate_summary,
        paired_counts: PairedCounts {
            margin_improved: accumulator.margin_improved,
            margin_worsened: accumulator.margin_worsened,
            margin_unchanged: accumulator.margin_unchanged,
            accuracy_gained: accumulator.accuracy_gained,
            accuracy_lost: accumulator.accuracy_lost,
        },
    };
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.output, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", args.output.display()))?;
    println!(
        "eligible={} loss={:.8}->{:.8} margin={:.4}->{:.4}cp accuracy={:.6}->{:.6} improved={}",
        report.eligible_records,
        report.baseline.weighted_objective_loss,
        report.candidate.weighted_objective_loss,
        report.baseline.mean_signed_margin_cp,
        report.candidate.mean_signed_margin_cp,
        report.baseline.accuracy,
        report.candidate.accuracy,
        report.objective_improved,
    );
    Ok(())
}

fn load_model(path: &Path) -> Result<HalfKpFlatModel> {
    HalfKpFlatModel::decode(&fs::read(path).with_context(|| format!("read {}", path.display()))?)
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
    let actual_sha = sha256_file(path)?;
    if output.sha256 != actual_sha {
        return Err(anyhow!(
            "{} SHA mismatch: manifest {}, actual {}",
            path.display(),
            output.sha256,
            actual_sha
        ));
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

fn accumulate(
    accumulator: &mut Accumulator,
    sample_weight: f32,
    baseline: GameVsSearchObservation,
    candidate: GameVsSearchObservation,
) {
    let sample_weight = f64::from(sample_weight);
    accumulator.eligible += 1;
    accumulator.sample_weight_sum += sample_weight;
    accumulator.baseline_correct += usize::from(baseline.correct);
    accumulator.candidate_correct += usize::from(candidate.correct);
    accumulator.baseline_margin_sum_cp += f64::from(baseline.signed_margin_cp);
    accumulator.candidate_margin_sum_cp += f64::from(candidate.signed_margin_cp);
    accumulator.baseline_weighted_loss_sum += f64::from(baseline.objective_loss) * sample_weight;
    accumulator.candidate_weighted_loss_sum += f64::from(candidate.objective_loss) * sample_weight;
    accumulator.margins_baseline.push(baseline.signed_margin_cp);
    accumulator
        .margins_candidate
        .push(candidate.signed_margin_cp);
    let delta = candidate.signed_margin_cp - baseline.signed_margin_cp;
    if delta > 1e-6 {
        accumulator.margin_improved += 1;
    } else if delta < -1e-6 {
        accumulator.margin_worsened += 1;
    } else {
        accumulator.margin_unchanged += 1;
    }
    accumulator.accuracy_gained += usize::from(!baseline.correct && candidate.correct);
    accumulator.accuracy_lost += usize::from(baseline.correct && !candidate.correct);
}

fn summarize(
    correct: usize,
    margin_sum_cp: f64,
    weighted_loss_sum: f64,
    margins: &mut [f32],
    eligible: usize,
    sample_weight_sum: f64,
) -> ModelSummary {
    margins.sort_by(f32::total_cmp);
    let median_signed_margin_cp = if margins.len().is_multiple_of(2) {
        f64::from(margins[margins.len() / 2 - 1] + margins[margins.len() / 2]) / 2.0
    } else {
        f64::from(margins[margins.len() / 2])
    };
    ModelSummary {
        accuracy: correct as f64 / eligible as f64,
        mean_signed_margin_cp: margin_sum_cp / eligible as f64,
        median_signed_margin_cp,
        weighted_objective_loss: weighted_loss_sum / sample_weight_sum,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_accumulator_detects_margin_improvement_without_accuracy_flip() {
        let mut accumulator = Accumulator::default();
        accumulate(
            &mut accumulator,
            2.0,
            GameVsSearchObservation {
                signed_margin_cp: -30.0,
                objective_loss: 1.0,
                correct: false,
            },
            GameVsSearchObservation {
                signed_margin_cp: -20.0,
                objective_loss: 0.8,
                correct: false,
            },
        );
        assert_eq!(1, accumulator.eligible);
        assert_eq!(1, accumulator.margin_improved);
        assert_eq!(0, accumulator.accuracy_gained);
        assert_eq!(0, accumulator.accuracy_lost);
        assert_eq!(2.0, accumulator.sample_weight_sum);
        assert_eq!(2.0, accumulator.baseline_weighted_loss_sum);
        assert!((accumulator.candidate_weighted_loss_sum - 1.6).abs() < 1e-6);
    }
}
