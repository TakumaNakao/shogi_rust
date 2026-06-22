use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use shogi_ai::evaluation::{extract_kpp_features_and_material, SparseModel};
use shogi_ai::utils::{parse_usi_move, position_from_sfen_or_usi};
use shogi_core::{Move, Piece};
use shogi_lib::Position;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Train KPP weights from MMTO-lite root rank JSONL (listwise)")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long)]
    train: PathBuf,
    #[arg(long)]
    valid: PathBuf,
    #[arg(long, value_name = "LABEL=PATH")]
    extra_valid: Vec<String>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
    #[arg(long, default_value_t = 600.0)]
    model_temperature: f32,
    #[arg(long, default_value_t = 600.0)]
    teacher_temperature: f32,
    #[arg(long, value_enum, default_value_t = CandidateScope::Scored)]
    candidate_scope: CandidateScope,
    #[arg(long, value_enum, default_value_t = LossMode::Listwise)]
    loss: LossMode,
    #[arg(long, default_value_t = 0.0)]
    train_min_teacher_gap: f32,
    #[arg(long)]
    train_max_teacher_gap: Option<f32>,
    #[arg(long, default_value_t = 0.0)]
    train_min_score_span: f32,
    #[arg(long, value_enum, default_value_t = ValidFilter::None)]
    valid_filter: ValidFilter,
    #[arg(long, default_value_t = 2)]
    min_candidates: usize,
    #[arg(long, default_value_t = 0.0)]
    pairwise_weight: f32,
    #[arg(long, alias = "pairwise-gap-cp", default_value_t = 1.0)]
    pairwise_gap: f32,
    #[arg(long, alias = "pairwise-margin-cp", default_value_t = 0.5)]
    pairwise_margin: f32,
    #[arg(long, default_value_t = 512)]
    pairwise_max_pairs_per_sample: usize,
    #[arg(long, default_value_t = true)]
    freeze_material: bool,
    #[arg(long, default_value_t = 0.0)]
    anchor_l2: f32,
    #[arg(long)]
    max_weight_delta: Option<f32>,
    #[arg(long)]
    best_checkpoint_path: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = BestMetric::SelectedRegret)]
    best_metric: BestMetric,
    #[arg(long, default_value_t = 300.0)]
    bad_regret_cp: f32,
    #[arg(long)]
    log_path: Option<PathBuf>,
    #[arg(long, default_value_t = 0.02)]
    learning_rate: f32,
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CandidateScope {
    Scored,
    Legal,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum LossMode {
    Listwise,
    #[value(name = "listwise-pairwise")]
    ListwisePairwise,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ValidFilter {
    None,
    Same,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BestMetric {
    #[value(name = "selected-regret")]
    SelectedRegret,
    #[value(name = "valid-ce")]
    ValidCe,
    #[value(name = "bad-regret")]
    BadRegret,
}

#[derive(Debug, Deserialize)]
struct RankRecord {
    sfen: String,
    #[serde(rename = "teacher_scores")]
    #[serde(default)]
    teacher_scores: Vec<LegacyTeacherScore>,
    #[serde(default)]
    candidates: Vec<RankCandidate>,
    #[serde(default)]
    root_score: Option<f32>,
    #[serde(default)]
    schema: Option<String>,
    #[serde(default)]
    version: Option<u8>,
}

#[derive(Debug, Deserialize)]
struct RankCandidate {
    #[serde(rename = "move")]
    move_usi: String,
    teacher_score: f32,
}

#[derive(Debug, Deserialize)]
struct LegacyTeacherScore {
    #[serde(rename = "move_usi")]
    move_usi: String,
    score: f32,
}

#[derive(Debug, Clone)]
struct Candidate {
    mv: Move,
    teacher_score: f32,
}

#[derive(Debug, Clone)]
struct Sample {
    position: Position,
    candidates: Vec<Candidate>,
    root_score: f32,
}

#[derive(Default)]
struct Metrics {
    samples: usize,
    ce: f32,
    top1: f32,
    selected_regret_sum: f32,
    expected_regret_sum: f32,
    bad_regret_count: usize,
    regrets: Vec<f32>,
    samples_with_model_score: usize,
    teacher_gap_sum: f32,
    teacher_gaps: Vec<f32>,
    pairwise_total: usize,
    pairwise_correct: usize,
}

#[derive(Default)]
struct ConstraintState {
    max_abs_delta: f32,
    clamped_count: usize,
}

#[derive(Debug)]
struct NamedBatch {
    label: String,
    samples: Vec<Sample>,
}

#[derive(Clone, Copy)]
struct LoadOptions {
    candidate_scope: CandidateScope,
    min_teacher_gap: f32,
    max_teacher_gap: Option<f32>,
    min_score_span: f32,
    min_candidates: usize,
}

#[derive(Clone, Copy)]
struct TrainOptions {
    model_temperature: f32,
    teacher_temperature: f32,
    loss: LossMode,
    pairwise_weight: f32,
    pairwise_gap: f32,
    pairwise_margin: f32,
    pairwise_max_pairs_per_sample: usize,
}

fn parse_move_for_position(position: &Position, move_text: &str) -> Option<Move> {
    parse_usi_move(move_text).map(|mv| match mv {
        Move::Drop { piece, to } => Move::Drop {
            piece: Piece::new(piece.piece_kind(), position.side_to_move()),
            to,
        },
        normal => normal,
    })
}

fn parse_extra_valid(spec: &str) -> Result<(String, PathBuf)> {
    let (label, path) = spec.split_once('=').ok_or_else(|| {
        anyhow!(
            "--extra-valid must use LABEL=PATH format (for example: --extra-valid hard=...jsonl)"
        )
    })?;
    if label.trim().is_empty() {
        return Err(anyhow!("--extra-valid label must not be empty: {spec}"));
    }
    if path.trim().is_empty() {
        return Err(anyhow!("--extra-valid path must not be empty: {spec}"));
    }
    Ok((label.trim().to_string(), PathBuf::from(path.trim())))
}

fn ensure_finite_model(model: &SparseModel) -> Result<()> {
    if !model.w.iter().all(|value| value.is_finite()) || !model.material_coeff.is_finite() {
        return Err(anyhow!("model contains NaN or inf"));
    }
    Ok(())
}

fn percentile(mut values: Vec<f32>, p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((values.len() - 1) as f32 * p).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn teacher_gap(candidates: &[Candidate]) -> Option<f32> {
    if candidates.len() < 2 {
        return None;
    }
    Some((candidates[0].teacher_score - candidates[1].teacher_score).max(0.0))
}

fn score_span(candidates: &[Candidate]) -> Option<f32> {
    let best = candidates.first()?.teacher_score;
    let worst = candidates.last()?.teacher_score;
    Some((best - worst).max(0.0))
}

fn load_samples(path: &PathBuf, options: LoadOptions) -> Result<Vec<Sample>> {
    if let CandidateScope::Legal = options.candidate_scope {
        return Err(anyhow!("--candidate-scope legal is not implemented yet"));
    }
    if options.min_candidates < 2 {
        return Err(anyhow!("--min-candidates must be at least 2"));
    }
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut samples = Vec::new();

    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: RankRecord = serde_json::from_str(&line)
            .map_err(|e| anyhow!("{}:{} invalid json: {}", path.display(), line_index + 1, e))?;
        if record
            .schema
            .as_deref()
            .is_some_and(|schema| schema != "kpp_rank_v1")
        {
            continue;
        }
        if record.version.is_some_and(|version| version == 0) {
            continue;
        }

        let Some(position) = position_from_sfen_or_usi(&record.sfen) else {
            continue;
        };

        let mut candidates = Vec::new();
        if !record.candidates.is_empty() {
            for candidate in record.candidates {
                let Some(mv) = parse_move_for_position(&position, &candidate.move_usi) else {
                    continue;
                };
                if !position.legal_moves().contains(&mv) {
                    continue;
                }
                if !candidate.teacher_score.is_finite() {
                    continue;
                }
                candidates.push(Candidate {
                    mv,
                    teacher_score: candidate.teacher_score,
                });
            }
        } else {
            for teacher_score in record.teacher_scores {
                let Some(mv) = parse_move_for_position(&position, &teacher_score.move_usi) else {
                    continue;
                };
                if !position.legal_moves().contains(&mv) {
                    continue;
                }
                if !teacher_score.score.is_finite() {
                    continue;
                }
                candidates.push(Candidate {
                    mv,
                    teacher_score: teacher_score.score,
                });
            }
        }

        if candidates.is_empty() {
            continue;
        }

        candidates.sort_by(|a, b| {
            b.teacher_score
                .partial_cmp(&a.teacher_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut deduped = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if !deduped
                .iter()
                .any(|item: &Candidate| item.mv == candidate.mv)
            {
                deduped.push(candidate);
            }
        }
        if deduped.len() < options.min_candidates {
            continue;
        }
        if options.min_teacher_gap > 0.0 || options.max_teacher_gap.is_some() {
            let Some(gap) = teacher_gap(&deduped) else {
                continue;
            };
            if gap < options.min_teacher_gap {
                continue;
            }
            if options.max_teacher_gap.is_some_and(|max_gap| gap > max_gap) {
                continue;
            }
        }
        if options.min_score_span > 0.0 {
            let Some(span) = score_span(&deduped) else {
                continue;
            };
            if span < options.min_score_span {
                continue;
            }
        }

        let root_score = if let Some(score) = record.root_score {
            score
        } else {
            deduped
                .iter()
                .map(|candidate| candidate.teacher_score)
                .fold(f32::NEG_INFINITY, f32::max)
        };
        if !root_score.is_finite() {
            continue;
        }

        samples.push(Sample {
            position,
            candidates: deduped,
            root_score,
        });
    }

    if samples.is_empty() {
        return Err(anyhow!("{} contains no usable samples", path.display()));
    }
    Ok(samples)
}

fn evaluate_batch(
    model: &SparseModel,
    batch: &[Sample],
    model_temperature: f32,
    teacher_temperature: f32,
    bad_regret_cp: f32,
) -> Metrics {
    let mut metrics = Metrics::default();
    if model_temperature <= 0.0 || !model_temperature.is_finite() {
        return metrics;
    }
    if teacher_temperature <= 0.0 || !teacher_temperature.is_finite() {
        return metrics;
    }

    for sample in batch {
        let mut entries = Vec::with_capacity(sample.candidates.len());
        for candidate in &sample.candidates {
            let mut child = sample.position.clone();
            child.do_move(candidate.mv);
            child.switch_turn();
            let (features, material) = extract_kpp_features_and_material(&child);
            let model_score = model.predict_with_material(&features, material);
            entries.push((
                candidate.mv,
                candidate.teacher_score,
                model_score,
                material,
                features,
            ));
        }
        if entries.is_empty() {
            continue;
        }
        let legal_teacher_scores: Vec<f32> =
            entries.iter().map(|(_, score, _, _, _)| *score).collect();
        let model_scores: Vec<f32> = entries.iter().map(|(_, _, score, _, _)| *score).collect();

        let best_teacher = sample.root_score;
        if !best_teacher.is_finite() {
            continue;
        }

        let max_teacher = legal_teacher_scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if !max_teacher.is_finite() {
            continue;
        }
        let teacher_total = legal_teacher_scores
            .iter()
            .map(|score| ((*score - max_teacher) / teacher_temperature).exp())
            .sum::<f32>();
        if teacher_total <= 0.0 || !teacher_total.is_finite() {
            continue;
        }

        let max_model = model_scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let model_exp = model_scores
            .iter()
            .map(|score| ((*score - max_model) / model_temperature).exp())
            .collect::<Vec<_>>();
        let model_total = model_exp.iter().sum::<f32>();
        if model_total <= 0.0 || !model_total.is_finite() {
            continue;
        }

        let mut selected_model_idx = 0usize;
        let mut selected_model_score = f32::NEG_INFINITY;
        for (idx, &(_, _, model_score, _, _)) in entries.iter().enumerate() {
            if model_score > selected_model_score {
                selected_model_score = model_score;
                selected_model_idx = idx;
            }
        }

        metrics.samples += 1;
        metrics.samples_with_model_score += 1;
        if let Some(gap) = teacher_gap(&sample.candidates) {
            metrics.teacher_gap_sum += gap;
            metrics.teacher_gaps.push(gap);
        }
        let mut sample_loss = 0.0;
        let mut selected_regret = 0.0;
        let mut expected_regret = 0.0;

        for (idx, (_, teacher_score, _, _, _)) in entries.iter().enumerate() {
            let model_prob = model_exp[idx] / model_total;
            let target_prob =
                (((*teacher_score - max_teacher) / teacher_temperature).exp()) / teacher_total;
            if target_prob > 0.0 {
                sample_loss += -target_prob * model_prob.max(1e-7).ln();
            }
            let regret = (best_teacher - *teacher_score).max(0.0);
            expected_regret += model_prob * regret;
            if idx == selected_model_idx {
                selected_regret = regret;
            }
        }
        metrics.ce += sample_loss;
        metrics.selected_regret_sum += selected_regret;
        metrics.expected_regret_sum += expected_regret;
        metrics.regrets.push(selected_regret);
        if selected_regret > bad_regret_cp {
            metrics.bad_regret_count += 1;
        }

        if selected_model_idx == 0 {
            // entries are sorted by teacher score, so idx=0 is teacher best.
            metrics.top1 += 1.0;
        }

        for good_idx in 0..entries.len() {
            for bad_idx in (good_idx + 1)..entries.len() {
                let teacher_gap = entries[good_idx].1 - entries[bad_idx].1;
                if teacher_gap <= 0.0 {
                    continue;
                }
                metrics.pairwise_total += 1;
                if entries[good_idx].2 >= entries[bad_idx].2 {
                    metrics.pairwise_correct += 1;
                }
            }
        }
    }

    if metrics.samples > 0 {
        metrics.ce /= metrics.samples as f32;
        metrics.top1 /= metrics.samples as f32;
        metrics.selected_regret_sum /= metrics.samples as f32;
        metrics.expected_regret_sum /= metrics.samples as f32;
        metrics.teacher_gap_sum /= metrics.samples as f32;
        metrics.top1 = metrics.top1 * 100.0;
    }
    metrics
}

fn update_batch_with_soft_targets(
    model: &mut SparseModel,
    batch: &[Sample],
    options: TrainOptions,
) -> (f32, usize) {
    let model_temperature = options.model_temperature;
    let teacher_temperature = options.teacher_temperature;
    if model_temperature <= 0.0 || !model_temperature.is_finite() {
        return (0.0, 0);
    }
    if teacher_temperature <= 0.0 || !teacher_temperature.is_finite() {
        return (0.0, 0);
    }

    let mut w_grads: HashMap<usize, f32> = HashMap::new();
    let mut material_grad_total = 0.0f32;
    let mut loss = 0.0f32;
    let mut valid_samples = 0usize;

    for sample in batch {
        let mut entries = Vec::with_capacity(sample.candidates.len());
        for candidate in &sample.candidates {
            let mut child = sample.position.clone();
            child.do_move(candidate.mv);
            child.switch_turn();
            let (features, material) = extract_kpp_features_and_material(&child);
            let model_score = model.predict_with_material(&features, material);
            entries.push((candidate.teacher_score, features, material, model_score));
        }
        if entries.is_empty() {
            continue;
        }

        let max_teacher = entries
            .iter()
            .map(|(teacher_score, _, _, _)| *teacher_score)
            .fold(f32::NEG_INFINITY, f32::max);
        if !max_teacher.is_finite() {
            continue;
        }
        let teacher_total = entries
            .iter()
            .map(|(teacher_score, _, _, _)| {
                ((*teacher_score - max_teacher) / teacher_temperature).exp()
            })
            .sum::<f32>();
        if teacher_total <= 0.0 || !teacher_total.is_finite() {
            continue;
        }

        let max_model = entries
            .iter()
            .map(|(_, _, _, model_score)| *model_score)
            .fold(f32::NEG_INFINITY, f32::max);
        let model_exp = entries
            .iter()
            .map(|(_, _, _, model_score)| ((*model_score - max_model) / model_temperature).exp())
            .collect::<Vec<_>>();
        let model_total = model_exp.iter().sum::<f32>();
        if model_total <= 0.0 || !model_total.is_finite() {
            continue;
        }

        valid_samples += 1;
        for (idx, (teacher_score, features, material, _)) in entries.iter().enumerate() {
            let model_prob = model_exp[idx] / model_total;
            let target_prob =
                ((*teacher_score - max_teacher) / teacher_temperature).exp() / teacher_total;
            if target_prob > 0.0 {
                loss += -target_prob * model_prob.max(1e-7).ln();
            }
            let delta = (model_prob - target_prob) / model_temperature;
            for &feature_idx in features {
                *w_grads.entry(feature_idx).or_insert(0.0) += delta;
            }
            material_grad_total += delta * *material;
        }

        if matches!(options.loss, LossMode::ListwisePairwise) && options.pairwise_weight > 0.0 {
            let mut violated_pairs = Vec::new();
            'pairs: for good_idx in 0..entries.len() {
                for bad_idx in (good_idx + 1)..entries.len() {
                    let teacher_gap = entries[good_idx].0 - entries[bad_idx].0;
                    if teacher_gap < options.pairwise_gap {
                        continue;
                    }
                    let model_gap = entries[good_idx].3 - entries[bad_idx].3;
                    let violation = options.pairwise_margin - model_gap;
                    if violation <= 0.0 {
                        continue;
                    }
                    violated_pairs.push((good_idx, bad_idx, violation));
                    if violated_pairs.len() >= options.pairwise_max_pairs_per_sample {
                        break 'pairs;
                    }
                }
            }
            if !violated_pairs.is_empty() {
                let scale = options.pairwise_weight / violated_pairs.len() as f32;
                for (good_idx, bad_idx, violation) in violated_pairs {
                    loss += scale * violation;
                    let (good_features, good_material) =
                        (&entries[good_idx].1, entries[good_idx].2);
                    let (bad_features, bad_material) = (&entries[bad_idx].1, entries[bad_idx].2);
                    for &feature_idx in good_features {
                        *w_grads.entry(feature_idx).or_insert(0.0) -= scale;
                    }
                    for &feature_idx in bad_features {
                        *w_grads.entry(feature_idx).or_insert(0.0) += scale;
                    }
                    material_grad_total += scale * (bad_material - good_material);
                }
            }
        }
    }

    if valid_samples == 0 {
        return (0.0, 0);
    }

    let avg_loss = loss / valid_samples as f32;
    for (idx, grad_sum) in w_grads {
        model.w[idx] -=
            model.kpp_eta * (grad_sum / valid_samples as f32 + model.l2_lambda * model.w[idx]);
    }
    model.material_coeff -= model.kpp_eta
        * (material_grad_total / valid_samples as f32 + model.l2_lambda * model.material_coeff);

    (avg_loss, valid_samples)
}

fn apply_weight_constraints(
    model: &mut SparseModel,
    initial_weights: &[f32],
    anchor_l2: f32,
    max_weight_delta: Option<f32>,
) -> ConstraintState {
    if initial_weights.len() != model.w.len() {
        return ConstraintState::default();
    }

    let limit = max_weight_delta;
    let mut state = ConstraintState::default();
    for (weight, anchor_weight) in model.w.iter_mut().zip(initial_weights.iter()) {
        if anchor_l2 > 0.0 {
            *weight += anchor_l2 * (*anchor_weight - *weight);
        }
        if let Some(max_delta) = limit {
            let raw_delta = *weight - *anchor_weight;
            let clamped_delta = raw_delta.clamp(-max_delta, max_delta);
            if clamped_delta != raw_delta {
                state.clamped_count += 1;
            }
            *weight = *anchor_weight + clamped_delta;
        }
        let delta = (*weight - *anchor_weight).abs();
        if delta > state.max_abs_delta {
            state.max_abs_delta = delta;
        }
    }
    state
}

fn log_summary(name: &str, metrics: &Metrics, bad_regret_cp: f32) -> String {
    let pairwise_accuracy = if metrics.pairwise_total == 0 {
        0.0
    } else {
        metrics.pairwise_correct as f32 / metrics.pairwise_total as f32 * 100.0
    };
    format!(
        "{name} ce={:.6}, top1={:.2}%, mean={:.2}, p90={:.2}, p95={:.2}, bad>{:.0}={:.4}, expected={:.2}, gap_mean={:.2}, gap_p50={:.2}, pairwise={:.2}%, samples={}",
        metrics.ce,
        metrics.top1,
        metrics.selected_regret_sum,
        percentile(metrics.regrets.clone(), 0.90),
        percentile(metrics.regrets.clone(), 0.95),
        bad_regret_cp,
        (metrics.bad_regret_count as f32 / metrics.samples.max(1) as f32) * 100.0,
        metrics.expected_regret_sum,
        metrics.teacher_gap_sum,
        percentile(metrics.teacher_gaps.clone(), 0.50),
        pairwise_accuracy,
        metrics.samples
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.epochs == 0 {
        return Err(anyhow!("--epochs must be greater than zero"));
    }
    if args.batch_size == 0 {
        return Err(anyhow!("--batch-size must be greater than zero"));
    }
    if !args.model_temperature.is_finite() || args.model_temperature <= 0.0 {
        return Err(anyhow!("--model-temperature must be positive"));
    }
    if !args.teacher_temperature.is_finite() || args.teacher_temperature <= 0.0 {
        return Err(anyhow!("--teacher-temperature must be positive"));
    }
    if !args.bad_regret_cp.is_finite() || args.bad_regret_cp < 0.0 {
        return Err(anyhow!("--bad-regret-cp must be non-negative"));
    }
    if !args.learning_rate.is_finite() || args.learning_rate <= 0.0 {
        return Err(anyhow!("--learning-rate must be positive"));
    }
    if !args.train_min_teacher_gap.is_finite() || args.train_min_teacher_gap < 0.0 {
        return Err(anyhow!("--train-min-teacher-gap must be non-negative"));
    }
    if let Some(max_teacher_gap) = args.train_max_teacher_gap {
        if !max_teacher_gap.is_finite() || max_teacher_gap < 0.0 {
            return Err(anyhow!("--train-max-teacher-gap must be non-negative"));
        }
        if max_teacher_gap < args.train_min_teacher_gap {
            return Err(anyhow!(
                "--train-max-teacher-gap must be greater than or equal to --train-min-teacher-gap"
            ));
        }
    }
    if !args.train_min_score_span.is_finite() || args.train_min_score_span < 0.0 {
        return Err(anyhow!("--train-min-score-span must be non-negative"));
    }
    if args.min_candidates < 2 {
        return Err(anyhow!("--min-candidates must be at least 2"));
    }
    if !args.pairwise_weight.is_finite() || args.pairwise_weight < 0.0 {
        return Err(anyhow!("--pairwise-weight must be non-negative"));
    }
    if !args.pairwise_gap.is_finite() || args.pairwise_gap < 0.0 {
        return Err(anyhow!("--pairwise-gap must be non-negative"));
    }
    if !args.pairwise_margin.is_finite() || args.pairwise_margin < 0.0 {
        return Err(anyhow!("--pairwise-margin must be non-negative"));
    }
    if args.pairwise_max_pairs_per_sample == 0 {
        return Err(anyhow!(
            "--pairwise-max-pairs-per-sample must be greater than zero"
        ));
    }
    if let Some(max_delta) = args.max_weight_delta {
        if !max_delta.is_finite() || max_delta <= 0.0 {
            return Err(anyhow!("--max-weight-delta must be > 0"));
        }
    }
    if args.anchor_l2 < 0.0 || !args.anchor_l2.is_finite() {
        return Err(anyhow!("--anchor-l2 must be non-negative"));
    }

    let mut model = SparseModel::new(args.learning_rate, 0.0);
    model
        .load(&args.weights)
        .map_err(|e| anyhow!("failed to load {}: {}", args.weights.display(), e))?;
    model.kpp_eta = args.learning_rate;
    let initial_model = SparseModel {
        w: model.w.clone(),
        bias: model.bias,
        material_coeff: model.material_coeff,
        kpp_eta: model.kpp_eta,
        l2_lambda: model.l2_lambda,
    };

    let train_load_options = LoadOptions {
        candidate_scope: args.candidate_scope,
        min_teacher_gap: args.train_min_teacher_gap,
        max_teacher_gap: args.train_max_teacher_gap,
        min_score_span: args.train_min_score_span,
        min_candidates: args.min_candidates,
    };
    let valid_load_options = match args.valid_filter {
        ValidFilter::None => LoadOptions {
            candidate_scope: args.candidate_scope,
            min_teacher_gap: 0.0,
            max_teacher_gap: None,
            min_score_span: 0.0,
            min_candidates: args.min_candidates,
        },
        ValidFilter::Same => train_load_options,
    };
    let train_samples = load_samples(&args.train, train_load_options)?;
    let valid_samples = load_samples(&args.valid, valid_load_options)?;
    let extra_valid = args
        .extra_valid
        .iter()
        .map(|spec| {
            let (label, path) = parse_extra_valid(spec)?;
            let samples = load_samples(&path, valid_load_options)?;
            Ok(NamedBatch { label, samples })
        })
        .collect::<Result<Vec<_>>>()?;

    let baseline_train = evaluate_batch(
        &model,
        &train_samples,
        args.model_temperature,
        args.teacher_temperature,
        args.bad_regret_cp,
    );
    let baseline_valid = evaluate_batch(
        &model,
        &valid_samples,
        args.model_temperature,
        args.teacher_temperature,
        args.bad_regret_cp,
    );

    let mut log_file = if let Some(path) = &args.log_path {
        Some(BufWriter::new(File::create(path)?))
    } else {
        None
    };

    let baseline_metric = match args.best_metric {
        BestMetric::SelectedRegret => {
            if baseline_valid.samples > 0 {
                baseline_valid.selected_regret_sum
            } else if baseline_train.samples > 0 {
                baseline_train.selected_regret_sum
            } else {
                f32::INFINITY
            }
        }
        BestMetric::ValidCe => {
            if baseline_valid.samples > 0 {
                baseline_valid.ce
            } else {
                baseline_train.ce
            }
        }
        BestMetric::BadRegret => {
            baseline_valid.bad_regret_count as f32 / baseline_valid.samples.max(1) as f32
        }
    };
    let mut best_metric_value: Option<f32> = Some(baseline_metric);
    let mut best_model_checkpoint = model.clone();
    let mut best_epoch = 0usize;
    let train_options = TrainOptions {
        model_temperature: args.model_temperature,
        teacher_temperature: args.teacher_temperature,
        loss: args.loss,
        pairwise_weight: args.pairwise_weight,
        pairwise_gap: args.pairwise_gap,
        pairwise_margin: args.pairwise_margin,
        pairwise_max_pairs_per_sample: args.pairwise_max_pairs_per_sample,
    };

    if let Some(file) = log_file.as_mut() {
        writeln!(
            file,
            "epoch,train_ce,train_top1,train_selected_regret_mean,train_p90_regret,train_p95_regret,train_bad_regret_ratio,train_expected_regret,train_teacher_gap_mean,train_teacher_gap_p50,train_pairwise_accuracy,train_samples,valid_ce,valid_top1,valid_selected_regret_mean,valid_p90_regret,valid_p95_regret,valid_bad_regret_ratio,valid_expected_regret,valid_teacher_gap_mean,valid_teacher_gap_p50,valid_pairwise_accuracy,valid_samples,max_abs_delta,p95_abs_delta,clamped_weights,material_coeff"
        )?;
        let baseline_train_pairwise = if baseline_train.pairwise_total == 0 {
            0.0
        } else {
            baseline_train.pairwise_correct as f32 / baseline_train.pairwise_total as f32 * 100.0
        };
        let baseline_valid_pairwise = if baseline_valid.pairwise_total == 0 {
            0.0
        } else {
            baseline_valid.pairwise_correct as f32 / baseline_valid.pairwise_total as f32 * 100.0
        };
        writeln!(
            file,
            "0,{:.6},{:.2},{:.2},{:.2},{:.2},{:.6},{:.2},{:.2},{:.2},{:.2},{},{:.6},{:.2},{:.2},{:.2},{:.2},{:.6},{:.2},{:.2},{:.2},{:.2},{},{:.6},{:.4},{},{}",
            baseline_train.ce,
            baseline_train.top1,
            baseline_train.selected_regret_sum,
            percentile(baseline_train.regrets.clone(), 0.90),
            percentile(baseline_train.regrets.clone(), 0.95),
            baseline_train.bad_regret_count as f32 / baseline_train.samples.max(1) as f32,
            baseline_train.expected_regret_sum,
            baseline_train.teacher_gap_sum,
            percentile(baseline_train.teacher_gaps.clone(), 0.50),
            baseline_train_pairwise,
            baseline_train.samples,
            baseline_valid.ce,
            baseline_valid.top1,
            baseline_valid.selected_regret_sum,
            percentile(baseline_valid.regrets.clone(), 0.90),
            percentile(baseline_valid.regrets.clone(), 0.95),
            baseline_valid.bad_regret_count as f32 / baseline_valid.samples.max(1) as f32,
            baseline_valid.expected_regret_sum,
            baseline_valid.teacher_gap_sum,
            percentile(baseline_valid.teacher_gaps.clone(), 0.50),
            baseline_valid_pairwise,
            baseline_valid.samples,
            0.0,
            0.0,
            0,
            model.material_coeff
        )?;
        for batch in &extra_valid {
            let metrics = evaluate_batch(
                &model,
                &batch.samples,
                args.model_temperature,
                args.teacher_temperature,
                args.bad_regret_cp,
            );
            let bad_ratio = metrics.bad_regret_count as f32 / metrics.samples.max(1) as f32;
            println!(
                "baseline extra_valid[{}] samples={} ce={:.6} top1={:.2}% mean={:.2} p90={:.2} p95={:.2} bad_ratio={:.4}",
                batch.label,
                metrics.samples,
                metrics.ce,
                metrics.top1,
                metrics.selected_regret_sum,
                percentile(metrics.regrets.clone(), 0.90),
                percentile(metrics.regrets.clone(), 0.95),
                bad_ratio
            );
        }
        file.flush()?;
    }

    println!(
        "baseline train: {}",
        log_summary("train", &baseline_train, args.bad_regret_cp)
    );
    println!(
        "baseline valid: {}",
        log_summary("valid", &baseline_valid, args.bad_regret_cp)
    );
    if args.dry_run {
        ensure_finite_model(&model)?;
        return Ok(());
    }

    let mut rng = ChaCha8Rng::seed_from_u64(0x1234);
    for epoch in 1..=args.epochs {
        let mut shuffled_train = train_samples.clone();
        shuffled_train.shuffle(&mut rng);
        let mut clamped_weights = 0usize;

        for chunk in shuffled_train.chunks(args.batch_size) {
            let _ = update_batch_with_soft_targets(&mut model, chunk, train_options);
            if args.freeze_material {
                model.material_coeff = initial_model.material_coeff;
            }
            if args.anchor_l2 > 0.0 || args.max_weight_delta.is_some() {
                let state = apply_weight_constraints(
                    &mut model,
                    &initial_model.w,
                    args.anchor_l2,
                    args.max_weight_delta,
                );
                clamped_weights += state.clamped_count;
            }
            ensure_finite_model(&model)?;
        }

        let train_metrics = evaluate_batch(
            &model,
            &train_samples,
            args.model_temperature,
            args.teacher_temperature,
            args.bad_regret_cp,
        );
        let valid_metrics = evaluate_batch(
            &model,
            &valid_samples,
            args.model_temperature,
            args.teacher_temperature,
            args.bad_regret_cp,
        );

        let max_abs_delta = model
            .w
            .iter()
            .zip(initial_model.w.iter())
            .map(|(w, anchor)| (w - anchor).abs())
            .fold(0.0, f32::max);
        let mut abs_deltas = model
            .w
            .iter()
            .zip(initial_model.w.iter())
            .map(|(w, anchor)| (w - anchor).abs())
            .collect::<Vec<_>>();
        abs_deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p95_abs_delta = percentile(abs_deltas, 0.95);

        let train_mean_regret = train_metrics.selected_regret_sum;
        let valid_mean_regret = valid_metrics.selected_regret_sum;
        let train_p90 = percentile(train_metrics.regrets.clone(), 0.90);
        let train_p95 = percentile(train_metrics.regrets.clone(), 0.95);
        let valid_p90 = percentile(valid_metrics.regrets.clone(), 0.90);
        let valid_p95 = percentile(valid_metrics.regrets.clone(), 0.95);
        let train_bad_ratio =
            train_metrics.bad_regret_count as f32 / train_metrics.samples.max(1) as f32;
        let valid_bad_ratio =
            valid_metrics.bad_regret_count as f32 / valid_metrics.samples.max(1) as f32;
        let train_pairwise_accuracy = if train_metrics.pairwise_total == 0 {
            0.0
        } else {
            train_metrics.pairwise_correct as f32 / train_metrics.pairwise_total as f32 * 100.0
        };
        let valid_pairwise_accuracy = if valid_metrics.pairwise_total == 0 {
            0.0
        } else {
            valid_metrics.pairwise_correct as f32 / valid_metrics.pairwise_total as f32 * 100.0
        };

        let current_metric = match args.best_metric {
            BestMetric::SelectedRegret => {
                if valid_metrics.samples > 0 {
                    valid_metrics.selected_regret_sum
                } else if train_metrics.samples > 0 {
                    train_metrics.selected_regret_sum
                } else {
                    f32::INFINITY
                }
            }
            BestMetric::ValidCe => {
                if valid_metrics.samples > 0 {
                    valid_metrics.ce
                } else {
                    train_metrics.ce
                }
            }
            BestMetric::BadRegret => valid_bad_ratio,
        };

        if best_metric_value.is_none()
            || current_metric < best_metric_value.unwrap_or(f32::INFINITY)
        {
            best_metric_value = Some(current_metric);
            best_model_checkpoint = model.clone();
            best_epoch = epoch;
        }

        println!(
            "epoch {} train_ce={:.6} train_top1={:.2}% train_mean_regret={:.2} train_p90={:.2} train_p95={:.2} train_bad_ratio={:.4} expected_regret={:.2} train_gap={:.2} train_pairwise={:.2}% valid_ce={:.6} valid_top1={:.2}% valid_mean_regret={:.2} valid_p90={:.2} valid_p95={:.2} valid_bad_ratio={:.4} expected_regret={:.2} valid_gap={:.2} valid_pairwise={:.2}% max_abs_delta={:.4} p95_abs_delta={:.4} clamped_weights={}",
            epoch,
            train_metrics.ce,
            if train_metrics.samples > 0 { train_metrics.top1 } else { 0.0 },
            train_mean_regret,
            train_p90,
            train_p95,
            train_bad_ratio,
            train_metrics.expected_regret_sum,
            train_metrics.teacher_gap_sum,
            train_pairwise_accuracy,
            valid_metrics.ce,
            if valid_metrics.samples > 0 { valid_metrics.top1 } else { 0.0 },
            valid_mean_regret,
            valid_p90,
            valid_p95,
            valid_bad_ratio,
            valid_metrics.expected_regret_sum,
            valid_metrics.teacher_gap_sum,
            valid_pairwise_accuracy,
            max_abs_delta,
            p95_abs_delta,
            clamped_weights
        );

        if let Some(file) = log_file.as_mut() {
            writeln!(
            file,
            "{},{:.6},{:.2},{:.2},{:.2},{:.2},{:.6},{:.2},{:.2},{:.2},{:.2},{},{:.6},{:.2},{:.2},{:.2},{:.2},{:.6},{:.2},{:.2},{:.2},{:.2},{},{:.6},{:.4},{},{}",
            epoch,
            train_metrics.ce,
            if train_metrics.samples > 0 { train_metrics.top1 } else { 0.0 },
                train_mean_regret,
                train_p90,
                train_p95,
                train_bad_ratio,
                train_metrics.expected_regret_sum,
                train_metrics.teacher_gap_sum,
                percentile(train_metrics.teacher_gaps.clone(), 0.50),
                train_pairwise_accuracy,
                train_metrics.samples,
                valid_metrics.ce,
                if valid_metrics.samples > 0 { valid_metrics.top1 } else { 0.0 },
                valid_mean_regret,
                valid_p90,
                valid_p95,
                valid_bad_ratio,
                valid_metrics.expected_regret_sum,
                valid_metrics.teacher_gap_sum,
                percentile(valid_metrics.teacher_gaps.clone(), 0.50),
                valid_pairwise_accuracy,
                valid_metrics.samples,
                max_abs_delta,
                p95_abs_delta,
                clamped_weights,
                model.material_coeff
            )?;
            file.flush()?;
        }
    }

    if let Some(path) = args.best_checkpoint_path.as_deref() {
        best_model_checkpoint.save(path)?;
        println!(
            "Best checkpoint saved at epoch {} to {}",
            best_epoch,
            path.display()
        );
    }

    for batch in &extra_valid {
        let metrics = evaluate_batch(
            &best_model_checkpoint,
            &batch.samples,
            args.model_temperature,
            args.teacher_temperature,
            args.bad_regret_cp,
        );
        let bad_ratio = metrics.bad_regret_count as f32 / metrics.samples.max(1) as f32;
        println!(
            "final extra_valid[{}] samples={} ce={:.6} top1={:.2}% mean={:.2} p90={:.2} p95={:.2} bad_ratio={:.4}",
            batch.label,
            metrics.samples,
            metrics.ce,
            metrics.top1,
            metrics.selected_regret_sum,
            percentile(metrics.regrets.clone(), 0.90),
            percentile(metrics.regrets.clone(), 0.95),
            bad_ratio
        );
    }

    best_model_checkpoint.save(&args.output)?;
    println!(
        "saved model to {} (best_epoch={})",
        args.output.display(),
        best_epoch
    );
    ensure_finite_model(&best_model_checkpoint)?;
    Ok(())
}
