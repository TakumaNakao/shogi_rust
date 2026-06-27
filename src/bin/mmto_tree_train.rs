use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use shogi_ai::evaluation::{extract_kpp_features_and_material, SparseModel};
use shogi_ai::utils::{parse_usi_move, position_from_sfen_or_usi};
use shogi_core::{Color, Move, Piece};
use shogi_lib::Position;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

const MISSING_RANK: usize = usize::MAX;

#[derive(Parser, Debug)]
#[command(about = "Train KPP weights from mmto_tree_v1 leaf pairwise JSONL")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long)]
    train: PathBuf,
    #[arg(long)]
    valid: PathBuf,
    #[arg(long, value_name = "LABEL=PATH")]
    extra_valid: Vec<String>,
    #[arg(long, default_value_t = 0.0)]
    extra_valid_best_weight: f32,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    #[arg(long, default_value_t = 128)]
    batch_size: usize,
    #[arg(long, default_value_t = 0.05)]
    learning_rate: f32,
    #[arg(long, default_value_t = 0.0)]
    l2_lambda: f32,
    #[arg(long, default_value_t = 50.0)]
    margin_cp: f32,
    #[arg(long, default_value_t = 100.0)]
    softplus_temp_cp: f32,
    #[arg(long, default_value_t = 80.0)]
    teacher_temperature_cp: f32,
    #[arg(long, default_value_t = 80.0)]
    model_temperature_cp: f32,
    #[arg(long, default_value_t = 2)]
    teacher_top_k: usize,
    #[arg(long, default_value_t = 5)]
    student_bad_top_k: usize,
    #[arg(long, default_value_t = 50.0)]
    min_regret_cp: f32,
    #[arg(long, value_enum, default_value_t = BadCandidateScope::StudentTop)]
    bad_candidate_scope: BadCandidateScope,
    #[arg(long, default_value_t = 16)]
    max_pairs_per_sample: usize,
    #[arg(long, value_enum, default_value_t = PairMiningMode::First)]
    pair_mining: PairMiningMode,
    #[arg(long, value_enum, default_value_t = PairWeightMode::None)]
    pair_weight_mode: PairWeightMode,
    #[arg(long, default_value_t = 100.0)]
    pair_weight_scale_cp: f32,
    #[arg(long, default_value_t = 4.0)]
    max_pair_weight: f32,
    #[arg(long, value_enum, default_value_t = OptimizerKind::Adagrad)]
    optimizer: OptimizerKind,
    #[arg(long, default_value_t = 1e-6)]
    adagrad_epsilon: f32,
    #[arg(long, default_value_t = true)]
    freeze_material: bool,
    #[arg(long, default_value_t = 0.0)]
    anchor_l2: f32,
    #[arg(long)]
    max_weight_delta: Option<f32>,
    #[arg(long)]
    best_checkpoint_path: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = BestMetric::ValidLoss)]
    best_metric: BestMetric,
    #[arg(long, default_value_t = 300.0)]
    selected_regret_cap_cp: f32,
    #[arg(long, default_value_t = 100.0)]
    bad_regret_cp: f32,
    #[arg(long, default_value = "50,100,200,300")]
    bad_regret_thresholds_cp: String,
    #[arg(long)]
    log_path: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    dry_run: bool,
    #[arg(long, default_value_t = false)]
    stream_train: bool,
    #[arg(long)]
    replay_train: Vec<PathBuf>,
    #[arg(long, default_value_t = 0.0)]
    replay_weight: f32,
    #[arg(long, default_value_t = 0)]
    replay_max_samples: usize,
    #[arg(long, value_enum, default_value_t = LossMode::Pairwise)]
    loss_mode: LossMode,
    #[arg(long, value_enum, default_value_t = ListwiseFeatureSource::TeacherLeaf)]
    listwise_feature_source: ListwiseFeatureSource,
    #[arg(long, default_value_t = 0.0)]
    listwise_hard_negative_weight: f32,
    #[arg(long, default_value_t = 50.0)]
    listwise_hard_negative_min_regret_cp: f32,
    #[arg(long, default_value_t = 0.0)]
    game_teacher_margin_weight: f32,
    #[arg(long, default_value_t = 150.0)]
    game_teacher_max_regret_cp: f32,
    #[arg(long, default_value_t = 15.0)]
    game_teacher_min_bad_regret_cp: f32,
    #[arg(long, default_value_t = 0.0)]
    current_top_margin_weight: f32,
    #[arg(long, default_value_t = 15.0)]
    current_top_min_bad_regret_cp: f32,
    #[arg(long, default_value_t = 0)]
    stream_train_eval_max_samples: usize,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OptimizerKind {
    Sgd,
    Adagrad,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BestMetric {
    #[value(name = "valid-loss")]
    ValidLoss,
    #[value(name = "selected-regret")]
    SelectedRegret,
    #[value(name = "bad-regret")]
    BadRegret,
    #[value(name = "p90-regret")]
    P90Regret,
    #[value(name = "p95-regret")]
    P95Regret,
    #[value(name = "p99-regret")]
    P99Regret,
    #[value(name = "bad50-regret")]
    #[value(alias = "bad-regret-50")]
    Bad50Regret,
    #[value(name = "bad100-regret")]
    #[value(alias = "bad-regret-100")]
    Bad100Regret,
    #[value(name = "bad200-regret")]
    #[value(alias = "bad-regret-200")]
    Bad200Regret,
    #[value(name = "max-regret")]
    MaxRegret,
    #[value(name = "capped-selected-regret")]
    CappedSelectedRegret,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum LossMode {
    Pairwise,
    #[value(name = "listwise-leaf")]
    ListwiseLeaf,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ListwiseFeatureSource {
    #[value(name = "teacher-leaf")]
    TeacherLeaf,
    #[value(name = "student-leaf")]
    StudentLeaf,
    Move,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BadCandidateScope {
    #[value(name = "student-top")]
    StudentTop,
    #[value(name = "model-top")]
    ModelTop,
    #[value(name = "all-candidates")]
    AllCandidates,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PairMiningMode {
    First,
    #[value(name = "loss-top")]
    LossTop,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PairWeightMode {
    None,
    #[value(name = "bad-regret")]
    BadRegret,
    #[value(name = "score-gap")]
    ScoreGap,
}

#[derive(Debug, Deserialize)]
struct TreeRecord {
    #[serde(default)]
    schema: Option<String>,
    #[serde(default)]
    version: Option<u8>,
    #[serde(default)]
    sample_weight: Option<f32>,
    #[serde(default)]
    sfen: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    game_teacher_move: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    teacher_root_score: Option<f32>,
    #[serde(default)]
    #[allow(dead_code)]
    student_root_score: Option<f32>,
    #[serde(default)]
    #[allow(dead_code)]
    teacher_best_move: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    student_best_move: Option<String>,
    #[serde(default)]
    candidates: Vec<TreeCandidateRecord>,
}

#[derive(Debug, Deserialize)]
struct TreeCandidateRecord {
    #[serde(rename = "move")]
    #[serde(default)]
    move_usi: String,
    #[serde(default)]
    selected_by_student: bool,
    #[serde(default)]
    #[allow(dead_code)]
    is_game_teacher_move: bool,
    #[serde(default)]
    teacher_score: Option<f32>,
    #[serde(default)]
    teacher_rank: Option<usize>,
    #[serde(default)]
    regret: Option<f32>,
    #[serde(default)]
    student_score: Option<f32>,
    #[serde(default)]
    student_rank: Option<usize>,
    #[serde(default)]
    teacher_leaf_sfen: Option<String>,
    #[serde(default)]
    student_leaf_sfen: Option<String>,
}

#[derive(Debug, Clone)]
struct CandidateSample {
    mv: Move,
    selected_by_student: bool,
    is_game_teacher_move: bool,
    teacher_score: f32,
    teacher_rank: usize,
    student_score: f32,
    student_rank: usize,
    regret: f32,
    move_features: Vec<usize>,
    move_material: f32,
    teacher_leaf: Option<(Vec<usize>, f32)>,
    student_leaf: Option<(Vec<usize>, f32)>,
}

#[derive(Debug, Clone)]
struct Sample {
    #[allow(dead_code)]
    position: Position,
    sample_weight: f32,
    teacher_root_score: f32,
    #[allow(dead_code)]
    student_root_score: f32,
    candidates: Vec<CandidateSample>,
}

#[derive(Clone)]
struct NamedBatch {
    label: String,
    samples: Vec<Sample>,
}

#[derive(Default)]
struct Metrics {
    samples: usize,
    pair_count: usize,
    pair_weight_sum: f32,
    loss: f32,
    selected_regret_sum: f32,
    regrets: Vec<f32>,
    bad_regret_count: usize,
    bad_regret_threshold_counts: Vec<usize>,
}

#[derive(Clone, Copy)]
struct PairOptions {
    teacher_top_k: usize,
    student_bad_top_k: usize,
    bad_candidate_scope: BadCandidateScope,
    min_regret_cp: f32,
    max_pairs_per_sample: usize,
    pair_mining: PairMiningMode,
    pair_weight_mode: PairWeightMode,
    pair_weight_scale_cp: f32,
    max_pair_weight: f32,
}

#[derive(Clone, Copy)]
struct LossOptions {
    mode: LossMode,
    margin_cp: f32,
    softplus_temp_cp: f32,
    model_temperature_cp: f32,
    teacher_temperature_cp: f32,
    listwise_feature_source: ListwiseFeatureSource,
    listwise_hard_negative_weight: f32,
    listwise_hard_negative_min_regret_cp: f32,
    game_teacher_margin_weight: f32,
    game_teacher_max_regret_cp: f32,
    game_teacher_min_bad_regret_cp: f32,
    current_top_margin_weight: f32,
    current_top_min_bad_regret_cp: f32,
}

#[derive(Clone, Copy)]
struct TrainOptions {
    loss: LossOptions,
    l2_lambda: f32,
}

#[derive(Default)]
struct AdagradState {
    w_acc: HashMap<usize, f32>,
    material_acc: f32,
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
        anyhow!("--extra-valid must use LABEL=PATH format (for example: --extra-valid valid=some.jsonl)")
    })?;
    if label.trim().is_empty() {
        return Err(anyhow!("--extra-valid label must not be empty: {spec}"));
    }
    if path.trim().is_empty() {
        return Err(anyhow!("--extra-valid path must not be empty: {spec}"));
    }
    Ok((label.trim().to_string(), PathBuf::from(path.trim())))
}

fn parse_bad_regret_thresholds(raw: &str) -> Result<Vec<f32>> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(anyhow!("--bad-regret-thresholds-cp must not be empty"));
    }

    let mut thresholds = Vec::new();
    for token in raw.split(',') {
        let token = token.trim();
        if token.is_empty() {
            return Err(anyhow!(
                "--bad-regret-thresholds-cp must be comma-separated finite numbers (example: 50,100,200)"
            ));
        }
        let value: f32 = token.parse::<f32>().map_err(|error| {
            anyhow!("--bad-regret-thresholds-cp contains invalid value '{token}': {error}")
        })?;
        if !value.is_finite() {
            return Err(anyhow!(
                "--bad-regret-thresholds-cp contains non-finite value '{token}'"
            ));
        }
        if value < 0.0 {
            return Err(anyhow!(
                "--bad-regret-thresholds-cp contains negative value '{token}'"
            ));
        }
        thresholds.push(value);
    }

    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    thresholds.dedup_by(|a, b| a == b);
    if thresholds.is_empty() {
        return Err(anyhow!(
            "--bad-regret-thresholds-cp must contain at least one threshold"
        ));
    }
    Ok(thresholds)
}

fn bad_regret_threshold_label(threshold: f32) -> String {
    let rendered = format!("{}", threshold);
    if rendered.ends_with(".0") {
        rendered[..rendered.len() - 2].to_string()
    } else {
        rendered
    }
}

fn bad_regret_thresholds_summary(metrics: &Metrics, thresholds: &[f32]) -> String {
    let denom = metrics.samples.max(1) as f32;
    let mut parts = String::new();
    for (i, threshold) in thresholds.iter().enumerate() {
        if i > 0 {
            parts.push(' ');
        }
        let count = metrics
            .bad_regret_threshold_counts
            .get(i)
            .copied()
            .unwrap_or(0);
        let ratio = count as f32 / denom;
        parts.push_str(&format!(
            "bad{}={:.4}",
            bad_regret_threshold_label(*threshold),
            ratio
        ));
    }
    parts
}

fn bad_regret_threshold_ratios(metrics: &Metrics, thresholds: &[f32]) -> Vec<f32> {
    let denom = metrics.samples.max(1) as f32;
    thresholds
        .iter()
        .enumerate()
        .map(|(idx, _)| {
            let count = metrics
                .bad_regret_threshold_counts
                .get(idx)
                .copied()
                .unwrap_or(0);
            count as f32 / denom
        })
        .collect()
}

fn percentile(mut values: Vec<f32>, p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((values.len() - 1) as f32 * p).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

fn listwise_distribution(
    sample: &Sample,
    model: &SparseModel,
    model_temperature_cp: f32,
    teacher_temperature_cp: f32,
    feature_source: ListwiseFeatureSource,
) -> Option<(Vec<f32>, Vec<f32>)> {
    if sample.candidates.is_empty()
        || !model_temperature_cp.is_finite()
        || model_temperature_cp <= 0.0
    {
        return None;
    }
    if !teacher_temperature_cp.is_finite() || teacher_temperature_cp <= 0.0 {
        return None;
    }

    let mut teacher_scores = Vec::with_capacity(sample.candidates.len());
    let mut model_scores = Vec::with_capacity(sample.candidates.len());
    for candidate in &sample.candidates {
        let (features, material) = candidate_listwise_features(candidate, feature_source);
        teacher_scores.push(candidate.teacher_score);
        model_scores.push(model.predict_with_material(features, material));
    }

    let max_teacher = teacher_scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    if !max_teacher.is_finite() {
        return None;
    }
    let max_model = model_scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    if !max_model.is_finite() {
        return None;
    }

    let mut teacher_exp = Vec::with_capacity(sample.candidates.len());
    let mut teacher_total = 0.0f32;
    for score in teacher_scores {
        let exp = ((score - max_teacher) / teacher_temperature_cp).exp();
        if !exp.is_finite() {
            return None;
        }
        teacher_exp.push(exp);
        teacher_total += exp;
    }
    if teacher_total <= 0.0 || !teacher_total.is_finite() {
        return None;
    }

    let mut model_exp = Vec::with_capacity(sample.candidates.len());
    let mut model_total = 0.0f32;
    for score in model_scores {
        let exp = ((score - max_model) / model_temperature_cp).exp();
        if !exp.is_finite() {
            return None;
        }
        model_exp.push(exp);
        model_total += exp;
    }
    if model_total <= 0.0 || !model_total.is_finite() {
        return None;
    }

    for p in &mut teacher_exp {
        *p /= teacher_total;
    }
    for p in &mut model_exp {
        *p /= model_total;
    }
    Some((teacher_exp, model_exp))
}

fn ensure_finite_model(model: &SparseModel) -> Result<()> {
    if !model.w.iter().all(|value| value.is_finite()) || !model.material_coeff.is_finite() {
        return Err(anyhow!("model contains NaN or inf"));
    }
    Ok(())
}

fn copy_model_into(dst: &mut SparseModel, src: &SparseModel) {
    dst.w.clone_from(&src.w);
    dst.bias = src.bias;
    dst.material_coeff = src.material_coeff;
    dst.kpp_eta = src.kpp_eta;
    dst.l2_lambda = src.l2_lambda;
}

fn create_writer(path: &PathBuf) -> Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(BufWriter::new(File::create(path)?))
}

fn features_from_position(position: &Position, mv: Move) -> (Vec<usize>, f32) {
    let mut child = position.clone();
    child.do_move(mv);
    child.switch_turn();
    extract_kpp_features_and_material(&child)
}

fn features_from_sfen(sfen: &str, root_turn: Color) -> Option<(Vec<usize>, f32)> {
    let mut position = position_from_sfen_or_usi(sfen)?;
    if position.side_to_move() != root_turn {
        position.switch_turn();
    }
    Some(extract_kpp_features_and_material(&position))
}

fn candidate_leaf_features<'a>(
    candidate: &'a CandidateSample,
    teacher_preferred: bool,
) -> (&'a Vec<usize>, f32) {
    if teacher_preferred {
        if let Some((features, material)) = candidate.teacher_leaf.as_ref() {
            return (features, *material);
        }
    } else if let Some((features, material)) = candidate.student_leaf.as_ref() {
        return (features, *material);
    }
    (&candidate.move_features, candidate.move_material)
}

fn candidate_listwise_features<'a>(
    candidate: &'a CandidateSample,
    source: ListwiseFeatureSource,
) -> (&'a Vec<usize>, f32) {
    match source {
        ListwiseFeatureSource::TeacherLeaf => {
            if let Some((features, material)) = candidate.teacher_leaf.as_ref() {
                return (features, *material);
            }
            if let Some((features, material)) = candidate.student_leaf.as_ref() {
                return (features, *material);
            }
        }
        ListwiseFeatureSource::StudentLeaf => {
            if let Some((features, material)) = candidate.student_leaf.as_ref() {
                return (features, *material);
            }
            if let Some((features, material)) = candidate.teacher_leaf.as_ref() {
                return (features, *material);
            }
        }
        ListwiseFeatureSource::Move => {}
    }

    (&candidate.move_features, candidate.move_material)
}

fn pair_indices(
    sample: &Sample,
    options: &PairOptions,
    model: Option<&SparseModel>,
    loss_options: &LossOptions,
) -> Vec<(usize, usize)> {
    let model_bad_ranks = if matches!(options.bad_candidate_scope, BadCandidateScope::ModelTop) {
        model.map(|model| {
            let mut indices = (0..sample.candidates.len()).collect::<Vec<_>>();
            indices.sort_by(|&lhs, &rhs| {
                let lhs_candidate = &sample.candidates[lhs];
                let rhs_candidate = &sample.candidates[rhs];
                let lhs_score = model.predict_with_material(
                    &lhs_candidate.move_features,
                    lhs_candidate.move_material,
                );
                let rhs_score = model.predict_with_material(
                    &rhs_candidate.move_features,
                    rhs_candidate.move_material,
                );
                rhs_score
                    .partial_cmp(&lhs_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut ranks = vec![MISSING_RANK; sample.candidates.len()];
            for (rank, idx) in indices.into_iter().enumerate() {
                ranks[idx] = rank;
            }
            ranks
        })
    } else {
        None
    };

    let mut pairs = Vec::new();
    for (good_idx, good) in sample.candidates.iter().enumerate() {
        if good.teacher_rank >= options.teacher_top_k {
            continue;
        }
        for (bad_idx, bad) in sample.candidates.iter().enumerate() {
            if good_idx == bad_idx {
                continue;
            }
            match options.bad_candidate_scope {
                BadCandidateScope::StudentTop => {
                    if bad.student_rank >= options.student_bad_top_k {
                        continue;
                    }
                }
                BadCandidateScope::ModelTop => {
                    let Some(ranks) = model_bad_ranks.as_ref() else {
                        continue;
                    };
                    if ranks[bad_idx] >= options.student_bad_top_k {
                        continue;
                    }
                }
                BadCandidateScope::AllCandidates => {}
            }
            if bad.regret < options.min_regret_cp {
                continue;
            }
            if good.teacher_score <= bad.teacher_score {
                continue;
            }
            pairs.push((good_idx, bad_idx));
            if matches!(options.pair_mining, PairMiningMode::First)
                && pairs.len() >= options.max_pairs_per_sample
            {
                return pairs;
            }
        }
    }
    if matches!(options.pair_mining, PairMiningMode::LossTop) {
        if let Some(model) = model {
            pairs.sort_by(|&(lhs_good, lhs_bad), &(rhs_good, rhs_bad)| {
                let lhs_priority =
                    pair_loss_priority(sample, lhs_good, lhs_bad, options, loss_options, model);
                let rhs_priority =
                    pair_loss_priority(sample, rhs_good, rhs_bad, options, loss_options, model);
                rhs_priority
                    .partial_cmp(&lhs_priority)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        pairs.truncate(options.max_pairs_per_sample);
    }
    pairs
}

fn pair_weight(good: &CandidateSample, bad: &CandidateSample, options: &PairOptions) -> f32 {
    let raw = match options.pair_weight_mode {
        PairWeightMode::None => return 1.0,
        PairWeightMode::BadRegret => bad.regret / options.pair_weight_scale_cp,
        PairWeightMode::ScoreGap => {
            (good.teacher_score - bad.teacher_score).max(0.0) / options.pair_weight_scale_cp
        }
    };
    raw.clamp(1.0, options.max_pair_weight)
}

fn pair_loss_priority(
    sample: &Sample,
    good_idx: usize,
    bad_idx: usize,
    pair_options: &PairOptions,
    loss_options: &LossOptions,
    model: &SparseModel,
) -> f32 {
    let good = &sample.candidates[good_idx];
    let bad = &sample.candidates[bad_idx];
    let (good_features, good_material) = candidate_leaf_features(good, true);
    let (bad_features, bad_material) = candidate_leaf_features(bad, false);
    let diff = model.predict_with_material(good_features, good_material)
        - model.predict_with_material(bad_features, bad_material);
    let x = (loss_options.margin_cp - diff) / loss_options.softplus_temp_cp;
    if x.is_finite() {
        pair_weight(good, bad, pair_options) * loss_options.softplus_temp_cp * softplus(x)
    } else {
        f32::NEG_INFINITY
    }
}

fn current_selected_hard_pair(
    sample: &Sample,
    model: &SparseModel,
    min_regret_cp: f32,
) -> Option<(usize, usize)> {
    if sample.candidates.len() < 2 {
        return None;
    }

    let mut good_idx = 0usize;
    let mut good_score = f32::NEG_INFINITY;
    for (idx, candidate) in sample.candidates.iter().enumerate() {
        if candidate.teacher_score > good_score {
            good_score = candidate.teacher_score;
            good_idx = idx;
        }
    }
    if !good_score.is_finite() {
        return None;
    }

    let mut selected_score = f32::NEG_INFINITY;
    let mut selected_idx = None;
    for (idx, candidate) in sample.candidates.iter().enumerate() {
        if !candidate.selected_by_student {
            continue;
        }
        let score = model.predict_with_material(&candidate.move_features, candidate.move_material);
        if score > selected_score {
            selected_score = score;
            selected_idx = Some(idx);
        }
    }

    if selected_idx.is_none() {
        selected_score = f32::NEG_INFINITY;
        for (idx, candidate) in sample.candidates.iter().enumerate() {
            let score =
                model.predict_with_material(&candidate.move_features, candidate.move_material);
            if score > selected_score {
                selected_score = score;
                selected_idx = Some(idx);
            }
        }
    }

    let Some(selected_idx) = selected_idx else {
        return None;
    };
    let bad = &sample.candidates[selected_idx];
    if !selected_score.is_finite() || selected_idx == good_idx {
        return None;
    }
    let regret = (sample.teacher_root_score - bad.teacher_score).max(0.0);
    if regret < min_regret_cp || good_score <= bad.teacher_score {
        return None;
    }

    Some((good_idx, selected_idx))
}

fn current_game_teacher_hard_pair(
    sample: &Sample,
    model: &SparseModel,
    max_teacher_regret_cp: f32,
    min_bad_regret_cp: f32,
) -> Option<(usize, usize)> {
    if sample.candidates.len() < 2 {
        return None;
    }

    let mut good_idx = None;
    let mut good_score = f32::NEG_INFINITY;
    for (idx, candidate) in sample.candidates.iter().enumerate() {
        if candidate.is_game_teacher_move
            && candidate.regret <= max_teacher_regret_cp
            && candidate.teacher_score > good_score
        {
            good_idx = Some(idx);
            good_score = candidate.teacher_score;
        }
    }
    let Some(good_idx) = good_idx else {
        return None;
    };
    if !good_score.is_finite() {
        return None;
    }

    let mut selected_score = f32::NEG_INFINITY;
    let mut bad_idx = None;
    for (idx, candidate) in sample.candidates.iter().enumerate() {
        if !candidate.selected_by_student {
            continue;
        }
        let score = model.predict_with_material(&candidate.move_features, candidate.move_material);
        if score > selected_score {
            selected_score = score;
            bad_idx = Some(idx);
        }
    }
    if bad_idx.is_none() {
        selected_score = f32::NEG_INFINITY;
        for (idx, candidate) in sample.candidates.iter().enumerate() {
            let score =
                model.predict_with_material(&candidate.move_features, candidate.move_material);
            if score > selected_score {
                selected_score = score;
                bad_idx = Some(idx);
            }
        }
    }
    let Some(bad_idx) = bad_idx else {
        return None;
    };
    if !selected_score.is_finite() || bad_idx == good_idx {
        return None;
    }

    let bad = &sample.candidates[bad_idx];
    let regret = (sample.teacher_root_score - bad.teacher_score).max(0.0);
    if regret < min_bad_regret_cp || good_score <= bad.teacher_score {
        return None;
    }
    Some((good_idx, bad_idx))
}

fn current_model_top_hard_pair(
    sample: &Sample,
    model: &SparseModel,
    feature_source: ListwiseFeatureSource,
    min_bad_regret_cp: f32,
) -> Option<(usize, usize)> {
    if sample.candidates.len() < 2 {
        return None;
    }

    let mut good_idx = 0usize;
    let mut good_score = f32::NEG_INFINITY;
    for (idx, candidate) in sample.candidates.iter().enumerate() {
        if candidate.teacher_score > good_score {
            good_score = candidate.teacher_score;
            good_idx = idx;
        }
    }
    if !good_score.is_finite() {
        return None;
    }

    let mut bad_idx = None;
    let mut bad_model_score = f32::NEG_INFINITY;
    for (idx, candidate) in sample.candidates.iter().enumerate() {
        let (features, material) = candidate_listwise_features(candidate, feature_source);
        let score = model.predict_with_material(features, material);
        if score > bad_model_score {
            bad_model_score = score;
            bad_idx = Some(idx);
        }
    }
    let Some(bad_idx) = bad_idx else {
        return None;
    };
    if !bad_model_score.is_finite() || bad_idx == good_idx {
        return None;
    }

    let bad = &sample.candidates[bad_idx];
    let regret = (sample.teacher_root_score - bad.teacher_score).max(0.0);
    if regret < min_bad_regret_cp || good_score <= bad.teacher_score {
        return None;
    }
    Some((good_idx, bad_idx))
}

fn root_move_pair_loss_and_grad(
    sample: &Sample,
    good_idx: usize,
    bad_idx: usize,
    model: &SparseModel,
    margin_cp: f32,
    softplus_temp_cp: f32,
    feature_source: ListwiseFeatureSource,
    weight: f32,
) -> Option<(f32, f32)> {
    if weight <= 0.0 {
        return None;
    }
    let good = &sample.candidates[good_idx];
    let bad = &sample.candidates[bad_idx];
    let (good_features, good_material) = candidate_listwise_features(good, feature_source);
    let (bad_features, bad_material) = candidate_listwise_features(bad, feature_source);
    let diff = model.predict_with_material(good_features, good_material)
        - model.predict_with_material(bad_features, bad_material);
    let x = (margin_cp - diff) / softplus_temp_cp;
    if !x.is_finite() {
        return None;
    }
    Some((
        weight * softplus_temp_cp * softplus(x),
        -sigmoid(x) * weight,
    ))
}

fn accumulate_sample_metrics(
    sample: &Sample,
    model: &SparseModel,
    bad_regret_cp: f32,
    bad_regret_thresholds: &[f32],
    pair_options: &PairOptions,
    loss_options: &LossOptions,
    metrics: &mut Metrics,
) {
    if sample.candidates.is_empty() {
        return;
    }
    let sample_weight = sample.sample_weight;

    let mut selected_idx = 0usize;
    let mut selected_score = f32::NEG_INFINITY;
    for (idx, candidate) in sample.candidates.iter().enumerate() {
        let model_score =
            model.predict_with_material(&candidate.move_features, candidate.move_material);
        if model_score > selected_score {
            selected_score = model_score;
            selected_idx = idx;
        }
    }

    let selected_regret =
        (sample.teacher_root_score - sample.candidates[selected_idx].teacher_score).max(0.0);
    metrics.samples += 1;
    metrics.selected_regret_sum += selected_regret;
    metrics.regrets.push(selected_regret);
    if selected_regret > bad_regret_cp {
        metrics.bad_regret_count += 1;
    }
    for (idx, threshold) in bad_regret_thresholds.iter().enumerate() {
        if selected_regret > *threshold {
            metrics.bad_regret_threshold_counts[idx] += 1;
        }
    }

    match loss_options.mode {
        LossMode::Pairwise => {
            for (good_idx, bad_idx) in pair_indices(sample, pair_options, Some(model), loss_options)
            {
                let good = &sample.candidates[good_idx];
                let bad = &sample.candidates[bad_idx];
                let (good_features, good_material) = candidate_leaf_features(good, true);
                let (bad_features, bad_material) = candidate_leaf_features(bad, false);
                let diff = model.predict_with_material(good_features, good_material)
                    - model.predict_with_material(bad_features, bad_material);
                let x = (loss_options.margin_cp - diff) / loss_options.softplus_temp_cp;
                if x.is_finite() {
                    let weight = pair_weight(good, bad, pair_options);
                    let scaled_weight = sample_weight * weight;
                    metrics.loss += scaled_weight * loss_options.softplus_temp_cp * softplus(x);
                    metrics.pair_weight_sum += scaled_weight;
                    metrics.pair_count += 1;
                }
            }
        }
        LossMode::ListwiseLeaf => {
            if let Some((teacher_probs, model_probs)) = listwise_distribution(
                sample,
                model,
                loss_options.model_temperature_cp,
                loss_options.teacher_temperature_cp,
                loss_options.listwise_feature_source,
            ) {
                let mut sample_loss = 0.0;
                for (idx, target_prob) in teacher_probs.iter().copied().enumerate() {
                    let student_prob = model_probs[idx];
                    if target_prob > 0.0 {
                        sample_loss -= target_prob * student_prob.max(1e-7).ln();
                    }
                    metrics.pair_count += 1;
                }
                metrics.loss += sample_weight * sample_loss;
                metrics.pair_weight_sum += sample_weight;
            }
            if let Some((good_idx, bad_idx)) = current_selected_hard_pair(
                sample,
                model,
                loss_options.listwise_hard_negative_min_regret_cp,
            ) {
                if let Some((hard_loss, _)) = root_move_pair_loss_and_grad(
                    sample,
                    good_idx,
                    bad_idx,
                    model,
                    loss_options.margin_cp,
                    loss_options.softplus_temp_cp,
                    loss_options.listwise_feature_source,
                    loss_options.listwise_hard_negative_weight,
                ) {
                    let weight = sample_weight * loss_options.listwise_hard_negative_weight;
                    metrics.loss += sample_weight * hard_loss;
                    metrics.pair_weight_sum += weight;
                    metrics.pair_count += 1;
                }
            }
            if loss_options.game_teacher_margin_weight > 0.0 {
                if let Some((good_idx, bad_idx)) = current_game_teacher_hard_pair(
                    sample,
                    model,
                    loss_options.game_teacher_max_regret_cp,
                    loss_options.game_teacher_min_bad_regret_cp,
                ) {
                    if let Some((hard_loss, _)) = root_move_pair_loss_and_grad(
                        sample,
                        good_idx,
                        bad_idx,
                        model,
                        loss_options.margin_cp,
                        loss_options.softplus_temp_cp,
                        loss_options.listwise_feature_source,
                        loss_options.game_teacher_margin_weight,
                    ) {
                        metrics.loss += sample_weight * hard_loss;
                        metrics.pair_weight_sum +=
                            sample_weight * loss_options.game_teacher_margin_weight;
                        metrics.pair_count += 1;
                    }
                }
            }
            if loss_options.current_top_margin_weight > 0.0 {
                if let Some((good_idx, bad_idx)) = current_model_top_hard_pair(
                    sample,
                    model,
                    loss_options.listwise_feature_source,
                    loss_options.current_top_min_bad_regret_cp,
                ) {
                    if let Some((hard_loss, _)) = root_move_pair_loss_and_grad(
                        sample,
                        good_idx,
                        bad_idx,
                        model,
                        loss_options.margin_cp,
                        loss_options.softplus_temp_cp,
                        loss_options.listwise_feature_source,
                        loss_options.current_top_margin_weight,
                    ) {
                        metrics.loss += sample_weight * hard_loss;
                        metrics.pair_weight_sum +=
                            sample_weight * loss_options.current_top_margin_weight;
                        metrics.pair_count += 1;
                    }
                }
            }
        }
    }
}

fn evaluate_batch(
    model: &SparseModel,
    batch: &[Sample],
    bad_regret_cp: f32,
    bad_regret_thresholds: &[f32],
    pair_options: &PairOptions,
    loss_options: &LossOptions,
) -> Metrics {
    let mut metrics = Metrics {
        bad_regret_threshold_counts: vec![0; bad_regret_thresholds.len()],
        ..Metrics::default()
    };
    for sample in batch {
        accumulate_sample_metrics(
            sample,
            model,
            bad_regret_cp,
            bad_regret_thresholds,
            pair_options,
            loss_options,
            &mut metrics,
        );
    }

    if metrics.samples > 0 {
        metrics.selected_regret_sum /= metrics.samples as f32;
    }
    if metrics.pair_weight_sum > 0.0 {
        metrics.loss /= metrics.pair_weight_sum;
    }
    metrics
}

fn update_sample_refs_with_softplus(
    model: &mut SparseModel,
    samples: &[&Sample],
    options: &TrainOptions,
    pair_options: &PairOptions,
    optimizer: OptimizerKind,
    adagrad: &mut AdagradState,
    adagrad_epsilon: f32,
    freeze_material: bool,
) -> (f32, usize) {
    let mut w_grads: HashMap<usize, f32> = HashMap::new();
    let mut material_grad_total = 0.0f32;
    let mut loss = 0.0f32;
    let mut pair_count = 0usize;
    let mut pair_weight_sum = 0.0f32;

    for sample in samples {
        let sample_weight = sample.sample_weight;
        match options.loss.mode {
            LossMode::Pairwise => {
                for (good_idx, bad_idx) in
                    pair_indices(sample, pair_options, Some(model), &options.loss)
                {
                    let good = &sample.candidates[good_idx];
                    let bad = &sample.candidates[bad_idx];
                    let (good_features, good_material) = candidate_leaf_features(good, true);
                    let (bad_features, bad_material) = candidate_leaf_features(bad, false);
                    let diff = model.predict_with_material(good_features, good_material)
                        - model.predict_with_material(bad_features, bad_material);
                    let x = (options.loss.margin_cp - diff) / options.loss.softplus_temp_cp;
                    if !x.is_finite() {
                        continue;
                    }

                    let weight = pair_weight(good, bad, pair_options);
                    let scaled_weight = sample_weight * weight;
                    loss += scaled_weight * options.loss.softplus_temp_cp * softplus(x);
                    let grad_diff = -sigmoid(x) * weight * sample_weight;
                    pair_count += 1;
                    pair_weight_sum += scaled_weight;

                    for &feature_idx in good_features {
                        *w_grads.entry(feature_idx).or_insert(0.0) += grad_diff;
                    }
                    for &feature_idx in bad_features {
                        *w_grads.entry(feature_idx).or_insert(0.0) -= grad_diff;
                    }
                    if !freeze_material {
                        material_grad_total += grad_diff * (good_material - bad_material);
                    }
                }
            }
            LossMode::ListwiseLeaf => {
                if let Some((teacher_probs, model_probs)) = listwise_distribution(
                    sample,
                    model,
                    options.loss.model_temperature_cp,
                    options.loss.teacher_temperature_cp,
                    options.loss.listwise_feature_source,
                ) {
                    let mut sample_loss = 0.0f32;
                    for (idx, (teacher_prob, model_prob)) in
                        teacher_probs.iter().zip(model_probs.iter()).enumerate()
                    {
                        if *teacher_prob > 0.0 {
                            sample_loss -= *teacher_prob * model_prob.max(1e-7).ln();
                        }
                        let delta = (model_prob - teacher_prob) / options.loss.model_temperature_cp;
                        let delta = delta * sample_weight;
                        let (features, material) = candidate_listwise_features(
                            &sample.candidates[idx],
                            options.loss.listwise_feature_source,
                        );
                        for &feature_idx in features {
                            *w_grads.entry(feature_idx).or_insert(0.0) += delta;
                        }
                        if !freeze_material {
                            material_grad_total += delta * material;
                        }
                        pair_count += 1;
                    }
                    loss += sample_weight * sample_loss;
                    pair_weight_sum += sample_weight;
                }
                if let Some((good_idx, bad_idx)) = current_selected_hard_pair(
                    sample,
                    model,
                    options.loss.listwise_hard_negative_min_regret_cp,
                ) {
                    if let Some((hard_loss, grad_diff)) = root_move_pair_loss_and_grad(
                        sample,
                        good_idx,
                        bad_idx,
                        model,
                        options.loss.margin_cp,
                        options.loss.softplus_temp_cp,
                        options.loss.listwise_feature_source,
                        options.loss.listwise_hard_negative_weight,
                    ) {
                        let good = &sample.candidates[good_idx];
                        let bad = &sample.candidates[bad_idx];
                        let (good_features, good_material) =
                            candidate_listwise_features(good, options.loss.listwise_feature_source);
                        let (bad_features, bad_material) =
                            candidate_listwise_features(bad, options.loss.listwise_feature_source);
                        loss += sample_weight * hard_loss;
                        pair_count += 1;
                        pair_weight_sum +=
                            sample_weight * options.loss.listwise_hard_negative_weight;
                        for &feature_idx in good_features {
                            *w_grads.entry(feature_idx).or_insert(0.0) += grad_diff * sample_weight;
                        }
                        for &feature_idx in bad_features {
                            *w_grads.entry(feature_idx).or_insert(0.0) -= grad_diff * sample_weight;
                        }
                        if !freeze_material {
                            material_grad_total +=
                                grad_diff * sample_weight * (good_material - bad_material);
                        }
                    }
                }
                if options.loss.game_teacher_margin_weight > 0.0 {
                    if let Some((good_idx, bad_idx)) = current_game_teacher_hard_pair(
                        sample,
                        model,
                        options.loss.game_teacher_max_regret_cp,
                        options.loss.game_teacher_min_bad_regret_cp,
                    ) {
                        if let Some((hard_loss, grad_diff)) = root_move_pair_loss_and_grad(
                            sample,
                            good_idx,
                            bad_idx,
                            model,
                            options.loss.margin_cp,
                            options.loss.softplus_temp_cp,
                            options.loss.listwise_feature_source,
                            options.loss.game_teacher_margin_weight,
                        ) {
                            let good = &sample.candidates[good_idx];
                            let bad = &sample.candidates[bad_idx];
                            let (good_features, good_material) = candidate_listwise_features(
                                good,
                                options.loss.listwise_feature_source,
                            );
                            let (bad_features, bad_material) = candidate_listwise_features(
                                bad,
                                options.loss.listwise_feature_source,
                            );
                            loss += sample_weight * hard_loss;
                            pair_count += 1;
                            pair_weight_sum +=
                                sample_weight * options.loss.game_teacher_margin_weight;
                            for &feature_idx in good_features {
                                *w_grads.entry(feature_idx).or_insert(0.0) +=
                                    grad_diff * sample_weight;
                            }
                            for &feature_idx in bad_features {
                                *w_grads.entry(feature_idx).or_insert(0.0) -=
                                    grad_diff * sample_weight;
                            }
                            if !freeze_material {
                                material_grad_total +=
                                    grad_diff * sample_weight * (good_material - bad_material);
                            }
                        }
                    }
                }
                if options.loss.current_top_margin_weight > 0.0 {
                    if let Some((good_idx, bad_idx)) = current_model_top_hard_pair(
                        sample,
                        model,
                        options.loss.listwise_feature_source,
                        options.loss.current_top_min_bad_regret_cp,
                    ) {
                        if let Some((hard_loss, grad_diff)) = root_move_pair_loss_and_grad(
                            sample,
                            good_idx,
                            bad_idx,
                            model,
                            options.loss.margin_cp,
                            options.loss.softplus_temp_cp,
                            options.loss.listwise_feature_source,
                            options.loss.current_top_margin_weight,
                        ) {
                            let good = &sample.candidates[good_idx];
                            let bad = &sample.candidates[bad_idx];
                            let (good_features, good_material) = candidate_listwise_features(
                                good,
                                options.loss.listwise_feature_source,
                            );
                            let (bad_features, bad_material) = candidate_listwise_features(
                                bad,
                                options.loss.listwise_feature_source,
                            );
                            loss += sample_weight * hard_loss;
                            pair_count += 1;
                            pair_weight_sum +=
                                sample_weight * options.loss.current_top_margin_weight;
                            for &feature_idx in good_features {
                                *w_grads.entry(feature_idx).or_insert(0.0) +=
                                    grad_diff * sample_weight;
                            }
                            for &feature_idx in bad_features {
                                *w_grads.entry(feature_idx).or_insert(0.0) -=
                                    grad_diff * sample_weight;
                            }
                            if !freeze_material {
                                material_grad_total +=
                                    grad_diff * sample_weight * (good_material - bad_material);
                            }
                        }
                    }
                }
            }
        }
    }

    if pair_count == 0 {
        return (0.0, 0);
    }
    let avg = 1.0 / pair_weight_sum.max(f32::EPSILON);
    let l2 = options.l2_lambda;
    for (idx, grad_sum) in w_grads {
        let grad = grad_sum * avg;
        match optimizer {
            OptimizerKind::Sgd => {
                model.w[idx] -= model.kpp_eta * (grad + l2 * model.w[idx]);
            }
            OptimizerKind::Adagrad => {
                let acc = adagrad.w_acc.entry(idx).or_insert(adagrad_epsilon);
                *acc += grad * grad;
                let denom = acc.sqrt().max(adagrad_epsilon.sqrt());
                let lr = model.kpp_eta / denom;
                model.w[idx] -= lr * (grad + l2 * model.w[idx]);
            }
        }
    }

    if !freeze_material {
        let material_grad = material_grad_total * avg;
        match optimizer {
            OptimizerKind::Sgd => {
                model.material_coeff -= model.kpp_eta * (material_grad + l2 * model.material_coeff);
            }
            OptimizerKind::Adagrad => {
                if adagrad.material_acc < adagrad_epsilon {
                    adagrad.material_acc = adagrad_epsilon;
                }
                adagrad.material_acc += material_grad * material_grad;
                let denom = adagrad.material_acc.sqrt().max(adagrad_epsilon.sqrt());
                let lr = model.kpp_eta / denom;
                model.material_coeff -= lr * (material_grad + l2 * model.material_coeff);
            }
        }
    }

    (loss / pair_weight_sum.max(f32::EPSILON), pair_count)
}

fn parse_candidate(
    line_number: usize,
    raw_candidate: &TreeCandidateRecord,
    position: &Position,
    root_turn: Color,
) -> Result<CandidateSample> {
    let move_usi = raw_candidate.move_usi.trim();
    if move_usi.is_empty() {
        return Err(anyhow!("line {}: empty move string", line_number));
    }
    let mv = parse_move_for_position(position, move_usi)
        .ok_or_else(|| anyhow!("line {}: invalid move {}", line_number, move_usi))?;
    if !position.legal_moves().contains(&mv) {
        return Err(anyhow!(
            "line {}: illegal move {} in root position",
            line_number,
            move_usi
        ));
    }

    let teacher_score = raw_candidate
        .teacher_score
        .filter(|score| score.is_finite())
        .ok_or_else(|| {
            anyhow!(
                "line {}: missing or non-finite teacher_score for {}",
                line_number,
                move_usi
            )
        })?;
    let student_score = raw_candidate
        .student_score
        .and_then(|score| score.is_finite().then_some(score))
        .unwrap_or(f32::NEG_INFINITY);
    let regret = raw_candidate
        .regret
        .filter(|r| r.is_finite())
        .unwrap_or(f32::NAN);
    let (move_features, move_material) = features_from_position(position, mv);
    let teacher_leaf = raw_candidate
        .teacher_leaf_sfen
        .as_deref()
        .and_then(|sfen| features_from_sfen(sfen, root_turn));
    let student_leaf = raw_candidate
        .student_leaf_sfen
        .as_deref()
        .and_then(|sfen| features_from_sfen(sfen, root_turn));

    Ok(CandidateSample {
        mv,
        selected_by_student: raw_candidate.selected_by_student,
        is_game_teacher_move: raw_candidate.is_game_teacher_move,
        teacher_score,
        teacher_rank: raw_candidate.teacher_rank.unwrap_or(MISSING_RANK),
        student_score,
        student_rank: raw_candidate.student_rank.unwrap_or(MISSING_RANK),
        regret,
        move_features,
        move_material,
        teacher_leaf,
        student_leaf,
    })
}

fn dedupe_candidates(mut candidates: Vec<CandidateSample>) -> Vec<CandidateSample> {
    let mut deduped = Vec::new();
    while let Some(candidate) = candidates.pop() {
        if let Some(pos) = deduped
            .iter()
            .position(|item: &CandidateSample| item.mv == candidate.mv)
        {
            if candidate.teacher_score > deduped[pos].teacher_score {
                deduped[pos] = candidate;
            }
        } else {
            deduped.push(candidate);
        }
    }
    deduped
}

fn assign_missing_ranks(candidates: &mut [CandidateSample]) {
    let mut by_teacher: Vec<usize> = (0..candidates.len()).collect();
    by_teacher.sort_by(|&a, &b| {
        candidates[b]
            .teacher_score
            .partial_cmp(&candidates[a].teacher_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut rank = 0usize;
    for idx in by_teacher {
        if candidates[idx].teacher_rank == MISSING_RANK {
            candidates[idx].teacher_rank = rank;
            rank += 1;
        }
    }

    let mut by_student: Vec<usize> = (0..candidates.len()).collect();
    by_student.sort_by(|&a, &b| {
        candidates[b]
            .student_score
            .partial_cmp(&candidates[a].student_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rank = 0;
    for idx in by_student {
        if candidates[idx].student_rank == MISSING_RANK {
            candidates[idx].student_rank = rank;
            rank += 1;
        }
    }
}

fn sample_from_record(path: &PathBuf, line_number: usize, record: TreeRecord) -> Result<Sample> {
    let sfen = record.sfen.ok_or_else(|| {
        anyhow!(
            "{}:{} missing required field `sfen`",
            path.display(),
            line_number
        )
    })?;
    let position = position_from_sfen_or_usi(&sfen)
        .ok_or_else(|| anyhow!("{}:{} invalid sfen: {}", path.display(), line_number, sfen))?;
    let root_turn = position.side_to_move();

    let mut candidates = Vec::with_capacity(record.candidates.len());
    for raw_candidate in &record.candidates {
        candidates.push(parse_candidate(
            line_number,
            raw_candidate,
            &position,
            root_turn,
        )?);
    }
    let sample_weight = record.sample_weight.unwrap_or(1.0);
    if !sample_weight.is_finite() || sample_weight <= 0.0 {
        return Err(anyhow!(
            "{}:{} invalid sample_weight: must be finite and greater than 0",
            path.display(),
            line_number
        ));
    }
    if candidates.is_empty() {
        return Err(anyhow!(
            "{}:{} has no usable candidates",
            path.display(),
            line_number
        ));
    }

    let mut sample = Sample {
        position,
        sample_weight,
        teacher_root_score: record.teacher_root_score.unwrap_or(f32::NAN),
        student_root_score: record.student_root_score.unwrap_or(f32::NAN),
        candidates: dedupe_candidates(candidates),
    };

    if !sample.teacher_root_score.is_finite() {
        sample.teacher_root_score = sample
            .candidates
            .iter()
            .map(|candidate| candidate.teacher_score)
            .fold(f32::NEG_INFINITY, f32::max);
        if !sample.teacher_root_score.is_finite() {
            return Err(anyhow!(
                "{}:{} has no finite teacher_root_score nor teacher score",
                path.display(),
                line_number
            ));
        }
    }

    for candidate in sample.candidates.iter_mut() {
        if !candidate.regret.is_finite() {
            candidate.regret = (sample.teacher_root_score - candidate.teacher_score).max(0.0);
        }
    }
    assign_missing_ranks(&mut sample.candidates);
    Ok(sample)
}

fn for_each_sample_in_file<F>(path: &PathBuf, mut visit: F) -> Result<usize>
where
    F: FnMut(usize, Sample) -> Result<bool>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut sample_count = 0usize;
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        let line_number = line_index + 1;
        if line.trim().is_empty() {
            continue;
        }
        let record: TreeRecord = serde_json::from_str(&line)
            .map_err(|e| anyhow!("{}:{} invalid json: {}", path.display(), line_number, e))?;

        if record
            .schema
            .as_deref()
            .is_some_and(|schema| schema != "mmto_tree_v1")
            || record.version == Some(0)
        {
            continue;
        }

        let sample = sample_from_record(path, line_number, record)?;
        if !visit(sample_count, sample)? {
            break;
        }
        sample_count += 1;
    }

    if sample_count == 0 {
        return Err(anyhow!("{} contains no usable samples", path.display()));
    }
    Ok(sample_count)
}

fn load_samples(path: &PathBuf) -> Result<Vec<Sample>> {
    let mut samples = Vec::new();
    for_each_sample_in_file(path, |_, sample| {
        samples.push(sample);
        Ok(true)
    })?;
    Ok(samples)
}

fn evaluate_streaming_train(
    model: &SparseModel,
    path: &PathBuf,
    max_samples: usize,
    bad_regret_cp: f32,
    bad_regret_thresholds: &[f32],
    pair_options: &PairOptions,
    loss_options: &LossOptions,
) -> Result<Metrics> {
    let mut metrics = Metrics {
        bad_regret_threshold_counts: vec![0; bad_regret_thresholds.len()],
        ..Metrics::default()
    };
    for_each_sample_in_file(path, |sample_index, sample| {
        if max_samples > 0 && sample_index >= max_samples {
            return Ok(false);
        }
        accumulate_sample_metrics(
            &sample,
            model,
            bad_regret_cp,
            bad_regret_thresholds,
            pair_options,
            loss_options,
            &mut metrics,
        );
        Ok(true)
    })?;

    if metrics.pair_weight_sum > 0.0 {
        metrics.loss /= metrics.pair_weight_sum;
    }
    if metrics.samples > 0 {
        metrics.selected_regret_sum /= metrics.samples as f32;
    }
    Ok(metrics)
}

fn evaluate_replay_streaming_train(
    model: &SparseModel,
    paths: &[PathBuf],
    max_samples: usize,
    bad_regret_cp: f32,
    bad_regret_thresholds: &[f32],
    pair_options: &PairOptions,
    loss_options: &LossOptions,
) -> Result<(usize, usize, f32, f32)> {
    let mut total_samples = 0usize;
    let mut total_pairs = 0usize;
    let mut total_pair_weight_sum = 0.0f32;
    let mut weighted_loss_sum = 0.0f32;
    let mut weighted_selected_regret_sum = 0.0f32;
    let mut remaining = max_samples;

    for path in paths {
        let file_limit = if max_samples == 0 { 0 } else { remaining };
        if max_samples > 0 && remaining == 0 {
            break;
        }

        let metrics = evaluate_streaming_train(
            model,
            path,
            file_limit,
            bad_regret_cp,
            bad_regret_thresholds,
            pair_options,
            loss_options,
        )?;
        total_samples += metrics.samples;
        total_pairs += metrics.pair_count;
        total_pair_weight_sum += metrics.pair_weight_sum;
        weighted_loss_sum += metrics.loss * metrics.pair_weight_sum;
        weighted_selected_regret_sum += metrics.selected_regret_sum * metrics.samples as f32;

        if max_samples > 0 {
            remaining = remaining.saturating_sub(metrics.samples);
            if remaining == 0 {
                break;
            }
        }
    }

    let total_loss = if total_pair_weight_sum > 0.0 {
        weighted_loss_sum / total_pair_weight_sum
    } else {
        0.0
    };
    let selected_regret = if total_samples > 0 {
        weighted_selected_regret_sum / total_samples as f32
    } else {
        0.0
    };
    Ok((total_samples, total_pairs, total_loss, selected_regret))
}

fn update_streaming_train_epoch(
    model: &mut SparseModel,
    train_path: &PathBuf,
    batch_size: usize,
    max_samples: usize,
    train_options: &TrainOptions,
    pair_options: &PairOptions,
    optimizer: OptimizerKind,
    adagrad: &mut AdagradState,
    adagrad_epsilon: f32,
    freeze_material: bool,
    initial_material: f32,
    initial_weights: Option<&Vec<f32>>,
    anchor_l2: f32,
    max_weight_delta: Option<f32>,
) -> Result<(usize, usize)> {
    let mut clamped_weights = 0usize;
    let mut batch_samples: Vec<Sample> = Vec::with_capacity(batch_size);
    let mut processed_samples = 0usize;
    let mut process_batch = |batch: &mut Vec<Sample>| -> Result<()> {
        let batch_refs: Vec<&Sample> = batch.iter().collect();
        let _ = update_sample_refs_with_softplus(
            model,
            &batch_refs,
            train_options,
            pair_options,
            optimizer,
            adagrad,
            adagrad_epsilon,
            freeze_material,
        );
        if freeze_material {
            model.material_coeff = initial_material;
        }
        if let Some(initial_weights) = initial_weights {
            if anchor_l2 > 0.0 || max_weight_delta.is_some() {
                clamped_weights +=
                    apply_weight_constraints(model, initial_weights, anchor_l2, max_weight_delta);
            }
        }
        ensure_finite_model(model)?;
        batch.clear();
        Ok(())
    };

    let file = File::open(train_path)?;
    let reader = BufReader::new(file);
    for (line_index, line) in reader.lines().enumerate() {
        if max_samples > 0 && processed_samples >= max_samples {
            break;
        }

        let line = line?;
        let line_number = line_index + 1;
        if line.trim().is_empty() {
            continue;
        }
        let record: TreeRecord = serde_json::from_str(&line).map_err(|e| {
            anyhow!(
                "{}:{} invalid json: {}",
                train_path.display(),
                line_number,
                e
            )
        })?;

        if record
            .schema
            .as_deref()
            .is_some_and(|schema| schema != "mmto_tree_v1")
            || record.version == Some(0)
        {
            continue;
        }

        batch_samples.push(sample_from_record(train_path, line_number, record)?);
        processed_samples += 1;
        if batch_samples.len() == batch_size {
            process_batch(&mut batch_samples)?;
        }
    }

    if processed_samples == 0 {
        return Err(anyhow!(
            "{} contains no usable samples",
            train_path.display()
        ));
    }
    if !batch_samples.is_empty() {
        process_batch(&mut batch_samples)?;
    }
    Ok((processed_samples, clamped_weights))
}

fn log_summary(
    metrics: &Metrics,
    _bad_regret_cp: f32,
    bad_regret_thresholds: &[f32],
    sample_name: &str,
) -> String {
    let bad_ratio = metrics.bad_regret_count as f32 / metrics.samples.max(1) as f32;
    format!(
        "{}: samples={} pairs={} loss={:.6} selected_regret_mean={:.2} p90={:.2} p95={:.2} max={:.2} bad_regret_ratio={:.4} {}",
        sample_name,
        metrics.samples,
        metrics.pair_count,
        metrics.loss,
        metrics.selected_regret_sum,
        percentile(metrics.regrets.clone(), 0.90),
        percentile(metrics.regrets.clone(), 0.95),
        max_regret(metrics),
        bad_ratio,
        bad_regret_thresholds_summary(metrics, bad_regret_thresholds)
    )
}

fn regret_ratio_above(metrics: &Metrics, threshold_cp: f32) -> f32 {
    metrics
        .regrets
        .iter()
        .copied()
        .filter(|regret| *regret > threshold_cp)
        .count() as f32
        / metrics.samples.max(1) as f32
}

fn capped_selected_regret_mean(metrics: &Metrics, cap_cp: f32) -> f32 {
    if metrics.regrets.is_empty() {
        return 0.0;
    }
    let cap = if cap_cp.is_sign_negative() {
        0.0
    } else {
        cap_cp
    };
    let mut total = 0.0;
    for regret in &metrics.regrets {
        total += regret.min(cap);
    }
    total / metrics.regrets.len() as f32
}

fn max_regret(metrics: &Metrics) -> f32 {
    metrics
        .regrets
        .iter()
        .copied()
        .fold(0.0f32, |best, regret| best.max(regret))
}

fn compute_best_metric_value(
    metric: BestMetric,
    valid: &Metrics,
    train: &Metrics,
    selected_regret_cap_cp: f32,
) -> f32 {
    match metric {
        BestMetric::ValidLoss => {
            if valid.samples > 0 {
                valid.loss
            } else if train.samples > 0 {
                train.loss
            } else {
                f32::INFINITY
            }
        }
        BestMetric::SelectedRegret => {
            if valid.samples > 0 {
                valid.selected_regret_sum
            } else if train.samples > 0 {
                train.selected_regret_sum
            } else {
                f32::INFINITY
            }
        }
        BestMetric::BadRegret => {
            if valid.samples > 0 {
                valid.bad_regret_count as f32 / valid.samples.max(1) as f32
            } else if train.samples > 0 {
                train.bad_regret_count as f32 / train.samples.max(1) as f32
            } else {
                f32::INFINITY
            }
        }
        BestMetric::P90Regret => {
            if valid.samples > 0 {
                percentile(valid.regrets.clone(), 0.90)
            } else if train.samples > 0 {
                percentile(train.regrets.clone(), 0.90)
            } else {
                f32::INFINITY
            }
        }
        BestMetric::P95Regret => {
            if valid.samples > 0 {
                percentile(valid.regrets.clone(), 0.95)
            } else if train.samples > 0 {
                percentile(train.regrets.clone(), 0.95)
            } else {
                f32::INFINITY
            }
        }
        BestMetric::P99Regret => {
            if valid.samples > 0 {
                percentile(valid.regrets.clone(), 0.99)
            } else if train.samples > 0 {
                percentile(train.regrets.clone(), 0.99)
            } else {
                f32::INFINITY
            }
        }
        BestMetric::Bad50Regret => {
            if valid.samples > 0 {
                regret_ratio_above(valid, 50.0)
            } else if train.samples > 0 {
                regret_ratio_above(train, 50.0)
            } else {
                f32::INFINITY
            }
        }
        BestMetric::Bad100Regret => {
            if valid.samples > 0 {
                regret_ratio_above(valid, 100.0)
            } else if train.samples > 0 {
                regret_ratio_above(train, 100.0)
            } else {
                f32::INFINITY
            }
        }
        BestMetric::Bad200Regret => {
            if valid.samples > 0 {
                regret_ratio_above(valid, 200.0)
            } else if train.samples > 0 {
                regret_ratio_above(train, 200.0)
            } else {
                f32::INFINITY
            }
        }
        BestMetric::MaxRegret => {
            if valid.samples > 0 {
                max_regret(valid)
            } else if train.samples > 0 {
                max_regret(train)
            } else {
                f32::INFINITY
            }
        }
        BestMetric::CappedSelectedRegret => {
            if valid.samples > 0 {
                capped_selected_regret_mean(valid, selected_regret_cap_cp)
            } else if train.samples > 0 {
                capped_selected_regret_mean(train, selected_regret_cap_cp)
            } else {
                f32::INFINITY
            }
        }
    }
}

fn compute_best_metric_score(
    metric: BestMetric,
    valid: &Metrics,
    train: &Metrics,
    selected_regret_cap_cp: f32,
    extra_valid: &[Metrics],
    extra_valid_best_weight: f32,
) -> f32 {
    let main_metric = compute_best_metric_value(metric, valid, train, selected_regret_cap_cp);
    if extra_valid_best_weight <= 0.0 || extra_valid.is_empty() {
        return main_metric;
    }

    let extra_metric_sum: f32 = extra_valid
        .iter()
        .map(|batch| {
            compute_best_metric_value(metric, batch, &Metrics::default(), selected_regret_cap_cp)
        })
        .sum();
    let extra_metric_mean = extra_metric_sum / extra_valid.len() as f32;
    main_metric + extra_valid_best_weight * extra_metric_mean
}

fn apply_weight_constraints(
    model: &mut SparseModel,
    initial_weights: &[f32],
    anchor_l2: f32,
    max_weight_delta: Option<f32>,
) -> usize {
    let mut clamped = 0usize;
    if initial_weights.len() != model.w.len() {
        return 0;
    }
    for (weight, anchor) in model.w.iter_mut().zip(initial_weights.iter()) {
        if anchor_l2 > 0.0 {
            *weight += anchor_l2 * (*anchor - *weight);
        }
        if let Some(max_delta) = max_weight_delta {
            let delta = *weight - *anchor;
            let clamped_delta = delta.clamp(-max_delta, max_delta);
            if clamped_delta != delta {
                clamped += 1;
            }
            *weight = *anchor + clamped_delta;
        }
    }
    clamped
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.epochs == 0 {
        return Err(anyhow!("--epochs must be greater than 0"));
    }
    if args.batch_size == 0 {
        return Err(anyhow!("--batch-size must be greater than 0"));
    }
    if !args.learning_rate.is_finite() || args.learning_rate <= 0.0 {
        return Err(anyhow!("--learning-rate must be positive"));
    }
    if !args.l2_lambda.is_finite() || args.l2_lambda < 0.0 {
        return Err(anyhow!("--l2-lambda must be non-negative"));
    }
    if !args.margin_cp.is_finite() || args.margin_cp < 0.0 {
        return Err(anyhow!("--margin-cp must be non-negative"));
    }
    if !args.softplus_temp_cp.is_finite() || args.softplus_temp_cp <= 0.0 {
        return Err(anyhow!("--softplus-temp-cp must be positive"));
    }
    if !args.teacher_temperature_cp.is_finite() || args.teacher_temperature_cp <= 0.0 {
        return Err(anyhow!("--teacher-temperature-cp must be positive"));
    }
    if !args.model_temperature_cp.is_finite() || args.model_temperature_cp <= 0.0 {
        return Err(anyhow!("--model-temperature-cp must be positive"));
    }
    if args.teacher_top_k == 0 {
        return Err(anyhow!("--teacher-top-k must be greater than 0"));
    }
    if matches!(
        args.bad_candidate_scope,
        BadCandidateScope::StudentTop | BadCandidateScope::ModelTop
    ) && args.student_bad_top_k == 0
    {
        return Err(anyhow!(
            "--student-bad-top-k must be greater than 0 for student-top or model-top"
        ));
    }
    if !args.min_regret_cp.is_finite() || args.min_regret_cp < 0.0 {
        return Err(anyhow!("--min-regret-cp must be non-negative"));
    }
    if !args.extra_valid_best_weight.is_finite() || args.extra_valid_best_weight < 0.0 {
        return Err(anyhow!(
            "--extra-valid-best-weight must be finite and non-negative"
        ));
    }
    if !args.listwise_hard_negative_weight.is_finite() || args.listwise_hard_negative_weight < 0.0 {
        return Err(anyhow!(
            "--listwise-hard-negative-weight must be finite and non-negative"
        ));
    }
    if !args.listwise_hard_negative_min_regret_cp.is_finite()
        || args.listwise_hard_negative_min_regret_cp < 0.0
    {
        return Err(anyhow!(
            "--listwise-hard-negative-min-regret-cp must be finite and non-negative"
        ));
    }
    if !args.game_teacher_margin_weight.is_finite() || args.game_teacher_margin_weight < 0.0 {
        return Err(anyhow!(
            "--game-teacher-margin-weight must be finite and non-negative"
        ));
    }
    if !args.game_teacher_max_regret_cp.is_finite() || args.game_teacher_max_regret_cp < 0.0 {
        return Err(anyhow!(
            "--game-teacher-max-regret-cp must be finite and non-negative"
        ));
    }
    if !args.game_teacher_min_bad_regret_cp.is_finite() || args.game_teacher_min_bad_regret_cp < 0.0
    {
        return Err(anyhow!(
            "--game-teacher-min-bad-regret-cp must be finite and non-negative"
        ));
    }
    if !args.current_top_margin_weight.is_finite() || args.current_top_margin_weight < 0.0 {
        return Err(anyhow!(
            "--current-top-margin-weight must be finite and non-negative"
        ));
    }
    if !args.current_top_min_bad_regret_cp.is_finite() || args.current_top_min_bad_regret_cp < 0.0 {
        return Err(anyhow!(
            "--current-top-min-bad-regret-cp must be finite and non-negative"
        ));
    }
    if !args.replay_weight.is_finite() || args.replay_weight < 0.0 {
        return Err(anyhow!("--replay-weight must be finite and non-negative"));
    }
    if args.max_pairs_per_sample == 0 {
        return Err(anyhow!("--max-pairs-per-sample must be greater than 0"));
    }
    if !args.pair_weight_scale_cp.is_finite() || args.pair_weight_scale_cp <= 0.0 {
        return Err(anyhow!("--pair-weight-scale-cp must be positive"));
    }
    if !args.max_pair_weight.is_finite() || args.max_pair_weight < 1.0 {
        return Err(anyhow!("--max-pair-weight must be finite and >= 1"));
    }
    if !args.bad_regret_cp.is_finite() || args.bad_regret_cp < 0.0 {
        return Err(anyhow!("--bad-regret-cp must be non-negative"));
    }
    if !args.selected_regret_cap_cp.is_finite() || args.selected_regret_cap_cp < 0.0 {
        return Err(anyhow!("--selected-regret-cap-cp must be non-negative"));
    }
    let bad_regret_thresholds_cp = parse_bad_regret_thresholds(&args.bad_regret_thresholds_cp)?;
    if let Some(max_delta) = args.max_weight_delta {
        if !max_delta.is_finite() || max_delta <= 0.0 {
            return Err(anyhow!("--max-weight-delta must be positive"));
        }
    }
    if args.anchor_l2 < 0.0 || !args.anchor_l2.is_finite() {
        return Err(anyhow!("--anchor-l2 must be non-negative"));
    }
    if matches!(args.optimizer, OptimizerKind::Adagrad) {
        if !args.adagrad_epsilon.is_finite() || args.adagrad_epsilon <= 0.0 {
            return Err(anyhow!(
                "--adagrad-epsilon must be positive when --optimizer adagrad"
            ));
        }
    }

    let pair_options = PairOptions {
        teacher_top_k: args.teacher_top_k,
        student_bad_top_k: args.student_bad_top_k,
        bad_candidate_scope: args.bad_candidate_scope,
        min_regret_cp: args.min_regret_cp,
        max_pairs_per_sample: args.max_pairs_per_sample,
        pair_mining: args.pair_mining,
        pair_weight_mode: args.pair_weight_mode,
        pair_weight_scale_cp: args.pair_weight_scale_cp,
        max_pair_weight: args.max_pair_weight,
    };
    let loss_options = LossOptions {
        mode: args.loss_mode,
        margin_cp: args.margin_cp,
        softplus_temp_cp: args.softplus_temp_cp,
        model_temperature_cp: args.model_temperature_cp,
        teacher_temperature_cp: args.teacher_temperature_cp,
        listwise_feature_source: args.listwise_feature_source,
        listwise_hard_negative_weight: args.listwise_hard_negative_weight,
        listwise_hard_negative_min_regret_cp: args.listwise_hard_negative_min_regret_cp,
        game_teacher_margin_weight: args.game_teacher_margin_weight,
        game_teacher_max_regret_cp: args.game_teacher_max_regret_cp,
        game_teacher_min_bad_regret_cp: args.game_teacher_min_bad_regret_cp,
        current_top_margin_weight: args.current_top_margin_weight,
        current_top_min_bad_regret_cp: args.current_top_min_bad_regret_cp,
    };
    let train_options = TrainOptions {
        loss: loss_options,
        l2_lambda: args.l2_lambda,
    };

    let mut model = SparseModel::new(args.learning_rate, args.l2_lambda);
    model
        .load(&args.weights)
        .map_err(|e| anyhow!("failed to load {}: {}", args.weights.display(), e))?;
    model.kpp_eta = args.learning_rate;
    model.l2_lambda = args.l2_lambda;

    let needs_anchor_weights = args.anchor_l2 > 0.0 || args.max_weight_delta.is_some();
    let initial_weights = if needs_anchor_weights {
        Some(model.w.clone())
    } else {
        None
    };
    let initial_material = model.material_coeff;

    let mut best_model_checkpoint = if args.best_checkpoint_path.is_none() {
        Some(model.clone())
    } else {
        None
    };
    let mut best_checkpoint_written = false;

    let train_samples = if args.stream_train {
        None
    } else {
        Some(load_samples(&args.train)?)
    };
    let valid_samples = load_samples(&args.valid)?;
    let extra_valid = args
        .extra_valid
        .iter()
        .map(|spec| {
            let (label, path) = parse_extra_valid(spec)?;
            let samples = load_samples(&path)?;
            Ok(NamedBatch { label, samples })
        })
        .collect::<Result<Vec<_>>>()?;

    let baseline_train = if args.stream_train {
        evaluate_streaming_train(
            &model,
            &args.train,
            args.stream_train_eval_max_samples,
            args.bad_regret_cp,
            &bad_regret_thresholds_cp,
            &pair_options,
            &loss_options,
        )?
    } else {
        evaluate_batch(
            &model,
            train_samples
                .as_ref()
                .map(Vec::as_slice)
                .expect("non-stream mode requires train samples loaded"),
            args.bad_regret_cp,
            &bad_regret_thresholds_cp,
            &pair_options,
            &loss_options,
        )
    };
    let baseline_valid = evaluate_batch(
        &model,
        &valid_samples,
        args.bad_regret_cp,
        &bad_regret_thresholds_cp,
        &pair_options,
        &loss_options,
    );
    let mut baseline_extra_metrics = Vec::with_capacity(extra_valid.len());

    println!(
        "baseline train: {}",
        log_summary(
            &baseline_train,
            args.bad_regret_cp,
            &bad_regret_thresholds_cp,
            "train",
        )
    );
    println!(
        "baseline valid: {}",
        log_summary(
            &baseline_valid,
            args.bad_regret_cp,
            &bad_regret_thresholds_cp,
            "valid"
        )
    );
    for batch in &extra_valid {
        let metrics = evaluate_batch(
            &model,
            &batch.samples,
            args.bad_regret_cp,
            &bad_regret_thresholds_cp,
            &pair_options,
            &loss_options,
        );
        baseline_extra_metrics.push(metrics);
        let metrics = baseline_extra_metrics
            .last()
            .expect("extra metric should exist");
        println!(
            "baseline extra_valid[{}]: {}",
            batch.label,
            log_summary(
                &metrics,
                args.bad_regret_cp,
                &bad_regret_thresholds_cp,
                "extra_valid"
            )
        );
    }
    let baseline_best_metric_score = compute_best_metric_score(
        args.best_metric,
        &baseline_valid,
        &baseline_train,
        args.selected_regret_cap_cp,
        &baseline_extra_metrics,
        args.extra_valid_best_weight,
    );
    println!(
        "baseline best_metric_score={:.6}",
        baseline_best_metric_score
    );

    if args.dry_run {
        ensure_finite_model(&model)?;
        if args.freeze_material && model.material_coeff != initial_material {
            return Err(anyhow!(
                "material_coeff changed despite --freeze-material: {} -> {}",
                initial_material,
                model.material_coeff
            ));
        }
        return Ok(());
    }

    let mut log_file = if let Some(path) = args.log_path.clone() {
        Some(create_writer(&path)?)
    } else {
        None
    };
    if let Some(file) = log_file.as_mut() {
        let mut header = String::from(
            "epoch,train_loss,train_pairs,train_selected_regret,train_p90,train_p95,train_bad_regret_ratio,train_samples,valid_loss,valid_pairs,valid_selected_regret,valid_p90,valid_p95,valid_bad_regret_ratio,valid_samples,max_abs_delta,p95_abs_delta,clamped_weights,material_coeff",
        );
        for threshold in bad_regret_thresholds_cp.iter() {
            let label = bad_regret_threshold_label(*threshold);
            header.push_str(&format!(",train_bad{label}_ratio"));
        }
        for threshold in bad_regret_thresholds_cp.iter() {
            let label = bad_regret_threshold_label(*threshold);
            header.push_str(&format!(",valid_bad{label}_ratio"));
        }
        writeln!(file, "{}", header)?;
        let write_row = |metrics: &Metrics| {
            let ratios = bad_regret_threshold_ratios(metrics, &bad_regret_thresholds_cp);
            ratios
                .into_iter()
                .map(|ratio| format!(",{ratio:.4}"))
                .collect::<String>()
        };
        let train_threshold_cols = write_row(&baseline_train);
        let valid_threshold_cols = write_row(&baseline_valid);
        writeln!(
            file,
            "0,{:.6},{},{:.2},{:.2},{:.2},{:.4},{},{:.6},{},{:.2},{:.2},{:.2},{:.4},{},{:.6},{:.6},{},{}{}{}",
            baseline_train.loss,
            baseline_train.pair_count,
            baseline_train.selected_regret_sum,
            percentile(baseline_train.regrets.clone(), 0.90),
            percentile(baseline_train.regrets.clone(), 0.95),
            baseline_train.bad_regret_count as f32 / baseline_train.samples.max(1) as f32,
            baseline_train.samples,
            baseline_valid.loss,
            baseline_valid.pair_count,
            baseline_valid.selected_regret_sum,
            percentile(baseline_valid.regrets.clone(), 0.90),
            percentile(baseline_valid.regrets.clone(), 0.95),
            baseline_valid.bad_regret_count as f32 / baseline_valid.samples.max(1) as f32,
            baseline_valid.samples,
            0.0,
            0.0,
            0,
            model.material_coeff,
            train_threshold_cols,
            valid_threshold_cols
        )?;
        file.flush()?;
    }

    let mut optimizer_state = AdagradState {
        w_acc: HashMap::new(),
        material_acc: 0.0,
    };
    let mut best_metric_score = Some(compute_best_metric_score(
        args.best_metric,
        &baseline_valid,
        &baseline_train,
        args.selected_regret_cap_cp,
        &baseline_extra_metrics,
        args.extra_valid_best_weight,
    ));
    let mut best_epoch = 0usize;
    let mut rng = ChaCha8Rng::seed_from_u64(0x1234);
    let mut train_indices: Option<Vec<usize>> = if args.stream_train {
        None
    } else {
        Some((0..train_samples.as_ref().unwrap().len()).collect())
    };
    let replay_enabled = args.replay_weight > 0.0 && !args.replay_train.is_empty();

    for epoch in 1..=args.epochs {
        let mut clamped_weights = 0usize;
        if args.stream_train {
            let (_, clamped) = update_streaming_train_epoch(
                &mut model,
                &args.train,
                args.batch_size,
                0,
                &train_options,
                &pair_options,
                args.optimizer,
                &mut optimizer_state,
                args.adagrad_epsilon,
                args.freeze_material,
                initial_material,
                initial_weights.as_ref(),
                args.anchor_l2,
                args.max_weight_delta,
            )?;
            clamped_weights = clamped;
        } else {
            let train_samples = train_samples.as_ref().unwrap();
            let indices = train_indices.as_mut().unwrap();
            indices.shuffle(&mut rng);

            let mut batch_refs: Vec<&Sample> = Vec::with_capacity(args.batch_size);
            for chunk in indices.chunks(args.batch_size) {
                batch_refs.clear();
                batch_refs.extend(chunk.iter().map(|idx| &train_samples[*idx]));
                let _ = update_sample_refs_with_softplus(
                    &mut model,
                    &batch_refs,
                    &train_options,
                    &pair_options,
                    args.optimizer,
                    &mut optimizer_state,
                    args.adagrad_epsilon,
                    args.freeze_material,
                );
                if args.freeze_material {
                    model.material_coeff = initial_material;
                }
                if let Some(initial_weights) = initial_weights.as_ref() {
                    if args.anchor_l2 > 0.0 || args.max_weight_delta.is_some() {
                        clamped_weights += apply_weight_constraints(
                            &mut model,
                            initial_weights,
                            args.anchor_l2,
                            args.max_weight_delta,
                        );
                    }
                }
                ensure_finite_model(&model)?;
            }
        }

        let mut replay_samples = 0usize;
        let mut replay_pairs = 0usize;
        let mut replay_loss = 0.0f32;
        let mut replay_selected_regret = 0.0f32;
        if replay_enabled {
            let original_learning_rate = model.kpp_eta;
            model.kpp_eta = original_learning_rate * args.replay_weight;
            let mut remaining_replay_samples = args.replay_max_samples;
            for replay_path in args.replay_train.iter() {
                if args.replay_max_samples > 0 && remaining_replay_samples == 0 {
                    break;
                }
                let replay_limit = if args.replay_max_samples == 0 {
                    0
                } else {
                    remaining_replay_samples
                };
                let (used_samples, replay_clamped) = update_streaming_train_epoch(
                    &mut model,
                    replay_path,
                    args.batch_size,
                    replay_limit,
                    &train_options,
                    &pair_options,
                    args.optimizer,
                    &mut optimizer_state,
                    args.adagrad_epsilon,
                    args.freeze_material,
                    initial_material,
                    initial_weights.as_ref(),
                    args.anchor_l2,
                    args.max_weight_delta,
                )?;
                replay_samples += used_samples;
                clamped_weights += replay_clamped;
                if args.replay_max_samples > 0 {
                    remaining_replay_samples =
                        remaining_replay_samples.saturating_sub(used_samples);
                }
            }
            model.kpp_eta = original_learning_rate;
            ensure_finite_model(&model)?;

            if replay_samples > 0 {
                let (samples, pairs, loss, selected_regret) = evaluate_replay_streaming_train(
                    &model,
                    &args.replay_train,
                    args.replay_max_samples,
                    args.bad_regret_cp,
                    &bad_regret_thresholds_cp,
                    &pair_options,
                    &loss_options,
                )?;
                replay_samples = samples;
                replay_pairs = pairs;
                replay_loss = loss;
                replay_selected_regret = selected_regret;
            }
        }

        let train = if args.stream_train {
            evaluate_streaming_train(
                &model,
                &args.train,
                args.stream_train_eval_max_samples,
                args.bad_regret_cp,
                &bad_regret_thresholds_cp,
                &pair_options,
                &loss_options,
            )?
        } else {
            let train_samples = train_samples.as_ref().unwrap();
            evaluate_batch(
                &model,
                train_samples,
                args.bad_regret_cp,
                &bad_regret_thresholds_cp,
                &pair_options,
                &loss_options,
            )
        };
        let valid = evaluate_batch(
            &model,
            &valid_samples,
            args.bad_regret_cp,
            &bad_regret_thresholds_cp,
            &pair_options,
            &loss_options,
        );
        let mut current_epoch_extra_metrics = Vec::with_capacity(extra_valid.len());

        let (max_abs_delta, p95_abs_delta) = if let Some(initial_weights) = initial_weights.as_ref()
        {
            let delta_abs: Vec<f32> = model
                .w
                .iter()
                .zip(initial_weights.iter())
                .map(|(w, anchor)| (w - anchor).abs())
                .collect();
            (
                delta_abs.iter().copied().fold(0.0_f32, f32::max),
                percentile(delta_abs, 0.95),
            )
        } else {
            (0.0, 0.0)
        };
        for batch in &extra_valid {
            let metrics = evaluate_batch(
                &model,
                &batch.samples,
                args.bad_regret_cp,
                &bad_regret_thresholds_cp,
                &pair_options,
                &loss_options,
            );
            current_epoch_extra_metrics.push(metrics);
        }
        let current_metric = compute_best_metric_score(
            args.best_metric,
            &valid,
            &train,
            args.selected_regret_cap_cp,
            &current_epoch_extra_metrics,
            args.extra_valid_best_weight,
        );
        if best_metric_score.is_none()
            || current_metric < best_metric_score.unwrap_or(f32::INFINITY)
        {
            if let Some(path) = &args.best_checkpoint_path {
                model.save(path)?;
                best_checkpoint_written = true;
            } else if let Some(best_model_checkpoint) = best_model_checkpoint.as_mut() {
                copy_model_into(best_model_checkpoint, &model);
            }
            best_metric_score = Some(current_metric);
            best_epoch = epoch;
        }

        println!(
            "epoch {} train: loss={:.6} pairs={} selected_regret={:.2} p90={:.2} p95={:.2} bad_ratio={:.4} {}",
            epoch,
            train.loss,
            train.pair_count,
            train.selected_regret_sum,
            percentile(train.regrets.clone(), 0.90),
            percentile(train.regrets.clone(), 0.95),
            train.bad_regret_count as f32 / train.samples.max(1) as f32,
            bad_regret_thresholds_summary(&train, &bad_regret_thresholds_cp)
        );
        println!(
            "epoch {} valid: loss={:.6} pairs={} selected_regret={:.2} p90={:.2} p95={:.2} bad_ratio={:.4} {} max_abs_delta={:.6} p95_abs_delta={:.6} clamped_weights={}",
            epoch,
            valid.loss,
            valid.pair_count,
            valid.selected_regret_sum,
            percentile(valid.regrets.clone(), 0.90),
            percentile(valid.regrets.clone(), 0.95),
            valid.bad_regret_count as f32 / valid.samples.max(1) as f32,
            bad_regret_thresholds_summary(&valid, &bad_regret_thresholds_cp),
            max_abs_delta,
            p95_abs_delta,
            clamped_weights
        );
        if replay_enabled {
            println!(
                "epoch {} replay: samples={} pairs={} loss={:.6} selected_regret={:.2}",
                epoch, replay_samples, replay_pairs, replay_loss, replay_selected_regret
            );
        }
        println!("epoch {} best_metric_score={:.6}", epoch, current_metric);
        for (batch, metrics) in extra_valid.iter().zip(current_epoch_extra_metrics.iter()) {
            println!(
                "epoch {} extra_valid[{}]: {}",
                epoch,
                batch.label,
                log_summary(
                    metrics,
                    args.bad_regret_cp,
                    &bad_regret_thresholds_cp,
                    "extra_valid"
                )
            );
        }

        if let Some(file) = log_file.as_mut() {
            let write_row = |metrics: &Metrics| {
                let ratios = bad_regret_threshold_ratios(metrics, &bad_regret_thresholds_cp);
                ratios
                    .into_iter()
                    .map(|ratio| format!(",{ratio:.4}"))
                    .collect::<String>()
            };
            let train_threshold_cols = write_row(&train);
            let valid_threshold_cols = write_row(&valid);
            writeln!(
                file,
                "{},{:.6},{},{:.2},{:.2},{:.2},{:.4},{},{:.6},{},{:.2},{:.2},{:.2},{:.4},{},{:.6},{:.6},{},{}{}{}",
                epoch,
                train.loss,
                train.pair_count,
                train.selected_regret_sum,
                percentile(train.regrets.clone(), 0.90),
                percentile(train.regrets.clone(), 0.95),
                train.bad_regret_count as f32 / train.samples.max(1) as f32,
                train.samples,
                valid.loss,
                valid.pair_count,
                valid.selected_regret_sum,
                percentile(valid.regrets.clone(), 0.90),
                percentile(valid.regrets.clone(), 0.95),
                valid.bad_regret_count as f32 / valid.samples.max(1) as f32,
                valid.samples,
                max_abs_delta,
                p95_abs_delta,
                clamped_weights,
                model.material_coeff,
                train_threshold_cols,
                valid_threshold_cols
            )?;
            file.flush()?;
        }
    }

    if let Some(path) = &args.best_checkpoint_path {
        if !best_checkpoint_written {
            fs::copy(&args.weights, path)?;
        }
        println!(
            "Best checkpoint saved to {} (epoch={})",
            path.display(),
            best_epoch
        );
    }

    if let Some(path) = &args.best_checkpoint_path {
        fs::copy(path, &args.output)?;
    } else if let Some(best_model_checkpoint) = best_model_checkpoint.as_ref() {
        best_model_checkpoint.save(&args.output)?;
    } else {
        return Err(anyhow!("internal error: no best model available"));
    }
    if !args.freeze_material {
        println!("saved final model to {}", args.output.display());
    } else {
        println!(
            "saved final model to {} (material restored to frozen value)",
            args.output.display()
        );
    }

    let mut final_model = SparseModel::new(args.learning_rate, args.l2_lambda);
    final_model
        .load(&args.output)
        .map_err(|e| anyhow!("failed to load {}: {}", args.output.display(), e))?;
    ensure_finite_model(&final_model)?;
    if args.freeze_material && final_model.material_coeff != initial_material {
        return Err(anyhow!(
            "material_coeff changed despite --freeze-material: {} -> {}",
            initial_material,
            final_model.material_coeff
        ));
    }
    for batch in &extra_valid {
        let metrics = evaluate_batch(
            &final_model,
            &batch.samples,
            args.bad_regret_cp,
            &bad_regret_thresholds_cp,
            &pair_options,
            &loss_options,
        );
        println!(
            "final extra_valid[{}]: {}",
            batch.label,
            log_summary(
                &metrics,
                args.bad_regret_cp,
                &bad_regret_thresholds_cp,
                "extra_valid"
            )
        );
    }

    println!(
        "best_epoch={} best_value={:.6}",
        best_epoch,
        best_metric_score.unwrap_or(f32::INFINITY)
    );
    Ok(())
}
