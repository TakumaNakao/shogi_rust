use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use shogi_ai::evaluation::{extract_kpp_features_and_material, SparseModel};
use shogi_ai::utils::{parse_usi_move, position_from_sfen_or_usi};
use shogi_core::{Color, Move, Piece};
use shogi_lib::Position;
use std::collections::{HashMap, HashSet};
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
    #[arg(long, default_value_t = -1.0)]
    best_guard_max_regret_increase_cp: f32,
    #[arg(long, default_value_t = -1.0)]
    best_guard_bad100_increase: f32,
    #[arg(long, default_value_t = -1.0)]
    best_guard_teacher_match_drop_pct: f32,
    #[arg(long, default_value_t = -1.0)]
    best_guard_feedback_loss_increase: f32,
    #[arg(long, default_value_t = -1.0)]
    best_guard_feedback_violation_increase: f32,
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
    #[arg(long)]
    feedback_json: Vec<PathBuf>,
    #[arg(long)]
    feedback_guard_json: Vec<PathBuf>,
    #[arg(long, default_value_t = 0.0)]
    feedback_weight: f32,
    #[arg(long, value_enum, default_value_t = FeedbackGoodMove::Baseline)]
    feedback_good_move: FeedbackGoodMove,
    #[arg(long, default_value_t = 0.0)]
    feedback_min_regret_delta_cp: f32,
    #[arg(long, default_value_t = 0.0)]
    feedback_min_candidate_regret_cp: f32,
    #[arg(long, default_value_t = 100.0)]
    feedback_weight_scale_cp: f32,
    #[arg(long, default_value_t = 5.0)]
    feedback_max_sample_weight: f32,
    #[arg(long, default_value_t = 0)]
    feedback_limit: usize,
    #[arg(long, default_value_t = false)]
    feedback_dedupe_sfen: bool,
    #[arg(long)]
    policy_anchor_weights: Option<PathBuf>,
    #[arg(long, default_value_t = 0.0)]
    policy_anchor_weight: f32,
    #[arg(long, default_value_t = 100.0)]
    policy_anchor_temperature_cp: f32,
    #[arg(long, value_enum, default_value_t = ListwiseFeatureSource::Move)]
    policy_anchor_feature_source: ListwiseFeatureSource,
    #[arg(long, default_value_t = 0.0)]
    policy_anchor_margin_weight: f32,
    #[arg(long, default_value_t = 50.0)]
    policy_anchor_margin_cp: f32,
    #[arg(long, default_value_t = 100.0)]
    policy_anchor_margin_softplus_temp_cp: f32,
    #[arg(long, default_value_t = false)]
    separate_aux_adagrad: bool,
    #[arg(long, value_enum, default_value_t = LossMode::Pairwise)]
    loss_mode: LossMode,
    #[arg(long, value_enum, default_value_t = ListwiseFeatureSource::TeacherLeaf)]
    listwise_feature_source: ListwiseFeatureSource,
    #[arg(long, default_value_t = 0.0)]
    listwise_hard_negative_weight: f32,
    #[arg(long, default_value_t = 50.0)]
    listwise_hard_negative_min_regret_cp: f32,
    #[arg(long, default_value_t = 0.0)]
    teacher_top_ce_weight: f32,
    #[arg(long, default_value_t = 0.0)]
    explicit_student_margin_weight: f32,
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
    #[arg(long, default_value_t = 0.0)]
    incumbent_protection_weight: f32,
    #[arg(long, default_value_t = 80.0)]
    incumbent_protection_max_regret_cp: f32,
    #[arg(long, default_value_t = 50.0)]
    incumbent_protection_allow_teacher_better_cp: f32,
    #[arg(long, default_value_t = 0.0)]
    tail_regret_penalty_weight: f32,
    #[arg(long, default_value_t = 50.0)]
    tail_regret_threshold_cp: f32,
    #[arg(long, default_value_t = 100.0)]
    tail_regret_weight_scale_cp: f32,
    #[arg(long, default_value_t = 3.0)]
    tail_regret_max_weight: f32,
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
    #[value(name = "teacher-mismatch")]
    TeacherMismatch,
    #[value(name = "feedback-loss")]
    FeedbackLoss,
    #[value(name = "feedback-violation")]
    FeedbackViolation,
}

fn best_metric_requires_feedback(metric: BestMetric) -> bool {
    matches!(
        metric,
        BestMetric::FeedbackLoss | BestMetric::FeedbackViolation
    )
}

fn best_guard_requires_feedback(args: &Args) -> bool {
    args.best_guard_feedback_loss_increase >= 0.0
        || args.best_guard_feedback_violation_increase >= 0.0
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

#[derive(Clone, Copy, Debug, ValueEnum)]
enum FeedbackGoodMove {
    Baseline,
    Teacher,
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
    ply: Option<u16>,
    #[serde(default)]
    in_check: Option<bool>,
    #[serde(default)]
    legal_moves: Option<usize>,
    #[serde(default)]
    candidates: Vec<TreeCandidateRecord>,
}

#[derive(Debug, Deserialize)]
struct RerankFeedbackReport {
    #[serde(default)]
    hard_positions: Vec<RerankFeedbackRecord>,
}

#[derive(Debug, Deserialize)]
struct RerankFeedbackRecord {
    sfen: String,
    #[serde(default)]
    teacher_best_move: Option<String>,
    #[serde(default)]
    baseline_move: Option<String>,
    #[serde(default)]
    candidate_move: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    baseline_regret: Option<f32>,
    #[serde(default)]
    candidate_regret: Option<f32>,
    #[serde(default)]
    regret_delta: Option<f32>,
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
    ply: u16,
    in_check: bool,
    legal_moves: usize,
    sample_weight: f32,
    teacher_root_score: f32,
    #[allow(dead_code)]
    student_root_score: f32,
    candidates: Vec<CandidateSample>,
}

#[derive(Debug, Clone)]
struct FeedbackSample {
    sample_weight: f32,
    good_features: Vec<usize>,
    good_material: f32,
    bad_features: Vec<usize>,
    bad_material: f32,
    regret_delta: f32,
    candidate_regret: f32,
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
    teacher_match_count: usize,
    bad_regret_count: usize,
    bad_regret_threshold_counts: Vec<usize>,
    buckets: BucketMetrics,
}

#[derive(Default)]
struct FeedbackMetrics {
    samples: usize,
    loss: f32,
    pair_weight_sum: f32,
    violation_count: usize,
    margin_sum: f32,
    regret_delta_sum: f32,
    candidate_regret_sum: f32,
}

#[derive(Default)]
struct PolicyAnchorMetrics {
    samples: usize,
    loss: f32,
    pair_weight_sum: f32,
    top_match_count: usize,
}

#[derive(Default)]
struct PolicyAnchorMarginMetrics {
    samples: usize,
    loss: f32,
    pair_weight_sum: f32,
    top_match_count: usize,
    violation_count: usize,
    margin_sum: f32,
}

#[derive(Clone, Copy, Debug)]
enum PhaseBucket {
    Opening,
    Middle,
    Late,
}

impl PhaseBucket {
    fn label(self) -> &'static str {
        match self {
            PhaseBucket::Opening => "opening",
            PhaseBucket::Middle => "middle",
            PhaseBucket::Late => "late",
        }
    }

    fn index(self) -> usize {
        match self {
            PhaseBucket::Opening => 0,
            PhaseBucket::Middle => 1,
            PhaseBucket::Late => 2,
        }
    }
}

#[derive(Default)]
struct BucketMetric {
    samples: usize,
    regret_sum: f32,
    regrets: Vec<f32>,
    teacher_match_count: usize,
    bad50_count: usize,
    bad100_count: usize,
}

#[derive(Default)]
struct BucketMetrics {
    phases: [BucketMetric; 3],
    in_check: BucketMetric,
    low_legal: BucketMetric,
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
    teacher_top_ce_weight: f32,
    explicit_student_margin_weight: f32,
    game_teacher_margin_weight: f32,
    game_teacher_max_regret_cp: f32,
    game_teacher_min_bad_regret_cp: f32,
    current_top_margin_weight: f32,
    current_top_min_bad_regret_cp: f32,
    incumbent_protection_weight: f32,
    incumbent_protection_max_regret_cp: f32,
    incumbent_protection_allow_teacher_better_cp: f32,
    tail_regret_penalty_weight: f32,
    tail_regret_threshold_cp: f32,
    tail_regret_weight_scale_cp: f32,
    tail_regret_max_weight: f32,
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
) -> Option<(Vec<f32>, Vec<f32>, usize)> {
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
    let mut max_teacher_score = f32::NEG_INFINITY;
    let mut teacher_top_idx = 0usize;
    for (idx, candidate) in sample.candidates.iter().enumerate() {
        let (features, material) = candidate_listwise_features(candidate, feature_source);
        teacher_scores.push(candidate.teacher_score);
        model_scores.push(model.predict_with_material(features, material));
        if candidate.teacher_score > max_teacher_score {
            max_teacher_score = candidate.teacher_score;
            teacher_top_idx = idx;
        }
    }

    if !max_teacher_score.is_finite() {
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
        let exp = ((score - max_teacher_score) / teacher_temperature_cp).exp();
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
    Some((teacher_exp, model_exp, teacher_top_idx))
}

fn model_policy_distribution(
    sample: &Sample,
    model: &SparseModel,
    temperature_cp: f32,
    feature_source: ListwiseFeatureSource,
) -> Option<Vec<f32>> {
    if sample.candidates.is_empty() || !temperature_cp.is_finite() || temperature_cp <= 0.0 {
        return None;
    }

    let mut scores = Vec::with_capacity(sample.candidates.len());
    let mut max_score = f32::NEG_INFINITY;
    for candidate in &sample.candidates {
        let (features, material) = candidate_listwise_features(candidate, feature_source);
        let score = model.predict_with_material(features, material);
        if !score.is_finite() {
            return None;
        }
        max_score = max_score.max(score);
        scores.push(score);
    }
    if !max_score.is_finite() {
        return None;
    }

    let mut total = 0.0f32;
    for score in &mut scores {
        *score = ((*score - max_score) / temperature_cp).exp();
        if !score.is_finite() {
            return None;
        }
        total += *score;
    }
    if total <= 0.0 || !total.is_finite() {
        return None;
    }
    for score in &mut scores {
        *score /= total;
    }
    Some(scores)
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

fn student_selected_idx(sample: &Sample) -> Option<usize> {
    let mut selected_idx = None;
    let mut best_student_rank = MISSING_RANK;
    for (idx, candidate) in sample.candidates.iter().enumerate() {
        if candidate.selected_by_student && candidate.student_rank < best_student_rank {
            selected_idx = Some(idx);
            best_student_rank = candidate.student_rank;
        }
    }
    selected_idx
}

fn teacher_best_idx(sample: &Sample) -> Option<usize> {
    let mut best_idx = None;
    let mut best_score = f32::NEG_INFINITY;
    for (idx, candidate) in sample.candidates.iter().enumerate() {
        if candidate.teacher_score > best_score {
            best_score = candidate.teacher_score;
            best_idx = Some(idx);
        }
    }
    best_idx.filter(|_| best_score.is_finite())
}

fn model_top_idx(
    sample: &Sample,
    model: &SparseModel,
    feature_source: ListwiseFeatureSource,
) -> Option<usize> {
    let mut top_idx = None;
    let mut top_score = f32::NEG_INFINITY;
    for (idx, candidate) in sample.candidates.iter().enumerate() {
        let (features, material) = candidate_listwise_features(candidate, feature_source);
        let score = model.predict_with_material(features, material);
        if score > top_score {
            top_score = score;
            top_idx = Some(idx);
        }
    }
    top_idx.filter(|_| top_score.is_finite())
}

fn incumbent_protection_pair(
    sample: &Sample,
    model: &SparseModel,
    feature_source: ListwiseFeatureSource,
    max_incumbent_regret_cp: f32,
    allow_teacher_better_cp: f32,
) -> Option<(usize, usize)> {
    if sample.candidates.len() < 2 {
        return None;
    }
    let good_idx = student_selected_idx(sample)?;
    let good = &sample.candidates[good_idx];
    if good.regret > max_incumbent_regret_cp {
        return None;
    }
    let bad_idx = model_top_idx(sample, model, feature_source)?;
    if bad_idx == good_idx {
        return None;
    }
    let bad = &sample.candidates[bad_idx];
    if bad.teacher_score > good.teacher_score + allow_teacher_better_cp {
        return None;
    }
    Some((good_idx, bad_idx))
}

fn tail_regret_penalty_pair(
    sample: &Sample,
    model: &SparseModel,
    feature_source: ListwiseFeatureSource,
    threshold_cp: f32,
    weight_scale_cp: f32,
    max_weight: f32,
) -> Option<(usize, usize, f32)> {
    if sample.candidates.len() < 2 {
        return None;
    }
    let good_idx = teacher_best_idx(sample)?;
    let bad_idx = model_top_idx(sample, model, feature_source)?;
    if bad_idx == good_idx {
        return None;
    }
    let good = &sample.candidates[good_idx];
    let bad = &sample.candidates[bad_idx];
    if bad.regret <= threshold_cp || good.teacher_score <= bad.teacher_score {
        return None;
    }
    let excess = (bad.regret - threshold_cp).max(0.0);
    let dynamic_weight = (1.0 + excess / weight_scale_cp.max(f32::EPSILON)).min(max_weight);
    Some((good_idx, bad_idx, dynamic_weight.max(1.0)))
}

fn explicit_student_hard_pair(sample: &Sample) -> Option<(usize, usize)> {
    if sample.candidates.len() < 2 {
        return None;
    }

    let mut good_idx = None;
    let mut good_score = f32::NEG_INFINITY;
    for (idx, candidate) in sample.candidates.iter().enumerate() {
        if candidate.is_game_teacher_move && candidate.teacher_score > good_score {
            good_score = candidate.teacher_score;
            good_idx = Some(idx);
        }
    }
    let Some(good_idx) = good_idx else {
        return None;
    };

    let mut bad_idx = None;
    let mut bad_student_rank = MISSING_RANK;
    for (idx, candidate) in sample.candidates.iter().enumerate() {
        if candidate.selected_by_student && candidate.student_rank < bad_student_rank {
            bad_student_rank = candidate.student_rank;
            bad_idx = Some(idx);
        }
    }
    let Some(bad_idx) = bad_idx else {
        return None;
    };

    if good_idx == bad_idx {
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

fn feedback_loss_and_grad(
    sample: &FeedbackSample,
    model: &SparseModel,
    margin_cp: f32,
    softplus_temp_cp: f32,
) -> Option<(f32, f32, f32)> {
    let good_score = model.predict_with_material(&sample.good_features, sample.good_material);
    let bad_score = model.predict_with_material(&sample.bad_features, sample.bad_material);
    let diff = good_score - bad_score;
    let x = (margin_cp - diff) / softplus_temp_cp;
    if !x.is_finite() {
        return None;
    }
    Some((
        sample.sample_weight * softplus_temp_cp * softplus(x),
        -sigmoid(x) * sample.sample_weight,
        diff,
    ))
}

fn phase_bucket(sample: &Sample) -> PhaseBucket {
    if sample.ply <= 40 {
        PhaseBucket::Opening
    } else if sample.ply <= 90 {
        PhaseBucket::Middle
    } else {
        PhaseBucket::Late
    }
}

fn accumulate_bucket_metric(
    bucket: &mut BucketMetric,
    selected_regret: f32,
    teacher_matched: bool,
) {
    bucket.samples += 1;
    bucket.regret_sum += selected_regret;
    bucket.regrets.push(selected_regret);
    if teacher_matched {
        bucket.teacher_match_count += 1;
    }
    if selected_regret > 50.0 {
        bucket.bad50_count += 1;
    }
    if selected_regret > 100.0 {
        bucket.bad100_count += 1;
    }
}

fn accumulate_bucket_metrics(
    sample: &Sample,
    selected_regret: f32,
    teacher_matched: bool,
    buckets: &mut BucketMetrics,
) {
    accumulate_bucket_metric(
        &mut buckets.phases[phase_bucket(sample).index()],
        selected_regret,
        teacher_matched,
    );
    if sample.in_check {
        accumulate_bucket_metric(&mut buckets.in_check, selected_regret, teacher_matched);
    }
    if sample.legal_moves <= 3 {
        accumulate_bucket_metric(&mut buckets.low_legal, selected_regret, teacher_matched);
    }
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
    let mut teacher_best_idx = 0usize;
    let mut teacher_best_score = f32::NEG_INFINITY;
    for (idx, candidate) in sample.candidates.iter().enumerate() {
        let model_score =
            model.predict_with_material(&candidate.move_features, candidate.move_material);
        if model_score > selected_score {
            selected_score = model_score;
            selected_idx = idx;
        }
        if candidate.teacher_score > teacher_best_score {
            teacher_best_score = candidate.teacher_score;
            teacher_best_idx = idx;
        }
    }

    let selected_regret =
        (sample.teacher_root_score - sample.candidates[selected_idx].teacher_score).max(0.0);
    metrics.samples += 1;
    metrics.selected_regret_sum += selected_regret;
    metrics.regrets.push(selected_regret);
    if selected_idx == teacher_best_idx {
        metrics.teacher_match_count += 1;
    }
    if selected_regret > bad_regret_cp {
        metrics.bad_regret_count += 1;
    }
    for (idx, threshold) in bad_regret_thresholds.iter().enumerate() {
        if selected_regret > *threshold {
            metrics.bad_regret_threshold_counts[idx] += 1;
        }
    }
    accumulate_bucket_metrics(
        sample,
        selected_regret,
        selected_idx == teacher_best_idx,
        &mut metrics.buckets,
    );

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
            if loss_options.explicit_student_margin_weight > 0.0 {
                if let Some((good_idx, bad_idx)) = explicit_student_hard_pair(sample) {
                    if let Some((hard_loss, _)) = root_move_pair_loss_and_grad(
                        sample,
                        good_idx,
                        bad_idx,
                        model,
                        loss_options.margin_cp,
                        loss_options.softplus_temp_cp,
                        loss_options.listwise_feature_source,
                        loss_options.explicit_student_margin_weight,
                    ) {
                        metrics.loss += sample_weight * hard_loss;
                        metrics.pair_weight_sum +=
                            sample_weight * loss_options.explicit_student_margin_weight;
                        metrics.pair_count += 1;
                    }
                }
            }
        }
        LossMode::ListwiseLeaf => {
            if let Some((teacher_probs, model_probs, teacher_top_idx)) = listwise_distribution(
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
                if loss_options.teacher_top_ce_weight > 0.0 {
                    sample_loss -= loss_options.teacher_top_ce_weight
                        * model_probs[teacher_top_idx].max(1e-7).ln();
                }
                metrics.loss += sample_weight * sample_loss;
                metrics.pair_weight_sum += sample_weight;
            }
            if loss_options.explicit_student_margin_weight > 0.0 {
                if let Some((good_idx, bad_idx)) = explicit_student_hard_pair(sample) {
                    if let Some((hard_loss, _)) = root_move_pair_loss_and_grad(
                        sample,
                        good_idx,
                        bad_idx,
                        model,
                        loss_options.margin_cp,
                        loss_options.softplus_temp_cp,
                        loss_options.listwise_feature_source,
                        loss_options.explicit_student_margin_weight,
                    ) {
                        metrics.loss += sample_weight * hard_loss;
                        metrics.pair_weight_sum +=
                            sample_weight * loss_options.explicit_student_margin_weight;
                        metrics.pair_count += 1;
                    }
                }
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
    if loss_options.incumbent_protection_weight > 0.0 {
        if let Some((good_idx, bad_idx)) = incumbent_protection_pair(
            sample,
            model,
            loss_options.listwise_feature_source,
            loss_options.incumbent_protection_max_regret_cp,
            loss_options.incumbent_protection_allow_teacher_better_cp,
        ) {
            if let Some((hard_loss, _)) = root_move_pair_loss_and_grad(
                sample,
                good_idx,
                bad_idx,
                model,
                loss_options.margin_cp,
                loss_options.softplus_temp_cp,
                loss_options.listwise_feature_source,
                loss_options.incumbent_protection_weight,
            ) {
                metrics.loss += sample_weight * hard_loss;
                metrics.pair_weight_sum += sample_weight * loss_options.incumbent_protection_weight;
                metrics.pair_count += 1;
            }
        }
    }
    if loss_options.tail_regret_penalty_weight > 0.0 {
        if let Some((good_idx, bad_idx, dynamic_weight)) = tail_regret_penalty_pair(
            sample,
            model,
            loss_options.listwise_feature_source,
            loss_options.tail_regret_threshold_cp,
            loss_options.tail_regret_weight_scale_cp,
            loss_options.tail_regret_max_weight,
        ) {
            let effective_weight = loss_options.tail_regret_penalty_weight * dynamic_weight;
            if let Some((hard_loss, _)) = root_move_pair_loss_and_grad(
                sample,
                good_idx,
                bad_idx,
                model,
                loss_options.margin_cp,
                loss_options.softplus_temp_cp,
                loss_options.listwise_feature_source,
                effective_weight,
            ) {
                metrics.loss += sample_weight * hard_loss;
                metrics.pair_weight_sum += sample_weight * effective_weight;
                metrics.pair_count += 1;
            }
        }
    }
}

fn accumulate_feedback_metrics(
    sample: &FeedbackSample,
    model: &SparseModel,
    loss_options: &LossOptions,
    metrics: &mut FeedbackMetrics,
) {
    if let Some((loss, _, diff)) = feedback_loss_and_grad(
        sample,
        model,
        loss_options.margin_cp,
        loss_options.softplus_temp_cp,
    ) {
        metrics.samples += 1;
        metrics.loss += loss;
        metrics.pair_weight_sum += sample.sample_weight;
        metrics.margin_sum += diff;
        metrics.regret_delta_sum += sample.regret_delta;
        metrics.candidate_regret_sum += sample.candidate_regret;
        if diff <= 0.0 {
            metrics.violation_count += 1;
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

fn evaluate_feedback_batch(
    model: &SparseModel,
    batch: &[FeedbackSample],
    loss_options: &LossOptions,
) -> FeedbackMetrics {
    let mut metrics = FeedbackMetrics::default();
    for sample in batch {
        accumulate_feedback_metrics(sample, model, loss_options, &mut metrics);
    }
    if metrics.pair_weight_sum > 0.0 {
        metrics.loss /= metrics.pair_weight_sum;
    }
    if metrics.samples > 0 {
        let denom = metrics.samples as f32;
        metrics.margin_sum /= denom;
        metrics.regret_delta_sum /= denom;
        metrics.candidate_regret_sum /= denom;
    }
    metrics
}

fn accumulate_policy_anchor_metrics(
    sample: &Sample,
    model: &SparseModel,
    anchor_model: &SparseModel,
    temperature_cp: f32,
    feature_source: ListwiseFeatureSource,
    metrics: &mut PolicyAnchorMetrics,
) {
    let Some(anchor_probs) =
        model_policy_distribution(sample, anchor_model, temperature_cp, feature_source)
    else {
        return;
    };
    let Some(model_probs) =
        model_policy_distribution(sample, model, temperature_cp, feature_source)
    else {
        return;
    };

    let mut loss = 0.0f32;
    let mut anchor_top_idx = 0usize;
    let mut model_top_idx = 0usize;
    let mut anchor_top_prob = f32::NEG_INFINITY;
    let mut model_top_prob = f32::NEG_INFINITY;
    for (idx, (anchor_prob, model_prob)) in anchor_probs.iter().zip(model_probs.iter()).enumerate()
    {
        if *anchor_prob > 0.0 {
            loss -= *anchor_prob * model_prob.max(1e-7).ln();
        }
        if *anchor_prob > anchor_top_prob {
            anchor_top_prob = *anchor_prob;
            anchor_top_idx = idx;
        }
        if *model_prob > model_top_prob {
            model_top_prob = *model_prob;
            model_top_idx = idx;
        }
    }

    metrics.samples += 1;
    metrics.loss += sample.sample_weight * loss;
    metrics.pair_weight_sum += sample.sample_weight;
    if anchor_top_idx == model_top_idx {
        metrics.top_match_count += 1;
    }
}

fn evaluate_policy_anchor_batch(
    model: &SparseModel,
    anchor_model: &SparseModel,
    batch: &[Sample],
    temperature_cp: f32,
    feature_source: ListwiseFeatureSource,
) -> PolicyAnchorMetrics {
    let mut metrics = PolicyAnchorMetrics::default();
    for sample in batch {
        accumulate_policy_anchor_metrics(
            sample,
            model,
            anchor_model,
            temperature_cp,
            feature_source,
            &mut metrics,
        );
    }
    if metrics.pair_weight_sum > 0.0 {
        metrics.loss /= metrics.pair_weight_sum;
    }
    metrics
}

fn policy_anchor_margin_pair(
    sample: &Sample,
    model: &SparseModel,
    anchor_model: &SparseModel,
    feature_source: ListwiseFeatureSource,
) -> Option<(usize, usize, f32, bool)> {
    if sample.candidates.len() < 2 {
        return None;
    }

    let mut anchor_top_idx = 0usize;
    let mut anchor_top_score = f32::NEG_INFINITY;
    let mut model_top_idx = 0usize;
    let mut model_top_score = f32::NEG_INFINITY;
    let mut model_scores = Vec::with_capacity(sample.candidates.len());

    for (idx, candidate) in sample.candidates.iter().enumerate() {
        let (features, material) = candidate_listwise_features(candidate, feature_source);
        let anchor_score = anchor_model.predict_with_material(features, material);
        let model_score = model.predict_with_material(features, material);
        if !anchor_score.is_finite() || !model_score.is_finite() {
            return None;
        }
        if anchor_score > anchor_top_score {
            anchor_top_score = anchor_score;
            anchor_top_idx = idx;
        }
        if model_score > model_top_score {
            model_top_score = model_score;
            model_top_idx = idx;
        }
        model_scores.push(model_score);
    }

    if !anchor_top_score.is_finite() || !model_top_score.is_finite() {
        return None;
    }

    let top_match = anchor_top_idx == model_top_idx;
    let bad_idx = if top_match {
        let mut runner_up_idx = None;
        let mut runner_up_score = f32::NEG_INFINITY;
        for (idx, score) in model_scores.iter().copied().enumerate() {
            if idx != anchor_top_idx && score > runner_up_score {
                runner_up_score = score;
                runner_up_idx = Some(idx);
            }
        }
        runner_up_idx?
    } else {
        model_top_idx
    };

    let margin = model_scores[anchor_top_idx] - model_scores[bad_idx];
    if !margin.is_finite() {
        return None;
    }
    Some((anchor_top_idx, bad_idx, margin, top_match))
}

fn accumulate_policy_anchor_margin_metrics(
    sample: &Sample,
    model: &SparseModel,
    anchor_model: &SparseModel,
    margin_cp: f32,
    softplus_temp_cp: f32,
    feature_source: ListwiseFeatureSource,
    metrics: &mut PolicyAnchorMarginMetrics,
) {
    let Some((_, _, margin, top_match)) =
        policy_anchor_margin_pair(sample, model, anchor_model, feature_source)
    else {
        return;
    };
    let x = (margin_cp - margin) / softplus_temp_cp;
    if !x.is_finite() {
        return;
    }

    metrics.samples += 1;
    metrics.loss += sample.sample_weight * softplus_temp_cp * softplus(x);
    metrics.pair_weight_sum += sample.sample_weight;
    metrics.margin_sum += margin;
    if top_match {
        metrics.top_match_count += 1;
    }
    if margin < margin_cp {
        metrics.violation_count += 1;
    }
}

fn evaluate_policy_anchor_margin_batch(
    model: &SparseModel,
    anchor_model: &SparseModel,
    batch: &[Sample],
    margin_cp: f32,
    softplus_temp_cp: f32,
    feature_source: ListwiseFeatureSource,
) -> PolicyAnchorMarginMetrics {
    let mut metrics = PolicyAnchorMarginMetrics::default();
    for sample in batch {
        accumulate_policy_anchor_margin_metrics(
            sample,
            model,
            anchor_model,
            margin_cp,
            softplus_temp_cp,
            feature_source,
            &mut metrics,
        );
    }
    if metrics.pair_weight_sum > 0.0 {
        metrics.loss /= metrics.pair_weight_sum;
    }
    if metrics.samples > 0 {
        metrics.margin_sum /= metrics.samples as f32;
    }
    metrics
}

#[allow(clippy::too_many_arguments)]
fn accumulate_root_pair_gradient(
    sample: &Sample,
    model: &SparseModel,
    loss_options: &LossOptions,
    good_idx: usize,
    bad_idx: usize,
    weight: f32,
    sample_weight: f32,
    freeze_material: bool,
    w_grads: &mut HashMap<usize, f32>,
    material_grad_total: &mut f32,
    loss: &mut f32,
    pair_count: &mut usize,
    pair_weight_sum: &mut f32,
) {
    let Some((hard_loss, grad_diff)) = root_move_pair_loss_and_grad(
        sample,
        good_idx,
        bad_idx,
        model,
        loss_options.margin_cp,
        loss_options.softplus_temp_cp,
        loss_options.listwise_feature_source,
        weight,
    ) else {
        return;
    };

    let good = &sample.candidates[good_idx];
    let bad = &sample.candidates[bad_idx];
    let (good_features, good_material) =
        candidate_listwise_features(good, loss_options.listwise_feature_source);
    let (bad_features, bad_material) =
        candidate_listwise_features(bad, loss_options.listwise_feature_source);

    *loss += sample_weight * hard_loss;
    *pair_count += 1;
    *pair_weight_sum += sample_weight * weight;
    for &feature_idx in good_features {
        *w_grads.entry(feature_idx).or_insert(0.0) += grad_diff * sample_weight;
    }
    for &feature_idx in bad_features {
        *w_grads.entry(feature_idx).or_insert(0.0) -= grad_diff * sample_weight;
    }
    if !freeze_material {
        *material_grad_total += grad_diff * sample_weight * (good_material - bad_material);
    }
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
                if options.loss.explicit_student_margin_weight > 0.0 {
                    if let Some((good_idx, bad_idx)) = explicit_student_hard_pair(sample) {
                        if let Some((hard_loss, grad_diff)) = root_move_pair_loss_and_grad(
                            sample,
                            good_idx,
                            bad_idx,
                            model,
                            options.loss.margin_cp,
                            options.loss.softplus_temp_cp,
                            options.loss.listwise_feature_source,
                            options.loss.explicit_student_margin_weight,
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
                                sample_weight * options.loss.explicit_student_margin_weight;
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
            LossMode::ListwiseLeaf => {
                if let Some((teacher_probs, model_probs, teacher_top_idx)) = listwise_distribution(
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
                        let mut delta =
                            (model_prob - teacher_prob) / options.loss.model_temperature_cp;
                        if options.loss.teacher_top_ce_weight > 0.0 {
                            let top_target = (idx == teacher_top_idx) as u8 as f32;
                            delta += options.loss.teacher_top_ce_weight * (model_prob - top_target)
                                / options.loss.model_temperature_cp;
                        }
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
                    if options.loss.teacher_top_ce_weight > 0.0 {
                        sample_loss += options.loss.teacher_top_ce_weight
                            * -model_probs[teacher_top_idx].max(1e-7).ln();
                    }
                    loss += sample_weight * sample_loss;
                    pair_weight_sum += sample_weight;
                }
                if options.loss.explicit_student_margin_weight > 0.0 {
                    if let Some((good_idx, bad_idx)) = explicit_student_hard_pair(sample) {
                        if let Some((hard_loss, grad_diff)) = root_move_pair_loss_and_grad(
                            sample,
                            good_idx,
                            bad_idx,
                            model,
                            options.loss.margin_cp,
                            options.loss.softplus_temp_cp,
                            options.loss.listwise_feature_source,
                            options.loss.explicit_student_margin_weight,
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
                                sample_weight * options.loss.explicit_student_margin_weight;
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
        if options.loss.incumbent_protection_weight > 0.0 {
            if let Some((good_idx, bad_idx)) = incumbent_protection_pair(
                sample,
                model,
                options.loss.listwise_feature_source,
                options.loss.incumbent_protection_max_regret_cp,
                options.loss.incumbent_protection_allow_teacher_better_cp,
            ) {
                accumulate_root_pair_gradient(
                    sample,
                    model,
                    &options.loss,
                    good_idx,
                    bad_idx,
                    options.loss.incumbent_protection_weight,
                    sample_weight,
                    freeze_material,
                    &mut w_grads,
                    &mut material_grad_total,
                    &mut loss,
                    &mut pair_count,
                    &mut pair_weight_sum,
                );
            }
        }
        if options.loss.tail_regret_penalty_weight > 0.0 {
            if let Some((good_idx, bad_idx, dynamic_weight)) = tail_regret_penalty_pair(
                sample,
                model,
                options.loss.listwise_feature_source,
                options.loss.tail_regret_threshold_cp,
                options.loss.tail_regret_weight_scale_cp,
                options.loss.tail_regret_max_weight,
            ) {
                accumulate_root_pair_gradient(
                    sample,
                    model,
                    &options.loss,
                    good_idx,
                    bad_idx,
                    options.loss.tail_regret_penalty_weight * dynamic_weight,
                    sample_weight,
                    freeze_material,
                    &mut w_grads,
                    &mut material_grad_total,
                    &mut loss,
                    &mut pair_count,
                    &mut pair_weight_sum,
                );
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

fn update_feedback_refs_with_softplus(
    model: &mut SparseModel,
    samples: &[&FeedbackSample],
    loss_options: &LossOptions,
    l2_lambda: f32,
    optimizer: OptimizerKind,
    adagrad: &mut AdagradState,
    adagrad_epsilon: f32,
    freeze_material: bool,
) -> (f32, usize) {
    let mut w_grads: HashMap<usize, f32> = HashMap::new();
    let mut material_grad_total = 0.0f32;
    let mut loss = 0.0f32;
    let mut pair_weight_sum = 0.0f32;
    let mut pair_count = 0usize;

    for sample in samples {
        let Some((pair_loss, grad_diff, _)) = feedback_loss_and_grad(
            sample,
            model,
            loss_options.margin_cp,
            loss_options.softplus_temp_cp,
        ) else {
            continue;
        };
        loss += pair_loss;
        pair_weight_sum += sample.sample_weight;
        pair_count += 1;

        for &feature_idx in &sample.good_features {
            *w_grads.entry(feature_idx).or_insert(0.0) += grad_diff;
        }
        for &feature_idx in &sample.bad_features {
            *w_grads.entry(feature_idx).or_insert(0.0) -= grad_diff;
        }
        if !freeze_material {
            material_grad_total += grad_diff * (sample.good_material - sample.bad_material);
        }
    }

    if pair_count == 0 {
        return (0.0, 0);
    }
    let avg = 1.0 / pair_weight_sum.max(f32::EPSILON);
    for (idx, grad_sum) in w_grads {
        let grad = grad_sum * avg;
        match optimizer {
            OptimizerKind::Sgd => {
                model.w[idx] -= model.kpp_eta * (grad + l2_lambda * model.w[idx]);
            }
            OptimizerKind::Adagrad => {
                let acc = adagrad.w_acc.entry(idx).or_insert(adagrad_epsilon);
                *acc += grad * grad;
                let denom = acc.sqrt().max(adagrad_epsilon.sqrt());
                let lr = model.kpp_eta / denom;
                model.w[idx] -= lr * (grad + l2_lambda * model.w[idx]);
            }
        }
    }

    if !freeze_material {
        let material_grad = material_grad_total * avg;
        match optimizer {
            OptimizerKind::Sgd => {
                model.material_coeff -=
                    model.kpp_eta * (material_grad + l2_lambda * model.material_coeff);
            }
            OptimizerKind::Adagrad => {
                if adagrad.material_acc < adagrad_epsilon {
                    adagrad.material_acc = adagrad_epsilon;
                }
                adagrad.material_acc += material_grad * material_grad;
                let denom = adagrad.material_acc.sqrt().max(adagrad_epsilon.sqrt());
                let lr = model.kpp_eta / denom;
                model.material_coeff -= lr * (material_grad + l2_lambda * model.material_coeff);
            }
        }
    }

    (loss / pair_weight_sum.max(f32::EPSILON), pair_count)
}

fn update_policy_anchor_refs(
    model: &mut SparseModel,
    anchor_model: &SparseModel,
    samples: &[&Sample],
    temperature_cp: f32,
    feature_source: ListwiseFeatureSource,
    l2_lambda: f32,
    optimizer: OptimizerKind,
    adagrad: &mut AdagradState,
    adagrad_epsilon: f32,
    freeze_material: bool,
) -> (f32, usize) {
    let mut w_grads: HashMap<usize, f32> = HashMap::new();
    let mut material_grad_total = 0.0f32;
    let mut loss = 0.0f32;
    let mut pair_weight_sum = 0.0f32;
    let mut pair_count = 0usize;

    for sample in samples {
        let Some(anchor_probs) =
            model_policy_distribution(sample, anchor_model, temperature_cp, feature_source)
        else {
            continue;
        };
        let Some(model_probs) =
            model_policy_distribution(sample, model, temperature_cp, feature_source)
        else {
            continue;
        };

        let mut sample_loss = 0.0f32;
        for (idx, (anchor_prob, model_prob)) in
            anchor_probs.iter().zip(model_probs.iter()).enumerate()
        {
            if *anchor_prob > 0.0 {
                sample_loss -= *anchor_prob * model_prob.max(1e-7).ln();
            }
            let delta = (*model_prob - *anchor_prob) / temperature_cp * sample.sample_weight;
            let (features, material) =
                candidate_listwise_features(&sample.candidates[idx], feature_source);
            for &feature_idx in features {
                *w_grads.entry(feature_idx).or_insert(0.0) += delta;
            }
            if !freeze_material {
                material_grad_total += delta * material;
            }
            pair_count += 1;
        }
        loss += sample.sample_weight * sample_loss;
        pair_weight_sum += sample.sample_weight;
    }

    if pair_count == 0 {
        return (0.0, 0);
    }
    let avg = 1.0 / pair_weight_sum.max(f32::EPSILON);
    for (idx, grad_sum) in w_grads {
        let grad = grad_sum * avg;
        match optimizer {
            OptimizerKind::Sgd => {
                model.w[idx] -= model.kpp_eta * (grad + l2_lambda * model.w[idx]);
            }
            OptimizerKind::Adagrad => {
                let acc = adagrad.w_acc.entry(idx).or_insert(adagrad_epsilon);
                *acc += grad * grad;
                let denom = acc.sqrt().max(adagrad_epsilon.sqrt());
                let lr = model.kpp_eta / denom;
                model.w[idx] -= lr * (grad + l2_lambda * model.w[idx]);
            }
        }
    }

    if !freeze_material {
        let material_grad = material_grad_total * avg;
        match optimizer {
            OptimizerKind::Sgd => {
                model.material_coeff -=
                    model.kpp_eta * (material_grad + l2_lambda * model.material_coeff);
            }
            OptimizerKind::Adagrad => {
                if adagrad.material_acc < adagrad_epsilon {
                    adagrad.material_acc = adagrad_epsilon;
                }
                adagrad.material_acc += material_grad * material_grad;
                let denom = adagrad.material_acc.sqrt().max(adagrad_epsilon.sqrt());
                let lr = model.kpp_eta / denom;
                model.material_coeff -= lr * (material_grad + l2_lambda * model.material_coeff);
            }
        }
    }

    (loss / pair_weight_sum.max(f32::EPSILON), pair_count)
}

fn update_policy_anchor_margin_refs(
    model: &mut SparseModel,
    anchor_model: &SparseModel,
    samples: &[&Sample],
    margin_cp: f32,
    softplus_temp_cp: f32,
    feature_source: ListwiseFeatureSource,
    l2_lambda: f32,
    optimizer: OptimizerKind,
    adagrad: &mut AdagradState,
    adagrad_epsilon: f32,
    freeze_material: bool,
) -> (f32, usize) {
    let mut w_grads: HashMap<usize, f32> = HashMap::new();
    let mut material_grad_total = 0.0f32;
    let mut loss = 0.0f32;
    let mut pair_weight_sum = 0.0f32;
    let mut pair_count = 0usize;

    for sample in samples {
        let Some((good_idx, bad_idx, margin, _)) =
            policy_anchor_margin_pair(sample, model, anchor_model, feature_source)
        else {
            continue;
        };
        let x = (margin_cp - margin) / softplus_temp_cp;
        if !x.is_finite() {
            continue;
        }

        let scaled_weight = sample.sample_weight;
        loss += scaled_weight * softplus_temp_cp * softplus(x);
        let grad_diff = -sigmoid(x) * scaled_weight;
        pair_count += 1;
        pair_weight_sum += scaled_weight;

        let good = &sample.candidates[good_idx];
        let bad = &sample.candidates[bad_idx];
        let (good_features, good_material) = candidate_listwise_features(good, feature_source);
        let (bad_features, bad_material) = candidate_listwise_features(bad, feature_source);
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

    if pair_count == 0 {
        return (0.0, 0);
    }
    let avg = 1.0 / pair_weight_sum.max(f32::EPSILON);
    for (idx, grad_sum) in w_grads {
        let grad = grad_sum * avg;
        match optimizer {
            OptimizerKind::Sgd => {
                model.w[idx] -= model.kpp_eta * (grad + l2_lambda * model.w[idx]);
            }
            OptimizerKind::Adagrad => {
                let acc = adagrad.w_acc.entry(idx).or_insert(adagrad_epsilon);
                *acc += grad * grad;
                let denom = acc.sqrt().max(adagrad_epsilon.sqrt());
                let lr = model.kpp_eta / denom;
                model.w[idx] -= lr * (grad + l2_lambda * model.w[idx]);
            }
        }
    }

    if !freeze_material {
        let material_grad = material_grad_total * avg;
        match optimizer {
            OptimizerKind::Sgd => {
                model.material_coeff -=
                    model.kpp_eta * (material_grad + l2_lambda * model.material_coeff);
            }
            OptimizerKind::Adagrad => {
                if adagrad.material_acc < adagrad_epsilon {
                    adagrad.material_acc = adagrad_epsilon;
                }
                adagrad.material_acc += material_grad * material_grad;
                let denom = adagrad.material_acc.sqrt().max(adagrad_epsilon.sqrt());
                let lr = model.kpp_eta / denom;
                model.material_coeff -= lr * (material_grad + l2_lambda * model.material_coeff);
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
    let ply = record.ply.unwrap_or_else(|| position.ply());
    let in_check = record.in_check.unwrap_or_else(|| position.in_check());
    let legal_moves = record
        .legal_moves
        .unwrap_or_else(|| position.legal_moves().len());

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
        ply,
        in_check,
        legal_moves,
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

fn finite_or(value: Option<f32>, fallback: f32) -> f32 {
    value.filter(|value| value.is_finite()).unwrap_or(fallback)
}

fn feedback_sample_from_record(
    path: &PathBuf,
    record_index: usize,
    record: &RerankFeedbackRecord,
    good_move: FeedbackGoodMove,
    min_regret_delta_cp: f32,
    min_candidate_regret_cp: f32,
    weight_scale_cp: f32,
    max_sample_weight: f32,
) -> Result<Option<FeedbackSample>> {
    let regret_delta = finite_or(record.regret_delta, f32::NEG_INFINITY);
    let candidate_regret = finite_or(record.candidate_regret, f32::NEG_INFINITY);
    if regret_delta < min_regret_delta_cp || candidate_regret < min_candidate_regret_cp {
        return Ok(None);
    }

    let candidate_move_text = record
        .candidate_move
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let Some(candidate_move_text) = candidate_move_text else {
        return Ok(None);
    };

    let good_move_text = match good_move {
        FeedbackGoodMove::Baseline => record
            .baseline_move
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .or_else(|| {
                record
                    .teacher_best_move
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
            }),
        FeedbackGoodMove::Teacher => record
            .teacher_best_move
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .or_else(|| {
                record
                    .baseline_move
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
            }),
    };
    let Some(good_move_text) = good_move_text else {
        return Ok(None);
    };
    if good_move_text == candidate_move_text {
        return Ok(None);
    }

    let position = position_from_sfen_or_usi(&record.sfen).ok_or_else(|| {
        anyhow!(
            "{} hard_positions[{}] invalid sfen: {}",
            path.display(),
            record_index,
            record.sfen
        )
    })?;
    let good_move = parse_move_for_position(&position, good_move_text).ok_or_else(|| {
        anyhow!(
            "{} hard_positions[{}] invalid good move {}",
            path.display(),
            record_index,
            good_move_text
        )
    })?;
    let bad_move = parse_move_for_position(&position, candidate_move_text).ok_or_else(|| {
        anyhow!(
            "{} hard_positions[{}] invalid candidate move {}",
            path.display(),
            record_index,
            candidate_move_text
        )
    })?;
    let legal_moves = position.legal_moves();
    if !legal_moves.contains(&good_move) || !legal_moves.contains(&bad_move) {
        return Ok(None);
    }

    let signal = regret_delta.max(candidate_regret).max(0.0);
    let sample_weight = (1.0 + signal / weight_scale_cp).clamp(1.0, max_sample_weight);
    let (good_features, good_material) = features_from_position(&position, good_move);
    let (bad_features, bad_material) = features_from_position(&position, bad_move);
    Ok(Some(FeedbackSample {
        sample_weight,
        good_features,
        good_material,
        bad_features,
        bad_material,
        regret_delta,
        candidate_regret,
    }))
}

fn load_feedback_samples(
    paths: &[PathBuf],
    good_move: FeedbackGoodMove,
    min_regret_delta_cp: f32,
    min_candidate_regret_cp: f32,
    weight_scale_cp: f32,
    max_sample_weight: f32,
    limit: usize,
    dedupe_sfen: bool,
) -> Result<Vec<FeedbackSample>> {
    let mut samples = Vec::new();
    let mut seen_sfens = HashSet::new();
    for path in paths {
        let reader = BufReader::new(File::open(path)?);
        let report: RerankFeedbackReport = serde_json::from_reader(reader)
            .map_err(|e| anyhow!("{} invalid feedback json: {}", path.display(), e))?;
        for (idx, record) in report.hard_positions.iter().enumerate() {
            if limit > 0 && samples.len() >= limit {
                return Ok(samples);
            }
            let sfen = record.sfen.trim();
            if dedupe_sfen && !sfen.is_empty() && seen_sfens.contains(sfen) {
                continue;
            }
            if let Some(sample) = feedback_sample_from_record(
                path,
                idx,
                record,
                good_move,
                min_regret_delta_cp,
                min_candidate_regret_cp,
                weight_scale_cp,
                max_sample_weight,
            )? {
                if dedupe_sfen && !sfen.is_empty() {
                    seen_sfens.insert(sfen.to_owned());
                }
                samples.push(sample);
            }
        }
    }
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

fn update_streaming_policy_anchor_epoch(
    model: &mut SparseModel,
    anchor_model: &SparseModel,
    train_path: &PathBuf,
    batch_size: usize,
    max_samples: usize,
    temperature_cp: f32,
    feature_source: ListwiseFeatureSource,
    l2_lambda: f32,
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
        let _ = update_policy_anchor_refs(
            model,
            anchor_model,
            &batch_refs,
            temperature_cp,
            feature_source,
            l2_lambda,
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

fn update_streaming_policy_anchor_margin_epoch(
    model: &mut SparseModel,
    anchor_model: &SparseModel,
    train_path: &PathBuf,
    batch_size: usize,
    max_samples: usize,
    margin_cp: f32,
    softplus_temp_cp: f32,
    feature_source: ListwiseFeatureSource,
    l2_lambda: f32,
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
        let _ = update_policy_anchor_margin_refs(
            model,
            anchor_model,
            &batch_refs,
            margin_cp,
            softplus_temp_cp,
            feature_source,
            l2_lambda,
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

    for_each_sample_in_file(train_path, |sample_index, sample| {
        if max_samples > 0 && sample_index >= max_samples {
            return Ok(false);
        }
        processed_samples += 1;
        batch_samples.push(sample);
        if batch_samples.len() >= batch_size {
            process_batch(&mut batch_samples)?;
        }
        Ok(true)
    })?;
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
    let teacher_match = teacher_match_rate(metrics);
    format!(
        "{}: samples={} pairs={} loss={:.6} selected_regret_mean={:.2} p90={:.2} p95={:.2} max={:.2} teacher_match={:.2}% bad_regret_ratio={:.4} {}",
        sample_name,
        metrics.samples,
        metrics.pair_count,
        metrics.loss,
        metrics.selected_regret_sum,
        percentile(metrics.regrets.clone(), 0.90),
        percentile(metrics.regrets.clone(), 0.95),
        max_regret(metrics),
        teacher_match * 100.0,
        bad_ratio,
        bad_regret_thresholds_summary(metrics, bad_regret_thresholds)
    )
}

fn bucket_metric_summary(bucket: &BucketMetric) -> String {
    if bucket.samples == 0 {
        return "samples=0".to_string();
    }
    let samples = bucket.samples.max(1) as f32;
    format!(
        "samples={} mean={:.2} p95={:.2} match={:.2}% bad50={:.4} bad100={:.4}",
        bucket.samples,
        bucket.regret_sum / samples,
        percentile(bucket.regrets.clone(), 0.95),
        bucket.teacher_match_count as f32 / samples * 100.0,
        bucket.bad50_count as f32 / samples,
        bucket.bad100_count as f32 / samples,
    )
}

fn log_bucket_summary(metrics: &Metrics) -> String {
    let phases = [PhaseBucket::Opening, PhaseBucket::Middle, PhaseBucket::Late]
        .into_iter()
        .map(|phase| {
            format!(
                "{}: {}",
                phase.label(),
                bucket_metric_summary(&metrics.buckets.phases[phase.index()])
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        "buckets phase=[{}] in_check=[{}] low_legal=[{}]",
        phases,
        bucket_metric_summary(&metrics.buckets.in_check),
        bucket_metric_summary(&metrics.buckets.low_legal)
    )
}

fn log_feedback_summary(metrics: &FeedbackMetrics, sample_name: &str) -> String {
    format!(
        "{}: samples={} loss={:.6} margin_mean={:.2} violation_ratio={:.4} regret_delta_mean={:.2} candidate_regret_mean={:.2}",
        sample_name,
        metrics.samples,
        metrics.loss,
        metrics.margin_sum,
        feedback_violation_rate(metrics),
        metrics.regret_delta_sum,
        metrics.candidate_regret_sum,
    )
}

fn log_policy_anchor_summary(metrics: &PolicyAnchorMetrics, sample_name: &str) -> String {
    let top_match = metrics.top_match_count as f32 / metrics.samples.max(1) as f32;
    format!(
        "{}: samples={} loss={:.6} top_match={:.2}%",
        sample_name,
        metrics.samples,
        metrics.loss,
        top_match * 100.0,
    )
}

fn log_policy_anchor_margin_summary(
    metrics: &PolicyAnchorMarginMetrics,
    sample_name: &str,
) -> String {
    let top_match = metrics.top_match_count as f32 / metrics.samples.max(1) as f32;
    let violation_ratio = metrics.violation_count as f32 / metrics.samples.max(1) as f32;
    format!(
        "{}: samples={} loss={:.6} margin_mean={:.2} violation_ratio={:.4} top_match={:.2}%",
        sample_name,
        metrics.samples,
        metrics.loss,
        metrics.margin_sum,
        violation_ratio,
        top_match * 100.0,
    )
}

fn teacher_match_rate(metrics: &Metrics) -> f32 {
    metrics.teacher_match_count as f32 / metrics.samples.max(1) as f32
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

fn feedback_violation_rate(metrics: &FeedbackMetrics) -> f32 {
    metrics.violation_count as f32 / metrics.samples.max(1) as f32
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
        BestMetric::TeacherMismatch => {
            if valid.samples > 0 {
                1.0 - teacher_match_rate(valid)
            } else if train.samples > 0 {
                1.0 - teacher_match_rate(train)
            } else {
                f32::INFINITY
            }
        }
        BestMetric::FeedbackLoss | BestMetric::FeedbackViolation => f32::INFINITY,
    }
}

fn compute_best_metric_score(
    metric: BestMetric,
    valid: &Metrics,
    train: &Metrics,
    selected_regret_cap_cp: f32,
    extra_valid: &[Metrics],
    extra_valid_best_weight: f32,
    feedback: Option<&FeedbackMetrics>,
) -> f32 {
    match metric {
        BestMetric::FeedbackLoss => {
            return feedback
                .filter(|metrics| metrics.samples > 0)
                .map(|metrics| metrics.loss)
                .unwrap_or(f32::INFINITY);
        }
        BestMetric::FeedbackViolation => {
            return feedback
                .filter(|metrics| metrics.samples > 0)
                .map(|metrics| metrics.violation_count as f32 / metrics.samples.max(1) as f32)
                .unwrap_or(f32::INFINITY);
        }
        _ => {}
    }
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

fn compute_best_guard_scores(
    valid: &Metrics,
    train: &Metrics,
    selected_regret_cap_cp: f32,
    extra_valid: &[Metrics],
    extra_valid_best_weight: f32,
) -> (f32, f32) {
    let max_regret = compute_best_metric_score(
        BestMetric::MaxRegret,
        valid,
        train,
        selected_regret_cap_cp,
        extra_valid,
        extra_valid_best_weight,
        None,
    );
    let bad100 = compute_best_metric_score(
        BestMetric::Bad100Regret,
        valid,
        train,
        selected_regret_cap_cp,
        extra_valid,
        extra_valid_best_weight,
        None,
    );
    (max_regret, bad100)
}

fn compute_teacher_match_score(
    valid: &Metrics,
    train: &Metrics,
    extra_valid: &[Metrics],
    extra_valid_best_weight: f32,
) -> f32 {
    let main_match = if valid.samples > 0 {
        teacher_match_rate(valid)
    } else if train.samples > 0 {
        teacher_match_rate(train)
    } else {
        0.0
    };
    if extra_valid_best_weight <= 0.0 || extra_valid.is_empty() {
        return main_match;
    }
    let extra_match_sum: f32 = extra_valid.iter().map(teacher_match_rate).sum();
    main_match + extra_valid_best_weight * (extra_match_sum / extra_valid.len() as f32)
}

fn best_guard_passes(
    current_max_regret: f32,
    current_bad100: f32,
    current_teacher_match: f32,
    baseline_max_regret: f32,
    baseline_bad100: f32,
    baseline_teacher_match: f32,
    max_regret_increase_cp: f32,
    bad100_increase: f32,
    teacher_match_drop_pct: f32,
) -> bool {
    if max_regret_increase_cp >= 0.0
        && current_max_regret > baseline_max_regret + max_regret_increase_cp
    {
        return false;
    }
    if bad100_increase >= 0.0 && current_bad100 > baseline_bad100 + bad100_increase {
        return false;
    }
    if teacher_match_drop_pct >= 0.0
        && current_teacher_match < baseline_teacher_match - teacher_match_drop_pct / 100.0
    {
        return false;
    }
    true
}

fn best_feedback_guard_passes(
    current: Option<&FeedbackMetrics>,
    baseline: Option<&FeedbackMetrics>,
    loss_increase: f32,
    violation_increase: f32,
) -> bool {
    if loss_increase < 0.0 && violation_increase < 0.0 {
        return true;
    }
    let (Some(current), Some(baseline)) = (current, baseline) else {
        return false;
    };
    if loss_increase >= 0.0 && current.loss > baseline.loss + loss_increase {
        return false;
    }
    if violation_increase >= 0.0
        && feedback_violation_rate(current) > feedback_violation_rate(baseline) + violation_increase
    {
        return false;
    }
    true
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
    if !args.best_guard_max_regret_increase_cp.is_finite() {
        return Err(anyhow!(
            "--best-guard-max-regret-increase-cp must be finite; use a negative value to disable"
        ));
    }
    if !args.best_guard_bad100_increase.is_finite() {
        return Err(anyhow!(
            "--best-guard-bad100-increase must be finite; use a negative value to disable"
        ));
    }
    if !args.best_guard_teacher_match_drop_pct.is_finite() {
        return Err(anyhow!(
            "--best-guard-teacher-match-drop-pct must be finite; use a negative value to disable"
        ));
    }
    if !args.best_guard_feedback_loss_increase.is_finite() {
        return Err(anyhow!(
            "--best-guard-feedback-loss-increase must be finite; use a negative value to disable"
        ));
    }
    if !args.best_guard_feedback_violation_increase.is_finite() {
        return Err(anyhow!(
            "--best-guard-feedback-violation-increase must be finite; use a negative value to disable"
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
    if !args.teacher_top_ce_weight.is_finite() || args.teacher_top_ce_weight < 0.0 {
        return Err(anyhow!(
            "--teacher-top-ce-weight must be finite and non-negative"
        ));
    }
    if !args.explicit_student_margin_weight.is_finite() || args.explicit_student_margin_weight < 0.0
    {
        return Err(anyhow!(
            "--explicit-student-margin-weight must be finite and non-negative"
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
    if !args.incumbent_protection_weight.is_finite() || args.incumbent_protection_weight < 0.0 {
        return Err(anyhow!(
            "--incumbent-protection-weight must be finite and non-negative"
        ));
    }
    if !args.incumbent_protection_max_regret_cp.is_finite()
        || args.incumbent_protection_max_regret_cp < 0.0
    {
        return Err(anyhow!(
            "--incumbent-protection-max-regret-cp must be finite and non-negative"
        ));
    }
    if !args
        .incumbent_protection_allow_teacher_better_cp
        .is_finite()
        || args.incumbent_protection_allow_teacher_better_cp < 0.0
    {
        return Err(anyhow!(
            "--incumbent-protection-allow-teacher-better-cp must be finite and non-negative"
        ));
    }
    if !args.tail_regret_penalty_weight.is_finite() || args.tail_regret_penalty_weight < 0.0 {
        return Err(anyhow!(
            "--tail-regret-penalty-weight must be finite and non-negative"
        ));
    }
    if !args.tail_regret_threshold_cp.is_finite() || args.tail_regret_threshold_cp < 0.0 {
        return Err(anyhow!(
            "--tail-regret-threshold-cp must be finite and non-negative"
        ));
    }
    if !args.tail_regret_weight_scale_cp.is_finite() || args.tail_regret_weight_scale_cp <= 0.0 {
        return Err(anyhow!(
            "--tail-regret-weight-scale-cp must be finite and positive"
        ));
    }
    if !args.tail_regret_max_weight.is_finite() || args.tail_regret_max_weight < 1.0 {
        return Err(anyhow!(
            "--tail-regret-max-weight must be finite and at least 1"
        ));
    }
    if !args.replay_weight.is_finite() || args.replay_weight < 0.0 {
        return Err(anyhow!("--replay-weight must be finite and non-negative"));
    }
    if !args.feedback_weight.is_finite() || args.feedback_weight < 0.0 {
        return Err(anyhow!("--feedback-weight must be finite and non-negative"));
    }
    if args.feedback_weight > 0.0 && args.feedback_json.is_empty() {
        return Err(anyhow!(
            "--feedback-json is required when --feedback-weight is positive"
        ));
    }
    let has_feedback_eval_input =
        !args.feedback_json.is_empty() || !args.feedback_guard_json.is_empty();
    if best_metric_requires_feedback(args.best_metric) && !has_feedback_eval_input {
        return Err(anyhow!(
            "--feedback-json or --feedback-guard-json is required when --best-metric is feedback-loss or feedback-violation"
        ));
    }
    if best_guard_requires_feedback(&args) && !has_feedback_eval_input {
        return Err(anyhow!(
            "--feedback-json or --feedback-guard-json is required when a feedback best-guard is enabled"
        ));
    }
    if !args.feedback_min_regret_delta_cp.is_finite() || args.feedback_min_regret_delta_cp < 0.0 {
        return Err(anyhow!(
            "--feedback-min-regret-delta-cp must be finite and non-negative"
        ));
    }
    if !args.feedback_min_candidate_regret_cp.is_finite()
        || args.feedback_min_candidate_regret_cp < 0.0
    {
        return Err(anyhow!(
            "--feedback-min-candidate-regret-cp must be finite and non-negative"
        ));
    }
    if !args.feedback_weight_scale_cp.is_finite() || args.feedback_weight_scale_cp <= 0.0 {
        return Err(anyhow!("--feedback-weight-scale-cp must be positive"));
    }
    if !args.feedback_max_sample_weight.is_finite() || args.feedback_max_sample_weight < 1.0 {
        return Err(anyhow!(
            "--feedback-max-sample-weight must be finite and >= 1"
        ));
    }
    if !args.policy_anchor_weight.is_finite() || args.policy_anchor_weight < 0.0 {
        return Err(anyhow!(
            "--policy-anchor-weight must be finite and non-negative"
        ));
    }
    if args.policy_anchor_weight > 0.0 && args.policy_anchor_weights.is_none() {
        return Err(anyhow!(
            "--policy-anchor-weights is required when --policy-anchor-weight is positive"
        ));
    }
    if !args.policy_anchor_temperature_cp.is_finite() || args.policy_anchor_temperature_cp <= 0.0 {
        return Err(anyhow!("--policy-anchor-temperature-cp must be positive"));
    }
    if !args.policy_anchor_margin_weight.is_finite() || args.policy_anchor_margin_weight < 0.0 {
        return Err(anyhow!(
            "--policy-anchor-margin-weight must be finite and non-negative"
        ));
    }
    if args.policy_anchor_margin_weight > 0.0 && args.policy_anchor_weights.is_none() {
        return Err(anyhow!(
            "--policy-anchor-weights is required when --policy-anchor-margin-weight is positive"
        ));
    }
    if !args.policy_anchor_margin_cp.is_finite() || args.policy_anchor_margin_cp < 0.0 {
        return Err(anyhow!(
            "--policy-anchor-margin-cp must be finite and non-negative"
        ));
    }
    if !args.policy_anchor_margin_softplus_temp_cp.is_finite()
        || args.policy_anchor_margin_softplus_temp_cp <= 0.0
    {
        return Err(anyhow!(
            "--policy-anchor-margin-softplus-temp-cp must be positive"
        ));
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
        teacher_top_ce_weight: args.teacher_top_ce_weight,
        explicit_student_margin_weight: args.explicit_student_margin_weight,
        game_teacher_margin_weight: args.game_teacher_margin_weight,
        game_teacher_max_regret_cp: args.game_teacher_max_regret_cp,
        game_teacher_min_bad_regret_cp: args.game_teacher_min_bad_regret_cp,
        current_top_margin_weight: args.current_top_margin_weight,
        current_top_min_bad_regret_cp: args.current_top_min_bad_regret_cp,
        incumbent_protection_weight: args.incumbent_protection_weight,
        incumbent_protection_max_regret_cp: args.incumbent_protection_max_regret_cp,
        incumbent_protection_allow_teacher_better_cp: args
            .incumbent_protection_allow_teacher_better_cp,
        tail_regret_penalty_weight: args.tail_regret_penalty_weight,
        tail_regret_threshold_cp: args.tail_regret_threshold_cp,
        tail_regret_weight_scale_cp: args.tail_regret_weight_scale_cp,
        tail_regret_max_weight: args.tail_regret_max_weight,
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
    let policy_anchor_model = if let Some(path) = args.policy_anchor_weights.as_ref() {
        let mut anchor_model = SparseModel::new(args.learning_rate, args.l2_lambda);
        anchor_model
            .load(path)
            .map_err(|e| anyhow!("failed to load {}: {}", path.display(), e))?;
        Some(anchor_model)
    } else {
        None
    };

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
    let feedback_samples = if args.feedback_json.is_empty() {
        Vec::new()
    } else {
        load_feedback_samples(
            &args.feedback_json,
            args.feedback_good_move,
            args.feedback_min_regret_delta_cp,
            args.feedback_min_candidate_regret_cp,
            args.feedback_weight_scale_cp,
            args.feedback_max_sample_weight,
            args.feedback_limit,
            args.feedback_dedupe_sfen,
        )?
    };
    let feedback_eval_samples = if args.feedback_guard_json.is_empty() {
        feedback_samples.clone()
    } else {
        load_feedback_samples(
            &args.feedback_guard_json,
            args.feedback_good_move,
            args.feedback_min_regret_delta_cp,
            args.feedback_min_candidate_regret_cp,
            args.feedback_weight_scale_cp,
            args.feedback_max_sample_weight,
            args.feedback_limit,
            args.feedback_dedupe_sfen,
        )?
    };
    if args.feedback_weight > 0.0 && feedback_samples.is_empty() {
        return Err(anyhow!(
            "--feedback-json produced no usable samples; lower feedback filters or check the input"
        ));
    }
    if best_guard_requires_feedback(&args) && feedback_eval_samples.is_empty() {
        return Err(anyhow!(
            "--feedback-json or --feedback-guard-json produced no usable samples for the feedback best-guard"
        ));
    }
    if !feedback_samples.is_empty() || !feedback_eval_samples.is_empty() {
        println!(
            "feedback samples: train={} eval={} guard_json={}",
            feedback_samples.len(),
            feedback_eval_samples.len(),
            if args.feedback_guard_json.is_empty() {
                0
            } else {
                args.feedback_guard_json.len()
            }
        );
    }

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
    println!("baseline train {}", log_bucket_summary(&baseline_train));
    println!(
        "baseline valid: {}",
        log_summary(
            &baseline_valid,
            args.bad_regret_cp,
            &bad_regret_thresholds_cp,
            "valid"
        )
    );
    println!("baseline valid {}", log_bucket_summary(&baseline_valid));
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
    let baseline_feedback_metrics = if !feedback_eval_samples.is_empty() {
        Some(evaluate_feedback_batch(
            &model,
            &feedback_eval_samples,
            &loss_options,
        ))
    } else {
        None
    };
    let baseline_best_metric_score = compute_best_metric_score(
        args.best_metric,
        &baseline_valid,
        &baseline_train,
        args.selected_regret_cap_cp,
        &baseline_extra_metrics,
        args.extra_valid_best_weight,
        baseline_feedback_metrics.as_ref(),
    );
    println!(
        "baseline best_metric_score={:.6}",
        baseline_best_metric_score
    );
    let (baseline_guard_max_regret, baseline_guard_bad100) = compute_best_guard_scores(
        &baseline_valid,
        &baseline_train,
        args.selected_regret_cap_cp,
        &baseline_extra_metrics,
        args.extra_valid_best_weight,
    );
    let baseline_guard_teacher_match = compute_teacher_match_score(
        &baseline_valid,
        &baseline_train,
        &baseline_extra_metrics,
        args.extra_valid_best_weight,
    );
    println!(
        "baseline best_guard max_regret_score={:.6} bad100_score={:.6} teacher_match_score={:.6}",
        baseline_guard_max_regret, baseline_guard_bad100, baseline_guard_teacher_match
    );
    let baseline_feedback_loss = baseline_feedback_metrics
        .as_ref()
        .map(|metrics| metrics.loss)
        .unwrap_or(f32::INFINITY);
    let baseline_feedback_violation = baseline_feedback_metrics
        .as_ref()
        .map(feedback_violation_rate)
        .unwrap_or(f32::INFINITY);
    if let Some(baseline_feedback) = baseline_feedback_metrics.as_ref() {
        println!(
            "baseline feedback: {} weight={:.4}",
            log_feedback_summary(&baseline_feedback, "feedback"),
            args.feedback_weight
        );
    }
    if let Some(anchor_model) = policy_anchor_model.as_ref() {
        let baseline_anchor = evaluate_policy_anchor_batch(
            &model,
            anchor_model,
            &valid_samples,
            args.policy_anchor_temperature_cp,
            args.policy_anchor_feature_source,
        );
        println!(
            "baseline policy_anchor: {} weight={:.4}",
            log_policy_anchor_summary(&baseline_anchor, "policy_anchor"),
            args.policy_anchor_weight
        );
        if args.policy_anchor_margin_weight > 0.0 {
            let baseline_anchor_margin = evaluate_policy_anchor_margin_batch(
                &model,
                anchor_model,
                &valid_samples,
                args.policy_anchor_margin_cp,
                args.policy_anchor_margin_softplus_temp_cp,
                args.policy_anchor_feature_source,
            );
            println!(
                "baseline policy_anchor_margin: {} weight={:.4}",
                log_policy_anchor_margin_summary(&baseline_anchor_margin, "policy_anchor_margin"),
                args.policy_anchor_margin_weight
            );
        }
    }

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
            "epoch,train_loss,train_pairs,train_selected_regret,train_p90,train_p95,train_teacher_match,train_bad_regret_ratio,train_samples,valid_loss,valid_pairs,valid_selected_regret,valid_p90,valid_p95,valid_teacher_match,valid_bad_regret_ratio,valid_samples,max_abs_delta,p95_abs_delta,clamped_weights,material_coeff",
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
            "0,{:.6},{},{:.2},{:.2},{:.2},{:.6},{:.4},{},{:.6},{},{:.2},{:.2},{:.2},{:.6},{:.4},{},{:.6},{:.6},{},{}{}{}",
            baseline_train.loss,
            baseline_train.pair_count,
            baseline_train.selected_regret_sum,
            percentile(baseline_train.regrets.clone(), 0.90),
            percentile(baseline_train.regrets.clone(), 0.95),
            teacher_match_rate(&baseline_train),
            baseline_train.bad_regret_count as f32 / baseline_train.samples.max(1) as f32,
            baseline_train.samples,
            baseline_valid.loss,
            baseline_valid.pair_count,
            baseline_valid.selected_regret_sum,
            percentile(baseline_valid.regrets.clone(), 0.90),
            percentile(baseline_valid.regrets.clone(), 0.95),
            teacher_match_rate(&baseline_valid),
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
    let mut replay_optimizer_state = AdagradState::default();
    let mut feedback_optimizer_state = AdagradState::default();
    let mut policy_anchor_optimizer_state = AdagradState::default();
    let mut policy_anchor_margin_optimizer_state = AdagradState::default();
    let mut best_metric_score = Some(baseline_best_metric_score);
    let mut best_epoch = 0usize;
    let mut rng = ChaCha8Rng::seed_from_u64(0x1234);
    let mut train_indices: Option<Vec<usize>> = if args.stream_train {
        None
    } else {
        Some((0..train_samples.as_ref().unwrap().len()).collect())
    };
    let replay_enabled = args.replay_weight > 0.0 && !args.replay_train.is_empty();
    let feedback_enabled = args.feedback_weight > 0.0 && !feedback_samples.is_empty();
    let mut feedback_indices: Vec<usize> = (0..feedback_samples.len()).collect();
    let policy_anchor_enabled = args.policy_anchor_weight > 0.0 && policy_anchor_model.is_some();
    let policy_anchor_margin_enabled =
        args.policy_anchor_margin_weight > 0.0 && policy_anchor_model.is_some();
    println!(
        "separate_aux_adagrad={}",
        if args.separate_aux_adagrad { 1 } else { 0 }
    );

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
            let active_optimizer_state: &mut AdagradState = if args.separate_aux_adagrad {
                &mut replay_optimizer_state
            } else {
                &mut optimizer_state
            };
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
                    &mut *active_optimizer_state,
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

        let mut feedback_metrics = None;
        if feedback_enabled {
            let original_learning_rate = model.kpp_eta;
            model.kpp_eta = original_learning_rate * args.feedback_weight;
            let active_optimizer_state: &mut AdagradState = if args.separate_aux_adagrad {
                &mut feedback_optimizer_state
            } else {
                &mut optimizer_state
            };
            feedback_indices.shuffle(&mut rng);
            let mut feedback_batch_refs: Vec<&FeedbackSample> = Vec::with_capacity(args.batch_size);
            for chunk in feedback_indices.chunks(args.batch_size) {
                feedback_batch_refs.clear();
                feedback_batch_refs.extend(chunk.iter().map(|idx| &feedback_samples[*idx]));
                let _ = update_feedback_refs_with_softplus(
                    &mut model,
                    &feedback_batch_refs,
                    &loss_options,
                    args.l2_lambda,
                    args.optimizer,
                    &mut *active_optimizer_state,
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
            model.kpp_eta = original_learning_rate;
            feedback_metrics = Some(evaluate_feedback_batch(
                &model,
                &feedback_eval_samples,
                &loss_options,
            ));
        } else if !feedback_eval_samples.is_empty() {
            feedback_metrics = Some(evaluate_feedback_batch(
                &model,
                &feedback_eval_samples,
                &loss_options,
            ));
        }

        let mut policy_anchor_metrics = None;
        if policy_anchor_enabled {
            let anchor_model = policy_anchor_model
                .as_ref()
                .expect("policy anchor enabled requires anchor model");
            let original_learning_rate = model.kpp_eta;
            model.kpp_eta = original_learning_rate * args.policy_anchor_weight;
            let active_optimizer_state: &mut AdagradState = if args.separate_aux_adagrad {
                &mut policy_anchor_optimizer_state
            } else {
                &mut optimizer_state
            };
            if args.stream_train {
                let (_, anchor_clamped) = update_streaming_policy_anchor_epoch(
                    &mut model,
                    anchor_model,
                    &args.train,
                    args.batch_size,
                    0,
                    args.policy_anchor_temperature_cp,
                    args.policy_anchor_feature_source,
                    args.l2_lambda,
                    args.optimizer,
                    &mut *active_optimizer_state,
                    args.adagrad_epsilon,
                    args.freeze_material,
                    initial_material,
                    initial_weights.as_ref(),
                    args.anchor_l2,
                    args.max_weight_delta,
                )?;
                clamped_weights += anchor_clamped;
            } else {
                let train_samples = train_samples.as_ref().unwrap();
                let indices = train_indices.as_mut().unwrap();
                indices.shuffle(&mut rng);
                let mut anchor_batch_refs: Vec<&Sample> = Vec::with_capacity(args.batch_size);
                for chunk in indices.chunks(args.batch_size) {
                    anchor_batch_refs.clear();
                    anchor_batch_refs.extend(chunk.iter().map(|idx| &train_samples[*idx]));
                    let _ = update_policy_anchor_refs(
                        &mut model,
                        anchor_model,
                        &anchor_batch_refs,
                        args.policy_anchor_temperature_cp,
                        args.policy_anchor_feature_source,
                        args.l2_lambda,
                        args.optimizer,
                        &mut *active_optimizer_state,
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
            model.kpp_eta = original_learning_rate;
            policy_anchor_metrics = Some(evaluate_policy_anchor_batch(
                &model,
                anchor_model,
                &valid_samples,
                args.policy_anchor_temperature_cp,
                args.policy_anchor_feature_source,
            ));
        }

        let mut policy_anchor_margin_metrics = None;
        if policy_anchor_margin_enabled {
            let anchor_model = policy_anchor_model
                .as_ref()
                .expect("policy anchor margin enabled requires anchor model");
            let original_learning_rate = model.kpp_eta;
            model.kpp_eta = original_learning_rate * args.policy_anchor_margin_weight;
            let active_optimizer_state: &mut AdagradState = if args.separate_aux_adagrad {
                &mut policy_anchor_margin_optimizer_state
            } else {
                &mut optimizer_state
            };
            if args.stream_train {
                let (_, anchor_margin_clamped) = update_streaming_policy_anchor_margin_epoch(
                    &mut model,
                    anchor_model,
                    &args.train,
                    args.batch_size,
                    0,
                    args.policy_anchor_margin_cp,
                    args.policy_anchor_margin_softplus_temp_cp,
                    args.policy_anchor_feature_source,
                    args.l2_lambda,
                    args.optimizer,
                    &mut *active_optimizer_state,
                    args.adagrad_epsilon,
                    args.freeze_material,
                    initial_material,
                    initial_weights.as_ref(),
                    args.anchor_l2,
                    args.max_weight_delta,
                )?;
                clamped_weights += anchor_margin_clamped;
            } else {
                let train_samples = train_samples.as_ref().unwrap();
                let indices = train_indices.as_mut().unwrap();
                indices.shuffle(&mut rng);
                let mut anchor_margin_batch_refs: Vec<&Sample> =
                    Vec::with_capacity(args.batch_size);
                for chunk in indices.chunks(args.batch_size) {
                    anchor_margin_batch_refs.clear();
                    anchor_margin_batch_refs.extend(chunk.iter().map(|idx| &train_samples[*idx]));
                    let _ = update_policy_anchor_margin_refs(
                        &mut model,
                        anchor_model,
                        &anchor_margin_batch_refs,
                        args.policy_anchor_margin_cp,
                        args.policy_anchor_margin_softplus_temp_cp,
                        args.policy_anchor_feature_source,
                        args.l2_lambda,
                        args.optimizer,
                        &mut *active_optimizer_state,
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
            model.kpp_eta = original_learning_rate;
            policy_anchor_margin_metrics = Some(evaluate_policy_anchor_margin_batch(
                &model,
                anchor_model,
                &valid_samples,
                args.policy_anchor_margin_cp,
                args.policy_anchor_margin_softplus_temp_cp,
                args.policy_anchor_feature_source,
            ));
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
            feedback_metrics.as_ref(),
        );
        let (current_guard_max_regret, current_guard_bad100) = compute_best_guard_scores(
            &valid,
            &train,
            args.selected_regret_cap_cp,
            &current_epoch_extra_metrics,
            args.extra_valid_best_weight,
        );
        let current_guard_teacher_match = compute_teacher_match_score(
            &valid,
            &train,
            &current_epoch_extra_metrics,
            args.extra_valid_best_weight,
        );
        let metric_guard_passed = best_guard_passes(
            current_guard_max_regret,
            current_guard_bad100,
            current_guard_teacher_match,
            baseline_guard_max_regret,
            baseline_guard_bad100,
            baseline_guard_teacher_match,
            args.best_guard_max_regret_increase_cp,
            args.best_guard_bad100_increase,
            args.best_guard_teacher_match_drop_pct,
        );
        let feedback_guard_passed = best_feedback_guard_passes(
            feedback_metrics.as_ref(),
            baseline_feedback_metrics.as_ref(),
            args.best_guard_feedback_loss_increase,
            args.best_guard_feedback_violation_increase,
        );
        let guard_passed = metric_guard_passed && feedback_guard_passed;
        if guard_passed
            && (best_metric_score.is_none()
                || current_metric < best_metric_score.unwrap_or(f32::INFINITY))
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
            "epoch {} train: loss={:.6} pairs={} selected_regret={:.2} p90={:.2} p95={:.2} teacher_match={:.2}% bad_ratio={:.4} {}",
            epoch,
            train.loss,
            train.pair_count,
            train.selected_regret_sum,
            percentile(train.regrets.clone(), 0.90),
            percentile(train.regrets.clone(), 0.95),
            teacher_match_rate(&train) * 100.0,
            train.bad_regret_count as f32 / train.samples.max(1) as f32,
            bad_regret_thresholds_summary(&train, &bad_regret_thresholds_cp)
        );
        println!("epoch {} train {}", epoch, log_bucket_summary(&train));
        println!(
            "epoch {} valid: loss={:.6} pairs={} selected_regret={:.2} p90={:.2} p95={:.2} teacher_match={:.2}% bad_ratio={:.4} {} max_abs_delta={:.6} p95_abs_delta={:.6} clamped_weights={}",
            epoch,
            valid.loss,
            valid.pair_count,
            valid.selected_regret_sum,
            percentile(valid.regrets.clone(), 0.90),
            percentile(valid.regrets.clone(), 0.95),
            teacher_match_rate(&valid) * 100.0,
            valid.bad_regret_count as f32 / valid.samples.max(1) as f32,
            bad_regret_thresholds_summary(&valid, &bad_regret_thresholds_cp),
            max_abs_delta,
            p95_abs_delta,
            clamped_weights
        );
        println!("epoch {} valid {}", epoch, log_bucket_summary(&valid));
        if replay_enabled {
            println!(
                "epoch {} replay: samples={} pairs={} loss={:.6} selected_regret={:.2}",
                epoch, replay_samples, replay_pairs, replay_loss, replay_selected_regret
            );
        }
        if let Some(metrics) = feedback_metrics.as_ref() {
            println!(
                "epoch {} feedback: {}",
                epoch,
                log_feedback_summary(metrics, "feedback")
            );
        }
        if let Some(metrics) = policy_anchor_metrics.as_ref() {
            println!(
                "epoch {} policy_anchor: {}",
                epoch,
                log_policy_anchor_summary(metrics, "policy_anchor")
            );
        }
        if let Some(metrics) = policy_anchor_margin_metrics.as_ref() {
            println!(
                "epoch {} policy_anchor_margin: {}",
                epoch,
                log_policy_anchor_margin_summary(metrics, "policy_anchor_margin")
            );
        }
        println!(
            "epoch {} best_metric_score={:.6} best_guard_max_regret={:.6} best_guard_bad100={:.6} best_guard_teacher_match={:.6} best_guard_metric_passed={} best_guard_feedback_loss={:.6} best_guard_feedback_violation={:.6} best_guard_feedback_passed={} best_guard_passed={}",
            epoch,
            current_metric,
            current_guard_max_regret,
            current_guard_bad100,
            current_guard_teacher_match,
            metric_guard_passed,
            feedback_metrics
                .as_ref()
                .map(|metrics| metrics.loss)
                .unwrap_or(f32::INFINITY),
            feedback_metrics
                .as_ref()
                .map(feedback_violation_rate)
                .unwrap_or(f32::INFINITY),
            feedback_guard_passed,
            guard_passed
        );
        if baseline_feedback_metrics.is_some() {
            println!(
                "epoch {} baseline_feedback_guard loss={:.6} violation={:.6}",
                epoch, baseline_feedback_loss, baseline_feedback_violation
            );
        }
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
                "{},{:.6},{},{:.2},{:.2},{:.2},{:.6},{:.4},{},{:.6},{},{:.2},{:.2},{:.2},{:.6},{:.4},{},{:.6},{:.6},{},{}{}{}",
                epoch,
                train.loss,
                train.pair_count,
                train.selected_regret_sum,
                percentile(train.regrets.clone(), 0.90),
                percentile(train.regrets.clone(), 0.95),
                teacher_match_rate(&train),
                train.bad_regret_count as f32 / train.samples.max(1) as f32,
                train.samples,
                valid.loss,
                valid.pair_count,
                valid.selected_regret_sum,
                percentile(valid.regrets.clone(), 0.90),
                percentile(valid.regrets.clone(), 0.95),
                teacher_match_rate(&valid),
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
    if !feedback_eval_samples.is_empty() {
        let metrics = evaluate_feedback_batch(&final_model, &feedback_eval_samples, &loss_options);
        println!(
            "final feedback: {}",
            log_feedback_summary(&metrics, "feedback")
        );
    }
    if let Some(anchor_model) = policy_anchor_model.as_ref() {
        let metrics = evaluate_policy_anchor_batch(
            &final_model,
            anchor_model,
            &valid_samples,
            args.policy_anchor_temperature_cp,
            args.policy_anchor_feature_source,
        );
        println!(
            "final policy_anchor: {}",
            log_policy_anchor_summary(&metrics, "policy_anchor")
        );
        if args.policy_anchor_margin_weight > 0.0 {
            let margin_metrics = evaluate_policy_anchor_margin_batch(
                &final_model,
                anchor_model,
                &valid_samples,
                args.policy_anchor_margin_cp,
                args.policy_anchor_margin_softplus_temp_cp,
                args.policy_anchor_feature_source,
            );
            println!(
                "final policy_anchor_margin: {}",
                log_policy_anchor_margin_summary(&margin_metrics, "policy_anchor_margin")
            );
        }
    }

    println!(
        "best_epoch={} best_value={:.6}",
        best_epoch,
        best_metric_score.unwrap_or(f32::INFINITY)
    );
    Ok(())
}
