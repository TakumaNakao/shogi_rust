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
    #[value(name = "bad50-regret")]
    #[value(alias = "bad-regret-50")]
    Bad50Regret,
    #[value(name = "capped-selected-regret")]
    CappedSelectedRegret,
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

#[derive(Debug, Deserialize)]
struct TreeRecord {
    #[serde(default)]
    schema: Option<String>,
    #[serde(default)]
    version: Option<u8>,
    #[serde(default)]
    sfen: Option<String>,
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
}

#[derive(Clone, Copy)]
struct LossOptions {
    margin_cp: f32,
    softplus_temp_cp: f32,
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

fn pair_indices(
    sample: &Sample,
    options: &PairOptions,
    model: Option<&SparseModel>,
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
            if pairs.len() >= options.max_pairs_per_sample {
                return pairs;
            }
        }
    }
    pairs
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
        if sample.candidates.is_empty() {
            continue;
        }

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

        for (good_idx, bad_idx) in pair_indices(sample, pair_options, Some(model)) {
            let good = &sample.candidates[good_idx];
            let bad = &sample.candidates[bad_idx];
            let (good_features, good_material) = candidate_leaf_features(good, true);
            let (bad_features, bad_material) = candidate_leaf_features(bad, false);
            let diff = model.predict_with_material(good_features, good_material)
                - model.predict_with_material(bad_features, bad_material);
            let x = (loss_options.margin_cp - diff) / loss_options.softplus_temp_cp;
            if x.is_finite() {
                metrics.loss += loss_options.softplus_temp_cp * softplus(x);
                metrics.pair_count += 1;
            }
        }
    }

    if metrics.samples > 0 {
        metrics.selected_regret_sum /= metrics.samples as f32;
    }
    if metrics.pair_count > 0 {
        metrics.loss /= metrics.pair_count as f32;
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

    for sample in samples {
        for (good_idx, bad_idx) in pair_indices(sample, pair_options, Some(model)) {
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

            loss += options.loss.softplus_temp_cp * softplus(x);
            let grad_diff = -sigmoid(x);
            pair_count += 1;

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

    if pair_count == 0 {
        return (0.0, 0);
    }
    let avg = 1.0 / pair_count as f32;
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

    (loss / pair_count as f32, pair_count)
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

fn load_samples(path: &PathBuf) -> Result<Vec<Sample>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut samples = Vec::new();

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
        {
            continue;
        }
        if record.version == Some(0) {
            continue;
        }

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
        if candidates.is_empty() {
            return Err(anyhow!(
                "{}:{} has no usable candidates",
                path.display(),
                line_number
            ));
        }

        let mut sample = Sample {
            position,
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
        samples.push(sample);
    }

    if samples.is_empty() {
        return Err(anyhow!("{} contains no usable samples", path.display()));
    }
    Ok(samples)
}

fn log_summary(
    metrics: &Metrics,
    _bad_regret_cp: f32,
    bad_regret_thresholds: &[f32],
    sample_name: &str,
) -> String {
    let bad_ratio = metrics.bad_regret_count as f32 / metrics.samples.max(1) as f32;
    format!(
        "{}: samples={} pairs={} loss={:.6} selected_regret_mean={:.2} p90={:.2} p95={:.2} bad_regret_ratio={:.4} {}",
        sample_name,
        metrics.samples,
        metrics.pair_count,
        metrics.loss,
        metrics.selected_regret_sum,
        percentile(metrics.regrets.clone(), 0.90),
        percentile(metrics.regrets.clone(), 0.95),
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
        BestMetric::Bad50Regret => {
            if valid.samples > 0 {
                regret_ratio_above(valid, 50.0)
            } else if train.samples > 0 {
                regret_ratio_above(train, 50.0)
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
    if args.max_pairs_per_sample == 0 {
        return Err(anyhow!("--max-pairs-per-sample must be greater than 0"));
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
    };
    let loss_options = LossOptions {
        margin_cp: args.margin_cp,
        softplus_temp_cp: args.softplus_temp_cp,
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

    let initial_weights = model.w.clone();
    let initial_material = model.material_coeff;

    let mut best_model_checkpoint = model.clone();

    let train_samples = load_samples(&args.train)?;
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

    let baseline_train = evaluate_batch(
        &model,
        &train_samples,
        args.bad_regret_cp,
        &bad_regret_thresholds_cp,
        &pair_options,
        &loss_options,
    );
    let baseline_valid = evaluate_batch(
        &model,
        &valid_samples,
        args.bad_regret_cp,
        &bad_regret_thresholds_cp,
        &pair_options,
        &loss_options,
    );

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
    let mut best_metric_score = Some(compute_best_metric_value(
        args.best_metric,
        &baseline_valid,
        &baseline_train,
        args.selected_regret_cap_cp,
    ));
    let mut best_epoch = 0usize;
    let mut rng = ChaCha8Rng::seed_from_u64(0x1234);
    let mut train_indices: Vec<usize> = (0..train_samples.len()).collect();
    let mut batch_refs: Vec<&Sample> = Vec::with_capacity(args.batch_size);

    for epoch in 1..=args.epochs {
        train_indices.shuffle(&mut rng);

        let mut clamped_weights = 0usize;
        for chunk in train_indices.chunks(args.batch_size) {
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
            if args.anchor_l2 > 0.0 || args.max_weight_delta.is_some() {
                clamped_weights += apply_weight_constraints(
                    &mut model,
                    &initial_weights,
                    args.anchor_l2,
                    args.max_weight_delta,
                );
            }
            ensure_finite_model(&model)?;
        }

        let train = evaluate_batch(
            &model,
            &train_samples,
            args.bad_regret_cp,
            &bad_regret_thresholds_cp,
            &pair_options,
            &loss_options,
        );
        let valid = evaluate_batch(
            &model,
            &valid_samples,
            args.bad_regret_cp,
            &bad_regret_thresholds_cp,
            &pair_options,
            &loss_options,
        );

        let delta_abs: Vec<f32> = model
            .w
            .iter()
            .zip(initial_weights.iter())
            .map(|(w, anchor)| (w - anchor).abs())
            .collect();
        let max_abs_delta = delta_abs.iter().copied().fold(0.0_f32, f32::max);
        let p95_abs_delta = percentile(delta_abs, 0.95);

        let current_metric = compute_best_metric_value(
            args.best_metric,
            &valid,
            &train,
            args.selected_regret_cap_cp,
        );
        if best_metric_score.is_none()
            || current_metric < best_metric_score.unwrap_or(f32::INFINITY)
        {
            copy_model_into(&mut best_model_checkpoint, &model);
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
        for batch in &extra_valid {
            let metrics = evaluate_batch(
                &model,
                &batch.samples,
                args.bad_regret_cp,
                &bad_regret_thresholds_cp,
                &pair_options,
                &loss_options,
            );
            println!(
                "epoch {} extra_valid[{}]: {}",
                epoch,
                batch.label,
                log_summary(
                    &metrics,
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
        best_model_checkpoint.save(path)?;
        println!(
            "Best checkpoint saved to {} (epoch={})",
            path.display(),
            best_epoch
        );
    }

    best_model_checkpoint.save(&args.output)?;
    if !args.freeze_material {
        println!("saved final model to {}", args.output.display());
    } else {
        println!(
            "saved final model to {} (material restored to frozen value)",
            args.output.display()
        );
    }

    ensure_finite_model(&best_model_checkpoint)?;
    if args.freeze_material && best_model_checkpoint.material_coeff != initial_material {
        return Err(anyhow!(
            "material_coeff changed despite --freeze-material: {} -> {}",
            initial_material,
            best_model_checkpoint.material_coeff
        ));
    }
    for batch in &extra_valid {
        let metrics = evaluate_batch(
            &best_model_checkpoint,
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
