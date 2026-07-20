use anyhow::{anyhow, Result};
use clap::{ArgAction, Parser, ValueEnum};
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use shogi_ai::evaluation::{extract_kpp_features_and_material, SparseModel};
use shogi_ai::utils::{parse_usi_move, position_from_sfen_or_usi};
use shogi_core::{Move, Piece};
use shogi_lib::Position;
use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Train Bonanza-style pairwise policy KPP from csa_policy_dump JSONL")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long)]
    train: Vec<PathBuf>,
    #[arg(long)]
    valid: Vec<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
    #[arg(long, default_value_t = 0.02)]
    learning_rate: f32,
    #[arg(long, value_enum, default_value_t = OptimizerKind::Sgd)]
    optimizer: OptimizerKind,
    #[arg(long, default_value_t = 1.0e-6)]
    adagrad_epsilon: f32,
    #[arg(long, default_value_t = 0.0)]
    l2_lambda: f32,
    #[arg(long, default_value_t = 0.0)]
    anchor_l2: f32,
    #[arg(long)]
    max_weight_delta: Option<f32>,
    #[arg(long, default_value_t = 4)]
    hard_negatives: usize,
    #[arg(long, default_value_t = 0.5)]
    margin_cp: f32,
    #[arg(long, default_value_t = 100.0)]
    softplus_temp_cp: f32,
    #[arg(long)]
    valid_max_samples: Option<usize>,
    #[arg(long)]
    best_checkpoint_path: Option<PathBuf>,
    #[arg(long)]
    log_path: Option<PathBuf>,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    freeze_material: bool,
    #[arg(long, default_value_t = 9601)]
    seed: u64,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OptimizerKind {
    Sgd,
    Adagrad,
}

#[derive(Debug, Deserialize)]
struct PolicyRecord {
    sfen: String,
    teacher_move: String,
}

#[derive(Clone)]
struct Sample {
    position: Position,
    teacher_move: Move,
}

#[derive(Clone)]
struct CandidateEval {
    mv: Move,
    score: f32,
    features: Vec<usize>,
    material: f32,
}

#[derive(Default)]
struct BatchMetrics {
    samples: usize,
    top1: usize,
    pair_correct: usize,
    pair_total: usize,
    rank_sum: f32,
    loss_sum: f32,
    loss_pairs: usize,
}

#[derive(Default)]
struct ConstraintStats {
    clamped_weights: usize,
    max_abs_delta: f32,
}

#[derive(Default)]
struct EpochMetrics {
    samples: usize,
    top1: usize,
    pair_correct: usize,
    pair_total: usize,
    rank_sum: f32,
    loss_sum: f32,
    loss_pairs: usize,
    clamped_weights: usize,
    max_abs_delta: f32,
}

#[derive(Default)]
struct OptimizerState {
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

fn parse_sample_line(path: &Path, line_no: usize, line: &str) -> Result<Option<Sample>> {
    if line.trim().is_empty() {
        return Ok(None);
    }
    let record: PolicyRecord = serde_json::from_str(line)
        .map_err(|e| anyhow!("{}:{} invalid json: {}", path.display(), line_no, e))?;
    let position = position_from_sfen_or_usi(&record.sfen).ok_or_else(|| {
        anyhow!(
            "{}:{} invalid sfen: {}",
            path.display(),
            line_no,
            record.sfen
        )
    })?;
    let teacher_move =
        parse_move_for_position(&position, &record.teacher_move).ok_or_else(|| {
            anyhow!(
                "{}:{} invalid move: {}",
                path.display(),
                line_no,
                record.teacher_move
            )
        })?;
    if !position.legal_moves().contains(&teacher_move) {
        return Err(anyhow!(
            "{}:{} illegal teacher move: {}",
            path.display(),
            line_no,
            record.teacher_move
        ));
    }
    Ok(Some(Sample {
        position,
        teacher_move,
    }))
}

fn softplus(x: f32) -> f32 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn gather_candidate_evals(
    sample: &Sample,
    model: &SparseModel,
    hard_negatives: usize,
) -> Option<(CandidateEval, Vec<CandidateEval>, usize, usize, usize)> {
    let legal_moves = sample.position.legal_moves();
    if legal_moves.is_empty() {
        return None;
    }

    let mut candidates = Vec::with_capacity(legal_moves.len());
    for &mv in legal_moves.iter() {
        let mut child = sample.position.clone();
        child.do_move(mv);
        child.switch_turn();
        let (features, material) = extract_kpp_features_and_material(&child);
        let score = model.predict_with_material(&features, material);
        candidates.push(CandidateEval {
            mv,
            score,
            features,
            material,
        });
    }
    if !candidates
        .iter()
        .any(|candidate| candidate.mv == sample.teacher_move)
    {
        return None;
    }

    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

    let teacher_rank = candidates
        .iter()
        .position(|candidate| candidate.mv == sample.teacher_move)?
        + 1;
    let teacher = candidates
        .iter()
        .find(|candidate| candidate.mv == sample.teacher_move)?
        .clone();

    let mut pair_correct = 0usize;
    let mut pair_total = 0usize;
    let mut hard_negatives_batch = Vec::new();
    for candidate in candidates.iter() {
        if candidate.mv == sample.teacher_move {
            continue;
        }
        pair_total += 1;
        if teacher.score >= candidate.score {
            pair_correct += 1;
        }
        if hard_negatives_batch.len() < hard_negatives {
            hard_negatives_batch.push(candidate.clone());
        }
    }

    Some((
        teacher,
        hard_negatives_batch,
        teacher_rank,
        pair_correct,
        pair_total,
    ))
}

fn update_batch(
    model: &mut SparseModel,
    optimizer_state: &mut OptimizerState,
    batch: &[Sample],
    optimizer: OptimizerKind,
    adagrad_epsilon: f32,
    margin_cp: f32,
    softplus_temp_cp: f32,
    hard_negatives: usize,
    l2_lambda: f32,
    initial_weights: &[f32],
    initial_material: f32,
    anchor_l2: f32,
    max_weight_delta: Option<f32>,
    freeze_material: bool,
) -> (BatchMetrics, ConstraintStats) {
    let mut feature_grads: HashMap<usize, f32> = HashMap::new();
    let mut material_grad = 0.0f32;
    let mut pairwise_feature_updates = BatchMetrics::default();
    let mut touched_features = Vec::with_capacity(1024);

    for sample in batch {
        let Some((teacher, siblings, teacher_rank, pair_correct, pair_total)) =
            gather_candidate_evals(sample, model, hard_negatives)
        else {
            continue;
        };

        pairwise_feature_updates.samples += 1;
        pairwise_feature_updates.top1 += usize::from(teacher_rank == 1);
        pairwise_feature_updates.rank_sum += teacher_rank as f32;
        pairwise_feature_updates.pair_correct += pair_correct;
        pairwise_feature_updates.pair_total += pair_total;

        for sibling in siblings {
            let x = (margin_cp - (teacher.score - sibling.score)) / softplus_temp_cp;
            if !x.is_finite() || !teacher.score.is_finite() || !sibling.score.is_finite() {
                continue;
            }
            let sample_loss = softplus(x);
            let grad = -sigmoid(x) / softplus_temp_cp;

            pairwise_feature_updates.loss_sum += sample_loss;
            pairwise_feature_updates.loss_pairs += 1;

            for &idx in &teacher.features {
                match feature_grads.entry(idx) {
                    Entry::Occupied(mut e) => {
                        *e.get_mut() += grad;
                    }
                    Entry::Vacant(e) => {
                        e.insert(grad);
                        touched_features.push(idx);
                    }
                }
            }
            for &idx in &sibling.features {
                match feature_grads.entry(idx) {
                    Entry::Occupied(mut e) => {
                        *e.get_mut() -= grad;
                    }
                    Entry::Vacant(e) => {
                        e.insert(-grad);
                        touched_features.push(idx);
                    }
                }
            }
            if !freeze_material {
                material_grad += grad * (teacher.material - sibling.material);
            }
        }
    }

    let mut constraints = ConstraintStats::default();
    let pair_count = pairwise_feature_updates.loss_pairs as f32;
    if pair_count == 0.0 {
        return (pairwise_feature_updates, constraints);
    }

    for idx in touched_features {
        let grad = feature_grads.remove(&idx).unwrap_or(0.0) / pair_count;
        let grad = grad + l2_lambda * model.w[idx];
        let step = match optimizer {
            OptimizerKind::Sgd => model.kpp_eta * grad,
            OptimizerKind::Adagrad => {
                let acc = optimizer_state.w_acc.entry(idx).or_insert(0.0);
                *acc += grad * grad;
                model.kpp_eta * grad / (acc.sqrt() + adagrad_epsilon)
            }
        };
        let mut updated = model.w[idx] - step;

        if anchor_l2 > 0.0 {
            updated += anchor_l2 * (initial_weights[idx] - updated);
        }
        if let Some(limit) = max_weight_delta {
            let delta = updated - initial_weights[idx];
            let clamped = delta.clamp(-limit, limit);
            if clamped != delta {
                constraints.clamped_weights += 1;
            }
            updated = initial_weights[idx] + clamped;
        }

        let delta = (updated - initial_weights[idx]).abs();
        if delta > constraints.max_abs_delta {
            constraints.max_abs_delta = delta;
        }
        model.w[idx] = updated;
    }

    if !freeze_material {
        let material_grad = material_grad / pair_count + l2_lambda * model.material_coeff;
        let material_step = match optimizer {
            OptimizerKind::Sgd => model.kpp_eta * material_grad,
            OptimizerKind::Adagrad => {
                optimizer_state.material_acc += material_grad * material_grad;
                model.kpp_eta * material_grad
                    / (optimizer_state.material_acc.sqrt() + adagrad_epsilon)
            }
        };
        let mut updated = model.material_coeff - material_step;
        if anchor_l2 > 0.0 {
            updated += anchor_l2 * (initial_material - updated);
        }
        if let Some(limit) = max_weight_delta {
            let delta = updated - initial_material;
            let clamped = delta.clamp(-limit, limit);
            if clamped != delta {
                constraints.clamped_weights += 1;
            }
            updated = initial_material + clamped;
        }

        let material_delta = (updated - initial_material).abs();
        if material_delta > constraints.max_abs_delta {
            constraints.max_abs_delta = material_delta;
        }
        model.material_coeff = updated;
    }

    (pairwise_feature_updates, constraints)
}

fn evaluate_batch(
    model: &SparseModel,
    paths: &[PathBuf],
    hard_negatives: usize,
    margin_cp: f32,
    softplus_temp_cp: f32,
    max_samples: Option<usize>,
) -> Result<BatchMetrics> {
    let mut metrics = BatchMetrics::default();
    let mut seen = 0usize;

    'dataset: for path in paths {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        for (line_no, line) in reader.lines().enumerate() {
            if let Some(limit) = max_samples {
                if seen >= limit {
                    break 'dataset;
                }
            }
            let line = line?;
            let sample = match parse_sample_line(path, line_no + 1, &line) {
                Ok(sample) => sample,
                Err(err) => {
                    eprintln!("skip {}:{}: {}", path.display(), line_no + 1, err);
                    continue;
                }
            };
            let Some(sample) = sample else {
                continue;
            };

            let Some((teacher, siblings, rank, pair_correct, pair_total)) =
                gather_candidate_evals(&sample, model, hard_negatives)
            else {
                continue;
            };
            if !teacher.score.is_finite() {
                continue;
            }

            seen += 1;
            metrics.samples += 1;
            metrics.top1 += usize::from(rank == 1);
            metrics.rank_sum += rank as f32;
            metrics.pair_correct += pair_correct;
            metrics.pair_total += pair_total;
            for sibling in siblings {
                let x = (margin_cp - (teacher.score - sibling.score)) / softplus_temp_cp;
                if !x.is_finite() {
                    continue;
                }
                metrics.loss_sum += softplus(x);
                metrics.loss_pairs += 1;
            }
        }
    }

    Ok(metrics)
}

fn percent(numer: usize, denom: usize) -> f32 {
    if denom == 0 {
        0.0
    } else {
        numer as f32 / denom as f32 * 100.0
    }
}

fn mean_rank(sum: f32, samples: usize) -> f32 {
    if samples == 0 {
        0.0
    } else {
        sum / samples as f32
    }
}

fn pair_accuracy(correct: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        correct as f32 / total as f32 * 100.0
    }
}

fn mean_loss(loss_sum: f32, pairs: usize) -> f32 {
    if pairs == 0 {
        0.0
    } else {
        loss_sum / pairs as f32
    }
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn run_training_epoch(
    model: &mut SparseModel,
    optimizer_state: &mut OptimizerState,
    args: &Args,
    train_paths: &[PathBuf],
    initial_w: &[f32],
    initial_material: f32,
    rng: &mut ChaCha8Rng,
) -> Result<EpochMetrics> {
    let mut shuffled_train_paths = train_paths.to_vec();
    shuffled_train_paths.shuffle(rng);

    let mut batch = Vec::with_capacity(args.batch_size);
    let mut metrics = EpochMetrics::default();

    for path in shuffled_train_paths {
        let file = File::open(&path)?;
        let reader = BufReader::new(file);

        for (line_no, line) in reader.lines().enumerate() {
            let line = line?;
            let sample = match parse_sample_line(&path, line_no + 1, &line) {
                Ok(sample) => sample,
                Err(err) => {
                    eprintln!("skip {}:{}: {}", path.display(), line_no + 1, err);
                    continue;
                }
            };
            let Some(sample) = sample else {
                continue;
            };
            batch.push(sample);
            if batch.len() >= args.batch_size {
                let (batch_metrics, constraints) = update_batch(
                    model,
                    optimizer_state,
                    &batch,
                    args.optimizer,
                    args.adagrad_epsilon,
                    args.margin_cp,
                    args.softplus_temp_cp,
                    args.hard_negatives,
                    args.l2_lambda,
                    initial_w,
                    initial_material,
                    args.anchor_l2,
                    args.max_weight_delta,
                    args.freeze_material,
                );
                metrics.samples += batch_metrics.samples;
                metrics.top1 += batch_metrics.top1;
                metrics.pair_correct += batch_metrics.pair_correct;
                metrics.pair_total += batch_metrics.pair_total;
                metrics.rank_sum += batch_metrics.rank_sum;
                metrics.loss_sum += batch_metrics.loss_sum;
                metrics.loss_pairs += batch_metrics.loss_pairs;
                metrics.clamped_weights += constraints.clamped_weights;
                if constraints.max_abs_delta > metrics.max_abs_delta {
                    metrics.max_abs_delta = constraints.max_abs_delta;
                }
                batch.clear();
            }
        }
    }
    if !batch.is_empty() {
        let (batch_metrics, constraints) = update_batch(
            model,
            optimizer_state,
            &batch,
            args.optimizer,
            args.adagrad_epsilon,
            args.margin_cp,
            args.softplus_temp_cp,
            args.hard_negatives,
            args.l2_lambda,
            initial_w,
            initial_material,
            args.anchor_l2,
            args.max_weight_delta,
            args.freeze_material,
        );
        metrics.samples += batch_metrics.samples;
        metrics.top1 += batch_metrics.top1;
        metrics.pair_correct += batch_metrics.pair_correct;
        metrics.pair_total += batch_metrics.pair_total;
        metrics.rank_sum += batch_metrics.rank_sum;
        metrics.loss_sum += batch_metrics.loss_sum;
        metrics.loss_pairs += batch_metrics.loss_pairs;
        metrics.clamped_weights += constraints.clamped_weights;
        if constraints.max_abs_delta > metrics.max_abs_delta {
            metrics.max_abs_delta = constraints.max_abs_delta;
        }
    }

    Ok(metrics)
}

fn ensure_finite_model(model: &SparseModel) -> Result<()> {
    if !model.w.iter().all(|value| value.is_finite()) || !model.material_coeff.is_finite() {
        return Err(anyhow!("model contains NaN or inf"));
    }
    Ok(())
}

fn is_better(candidate_top1: f32, candidate_loss: f32, best_top1: f32, best_loss: f32) -> bool {
    if candidate_top1 > best_top1 + 1e-9 {
        return true;
    }
    if (candidate_top1 - best_top1).abs() <= 1e-9 && candidate_loss < best_loss {
        return true;
    }
    false
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    if args.train.is_empty() {
        return Err(anyhow!("--train is required"));
    }
    if args.valid.is_empty() {
        return Err(anyhow!("--valid is required"));
    }
    if args.epochs == 0 {
        return Err(anyhow!("--epochs must be greater than zero"));
    }
    if args.batch_size == 0 {
        return Err(anyhow!("--batch-size must be greater than zero"));
    }
    if args.hard_negatives == 0 {
        return Err(anyhow!("--hard-negatives must be greater than zero"));
    }
    if args.learning_rate <= 0.0 || !args.learning_rate.is_finite() {
        return Err(anyhow!("--learning-rate must be positive"));
    }
    if args.adagrad_epsilon <= 0.0 || !args.adagrad_epsilon.is_finite() {
        return Err(anyhow!("--adagrad-epsilon must be positive"));
    }
    if args.l2_lambda < 0.0 || !args.l2_lambda.is_finite() {
        return Err(anyhow!("--l2-lambda must be non-negative"));
    }
    if args.anchor_l2 < 0.0 || !args.anchor_l2.is_finite() {
        return Err(anyhow!("--anchor-l2 must be non-negative"));
    }
    if let Some(max_delta) = args.max_weight_delta {
        if max_delta <= 0.0 || !max_delta.is_finite() {
            return Err(anyhow!("--max-weight-delta must be positive"));
        }
    }
    if args.margin_cp < 0.0 || !args.margin_cp.is_finite() {
        return Err(anyhow!("--margin-cp must be non-negative"));
    }
    if args.softplus_temp_cp <= 0.0 || !args.softplus_temp_cp.is_finite() {
        return Err(anyhow!("--softplus-temp-cp must be positive"));
    }

    let mut model = SparseModel::new(args.learning_rate, args.l2_lambda);
    model
        .load(&args.weights)
        .map_err(|e| anyhow!("failed to load {}: {e}", args.weights.display()))?;
    model.kpp_eta = args.learning_rate;
    model.l2_lambda = args.l2_lambda;
    ensure_finite_model(&model)?;

    let initial_model = SparseModel {
        w: model.w.clone(),
        bias: model.bias,
        material_coeff: model.material_coeff,
        kpp_eta: model.kpp_eta,
        l2_lambda: model.l2_lambda,
    };

    println!(
        "optimizer={:?} learning_rate={} adagrad_epsilon={}",
        args.optimizer, args.learning_rate, args.adagrad_epsilon
    );

    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let mut optimizer_state = OptimizerState::default();

    let mut log_file = if let Some(path) = &args.log_path {
        ensure_parent_dir(path)?;
        Some(BufWriter::new(File::create(path)?))
    } else {
        None
    };
    if let Some(file) = log_file.as_mut() {
        writeln!(
            file,
            "epoch,train_samples,train_top1_pct,train_pair_accuracy_pct,train_mean_rank,train_loss,valid_samples,valid_top1_pct,valid_pair_accuracy_pct,valid_mean_rank,valid_loss,clamped_weights,max_abs_delta,material_coeff"
        )?;
        file.flush()?;
    }

    let baseline_valid = evaluate_batch(
        &model,
        &args.valid,
        args.hard_negatives,
        args.margin_cp,
        args.softplus_temp_cp,
        args.valid_max_samples,
    )?;
    if baseline_valid.samples == 0 {
        println!("baseline valid: 0 samples, skip best-checkpoint selection");
    }
    let baseline_top1 = percent(baseline_valid.top1, baseline_valid.samples);
    let baseline_pair_accuracy =
        pair_accuracy(baseline_valid.pair_correct, baseline_valid.pair_total);
    let baseline_mean_rank = mean_rank(baseline_valid.rank_sum, baseline_valid.samples);
    let baseline_loss = mean_loss(baseline_valid.loss_sum, baseline_valid.loss_pairs);
    println!(
        "baseline valid: top1={:.2}% pair_acc={:.2}% mean_rank={:.2} loss={:.6}",
        baseline_top1, baseline_pair_accuracy, baseline_mean_rank, baseline_loss,
    );

    let mut best_model = model.clone();
    let mut best_epoch = 0usize;
    let mut best_top1 = baseline_top1;
    let mut best_loss = baseline_loss;

    if let Some(file) = log_file.as_mut() {
        writeln!(
            file,
            "0,0,0.000000,0.000000,0.000000,0.000000,{},{:.6},{:.6},{:.6},{:.6},0,0.000000,{}",
            baseline_valid.samples,
            baseline_top1,
            baseline_pair_accuracy,
            baseline_mean_rank,
            baseline_loss,
            model.material_coeff,
        )?;
        file.flush()?;
    }

    for epoch in 1..=args.epochs {
        let train_metrics = run_training_epoch(
            &mut model,
            &mut optimizer_state,
            &args,
            &args.train,
            &initial_model.w,
            initial_model.material_coeff,
            &mut rng,
        )?;
        ensure_finite_model(&model)?;

        let valid_metrics = evaluate_batch(
            &model,
            &args.valid,
            args.hard_negatives,
            args.margin_cp,
            args.softplus_temp_cp,
            args.valid_max_samples,
        )?;

        let train_top1 = percent(train_metrics.top1, train_metrics.samples);
        let train_pair_accuracy =
            pair_accuracy(train_metrics.pair_correct, train_metrics.pair_total);
        let train_mean_rank = mean_rank(train_metrics.rank_sum, train_metrics.samples);
        let train_loss = mean_loss(train_metrics.loss_sum, train_metrics.loss_pairs);
        let valid_top1 = percent(valid_metrics.top1, valid_metrics.samples);
        let valid_pair_accuracy =
            pair_accuracy(valid_metrics.pair_correct, valid_metrics.pair_total);
        let valid_mean_rank = mean_rank(valid_metrics.rank_sum, valid_metrics.samples);
        let valid_loss = mean_loss(valid_metrics.loss_sum, valid_metrics.loss_pairs);

        println!(
            "epoch {epoch}: train top1={train_top1:.2}% pair_acc={train_pair_accuracy:.2}% mean_rank={train_mean_rank:.2} loss={train_loss:.6} | valid top1={valid_top1:.2}% pair_acc={valid_pair_accuracy:.2}% mean_rank={valid_mean_rank:.2} loss={valid_loss:.6} | clamped={}",
            train_metrics.clamped_weights,
        );

        if let Some(file) = log_file.as_mut() {
            writeln!(
                file,
                "{},{},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{},{:.6},{}",
                epoch,
                train_metrics.samples,
                train_top1,
                train_pair_accuracy,
                train_mean_rank,
                train_loss,
                valid_metrics.samples,
                valid_top1,
                valid_pair_accuracy,
                valid_mean_rank,
                valid_loss,
                train_metrics.clamped_weights,
                train_metrics.max_abs_delta,
                model.material_coeff
            )?;
            file.flush()?;
        }

        if is_better(valid_top1, valid_loss, best_top1, best_loss) {
            best_top1 = valid_top1;
            best_loss = valid_loss;
            best_epoch = epoch;
            best_model = model.clone();
        }
    }

    if let Some(path) = args.best_checkpoint_path.as_deref() {
        ensure_parent_dir(path)?;
        best_model.save(path)?;
        println!(
            "best checkpoint saved at epoch {} to {}",
            best_epoch,
            path.display()
        );
    }
    println!(
        "best: epoch={} top1={:.2}% loss={:.6}",
        best_epoch, best_top1, best_loss
    );

    ensure_parent_dir(&args.output)?;
    model.save(&args.output)?;
    println!("final model saved to {}", args.output.display());
    ensure_finite_model(&model)?;
    Ok(())
}
