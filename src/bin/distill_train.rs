use anyhow::{anyhow, Result};
use clap::Parser;
use serde::Deserialize;
use shogi_ai::evaluation::{extract_kpp_features_and_material, SparseModel};
use shogi_ai::utils::{parse_usi_move, position_from_sfen_or_usi};
use shogi_core::{Move, Piece};
use shogi_lib::Position;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Train KPP policy weights from fixed teacher bestmove JSONL data")]
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
    #[arg(long, default_value_t = 0.02)]
    learning_rate: f32,
    #[arg(long, default_value_t = 600.0)]
    softmax_temperature: f32,
    #[arg(long, default_value_t = 600.0)]
    teacher_temperature: f32,
    #[arg(long, default_value_t = 0.0)]
    min_teacher_gap: f32,
    #[arg(long)]
    max_teacher_gap: Option<f32>,
    #[arg(long, default_value_t = true)]
    freeze_material: bool,
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

#[derive(Deserialize)]
struct DistillRecord {
    sfen: String,
    teacher_move: String,
    #[serde(default)]
    teacher_scores: Vec<TeacherScoreRecord>,
}

#[derive(Deserialize)]
struct TeacherScoreRecord {
    move_usi: String,
    score: f32,
}

struct TeacherScore {
    mv: Move,
    score: f32,
}

struct Sample {
    position: Position,
    teacher_move: Move,
    teacher_scores: Vec<TeacherScore>,
}

struct NamedBatch {
    label: String,
    samples: Vec<Sample>,
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

fn load_batch(
    path: &Path,
    min_teacher_gap: f32,
    max_teacher_gap: Option<f32>,
) -> Result<Vec<Sample>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut batch = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: DistillRecord = serde_json::from_str(&line)
            .map_err(|e| anyhow!("{}:{} invalid json: {}", path.display(), line_index + 1, e))?;
        let position = position_from_sfen_or_usi(&record.sfen).ok_or_else(|| {
            anyhow!(
                "{}:{} invalid sfen: {}",
                path.display(),
                line_index + 1,
                record.sfen
            )
        })?;
        let teacher_move =
            parse_move_for_position(&position, &record.teacher_move).ok_or_else(|| {
                anyhow!(
                    "{}:{} invalid move: {}",
                    path.display(),
                    line_index + 1,
                    record.teacher_move
                )
            })?;
        let legal_moves = position.legal_moves();
        if !legal_moves.contains(&teacher_move) {
            return Err(anyhow!(
                "{}:{} illegal teacher move: {}",
                path.display(),
                line_index + 1,
                record.teacher_move
            ));
        }
        let mut teacher_scores = Vec::with_capacity(record.teacher_scores.len());
        for teacher_score in record.teacher_scores {
            let mv =
                parse_move_for_position(&position, &teacher_score.move_usi).ok_or_else(|| {
                    anyhow!(
                        "{}:{} invalid teacher score move: {}",
                        path.display(),
                        line_index + 1,
                        teacher_score.move_usi
                    )
                })?;
            if !legal_moves.contains(&mv) {
                return Err(anyhow!(
                    "{}:{} illegal teacher score move: {}",
                    path.display(),
                    line_index + 1,
                    teacher_score.move_usi
                ));
            }
            if teacher_score.score.is_finite() {
                teacher_scores.push(TeacherScore {
                    mv,
                    score: teacher_score.score,
                });
            }
        }
        teacher_scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if min_teacher_gap > 0.0 && teacher_scores.len() >= 2 {
            let gap = teacher_scores[0].score - teacher_scores[1].score;
            if gap < min_teacher_gap {
                continue;
            }
        }
        if let Some(max_teacher_gap) = max_teacher_gap {
            if teacher_scores.len() >= 2 {
                let gap = teacher_scores[0].score - teacher_scores[1].score;
                if gap > max_teacher_gap {
                    continue;
                }
            }
        }
        batch.push(Sample {
            position,
            teacher_move,
            teacher_scores,
        });
    }
    if batch.is_empty() {
        return Err(anyhow!("{} contains no samples", path.display()));
    }
    Ok(batch)
}

fn parse_extra_valid(spec: &str) -> Result<(String, PathBuf)> {
    let (label, path) = spec
        .split_once('=')
        .ok_or_else(|| anyhow!("--extra-valid must use LABEL=PATH: {spec}"))?;
    if label.trim().is_empty() {
        return Err(anyhow!("--extra-valid label must not be empty: {spec}"));
    }
    if path.trim().is_empty() {
        return Err(anyhow!("--extra-valid path must not be empty: {spec}"));
    }
    Ok((label.trim().to_string(), PathBuf::from(path.trim())))
}

fn evaluate_policy(
    model: &SparseModel,
    batch: &[Sample],
    softmax_temperature: f32,
    teacher_temperature: f32,
) -> (f32, f32, usize) {
    let mut loss_sum = 0.0;
    let mut correct = 0usize;
    let mut valid = 0usize;

    for sample in batch {
        let legal_moves = sample.position.legal_moves();
        if legal_moves.is_empty() || !legal_moves.contains(&sample.teacher_move) {
            continue;
        }

        let mut scores = Vec::with_capacity(legal_moves.len());
        let mut best_move = legal_moves[0];
        let mut best_score = f32::NEG_INFINITY;
        for &mv in legal_moves.iter() {
            let mut child = sample.position.clone();
            child.do_move(mv);
            child.switch_turn();
            let score = model.predict_from_position(&child);
            if score > best_score {
                best_score = score;
                best_move = mv;
            }
            scores.push(score);
        }

        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores = scores
            .iter()
            .map(|score| ((*score - max_score) / softmax_temperature).exp())
            .collect::<Vec<_>>();
        let total_score = exp_scores.iter().sum::<f32>();

        let target_probs = target_probabilities(sample, &legal_moves, teacher_temperature);
        let mut sample_loss = 0.0;
        for (idx, _) in legal_moves.iter().enumerate() {
            let target = target_probs.get(&idx).copied().unwrap_or(0.0);
            if target > 0.0 {
                let prob = exp_scores[idx] / total_score;
                sample_loss += -target * prob.max(1e-7).ln();
            }
        }
        loss_sum += sample_loss;
        if best_move == sample.teacher_move {
            correct += 1;
        }
        valid += 1;
    }

    if valid == 0 {
        (0.0, 0.0, 0)
    } else {
        (
            loss_sum / valid as f32,
            correct as f32 / valid as f32,
            valid,
        )
    }
}

fn target_probabilities(
    sample: &Sample,
    legal_moves: &[Move],
    teacher_temperature: f32,
) -> HashMap<usize, f32> {
    let mut targets = HashMap::new();
    if sample.teacher_scores.is_empty() {
        if let Some(idx) = legal_moves.iter().position(|&mv| mv == sample.teacher_move) {
            targets.insert(idx, 1.0);
        }
        return targets;
    }

    let max_score = sample
        .teacher_scores
        .iter()
        .map(|teacher| teacher.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    let mut weighted = Vec::new();
    for teacher in &sample.teacher_scores {
        if let Some(idx) = legal_moves.iter().position(|&mv| mv == teacher.mv) {
            let weight = ((teacher.score - max_score) / teacher_temperature).exp();
            if weight.is_finite() && weight > 0.0 {
                total += weight;
                weighted.push((idx, weight));
            }
        }
    }
    if total <= 0.0 {
        if let Some(idx) = legal_moves.iter().position(|&mv| mv == sample.teacher_move) {
            targets.insert(idx, 1.0);
        }
        return targets;
    }
    for (idx, weight) in weighted {
        *targets.entry(idx).or_insert(0.0) += weight / total;
    }
    targets
}

fn update_batch_with_soft_targets(
    model: &mut SparseModel,
    batch: &[Sample],
    softmax_temperature: f32,
    teacher_temperature: f32,
) -> (f32, usize) {
    let mut w_grads: HashMap<usize, f32> = HashMap::new();
    let mut material_grad_total = 0.0;
    let mut loss = 0.0;
    let mut valid_samples = 0usize;

    for sample in batch {
        let legal_moves = sample.position.legal_moves();
        if legal_moves.is_empty() || !legal_moves.contains(&sample.teacher_move) {
            continue;
        }
        let target_probs = target_probabilities(sample, &legal_moves, teacher_temperature);
        if target_probs.is_empty() {
            continue;
        }

        let move_data = legal_moves
            .iter()
            .map(|&mv| {
                let mut child = sample.position.clone();
                child.do_move(mv);
                child.switch_turn();
                let (features, material) = extract_kpp_features_and_material(&child);
                let score = model.predict_with_material(&features, material);
                (features, material, score)
            })
            .collect::<Vec<_>>();

        let max_score = move_data
            .iter()
            .map(|(_, _, score)| *score)
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_scores = move_data
            .iter()
            .map(|(_, _, score)| ((*score - max_score) / softmax_temperature).exp())
            .collect::<Vec<_>>();
        let total_score = exp_scores.iter().sum::<f32>();
        if total_score <= 0.0 {
            continue;
        }

        valid_samples += 1;
        for (idx, (features, material, _)) in move_data.iter().enumerate() {
            let prob = exp_scores[idx] / total_score;
            let target = target_probs.get(&idx).copied().unwrap_or(0.0);
            if target > 0.0 {
                loss += -target * prob.max(1e-7).ln();
            }
            let delta = (prob - target) / softmax_temperature;
            for &feature_idx in features {
                *w_grads.entry(feature_idx).or_insert(0.0) += delta;
            }
            material_grad_total += delta * *material;
        }
    }

    if valid_samples == 0 {
        return (0.0, 0);
    }

    for (i, grad) in w_grads {
        model.w[i] -= model.kpp_eta * (grad / valid_samples as f32 + model.l2_lambda * model.w[i]);
    }
    model.material_coeff -= model.kpp_eta
        * (material_grad_total / valid_samples as f32 + model.l2_lambda * model.material_coeff);

    (loss / valid_samples as f32, valid_samples)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.epochs == 0 {
        return Err(anyhow!("--epochs must be greater than zero"));
    }
    if args.batch_size == 0 {
        return Err(anyhow!("--batch-size must be greater than zero"));
    }
    if !args.softmax_temperature.is_finite() || args.softmax_temperature <= 0.0 {
        return Err(anyhow!("--softmax-temperature must be positive"));
    }
    if !args.teacher_temperature.is_finite() || args.teacher_temperature <= 0.0 {
        return Err(anyhow!("--teacher-temperature must be positive"));
    }
    if !args.min_teacher_gap.is_finite() || args.min_teacher_gap < 0.0 {
        return Err(anyhow!("--min-teacher-gap must be non-negative"));
    }
    if let Some(max_teacher_gap) = args.max_teacher_gap {
        if !max_teacher_gap.is_finite() || max_teacher_gap < 0.0 {
            return Err(anyhow!("--max-teacher-gap must be non-negative"));
        }
        if args.min_teacher_gap > 0.0 && max_teacher_gap < args.min_teacher_gap {
            return Err(anyhow!(
                "--max-teacher-gap must be greater than or equal to --min-teacher-gap"
            ));
        }
    }

    let train = load_batch(&args.train, args.min_teacher_gap, args.max_teacher_gap)?;
    let valid = load_batch(&args.valid, args.min_teacher_gap, args.max_teacher_gap)?;
    let extra_valid = args
        .extra_valid
        .iter()
        .map(|spec| {
            let (label, path) = parse_extra_valid(spec)?;
            let samples = load_batch(&path, args.min_teacher_gap, args.max_teacher_gap)?;
            Ok(NamedBatch { label, samples })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut model = SparseModel::new(args.learning_rate, 0.0);
    model.load(&args.weights)?;
    model.kpp_eta = args.learning_rate;
    let initial_material_coeff = model.material_coeff;

    let (base_train_loss, base_train_accuracy, train_valid) = evaluate_policy(
        &model,
        &train,
        args.softmax_temperature,
        args.teacher_temperature,
    );
    let (base_valid_loss, base_valid_accuracy, valid_valid) = evaluate_policy(
        &model,
        &valid,
        args.softmax_temperature,
        args.teacher_temperature,
    );
    println!(
        "baseline train samples={} ce={:.6} top1={:.4}",
        train_valid, base_train_loss, base_train_accuracy
    );
    println!(
        "baseline valid samples={} ce={:.6} top1={:.4}",
        valid_valid, base_valid_loss, base_valid_accuracy
    );
    for batch in &extra_valid {
        let (loss, accuracy, count) = evaluate_policy(
            &model,
            &batch.samples,
            args.softmax_temperature,
            args.teacher_temperature,
        );
        println!(
            "baseline extra_valid[{}] samples={} ce={:.6} top1={:.4}",
            batch.label, count, loss, accuracy
        );
    }
    if args.dry_run {
        return Ok(());
    }

    for epoch in 1..=args.epochs {
        for chunk in train.chunks(args.batch_size) {
            let material_before = model.material_coeff;
            let _ = update_batch_with_soft_targets(
                &mut model,
                chunk,
                args.softmax_temperature,
                args.teacher_temperature,
            );
            if args.freeze_material {
                model.material_coeff = material_before;
            }
        }
        if args.freeze_material {
            model.material_coeff = initial_material_coeff;
        }
        let (train_loss, train_accuracy, _) = evaluate_policy(
            &model,
            &train,
            args.softmax_temperature,
            args.teacher_temperature,
        );
        let (valid_loss, valid_accuracy, _) = evaluate_policy(
            &model,
            &valid,
            args.softmax_temperature,
            args.teacher_temperature,
        );
        println!(
            "epoch {} train_ce={:.6} train_top1={:.4} valid_ce={:.6} valid_top1={:.4} material_coeff={:.6}",
            epoch,
            train_loss,
            train_accuracy,
            valid_loss,
            valid_accuracy,
            model.material_coeff
        );
        for batch in &extra_valid {
            let (loss, accuracy, count) = evaluate_policy(
                &model,
                &batch.samples,
                args.softmax_temperature,
                args.teacher_temperature,
            );
            println!(
                "epoch {} extra_valid[{}] samples={} ce={:.6} top1={:.4}",
                epoch, batch.label, count, loss, accuracy
            );
        }
    }

    if model.w.iter().any(|value| !value.is_finite())
        || !model.bias.is_finite()
        || !model.material_coeff.is_finite()
    {
        return Err(anyhow!("model contains NaN or inf"));
    }
    if args.freeze_material && model.material_coeff != initial_material_coeff {
        return Err(anyhow!(
            "material_coeff changed despite --freeze-material: {} -> {}",
            initial_material_coeff,
            model.material_coeff
        ));
    }

    model.save(&args.output)?;
    println!("saved {}", args.output.display());
    Ok(())
}
