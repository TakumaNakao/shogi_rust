use anyhow::{anyhow, Result};
use clap::Parser;
use serde::Deserialize;
use shogi_ai::evaluation::SparseModel;
use shogi_ai::utils::{parse_usi_move, position_from_sfen_or_usi};
use shogi_core::{Move, Piece};
use shogi_lib::Position;
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
    #[arg(long, default_value_t = true)]
    freeze_material: bool,
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

#[derive(Deserialize)]
struct DistillRecord {
    sfen: String,
    teacher_move: String,
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

fn load_batch(path: &Path) -> Result<Vec<(Position, Move)>> {
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
        if !position.legal_moves().contains(&teacher_move) {
            return Err(anyhow!(
                "{}:{} illegal teacher move: {}",
                path.display(),
                line_index + 1,
                record.teacher_move
            ));
        }
        batch.push((position, teacher_move));
    }
    if batch.is_empty() {
        return Err(anyhow!("{} contains no samples", path.display()));
    }
    Ok(batch)
}

fn evaluate_policy(
    model: &SparseModel,
    batch: &[(Position, Move)],
    softmax_temperature: f32,
) -> (f32, f32, usize) {
    let mut loss_sum = 0.0;
    let mut correct = 0usize;
    let mut valid = 0usize;

    for (position, teacher_move) in batch {
        let legal_moves = position.legal_moves();
        if legal_moves.is_empty() || !legal_moves.contains(teacher_move) {
            continue;
        }

        let mut scores = Vec::with_capacity(legal_moves.len());
        let mut best_move = legal_moves[0];
        let mut best_score = f32::NEG_INFINITY;
        let mut teacher_index = None;
        for (idx, &mv) in legal_moves.iter().enumerate() {
            let mut child = position.clone();
            child.do_move(mv);
            child.switch_turn();
            let score = model.predict_from_position(&child);
            if score > best_score {
                best_score = score;
                best_move = mv;
            }
            if mv == *teacher_move {
                teacher_index = Some(idx);
            }
            scores.push(score);
        }

        let Some(teacher_index) = teacher_index else {
            continue;
        };
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores = scores
            .iter()
            .map(|score| ((*score - max_score) / softmax_temperature).exp())
            .collect::<Vec<_>>();
        let total_score = exp_scores.iter().sum::<f32>();
        let teacher_prob = exp_scores[teacher_index] / total_score;
        loss_sum += -teacher_prob.max(1e-7).ln();
        if best_move == *teacher_move {
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

    let train = load_batch(&args.train)?;
    let valid = load_batch(&args.valid)?;
    let mut model = SparseModel::new(args.learning_rate, 0.0);
    model.load(&args.weights)?;
    model.kpp_eta = args.learning_rate;
    let initial_material_coeff = model.material_coeff;

    let (base_train_loss, base_train_accuracy, train_valid) =
        evaluate_policy(&model, &train, args.softmax_temperature);
    let (base_valid_loss, base_valid_accuracy, valid_valid) =
        evaluate_policy(&model, &valid, args.softmax_temperature);
    println!(
        "baseline train samples={} ce={:.6} top1={:.4}",
        train_valid, base_train_loss, base_train_accuracy
    );
    println!(
        "baseline valid samples={} ce={:.6} top1={:.4}",
        valid_valid, base_valid_loss, base_valid_accuracy
    );

    for epoch in 1..=args.epochs {
        for chunk in train.chunks(args.batch_size) {
            let material_before = model.material_coeff;
            let _ =
                model.update_batch_with_cross_entropy_temperature(chunk, args.softmax_temperature);
            if args.freeze_material {
                model.material_coeff = material_before;
            }
        }
        if args.freeze_material {
            model.material_coeff = initial_material_coeff;
        }
        let (train_loss, train_accuracy, _) =
            evaluate_policy(&model, &train, args.softmax_temperature);
        let (valid_loss, valid_accuracy, _) =
            evaluate_policy(&model, &valid, args.softmax_temperature);
        println!(
            "epoch {} train_ce={:.6} train_top1={:.4} valid_ce={:.6} valid_top1={:.4} material_coeff={:.6}",
            epoch,
            train_loss,
            train_accuracy,
            valid_loss,
            valid_accuracy,
            model.material_coeff
        );
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

    if !args.dry_run {
        model.save(&args.output)?;
        println!("saved {}", args.output.display());
    }
    Ok(())
}
