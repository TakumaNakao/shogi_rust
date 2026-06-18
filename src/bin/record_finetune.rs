use anyhow::{anyhow, Context, Result};
use clap::Parser;
use glob::glob;
use rayon::prelude::*;
use shogi_ai::evaluation::{extract_kpp_features_and_material, SparseModel, MAX_FEATURES};
use shogi_ai::utils::{parse_usi_move, position_from_sfen_or_usi};
use shogi_core::{Color, Move, Piece};
use shogi_lib::Position;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

const WIN_RATE_SCALING_FACTOR: f32 = 600.0;

#[derive(Parser, Debug)]
#[command(about = "Fine-tune KPP weights from saved benchmark game records")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    record_dir: Option<PathBuf>,
    #[arg(required_unless_present = "record_dir")]
    records: Vec<PathBuf>,
    #[arg(long, default_value_t = 32)]
    tail_plies: usize,
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    #[arg(long, default_value_t = 0.0001)]
    learning_rate: f32,
    #[arg(long, default_value_t = false)]
    freeze_material: bool,
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

#[derive(Debug)]
struct Record {
    result: String,
    new_as: Option<Color>,
    positions: Vec<Position>,
}

#[derive(Clone)]
struct Sample {
    features: Vec<usize>,
    material: f32,
    predicted_win_rate: f32,
    target_win_rate: f32,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x / WIN_RATE_SCALING_FACTOR).exp())
}

fn load_model(path: &Path) -> Result<SparseModel> {
    let mut model = SparseModel::new(0.0, 0.0);
    model
        .load(path)
        .map_err(|e| anyhow!("failed to load {}: {}", path.display(), e))?;
    Ok(model)
}

fn parse_side(side: &str) -> Option<Color> {
    match side {
        "black" => Some(Color::Black),
        "white" => Some(Color::White),
        _ => None,
    }
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

fn parse_position_command(command: &str) -> Result<Vec<Position>> {
    let rest = command
        .trim()
        .strip_prefix("position ")
        .ok_or_else(|| anyhow!("record position line must start with 'position '"))?;
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    if tokens.is_empty() {
        return Err(anyhow!("empty position command"));
    }

    let (start_text, move_tokens): (String, &[&str]) = if tokens[0] == "startpos" {
        let moves_idx = tokens.iter().position(|&token| token == "moves");
        let move_tokens = moves_idx.map(|idx| &tokens[idx + 1..]).unwrap_or(&[]);
        ("startpos".to_string(), move_tokens)
    } else if tokens[0] == "sfen" {
        let moves_idx = tokens
            .iter()
            .position(|&token| token == "moves")
            .unwrap_or(tokens.len());
        let move_tokens = if moves_idx < tokens.len() {
            &tokens[moves_idx + 1..]
        } else {
            &[]
        };
        (tokens[..moves_idx].join(" "), move_tokens)
    } else {
        return Err(anyhow!("unsupported position command: {}", command));
    };

    let mut position = position_from_sfen_or_usi(&start_text)
        .ok_or_else(|| anyhow!("invalid start position: {}", start_text))?;
    let mut positions = vec![position.clone()];
    for move_text in move_tokens {
        let mv = parse_move_for_position(&position, move_text)
            .ok_or_else(|| anyhow!("invalid move '{}' in {}", move_text, command))?;
        if !position.legal_moves().contains(&mv) {
            return Err(anyhow!("illegal move '{}' in {}", move_text, command));
        }
        position.do_move(mv);
        positions.push(position.clone());
    }

    Ok(positions)
}

fn load_record(path: &Path) -> Result<Record> {
    let content =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut result = None;
    let mut new_as = None;
    let mut positions = None;

    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("result ") {
            result = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("new_as ") {
            new_as = parse_side(rest.trim());
        } else if line.starts_with("position ") {
            positions = Some(parse_position_command(line)?);
        }
    }

    Ok(Record {
        result: result.unwrap_or_else(|| "Unknown".to_string()),
        new_as,
        positions: positions.ok_or_else(|| anyhow!("missing position line in {}", path.display()))?,
    })
}

fn collect_records(args: &Args) -> Result<Vec<PathBuf>> {
    let mut paths = args.records.clone();
    if let Some(record_dir) = &args.record_dir {
        let pattern = record_dir.join("*.usi");
        let pattern = pattern
            .to_str()
            .ok_or_else(|| anyhow!("record path is not valid UTF-8: {}", pattern.display()))?;
        for entry in glob(pattern)? {
            paths.push(entry?);
        }
    }
    paths.sort();
    paths.dedup();
    if paths.is_empty() {
        return Err(anyhow!("no record files found"));
    }
    Ok(paths)
}

fn winner_for_record(record: &Record) -> Option<Color> {
    let new_as = record.new_as?;
    match record.result.as_str() {
        "NewWin" => Some(new_as),
        "BaselineWin" => Some(new_as.flip()),
        _ => None,
    }
}

fn build_samples(model: &SparseModel, records: &[Record], tail_plies: usize) -> Vec<Sample> {
    records
        .par_iter()
        .flat_map(|record| {
            let Some(winner) = winner_for_record(record) else {
                return Vec::new();
            };
            let start = record.positions.len().saturating_sub(tail_plies.max(1));
            record.positions[start..]
                .iter()
                .map(|position| {
                    let (features, material) = extract_kpp_features_and_material(position);
                    let predicted_score = model.predict_with_material(&features, material);
                    let target_win_rate = if position.side_to_move() == winner {
                        1.0
                    } else {
                        0.0
                    };
                    Sample {
                        features,
                        material,
                        predicted_win_rate: sigmoid(predicted_score),
                        target_win_rate,
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn apply_samples(
    model: &mut SparseModel,
    samples: &[Sample],
    learning_rate: f32,
    freeze_material: bool,
) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let (feature_grads, bias_grad, material_grad, loss_sum) = samples
        .par_iter()
        .map(|sample| {
            let error = sample.predicted_win_rate - sample.target_win_rate;
            let mut feature_grads = HashMap::new();
            for &feature in &sample.features {
                if feature < MAX_FEATURES {
                    *feature_grads.entry(feature).or_insert(0.0) += error;
                }
            }
            let epsilon = 1e-7;
            let p = sample.predicted_win_rate.clamp(epsilon, 1.0 - epsilon);
            let q = sample.target_win_rate;
            let loss = -(q * p.ln() + (1.0 - q) * (1.0 - p).ln());
            (feature_grads, error, error * sample.material, loss)
        })
        .reduce(
            || (HashMap::new(), 0.0, 0.0, 0.0),
            |mut a, b| {
                for (feature, grad) in b.0 {
                    *a.0.entry(feature).or_insert(0.0) += grad;
                }
                a.1 += b.1;
                a.2 += b.2;
                a.3 += b.3;
                a
            },
        );

    let sample_count = samples.len() as f32;
    for (feature, grad) in feature_grads {
        model.w[feature] -= learning_rate * grad / sample_count;
    }
    model.bias -= learning_rate * bias_grad / sample_count;
    if !freeze_material {
        model.material_coeff -= learning_rate * material_grad / sample_count;
    }

    loss_sum / sample_count
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.epochs == 0 {
        return Err(anyhow!("--epochs must be greater than zero"));
    }
    if !args.dry_run && args.output.is_none() {
        return Err(anyhow!("--output is required unless --dry-run is set"));
    }

    let mut model = load_model(&args.weights)?;
    let paths = collect_records(&args)?;
    let records: Vec<Record> = paths
        .iter()
        .map(|path| load_record(path))
        .collect::<Result<Vec<_>>>()?;
    let total_positions: usize = records.iter().map(|record| record.positions.len()).sum();
    println!(
        "loaded {} records ({} positions)",
        records.len(),
        total_positions
    );

    let mut last_loss = 0.0;
    for epoch in 1..=args.epochs {
        let samples = build_samples(&model, &records, args.tail_plies);
        last_loss = apply_samples(
            &mut model,
            &samples,
            args.learning_rate,
            args.freeze_material,
        );
        println!(
            "epoch {} samples={} loss={:.6} material_coeff={:.6}",
            epoch,
            samples.len(),
            last_loss,
            model.material_coeff
        );
    }

    if let Some(output) = &args.output {
        if !args.dry_run {
            model.save(output)?;
            println!("saved {}", output.display());
        }
    }
    println!("final loss: {:.6}", last_loss);

    Ok(())
}
