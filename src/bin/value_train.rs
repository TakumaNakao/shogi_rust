use anyhow::{anyhow, Result};
use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use shogi_ai::evaluation::{extract_kpp_features_and_material, SparseModel};
use shogi_ai::utils::position_from_sfen_or_usi;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Train KPP weights from searched root value JSONL data")]
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
    target_scale: f32,
    #[arg(long, default_value_t = 3000.0)]
    score_clip: f32,
    #[arg(long, default_value_t = 1.0)]
    huber_delta: f32,
    #[arg(long, default_value_t = 100.0)]
    sign_threshold: f32,
    #[arg(long, default_value_t = true)]
    freeze_material: bool,
    #[arg(long, default_value_t = false)]
    freeze_bias: bool,
    #[arg(long, default_value_t = 9901)]
    seed: u64,
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

#[derive(Deserialize)]
struct ValueRecord {
    sfen: String,
    teacher_score: f32,
}

struct Sample {
    features: Vec<usize>,
    material: f32,
    target_score: f32,
}

struct NamedBatch {
    label: String,
    samples: Vec<Sample>,
}

#[derive(Clone, Copy)]
struct Metrics {
    samples: usize,
    huber: f32,
    rmse_cp: f32,
    mae_cp: f32,
    sign_accuracy: f32,
    sign_samples: usize,
    correlation: f32,
}

fn load_batch(path: &Path, score_clip: f32) -> Result<Vec<Sample>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut batch = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: ValueRecord = serde_json::from_str(&line)
            .map_err(|e| anyhow!("{}:{} invalid json: {}", path.display(), line_index + 1, e))?;
        if !record.teacher_score.is_finite() {
            continue;
        }
        let position = position_from_sfen_or_usi(&record.sfen).ok_or_else(|| {
            anyhow!(
                "{}:{} invalid sfen: {}",
                path.display(),
                line_index + 1,
                record.sfen
            )
        })?;
        let (features, material) = extract_kpp_features_and_material(&position);
        batch.push(Sample {
            features,
            material,
            target_score: record.teacher_score.clamp(-score_clip, score_clip),
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

fn evaluate_value(
    model: &SparseModel,
    batch: &[Sample],
    target_scale: f32,
    huber_delta: f32,
    sign_threshold: f32,
) -> Metrics {
    let mut squared_error = 0.0;
    let mut huber_loss = 0.0;
    let mut absolute_error = 0.0;
    let mut sign_correct = 0usize;
    let mut sign_samples = 0usize;
    let mut sum_pred = 0.0;
    let mut sum_target = 0.0;
    let mut sum_pred_sq = 0.0;
    let mut sum_target_sq = 0.0;
    let mut sum_cross = 0.0;

    for sample in batch {
        let pred = model.predict_with_material(&sample.features, sample.material);
        let target = sample.target_score;
        let diff = pred - target;
        let normalized_diff = diff / target_scale;
        squared_error += normalized_diff * normalized_diff;
        huber_loss += huber_value(normalized_diff, huber_delta);
        absolute_error += diff.abs();
        if target.abs() >= sign_threshold {
            sign_samples += 1;
            if pred.signum() == target.signum() {
                sign_correct += 1;
            }
        }
        sum_pred += pred;
        sum_target += target;
        sum_pred_sq += pred * pred;
        sum_target_sq += target * target;
        sum_cross += pred * target;
    }

    let samples = batch.len();
    if samples == 0 {
        return Metrics {
            samples: 0,
            huber: 0.0,
            rmse_cp: 0.0,
            mae_cp: 0.0,
            sign_accuracy: 0.0,
            sign_samples: 0,
            correlation: 0.0,
        };
    }

    let n = samples as f32;
    let pred_var = sum_pred_sq - sum_pred * sum_pred / n;
    let target_var = sum_target_sq - sum_target * sum_target / n;
    let covariance = sum_cross - sum_pred * sum_target / n;
    let correlation = if pred_var > 0.0 && target_var > 0.0 {
        covariance / (pred_var * target_var).sqrt()
    } else {
        0.0
    };

    Metrics {
        samples,
        huber: huber_loss / n,
        rmse_cp: (squared_error / n).sqrt() * target_scale,
        mae_cp: absolute_error / n,
        sign_accuracy: if sign_samples == 0 {
            0.0
        } else {
            sign_correct as f32 / sign_samples as f32
        },
        sign_samples,
        correlation,
    }
}

fn update_batch(
    model: &mut SparseModel,
    batch: &[Sample],
    target_scale: f32,
    huber_delta: f32,
    freeze_material: bool,
    freeze_bias: bool,
) -> (f32, usize) {
    let mut w_grads: HashMap<usize, f32> = HashMap::new();
    let mut material_grad_total = 0.0;
    let mut bias_grad_total = 0.0;
    let mut loss = 0.0;
    let mut valid_samples = 0usize;

    for sample in batch {
        let pred = model.predict_with_material(&sample.features, sample.material);
        let normalized_diff = (pred - sample.target_score) / target_scale;
        let delta = huber_grad(normalized_diff, huber_delta) / target_scale;
        if !normalized_diff.is_finite() || !delta.is_finite() {
            continue;
        }
        valid_samples += 1;
        loss += huber_value(normalized_diff, huber_delta);
        for &feature_idx in &sample.features {
            *w_grads.entry(feature_idx).or_insert(0.0) += delta;
        }
        material_grad_total += delta * sample.material;
        bias_grad_total += delta;
    }

    if valid_samples == 0 {
        return (0.0, 0);
    }

    for (i, grad) in w_grads {
        model.w[i] -= model.kpp_eta * (grad / valid_samples as f32 + model.l2_lambda * model.w[i]);
    }
    if !freeze_material {
        model.material_coeff -= model.kpp_eta
            * (material_grad_total / valid_samples as f32 + model.l2_lambda * model.material_coeff);
    }
    if !freeze_bias {
        model.bias -= model.kpp_eta * (bias_grad_total / valid_samples as f32);
    }

    (loss / valid_samples as f32, valid_samples)
}

fn print_metrics(prefix: &str, metrics: Metrics) {
    println!(
        "{} samples={} huber={:.6} rmse_cp={:.2} mae_cp={:.2} sign_acc={:.4} sign_samples={} corr={:.4}",
        prefix,
        metrics.samples,
        metrics.huber,
        metrics.rmse_cp,
        metrics.mae_cp,
        metrics.sign_accuracy,
        metrics.sign_samples,
        metrics.correlation
    );
}

fn huber_value(diff: f32, delta: f32) -> f32 {
    let abs_diff = diff.abs();
    if abs_diff <= delta {
        0.5 * diff * diff
    } else {
        delta * (abs_diff - 0.5 * delta)
    }
}

fn huber_grad(diff: f32, delta: f32) -> f32 {
    diff.clamp(-delta, delta)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.epochs == 0 {
        return Err(anyhow!("--epochs must be greater than zero"));
    }
    if args.batch_size == 0 {
        return Err(anyhow!("--batch-size must be greater than zero"));
    }
    if !args.learning_rate.is_finite() || args.learning_rate <= 0.0 {
        return Err(anyhow!("--learning-rate must be positive"));
    }
    if !args.target_scale.is_finite() || args.target_scale <= 0.0 {
        return Err(anyhow!("--target-scale must be positive"));
    }
    if !args.score_clip.is_finite() || args.score_clip <= 0.0 {
        return Err(anyhow!("--score-clip must be positive"));
    }
    if !args.huber_delta.is_finite() || args.huber_delta <= 0.0 {
        return Err(anyhow!("--huber-delta must be positive"));
    }
    if !args.sign_threshold.is_finite() || args.sign_threshold < 0.0 {
        return Err(anyhow!("--sign-threshold must be non-negative"));
    }

    let mut train = load_batch(&args.train, args.score_clip)?;
    let valid = load_batch(&args.valid, args.score_clip)?;
    let extra_valid = args
        .extra_valid
        .iter()
        .map(|spec| {
            let (label, path) = parse_extra_valid(spec)?;
            let samples = load_batch(&path, args.score_clip)?;
            Ok(NamedBatch { label, samples })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut model = SparseModel::new(args.learning_rate, 0.0);
    model.load(&args.weights)?;
    model.kpp_eta = args.learning_rate;
    let initial_material_coeff = model.material_coeff;
    let initial_bias = model.bias;

    print_metrics(
        "baseline train",
        evaluate_value(
            &model,
            &train,
            args.target_scale,
            args.huber_delta,
            args.sign_threshold,
        ),
    );
    print_metrics(
        "baseline valid",
        evaluate_value(
            &model,
            &valid,
            args.target_scale,
            args.huber_delta,
            args.sign_threshold,
        ),
    );
    for batch in &extra_valid {
        print_metrics(
            &format!("baseline extra_valid[{}]", batch.label),
            evaluate_value(
                &model,
                &batch.samples,
                args.target_scale,
                args.huber_delta,
                args.sign_threshold,
            ),
        );
    }
    if args.dry_run {
        return Ok(());
    }

    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    for epoch in 1..=args.epochs {
        train.shuffle(&mut rng);
        let mut epoch_loss = 0.0;
        let mut epoch_samples = 0usize;
        for chunk in train.chunks(args.batch_size) {
            let (loss, samples) = update_batch(
                &mut model,
                chunk,
                args.target_scale,
                args.huber_delta,
                args.freeze_material,
                args.freeze_bias,
            );
            epoch_loss += loss * samples as f32;
            epoch_samples += samples;
        }
        if args.freeze_material {
            model.material_coeff = initial_material_coeff;
        }
        if args.freeze_bias {
            model.bias = initial_bias;
        }
        let train_metrics = evaluate_value(
            &model,
            &train,
            args.target_scale,
            args.huber_delta,
            args.sign_threshold,
        );
        let valid_metrics = evaluate_value(
            &model,
            &valid,
            args.target_scale,
            args.huber_delta,
            args.sign_threshold,
        );
        println!(
            "epoch {} batch_loss={:.6} material_coeff={:.6} bias={:.6}",
            epoch,
            if epoch_samples == 0 {
                0.0
            } else {
                epoch_loss / epoch_samples as f32
            },
            model.material_coeff,
            model.bias
        );
        print_metrics("epoch train", train_metrics);
        print_metrics("epoch valid", valid_metrics);
        for batch in &extra_valid {
            print_metrics(
                &format!("epoch extra_valid[{}]", batch.label),
                evaluate_value(
                    &model,
                    &batch.samples,
                    args.target_scale,
                    args.huber_delta,
                    args.sign_threshold,
                ),
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
    if args.freeze_bias && model.bias != initial_bias {
        return Err(anyhow!(
            "bias changed despite --freeze-bias: {} -> {}",
            initial_bias,
            model.bias
        ));
    }

    model.save(&args.output)?;
    println!("saved {}", args.output.display());
    Ok(())
}
