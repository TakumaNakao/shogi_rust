use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Deserialize;
use shogi_ai::evaluation::{HalfKpHeader, HALFKP_HEADER_LEN, HALFKP_HIDDEN, HALFKP_INPUTS};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const TARGET_SCALE: f32 = 1000.0;

#[derive(Parser, Debug)]
#[command(about = "Fit HalfKP scores to v2.1.0 KPP teacher values")]
struct Args {
    #[arg(long)]
    train: Option<PathBuf>,
    #[arg(long)]
    train_dir: Vec<PathBuf>,
    #[arg(long)]
    valid: Option<PathBuf>,
    #[arg(long)]
    valid_search: Option<PathBuf>,
    #[arg(long)]
    valid_random: Option<PathBuf>,
    #[arg(long)]
    valid_rank: Option<PathBuf>,
    #[arg(long)]
    rank_train: Option<PathBuf>,
    #[arg(long, default_value_t = 4)]
    rank_pairs_per_root: usize,
    #[arg(long)]
    max_rank_roots: Option<usize>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    init: Option<PathBuf>,
    #[arg(long, default_value_t = 3)]
    epochs: usize,
    #[arg(long, default_value_t = 0.003)]
    learning_rate: f32,
    #[arg(long, default_value_t = 0.0003)]
    output_learning_rate: f32,
    #[arg(long, default_value_t = 0.1)]
    huber_delta: f32,
    #[arg(long, default_value_t = 0.05)]
    output_limit: f32,
    #[arg(long, default_value_t = 2)]
    early_stop_patience: usize,
    #[arg(long, default_value_t = 0.01)]
    min_valid_improvement_cp: f32,
    #[arg(long)]
    max_records: Option<usize>,
    #[arg(long)]
    max_valid_records: Option<usize>,
    #[arg(long, default_value_t = 20260716)]
    seed: u64,
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, default_value_t = 1)]
    train_repeat: usize,
    #[arg(long)]
    checkpoint_dir: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    resume: bool,
    #[arg(long)]
    log: Option<PathBuf>,
}

#[derive(Clone, Deserialize)]
struct Record {
    features_black: Vec<usize>,
    features_white: Vec<usize>,
    material_black: f32,
    material_white: f32,
    side_to_move: String,
    static_eval: Option<f32>,
}

struct Weights {
    feature_emb: Vec<f32>,
    hidden_b: [f32; HALFKP_HIDDEN],
    out_w: [f32; HALFKP_HIDDEN * 2 + 1],
    out_b: f32,
}

struct Accum {
    feature_emb: Vec<f32>,
    hidden_b: [f32; HALFKP_HIDDEN],
    out_w: [f32; HALFKP_HIDDEN * 2 + 1],
    out_b: f32,
}

struct RecordGradient {
    loss: f64,
    error_cp: f32,
    out_w: [f32; HALFKP_HIDDEN * 2 + 1],
    out_b: f32,
    hidden_b: [f32; HALFKP_HIDDEN],
    features: Vec<(usize, [f32; HALFKP_HIDDEN])>,
}

#[derive(Default)]
struct EvalMetrics {
    samples: usize,
    mae: f64,
    rmse: f64,
    bias: f64,
    p95: f64,
    p99: f64,
    sign_mismatches: usize,
}

#[derive(Deserialize)]
struct RankGroup {
    candidates: Vec<Record>,
}

#[derive(Default)]
struct RankMetrics {
    roots: usize,
    top1_matches: usize,
    pair_correct: u64,
    pair_total: u64,
    mean_regret: f64,
    p95_regret: f64,
}

impl Weights {
    fn random(seed: u64) -> Self {
        let mut state = seed ^ 0x9e3779b97f4a7c15;
        let mut next = || {
            state ^= state << 7;
            state ^= state >> 9;
            (state as i32 as f32) / (i32::MAX as f32) * 0.005
        };
        let mut feature_emb = vec![0.0; HALFKP_INPUTS * HALFKP_HIDDEN];
        for value in &mut feature_emb {
            *value = next();
        }
        Self {
            feature_emb,
            hidden_b: [0.5; HALFKP_HIDDEN],
            out_w: [0.0; HALFKP_HIDDEN * 2 + 1],
            out_b: 0.0,
        }
    }

    fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("read initial HalfKP weights {}", path.display()))?;
        if bytes.len() < HALFKP_HEADER_LEN {
            return Err(anyhow!("invalid HalfKP weight file"));
        }
        HalfKpHeader::decode(&bytes)?;
        let count = HALFKP_INPUTS * HALFKP_HIDDEN + HALFKP_HIDDEN + HALFKP_HIDDEN * 2 + 2;
        if bytes.len() != HALFKP_HEADER_LEN + count * 4 {
            return Err(anyhow!("invalid HalfKP weight length"));
        }
        let mut offset = HALFKP_HEADER_LEN;
        let mut next = || {
            let value = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            offset += 4;
            value
        };
        let mut feature_emb = vec![0.0; HALFKP_INPUTS * HALFKP_HIDDEN];
        for value in &mut feature_emb {
            *value = next();
        }
        let hidden_b = std::array::from_fn(|_| next());
        let out_w = std::array::from_fn(|_| next());
        Ok(Self {
            feature_emb,
            hidden_b,
            out_w,
            out_b: next(),
        })
    }

    fn accum(&self, features: &[usize]) -> [f32; HALFKP_HIDDEN] {
        let mut hidden = self.hidden_b;
        for &feature in features {
            if feature >= HALFKP_INPUTS {
                continue;
            }
            let start = feature * HALFKP_HIDDEN;
            for h in 0..HALFKP_HIDDEN {
                hidden[h] += self.feature_emb[start + h];
            }
        }
        hidden
    }

    fn raw_score(&self, record: &Record) -> ([f32; HALFKP_HIDDEN], [f32; HALFKP_HIDDEN], f32) {
        let black = self.accum(&record.features_black);
        let white = self.accum(&record.features_white);
        let black_side = record.side_to_move == "black";
        let (stm, nstm, material) = if black_side {
            (&black, &white, record.material_black)
        } else {
            (&white, &black, record.material_white)
        };
        let mut raw = self.out_b + material / 1000.0 * self.out_w[HALFKP_HIDDEN * 2];
        for h in 0..HALFKP_HIDDEN {
            raw += stm[h].clamp(0.0, 1.0) * self.out_w[h];
            raw += nstm[h].clamp(0.0, 1.0) * self.out_w[HALFKP_HIDDEN + h];
        }
        (black, white, raw)
    }

    fn zero_accum(&self) -> Accum {
        Accum {
            feature_emb: vec![0.0; self.feature_emb.len()],
            hidden_b: [0.0; HALFKP_HIDDEN],
            out_w: [0.0; HALFKP_HIDDEN * 2 + 1],
            out_b: 0.0,
        }
    }
}

fn clamp_update(value: f32, limit: f32) -> f32 {
    value.clamp(-limit, limit)
}

fn active_derivative(value: f32) -> f32 {
    if value > 0.0 && value < 1.0 {
        1.0
    } else {
        0.0
    }
}

fn parse_record(line: &str) -> Option<Record> {
    let record: Record = serde_json::from_str(line).ok()?;
    record.static_eval.filter(|value| value.is_finite())?;
    Some(record)
}

#[allow(dead_code)]
fn train_record(
    weights: &mut Weights,
    accum: &mut Accum,
    record: &Record,
    args: &Args,
) -> Option<(f64, f32)> {
    let target = (record.static_eval? / TARGET_SCALE).clamp(-6.0, 6.0);
    let (black, white, prediction) = weights.raw_score(record);
    let error = prediction - target;
    let abs_error = error.abs();
    let loss = if abs_error <= args.huber_delta {
        0.5 * error * error
    } else {
        args.huber_delta * (abs_error - 0.5 * args.huber_delta)
    };
    let derivative = error.clamp(-args.huber_delta, args.huber_delta);
    let black_side = record.side_to_move == "black";
    let (stm_hidden, nstm_hidden) = if black_side {
        (&black, &white)
    } else {
        (&white, &black)
    };
    let mut stm_grad = [0.0; HALFKP_HIDDEN];
    let mut nstm_grad = [0.0; HALFKP_HIDDEN];
    for h in 0..HALFKP_HIDDEN {
        stm_grad[h] = derivative * weights.out_w[h];
        nstm_grad[h] = derivative * weights.out_w[HALFKP_HIDDEN + h];
        let a = derivative * stm_hidden[h].clamp(0.0, 1.0);
        accum.out_w[h] += a * a;
        weights.out_w[h] = clamp_update(
            weights.out_w[h] - args.output_learning_rate * a / (accum.out_w[h] + 1e-8).sqrt(),
            args.output_limit,
        );
        let index = HALFKP_HIDDEN + h;
        let b = derivative * nstm_hidden[h].clamp(0.0, 1.0);
        accum.out_w[index] += b * b;
        weights.out_w[index] = clamp_update(
            weights.out_w[index]
                - args.output_learning_rate * b / (accum.out_w[index] + 1e-8).sqrt(),
            args.output_limit,
        );
    }
    let material_index = HALFKP_HIDDEN * 2;
    let material_grad = derivative
        * if black_side {
            record.material_black / 1000.0
        } else {
            record.material_white / 1000.0
        };
    accum.out_w[material_index] += material_grad * material_grad;
    weights.out_w[material_index] = clamp_update(
        weights.out_w[material_index]
            - args.output_learning_rate * material_grad
                / (accum.out_w[material_index] + 1e-8).sqrt(),
        args.output_limit,
    );
    accum.out_b += derivative * derivative;
    weights.out_b = clamp_update(
        weights.out_b - args.output_learning_rate * derivative / (accum.out_b + 1e-8).sqrt(),
        args.output_limit,
    );

    for h in 0..HALFKP_HIDDEN {
        let hidden_grad_black = if black_side {
            stm_grad[h]
        } else {
            nstm_grad[h]
        };
        let hidden_grad_white = if black_side {
            nstm_grad[h]
        } else {
            stm_grad[h]
        };
        let gb = hidden_grad_black * active_derivative(black[h]);
        accum.hidden_b[h] += (gb + hidden_grad_white * active_derivative(white[h])).powi(2);
        weights.hidden_b[h] = clamp_update(
            weights.hidden_b[h]
                - args.learning_rate * (gb + hidden_grad_white * active_derivative(white[h]))
                    / (accum.hidden_b[h] + 1e-8).sqrt(),
            1.0,
        );
    }
    for &feature in &record.features_black {
        if feature >= HALFKP_INPUTS {
            continue;
        }
        let start = feature * HALFKP_HIDDEN;
        for h in 0..HALFKP_HIDDEN {
            let gradient = (if black_side {
                stm_grad[h]
            } else {
                nstm_grad[h]
            }) * active_derivative(black[h]);
            let index = start + h;
            accum.feature_emb[index] += gradient * gradient;
            weights.feature_emb[index] = clamp_update(
                weights.feature_emb[index]
                    - args.learning_rate * gradient / (accum.feature_emb[index] + 1e-8).sqrt(),
                1.0,
            );
        }
    }
    for &feature in &record.features_white {
        if feature >= HALFKP_INPUTS {
            continue;
        }
        let start = feature * HALFKP_HIDDEN;
        for h in 0..HALFKP_HIDDEN {
            let gradient = (if black_side {
                nstm_grad[h]
            } else {
                stm_grad[h]
            }) * active_derivative(white[h]);
            let index = start + h;
            accum.feature_emb[index] += gradient * gradient;
            weights.feature_emb[index] = clamp_update(
                weights.feature_emb[index]
                    - args.learning_rate * gradient / (accum.feature_emb[index] + 1e-8).sqrt(),
                1.0,
            );
        }
    }
    Some((loss as f64, error * TARGET_SCALE))
}

fn compute_gradient(weights: &Weights, record: &Record, args: &Args) -> Option<RecordGradient> {
    let target = (record.static_eval? / TARGET_SCALE).clamp(-6.0, 6.0);
    let (black, white, prediction) = weights.raw_score(record);
    let error = prediction - target;
    let abs_error = error.abs();
    let loss = if abs_error <= args.huber_delta {
        0.5 * error * error
    } else {
        args.huber_delta * (abs_error - 0.5 * args.huber_delta)
    };
    let derivative = error.clamp(-args.huber_delta, args.huber_delta);
    let black_side = record.side_to_move == "black";
    let (stm_hidden, nstm_hidden) = if black_side {
        (&black, &white)
    } else {
        (&white, &black)
    };
    let mut out_w = [0.0; HALFKP_HIDDEN * 2 + 1];
    let mut stm_grad = [0.0; HALFKP_HIDDEN];
    let mut nstm_grad = [0.0; HALFKP_HIDDEN];
    for h in 0..HALFKP_HIDDEN {
        stm_grad[h] = derivative * weights.out_w[h];
        nstm_grad[h] = derivative * weights.out_w[HALFKP_HIDDEN + h];
        out_w[h] = derivative * stm_hidden[h].clamp(0.0, 1.0);
        out_w[HALFKP_HIDDEN + h] = derivative * nstm_hidden[h].clamp(0.0, 1.0);
    }
    out_w[HALFKP_HIDDEN * 2] = derivative
        * if black_side {
            record.material_black / 1000.0
        } else {
            record.material_white / 1000.0
        };
    let mut hidden_b = [0.0; HALFKP_HIDDEN];
    let mut black_gradient = [0.0; HALFKP_HIDDEN];
    let mut white_gradient = [0.0; HALFKP_HIDDEN];
    for h in 0..HALFKP_HIDDEN {
        black_gradient[h] = (if black_side {
            stm_grad[h]
        } else {
            nstm_grad[h]
        }) * active_derivative(black[h]);
        white_gradient[h] = (if black_side {
            nstm_grad[h]
        } else {
            stm_grad[h]
        }) * active_derivative(white[h]);
        hidden_b[h] = black_gradient[h] + white_gradient[h];
    }
    let mut features =
        Vec::with_capacity(record.features_black.len() + record.features_white.len());
    for &feature in &record.features_black {
        if feature < HALFKP_INPUTS {
            features.push((feature, black_gradient));
        }
    }
    for &feature in &record.features_white {
        if feature < HALFKP_INPUTS {
            features.push((feature, white_gradient));
        }
    }
    Some(RecordGradient {
        loss: loss as f64,
        error_cp: error * TARGET_SCALE,
        out_w,
        out_b: derivative,
        hidden_b,
        features,
    })
}

fn apply_gradient(
    weights: &mut Weights,
    accum: &mut Accum,
    gradient: &RecordGradient,
    args: &Args,
) {
    for index in 0..weights.out_w.len() {
        let value = gradient.out_w[index];
        accum.out_w[index] += value * value;
        weights.out_w[index] = clamp_update(
            weights.out_w[index]
                - args.output_learning_rate * value / (accum.out_w[index] + 1e-8).sqrt(),
            args.output_limit,
        );
    }
    accum.out_b += gradient.out_b * gradient.out_b;
    weights.out_b = clamp_update(
        weights.out_b - args.output_learning_rate * gradient.out_b / (accum.out_b + 1e-8).sqrt(),
        args.output_limit,
    );
    for h in 0..HALFKP_HIDDEN {
        let value = gradient.hidden_b[h];
        accum.hidden_b[h] += value * value;
        weights.hidden_b[h] = clamp_update(
            weights.hidden_b[h] - args.learning_rate * value / (accum.hidden_b[h] + 1e-8).sqrt(),
            1.0,
        );
    }
    for (feature, values) in &gradient.features {
        let start = feature * HALFKP_HIDDEN;
        for h in 0..HALFKP_HIDDEN {
            let index = start + h;
            let value = values[h];
            accum.feature_emb[index] += value * value;
            weights.feature_emb[index] = clamp_update(
                weights.feature_emb[index]
                    - args.learning_rate * value / (accum.feature_emb[index] + 1e-8).sqrt(),
                1.0,
            );
        }
    }
}

fn evaluate_file(weights: &Weights, path: &Path, limit: Option<usize>) -> Result<EvalMetrics> {
    let reader = BufReader::new(File::open(path)?);
    let mut count = 0usize;
    let mut abs_sum = 0.0f64;
    let mut square_sum = 0.0f64;
    let mut bias_sum = 0.0f64;
    let mut absolute_errors = Vec::new();
    let mut sign_mismatches = 0usize;
    for line in reader.lines() {
        if limit.is_some_and(|value| count >= value) {
            break;
        }
        let line = line?;
        let Some(record) = parse_record(&line) else {
            continue;
        };
        let target = record.static_eval.unwrap() / TARGET_SCALE;
        let (_, _, prediction) = weights.raw_score(&record);
        let error_cp = (prediction - target) * TARGET_SCALE;
        abs_sum += error_cp.abs() as f64;
        square_sum += (error_cp * error_cp) as f64;
        bias_sum += error_cp as f64;
        absolute_errors.push(error_cp.abs() as f64);
        if target.abs() > 1e-6 && prediction.signum() != target.signum() {
            sign_mismatches += 1;
        }
        count += 1;
    }
    if count == 0 {
        return Ok(EvalMetrics::default());
    }
    absolute_errors.sort_by(|a, b| a.total_cmp(b));
    let percentile = |fraction: f64| {
        let index = ((count - 1) as f64 * fraction).round() as usize;
        absolute_errors[index]
    };
    Ok(EvalMetrics {
        samples: count,
        mae: abs_sum / count as f64,
        rmse: (square_sum / count as f64).sqrt(),
        bias: bias_sum / count as f64,
        p95: percentile(0.95),
        p99: percentile(0.99),
        sign_mismatches,
    })
}

fn evaluate_rank_file(weights: &Weights, path: &Path, limit: Option<usize>) -> Result<RankMetrics> {
    let reader = BufReader::new(File::open(path)?);
    let mut metrics = RankMetrics::default();
    let mut regret_sum = 0.0f64;
    let mut regrets = Vec::new();
    for line in reader.lines() {
        if limit.is_some_and(|value| metrics.roots >= value) {
            break;
        }
        let group: RankGroup = serde_json::from_str(&line?)?;
        let candidates = group
            .candidates
            .iter()
            .filter_map(|record| {
                let teacher = record.static_eval?;
                let (_, _, student) = weights.raw_score(record);
                Some((teacher, student * TARGET_SCALE))
            })
            .collect::<Vec<_>>();
        if candidates.len() < 2 {
            continue;
        }
        let teacher_best = candidates
            .iter()
            .enumerate()
            .min_by(|a, b| a.1 .0.total_cmp(&b.1 .0))
            .map(|(index, _)| index)
            .unwrap();
        let student_best = candidates
            .iter()
            .enumerate()
            .min_by(|a, b| a.1 .1.total_cmp(&b.1 .1))
            .map(|(index, _)| index)
            .unwrap();
        metrics.top1_matches += usize::from(teacher_best == student_best);
        let regret = (candidates[student_best].0 - candidates[teacher_best].0).max(0.0) as f64;
        regret_sum += regret;
        regrets.push(regret);
        for left in 0..candidates.len() {
            for right in left + 1..candidates.len() {
                let teacher_order = candidates[left].0.total_cmp(&candidates[right].0);
                if teacher_order == std::cmp::Ordering::Equal {
                    continue;
                }
                let student_order = candidates[left].1.total_cmp(&candidates[right].1);
                metrics.pair_correct += u64::from(student_order == teacher_order);
                metrics.pair_total += 1;
            }
        }
        metrics.roots += 1;
    }
    if metrics.roots > 0 {
        metrics.mean_regret = regret_sum / metrics.roots as f64;
        regrets.sort_by(|a, b| a.total_cmp(b));
        metrics.p95_regret = regrets[((regrets.len() - 1) as f64 * 0.95).round() as usize];
    }
    Ok(metrics)
}

fn train_rank_file(
    weights: &mut Weights,
    accum: &mut Accum,
    path: &Path,
    args: &Args,
) -> Result<(usize, usize, f64)> {
    let reader = BufReader::new(File::open(path)?);
    let mut roots = 0usize;
    let mut pairs = 0usize;
    let mut delta_abs_sum = 0.0f64;
    for line in reader.lines() {
        if args.max_rank_roots.is_some_and(|limit| roots >= limit) {
            break;
        }
        let group: RankGroup = serde_json::from_str(&line?)?;
        let candidates = group
            .candidates
            .into_iter()
            .filter(|record| record.static_eval.is_some_and(f32::is_finite))
            .collect::<Vec<_>>();
        if candidates.len() < 2 {
            continue;
        }
        let teacher_best = candidates
            .iter()
            .enumerate()
            .min_by(|a, b| {
                a.1.static_eval
                    .unwrap()
                    .total_cmp(&b.1.static_eval.unwrap())
            })
            .map(|(index, _)| index)
            .unwrap();
        let predictions = candidates
            .iter()
            .map(|record| weights.raw_score(record).2)
            .collect::<Vec<_>>();
        let student_best = predictions
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .unwrap();
        let mut alternatives = candidates
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != teacher_best)
            .collect::<Vec<_>>();
        alternatives.sort_by(|a, b| {
            a.1.static_eval
                .unwrap()
                .total_cmp(&b.1.static_eval.unwrap())
        });
        let mut selected = alternatives
            .into_iter()
            .take(args.rank_pairs_per_root)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if student_best != teacher_best && !selected.contains(&student_best) {
            selected.push(student_best);
        }
        let mut seen = HashSet::new();
        for other_index in selected {
            if !seen.insert(other_index) {
                continue;
            }
            let best = &candidates[teacher_best];
            let other = &candidates[other_index];
            let best_prediction = predictions[teacher_best];
            let other_prediction = predictions[other_index];
            let teacher_delta =
                (best.static_eval.unwrap() - other.static_eval.unwrap()) / TARGET_SCALE;
            let pair_error = (best_prediction - other_prediction) - teacher_delta;
            delta_abs_sum += (pair_error * TARGET_SCALE).abs() as f64;
            let correction = pair_error.clamp(-args.huber_delta * 2.0, args.huber_delta * 2.0);
            let mut best_target = best.clone();
            best_target.static_eval = Some((best_prediction - correction * 0.5) * TARGET_SCALE);
            let mut other_target = other.clone();
            other_target.static_eval = Some((other_prediction + correction * 0.5) * TARGET_SCALE);
            if let Some(gradient) = compute_gradient(weights, &best_target, args) {
                apply_gradient(weights, accum, &gradient, args);
            }
            if let Some(gradient) = compute_gradient(weights, &other_target, args) {
                apply_gradient(weights, accum, &gradient, args);
            }
            pairs += 1;
        }
        roots += 1;
    }
    Ok((roots, pairs, delta_abs_sum / pairs.max(1) as f64))
}

fn save(weights: &Weights, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(path)?);
    HalfKpHeader::current(TARGET_SCALE)?.write_to(&mut writer)?;
    for value in &weights.feature_emb {
        writer.write_all(&value.to_le_bytes())?;
    }
    for value in weights.hidden_b {
        writer.write_all(&value.to_le_bytes())?;
    }
    for value in weights.out_w {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.write_all(&weights.out_b.to_le_bytes())?;
    writer.flush()?;
    Ok(())
}

fn collect_jsonl(path: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_file() {
        if path.extension().is_some_and(|ext| ext == "jsonl") {
            output.push(path.to_path_buf());
        }
        return Ok(());
    }
    for entry in fs::read_dir(path).with_context(|| format!("read directory {}", path.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_jsonl(&path, output)?;
        } else if path.extension().is_some_and(|ext| ext == "jsonl") {
            output.push(path);
        }
    }
    Ok(())
}

fn training_paths(args: &Args) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    if let Some(path) = &args.train {
        collect_jsonl(path, &mut paths)?;
    }
    for path in &args.train_dir {
        collect_jsonl(path, &mut paths)?;
    }
    if paths.is_empty() {
        return Err(anyhow!("no training JSONL files found"));
    }
    paths.sort();
    let base = paths.clone();
    for _ in 1..args.train_repeat {
        paths.extend(base.iter().cloned());
    }
    Ok(paths)
}

const OPT_MAGIC: &[u8; 8] = b"HKOPT001";

fn save_optimizer(
    accum: &Accum,
    epoch: usize,
    best_valid: f64,
    stale: usize,
    path: &Path,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(OPT_MAGIC)?;
    writer.write_all(&(HALFKP_HIDDEN as u32).to_le_bytes())?;
    writer.write_all(&(HALFKP_INPUTS as u32).to_le_bytes())?;
    writer.write_all(&(epoch as u64).to_le_bytes())?;
    writer.write_all(&best_valid.to_le_bytes())?;
    writer.write_all(&(stale as u64).to_le_bytes())?;
    for value in &accum.feature_emb {
        writer.write_all(&value.to_le_bytes())?;
    }
    for value in accum.hidden_b {
        writer.write_all(&value.to_le_bytes())?;
    }
    for value in accum.out_w {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.write_all(&accum.out_b.to_le_bytes())?;
    writer.flush()?;
    Ok(())
}

fn read_f32(bytes: &[u8], offset: &mut usize) -> Result<f32> {
    let end = *offset + 4;
    let chunk = bytes
        .get(*offset..end)
        .ok_or_else(|| anyhow!("truncated optimizer checkpoint"))?;
    *offset = end;
    Ok(f32::from_le_bytes(chunk.try_into().unwrap()))
}

fn load_optimizer(path: &Path, weights: &Weights) -> Result<(Accum, usize, f64, usize)> {
    let bytes = fs::read(path)?;
    if bytes.len() < 40 || &bytes[..8] != OPT_MAGIC {
        return Err(anyhow!("invalid optimizer checkpoint"));
    }
    let hidden = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let inputs = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let epoch = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    let best_valid = f64::from_le_bytes(bytes[24..32].try_into().unwrap());
    let stale = u64::from_le_bytes(bytes[32..40].try_into().unwrap()) as usize;
    if hidden != HALFKP_HIDDEN || inputs != HALFKP_INPUTS {
        return Err(anyhow!("optimizer checkpoint dimension mismatch"));
    }
    let mut offset = 40;
    let mut accum = weights.zero_accum();
    for value in &mut accum.feature_emb {
        *value = read_f32(&bytes, &mut offset)?;
    }
    for value in &mut accum.hidden_b {
        *value = read_f32(&bytes, &mut offset)?;
    }
    for value in &mut accum.out_w {
        *value = read_f32(&bytes, &mut offset)?;
    }
    accum.out_b = read_f32(&bytes, &mut offset)?;
    if offset != bytes.len() {
        return Err(anyhow!("invalid optimizer checkpoint length"));
    }
    Ok((accum, epoch, best_valid, stale))
}

fn save_checkpoint(
    weights: &Weights,
    accum: &Accum,
    epoch: usize,
    best_valid: f64,
    stale: usize,
    dir: &Path,
) -> Result<()> {
    fs::create_dir_all(dir)?;
    let weights_tmp = dir.join("latest.bin.tmp");
    let optimizer_tmp = dir.join("optimizer.bin.tmp");
    save(weights, &weights_tmp)?;
    save_optimizer(accum, epoch, best_valid, stale, &optimizer_tmp)?;
    fs::rename(weights_tmp, dir.join("latest.bin"))?;
    fs::rename(optimizer_tmp, dir.join("optimizer.bin"))?;
    Ok(())
}

fn composite_score(
    mainline: &EvalMetrics,
    search: &EvalMetrics,
    random: &EvalMetrics,
    rank: &RankMetrics,
) -> f64 {
    if search.samples == 0 || random.samples == 0 {
        return mainline.mae;
    }
    let base = 0.35 * mainline.mae
        + 0.35 * search.mae
        + 0.15 * random.mae
        + 0.10 * search.p95
        + 0.05 * random.p95;
    if rank.roots == 0 {
        base
    } else {
        0.8 * base + 0.2 * rank.mean_regret
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.epochs == 0
        || args.learning_rate <= 0.0
        || args.output_learning_rate <= 0.0
        || args.huber_delta <= 0.0
        || args.output_limit <= 0.0
        || args.early_stop_patience == 0
        || args.batch_size == 0
        || args.train_repeat == 0
    {
        return Err(anyhow!("invalid training parameters"));
    }
    let mut paths = training_paths(&args)?;
    let pool = if args.threads > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(args.threads)
                .build()?,
        )
    } else {
        None
    };
    let (mut weights, mut accum, start_epoch, mut best_valid, mut stale) = if args.resume {
        let dir = args
            .checkpoint_dir
            .as_ref()
            .ok_or_else(|| anyhow!("--resume requires --checkpoint-dir"))?;
        let weights = Weights::load(&dir.join("latest.bin"))?;
        let (accum, epoch, best_valid, stale) =
            load_optimizer(&dir.join("optimizer.bin"), &weights)?;
        (weights, accum, epoch + 1, best_valid, stale)
    } else {
        let weights = args.init.as_ref().map_or_else(
            || Ok(Weights::random(args.seed)),
            |path| Weights::load(path),
        )?;
        let accum = weights.zero_accum();
        (weights, accum, 1, f64::INFINITY, 0)
    };
    let log_existed = args.log.as_ref().is_some_and(|path| path.exists());
    let mut log = args.log.as_ref().map_or(Ok(None), |path| {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(args.resume)
            .truncate(!args.resume)
            .write(true)
            .open(path)?;
        Ok::<_, anyhow::Error>(Some(BufWriter::new(file)))
    })?;
    if let Some(writer) = log.as_mut().filter(|_| !args.resume || !log_existed) {
        writeln!(
            writer,
            "epoch,train_samples,train_mae,rank_train_roots,rank_train_pairs,rank_delta_mae,main_mae,main_rmse,main_p95,main_p99,search_mae,search_rmse,search_p95,search_p99,random_mae,random_rmse,random_p95,random_p99,rank_top1,rank_pair_accuracy,rank_regret,rank_p95_regret,composite"
        )?;
    }
    let final_epoch = start_epoch.saturating_add(args.epochs).saturating_sub(1);
    for epoch in start_epoch..=final_epoch {
        let mut rng = ChaCha8Rng::seed_from_u64(args.seed ^ epoch as u64);
        paths.shuffle(&mut rng);
        let mut samples = 0usize;
        let mut loss = 0.0f64;
        let mut train_abs = 0.0f64;
        let mut batch = Vec::with_capacity(args.batch_size);
        for path in &paths {
            let reader = BufReader::new(File::open(path)?);
            for line in reader.lines() {
                if args.max_records.is_some_and(|limit| samples >= limit) {
                    break;
                }
                let line = line?;
                let Some(record) = parse_record(&line) else {
                    continue;
                };
                batch.push(record);
                if batch.len() < args.batch_size {
                    continue;
                }
                batch.shuffle(&mut rng);
                let compute = || {
                    batch
                        .par_iter()
                        .filter_map(|record| compute_gradient(&weights, record, &args))
                        .collect::<Vec<_>>()
                };
                let gradients = if let Some(pool) = &pool {
                    pool.install(compute)
                } else {
                    compute()
                };
                for gradient in &gradients {
                    apply_gradient(&mut weights, &mut accum, gradient, &args);
                    loss += gradient.loss;
                    train_abs += gradient.error_cp.abs() as f64;
                    samples += 1;
                }
                batch.clear();
            }
            if args.max_records.is_some_and(|limit| samples >= limit) {
                break;
            }
        }
        if !batch.is_empty() {
            batch.shuffle(&mut rng);
            let compute = || {
                batch
                    .par_iter()
                    .filter_map(|record| compute_gradient(&weights, record, &args))
                    .collect::<Vec<_>>()
            };
            let gradients = if let Some(pool) = &pool {
                pool.install(compute)
            } else {
                compute()
            };
            for gradient in &gradients {
                apply_gradient(&mut weights, &mut accum, gradient, &args);
                loss += gradient.loss;
                train_abs += gradient.error_cp.abs() as f64;
                samples += 1;
            }
        }
        let (rank_train_roots, rank_train_pairs, rank_delta_mae) =
            if let Some(path) = args.rank_train.as_ref() {
                train_rank_file(&mut weights, &mut accum, path, &args)?
            } else {
                (0, 0, 0.0)
            };
        let mainline = if let Some(path) = args.valid.as_ref() {
            evaluate_file(&weights, path, args.max_valid_records)?
        } else {
            EvalMetrics::default()
        };
        let search = if let Some(path) = args.valid_search.as_ref() {
            evaluate_file(&weights, path, args.max_valid_records)?
        } else {
            EvalMetrics::default()
        };
        let random = if let Some(path) = args.valid_random.as_ref() {
            evaluate_file(&weights, path, args.max_valid_records)?
        } else {
            EvalMetrics::default()
        };
        let rank = if let Some(path) = args.valid_rank.as_ref() {
            evaluate_rank_file(&weights, path, args.max_valid_records)?
        } else {
            RankMetrics::default()
        };
        let rank_top1 = rank.top1_matches as f64 / rank.roots.max(1) as f64;
        let rank_pair_accuracy = rank.pair_correct as f64 / rank.pair_total.max(1) as f64;
        let score = composite_score(&mainline, &search, &random, &rank);
        let train_mae = if samples == 0 {
            0.0
        } else {
            train_abs / samples as f64
        };
        println!("epoch={epoch} train_samples={samples} train_huber={:.6} train_mae={train_mae:.3} rank_train_roots={rank_train_roots} rank_train_pairs={rank_train_pairs} rank_delta_mae={rank_delta_mae:.3} main_mae={:.3} main_rmse={:.3} main_bias={:.3} main_p95={:.3} main_p99={:.3} main_sign={} search_mae={:.3} search_rmse={:.3} search_bias={:.3} search_p95={:.3} search_p99={:.3} search_sign={} random_mae={:.3} random_rmse={:.3} random_bias={:.3} random_p95={:.3} random_p99={:.3} random_sign={} rank_top1={rank_top1:.4} rank_pair={rank_pair_accuracy:.4} rank_regret={:.3} rank_p95_regret={:.3} composite={score:.3}", loss / samples.max(1) as f64, mainline.mae, mainline.rmse, mainline.bias, mainline.p95, mainline.p99, mainline.sign_mismatches, search.mae, search.rmse, search.bias, search.p95, search.p99, search.sign_mismatches, random.mae, random.rmse, random.bias, random.p95, random.p99, random.sign_mismatches, rank.mean_regret, rank.p95_regret);
        if let Some(writer) = log.as_mut() {
            writeln!(writer, "{epoch},{samples},{train_mae:.6},{rank_train_roots},{rank_train_pairs},{rank_delta_mae:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{rank_top1:.6},{rank_pair_accuracy:.6},{:.6},{:.6},{score:.6}", mainline.mae, mainline.rmse, mainline.p95, mainline.p99, search.mae, search.rmse, search.p95, search.p99, random.mae, random.rmse, random.p95, random.p99, rank.mean_regret, rank.p95_regret)?;
            writer.flush()?;
        }
        let mut should_stop = false;
        if mainline.samples > 0 && best_valid - score >= args.min_valid_improvement_cp as f64 {
            best_valid = score;
            stale = 0;
            save(&weights, &args.output)?;
        } else if mainline.samples > 0 {
            stale += 1;
            if stale >= args.early_stop_patience {
                should_stop = true;
            }
        } else if epoch == final_epoch {
            save(&weights, &args.output)?;
        }
        if let Some(dir) = args.checkpoint_dir.as_ref() {
            save_checkpoint(&weights, &accum, epoch, best_valid, stale, dir)?;
        }
        if should_stop {
            break;
        }
    }
    if let Some(writer) = log.as_mut() {
        writer.flush()?;
    }
    println!("output={}", args.output.display());
    Ok(())
}
