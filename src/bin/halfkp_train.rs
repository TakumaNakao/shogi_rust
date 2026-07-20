use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::Deserialize;
use shogi_ai::evaluation::{
    extract_halfkp_features_for, HalfKpFlatModel, HalfKpHeader, HALFKP_HIDDEN, HALFKP_INPUTS,
};
use shogi_ai::utils::position_from_sfen_or_usi;
use shogi_core::Color;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const TARGET_SCALE: f32 = 1000.0;

#[derive(Parser, Debug)]
#[command(about = "Train HalfKP weights from decisive game outcomes")]
struct Args {
    #[arg(long, required = true)]
    train: PathBuf,
    #[arg(long)]
    valid: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    init: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    #[arg(long, default_value_t = 0.01)]
    learning_rate: f32,
    #[arg(long, default_value_t = 600.0)]
    temperature: f32,
    #[arg(long, default_value_t = 1e-6)]
    l2: f32,
    #[arg(long, default_value_t = 2)]
    early_stop_patience: usize,
    #[arg(long, default_value_t = 1e-4)]
    min_valid_improvement: f64,
    #[arg(long)]
    max_records: Option<usize>,
    #[arg(long)]
    max_valid_records: Option<usize>,
    #[arg(long, default_value_t = 20260715)]
    seed: u64,
    #[arg(long)]
    log: Option<PathBuf>,
}

#[derive(Deserialize)]
struct DatasetRecord {
    sfen: String,
    side_to_move: String,
    winner: Option<String>,
    result_known: Option<bool>,
}

struct Weights {
    feature_emb: Vec<f32>,
    hidden_b: [f32; HALFKP_HIDDEN],
    out_w: [f32; HALFKP_HIDDEN * 2 + 1],
    out_b: f32,
}

struct Accumulators {
    feature_emb: Vec<f32>,
    hidden_b: [f32; HALFKP_HIDDEN],
    out_w: [f32; HALFKP_HIDDEN * 2 + 1],
    out_b: f32,
}

impl Weights {
    fn new(seed: u64) -> Self {
        let mut rng = seed ^ 0x9e3779b97f4a7c15;
        let mut next = || {
            rng ^= rng << 7;
            rng ^= rng >> 9;
            rng ^= rng << 8;
            (rng as i32 as f32) / (i32::MAX as f32) * 0.01
        };
        let mut feature_emb = vec![0.0; HALFKP_INPUTS * HALFKP_HIDDEN];
        for value in &mut feature_emb {
            *value = next();
        }
        let out_w = [0.0; HALFKP_HIDDEN * 2 + 1];
        Self {
            feature_emb,
            hidden_b: [0.5; HALFKP_HIDDEN],
            out_w,
            out_b: 0.0,
        }
    }

    fn zero_accumulators(&self) -> Accumulators {
        Accumulators {
            feature_emb: vec![0.0; self.feature_emb.len()],
            hidden_b: [0.0; HALFKP_HIDDEN],
            out_w: [0.0; HALFKP_HIDDEN * 2 + 1],
            out_b: 0.0,
        }
    }

    fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("read initial weights {}", path.display()))?;
        let flat = HalfKpFlatModel::decode(&bytes)?;
        Ok(Self {
            feature_emb: flat.feature_emb,
            hidden_b: flat.hidden_b,
            out_w: flat.out_w,
            out_b: flat.out_b,
        })
    }

    fn accumulate(&self, features: &[usize]) -> [f32; HALFKP_HIDDEN] {
        let mut hidden = self.hidden_b;
        for &feature in features {
            let start = feature * HALFKP_HIDDEN;
            for h in 0..HALFKP_HIDDEN {
                hidden[h] += self.feature_emb[start + h];
            }
        }
        hidden
    }

    fn score(
        &self,
        black_features: &[usize],
        white_features: &[usize],
        black_material: f32,
        white_material: f32,
        side: Color,
    ) -> (f32, [f32; HALFKP_HIDDEN], [f32; HALFKP_HIDDEN], f32) {
        let black = self.accumulate(black_features);
        let white = self.accumulate(white_features);
        let (stm, nstm, material) = if side == Color::Black {
            (&black, &white, black_material)
        } else {
            (&white, &black, white_material)
        };
        let mut raw = self.out_b + material / 1000.0 * self.out_w[HALFKP_HIDDEN * 2];
        for h in 0..HALFKP_HIDDEN {
            raw += stm[h].clamp(0.0, 1.0) * self.out_w[h];
            raw += nstm[h].clamp(0.0, 1.0) * self.out_w[HALFKP_HIDDEN + h];
        }
        (raw * TARGET_SCALE, black, white, material)
    }
}

fn parse_side(text: &str) -> Result<Color> {
    match text {
        "black" | "b" => Ok(Color::Black),
        "white" | "w" => Ok(Color::White),
        _ => Err(anyhow!("invalid side_to_move: {text}")),
    }
}

fn target_for(record: &DatasetRecord, side: Color) -> Option<f32> {
    let Some(winner) = record.winner.as_deref() else {
        return record.result_known.filter(|known| *known).map(|_| 0.5);
    };
    match winner {
        "black" => Some((side == Color::Black) as u8 as f32),
        "white" => Some((side == Color::White) as u8 as f32),
        _ => None,
    }
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn clamp_weight(value: f32, limit: f32) -> f32 {
    value.clamp(-limit, limit)
}

fn train_record(
    weights: &mut Weights,
    accumulators: &mut Accumulators,
    record: &DatasetRecord,
    learning_rate: f32,
    temperature: f32,
    l2: f32,
) -> Result<Option<(f64, bool)>> {
    let Some(position) = position_from_sfen_or_usi(&record.sfen) else {
        return Ok(None);
    };
    let side = parse_side(&record.side_to_move)?;
    let Some(target) = target_for(record, side) else {
        return Ok(None);
    };
    let Some(black) = extract_halfkp_features_for(&position, Color::Black) else {
        return Ok(None);
    };
    let Some(white) = extract_halfkp_features_for(&position, Color::White) else {
        return Ok(None);
    };
    let (score, black_hidden, white_hidden, material) = weights.score(
        &black.features,
        &white.features,
        black.material,
        white.material,
        side,
    );
    let probability = sigmoid(score / temperature);
    let probability_clamped = probability.clamp(1e-7, 1.0 - 1e-7);
    let loss = -(target as f64) * (probability_clamped as f64).ln()
        - ((1.0 - target) as f64) * (1.0 - probability_clamped as f64).ln();
    let derivative = ((probability - target) * TARGET_SCALE / temperature).clamp(-5.0, 5.0);
    let stm_hidden = if side == Color::Black {
        &black_hidden
    } else {
        &white_hidden
    };
    let nstm_hidden = if side == Color::Black {
        &white_hidden
    } else {
        &black_hidden
    };
    let mut stm_grad = [0.0; HALFKP_HIDDEN];
    let mut nstm_grad = [0.0; HALFKP_HIDDEN];
    for h in 0..HALFKP_HIDDEN {
        stm_grad[h] = derivative * weights.out_w[h];
        nstm_grad[h] = derivative * weights.out_w[HALFKP_HIDDEN + h];
        let stm_update = derivative * stm_hidden[h].clamp(0.0, 1.0) + l2 * weights.out_w[h];
        accumulators.out_w[h] += stm_update * stm_update;
        weights.out_w[h] = clamp_weight(
            weights.out_w[h] - learning_rate * stm_update / (accumulators.out_w[h] + 1e-8).sqrt(),
            8.0,
        );
        let nstm_index = HALFKP_HIDDEN + h;
        let nstm_update =
            derivative * nstm_hidden[h].clamp(0.0, 1.0) + l2 * weights.out_w[nstm_index];
        accumulators.out_w[nstm_index] += nstm_update * nstm_update;
        weights.out_w[nstm_index] = clamp_weight(
            weights.out_w[nstm_index]
                - learning_rate * nstm_update / (accumulators.out_w[nstm_index] + 1e-8).sqrt(),
            8.0,
        );
    }
    let material_index = HALFKP_HIDDEN * 2;
    let material_update = derivative * material / 1000.0 + l2 * weights.out_w[material_index];
    accumulators.out_w[material_index] += material_update * material_update;
    weights.out_w[material_index] = clamp_weight(
        weights.out_w[material_index]
            - learning_rate * material_update / (accumulators.out_w[material_index] + 1e-8).sqrt(),
        8.0,
    );
    let bias_update = derivative + l2 * weights.out_b;
    accumulators.out_b += bias_update * bias_update;
    weights.out_b = clamp_weight(
        weights.out_b - learning_rate * bias_update / (accumulators.out_b + 1e-8).sqrt(),
        8.0,
    );
    for h in 0..HALFKP_HIDDEN {
        let grad_black = if side == Color::Black {
            stm_grad[h]
        } else {
            nstm_grad[h]
        };
        let grad_white = if side == Color::Black {
            nstm_grad[h]
        } else {
            stm_grad[h]
        };
        let hidden_update = grad_black * active_derivative(black_hidden[h])
            + grad_white * active_derivative(white_hidden[h])
            + l2 * weights.hidden_b[h];
        accumulators.hidden_b[h] += hidden_update * hidden_update;
        weights.hidden_b[h] = clamp_weight(
            weights.hidden_b[h]
                - learning_rate * hidden_update / (accumulators.hidden_b[h] + 1e-8).sqrt(),
            1.0,
        );
    }
    for &feature in &black.features {
        let start = feature * HALFKP_HIDDEN;
        for h in 0..HALFKP_HIDDEN {
            let index = start + h;
            let update = grad_black_for_side(side, &stm_grad, &nstm_grad)[h]
                * active_derivative(black_hidden[h])
                + l2 * weights.feature_emb[index];
            accumulators.feature_emb[index] += update * update;
            weights.feature_emb[index] = clamp_weight(
                weights.feature_emb[index]
                    - learning_rate * update / (accumulators.feature_emb[index] + 1e-8).sqrt(),
                1.0,
            );
        }
    }
    for &feature in &white.features {
        let start = feature * HALFKP_HIDDEN;
        for h in 0..HALFKP_HIDDEN {
            let index = start + h;
            let update = grad_white_for_side(side, &stm_grad, &nstm_grad)[h]
                * active_derivative(white_hidden[h])
                + l2 * weights.feature_emb[index];
            accumulators.feature_emb[index] += update * update;
            weights.feature_emb[index] = clamp_weight(
                weights.feature_emb[index]
                    - learning_rate * update / (accumulators.feature_emb[index] + 1e-8).sqrt(),
                1.0,
            );
        }
    }
    Ok(Some((loss, (probability >= 0.5) == (target >= 0.5))))
}

fn grad_black_for_side<'a>(
    side: Color,
    stm: &'a [f32; HALFKP_HIDDEN],
    nstm: &'a [f32; HALFKP_HIDDEN],
) -> &'a [f32; HALFKP_HIDDEN] {
    if side == Color::Black {
        stm
    } else {
        nstm
    }
}

fn grad_white_for_side<'a>(
    side: Color,
    stm: &'a [f32; HALFKP_HIDDEN],
    nstm: &'a [f32; HALFKP_HIDDEN],
) -> &'a [f32; HALFKP_HIDDEN] {
    if side == Color::Black {
        nstm
    } else {
        stm
    }
}

fn active_derivative(hidden: f32) -> f32 {
    if hidden > 0.0 && hidden < 1.0 {
        1.0
    } else {
        0.0
    }
}

fn evaluate_file(
    weights: &Weights,
    path: &Path,
    max_records: Option<usize>,
    temperature: f32,
) -> Result<(usize, f64, f64)> {
    let reader = BufReader::new(File::open(path)?);
    let mut count = 0usize;
    let mut loss = 0.0f64;
    let mut correct = 0usize;
    for line in reader.lines() {
        if max_records.is_some_and(|limit| count >= limit) {
            break;
        }
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: DatasetRecord = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let Some(position) = position_from_sfen_or_usi(&record.sfen) else {
            continue;
        };
        let side = parse_side(&record.side_to_move)?;
        let Some(target) = target_for(&record, side) else {
            continue;
        };
        let Some(black) = extract_halfkp_features_for(&position, Color::Black) else {
            continue;
        };
        let Some(white) = extract_halfkp_features_for(&position, Color::White) else {
            continue;
        };
        let (score, _, _, _) = weights.score(
            &black.features,
            &white.features,
            black.material,
            white.material,
            side,
        );
        let p = sigmoid(score / temperature).clamp(1e-7, 1.0 - 1e-7);
        loss +=
            -(target as f64) * (p as f64).ln() - ((1.0 - target) as f64) * (1.0 - p as f64).ln();
        correct += usize::from((p >= 0.5) == (target >= 0.5));
        count += 1;
    }
    if count == 0 {
        return Ok((0, 0.0, 0.0));
    }
    Ok((count, loss / count as f64, correct as f64 / count as f64))
}

fn save_weights(weights: &Weights, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(path)?);
    HalfKpFlatModel::write_parts(
        &mut writer,
        HalfKpHeader::current(TARGET_SCALE)?,
        &weights.feature_emb,
        &weights.hidden_b,
        &weights.out_w,
        weights.out_b,
    )?;
    writer.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.epochs == 0
        || !args.learning_rate.is_finite()
        || args.learning_rate <= 0.0
        || !args.temperature.is_finite()
        || args.temperature <= 0.0
        || args.l2 < 0.0
        || args.early_stop_patience == 0
        || !args.min_valid_improvement.is_finite()
        || args.min_valid_improvement < 0.0
    {
        return Err(anyhow!("invalid training hyperparameters"));
    }
    let mut weights = if let Some(path) = args.init.as_ref() {
        Weights::load(path)?
    } else {
        Weights::new(args.seed)
    };
    let mut accumulators = weights.zero_accumulators();
    let mut log = args
        .log
        .as_ref()
        .map(|path| File::create(path).map(BufWriter::new))
        .transpose()?;
    if let Some(writer) = log.as_mut() {
        writeln!(
            writer,
            "epoch,train_samples,train_bce,train_accuracy,valid_samples,valid_bce,valid_accuracy"
        )?;
    }
    let mut best_valid = f64::INFINITY;
    let mut stale_epochs = 0usize;
    for epoch in 1..=args.epochs {
        let reader = BufReader::new(
            File::open(&args.train)
                .with_context(|| format!("open train {}", args.train.display()))?,
        );
        let mut samples = 0usize;
        let mut loss = 0.0f64;
        let mut correct = 0usize;
        for line in reader.lines() {
            if args.max_records.is_some_and(|limit| samples >= limit) {
                break;
            }
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let record: DatasetRecord = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some((sample_loss, is_correct)) = train_record(
                &mut weights,
                &mut accumulators,
                &record,
                args.learning_rate,
                args.temperature,
                args.l2,
            )? {
                samples += 1;
                loss += sample_loss;
                correct += usize::from(is_correct);
            }
        }
        let train_bce = if samples == 0 {
            0.0
        } else {
            loss / samples as f64
        };
        let train_accuracy = if samples == 0 {
            0.0
        } else {
            correct as f64 / samples as f64
        };
        let (valid_samples, valid_bce, valid_accuracy) = if let Some(path) = args.valid.as_ref() {
            evaluate_file(&weights, path, args.max_valid_records, args.temperature)?
        } else {
            (0, 0.0, 0.0)
        };
        println!("epoch={epoch} train_samples={samples} train_bce={train_bce:.6} train_accuracy={train_accuracy:.4} valid_samples={valid_samples} valid_bce={valid_bce:.6} valid_accuracy={valid_accuracy:.4}");
        if let Some(writer) = log.as_mut() {
            writeln!(writer, "{epoch},{samples},{train_bce:.8},{train_accuracy:.6},{valid_samples},{valid_bce:.8},{valid_accuracy:.6}")?;
        }
        if valid_samples > 0 {
            if best_valid - valid_bce >= args.min_valid_improvement {
                best_valid = valid_bce;
                stale_epochs = 0;
                save_weights(&weights, &args.output)?;
            } else {
                stale_epochs += 1;
                if stale_epochs >= args.early_stop_patience {
                    println!(
                        "early_stop epoch={epoch} patience={}",
                        args.early_stop_patience
                    );
                    break;
                }
            }
        } else if epoch == args.epochs {
            save_weights(&weights, &args.output)?;
        }
    }
    if let Some(writer) = log.as_mut() {
        writer.flush()?;
    }
    println!("output={}", args.output.display());
    Ok(())
}
