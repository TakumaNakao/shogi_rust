use anyhow::{anyhow, Result};
use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use shogi_ai::evaluation::{calculate_material_advantage, SparseModel};
use shogi_ai::utils::position_from_sfen_or_usi;
use shogi_lib::Position;
use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process;

const SCORE_LIMIT: f32 = 100_000.0;
const DEFAULT_P95_LIMIT_CP: f32 = 50.0;
const DEFAULT_MAX_LIMIT_CP: f32 = 200.0;

#[derive(Parser, Debug)]
#[command(about = "Gate KPP weights by score-space delta on sampled positions")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    baseline_weights: PathBuf,
    #[arg(long, required = true)]
    candidate_weights: PathBuf,
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long)]
    max_positions: Option<usize>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = DEFAULT_P95_LIMIT_CP)]
    p95_limit_cp: f32,
    #[arg(long, default_value_t = DEFAULT_MAX_LIMIT_CP)]
    max_limit_cp: f32,
    #[arg(long)]
    mean_limit_cp: Option<f32>,
    #[arg(long)]
    fail_on_material_drift_cp: Option<f32>,
    #[arg(long)]
    json_output: Option<PathBuf>,
    #[arg(long, default_value_t = 20)]
    print_worst: usize,
}

#[derive(Clone, Serialize)]
struct PositionDelta {
    index: usize,
    sfen: String,
    baseline_score: f32,
    candidate_score: f32,
    delta_cp: f32,
    abs_delta_cp: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    material_advantage: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    material_term_delta_cp: Option<f32>,
}

#[derive(Serialize)]
struct MaterialSummary {
    baseline_material_coeff: f32,
    candidate_material_coeff: f32,
    material_coeff_delta: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    mean_abs_material_term_delta_cp: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p95_abs_material_term_delta_cp: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_abs_material_term_delta_cp: Option<f32>,
}

#[derive(Serialize)]
struct ScoreSummary {
    requested_positions: usize,
    used_positions: usize,
    invalid_positions: usize,
    seed: u64,
    max_positions: Option<usize>,
    mean_abs_delta_cp: f32,
    p50_abs_delta_cp: f32,
    p90_abs_delta_cp: f32,
    p95_abs_delta_cp: f32,
    p99_abs_delta_cp: f32,
    max_abs_delta_cp: f32,
    p95_limit_cp: f32,
    max_limit_cp: f32,
    mean_limit_cp: Option<f32>,
    fail_on_material_drift_cp: Option<f32>,
    p95_ok: bool,
    max_ok: bool,
    mean_ok: bool,
    material_drift_ok: bool,
    material: Option<MaterialSummary>,
}

#[derive(Serialize)]
struct GateReport {
    baseline_weights: String,
    candidate_weights: String,
    summary: ScoreSummary,
    worst_positions: Vec<PositionDelta>,
}

#[derive(Clone)]
struct PositionRecord {
    sfen: String,
    position: Position,
}

#[derive(Clone)]
struct LoadedPositions {
    records: Vec<PositionRecord>,
    requested_positions: usize,
    invalid_positions: usize,
}

fn sanitize_score(score: f32) -> f32 {
    if score == f32::INFINITY {
        SCORE_LIMIT
    } else if score == -f32::INFINITY {
        -SCORE_LIMIT
    } else {
        score.clamp(-SCORE_LIMIT, SCORE_LIMIT)
    }
}

fn percentile(mut values: Vec<f32>, p: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let idx = ((values.len() - 1) as f64 * p).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn load_model(path: &Path) -> Result<SparseModel> {
    let mut model = SparseModel::new(0.0, 0.0);
    model
        .load(path)
        .map_err(|e| anyhow!("failed to load {}: {}", path.display(), e))?;
    Ok(model)
}

fn load_positions(
    paths: &[PathBuf],
    max_positions: Option<usize>,
    seed: u64,
) -> Result<LoadedPositions> {
    let mut records = Vec::new();
    let mut requested_positions = 0usize;
    let mut invalid_positions = 0usize;

    for path in paths {
        let content = fs::read_to_string(path)
            .map_err(|e| anyhow!("failed to read {}: {}", path.display(), e))?;
        for line in content.lines() {
            let sfen = line.trim();
            if sfen.is_empty() {
                continue;
            }
            requested_positions += 1;
            if let Some(position) = position_from_sfen_or_usi(sfen) {
                records.push(PositionRecord {
                    sfen: position.to_sfen_owned(),
                    position,
                });
            } else {
                invalid_positions += 1;
            }
        }
    }

    if records.is_empty() {
        return Err(anyhow!(
            "no valid positions loaded (total lines: {}, invalid: {})",
            requested_positions,
            invalid_positions
        ));
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    records.shuffle(&mut rng);
    if let Some(max_positions) = max_positions {
        if max_positions == 0 {
            return Err(anyhow!("--max-positions must be > 0"));
        }
        records.truncate(max_positions);
    }

    Ok(LoadedPositions {
        records,
        requested_positions,
        invalid_positions,
    })
}

fn create_writer(path: &Path) -> Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(BufWriter::new(File::create(path)?))
}

fn format_fail_reasons(
    p95_ok: bool,
    max_ok: bool,
    mean_ok: bool,
    material_drift_ok: bool,
) -> Vec<&'static str> {
    let mut reasons = Vec::new();
    if !p95_ok {
        reasons.push("p95_abs_delta_cp exceeds limit");
    }
    if !max_ok {
        reasons.push("max_abs_delta_cp exceeds limit");
    }
    if !mean_ok {
        reasons.push("mean_abs_delta_cp exceeds limit");
    }
    if !material_drift_ok {
        reasons.push("material-term drift exceeds limit");
    }
    reasons
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.p95_limit_cp < 0.0 {
        return Err(anyhow!("--p95-limit-cp must be >= 0"));
    }
    if args.max_limit_cp < 0.0 {
        return Err(anyhow!("--max-limit-cp must be >= 0"));
    }
    if let Some(limit) = args.mean_limit_cp {
        if limit < 0.0 {
            return Err(anyhow!("--mean-limit-cp must be >= 0"));
        }
    }
    if let Some(limit) = args.fail_on_material_drift_cp {
        if limit < 0.0 {
            return Err(anyhow!("--fail-on-material-drift-cp must be >= 0"));
        }
    }
    if args.input.is_empty() {
        return Err(anyhow!("--input is required"));
    }

    let baseline = load_model(&args.baseline_weights)?;
    let candidate = load_model(&args.candidate_weights)?;

    let loaded = load_positions(&args.input, args.max_positions, args.seed)?;
    let records = loaded.records;
    let used_positions = records.len();

    let mut results = Vec::with_capacity(used_positions);
    let mut abs_deltas = Vec::with_capacity(used_positions);
    let mut material_term_deltas = Vec::new();

    for (idx, record) in records.iter().enumerate() {
        let baseline_score = sanitize_score(baseline.predict_from_position(&record.position));
        let candidate_score = sanitize_score(candidate.predict_from_position(&record.position));
        let delta = candidate_score - baseline_score;
        let abs_delta = delta.abs();

        let mut material_advantage = None;
        let mut material_term_delta = None;
        if baseline.material_coeff.is_finite() && candidate.material_coeff.is_finite() {
            let material = calculate_material_advantage(&record.position);
            let baseline_term = baseline.material_coeff * material;
            let candidate_term = candidate.material_coeff * material;
            let drift = (candidate_term - baseline_term).abs();
            material_advantage = Some(material);
            material_term_delta = Some(drift);
            material_term_deltas.push(drift);
        }

        abs_deltas.push(abs_delta);
        results.push(PositionDelta {
            index: idx,
            sfen: record.sfen.clone(),
            baseline_score,
            candidate_score,
            delta_cp: delta,
            abs_delta_cp: abs_delta,
            material_advantage,
            material_term_delta_cp: material_term_delta,
        });
    }

    let p50 = percentile(abs_deltas.clone(), 0.50);
    let p90 = percentile(abs_deltas.clone(), 0.90);
    let p95 = percentile(abs_deltas.clone(), 0.95);
    let p99 = percentile(abs_deltas.clone(), 0.99);
    let max_abs_delta = abs_deltas
        .iter()
        .copied()
        .fold(0.0, |acc, x| if x > acc { x } else { acc });
    let mean_abs_delta = mean(&abs_deltas);

    let p95_ok = p95 <= args.p95_limit_cp;
    let max_ok = max_abs_delta <= args.max_limit_cp;
    let mean_ok = match args.mean_limit_cp {
        Some(limit) => mean_abs_delta <= limit,
        None => true,
    };

    let material_mean_abs_term_delta = if material_term_deltas.is_empty() {
        None
    } else {
        Some(mean(&material_term_deltas))
    };
    let material_p95_term_delta = if material_term_deltas.is_empty() {
        None
    } else {
        Some(percentile(material_term_deltas.clone(), 0.95))
    };
    let material_max_term_delta = if material_term_deltas.is_empty() {
        None
    } else {
        Some(
            material_term_deltas
                .iter()
                .copied()
                .fold(0.0, |acc, x| if x > acc { x } else { acc }),
        )
    };
    let material_drift_ok = if let Some(limit) = args.fail_on_material_drift_cp {
        if let Some(max_drift) = material_max_term_delta {
            max_drift <= limit
        } else {
            true
        }
    } else {
        true
    };

    let mut worst_positions = results.clone();
    worst_positions.sort_by(|a, b| {
        b.abs_delta_cp
            .partial_cmp(&a.abs_delta_cp)
            .unwrap_or(Ordering::Equal)
    });
    if args.print_worst < worst_positions.len() {
        worst_positions.truncate(args.print_worst);
    }

    let summary = ScoreSummary {
        requested_positions: loaded.requested_positions,
        used_positions,
        invalid_positions: loaded.invalid_positions,
        seed: args.seed,
        max_positions: args.max_positions,
        mean_abs_delta_cp: mean_abs_delta,
        p50_abs_delta_cp: p50,
        p90_abs_delta_cp: p90,
        p95_abs_delta_cp: p95,
        p99_abs_delta_cp: p99,
        max_abs_delta_cp: max_abs_delta,
        p95_limit_cp: args.p95_limit_cp,
        max_limit_cp: args.max_limit_cp,
        mean_limit_cp: args.mean_limit_cp,
        fail_on_material_drift_cp: args.fail_on_material_drift_cp,
        p95_ok,
        max_ok,
        mean_ok,
        material_drift_ok,
        material: if baseline.material_coeff.is_finite() && candidate.material_coeff.is_finite() {
            Some(MaterialSummary {
                baseline_material_coeff: baseline.material_coeff,
                candidate_material_coeff: candidate.material_coeff,
                material_coeff_delta: (candidate.material_coeff - baseline.material_coeff).abs(),
                mean_abs_material_term_delta_cp: material_mean_abs_term_delta,
                p95_abs_material_term_delta_cp: material_p95_term_delta,
                max_abs_material_term_delta_cp: material_max_term_delta,
            })
        } else {
            None
        },
    };

    println!("samples={}", used_positions);
    println!(
        "mean_abs_delta_cp={:.2}, p50={:.2}, p90={:.2}, p95={:.2}, p99={:.2}, max={:.2}",
        mean_abs_delta, p50, p90, p95, p99, max_abs_delta
    );
    println!(
        "limits: p95<= {:.2}, max<= {:.2}, mean={}",
        args.p95_limit_cp,
        args.max_limit_cp,
        args.mean_limit_cp
            .map(|v| format!("{:.2}", v))
            .unwrap_or_else(|| "none".to_string())
    );
    println!(
        "material_coeff: baseline={:.6}, candidate={:.6}, delta={:.6}",
        baseline.material_coeff,
        candidate.material_coeff,
        (candidate.material_coeff - baseline.material_coeff).abs()
    );

    if let Some(path) = args.json_output.as_ref() {
        let report = GateReport {
            baseline_weights: args.baseline_weights.display().to_string(),
            candidate_weights: args.candidate_weights.display().to_string(),
            summary,
            worst_positions: worst_positions.clone(),
        };
        let mut writer = create_writer(path)?;
        serde_json::to_writer_pretty(&mut writer, &report)?;
        writeln!(writer)?;
        println!("json-output: {}", path.display());
    }

    let failed = !(p95_ok
        && max_ok
        && (!args.mean_limit_cp.is_some() || mean_ok)
        && (!args.fail_on_material_drift_cp.is_some() || material_drift_ok));
    if failed {
        println!(
            "SCORE GATE FAILED: {:?}",
            format_fail_reasons(p95_ok, max_ok, mean_ok, material_drift_ok)
        );
        process::exit(2);
    }

    println!("SCORE GATE PASSED");
    Ok(())
}
