use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Normal};
use serde::Serialize;
use shogi_ai::evaluation::{extract_halfkp_features_for, HalfKpModel};
use shogi_ai::evaluation::{HalfKpFlatModel, HalfKpHeader, HALFKP_HIDDEN, HALFKP_INPUTS};
use shogi_ai::training_data::{
    artifact_metadata, capture_run_environment, ArtifactMetadata, RunEnvironment,
};
use shogi_core::Color;
use shogi_lib::Position;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const LINEAGE_ID: &str = "halfkp64-scratch-v1";
const SCHEMA: &str = "halfkp_scratch_genesis_v1";
const INITIALIZER: &str = "independent-cha-cha20-centered-normal-v2";
const DEFAULT_TARGET_SCALE: f32 = 1000.0;
const DEFAULT_ACTIVE_FEATURES: f32 = 38.0;
const DEFAULT_FEATURE_SUM_STDDEV: f32 = 0.04;
const DEFAULT_OUTPUT_STDDEV: f32 = 0.01;

#[derive(Parser, Debug)]
#[command(about = "Create a parentless HalfKP64 model from independent random parameters")]
struct Args {
    /// New model path. Existing files are never overwritten.
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    seed: u64,
    #[arg(long, default_value_t = DEFAULT_TARGET_SCALE)]
    target_scale: f32,
    /// Deterministic legal positions used for the genesis health check.
    #[arg(long, default_value_t = 10_000)]
    validation_positions: usize,
}

#[derive(Debug, Serialize)]
struct GenesisManifest {
    schema: &'static str,
    lineage_id: &'static str,
    generation: u32,
    parent: Option<ArtifactMetadata>,
    initializer: InitializerManifest,
    model: ArtifactMetadata,
    validation: ValidationStats,
    environment: RunEnvironment,
}

#[derive(Debug, Serialize)]
struct InitializerManifest {
    id: &'static str,
    seed: u64,
    rng: &'static str,
    distribution: &'static str,
    feature_stddev: f32,
    hidden_bias: f32,
    output_stddev: f32,
    material_coefficient: f32,
    output_bias: f32,
    target_scale: f32,
}

struct Genesis {
    model: HalfKpFlatModel,
    feature_stddev: f32,
}

#[derive(Debug, Serialize)]
struct ValidationStats {
    positions: usize,
    score_mean_cp: f64,
    score_stddev_cp: f64,
    score_abs_p99_cp: f32,
    activation_below_zero_rate: f64,
    activation_above_one_rate: f64,
    reload_max_abs_delta_cp: f32,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    validate_output(&args.output)?;
    let manifest_path = manifest_path(&args.output);
    validate_output(&manifest_path)?;

    let genesis = generate(args.seed, args.target_scale)?;
    write_model_atomic(&args.output, &genesis.model)?;
    let validation = validate_genesis(
        &args.output,
        &genesis.model,
        args.seed ^ 0x5641_4c49_4441_5445,
        args.validation_positions,
    )?;
    let manifest = GenesisManifest {
        schema: SCHEMA,
        lineage_id: LINEAGE_ID,
        generation: 0,
        parent: None,
        initializer: InitializerManifest {
            id: INITIALIZER,
            seed: args.seed,
            rng: "ChaCha20Rng",
            distribution: "normal",
            feature_stddev: genesis.feature_stddev,
            hidden_bias: 0.5,
            output_stddev: DEFAULT_OUTPUT_STDDEV,
            material_coefficient: 0.0,
            output_bias: 0.0,
            target_scale: args.target_scale,
        },
        model: artifact_metadata(&args.output, None)?,
        validation,
        environment: capture_run_environment(),
    };
    write_json_atomic(&manifest_path, &manifest)?;
    println!("model: {}", args.output.display());
    println!("manifest: {}", manifest_path.display());
    println!("sha256: {}", manifest.model.sha256);
    Ok(())
}

fn validate_genesis(
    path: &Path,
    flat: &HalfKpFlatModel,
    seed: u64,
    samples: usize,
) -> Result<ValidationStats> {
    if samples == 0 {
        return Err(anyhow!("validation-positions must be greater than zero"));
    }
    let loaded = HalfKpModel::load(path)?;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut position = Position::default();
    let mut game_ply = 0usize;
    let mut scores = Vec::with_capacity(samples);
    let mut below = 0usize;
    let mut above = 0usize;
    let mut activations = 0usize;
    let mut reload_max_abs_delta_cp = 0.0f32;
    while scores.len() < samples {
        let black = extract_halfkp_features_for(&position, Color::Black)
            .ok_or_else(|| anyhow!("legal validation position has no black king"))?;
        let white = extract_halfkp_features_for(&position, Color::White)
            .ok_or_else(|| anyhow!("legal validation position has no white king"))?;
        let forward = HalfKpFlatModel::forward_parts(
            &flat.feature_emb,
            &flat.hidden_b,
            &flat.out_w,
            flat.out_b,
            black.features.iter().copied(),
            white.features.iter().copied(),
            black.material,
            white.material,
            position.side_to_move() == Color::Black,
            flat.header.target_scale,
        );
        for value in forward.black.iter().chain(forward.white.iter()) {
            below += usize::from(*value < 0.0);
            above += usize::from(*value > 1.0);
            activations += 1;
        }
        let score_cp = forward.raw * flat.header.target_scale;
        reload_max_abs_delta_cp =
            reload_max_abs_delta_cp.max((score_cp - loaded.predict_from_position(&position)).abs());
        scores.push(score_cp);

        let legal = position.legal_moves();
        game_ply += 1;
        if legal.is_empty() || game_ply >= 160 {
            position = Position::default();
            game_ply = 0;
        } else {
            position.do_move(legal[rng.gen_range(0..legal.len())]);
        }
    }
    scores.sort_by(f32::total_cmp);
    let mean = scores.iter().map(|&value| value as f64).sum::<f64>() / samples as f64;
    let variance = scores
        .iter()
        .map(|&value| {
            let centered = value as f64 - mean;
            centered * centered
        })
        .sum::<f64>()
        / samples as f64;
    let mut absolute = scores.iter().map(|value| value.abs()).collect::<Vec<_>>();
    absolute.sort_by(f32::total_cmp);
    let p99_index = ((samples - 1) * 99) / 100;
    Ok(ValidationStats {
        positions: samples,
        score_mean_cp: mean,
        score_stddev_cp: variance.sqrt(),
        score_abs_p99_cp: absolute[p99_index],
        activation_below_zero_rate: below as f64 / activations as f64,
        activation_above_one_rate: above as f64 / activations as f64,
        reload_max_abs_delta_cp,
    })
}

fn generate(seed: u64, target_scale: f32) -> Result<Genesis> {
    let header = HalfKpHeader::current(target_scale)?;
    let feature_stddev = DEFAULT_FEATURE_SUM_STDDEV / DEFAULT_ACTIVE_FEATURES.sqrt();
    let feature_distribution = Normal::new(0.0, feature_stddev as f64)?;
    let output_distribution = Normal::new(0.0, DEFAULT_OUTPUT_STDDEV as f64)?;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let feature_emb = (0..HALFKP_INPUTS * HALFKP_HIDDEN)
        .map(|_| feature_distribution.sample(&mut rng) as f32)
        .collect();
    let hidden_b = [0.5; HALFKP_HIDDEN];
    let mut out_w = [0.0; HALFKP_HIDDEN * 2 + 1];
    for value in &mut out_w[..HALFKP_HIDDEN * 2] {
        *value = output_distribution.sample(&mut rng) as f32;
    }
    // Remove the arbitrary constant offset introduced by a finite random draw.
    // Each perspective retains independently random directions while a neutral
    // activation vector maps to exactly zero.
    for bank in out_w[..HALFKP_HIDDEN * 2].chunks_exact_mut(HALFKP_HIDDEN) {
        let mean = bank.iter().sum::<f32>() / HALFKP_HIDDEN as f32;
        for value in bank {
            *value -= mean;
        }
    }
    // The final output weight is the hand-written material bypass. Scratch
    // lineage generation zero deliberately contains no such prior knowledge.
    out_w[HALFKP_HIDDEN * 2] = 0.0;
    Ok(Genesis {
        model: HalfKpFlatModel {
            header,
            feature_emb,
            hidden_b,
            out_w,
            out_b: 0.0,
        },
        feature_stddev,
    })
}

fn validate_output(path: &Path) -> Result<()> {
    if path.exists() {
        return Err(anyhow!("refusing to overwrite {}", path.display()));
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }
    Ok(())
}

fn write_model_atomic(path: &Path, model: &HalfKpFlatModel) -> Result<()> {
    let temporary = temporary_path(path);
    let mut writer = BufWriter::new(File::create(&temporary)?);
    model.write_to(&mut writer)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    fs::rename(&temporary, path)?;
    Ok(())
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = temporary_path(path);
    let mut writer = BufWriter::new(File::create(&temporary)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    fs::rename(&temporary, path)?;
    Ok(())
}

fn manifest_path(model: &Path) -> PathBuf {
    PathBuf::from(format!("{}.manifest.json", model.display()))
}

fn temporary_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.tmp", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genesis_is_deterministic_and_contains_no_material_prior() {
        let first = generate(17, DEFAULT_TARGET_SCALE).expect("first genesis");
        let second = generate(17, DEFAULT_TARGET_SCALE).expect("second genesis");
        assert_eq!(first.model.feature_emb, second.model.feature_emb);
        assert_eq!(first.model.out_w, second.model.out_w);
        assert_eq!(first.model.out_w[HALFKP_HIDDEN * 2], 0.0);
        assert_eq!(first.model.out_b, 0.0);
        assert!(first
            .model
            .feature_emb
            .iter()
            .all(|value| value.is_finite()));
        assert!(first.model.out_w.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn independent_seeds_differ() {
        let first = generate(17, DEFAULT_TARGET_SCALE).expect("first genesis");
        let second = generate(18, DEFAULT_TARGET_SCALE).expect("second genesis");
        assert_ne!(
            &first.model.feature_emb[..1024],
            &second.model.feature_emb[..1024]
        );
    }
}
