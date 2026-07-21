use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use shogi_ai::evaluation::{HalfKpFlatModel, HalfKpHeader, HALFKP_HIDDEN};
use shogi_ai::halfkp_training::{
    CurriculumReader, CurriculumRecord, LabelSource, PackedHalfKpPosition, PreferenceKind,
    ProofStatus,
};
use shogi_ai::training_data::{
    artifact_metadata, capture_run_environment, sha256_file, sha256_hex,
};
use shogi_core::Color;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const LINEAGE_ID: &str = "halfkp64-scratch-v1";
const TARGET_SCALE: f32 = 1000.0;
const EPSILON: f32 = 1e-8;

#[derive(Parser, Debug)]
#[command(about = "Train HalfKP64 only from an HKCL scratch lineage")]
struct Args {
    #[arg(long, required = true)]
    train: Vec<PathBuf>,
    #[arg(long, required = true)]
    valid: Vec<PathBuf>,
    #[arg(long)]
    init: PathBuf,
    #[arg(long)]
    init_manifest: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 8)]
    epochs: usize,
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
    #[arg(long, default_value_t = 0.02)]
    learning_rate: f32,
    #[arg(long, default_value_t = 600.0)]
    kappa_cp: f32,
    #[arg(long, default_value_t = 300.0)]
    rank_temperature_cp: f32,
    #[arg(long, default_value_t = 1.0)]
    rank_weight: f32,
    #[arg(long, default_value_t = 1.0)]
    proof_weight: f32,
    #[arg(long, default_value_t = 1e-6)]
    l2: f32,
    #[arg(long)]
    max_train_records: Option<usize>,
    #[arg(long)]
    max_valid_records: Option<usize>,
    #[arg(long)]
    checkpoint_dir: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    resume: bool,
}

#[derive(Deserialize)]
struct ParentManifest {
    lineage_id: String,
    generation: u32,
    parent: Option<serde_json::Value>,
    model: ParentArtifact,
}

#[derive(Deserialize)]
struct ParentArtifact {
    sha256: String,
}

#[derive(Serialize)]
struct OutputManifest {
    schema: &'static str,
    lineage_id: &'static str,
    generation: u32,
    parent: ParentReference,
    parent_model_sha256: String,
    parent_manifest_sha256: String,
    train_dataset_sha256: Vec<String>,
    valid_dataset_sha256: Vec<String>,
    hyperparameters: serde_json::Value,
    best_epoch: usize,
    best_valid_loss: f64,
    model: shogi_ai::training_data::ArtifactMetadata,
    environment: shogi_ai::training_data::RunEnvironment,
}

#[derive(Serialize)]
struct ParentReference {
    model_sha256: String,
    manifest_sha256: String,
}

#[derive(Deserialize, Serialize)]
struct CheckpointMeta {
    schema: String,
    fingerprint: String,
    completed_epoch: usize,
    best_epoch: usize,
    best_valid_loss: f64,
    model_sha256: String,
    optimizer_sha256: String,
}

#[derive(Clone)]
struct Weights {
    feature: Vec<f32>,
    hidden_b: [f32; HALFKP_HIDDEN],
    out_w: [f32; HALFKP_HIDDEN * 2 + 1],
    out_b: f32,
}

#[derive(Clone)]
struct Optimizer {
    feature: Vec<f32>,
    hidden_b: [f32; HALFKP_HIDDEN],
    out_w: [f32; HALFKP_HIDDEN * 2 + 1],
    out_b: f32,
}

struct Gradient {
    feature: HashMap<usize, [f32; HALFKP_HIDDEN]>,
    hidden_b: [f32; HALFKP_HIDDEN],
    out_w: [f32; HALFKP_HIDDEN * 2 + 1],
    out_b: f32,
    loss: f64,
    objectives: usize,
}

impl Default for Gradient {
    fn default() -> Self {
        Self {
            feature: HashMap::new(),
            hidden_b: [0.0; HALFKP_HIDDEN],
            out_w: [0.0; HALFKP_HIDDEN * 2 + 1],
            out_b: 0.0,
            loss: 0.0,
            objectives: 0,
        }
    }
}

struct Forward {
    black: [f32; HALFKP_HIDDEN],
    white: [f32; HALFKP_HIDDEN],
    raw: f32,
}

#[derive(Default)]
struct Metrics {
    loss: f64,
    objectives: usize,
    records: usize,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let parent = validate_parent(&args.init, &args.init_manifest)?;
    validate_datasets(&args.train, false)?;
    validate_datasets(&args.valid, false)?;
    let fingerprint = training_fingerprint(&args, &parent)?;
    let (mut weights, mut optimizer, start_epoch, mut best_epoch, mut best_loss) =
        initialize_training(&args, &fingerprint)?;
    let initial = evaluate(&weights, &args.valid, args.max_valid_records, &args)?;
    print_metrics("initial", 0, &initial);
    if start_epoch == 0 {
        best_loss = mean_loss(&initial);
        save_weights(&weights, &args.output)?;
    }

    for epoch in start_epoch + 1..=args.epochs {
        let train = train_epoch(&mut weights, &mut optimizer, &args, epoch)?;
        print_metrics("train", epoch, &train);
        let valid = evaluate(&weights, &args.valid, args.max_valid_records, &args)?;
        print_metrics("valid", epoch, &valid);
        let loss = mean_loss(&valid);
        if loss < best_loss {
            best_loss = loss;
            best_epoch = epoch;
            save_weights(&weights, &args.output)?;
        }
        if let Some(directory) = &args.checkpoint_dir {
            save_checkpoint(
                directory,
                &fingerprint,
                epoch,
                best_epoch,
                best_loss,
                &weights,
                &optimizer,
            )?;
        }
    }
    let parent_manifest_sha256 = sha256_file(&args.init_manifest)?;
    let parent_model_sha256 = parent.model.sha256;
    let manifest = OutputManifest {
        schema: "halfkp_scratch_training_v1",
        lineage_id: LINEAGE_ID,
        generation: parent.generation + 1,
        parent: ParentReference {
            model_sha256: parent_model_sha256.clone(),
            manifest_sha256: parent_manifest_sha256.clone(),
        },
        parent_model_sha256,
        parent_manifest_sha256,
        train_dataset_sha256: args
            .train
            .iter()
            .map(|path| sha256_file(path))
            .collect::<Result<_>>()?,
        valid_dataset_sha256: args
            .valid
            .iter()
            .map(|path| sha256_file(path))
            .collect::<Result<_>>()?,
        hyperparameters: serde_json::json!({
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "kappa_cp": args.kappa_cp,
            "rank_temperature_cp": args.rank_temperature_cp, "rank_weight": args.rank_weight,
            "proof_weight": args.proof_weight, "l2": args.l2,
        }),
        best_epoch,
        best_valid_loss: best_loss,
        model: artifact_metadata(&args.output, None)?,
        environment: capture_run_environment(),
    };
    write_json_atomic(&manifest_path(&args.output), &manifest)?;
    Ok(())
}

impl Weights {
    fn from_flat(flat: HalfKpFlatModel) -> Self {
        Self {
            feature: flat.feature_emb,
            hidden_b: flat.hidden_b,
            out_w: flat.out_w,
            out_b: flat.out_b,
        }
    }

    fn forward(&self, position: &PackedHalfKpPosition) -> Forward {
        let value = HalfKpFlatModel::forward_parts(
            &self.feature,
            &self.hidden_b,
            &self.out_w,
            self.out_b,
            position.features_black.iter().map(|&x| x as usize),
            position.features_white.iter().map(|&x| x as usize),
            position.material_black,
            position.material_white,
            position.side_to_move == Color::Black,
            TARGET_SCALE,
        );
        Forward {
            black: value.black,
            white: value.white,
            raw: value.raw,
        }
    }
}

impl Optimizer {
    fn zeros(features: usize) -> Self {
        Self {
            feature: vec![0.0; features],
            hidden_b: [0.0; HALFKP_HIDDEN],
            out_w: [0.0; HALFKP_HIDDEN * 2 + 1],
            out_b: 0.0,
        }
    }
}

fn validate_args(args: &Args) -> Result<()> {
    if args.epochs == 0
        || args.batch_size == 0
        || args.learning_rate <= 0.0
        || args.kappa_cp <= 0.0
        || args.rank_temperature_cp <= 0.0
        || args.rank_weight < 0.0
        || args.proof_weight < 0.0
        || args.l2 < 0.0
        || (!args.resume && args.output.exists())
        || manifest_path(&args.output).exists()
        || (args.resume && args.checkpoint_dir.is_none())
    {
        return Err(anyhow!("invalid arguments or output already exists"));
    }
    Ok(())
}

fn initialize_training(
    args: &Args,
    fingerprint: &str,
) -> Result<(Weights, Optimizer, usize, usize, f64)> {
    if args.resume {
        let directory = args.checkpoint_dir.as_ref().expect("validated checkpoint");
        let meta: CheckpointMeta =
            serde_json::from_slice(&fs::read(directory.join("state.json"))?)?;
        if meta.schema != "halfkp_scratch_checkpoint_v1" || meta.fingerprint != fingerprint {
            return Err(anyhow!(
                "checkpoint does not match this scratch training run"
            ));
        }
        let model_path = directory.join("model.bin");
        let optimizer_path = directory.join("optimizer.bin");
        if sha256_file(&model_path)? != meta.model_sha256
            || sha256_file(&optimizer_path)? != meta.optimizer_sha256
        {
            return Err(anyhow!("checkpoint artifact hash mismatch"));
        }
        let flat = HalfKpFlatModel::decode(&fs::read(model_path)?)?;
        let weights = Weights::from_flat(flat);
        let optimizer = load_optimizer(&optimizer_path, weights.feature.len())?;
        Ok((
            weights,
            optimizer,
            meta.completed_epoch,
            meta.best_epoch,
            meta.best_valid_loss,
        ))
    } else {
        let flat = HalfKpFlatModel::decode(&fs::read(&args.init)?)?;
        if flat.header.target_scale != TARGET_SCALE {
            return Err(anyhow!(
                "scratch trainer requires target scale {TARGET_SCALE}"
            ));
        }
        let weights = Weights::from_flat(flat);
        let optimizer = Optimizer::zeros(weights.feature.len());
        Ok((weights, optimizer, 0, 0, f64::INFINITY))
    }
}

fn training_fingerprint(args: &Args, parent: &ParentManifest) -> Result<String> {
    let value = serde_json::json!({
        "schema": "halfkp_scratch_training_v1",
        "lineage_id": LINEAGE_ID,
        "parent_model_sha256": parent.model.sha256,
        "train": args.train.iter().map(|path| sha256_file(path)).collect::<Result<Vec<_>>>()?,
        "valid": args.valid.iter().map(|path| sha256_file(path)).collect::<Result<Vec<_>>>()?,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "kappa_cp": args.kappa_cp,
        "rank_temperature_cp": args.rank_temperature_cp,
        "rank_weight": args.rank_weight,
        "proof_weight": args.proof_weight,
        "l2": args.l2,
    });
    Ok(sha256_hex(&serde_json::to_vec(&value)?))
}

fn validate_parent(model: &Path, manifest: &Path) -> Result<ParentManifest> {
    let parent: ParentManifest = serde_json::from_slice(&fs::read(manifest)?)?;
    if parent.lineage_id != LINEAGE_ID || parent.model.sha256 != sha256_file(model)? {
        return Err(anyhow!(
            "initial model is not the declared scratch-lineage artifact"
        ));
    }
    if parent.generation == 0 && parent.parent.is_some() {
        return Err(anyhow!("generation zero must not have a parent"));
    }
    Ok(parent)
}

fn validate_datasets(paths: &[PathBuf], allow_teacher: bool) -> Result<()> {
    for path in paths {
        let reader = CurriculumReader::open(path)?;
        if reader.header().lineage_id != LINEAGE_ID {
            return Err(anyhow!("{} is outside scratch lineage", path.display()));
        }
        if !allow_teacher
            && reader
                .header()
                .teacher_lineage_id
                .as_deref()
                .is_some_and(|id| id != LINEAGE_ID)
        {
            return Err(anyhow!("{} has an external teacher", path.display()));
        }
    }
    Ok(())
}

fn train_epoch(
    weights: &mut Weights,
    optimizer: &mut Optimizer,
    args: &Args,
    _epoch: usize,
) -> Result<Metrics> {
    let mut metrics = Metrics::default();
    let mut seen = 0usize;
    let mut batch = Gradient::default();
    let mut batch_records = 0usize;
    for path in &args.train {
        let mut reader = CurriculumReader::open(path)?;
        while let Some(record) = reader.read_record()? {
            accumulate_record(weights, &record, args, &mut batch);
            seen += 1;
            batch_records += 1;
            if batch_records == args.batch_size {
                metrics.loss += batch.loss;
                metrics.objectives += batch.objectives;
                metrics.records += batch_records;
                apply_gradient(weights, optimizer, &mut batch, batch_records, args);
                batch_records = 0;
            }
            if args.max_train_records.is_some_and(|limit| seen >= limit) {
                break;
            }
        }
        if args.max_train_records.is_some_and(|limit| seen >= limit) {
            break;
        }
    }
    if batch_records > 0 {
        metrics.loss += batch.loss;
        metrics.objectives += batch.objectives;
        metrics.records += batch_records;
        apply_gradient(weights, optimizer, &mut batch, batch_records, args);
    }
    Ok(metrics)
}

fn evaluate(
    weights: &Weights,
    paths: &[PathBuf],
    limit: Option<usize>,
    args: &Args,
) -> Result<Metrics> {
    let mut metrics = Metrics::default();
    let mut gradient = Gradient::default();
    for path in paths {
        let mut reader = CurriculumReader::open(path)?;
        while let Some(record) = reader.read_record()? {
            accumulate_record(weights, &record, args, &mut gradient);
            metrics.records += 1;
            if limit.is_some_and(|limit| metrics.records >= limit) {
                break;
            }
        }
        if limit.is_some_and(|limit| metrics.records >= limit) {
            break;
        }
    }
    metrics.loss = gradient.loss;
    metrics.objectives = gradient.objectives;
    Ok(metrics)
}

fn accumulate_record(
    weights: &Weights,
    record: &CurriculumRecord,
    args: &Args,
    gradient: &mut Gradient,
) {
    if let Some(target) = record.result {
        add_value_objective(
            weights,
            &record.root,
            target,
            record.sample_weight,
            args.kappa_cp,
            gradient,
        );
    } else if record.source == LabelSource::ExactMate
        && record
            .proof
            .as_ref()
            .is_some_and(|proof| proof.status == ProofStatus::ProvenMate)
    {
        add_value_objective(
            weights,
            &record.root,
            1.0,
            record.sample_weight * args.proof_weight,
            args.kappa_cp,
            gradient,
        );
    }
    for preference in &record.preferences {
        let good = weights.forward(&preference.good_child);
        let bad = weights.forward(&preference.bad_child);
        if preference.kind == PreferenceKind::SymmetryEqual {
            let difference = good.raw - bad.raw;
            let scale = record.sample_weight * args.rank_weight;
            gradient.loss += 0.5 * (difference * difference * scale) as f64;
            add_position_gradient(
                weights,
                &preference.good_child,
                &good,
                difference * scale,
                gradient,
            );
            add_position_gradient(
                weights,
                &preference.bad_child,
                &bad,
                -difference * scale,
                gradient,
            );
        } else {
            // Child values are from the opponent-to-move perspective.
            let x = (bad.raw - good.raw) * TARGET_SCALE / args.rank_temperature_cp;
            let probability = sigmoid(x);
            let scale = record.sample_weight * args.rank_weight;
            gradient.loss += (softplus(-x) * scale) as f64;
            let common = (probability - 1.0) * TARGET_SCALE / args.rank_temperature_cp * scale;
            add_position_gradient(weights, &preference.good_child, &good, -common, gradient);
            add_position_gradient(weights, &preference.bad_child, &bad, common, gradient);
        }
        gradient.objectives += 1;
    }
}

fn add_value_objective(
    weights: &Weights,
    position: &PackedHalfKpPosition,
    target: f32,
    weight: f32,
    kappa: f32,
    gradient: &mut Gradient,
) {
    if weight <= 0.0 {
        return;
    }
    let forward = weights.forward(position);
    let x = forward.raw * TARGET_SCALE / kappa;
    let probability = sigmoid(x);
    gradient.loss += (-(target * probability.max(1e-7).ln()
        + (1.0 - target) * (1.0 - probability).max(1e-7).ln())
        * weight) as f64;
    let d_raw = (probability - target) * TARGET_SCALE / kappa * weight;
    add_position_gradient(weights, position, &forward, d_raw, gradient);
    gradient.objectives += 1;
}

fn add_position_gradient(
    weights: &Weights,
    position: &PackedHalfKpPosition,
    forward: &Forward,
    d_raw: f32,
    gradient: &mut Gradient,
) {
    let (stm, nstm, stm_features, nstm_features, material) =
        if position.side_to_move == Color::Black {
            (
                &forward.black,
                &forward.white,
                &position.features_black,
                &position.features_white,
                position.material_black,
            )
        } else {
            (
                &forward.white,
                &forward.black,
                &position.features_white,
                &position.features_black,
                position.material_white,
            )
        };
    gradient.out_b += d_raw;
    gradient.out_w[HALFKP_HIDDEN * 2] += d_raw * material / TARGET_SCALE;
    for h in 0..HALFKP_HIDDEN {
        gradient.out_w[h] += d_raw * stm[h].clamp(0.0, 1.0);
        gradient.out_w[HALFKP_HIDDEN + h] += d_raw * nstm[h].clamp(0.0, 1.0);
        let stm_hidden = if (0.0..1.0).contains(&stm[h]) {
            d_raw * weights.out_w[h]
        } else {
            0.0
        };
        let nstm_hidden = if (0.0..1.0).contains(&nstm[h]) {
            d_raw * weights.out_w[HALFKP_HIDDEN + h]
        } else {
            0.0
        };
        gradient.hidden_b[h] += stm_hidden + nstm_hidden;
        for &feature in stm_features {
            gradient
                .feature
                .entry(feature as usize)
                .or_insert([0.0; HALFKP_HIDDEN])[h] += stm_hidden;
        }
        for &feature in nstm_features {
            gradient
                .feature
                .entry(feature as usize)
                .or_insert([0.0; HALFKP_HIDDEN])[h] += nstm_hidden;
        }
    }
}

fn apply_gradient(
    weights: &mut Weights,
    optimizer: &mut Optimizer,
    gradient: &mut Gradient,
    records: usize,
    args: &Args,
) {
    let inv = 1.0 / records as f32;
    for (&feature, row) in &gradient.feature {
        let start = feature * HALFKP_HIDDEN;
        for h in 0..HALFKP_HIDDEN {
            let index = start + h;
            let g = row[h] * inv + args.l2 * weights.feature[index];
            optimizer.feature[index] += g * g;
            weights.feature[index] -=
                args.learning_rate * g / (optimizer.feature[index].sqrt() + EPSILON);
        }
    }
    for h in 0..HALFKP_HIDDEN {
        let g = gradient.hidden_b[h] * inv;
        optimizer.hidden_b[h] += g * g;
        weights.hidden_b[h] -= args.learning_rate * g / (optimizer.hidden_b[h].sqrt() + EPSILON);
    }
    for h in 0..HALFKP_HIDDEN * 2 + 1 {
        let g = gradient.out_w[h] * inv + args.l2 * weights.out_w[h];
        optimizer.out_w[h] += g * g;
        weights.out_w[h] -= args.learning_rate * g / (optimizer.out_w[h].sqrt() + EPSILON);
    }
    let g = gradient.out_b * inv;
    optimizer.out_b += g * g;
    weights.out_b -= args.learning_rate * g / (optimizer.out_b.sqrt() + EPSILON);
    *gradient = Gradient::default();
}

fn save_weights(weights: &Weights, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = PathBuf::from(format!("{}.tmp", path.display()));
    let mut writer = BufWriter::new(File::create(&temporary)?);
    HalfKpFlatModel::write_parts(
        &mut writer,
        HalfKpHeader::current(TARGET_SCALE)?,
        &weights.feature,
        &weights.hidden_b,
        &weights.out_w,
        weights.out_b,
    )?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    if path.exists() {
        fs::remove_file(path)?;
    }
    fs::rename(temporary, path)?;
    Ok(())
}

fn save_checkpoint(
    directory: &Path,
    fingerprint: &str,
    completed_epoch: usize,
    best_epoch: usize,
    best_valid_loss: f64,
    weights: &Weights,
    optimizer: &Optimizer,
) -> Result<()> {
    fs::create_dir_all(directory)?;
    let model_path = directory.join("model.bin");
    let optimizer_path = directory.join("optimizer.bin");
    save_weights(weights, &model_path)?;
    save_optimizer(optimizer, &optimizer_path)?;
    let meta = CheckpointMeta {
        schema: "halfkp_scratch_checkpoint_v1".to_string(),
        fingerprint: fingerprint.to_string(),
        completed_epoch,
        best_epoch,
        best_valid_loss,
        model_sha256: sha256_file(&model_path)?,
        optimizer_sha256: sha256_file(&optimizer_path)?,
    };
    write_json_replace_atomic(&directory.join("state.json"), &meta)
}

fn save_optimizer(optimizer: &Optimizer, path: &Path) -> Result<()> {
    let temporary = PathBuf::from(format!("{}.tmp", path.display()));
    let mut writer = BufWriter::new(File::create(&temporary)?);
    writer.write_all(b"HKOP0001")?;
    writer.write_all(&(optimizer.feature.len() as u64).to_le_bytes())?;
    for value in &optimizer.feature {
        writer.write_all(&value.to_le_bytes())?;
    }
    for value in &optimizer.hidden_b {
        writer.write_all(&value.to_le_bytes())?;
    }
    for value in &optimizer.out_w {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.write_all(&optimizer.out_b.to_le_bytes())?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    replace_file(&temporary, path)
}

fn load_optimizer(path: &Path, expected_features: usize) -> Result<Optimizer> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != b"HKOP0001" {
        return Err(anyhow!("invalid scratch optimizer checkpoint"));
    }
    let mut length = [0u8; 8];
    reader.read_exact(&mut length)?;
    if u64::from_le_bytes(length) as usize != expected_features {
        return Err(anyhow!("optimizer checkpoint shape mismatch"));
    }
    let mut read_f32 = || -> Result<f32> {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)?;
        Ok(f32::from_le_bytes(bytes))
    };
    let mut feature = Vec::with_capacity(expected_features);
    for _ in 0..expected_features {
        feature.push(read_f32()?);
    }
    let mut hidden_b = [0.0; HALFKP_HIDDEN];
    for value in &mut hidden_b {
        *value = read_f32()?;
    }
    let mut out_w = [0.0; HALFKP_HIDDEN * 2 + 1];
    for value in &mut out_w {
        *value = read_f32()?;
    }
    let out_b = read_f32()?;
    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        return Err(anyhow!("trailing bytes in optimizer checkpoint"));
    }
    if feature
        .iter()
        .chain(&hidden_b)
        .chain(&out_w)
        .chain(std::iter::once(&out_b))
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(anyhow!("invalid optimizer accumulator"));
    }
    Ok(Optimizer {
        feature,
        hidden_b,
        out_w,
        out_b,
    })
}

fn replace_file(source: &Path, destination: &Path) -> Result<()> {
    if destination.exists() {
        fs::remove_file(destination)?;
    }
    fs::rename(source, destination)?;
    Ok(())
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = PathBuf::from(format!("{}.tmp", path.display()));
    let mut writer = BufWriter::new(File::create(&temporary)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    fs::rename(temporary, path).with_context(|| format!("publish {}", path.display()))?;
    Ok(())
}

fn write_json_replace_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = PathBuf::from(format!("{}.tmp", path.display()));
    let mut writer = BufWriter::new(File::create(&temporary)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    replace_file(&temporary, path)
}

fn manifest_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.manifest.json", path.display()))
}
fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let e = value.exp();
        e / (1.0 + e)
    }
}
fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}
fn mean_loss(metrics: &Metrics) -> f64 {
    metrics.loss / metrics.objectives.max(1) as f64
}
fn print_metrics(label: &str, epoch: usize, metrics: &Metrics) {
    println!(
        "{label} epoch={epoch} records={} objectives={} loss={:.8}",
        metrics.records,
        metrics.objectives,
        mean_loss(metrics)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_args() -> Args {
        Args {
            train: vec![],
            valid: vec![],
            init: PathBuf::new(),
            init_manifest: PathBuf::new(),
            output: PathBuf::new(),
            epochs: 1,
            batch_size: 1,
            learning_rate: 0.05,
            kappa_cp: 600.0,
            rank_temperature_cp: 300.0,
            rank_weight: 1.0,
            proof_weight: 1.0,
            l2: 0.0,
            max_train_records: None,
            max_valid_records: None,
            checkpoint_dir: None,
            resume: false,
        }
    }

    fn test_weights() -> Weights {
        let mut out_w = [0.0; HALFKP_HIDDEN * 2 + 1];
        out_w[0] = 0.2;
        Weights {
            feature: vec![0.0; 2 * HALFKP_HIDDEN],
            hidden_b: [0.5; HALFKP_HIDDEN],
            out_w,
            out_b: 0.0,
        }
    }

    fn test_position() -> PackedHalfKpPosition {
        PackedHalfKpPosition {
            side_to_move: Color::Black,
            features_black: vec![0],
            features_white: vec![1],
            material_black: 0.0,
            material_white: 0.0,
        }
    }

    fn value_loss(weights: &Weights, target: f32, kappa: f32) -> f32 {
        let raw = weights.forward(&test_position()).raw;
        let probability = sigmoid(raw * TARGET_SCALE / kappa);
        -(target * probability.ln() + (1.0 - target) * (1.0 - probability).ln())
    }

    #[test]
    fn sigmoid_and_softplus_are_finite_at_extremes() {
        for value in [-10_000.0, -100.0, 0.0, 100.0, 10_000.0] {
            assert!(sigmoid(value).is_finite());
            assert!(softplus(value).is_finite());
        }
    }

    #[test]
    fn value_gradient_matches_finite_difference() {
        let mut weights = test_weights();
        let mut gradient = Gradient::default();
        add_value_objective(&weights, &test_position(), 1.0, 1.0, 600.0, &mut gradient);
        let analytic = gradient.feature[&0][0];
        let epsilon = 1e-3;
        weights.feature[0] += epsilon;
        let plus = value_loss(&weights, 1.0, 600.0);
        weights.feature[0] -= 2.0 * epsilon;
        let minus = value_loss(&weights, 1.0, 600.0);
        let numeric = (plus - minus) / (2.0 * epsilon);
        assert!(
            (analytic - numeric).abs() < 2e-4,
            "analytic={analytic} numeric={numeric}"
        );
    }

    #[test]
    fn repeated_updates_overfit_one_value_target() {
        let mut weights = test_weights();
        let mut optimizer = Optimizer::zeros(weights.feature.len());
        let args = test_args();
        let before = value_loss(&weights, 1.0, args.kappa_cp);
        for _ in 0..100 {
            let mut gradient = Gradient::default();
            add_value_objective(
                &weights,
                &test_position(),
                1.0,
                1.0,
                args.kappa_cp,
                &mut gradient,
            );
            apply_gradient(&mut weights, &mut optimizer, &mut gradient, 1, &args);
        }
        let after = value_loss(&weights, 1.0, args.kappa_cp);
        assert!(after < before * 0.25, "before={before} after={after}");
    }

    #[test]
    fn optimizer_checkpoint_round_trips_and_rejects_trailing_bytes() {
        let path = std::env::temp_dir().join(format!(
            "shogi-ai-scratch-optimizer-{}.bin",
            std::process::id()
        ));
        fs::remove_file(&path).ok();
        let mut optimizer = Optimizer::zeros(3);
        optimizer.feature = vec![1.0, 2.0, 3.0];
        optimizer.hidden_b[4] = 4.0;
        optimizer.out_w[5] = 5.0;
        optimizer.out_b = 6.0;
        save_optimizer(&optimizer, &path).expect("save");
        let loaded = load_optimizer(&path, 3).expect("load");
        assert_eq!(loaded.feature, optimizer.feature);
        assert_eq!(loaded.hidden_b, optimizer.hidden_b);
        assert_eq!(loaded.out_w, optimizer.out_w);
        assert_eq!(loaded.out_b, optimizer.out_b);
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("append");
        file.write_all(&[0]).expect("trailing byte");
        drop(file);
        assert!(load_optimizer(&path, 3).is_err());
        fs::remove_file(path).ok();
    }

    #[test]
    fn resumed_updates_match_uninterrupted_updates() {
        let path = std::env::temp_dir().join(format!(
            "shogi-ai-scratch-resume-{}.bin",
            std::process::id()
        ));
        fs::remove_file(&path).ok();
        let args = test_args();
        let mut uninterrupted = test_weights();
        let mut uninterrupted_optimizer = Optimizer::zeros(uninterrupted.feature.len());
        let mut resumed = uninterrupted.clone();
        let mut resumed_optimizer = uninterrupted_optimizer.clone();
        for _ in 0..20 {
            let mut gradient = Gradient::default();
            add_value_objective(
                &uninterrupted,
                &test_position(),
                1.0,
                1.0,
                args.kappa_cp,
                &mut gradient,
            );
            apply_gradient(
                &mut uninterrupted,
                &mut uninterrupted_optimizer,
                &mut gradient,
                1,
                &args,
            );
        }
        for _ in 0..10 {
            let mut gradient = Gradient::default();
            add_value_objective(
                &resumed,
                &test_position(),
                1.0,
                1.0,
                args.kappa_cp,
                &mut gradient,
            );
            apply_gradient(
                &mut resumed,
                &mut resumed_optimizer,
                &mut gradient,
                1,
                &args,
            );
        }
        save_optimizer(&resumed_optimizer, &path).expect("checkpoint optimizer");
        resumed_optimizer = load_optimizer(&path, resumed.feature.len()).expect("resume optimizer");
        for _ in 10..20 {
            let mut gradient = Gradient::default();
            add_value_objective(
                &resumed,
                &test_position(),
                1.0,
                1.0,
                args.kappa_cp,
                &mut gradient,
            );
            apply_gradient(
                &mut resumed,
                &mut resumed_optimizer,
                &mut gradient,
                1,
                &args,
            );
        }
        assert_eq!(uninterrupted.feature, resumed.feature);
        assert_eq!(uninterrupted.hidden_b, resumed.hidden_b);
        assert_eq!(uninterrupted.out_w, resumed.out_w);
        assert_eq!(uninterrupted.out_b, resumed.out_b);
        fs::remove_file(path).ok();
    }
}
