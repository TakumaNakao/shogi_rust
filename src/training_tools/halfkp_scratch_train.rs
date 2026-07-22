use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use shogi_ai::evaluation::{
    HalfKpFlatModel, HalfKpHeader, HALFKP_HIDDEN, HALFKP_INPUTS, HALFKP_PIECE_STATES,
    NUM_BOARD_PIECE_KINDS, NUM_SQUARES,
};
use shogi_ai::halfkp_training::{
    CurriculumReader, CurriculumRecord, LabelSource, PackedHalfKpPosition, PreferenceKind,
};
use shogi_ai::training_data::{
    artifact_metadata, capture_run_environment, sha256_file, sha256_hex,
};
use shogi_ai::training_tools::halfkp_factorization::{factor_indices, HalfKpFactorization};
use shogi_ai::training_tools::scratch_lineage::{
    require_accepted_parent, DEFAULT_PROMOTION_REGISTRY,
};
use shogi_core::Color;
use std::collections::{BTreeMap, HashMap};
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
    parent_gate_report: PathBuf,
    #[arg(long, default_value = DEFAULT_PROMOTION_REGISTRY)]
    promotion_registry: PathBuf,
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
    #[arg(long, default_value_t = 30.0)]
    symmetry_temperature_cp: f32,
    #[arg(long, default_value_t = 1.0)]
    rank_weight: f32,
    /// Additional weight for uncertain recorded-move preferences only.
    #[arg(long, default_value_t = 1.0)]
    expert_preference_weight: f32,
    /// Additional weight for game-result objectives only.
    #[arg(long, default_value_t = 1.0)]
    game_result_weight: f32,
    /// Target effective loss share for exact objectives; all three shares must sum to one.
    #[arg(long, default_value_t = 0.0)]
    source_mix_exact: f32,
    #[arg(long, default_value_t = 0.0)]
    source_mix_health: f32,
    #[arg(long, default_value_t = 0.0)]
    source_mix_game: f32,
    #[arg(long, value_enum)]
    stage: TrainingStage,
    #[arg(long, value_enum, default_value_t = FactorizationMode::Direct)]
    factorization: FactorizationMode,
    #[arg(long, default_value_t = 20260721)]
    seed: u64,
    #[arg(long, default_value_t = 1e-6)]
    l2: f32,
    /// Decoupled weight decay applied independently of AdaGrad state.
    #[arg(long, default_value_t = 0.0)]
    weight_decay: f32,
    /// Keep the Phase 1 output head fixed while learning shared input factors.
    #[arg(long, default_value_t = false)]
    freeze_output_head: bool,
    #[arg(long, default_value_t = 1.0)]
    output_learning_rate_scale: f32,
    /// Zero-centered score RMS anchor applied to symmetry/legal-health roots.
    #[arg(long, default_value_t = 0.0)]
    score_anchor_weight: f32,
    #[arg(long, default_value_t = 600.0)]
    score_anchor_cp: f32,
    /// Keep pre-ClippedReLU activations inside a safe interior range on legal-health roots.
    #[arg(long, default_value_t = 0.0)]
    activation_anchor_weight: f32,
    #[arg(long, default_value_t = 0.02)]
    activation_anchor_margin: f32,
    #[arg(long)]
    max_train_records: Option<usize>,
    #[arg(long)]
    max_valid_records: Option<usize>,
    #[arg(long)]
    checkpoint_dir: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    resume: bool,
    /// Save the final epoch instead of selecting by heterogeneous scalar validation loss.
    #[arg(long, default_value_t = false)]
    save_last_epoch: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum TrainingStage {
    Phase1,
    Phase2,
    Calibrated,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum FactorizationMode {
    Direct,
    PieceRelative,
    FactorOnly,
}

impl TrainingStage {
    fn freezes_scale(self) -> bool {
        matches!(self, Self::Phase1 | Self::Phase2)
    }
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
    parent_gate_report_sha256: String,
    promotion_registry_sha256: String,
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
    factors: Option<HalfKpFactorization>,
    train_direct: bool,
}

#[derive(Clone)]
struct Optimizer {
    feature: Vec<f32>,
    hidden_b: [f32; HALFKP_HIDDEN],
    out_w: [f32; HALFKP_HIDDEN * 2 + 1],
    out_b: f32,
    factor_piece: Vec<f32>,
    factor_relative: Vec<f32>,
}

struct Gradient {
    feature: HashMap<usize, [f32; HALFKP_HIDDEN]>,
    hidden_b: [f32; HALFKP_HIDDEN],
    out_w: [f32; HALFKP_HIDDEN * 2 + 1],
    out_b: f32,
    factor_piece: HashMap<usize, [f32; HALFKP_HIDDEN]>,
    factor_relative: HashMap<usize, [f32; HALFKP_HIDDEN]>,
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
            factor_piece: HashMap::new(),
            factor_relative: HashMap::new(),
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SourceGroup {
    Exact = 0,
    Health = 1,
    Game = 2,
}

struct RecordFamily {
    source: SourceGroup,
    records: Vec<CurriculumRecord>,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let parent = validate_parent(&args)?;
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
        if args.save_last_epoch || loss < best_loss {
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
        schema: "halfkp_scratch_training_v2",
        lineage_id: LINEAGE_ID,
        generation: parent.generation + 1,
        parent: ParentReference {
            model_sha256: parent_model_sha256.clone(),
            manifest_sha256: parent_manifest_sha256.clone(),
        },
        parent_model_sha256,
        parent_manifest_sha256,
        parent_gate_report_sha256: sha256_file(&args.parent_gate_report)?,
        promotion_registry_sha256: sha256_file(&args.promotion_registry)?,
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
            "rank_temperature_cp": args.rank_temperature_cp,
            "symmetry_temperature_cp": args.symmetry_temperature_cp,
            "rank_weight": args.rank_weight,
            "expert_preference_weight": args.expert_preference_weight,
            "game_result_weight": args.game_result_weight,
            "source_mix_exact": args.source_mix_exact,
            "source_mix_health": args.source_mix_health,
            "source_mix_game": args.source_mix_game,
            "stage": format!("{:?}", args.stage).to_ascii_lowercase(),
            "factorization": format!("{:?}", args.factorization).to_ascii_lowercase(),
            "seed": args.seed, "l2": args.l2,
            "weight_decay": args.weight_decay,
            "freeze_output_head": args.freeze_output_head,
            "output_learning_rate_scale": args.output_learning_rate_scale,
            "score_anchor_weight": args.score_anchor_weight,
            "score_anchor_cp": args.score_anchor_cp,
            "activation_anchor_weight": args.activation_anchor_weight,
            "activation_anchor_margin": args.activation_anchor_margin,
            "save_last_epoch": args.save_last_epoch,
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
    fn from_flat(flat: HalfKpFlatModel, factorization: FactorizationMode) -> Self {
        Self {
            feature: flat.feature_emb,
            hidden_b: flat.hidden_b,
            out_w: flat.out_w,
            out_b: flat.out_b,
            factors: (factorization != FactorizationMode::Direct).then(HalfKpFactorization::zeros),
            train_direct: factorization != FactorizationMode::FactorOnly,
        }
    }

    fn forward(&self, position: &PackedHalfKpPosition) -> Forward {
        let mut black = self.hidden_b;
        let mut white = self.hidden_b;
        self.accumulate_features(&position.features_black, &mut black);
        self.accumulate_features(&position.features_white, &mut white);
        let (stm, nstm, material) = if position.side_to_move == Color::Black {
            (&black, &white, position.material_black)
        } else {
            (&white, &black, position.material_white)
        };
        let mut raw = self.out_b + self.out_w[HALFKP_HIDDEN * 2] * material / TARGET_SCALE;
        for h in 0..HALFKP_HIDDEN {
            raw += self.out_w[h] * stm[h].clamp(0.0, 1.0);
            raw += self.out_w[HALFKP_HIDDEN + h] * nstm[h].clamp(0.0, 1.0);
        }
        Forward { black, white, raw }
    }

    fn accumulate_features(&self, features: &[u32], hidden: &mut [f32; HALFKP_HIDDEN]) {
        for &feature in features {
            let feature = feature as usize;
            let start = feature * HALFKP_HIDDEN;
            for h in 0..HALFKP_HIDDEN {
                hidden[h] += self.feature[start + h];
            }
            if let Some(factors) = &self.factors {
                factors
                    .add_feature(feature, hidden)
                    .expect("validated feature");
            }
        }
    }

    fn project_lr_symmetry(&mut self) {
        project_direct_lr_symmetry(&mut self.feature);
        if let Some(factors) = &mut self.factors {
            factors.project_lr_symmetry();
        }
    }
}

impl Optimizer {
    fn zeros(weights: &Weights) -> Self {
        Self {
            feature: vec![0.0; weights.feature.len()],
            hidden_b: [0.0; HALFKP_HIDDEN],
            out_w: [0.0; HALFKP_HIDDEN * 2 + 1],
            out_b: 0.0,
            factor_piece: weights
                .factors
                .as_ref()
                .map_or_else(Vec::new, |factors| vec![0.0; factors.piece.len()]),
            factor_relative: weights
                .factors
                .as_ref()
                .map_or_else(Vec::new, |factors| vec![0.0; factors.relative.len()]),
        }
    }
}

fn validate_args(args: &Args) -> Result<()> {
    if args.epochs == 0
        || args.batch_size == 0
        || args.learning_rate <= 0.0
        || args.kappa_cp <= 0.0
        || args.rank_temperature_cp <= 0.0
        || args.symmetry_temperature_cp <= 0.0
        || args.rank_weight < 0.0
        || args.expert_preference_weight < 0.0
        || args.game_result_weight < 0.0
        || !valid_source_mix(args)
        || args.l2 < 0.0
        || args.weight_decay < 0.0
        || args.score_anchor_weight < 0.0
        || args.score_anchor_cp <= 0.0
        || args.activation_anchor_weight < 0.0
        || !(0.0..0.5).contains(&args.activation_anchor_margin)
        || args.output_learning_rate_scale < 0.0
        || (!args.resume && args.output.exists())
        || manifest_path(&args.output).exists()
        || (args.resume && args.checkpoint_dir.is_none())
        || (args.factorization != FactorizationMode::Direct
            && (args.resume || args.checkpoint_dir.is_some()))
    {
        return Err(anyhow!("invalid arguments or output already exists"));
    }
    Ok(())
}

fn valid_source_mix(args: &Args) -> bool {
    let values = [
        args.source_mix_exact,
        args.source_mix_health,
        args.source_mix_game,
    ];
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return false;
    }
    let sum = values.iter().sum::<f32>();
    sum == 0.0 || ((sum - 1.0).abs() <= 1e-6 && values.iter().all(|value| *value > 0.0))
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
        let mut weights = Weights::from_flat(flat, args.factorization);
        weights.project_lr_symmetry();
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
        let mut weights = Weights::from_flat(flat, args.factorization);
        weights.project_lr_symmetry();
        let optimizer = Optimizer::zeros(&weights);
        Ok((weights, optimizer, 0, 0, f64::INFINITY))
    }
}

fn training_fingerprint(args: &Args, parent: &ParentManifest) -> Result<String> {
    let value = serde_json::json!({
        "schema": "halfkp_scratch_training_v2",
        "lineage_id": LINEAGE_ID,
        "parent_model_sha256": parent.model.sha256,
        "parent_gate_report_sha256": sha256_file(&args.parent_gate_report)?,
        "promotion_registry_sha256": sha256_file(&args.promotion_registry)?,
        "train": args.train.iter().map(|path| sha256_file(path)).collect::<Result<Vec<_>>>()?,
        "valid": args.valid.iter().map(|path| sha256_file(path)).collect::<Result<Vec<_>>>()?,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "kappa_cp": args.kappa_cp,
        "rank_temperature_cp": args.rank_temperature_cp,
        "symmetry_temperature_cp": args.symmetry_temperature_cp,
        "rank_weight": args.rank_weight,
        "expert_preference_weight": args.expert_preference_weight,
        "game_result_weight": args.game_result_weight,
        "source_mix_exact": args.source_mix_exact,
        "source_mix_health": args.source_mix_health,
        "source_mix_game": args.source_mix_game,
        "stage": format!("{:?}", args.stage).to_ascii_lowercase(),
        "factorization": format!("{:?}", args.factorization).to_ascii_lowercase(),
        "seed": args.seed,
        "l2": args.l2,
        "weight_decay": args.weight_decay,
        "freeze_output_head": args.freeze_output_head,
        "output_learning_rate_scale": args.output_learning_rate_scale,
        "score_anchor_weight": args.score_anchor_weight,
        "score_anchor_cp": args.score_anchor_cp,
        "activation_anchor_weight": args.activation_anchor_weight,
        "activation_anchor_margin": args.activation_anchor_margin,
        "save_last_epoch": args.save_last_epoch,
    });
    Ok(sha256_hex(&serde_json::to_vec(&value)?))
}

fn validate_parent(args: &Args) -> Result<ParentManifest> {
    let parent: ParentManifest = serde_json::from_slice(&fs::read(&args.init_manifest)?)?;
    if parent.lineage_id != LINEAGE_ID || parent.model.sha256 != sha256_file(&args.init)? {
        return Err(anyhow!(
            "initial model is not the declared scratch-lineage artifact"
        ));
    }
    if parent.generation == 0 && parent.parent.is_some() {
        return Err(anyhow!("generation zero must not have a parent"));
    }
    require_accepted_parent(
        &args.promotion_registry,
        &args.init,
        &args.parent_gate_report,
        parent.generation,
    )?;
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
    epoch: usize,
) -> Result<Metrics> {
    let mut metrics = Metrics::default();
    let mut families = BTreeMap::<[u8; 32], Vec<CurriculumRecord>>::new();
    let mut seen = 0usize;
    for path in &args.train {
        let mut reader = CurriculumReader::open(path)?;
        while let Some(record) = reader.read_record()? {
            families.entry(record.family_hash).or_default().push(record);
            seen += 1;
            if args.max_train_records.is_some_and(|limit| seen >= limit) {
                break;
            }
        }
        if args.max_train_records.is_some_and(|limit| seen >= limit) {
            break;
        }
    }
    let mut families = families
        .into_values()
        .map(|records| {
            let source = source_group(records[0].source);
            if records
                .iter()
                .any(|record| source_group(record.source) != source)
            {
                return Err(anyhow!("one family contains multiple source groups"));
            }
            Ok(RecordFamily { source, records })
        })
        .collect::<Result<Vec<_>>>()?;
    let source_multipliers = source_multipliers(&families, args)?;
    let mut rng = ChaCha20Rng::seed_from_u64(args.seed ^ epoch as u64);
    families.shuffle(&mut rng);
    let mut batch = Gradient::default();
    let mut batch_records = 0usize;
    let mut batch_families = 0usize;
    for family in families {
        let objectives = family
            .records
            .iter()
            .map(|record| objective_count(record, args))
            .sum::<usize>();
        if objectives == 0 {
            metrics.records += family.records.len();
            continue;
        }
        let family_scale = source_multipliers[family.source as usize] / objectives as f32;
        for record in &family.records {
            accumulate_record(weights, record, args, family_scale, &mut batch);
            batch_records += 1;
        }
        batch_families += 1;
        if batch_families == args.batch_size {
            metrics.loss += batch.loss;
            metrics.objectives += batch.objectives;
            metrics.records += batch_records;
            apply_gradient(weights, optimizer, &mut batch, batch_families, args);
            batch_records = 0;
            batch_families = 0;
        }
    }
    if batch_families > 0 {
        metrics.loss += batch.loss;
        metrics.objectives += batch.objectives;
        metrics.records += batch_records;
        apply_gradient(weights, optimizer, &mut batch, batch_families, args);
    }
    Ok(metrics)
}

fn source_group(source: LabelSource) -> SourceGroup {
    match source {
        LabelSource::ExactMate | LabelSource::ExactTactic => SourceGroup::Exact,
        LabelSource::SymmetryAnchor => SourceGroup::Health,
        LabelSource::ExpertMove
        | LabelSource::GameResult
        | LabelSource::SelfSearch
        | LabelSource::SelfPlay => SourceGroup::Game,
    }
}

fn family_nominal_weight(family: &RecordFamily, args: &Args) -> f64 {
    let objectives = family
        .records
        .iter()
        .map(|record| objective_count(record, args))
        .sum::<usize>();
    if objectives == 0 {
        return 0.0;
    }
    let weight = family
        .records
        .iter()
        .map(|record| {
            let result = record
                .result
                .map_or(0.0, |_| result_objective_weight(record, args));
            let preferences =
                record.preferences.len() as f32 * preference_objective_weight(record, args);
            let score_anchor = if record.source == LabelSource::SymmetryAnchor {
                record.sample_weight * args.score_anchor_weight
            } else {
                0.0
            };
            result + preferences + score_anchor
        })
        .sum::<f32>();
    (weight / objectives as f32) as f64
}

fn source_multipliers(families: &[RecordFamily], args: &Args) -> Result<[f32; 3]> {
    if args.source_mix_exact == 0.0 {
        return Ok([1.0; 3]);
    }
    let mut nominal = [0.0f64; 3];
    let mut counts = [0usize; 3];
    for family in families {
        nominal[family.source as usize] += family_nominal_weight(family, args);
        counts[family.source as usize] += 1;
    }
    if nominal.iter().any(|value| *value <= 0.0) {
        return Err(anyhow!(
            "source mix requested but one source has no enabled objective"
        ));
    }
    let targets = [
        args.source_mix_exact as f64,
        args.source_mix_health as f64,
        args.source_mix_game as f64,
    ];
    let total = nominal.iter().sum::<f64>();
    let multipliers = std::array::from_fn(|index| (total * targets[index] / nominal[index]) as f32);
    println!(
        "source_mix families={counts:?} nominal={nominal:?} target={targets:?} multipliers={multipliers:?}"
    );
    Ok(multipliers)
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
            accumulate_record(weights, &record, args, 1.0, &mut gradient);
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
    family_scale: f32,
    gradient: &mut Gradient,
) {
    if record.source == LabelSource::SymmetryAnchor && args.score_anchor_weight > 0.0 {
        let forward = weights.forward(&record.root);
        let normalized = forward.raw * TARGET_SCALE / args.score_anchor_cp;
        let scale = record.sample_weight * args.score_anchor_weight * family_scale;
        gradient.loss += 0.5 * (normalized * normalized * scale) as f64;
        let d_raw = normalized * TARGET_SCALE / args.score_anchor_cp * scale;
        add_position_gradient(weights, &record.root, &forward, d_raw, gradient);
        gradient.objectives += 1;
    }
    if record.source == LabelSource::SymmetryAnchor && args.activation_anchor_weight > 0.0 {
        let forward = weights.forward(&record.root);
        let scale = record.sample_weight * args.activation_anchor_weight * family_scale;
        add_activation_anchor(
            weights,
            &record.root,
            &forward,
            args.activation_anchor_margin,
            scale,
            gradient,
        );
    }
    if let Some(target) = record.result {
        add_value_objective(
            weights,
            &record.root,
            target,
            result_objective_weight(record, args) * family_scale,
            args.kappa_cp,
            gradient,
        );
    }
    for preference in &record.preferences {
        let preference_weight = preference_objective_weight(record, args);
        if preference_weight == 0.0 {
            continue;
        }
        let good = weights.forward(&preference.good_child);
        let bad = weights.forward(&preference.bad_child);
        if preference.kind == PreferenceKind::SymmetryEqual {
            let difference = good.raw - bad.raw;
            let normalized = difference * TARGET_SCALE / args.symmetry_temperature_cp;
            let scale = preference_weight * family_scale;
            gradient.loss += 0.5 * (normalized * normalized * scale) as f64;
            let d_raw = normalized * TARGET_SCALE / args.symmetry_temperature_cp * scale;
            add_position_gradient(weights, &preference.good_child, &good, d_raw, gradient);
            add_position_gradient(weights, &preference.bad_child, &bad, -d_raw, gradient);
        } else {
            // Child values are from the opponent-to-move perspective.
            let margin_cp = margin_cp(record.source, preference.kind, preference.margin_class);
            let x = ((bad.raw - good.raw) * TARGET_SCALE - margin_cp) / args.rank_temperature_cp;
            let probability = sigmoid(x);
            let scale = preference_weight * family_scale;
            gradient.loss += (softplus(-x) * scale) as f64;
            let common = (probability - 1.0) * TARGET_SCALE / args.rank_temperature_cp * scale;
            add_position_gradient(weights, &preference.good_child, &good, -common, gradient);
            add_position_gradient(weights, &preference.bad_child, &bad, common, gradient);
        }
        gradient.objectives += 1;
    }
}

fn result_objective_weight(record: &CurriculumRecord, args: &Args) -> f32 {
    let source_weight = if matches!(
        record.source,
        LabelSource::ExpertMove | LabelSource::GameResult
    ) {
        args.game_result_weight
    } else {
        1.0
    };
    record.sample_weight * source_weight
}

fn preference_objective_weight(record: &CurriculumRecord, args: &Args) -> f32 {
    let source_weight = if record.source == LabelSource::ExpertMove {
        args.expert_preference_weight
    } else {
        1.0
    };
    record.sample_weight * record.confidence * args.rank_weight * source_weight
}

fn objective_count(record: &CurriculumRecord, args: &Args) -> usize {
    let result_enabled = !matches!(
        record.source,
        LabelSource::ExpertMove | LabelSource::GameResult
    ) || args.game_result_weight > 0.0;
    let preference_enabled = args.rank_weight > 0.0
        && (record.source != LabelSource::ExpertMove
            || (args.expert_preference_weight > 0.0 && record.confidence > 0.0));
    usize::from(record.result.is_some() && result_enabled)
        + usize::from(preference_enabled) * record.preferences.len()
        + usize::from(
            record.source == LabelSource::SymmetryAnchor && args.score_anchor_weight > 0.0,
        )
}

fn add_activation_anchor(
    weights: &Weights,
    position: &PackedHalfKpPosition,
    forward: &Forward,
    margin: f32,
    scale: f32,
    gradient: &mut Gradient,
) {
    let normalization = scale / (2 * HALFKP_HIDDEN) as f32;
    for (hidden, features) in [
        (&forward.black, &position.features_black),
        (&forward.white, &position.features_white),
    ] {
        for h in 0..HALFKP_HIDDEN {
            let residual = if hidden[h] < margin {
                hidden[h] - margin
            } else if hidden[h] > 1.0 - margin {
                hidden[h] - (1.0 - margin)
            } else {
                0.0
            };
            if residual == 0.0 {
                continue;
            }
            gradient.loss += (0.5 * residual * residual * normalization) as f64;
            let value = residual * normalization;
            gradient.hidden_b[h] += value;
            for &feature in features {
                add_feature_gradient(
                    gradient,
                    feature as usize,
                    h,
                    value,
                    weights.factors.is_some(),
                    weights.train_direct,
                );
            }
        }
    }
}

fn margin_cp(source: LabelSource, kind: PreferenceKind, _class: u8) -> f32 {
    match (source, kind) {
        (_, PreferenceKind::SymmetryEqual) => 0.0,
        (LabelSource::ExactMate | LabelSource::ExactTactic, PreferenceKind::ProvenOutcome) => 0.0,
        (_, PreferenceKind::ExpertMove) => 50.0,
        (_, PreferenceKind::Search) => 75.0,
        _ => 100.0,
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
            add_feature_gradient(
                gradient,
                feature as usize,
                h,
                stm_hidden,
                weights.factors.is_some(),
                weights.train_direct,
            );
        }
        for &feature in nstm_features {
            add_feature_gradient(
                gradient,
                feature as usize,
                h,
                nstm_hidden,
                weights.factors.is_some(),
                weights.train_direct,
            );
        }
    }
}

fn add_feature_gradient(
    gradient: &mut Gradient,
    feature: usize,
    hidden: usize,
    value: f32,
    factorized: bool,
    train_direct: bool,
) {
    if train_direct {
        gradient
            .feature
            .entry(feature)
            .or_insert([0.0; HALFKP_HIDDEN])[hidden] += value;
    }
    if factorized {
        let indices = factor_indices(feature).expect("validated feature");
        gradient
            .factor_piece
            .entry(indices.piece)
            .or_insert([0.0; HALFKP_HIDDEN])[hidden] += value;
        if let Some(relative) = indices.relative {
            gradient
                .factor_relative
                .entry(relative)
                .or_insert([0.0; HALFKP_HIDDEN])[hidden] += value;
        }
    }
}

fn apply_gradient(
    weights: &mut Weights,
    optimizer: &mut Optimizer,
    gradient: &mut Gradient,
    families: usize,
    args: &Args,
) {
    let inv = 1.0 / families.max(1) as f32;
    let decay = (1.0 - args.learning_rate * args.weight_decay).max(0.0);
    if args.weight_decay > 0.0 {
        if weights.train_direct {
            weights.feature.iter_mut().for_each(|value| *value *= decay);
        }
        if let Some(factors) = &mut weights.factors {
            factors.piece.iter_mut().for_each(|value| *value *= decay);
            factors
                .relative
                .iter_mut()
                .for_each(|value| *value *= decay);
        }
    }
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
    if let Some(factors) = &mut weights.factors {
        apply_sparse_rows(
            &mut factors.piece,
            &mut optimizer.factor_piece,
            &gradient.factor_piece,
            inv,
            args.learning_rate,
            args.l2,
        );
        apply_sparse_rows(
            &mut factors.relative,
            &mut optimizer.factor_relative,
            &gradient.factor_relative,
            inv,
            args.learning_rate,
            args.l2,
        );
    }
    for h in 0..HALFKP_HIDDEN {
        let g = gradient.hidden_b[h] * inv;
        optimizer.hidden_b[h] += g * g;
        weights.hidden_b[h] -= args.learning_rate * g / (optimizer.hidden_b[h].sqrt() + EPSILON);
    }
    let output_end = if args.freeze_output_head {
        0
    } else if args.stage.freezes_scale() {
        HALFKP_HIDDEN * 2
    } else {
        HALFKP_HIDDEN * 2 + 1
    };
    for h in 0..output_end {
        let output_decay = (1.0
            - args.learning_rate * args.output_learning_rate_scale * args.weight_decay)
            .max(0.0);
        if args.weight_decay > 0.0 {
            weights.out_w[h] *= output_decay;
        }
        let g = gradient.out_w[h] * inv + args.l2 * weights.out_w[h];
        optimizer.out_w[h] += g * g;
        weights.out_w[h] -= args.learning_rate * args.output_learning_rate_scale * g
            / (optimizer.out_w[h].sqrt() + EPSILON);
    }
    if !args.stage.freezes_scale() {
        let g = gradient.out_b * inv;
        optimizer.out_b += g * g;
        weights.out_b -= args.learning_rate * g / (optimizer.out_b.sqrt() + EPSILON);
    }
    weights.project_lr_symmetry();
    *gradient = Gradient::default();
}

fn project_direct_lr_symmetry(feature: &mut [f32]) {
    if feature.len() != HALFKP_INPUTS * HALFKP_HIDDEN {
        return;
    }
    const BOARD_STATES: usize = 2 * NUM_BOARD_PIECE_KINDS * NUM_SQUARES;
    for rank in 0..9 {
        let king_bucket = 4 * 9 + rank;
        for color_kind in 0..2 * NUM_BOARD_PIECE_KINDS {
            for file in 0..4 {
                for piece_rank in 0..9 {
                    let a_state = color_kind * NUM_SQUARES + file * 9 + piece_rank;
                    let b_state = color_kind * NUM_SQUARES + (8 - file) * 9 + piece_rank;
                    debug_assert!(a_state < BOARD_STATES && b_state < BOARD_STATES);
                    let a = (king_bucket * HALFKP_PIECE_STATES + a_state) * HALFKP_HIDDEN;
                    let b = (king_bucket * HALFKP_PIECE_STATES + b_state) * HALFKP_HIDDEN;
                    for h in 0..HALFKP_HIDDEN {
                        let mean = 0.5 * (feature[a + h] + feature[b + h]);
                        feature[a + h] = mean;
                        feature[b + h] = mean;
                    }
                }
            }
        }
    }
}

fn apply_sparse_rows(
    weights: &mut [f32],
    optimizer: &mut [f32],
    gradient: &HashMap<usize, [f32; HALFKP_HIDDEN]>,
    inv: f32,
    learning_rate: f32,
    l2: f32,
) {
    for (&row_index, row) in gradient {
        let start = row_index * HALFKP_HIDDEN;
        for h in 0..HALFKP_HIDDEN {
            let index = start + h;
            let g = row[h] * inv + l2 * weights[index];
            optimizer[index] += g * g;
            weights[index] -= learning_rate * g / (optimizer[index].sqrt() + EPSILON);
        }
    }
}

fn save_weights(weights: &Weights, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = PathBuf::from(format!("{}.tmp", path.display()));
    let mut writer = BufWriter::new(File::create(&temporary)?);
    let mut feature = weights.feature.clone();
    if let Some(factors) = &weights.factors {
        factors.fold_into(&mut feature)?;
    }
    HalfKpFlatModel::write_parts(
        &mut writer,
        HalfKpHeader::current(TARGET_SCALE)?,
        &feature,
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
        factor_piece: vec![],
        factor_relative: vec![],
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
    use shogi_ai::halfkp_training::{CurriculumPreference, OpeningStratum};
    use shogi_ai::training_data::DatasetSplit;
    use shogi_core::PartialPosition;
    use shogi_lib::Position;
    use shogi_usi_parser::FromUsi;

    fn test_args() -> Args {
        Args {
            train: vec![],
            valid: vec![],
            init: PathBuf::new(),
            init_manifest: PathBuf::new(),
            parent_gate_report: PathBuf::new(),
            promotion_registry: PathBuf::new(),
            output: PathBuf::new(),
            epochs: 1,
            batch_size: 1,
            learning_rate: 0.05,
            kappa_cp: 600.0,
            rank_temperature_cp: 300.0,
            symmetry_temperature_cp: 30.0,
            rank_weight: 1.0,
            expert_preference_weight: 1.0,
            game_result_weight: 1.0,
            source_mix_exact: 0.0,
            source_mix_health: 0.0,
            source_mix_game: 0.0,
            stage: TrainingStage::Calibrated,
            factorization: FactorizationMode::Direct,
            seed: 7,
            l2: 0.0,
            weight_decay: 0.0,
            freeze_output_head: false,
            output_learning_rate_scale: 1.0,
            score_anchor_weight: 0.0,
            score_anchor_cp: 600.0,
            activation_anchor_weight: 0.0,
            activation_anchor_margin: 0.02,
            max_train_records: None,
            max_valid_records: None,
            checkpoint_dir: None,
            resume: false,
            save_last_epoch: false,
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
            factors: None,
            train_direct: true,
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

    fn expert_record() -> CurriculumRecord {
        let root = test_position();
        CurriculumRecord {
            source: LabelSource::ExpertMove,
            group_id: 1,
            position_hash: 1,
            family_hash: [1; 32],
            network_input_hash: root.network_input_hash(),
            root_sfen: "test-only".to_string(),
            split: DatasetSplit::Train,
            ply: 1,
            phase: 0,
            opening_stratum: OpeningStratum::UNKNOWN,
            confidence: 0.6,
            sample_weight: 0.1,
            root: root.clone(),
            result: Some(1.0),
            proof: None,
            preferences: vec![CurriculumPreference {
                good_child: root.clone(),
                bad_child: root,
                kind: PreferenceKind::ExpertMove,
                margin_class: 0,
                good_move_usi: None,
                bad_move_usi: None,
            }],
        }
    }

    #[test]
    fn expert_confidence_only_scales_preference_objective() {
        let record = expert_record();
        let mut args = test_args();
        args.expert_preference_weight = 2.0;
        args.game_result_weight = 3.0;
        assert!((preference_objective_weight(&record, &args) - 0.12).abs() < 1e-7);
        assert!((result_objective_weight(&record, &args) - 0.3).abs() < 1e-7);
        args.expert_preference_weight = 0.0;
        args.game_result_weight = 0.0;
        assert_eq!(preference_objective_weight(&record, &args), 0.0);
        assert_eq!(result_objective_weight(&record, &args), 0.0);
        assert_eq!(objective_count(&record, &args), 0);
    }

    #[test]
    fn source_mix_scales_effective_nominal_loss() {
        let mut args = test_args();
        args.source_mix_exact = 0.2;
        args.source_mix_health = 0.4;
        args.source_mix_game = 0.4;
        args.score_anchor_weight = 2.0;
        let game = expert_record();
        let mut exact = game.clone();
        exact.source = LabelSource::ExactTactic;
        exact.confidence = 1.0;
        exact.sample_weight = 1.0;
        exact.result = None;
        let mut health = exact.clone();
        health.source = LabelSource::SymmetryAnchor;
        health.sample_weight = 0.05;
        health.preferences[0].kind = PreferenceKind::SymmetryEqual;
        let families = vec![
            RecordFamily {
                source: SourceGroup::Exact,
                records: vec![exact],
            },
            RecordFamily {
                source: SourceGroup::Health,
                records: vec![health],
            },
            RecordFamily {
                source: SourceGroup::Game,
                records: vec![game],
            },
        ];
        let multipliers = source_multipliers(&families, &args).expect("mix");
        let effective: [f64; 3] = std::array::from_fn(|index| {
            family_nominal_weight(&families[index], &args) * multipliers[index] as f64
        });
        let total = effective.iter().sum::<f64>();
        for (actual, expected) in effective
            .iter()
            .map(|value| value / total)
            .zip([0.2, 0.4, 0.4])
        {
            assert!((actual - expected).abs() < 1e-6, "actual={actual}");
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
    fn factor_only_gradient_does_not_update_direct_rows() {
        let mut gradient = Gradient::default();
        add_feature_gradient(&mut gradient, 0, 0, 1.0, true, false);
        assert!(gradient.feature.is_empty());
        assert!(!gradient.factor_piece.is_empty());
    }

    #[test]
    fn projected_center_king_rows_make_mirror_scores_equal() {
        let original = Position::default();
        let mirrored = Position::new(
            PartialPosition::from_usi(
                "sfen lnsgkgsnl/1b5r1/ppppppppp/9/9/9/PPPPPPPPP/1R5B1/LNSGKGSNL b - 1",
            )
            .expect("mirrored start position"),
        );
        let mut weights = Weights {
            feature: (0..HALFKP_INPUTS * HALFKP_HIDDEN)
                .map(|index| (index % 997) as f32 * 1e-7)
                .collect(),
            hidden_b: [0.25; HALFKP_HIDDEN],
            out_w: [0.01; HALFKP_HIDDEN * 2 + 1],
            out_b: 0.0,
            factors: Some({
                let mut factors = HalfKpFactorization::zeros();
                for (index, value) in factors.relative.iter_mut().enumerate() {
                    *value = (index % 991) as f32 * 1e-7;
                }
                factors
            }),
            train_direct: true,
        };
        weights.project_lr_symmetry();
        let a = PackedHalfKpPosition::from_position(&original).expect("original");
        let b = PackedHalfKpPosition::from_position(&mirrored).expect("mirror");
        assert_eq!(weights.forward(&a).raw, weights.forward(&b).raw);
        let mut folded = weights.clone();
        let factors = folded.factors.take().expect("factors");
        factors.fold_into(&mut folded.feature).expect("fold");
        assert_eq!(folded.forward(&a).raw, folded.forward(&b).raw);
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
        let mut optimizer = Optimizer::zeros(&weights);
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
        let mut optimizer = Optimizer::zeros(&test_weights());
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
        let mut uninterrupted_optimizer = Optimizer::zeros(&uninterrupted);
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
