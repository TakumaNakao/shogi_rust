use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use shogi_ai::evaluation::{HalfKpFlatModel, HalfKpHeader, HALFKP_HIDDEN, HALFKP_INPUTS};
use shogi_ai::halfkp_training::{
    read_search_teacher_manifest, PackedHalfKpPosition, SearchTeacherReader, SearchTeacherRecord,
    CANDIDATE_GAME_MOVE, SEARCH_TEACHER_SEMANTICS_ID, SEARCH_TEACHER_SEMANTICS_VERSION,
};
use shogi_core::Color;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const TARGET_SCALE: f32 = 1000.0;

#[derive(Parser, Debug)]
#[command(about = "Train HalfKP from own-search teachers, game results, and candidate rankings")]
struct Args {
    #[arg(long, required = true)]
    train: Vec<PathBuf>,
    #[arg(long, required = true)]
    valid: Vec<PathBuf>,
    #[arg(long)]
    test: Vec<PathBuf>,
    #[arg(long)]
    init: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 5)]
    epochs: usize,
    #[arg(long, default_value_t = 128)]
    batch_size: usize,
    #[arg(long, default_value_t = 0.0003)]
    learning_rate: f32,
    #[arg(long, default_value_t = 0.00003)]
    output_learning_rate: f32,
    #[arg(long, default_value_t = 600.0)]
    kappa_cp: f32,
    #[arg(long, default_value_t = 0.5)]
    search_mix: f32,
    #[arg(long)]
    search_mix_end: Option<f32>,
    #[arg(long, default_value_t = 2.5)]
    loss_power: f32,
    #[arg(long, default_value_t = 0.10)]
    rank_weight: f32,
    #[arg(long, default_value_t = 0.025)]
    game_rank_weight: f32,
    #[arg(long, default_value_t = 20.0)]
    rank_margin_cp: f32,
    #[arg(long, default_value_t = 100.0)]
    rank_temperature_cp: f32,
    #[arg(long, default_value_t = 15.0)]
    min_rank_gap_cp: f32,
    #[arg(long, default_value_t = 150.0)]
    game_regret_cap_cp: f32,
    #[arg(long, default_value_t = 4)]
    max_pairs_per_record: usize,
    #[arg(long, default_value_t = 1.0)]
    gradient_clip_norm: f32,
    #[arg(long, default_value_t = 0.06)]
    output_limit: f32,
    #[arg(long, default_value_t = 2)]
    early_stop_patience: usize,
    #[arg(long, default_value_t = 0.000001)]
    min_valid_improvement: f64,
    #[arg(long, default_value_t = 20260718)]
    seed: u64,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, value_enum, default_value_t = OptimizerKind::Adagrad)]
    optimizer: OptimizerKind,
    #[arg(long, default_value_t = 0.9)]
    schedule_free_beta1: f32,
    #[arg(long, default_value_t = 0.999)]
    schedule_free_beta2: f32,
    #[arg(long, default_value_t = 10_000)]
    schedule_free_warmup_steps: u64,
    #[arg(long)]
    checkpoint_dir: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    resume: bool,
    #[arg(long, default_value_t = 4)]
    swa_start_epoch: usize,
    #[arg(long, default_value_t = false)]
    fit_kappa: bool,
    #[arg(long)]
    max_train_records: Option<usize>,
    #[arg(long)]
    max_valid_records: Option<usize>,
    /// Permit HKST0002 inputs created before teacher-semantics manifests.
    #[arg(long, default_value_t = false)]
    allow_legacy_teacher_semantics: bool,
    #[arg(long)]
    log: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
enum OptimizerKind {
    Adagrad,
    ScheduleFree,
}

#[derive(Clone)]
struct Weights {
    feature_emb: Vec<f32>,
    hidden_b: [f32; HALFKP_HIDDEN],
    out_w: [f32; HALFKP_HIDDEN * 2 + 1],
    out_b: f32,
}

struct ScheduleFreeState {
    z: Weights,
    variance: Weights,
    last_weight_sum: Vec<f64>,
    step: u64,
    weight_sum: f64,
    lr_max: f32,
}

enum OptimizerState {
    Adagrad(Box<Weights>),
    ScheduleFree(Box<ScheduleFreeState>),
}

struct SwaState {
    average: Weights,
    count: u64,
}

#[derive(Serialize, Deserialize)]
struct CheckpointMeta {
    completed_epoch: usize,
    best_score: f64,
    stale_epochs: usize,
    optimizer: OptimizerKind,
    schedule_step: u64,
    schedule_weight_sum: f64,
    schedule_lr_max: f32,
    swa_count: u64,
}

struct DenseGradient {
    hidden_b: [f32; HALFKP_HIDDEN],
    out_w: [f32; HALFKP_HIDDEN * 2 + 1],
    out_b: f32,
}

impl Default for DenseGradient {
    fn default() -> Self {
        Self {
            hidden_b: [0.0; HALFKP_HIDDEN],
            out_w: [0.0; HALFKP_HIDDEN * 2 + 1],
            out_b: 0.0,
        }
    }
}

struct Forward {
    black: [f32; HALFKP_HIDDEN],
    white: [f32; HALFKP_HIDDEN],
    raw: f32,
}

struct RecordGradient {
    dense: DenseGradient,
    features: Vec<(usize, [f32; HALFKP_HIDDEN])>,
    value_loss: f64,
    rank_loss: f64,
    pairs: usize,
}

#[derive(Clone)]
struct Metrics {
    records: usize,
    known_results: usize,
    value_loss: f64,
    search_abs_error: f64,
    pairs: usize,
    pair_correct: usize,
    top1_matches: usize,
    game_moves: usize,
    game_move_top1: usize,
    regret_sum: f64,
    regret_histogram: [u64; 121],
    log_loss: f64,
    phase_records: [usize; 3],
    phase_brier: [f64; 3],
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            records: 0,
            known_results: 0,
            value_loss: 0.0,
            search_abs_error: 0.0,
            pairs: 0,
            pair_correct: 0,
            top1_matches: 0,
            game_moves: 0,
            game_move_top1: 0,
            regret_sum: 0.0,
            regret_histogram: [0; 121],
            log_loss: 0.0,
            phase_records: [0; 3],
            phase_brier: [0.0; 3],
        }
    }
}

struct TrainOptions {
    kappa_cp: f32,
    search_mix: f32,
    loss_power: f32,
    rank_weight: f32,
    game_rank_weight: f32,
    rank_margin_cp: f32,
    rank_temperature_cp: f32,
    min_rank_gap_cp: f32,
    game_regret_cap_cp: f32,
    max_pairs_per_record: usize,
}

impl Weights {
    fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let flat = HalfKpFlatModel::decode(&bytes)?;
        if flat.header.target_scale != TARGET_SCALE {
            return Err(anyhow!("incompatible HalfKP model header"));
        }
        Ok(Self {
            feature_emb: flat.feature_emb,
            hidden_b: flat.hidden_b,
            out_w: flat.out_w,
            out_b: flat.out_b,
        })
    }

    fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension("tmp");
        let mut writer = BufWriter::new(File::create(&temporary)?);
        HalfKpFlatModel::write_parts(
            &mut writer,
            HalfKpHeader::current(TARGET_SCALE)?,
            &self.feature_emb,
            &self.hidden_b,
            &self.out_w,
            self.out_b,
        )?;
        writer.flush()?;
        drop(writer);
        fs::rename(temporary, path)?;
        Ok(())
    }

    fn zeros_like(&self) -> Self {
        Self {
            feature_emb: vec![0.0; self.feature_emb.len()],
            hidden_b: [0.0; HALFKP_HIDDEN],
            out_w: [0.0; HALFKP_HIDDEN * 2 + 1],
            out_b: 0.0,
        }
    }

    fn add_to_average(&mut self, value: &Self, count: u64) {
        let rate = 1.0 / count as f32;
        for (average, &current) in self.feature_emb.iter_mut().zip(&value.feature_emb) {
            *average += (current - *average) * rate;
        }
        for (average, &current) in self.hidden_b.iter_mut().zip(&value.hidden_b) {
            *average += (current - *average) * rate;
        }
        for (average, &current) in self.out_w.iter_mut().zip(&value.out_w) {
            *average += (current - *average) * rate;
        }
        self.out_b += (value.out_b - self.out_b) * rate;
    }

    fn forward(&self, position: &PackedHalfKpPosition) -> Forward {
        let forward = HalfKpFlatModel::forward_parts(
            &self.feature_emb,
            &self.hidden_b,
            &self.out_w,
            self.out_b,
            position
                .features_black
                .iter()
                .map(|&feature| feature as usize),
            position
                .features_white
                .iter()
                .map(|&feature| feature as usize),
            position.material_black,
            position.material_white,
            position.side_to_move == Color::Black,
            TARGET_SCALE,
        );
        Forward {
            black: forward.black,
            white: forward.white,
            raw: forward.raw,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let pool = if args.threads > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(args.threads)
                .build()?,
        )
    } else {
        None
    };
    let mut valid = read_datasets(
        &args.valid,
        args.max_valid_records,
        args.allow_legacy_teacher_semantics,
    )?;
    let test = read_datasets(&args.test, None, args.allow_legacy_teacher_semantics)?;
    if valid.is_empty() {
        return Err(anyhow!("train and valid datasets must not be empty"));
    }
    let train_records = count_datasets(
        &args.train,
        args.max_train_records,
        args.allow_legacy_teacher_semantics,
    )?;
    if train_records == 0 {
        return Err(anyhow!("train and valid datasets must not be empty"));
    }
    println!(
        "datasets train_shards={} train_records={} valid={} test={} model_parameters={}",
        args.train.len(),
        train_records,
        valid.len(),
        test.len(),
        HALFKP_INPUTS * HALFKP_HIDDEN + HALFKP_HIDDEN * 3 + 2
    );
    let mut kappa_cp = args.kappa_cp;
    if args.fit_kappa {
        kappa_cp = fit_kappa(&valid);
        println!("fitted_kappa_cp={kappa_cp:.1}");
    }
    let base_options = TrainOptions {
        kappa_cp: args.kappa_cp,
        search_mix: args.search_mix,
        loss_power: args.loss_power,
        rank_weight: args.rank_weight,
        game_rank_weight: args.game_rank_weight,
        rank_margin_cp: args.rank_margin_cp,
        rank_temperature_cp: args.rank_temperature_cp,
        min_rank_gap_cp: args.min_rank_gap_cp,
        game_regret_cap_cp: args.game_regret_cap_cp,
        max_pairs_per_record: args.max_pairs_per_record,
    };
    let mut options = TrainOptions {
        kappa_cp,
        ..base_options
    };
    let (mut weights, mut optimizer, mut swa, start_epoch, mut best, mut stale) =
        initialize_training(&args)?;
    let mut log = create_log(args.log.as_deref())?;
    options.search_mix = args.search_mix_end.unwrap_or(args.search_mix);
    valid.shuffle(&mut ChaCha8Rng::seed_from_u64(args.seed ^ 0x0056_414c_4944));
    let initial_weights = evaluation_weights(&weights, &optimizer, args.schedule_free_beta1);
    let initial = evaluate(&initial_weights, &valid, &options);
    print_metrics("initial_valid", 0, &initial);
    if start_epoch == 0 {
        best = validation_score(&initial);
        initial_weights.save(&args.output)?;
        println!(
            "initial checkpoint score={best:.8} output={}",
            args.output.display()
        );
    }
    for epoch in start_epoch + 1..=args.epochs {
        options.search_mix = epoch_search_mix(&args, epoch);
        let mut shard_order = args.train.clone();
        shard_order.shuffle(&mut ChaCha8Rng::seed_from_u64(
            args.seed ^ epoch as u64 ^ 0x0053_4841_5244,
        ));
        let mut epoch_value_loss = 0.0;
        let mut epoch_rank_loss = 0.0;
        let mut epoch_pairs = 0usize;
        let mut seen = 0usize;
        for (shard_index, path) in shard_order.iter().enumerate() {
            let remaining = args
                .max_train_records
                .map(|limit| limit.saturating_sub(seen));
            if remaining == Some(0) {
                break;
            }
            let mut shard = read_dataset(path, remaining, args.allow_legacy_teacher_semantics)?;
            shard.shuffle(&mut ChaCha8Rng::seed_from_u64(
                args.seed ^ epoch as u64 ^ shard_index as u64,
            ));
            for batch in shard.chunks(args.batch_size) {
                let compute = || {
                    batch
                        .par_iter()
                        .map(|record| compute_record_gradient(&weights, record, &options))
                        .collect::<Vec<_>>()
                };
                let gradients = pool
                    .as_ref()
                    .map_or_else(compute, |pool| pool.install(compute));
                let (value_loss, rank_loss, pairs) =
                    apply_batch(&mut weights, &mut optimizer, &gradients, &args);
                epoch_value_loss += value_loss;
                epoch_rank_loss += rank_loss;
                epoch_pairs += pairs;
            }
            seen += shard.len();
            println!(
                "epoch={epoch} shard={}/{} records={} total={seen}",
                shard_index + 1,
                shard_order.len(),
                shard.len()
            );
        }
        let eval_weights = evaluation_weights(&weights, &optimizer, args.schedule_free_beta1);
        if epoch >= args.swa_start_epoch {
            let state = swa.get_or_insert_with(|| SwaState {
                average: eval_weights.zeros_like(),
                count: 0,
            });
            state.count += 1;
            state.average.add_to_average(&eval_weights, state.count);
        }
        options.search_mix = args.search_mix_end.unwrap_or(args.search_mix);
        let valid_metrics = evaluate(&eval_weights, &valid, &options);
        print_metrics("valid", epoch, &valid_metrics);
        let mut score = validation_score(&valid_metrics);
        let mut selected = &eval_weights;
        if let Some(state) = &swa {
            let swa_metrics = evaluate(&state.average, &valid, &options);
            print_metrics("swa_valid", epoch, &swa_metrics);
            let swa_score = validation_score(&swa_metrics);
            if swa_score < score {
                score = swa_score;
                selected = &state.average;
            }
        }
        if let Some(writer) = log.as_mut() {
            writeln!(
                writer,
                "{epoch},{seen},{:.8},{:.8},{:.6},{:.6},{:.6},{:.6},{:.3},{:.8}",
                epoch_value_loss / seen.max(1) as f64,
                epoch_rank_loss / epoch_pairs.max(1) as f64,
                valid_metrics.value_loss / valid_metrics.records.max(1) as f64,
                valid_metrics.search_abs_error / valid_metrics.records.max(1) as f64,
                valid_metrics.top1_matches as f64 / valid_metrics.records.max(1) as f64,
                valid_metrics.pair_correct as f64 / valid_metrics.pairs.max(1) as f64,
                valid_metrics.regret_sum / valid_metrics.records.max(1) as f64,
                score
            )?;
            writer.flush()?;
        }
        if best - score >= args.min_valid_improvement {
            best = score;
            stale = 0;
            selected.save(&args.output)?;
            println!(
                "best epoch={epoch} score={score:.8} output={}",
                args.output.display()
            );
        } else {
            stale += 1;
            if stale >= args.early_stop_patience {
                println!("early_stop epoch={epoch} best={best:.8}");
            }
        }
        save_checkpoint(
            &args,
            &weights,
            &mut optimizer,
            swa.as_ref(),
            epoch,
            best,
            stale,
        )?;
        if stale >= args.early_stop_patience {
            break;
        }
    }
    if !test.is_empty() {
        let final_weights = Weights::load(&args.output)?;
        let test_metrics = evaluate(&final_weights, &test, &options);
        print_metrics("final_test", args.epochs, &test_metrics);
    }
    Ok(())
}

fn validate_args(args: &Args) -> Result<()> {
    if args.epochs == 0
        || args.batch_size == 0
        || args.learning_rate <= 0.0
        || args.output_learning_rate <= 0.0
        || args.kappa_cp <= 0.0
        || !(0.0..=1.0).contains(&args.search_mix)
        || args
            .search_mix_end
            .is_some_and(|value| !(0.0..=1.0).contains(&value))
        || !(1.0..=4.0).contains(&args.loss_power)
        || args.rank_weight < 0.0
        || args.game_rank_weight < 0.0
        || args.rank_temperature_cp <= 0.0
        || args.gradient_clip_norm <= 0.0
        || args.output_limit <= 0.0
        || args.early_stop_patience == 0
        || args.max_pairs_per_record == 0
        || !(0.0..1.0).contains(&args.schedule_free_beta1)
        || !(0.0..1.0).contains(&args.schedule_free_beta2)
    {
        return Err(anyhow!("invalid training arguments"));
    }
    if args.resume && args.checkpoint_dir.is_none() {
        return Err(anyhow!("--resume requires --checkpoint-dir"));
    }
    Ok(())
}

fn read_dataset(
    path: &Path,
    limit: Option<usize>,
    allow_legacy_teacher_semantics: bool,
) -> Result<Vec<SearchTeacherRecord>> {
    validate_teacher_semantics(path, allow_legacy_teacher_semantics)?;
    let mut reader = SearchTeacherReader::open(path)?;
    let mut records = Vec::new();
    while limit.is_none_or(|limit| records.len() < limit) {
        let Some(record) = reader.read_record()? else {
            break;
        };
        records.push(record);
    }
    Ok(records)
}

fn validate_teacher_semantics(path: &Path, allow_legacy_teacher_semantics: bool) -> Result<()> {
    let Some(manifest) = read_search_teacher_manifest(path)? else {
        return if allow_legacy_teacher_semantics {
            Ok(())
        } else {
            Err(anyhow!(
                "{} has no teacher-semantics manifest; regenerate it or pass \
                 --allow-legacy-teacher-semantics explicitly",
                path.display()
            ))
        };
    };
    if manifest.format != "HKST0002"
        || manifest.teacher_semantics_version != SEARCH_TEACHER_SEMANTICS_VERSION
        || manifest.teacher_semantics_id != SEARCH_TEACHER_SEMANTICS_ID
    {
        return Err(anyhow!(
            "{} uses unsupported teacher semantics {} ({}) for {}; expected {} ({})",
            path.display(),
            manifest.teacher_semantics_version,
            manifest.teacher_semantics_id,
            manifest.format,
            SEARCH_TEACHER_SEMANTICS_VERSION,
            SEARCH_TEACHER_SEMANTICS_ID
        ));
    }
    Ok(())
}

fn read_datasets(
    paths: &[PathBuf],
    limit: Option<usize>,
    allow_legacy_teacher_semantics: bool,
) -> Result<Vec<SearchTeacherRecord>> {
    let mut records = Vec::new();
    for path in paths {
        let remaining = limit.map(|limit| limit.saturating_sub(records.len()));
        if remaining == Some(0) {
            break;
        }
        records.extend(read_dataset(
            path,
            remaining,
            allow_legacy_teacher_semantics,
        )?);
    }
    Ok(records)
}

fn count_datasets(
    paths: &[PathBuf],
    limit: Option<usize>,
    allow_legacy_teacher_semantics: bool,
) -> Result<usize> {
    let mut count = 0;
    for path in paths {
        let remaining = limit.map(|limit| limit.saturating_sub(count));
        if remaining == Some(0) {
            break;
        }
        count += read_dataset(path, remaining, allow_legacy_teacher_semantics)?.len();
    }
    Ok(count)
}

fn create_log(path: Option<&Path>) -> Result<Option<BufWriter<File>>> {
    let Some(path) = path else {
        return Ok(None);
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let exists = path.exists();
    let mut writer = BufWriter::new(OpenOptions::new().create(true).append(true).open(path)?);
    if !exists {
        writeln!(
            writer,
            "epoch,train_records,train_value_loss,train_rank_loss,valid_value_loss,valid_search_mae,valid_top1,valid_pair_accuracy,valid_regret_cp,score"
        )?;
    }
    Ok(Some(writer))
}

fn epoch_search_mix(args: &Args, epoch: usize) -> f32 {
    let end = args.search_mix_end.unwrap_or(args.search_mix);
    if args.epochs <= 1 {
        return end;
    }
    let progress = (epoch.saturating_sub(1) as f32 / (args.epochs - 1) as f32).clamp(0.0, 1.0);
    args.search_mix + (end - args.search_mix) * progress
}

fn initialize_training(
    args: &Args,
) -> Result<(Weights, OptimizerState, Option<SwaState>, usize, f64, usize)> {
    if args.resume {
        let directory = args
            .checkpoint_dir
            .as_ref()
            .expect("validated checkpoint dir");
        let meta: CheckpointMeta =
            serde_json::from_reader(BufReader::new(File::open(directory.join("state.json"))?))?;
        if meta.optimizer != args.optimizer {
            return Err(anyhow!("checkpoint optimizer does not match --optimizer"));
        }
        let weights = Weights::load(&directory.join("current.binary"))?;
        let optimizer = match args.optimizer {
            OptimizerKind::Adagrad => {
                OptimizerState::Adagrad(Box::new(Weights::load(&directory.join("adagrad.binary"))?))
            }
            OptimizerKind::ScheduleFree => {
                let mut last_reader =
                    BufReader::new(File::open(directory.join("schedule_last.bin"))?);
                let mut last_weight_sum = vec![0.0; HALFKP_INPUTS];
                for value in &mut last_weight_sum {
                    let mut bytes = [0; 8];
                    last_reader.read_exact(&mut bytes)?;
                    *value = f64::from_le_bytes(bytes);
                }
                OptimizerState::ScheduleFree(Box::new(ScheduleFreeState {
                    z: Weights::load(&directory.join("schedule_z.binary"))?,
                    variance: Weights::load(&directory.join("schedule_variance.binary"))?,
                    last_weight_sum,
                    step: meta.schedule_step,
                    weight_sum: meta.schedule_weight_sum,
                    lr_max: meta.schedule_lr_max,
                }))
            }
        };
        let swa = if meta.swa_count > 0 {
            Some(SwaState {
                average: Weights::load(&directory.join("swa.binary"))?,
                count: meta.swa_count,
            })
        } else {
            None
        };
        println!("resumed checkpoint epoch={}", meta.completed_epoch);
        Ok((
            weights,
            optimizer,
            swa,
            meta.completed_epoch,
            meta.best_score,
            meta.stale_epochs,
        ))
    } else {
        let weights = Weights::load(&args.init)?;
        let optimizer = match args.optimizer {
            OptimizerKind::Adagrad => OptimizerState::Adagrad(Box::new(weights.zeros_like())),
            OptimizerKind::ScheduleFree => {
                OptimizerState::ScheduleFree(Box::new(ScheduleFreeState {
                    z: weights.clone(),
                    variance: weights.zeros_like(),
                    last_weight_sum: vec![0.0; HALFKP_INPUTS],
                    step: 0,
                    weight_sum: 0.0,
                    lr_max: 0.0,
                }))
            }
        };
        Ok((weights, optimizer, None, 0, f64::INFINITY, 0))
    }
}

fn save_checkpoint(
    args: &Args,
    weights: &Weights,
    optimizer: &mut OptimizerState,
    swa: Option<&SwaState>,
    epoch: usize,
    best_score: f64,
    stale_epochs: usize,
) -> Result<()> {
    let Some(directory) = &args.checkpoint_dir else {
        return Ok(());
    };
    fs::create_dir_all(directory)?;
    weights.save(&directory.join("current.binary"))?;
    let (schedule_step, schedule_weight_sum, schedule_lr_max) = match optimizer {
        OptimizerState::Adagrad(state) => {
            state.save(&directory.join("adagrad.binary"))?;
            (0, 0.0, 0.0)
        }
        OptimizerState::ScheduleFree(state) => {
            state.z.save(&directory.join("schedule_z.binary"))?;
            state
                .variance
                .save(&directory.join("schedule_variance.binary"))?;
            let mut writer = BufWriter::new(File::create(directory.join("schedule_last.bin.tmp"))?);
            for value in &state.last_weight_sum {
                writer.write_all(&value.to_le_bytes())?;
            }
            writer.flush()?;
            drop(writer);
            fs::rename(
                directory.join("schedule_last.bin.tmp"),
                directory.join("schedule_last.bin"),
            )?;
            (state.step, state.weight_sum, state.lr_max)
        }
    };
    if let Some(state) = swa {
        state.average.save(&directory.join("swa.binary"))?;
    }
    let meta = CheckpointMeta {
        completed_epoch: epoch,
        best_score,
        stale_epochs,
        optimizer: args.optimizer,
        schedule_step,
        schedule_weight_sum,
        schedule_lr_max,
        swa_count: swa.map_or(0, |state| state.count),
    };
    let temporary = directory.join("state.json.tmp");
    serde_json::to_writer_pretty(BufWriter::new(File::create(&temporary)?), &meta)?;
    fs::rename(temporary, directory.join("state.json"))?;
    println!("checkpoint epoch={epoch} directory={}", directory.display());
    Ok(())
}

fn fit_kappa(records: &[SearchTeacherRecord]) -> f32 {
    (100..=1500)
        .step_by(25)
        .map(|kappa| {
            let loss = records
                .iter()
                .filter_map(|record| {
                    record.result.map(|result| {
                        let prediction = sigmoid(record.root_search_score_cp / kappa as f32);
                        (prediction - result).powi(2) as f64
                    })
                })
                .sum::<f64>();
            (kappa, loss)
        })
        .min_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
        .map_or(600.0, |(kappa, _)| kappa as f32)
}

fn compute_record_gradient(
    weights: &Weights,
    record: &SearchTeacherRecord,
    options: &TrainOptions,
) -> RecordGradient {
    let mut gradient = RecordGradient {
        dense: DenseGradient::default(),
        features: Vec::new(),
        value_loss: 0.0,
        rank_loss: 0.0,
        pairs: 0,
    };
    let root_forward = weights.forward(&record.root);
    let model_cp = root_forward.raw * TARGET_SCALE;
    let model_p = sigmoid(model_cp / options.kappa_cp);
    let search_p = sigmoid(record.root_search_score_cp / options.kappa_cp);
    let target = record
        .result
        .map(|result| options.search_mix * search_p + (1.0 - options.search_mix) * result)
        .unwrap_or(search_p);
    let error = model_p - target;
    let absolute_error = error.abs().max(1e-8);
    gradient.value_loss = (absolute_error.powf(options.loss_power) * record.sample_weight) as f64;
    let d_probability = options.loss_power
        * error.signum()
        * absolute_error.powf(options.loss_power - 1.0)
        * record.sample_weight;
    let d_score_cp = d_probability * model_p * (1.0 - model_p) / options.kappa_cp;
    add_position_gradient(
        weights,
        &record.root,
        &root_forward,
        d_score_cp * TARGET_SCALE,
        &mut gradient,
    );

    let forwards = record
        .candidates
        .iter()
        .map(|candidate| weights.forward(&candidate.child))
        .collect::<Vec<_>>();
    let utilities = forwards
        .iter()
        .map(|forward| -forward.raw * TARGET_SCALE)
        .collect::<Vec<_>>();
    let teacher_best = record
        .candidates
        .iter()
        .enumerate()
        .max_by(|lhs, rhs| lhs.1.score_cp.total_cmp(&rhs.1.score_cp))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let mut bad = (0..record.candidates.len())
        .filter(|&index| {
            index != teacher_best
                && record.candidates[teacher_best].score_cp - record.candidates[index].score_cp
                    >= options.min_rank_gap_cp
        })
        .collect::<Vec<_>>();
    bad.sort_by(|&lhs, &rhs| utilities[rhs].total_cmp(&utilities[lhs]));
    for bad_index in bad.into_iter().take(options.max_pairs_per_record) {
        add_rank_pair(
            weights,
            record,
            &forwards,
            &utilities,
            teacher_best,
            bad_index,
            options.rank_weight * record.sample_weight,
            options,
            &mut gradient,
        );
    }

    if options.game_rank_weight > 0.0 {
        if let Some(game_index) = record
            .candidates
            .iter()
            .position(|candidate| candidate.flags & CANDIDATE_GAME_MOVE != 0)
        {
            let regret =
                record.candidates[teacher_best].score_cp - record.candidates[game_index].score_cp;
            if regret <= options.game_regret_cap_cp {
                if let Some(bad_index) = (0..record.candidates.len())
                    .filter(|&index| {
                        index != game_index
                            && record.candidates[game_index].score_cp
                                - record.candidates[index].score_cp
                                >= options.min_rank_gap_cp
                    })
                    .max_by(|&lhs, &rhs| utilities[lhs].total_cmp(&utilities[rhs]))
                {
                    add_rank_pair(
                        weights,
                        record,
                        &forwards,
                        &utilities,
                        game_index,
                        bad_index,
                        options.game_rank_weight * record.sample_weight,
                        options,
                        &mut gradient,
                    );
                }
            }
        }
    }
    gradient
}

#[allow(clippy::too_many_arguments)]
fn add_rank_pair(
    weights: &Weights,
    record: &SearchTeacherRecord,
    forwards: &[Forward],
    utilities: &[f32],
    good_index: usize,
    bad_index: usize,
    weight: f32,
    options: &TrainOptions,
    gradient: &mut RecordGradient,
) {
    if weight <= 0.0 {
        return;
    }
    let x = (utilities[bad_index] - utilities[good_index] + options.rank_margin_cp)
        / options.rank_temperature_cp;
    let derivative = sigmoid(x) / options.rank_temperature_cp * weight;
    gradient.rank_loss += (softplus(x) * weight) as f64;
    gradient.pairs += 1;
    // utility(child) = -score(child)
    add_position_gradient(
        weights,
        &record.candidates[good_index].child,
        &forwards[good_index],
        derivative * TARGET_SCALE,
        gradient,
    );
    add_position_gradient(
        weights,
        &record.candidates[bad_index].child,
        &forwards[bad_index],
        -derivative * TARGET_SCALE,
        gradient,
    );
}

fn add_position_gradient(
    weights: &Weights,
    position: &PackedHalfKpPosition,
    forward: &Forward,
    d_raw: f32,
    gradient: &mut RecordGradient,
) {
    let (stm, nstm, material) = if position.side_to_move == Color::Black {
        (&forward.black, &forward.white, position.material_black)
    } else {
        (&forward.white, &forward.black, position.material_white)
    };
    let mut stm_hidden = [0.0; HALFKP_HIDDEN];
    let mut nstm_hidden = [0.0; HALFKP_HIDDEN];
    for h in 0..HALFKP_HIDDEN {
        gradient.dense.out_w[h] += d_raw * stm[h].clamp(0.0, 1.0);
        gradient.dense.out_w[HALFKP_HIDDEN + h] += d_raw * nstm[h].clamp(0.0, 1.0);
        stm_hidden[h] = d_raw * weights.out_w[h] * active_derivative(stm[h]);
        nstm_hidden[h] = d_raw * weights.out_w[HALFKP_HIDDEN + h] * active_derivative(nstm[h]);
    }
    gradient.dense.out_w[HALFKP_HIDDEN * 2] += d_raw * material / TARGET_SCALE;
    gradient.dense.out_b += d_raw;
    let (black_hidden, white_hidden) = if position.side_to_move == Color::Black {
        (stm_hidden, nstm_hidden)
    } else {
        (nstm_hidden, stm_hidden)
    };
    for h in 0..HALFKP_HIDDEN {
        gradient.dense.hidden_b[h] += black_hidden[h] + white_hidden[h];
    }
    gradient.features.extend(
        position
            .features_black
            .iter()
            .map(|&feature| (feature as usize, black_hidden)),
    );
    gradient.features.extend(
        position
            .features_white
            .iter()
            .map(|&feature| (feature as usize, white_hidden)),
    );
}

fn apply_batch(
    weights: &mut Weights,
    optimizer: &mut OptimizerState,
    records: &[RecordGradient],
    args: &Args,
) -> (f64, f64, usize) {
    let mut dense = DenseGradient::default();
    let mut sparse: HashMap<usize, [f32; HALFKP_HIDDEN]> = HashMap::new();
    let mut value_loss = 0.0;
    let mut rank_loss = 0.0;
    let mut pairs = 0;
    for record in records {
        value_loss += record.value_loss;
        rank_loss += record.rank_loss;
        pairs += record.pairs;
        dense.out_b += record.dense.out_b;
        for h in 0..HALFKP_HIDDEN {
            dense.hidden_b[h] += record.dense.hidden_b[h];
        }
        for index in 0..dense.out_w.len() {
            dense.out_w[index] += record.dense.out_w[index];
        }
        for &(feature, values) in &record.features {
            let row = sparse.entry(feature).or_insert([0.0; HALFKP_HIDDEN]);
            for h in 0..HALFKP_HIDDEN {
                row[h] += values[h];
            }
        }
    }
    let scale = 1.0 / records.len().max(1) as f32;
    let mut norm_squared = (dense.out_b * scale).powi(2);
    for value in dense.hidden_b.iter_mut().chain(dense.out_w.iter_mut()) {
        *value *= scale;
        norm_squared += *value * *value;
    }
    dense.out_b *= scale;
    for row in sparse.values_mut() {
        for value in row {
            *value *= scale;
            norm_squared += *value * *value;
        }
    }
    let norm = norm_squared.sqrt();
    let clip = if norm > args.gradient_clip_norm {
        args.gradient_clip_norm / norm
    } else {
        1.0
    };
    dense.out_b *= clip;
    for value in dense.hidden_b.iter_mut().chain(dense.out_w.iter_mut()) {
        *value *= clip;
    }
    for row in sparse.values_mut() {
        for value in row {
            *value *= clip;
        }
    }
    match optimizer {
        OptimizerState::Adagrad(state) => apply_adagrad(weights, state, &dense, &sparse, args),
        OptimizerState::ScheduleFree(state) => {
            apply_schedule_free(weights, state, &dense, &sparse, args)
        }
    }
    (value_loss, rank_loss, pairs)
}

fn adagrad_scalar(
    weight: &mut f32,
    accumulator: &mut f32,
    gradient: f32,
    learning_rate: f32,
    limit: f32,
) {
    *accumulator += gradient * gradient;
    *weight =
        (*weight - learning_rate * gradient / (accumulator.sqrt() + 1e-8)).clamp(-limit, limit);
}

fn apply_adagrad(
    weights: &mut Weights,
    state: &mut Weights,
    dense: &DenseGradient,
    sparse: &HashMap<usize, [f32; HALFKP_HIDDEN]>,
    args: &Args,
) {
    adagrad_scalar(
        &mut weights.out_b,
        &mut state.out_b,
        dense.out_b,
        args.output_learning_rate,
        args.output_limit,
    );
    for h in 0..HALFKP_HIDDEN {
        adagrad_scalar(
            &mut weights.hidden_b[h],
            &mut state.hidden_b[h],
            dense.hidden_b[h],
            args.learning_rate,
            1.0,
        );
    }
    for index in 0..weights.out_w.len() {
        adagrad_scalar(
            &mut weights.out_w[index],
            &mut state.out_w[index],
            dense.out_w[index],
            args.output_learning_rate,
            args.output_limit,
        );
    }
    for (&feature, row) in sparse {
        let start = feature * HALFKP_HIDDEN;
        for (h, &gradient) in row.iter().enumerate() {
            adagrad_scalar(
                &mut weights.feature_emb[start + h],
                &mut state.feature_emb[start + h],
                gradient,
                args.learning_rate,
                1.0,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn schedule_free_scalar(
    y: &mut f32,
    z: &mut f32,
    variance: &mut f32,
    gradient: f32,
    learning_rate: f32,
    ckp1: f32,
    beta1: f32,
    beta2: f32,
    bias_correction2: f32,
    limit: f32,
) {
    *variance = beta2 * *variance + (1.0 - beta2) * gradient * gradient;
    let normalized = gradient / ((*variance / bias_correction2).sqrt() + 1e-8);
    *y =
        ((1.0 - ckp1) * *y + ckp1 * *z + learning_rate * (beta1 * (1.0 - ckp1) - 1.0) * normalized)
            .clamp(-limit, limit);
    *z = (*z - learning_rate * normalized).clamp(-limit, limit);
}

fn sync_sparse_row(weights: &mut Weights, state: &mut ScheduleFreeState, feature: usize) {
    let previous = state.last_weight_sum[feature];
    if state.weight_sum > 0.0 && previous < state.weight_sum {
        let keep = (previous / state.weight_sum) as f32;
        let start = feature * HALFKP_HIDDEN;
        for h in 0..HALFKP_HIDDEN {
            weights.feature_emb[start + h] = keep * weights.feature_emb[start + h]
                + (1.0 - keep) * state.z.feature_emb[start + h];
        }
    }
}

fn apply_schedule_free(
    weights: &mut Weights,
    state: &mut ScheduleFreeState,
    dense: &DenseGradient,
    sparse: &HashMap<usize, [f32; HALFKP_HIDDEN]>,
    args: &Args,
) {
    state.step += 1;
    let warmup = (state.step as f32 / args.schedule_free_warmup_steps.max(1) as f32).min(1.0);
    let scheduled_lr = args.learning_rate * warmup;
    state.lr_max = state.lr_max.max(scheduled_lr);
    let update_weight = (state.lr_max as f64).powi(2);
    let new_weight_sum = state.weight_sum + update_weight;
    let ckp1 = if new_weight_sum > 0.0 {
        (update_weight / new_weight_sum) as f32
    } else {
        1.0
    };
    let bias_correction2 = 1.0 - args.schedule_free_beta2.powf(state.step as f32);
    for &feature in sparse.keys() {
        sync_sparse_row(weights, state, feature);
    }
    schedule_free_scalar(
        &mut weights.out_b,
        &mut state.z.out_b,
        &mut state.variance.out_b,
        dense.out_b,
        args.output_learning_rate * warmup,
        ckp1,
        args.schedule_free_beta1,
        args.schedule_free_beta2,
        bias_correction2,
        args.output_limit,
    );
    for h in 0..HALFKP_HIDDEN {
        schedule_free_scalar(
            &mut weights.hidden_b[h],
            &mut state.z.hidden_b[h],
            &mut state.variance.hidden_b[h],
            dense.hidden_b[h],
            scheduled_lr,
            ckp1,
            args.schedule_free_beta1,
            args.schedule_free_beta2,
            bias_correction2,
            1.0,
        );
    }
    for index in 0..weights.out_w.len() {
        schedule_free_scalar(
            &mut weights.out_w[index],
            &mut state.z.out_w[index],
            &mut state.variance.out_w[index],
            dense.out_w[index],
            args.output_learning_rate * warmup,
            ckp1,
            args.schedule_free_beta1,
            args.schedule_free_beta2,
            bias_correction2,
            args.output_limit,
        );
    }
    for (&feature, row) in sparse {
        let start = feature * HALFKP_HIDDEN;
        for (h, &gradient) in row.iter().enumerate() {
            schedule_free_scalar(
                &mut weights.feature_emb[start + h],
                &mut state.z.feature_emb[start + h],
                &mut state.variance.feature_emb[start + h],
                gradient,
                scheduled_lr,
                ckp1,
                args.schedule_free_beta1,
                args.schedule_free_beta2,
                bias_correction2,
                1.0,
            );
        }
        state.last_weight_sum[feature] = new_weight_sum;
    }
    state.weight_sum = new_weight_sum;
}

fn evaluation_weights(weights: &Weights, optimizer: &OptimizerState, beta1: f32) -> Weights {
    let OptimizerState::ScheduleFree(state) = optimizer else {
        return weights.clone();
    };
    let mut evaluated = weights.clone();
    if state.weight_sum > 0.0 {
        for feature in 0..HALFKP_INPUTS {
            let keep = (state.last_weight_sum[feature] / state.weight_sum) as f32;
            let start = feature * HALFKP_HIDDEN;
            for h in 0..HALFKP_HIDDEN {
                evaluated.feature_emb[start + h] = keep * evaluated.feature_emb[start + h]
                    + (1.0 - keep) * state.z.feature_emb[start + h];
            }
        }
    }
    let interpolation = 1.0 - 1.0 / beta1;
    for (value, &z) in evaluated.feature_emb.iter_mut().zip(&state.z.feature_emb) {
        *value += interpolation * (z - *value);
    }
    for (value, &z) in evaluated.hidden_b.iter_mut().zip(&state.z.hidden_b) {
        *value += interpolation * (z - *value);
    }
    for (value, &z) in evaluated.out_w.iter_mut().zip(&state.z.out_w) {
        *value += interpolation * (z - *value);
    }
    evaluated.out_b += interpolation * (state.z.out_b - evaluated.out_b);
    evaluated
}

fn evaluate(weights: &Weights, records: &[SearchTeacherRecord], options: &TrainOptions) -> Metrics {
    records
        .par_iter()
        .map(|record| evaluate_record(weights, record, options))
        .reduce(Metrics::default, merge_metrics)
}

fn evaluate_record(
    weights: &Weights,
    record: &SearchTeacherRecord,
    options: &TrainOptions,
) -> Metrics {
    let root_score = weights.forward(&record.root).raw * TARGET_SCALE;
    let model_p = sigmoid(root_score / options.kappa_cp);
    let search_p = sigmoid(record.root_search_score_cp / options.kappa_cp);
    let target = record
        .result
        .map(|result| options.search_mix * search_p + (1.0 - options.search_mix) * result)
        .unwrap_or(search_p);
    let utilities = record
        .candidates
        .iter()
        .map(|candidate| -weights.forward(&candidate.child).raw * TARGET_SCALE)
        .collect::<Vec<_>>();
    let teacher_best = record
        .candidates
        .iter()
        .enumerate()
        .max_by(|lhs, rhs| lhs.1.score_cp.total_cmp(&rhs.1.score_cp))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let model_best = utilities
        .iter()
        .enumerate()
        .max_by(|lhs, rhs| lhs.1.total_cmp(rhs.1))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let mut metrics = Metrics {
        records: 1,
        known_results: usize::from(record.result.is_some()),
        value_loss: ((model_p - target).powi(2) * record.sample_weight) as f64,
        search_abs_error: (root_score - record.root_search_score_cp).abs() as f64,
        top1_matches: usize::from(teacher_best == model_best),
        regret_sum: (record.candidates[teacher_best].score_cp
            - record.candidates[model_best].score_cp) as f64,
        log_loss: (-(target * model_p.max(1e-7).ln()
            + (1.0 - target) * (1.0 - model_p).max(1e-7).ln())
            * record.sample_weight) as f64,
        ..Metrics::default()
    };
    let regret_bucket =
        ((metrics.regret_sum.max(0.0) / 50.0) as usize).min(metrics.regret_histogram.len() - 1);
    metrics.regret_histogram[regret_bucket] = 1;
    let phase = (record.phase as usize).min(2);
    metrics.phase_records[phase] = 1;
    metrics.phase_brier[phase] = metrics.value_loss;
    for bad_index in 0..record.candidates.len() {
        if bad_index == teacher_best
            || record.candidates[teacher_best].score_cp - record.candidates[bad_index].score_cp
                < options.min_rank_gap_cp
        {
            continue;
        }
        metrics.pairs += 1;
        metrics.pair_correct += usize::from(utilities[teacher_best] > utilities[bad_index]);
    }
    if let Some(game_index) = record
        .candidates
        .iter()
        .position(|candidate| candidate.flags & CANDIDATE_GAME_MOVE != 0)
    {
        metrics.game_moves = 1;
        metrics.game_move_top1 = usize::from(game_index == model_best);
    }
    metrics
}

fn merge_metrics(mut lhs: Metrics, rhs: Metrics) -> Metrics {
    lhs.records += rhs.records;
    lhs.known_results += rhs.known_results;
    lhs.value_loss += rhs.value_loss;
    lhs.search_abs_error += rhs.search_abs_error;
    lhs.pairs += rhs.pairs;
    lhs.pair_correct += rhs.pair_correct;
    lhs.top1_matches += rhs.top1_matches;
    lhs.game_moves += rhs.game_moves;
    lhs.game_move_top1 += rhs.game_move_top1;
    lhs.regret_sum += rhs.regret_sum;
    lhs.log_loss += rhs.log_loss;
    for index in 0..lhs.regret_histogram.len() {
        lhs.regret_histogram[index] += rhs.regret_histogram[index];
    }
    for phase in 0..3 {
        lhs.phase_records[phase] += rhs.phase_records[phase];
        lhs.phase_brier[phase] += rhs.phase_brier[phase];
    }
    lhs
}

fn validation_score(metrics: &Metrics) -> f64 {
    let brier = metrics.value_loss / metrics.records.max(1) as f64;
    let pair_error = 1.0 - metrics.pair_correct as f64 / metrics.pairs.max(1) as f64;
    let regret = metrics.regret_sum / metrics.records.max(1) as f64;
    brier + 0.02 * pair_error + regret / 10_000.0
}

fn regret_percentile(metrics: &Metrics, percentile: f64) -> f64 {
    let target = (metrics.records as f64 * percentile).ceil() as u64;
    let mut cumulative = 0;
    for (index, count) in metrics.regret_histogram.iter().enumerate() {
        cumulative += count;
        if cumulative >= target {
            return index as f64 * 50.0;
        }
    }
    6000.0
}

fn print_metrics(label: &str, epoch: usize, metrics: &Metrics) {
    println!(
        "{label} epoch={epoch} records={} known_results={} brier={:.8} logloss={:.6} search_mae={:.3} top1={:.4} pair={:.4} regret_mean={:.2} regret_p95={:.1} game_top1={:.4} phase_brier={:.8}/{:.8}/{:.8}",
        metrics.records,
        metrics.known_results,
        metrics.value_loss / metrics.records.max(1) as f64,
        metrics.log_loss / metrics.records.max(1) as f64,
        metrics.search_abs_error / metrics.records.max(1) as f64,
        metrics.top1_matches as f64 / metrics.records.max(1) as f64,
        metrics.pair_correct as f64 / metrics.pairs.max(1) as f64,
        metrics.regret_sum / metrics.records.max(1) as f64,
        regret_percentile(metrics, 0.95),
        metrics.game_move_top1 as f64 / metrics.game_moves.max(1) as f64,
        metrics.phase_brier[0] / metrics.phase_records[0].max(1) as f64,
        metrics.phase_brier[1] / metrics.phase_records[1].max(1) as f64,
        metrics.phase_brier[2] / metrics.phase_records[2].max(1) as f64,
    );
}

fn active_derivative(value: f32) -> f32 {
    if value > 0.0 && value < 1.0 {
        1.0
    } else {
        0.0
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_ai::halfkp_training::{search_teacher_manifest_path, SearchTeacherManifest};
    use shogi_ai::position_hash::PositionHasher;
    use shogi_lib::Position;

    #[test]
    fn rejects_manifest_with_incompatible_teacher_semantics() {
        let path = std::env::temp_dir().join(format!(
            "halfkp-search-train-semantics-{}.hkst",
            std::process::id()
        ));
        let manifest_path = search_teacher_manifest_path(&path);
        let manifest = SearchTeacherManifest {
            schema_version: 1,
            format: "HKST0002".to_string(),
            teacher_semantics_version: SEARCH_TEACHER_SEMANTICS_VERSION - 1,
            teacher_semantics_id: "legacy-truncated-history".to_string(),
            records: 0,
        };
        std::fs::write(
            &manifest_path,
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");

        let error = validate_teacher_semantics(&path, false).expect_err("reject old semantics");
        assert!(error.to_string().contains("unsupported teacher semantics"));
        std::fs::remove_file(manifest_path).expect("remove manifest");
    }

    #[test]
    fn legacy_teacher_requires_explicit_opt_in() {
        let path = Path::new("legacy-without-manifest.hkst");
        let error = validate_teacher_semantics(path, false).expect_err("reject implicit legacy");
        assert!(error
            .to_string()
            .contains("--allow-legacy-teacher-semantics"));
        validate_teacher_semantics(path, true).expect("explicit legacy opt-in");
    }

    #[test]
    fn value_gradient_matches_finite_difference() {
        let position = Position::default();
        let packed = PackedHalfKpPosition::from_position(&position).expect("start position");
        let record = SearchTeacherRecord {
            position_hash: PositionHasher::calculate_hash(&position),
            ply: 0,
            phase: 0,
            result: None,
            root_search_score_cp: 120.0,
            sample_weight: 0.75,
            teacher_depth: 4,
            root: packed.clone(),
            candidates: vec![
                shogi_ai::halfkp_training::SearchTeacherCandidate {
                    flags: 0,
                    score_cp: 20.0,
                    child: packed.clone(),
                },
                shogi_ai::halfkp_training::SearchTeacherCandidate {
                    flags: 0,
                    score_cp: 0.0,
                    child: packed,
                },
            ],
        };
        let options = TrainOptions {
            kappa_cp: 600.0,
            search_mix: 1.0,
            loss_power: 2.0,
            rank_weight: 0.0,
            game_rank_weight: 0.0,
            rank_margin_cp: 20.0,
            rank_temperature_cp: 100.0,
            min_rank_gap_cp: 15.0,
            game_regret_cap_cp: 150.0,
            max_pairs_per_record: 4,
        };
        let mut weights = Weights {
            feature_emb: vec![0.0; HALFKP_INPUTS * HALFKP_HIDDEN],
            hidden_b: [0.5; HALFKP_HIDDEN],
            out_w: [0.0; HALFKP_HIDDEN * 2 + 1],
            out_b: 0.03,
        };
        let analytic = compute_record_gradient(&weights, &record, &options)
            .dense
            .out_b;
        let epsilon = 1e-4;
        weights.out_b += epsilon;
        let plus = compute_record_gradient(&weights, &record, &options).value_loss;
        weights.out_b -= 2.0 * epsilon;
        let minus = compute_record_gradient(&weights, &record, &options).value_loss;
        let numerical = ((plus - minus) / (2.0 * epsilon as f64)) as f32;
        assert!(
            (analytic - numerical).abs() < 1e-4,
            "analytic={analytic} numerical={numerical}"
        );
    }
}
