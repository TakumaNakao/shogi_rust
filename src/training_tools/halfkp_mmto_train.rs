use super::halfkp_search_train::{
    project_center_mirror_sparse_gradient, Forward, Weights, TARGET_SCALE,
};
use super::scratch_lineage::{require_accepted_parent, require_scratch_model_manifest};
use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use shogi_ai::ai::{QsearchLeafKind, ShogiAI};
use shogi_ai::evaluation::{EngineEvaluator, Evaluator, HALFKP_HIDDEN};
use shogi_ai::halfkp_training::PackedHalfKpPosition;
use shogi_ai::training_data::{artifact_metadata, capture_run_environment, sha256_file};
use shogi_ai::utils::{parse_usi_move_for_color, position_from_sfen_or_usi};
use shogi_core::{Color, Move};
use shogi_lib::Position;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const OPTIMIZER_MAGIC: &[u8; 8] = b"MMTOAD01";
const CHECKPOINT_SCHEMA: u32 = 1;
const DELTA_HIST_BINS: usize = 81;

#[derive(Parser, Debug)]
#[command(about = "Train scratch HalfKP with refreshed forced-qsearch MMTO leaves")]
struct Args {
    #[arg(long, required = true)]
    train: Vec<PathBuf>,
    #[arg(long, required = true)]
    valid: Vec<PathBuf>,
    #[arg(long)]
    dataset_manifest: PathBuf,
    #[arg(long)]
    expected_dataset_manifest_sha256: String,
    #[arg(long)]
    init: PathBuf,
    #[arg(long)]
    model_manifest: PathBuf,
    #[arg(long)]
    promotion_registry: PathBuf,
    #[arg(long)]
    gate_report: PathBuf,
    #[arg(long)]
    parent_generation: u32,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 2)]
    outer_iterations: usize,
    #[arg(long, default_value_t = 1024)]
    search_block_roots: usize,
    #[arg(long, default_value_t = 1)]
    jobs: usize,
    #[arg(long, default_value_t = 8192)]
    pair_batch_size: usize,
    #[arg(long, default_value_t = 0.00001)]
    learning_rate: f32,
    #[arg(long, default_value_t = 0.1)]
    output_learning_rate_ratio: f32,
    #[arg(long, default_value_t = 1.0)]
    gradient_clip_norm: f32,
    #[arg(long, default_value_t = 36.63)]
    tau_cp: f32,
    /// Verification hook. Production MMTO uses exactly 1.0.
    #[arg(long, default_value_t = 1.0)]
    loss_weight: f32,
    #[arg(long, default_value_t = 1.0)]
    feature_limit: f32,
    #[arg(long, default_value_t = 0.20)]
    output_limit: f32,
    #[arg(long, default_value_t = 0.02)]
    max_feature_delta_per_outer: f32,
    #[arg(long, default_value_t = 0.004)]
    max_output_delta_per_outer: f32,
    #[arg(long)]
    max_train_roots: Option<usize>,
    #[arg(long, default_value_t = 10_000)]
    max_valid_roots: usize,
    #[arg(long, default_value_t = 1024)]
    pv_probe_roots: usize,
    #[arg(long, default_value_t = false)]
    resume: bool,
}

#[derive(Deserialize)]
struct InputRecord {
    schema: String,
    game_key: String,
    opening_family: String,
    split: String,
    side_to_move: String,
    sfen: String,
    teacher_move: String,
    license_id: String,
    source_uri: String,
}

#[derive(Deserialize)]
struct DatasetManifest {
    schema: String,
    min_player_rate: Option<i64>,
    min_opponent_rate: Option<i64>,
    output_files: Vec<DatasetOutput>,
}

#[derive(Deserialize)]
struct DatasetOutput {
    path: PathBuf,
    sha256: String,
}

struct Sample {
    root: Position,
    teacher: Move,
}

struct Leaf {
    packed: Option<PackedHalfKpPosition>,
    root_sign: f32,
    fixed_score_cp: f32,
    kind: QsearchLeafKind,
}

struct Pair {
    desired: Leaf,
    alternative: Leaf,
}

struct Gradient {
    hidden_b: [f32; HALFKP_HIDDEN],
    out_w: [f32; HALFKP_HIDDEN * 2 + 1],
    out_b: f32,
    sparse: HashMap<usize, [f32; HALFKP_HIDDEN]>,
    pairs: usize,
    loss_sum: f64,
    delta_hist: [u64; DELTA_HIST_BINS],
    active_derivative_pairs: u64,
}

impl Default for Gradient {
    fn default() -> Self {
        Self {
            hidden_b: [0.0; HALFKP_HIDDEN],
            out_w: [0.0; HALFKP_HIDDEN * 2 + 1],
            out_b: 0.0,
            sparse: HashMap::new(),
            pairs: 0,
            loss_sum: 0.0,
            delta_hist: [0; DELTA_HIST_BINS],
            active_derivative_pairs: 0,
        }
    }
}

#[derive(Serialize, Clone)]
struct UpdateDiagnostics {
    #[serde(skip)]
    delta_hist: [u64; DELTA_HIST_BINS],
    pairs: u64,
    delta_over_tau_p50: f64,
    delta_over_tau_p90: f64,
    delta_over_tau_p99: f64,
    active_derivative_fraction: f64,
    update_batches: u64,
    clipped_batches: u64,
}

impl Default for UpdateDiagnostics {
    fn default() -> Self {
        Self {
            delta_hist: [0; DELTA_HIST_BINS],
            pairs: 0,
            delta_over_tau_p50: 0.0,
            delta_over_tau_p90: 0.0,
            delta_over_tau_p99: 0.0,
            active_derivative_fraction: 0.0,
            update_batches: 0,
            clipped_batches: 0,
        }
    }
}

#[derive(Serialize)]
struct OuterDelta {
    feature_max_abs: f32,
    hidden_max_abs: f32,
    output_max_abs: f32,
}

#[derive(Serialize)]
struct OuterMetricsFile {
    outer: usize,
    train: MetricReport,
    valid: MetricReport,
    train_diagnostics: UpdateDiagnostics,
    outer_delta: OuterDelta,
    pv_probe_best_move_unchanged: f64,
    elapsed_ms: u128,
    model_sha256: String,
    optimizer_sha256: String,
}

#[derive(Default, Serialize, Clone)]
struct SearchMetrics {
    roots: usize,
    pairs: usize,
    exact_leaves: usize,
    terminal_leaves: usize,
    rejected_bounds: usize,
    illegal_records: usize,
    loss_sum: f64,
    agreement: usize,
    regret_sum_cp: f64,
    qnodes: u64,
    #[serde(skip)]
    best_moves: Vec<Move>,
}

impl SearchMetrics {
    fn merge(&mut self, other: &Self) {
        self.roots += other.roots;
        self.pairs += other.pairs;
        self.exact_leaves += other.exact_leaves;
        self.terminal_leaves += other.terminal_leaves;
        self.rejected_bounds += other.rejected_bounds;
        self.illegal_records += other.illegal_records;
        self.loss_sum += other.loss_sum;
        self.agreement += other.agreement;
        self.regret_sum_cp += other.regret_sum_cp;
        self.qnodes += other.qnodes;
        self.best_moves.extend_from_slice(&other.best_moves);
    }
}

#[derive(Serialize)]
struct MetricReport {
    roots: usize,
    pairs: usize,
    mean_loss_per_pair: f64,
    agreement: f64,
    mean_regret_cp: f64,
    exact_leaves: usize,
    terminal_leaves: usize,
    rejected_bounds: usize,
    qnodes: u64,
}

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    schema_version: u32,
    completed_outer: usize,
    init_model_sha256: String,
    dataset_manifest_sha256: String,
    material_bits: u32,
    previous_model_sha256: String,
    model_sha256: String,
    optimizer_sha256: String,
}

#[derive(Serialize)]
struct RunManifest {
    schema_version: u32,
    stage: &'static str,
    command: Vec<String>,
    environment: shogi_ai::training_data::RunEnvironment,
    completed_outer: usize,
    dataset_manifest_sha256: String,
    init_model: shogi_ai::training_data::ArtifactMetadata,
    output_model: shogi_ai::training_data::ArtifactMetadata,
    optimizer: shogi_ai::training_data::ArtifactMetadata,
    hyperparameters: BTreeMap<&'static str, serde_json::Value>,
    last_train: MetricReport,
    last_valid_fresh: MetricReport,
    pv_probe_best_move_unchanged: f64,
    train_diagnostics: UpdateDiagnostics,
    outer_delta: OuterDelta,
    elapsed_ms: u128,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    validate_lineage_and_dataset(&args)?;
    fs::create_dir_all(&args.output_dir)?;
    let model_path = args.output_dir.join("model.bin");
    let optimizer_path = args.output_dir.join("optimizer.bin");
    let checkpoint_path = args.output_dir.join("checkpoint.json");
    let manifest_sha = sha256_file(&args.dataset_manifest)?;
    let init_sha = sha256_file(&args.init)?;
    let search_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.jobs)
        .build()
        .context("build MMTO search pool")?;

    let (mut weights, mut adagrad, start_outer, material_bits, mut previous_sha) = if args.resume {
        let checkpoint: Checkpoint =
            serde_json::from_reader(BufReader::new(File::open(&checkpoint_path)?))?;
        if checkpoint.schema_version != CHECKPOINT_SCHEMA
            || checkpoint.init_model_sha256 != init_sha
            || checkpoint.dataset_manifest_sha256 != manifest_sha
        {
            return Err(anyhow!("resume checkpoint provenance mismatch"));
        }
        restore_checkpoint_artifacts_if_needed(
            &model_path,
            &optimizer_path,
            &checkpoint.model_sha256,
            &checkpoint.optimizer_sha256,
        )?;
        let weights = Weights::load(&model_path)?;
        let adagrad = load_optimizer(&optimizer_path, &weights)?;
        (
            weights,
            adagrad,
            checkpoint.completed_outer,
            checkpoint.material_bits,
            checkpoint.model_sha256,
        )
    } else {
        if model_path.exists() || optimizer_path.exists() || checkpoint_path.exists() {
            return Err(anyhow!(
                "output checkpoint exists; pass --resume or use an empty output directory"
            ));
        }
        let weights = Weights::load(&args.init)?;
        let material_bits = weights.out_w[HALFKP_HIDDEN * 2].to_bits();
        let adagrad = weights.zeros_like();
        weights.save(&model_path)?;
        save_optimizer(&optimizer_path, &adagrad)?;
        let checkpoint = Checkpoint {
            schema_version: CHECKPOINT_SCHEMA,
            completed_outer: 0,
            init_model_sha256: init_sha.clone(),
            dataset_manifest_sha256: manifest_sha.clone(),
            material_bits,
            previous_model_sha256: init_sha.clone(),
            model_sha256: sha256_file(&model_path)?,
            optimizer_sha256: sha256_file(&optimizer_path)?,
        };
        atomic_json(&checkpoint_path, &checkpoint)?;
        (weights, adagrad, 0, material_bits, init_sha.clone())
    };

    for outer in start_outer + 1..=args.outer_iterations {
        let outer_started = Instant::now();
        let before = weights.clone();
        weights.save(&model_path)?;
        // EngineEvaluator owns this immutable search-side copy while `weights`
        // is updated independently throughout the streaming pass.
        let search_snapshot = Arc::new(EngineEvaluator::new(&model_path, 0.0)?);
        let valid_before = fresh_evaluate(
            &model_path,
            &args.valid,
            "valid",
            Some(args.pv_probe_roots.min(args.max_valid_roots)),
            args.tau_cp,
            &search_pool,
            args.search_block_roots,
        )?;
        let mut gradient = Gradient::default();
        let mut diagnostics = UpdateDiagnostics::default();
        let mut last_train = SearchMetrics::default();
        let mut block = Vec::with_capacity(args.search_block_roots);
        visit_samples(&args.train, "train", args.max_train_roots, |sample| {
            block.push(sample);
            if block.len() == args.search_block_roots {
                train_block(
                    &mut block,
                    &search_pool,
                    &search_snapshot,
                    &mut weights,
                    &mut adagrad,
                    &mut gradient,
                    &mut last_train,
                    &mut diagnostics,
                    &args,
                    material_bits,
                )?;
            }
            Ok(())
        })?;
        train_block(
            &mut block,
            &search_pool,
            &search_snapshot,
            &mut weights,
            &mut adagrad,
            &mut gradient,
            &mut last_train,
            &mut diagnostics,
            &args,
            material_bits,
        )?;
        if gradient.pairs != 0 {
            apply_batch(
                &mut weights,
                &mut adagrad,
                &mut gradient,
                &args,
                material_bits,
                &mut diagnostics,
            )?;
        }
        let outer_delta = validate_outer_delta(&before, &weights, &args, material_bits)?;
        atomic_copy(&model_path, &model_path.with_extension("previous"))?;
        atomic_copy(&optimizer_path, &optimizer_path.with_extension("previous"))?;
        weights.save(&model_path)?;
        save_optimizer(&optimizer_path, &adagrad)?;
        let last_valid = fresh_evaluate(
            &model_path,
            &args.valid,
            "valid",
            Some(args.max_valid_roots),
            args.tau_cp,
            &search_pool,
            args.search_block_roots,
        )?;
        let pv_unchanged = best_move_unchanged(&valid_before, &last_valid)?;
        finalize_diagnostics(&mut diagnostics);
        let model_sha = sha256_file(&model_path)?;
        let optimizer_sha = sha256_file(&optimizer_path)?;
        let checkpoint = Checkpoint {
            schema_version: CHECKPOINT_SCHEMA,
            completed_outer: outer,
            init_model_sha256: init_sha.clone(),
            dataset_manifest_sha256: manifest_sha.clone(),
            material_bits,
            previous_model_sha256: previous_sha.clone(),
            model_sha256: model_sha.clone(),
            optimizer_sha256: optimizer_sha,
        };
        atomic_json(&checkpoint_path, &checkpoint)?;
        previous_sha = model_sha.clone();
        let metrics_dir = args.output_dir.join("metrics");
        fs::create_dir_all(&metrics_dir)?;
        let metrics_file = OuterMetricsFile {
            outer,
            train: report(&last_train),
            valid: report(&last_valid),
            train_diagnostics: diagnostics.clone(),
            outer_delta: OuterDelta {
                feature_max_abs: outer_delta.feature_max_abs,
                hidden_max_abs: outer_delta.hidden_max_abs,
                output_max_abs: outer_delta.output_max_abs,
            },
            pv_probe_best_move_unchanged: pv_unchanged,
            elapsed_ms: outer_started.elapsed().as_millis(),
            model_sha256: sha256_file(&model_path)?,
            optimizer_sha256: sha256_file(&optimizer_path)?,
        };
        atomic_json(
            &metrics_dir.join(format!("outer_{outer:04}.json")),
            &metrics_file,
        )?;
        write_run_manifest(
            &args,
            outer,
            &manifest_sha,
            &model_path,
            &optimizer_path,
            &last_train,
            &last_valid,
            pv_unchanged,
            diagnostics,
            outer_delta,
            outer_started.elapsed().as_millis(),
        )?;
        println!(
            "outer={outer} train_loss={:.9} valid_fresh_loss={:.9} valid_agreement={:.6}",
            report(&last_train).mean_loss_per_pair,
            report(&last_valid).mean_loss_per_pair,
            report(&last_valid).agreement
        );
    }
    let _ = previous_sha;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn train_block(
    block: &mut Vec<Sample>,
    search_pool: &rayon::ThreadPool,
    search_snapshot: &Arc<EngineEvaluator>,
    weights: &mut Weights,
    adagrad: &mut Weights,
    gradient: &mut Gradient,
    metrics: &mut SearchMetrics,
    diagnostics: &mut UpdateDiagnostics,
    args: &Args,
    material_bits: u32,
) -> Result<()> {
    let searched = search_pool.install(|| {
        block
            .par_iter()
            .map(|sample| search_sample(Arc::clone(search_snapshot), sample, args.tau_cp))
            .collect::<Result<Vec<_>>>()
    })?;
    block.clear();
    for (pairs, sample_metrics) in searched {
        metrics.merge(&sample_metrics);
        for pair in pairs {
            add_pair_gradient(weights, &pair, args.tau_cp, args.loss_weight, gradient)?;
            if gradient.pairs == args.pair_batch_size {
                apply_batch(weights, adagrad, gradient, args, material_bits, diagnostics)?;
            }
        }
    }
    Ok(())
}

fn validate_args(args: &Args) -> Result<()> {
    if args.outer_iterations == 0
        || args.search_block_roots == 0
        || args.jobs == 0
        || args.pair_batch_size == 0
        || args.learning_rate <= 0.0
        || args.output_learning_rate_ratio <= 0.0
        || args.gradient_clip_norm <= 0.0
        || args.tau_cp <= 0.0
        || args.loss_weight < 0.0
        || !args.loss_weight.is_finite()
        || args.feature_limit <= 0.0
        || args.output_limit <= 0.0
        || args.max_valid_roots == 0
        || args.pv_probe_roots == 0
    {
        return Err(anyhow!("invalid MMTO hyperparameter"));
    }
    Ok(())
}

fn validate_lineage_and_dataset(args: &Args) -> Result<()> {
    let accepted = require_accepted_parent(
        &args.promotion_registry,
        &args.init,
        &args.gate_report,
        args.parent_generation,
    )?;
    let manifest_parent =
        require_scratch_model_manifest(&args.model_manifest, &args.init, args.parent_generation)?;
    if accepted.model_sha256 != manifest_parent {
        return Err(anyhow!("accepted parent and model manifest disagree"));
    }
    let actual = sha256_file(&args.dataset_manifest)?;
    if actual != args.expected_dataset_manifest_sha256 {
        return Err(anyhow!(
            "dataset manifest SHA mismatch: expected {}, got {}",
            args.expected_dataset_manifest_sha256,
            actual
        ));
    }
    let manifest: DatasetManifest =
        serde_json::from_reader(BufReader::new(File::open(&args.dataset_manifest)?))?;
    if manifest.schema != "rank_value_dataset_v4_manifest"
        || manifest.min_player_rate != Some(4000)
        || manifest.min_opponent_rate != Some(4000)
    {
        return Err(anyhow!(
            "dataset manifest is not the frozen WDoor rate-4000 set"
        ));
    }
    validate_manifest_outputs(&args.dataset_manifest, &manifest, &args.train, &args.valid)?;
    validate_dataset_splits(&args.train, &args.valid)
}

fn validate_manifest_outputs(
    manifest_path: &Path,
    manifest: &DatasetManifest,
    train: &[PathBuf],
    valid: &[PathBuf],
) -> Result<()> {
    let base = manifest_path
        .parent()
        .ok_or_else(|| anyhow!("dataset manifest has no parent directory"))?;
    for (split, supplied) in [("train.jsonl", train), ("valid.jsonl", valid)] {
        let output = manifest
            .output_files
            .iter()
            .find(|output| output.path == Path::new(split))
            .ok_or_else(|| anyhow!("dataset manifest lacks {split}"))?;
        let expected = fs::canonicalize(base.join(&output.path))?;
        if supplied.len() != 1 || fs::canonicalize(&supplied[0])? != expected {
            return Err(anyhow!("{split} path does not match the frozen manifest"));
        }
        if sha256_file(&expected)? != output.sha256 {
            return Err(anyhow!(
                "{split} SHA-256 does not match the frozen manifest"
            ));
        }
    }
    Ok(())
}

fn validate_dataset_splits(train: &[PathBuf], valid: &[PathBuf]) -> Result<()> {
    let mut train_games = BTreeSet::new();
    let mut valid_games = BTreeSet::new();
    for (paths, expected, games) in [
        (train, "train", &mut train_games),
        (valid, "valid", &mut valid_games),
    ] {
        for path in paths {
            for line in BufReader::new(File::open(path)?).lines() {
                let record: InputRecord = serde_json::from_str(&line?)?;
                validate_record(&record, expected)?;
                games.insert(record.game_key);
            }
        }
    }
    let overlap = train_games.intersection(&valid_games).count();
    if overlap != 0 {
        return Err(anyhow!("train/valid game overlap: {overlap}"));
    }
    Ok(())
}

fn validate_record(record: &InputRecord, expected_split: &str) -> Result<()> {
    if record.schema != "rank_value_dataset_v3_position"
        || record.split != expected_split
        || record.game_key.is_empty()
        || record.opening_family.is_empty()
        || record.license_id.is_empty()
        || record.source_uri.is_empty()
    {
        return Err(anyhow!("invalid WDoor record provenance"));
    }
    Ok(())
}

fn visit_samples(
    paths: &[PathBuf],
    expected_split: &str,
    limit: Option<usize>,
    mut visitor: impl FnMut(Sample) -> Result<()>,
) -> Result<usize> {
    let mut count = 0usize;
    'files: for path in paths {
        for (line_no, line) in BufReader::new(File::open(path)?).lines().enumerate() {
            if limit.is_some_and(|limit| count >= limit) {
                break 'files;
            }
            let record: InputRecord = serde_json::from_str(&line?)
                .with_context(|| format!("parse {}:{}", path.display(), line_no + 1))?;
            validate_record(&record, expected_split)?;
            let position = position_from_sfen_or_usi(&record.sfen)
                .ok_or_else(|| anyhow!("invalid SFEN at {}:{}", path.display(), line_no + 1))?;
            let expected_side = if position.side_to_move() == Color::Black {
                "black"
            } else {
                "white"
            };
            if record.side_to_move != expected_side {
                return Err(anyhow!("side-to-move mismatch"));
            }
            let legal = position.legal_moves();
            let teacher = parse_usi_move_for_color(&record.teacher_move, position.side_to_move())
                .filter(|mv| legal.contains(mv))
                .ok_or_else(|| anyhow!("illegal teacher move"))?;
            if legal.len() < 2 {
                continue;
            }
            visitor(Sample {
                root: position,
                teacher,
            })?;
            count += 1;
        }
    }
    Ok(count)
}

fn search_sample(
    evaluator: Arc<EngineEvaluator>,
    sample: &Sample,
    tau_cp: f32,
) -> Result<(Vec<Pair>, SearchMetrics)> {
    let mut ai = ShogiAI::new(evaluator);
    ai.set_emit_info(false);
    let mut metrics = SearchMetrics::default();
    let pairs = search_pairs(&mut ai, sample, tau_cp, &mut metrics)?;
    Ok((pairs, metrics))
}

fn search_pairs<E: Evaluator>(
    ai: &mut ShogiAI<E, 1024>,
    sample: &Sample,
    tau_cp: f32,
    metrics: &mut SearchMetrics,
) -> Result<Vec<Pair>> {
    ai.reset_for_independent_search();
    ai.sennichite_detector.record_initial_position(&sample.root);
    let root_turn = sample.root.side_to_move();
    let mut root = sample.root.clone();
    let mut searched = Vec::new();
    for mv in root.legal_moves() {
        let trace = ai
            .forced_move_qsearch_leaf_trace(&mut root, mv)
            .ok_or_else(|| anyhow!("forced qsearch aborted"))?;
        let leaf = if trace.has_exact_evaluation_leaf() {
            let position = trace.leaf.ok_or_else(|| anyhow!("missing exact leaf"))?;
            let sign = if position.side_to_move() == root_turn {
                1.0
            } else {
                -1.0
            };
            metrics.exact_leaves += 1;
            Leaf {
                packed: Some(
                    PackedHalfKpPosition::from_position(&position)
                        .ok_or_else(|| anyhow!("qsearch leaf lacks king"))?,
                ),
                root_sign: sign,
                fixed_score_cp: trace.score,
                kind: trace.kind,
            }
        } else if matches!(
            trace.kind,
            QsearchLeafKind::Mate | QsearchLeafKind::Repetition
        ) {
            metrics.terminal_leaves += 1;
            Leaf {
                packed: None,
                root_sign: 0.0,
                fixed_score_cp: trace.score,
                kind: trace.kind,
            }
        } else {
            metrics.rejected_bounds += 1;
            return Err(anyhow!("root full-window qsearch returned a bound"));
        };
        searched.push((mv, leaf));
    }
    let teacher_index = searched
        .iter()
        .position(|(mv, _)| *mv == sample.teacher)
        .ok_or_else(|| anyhow!("teacher missing after search"))?;
    let teacher_score = searched[teacher_index].1.fixed_score_cp;
    let best_score = searched
        .iter()
        .map(|(_, leaf)| leaf.fixed_score_cp)
        .fold(f32::NEG_INFINITY, f32::max);
    let best_move = searched
        .iter()
        .max_by(|lhs, rhs| lhs.1.fixed_score_cp.total_cmp(&rhs.1.fixed_score_cp))
        .map(|(mv, _)| *mv)
        .expect("at least two legal moves");
    metrics.roots += 1;
    metrics.agreement += usize::from(teacher_score.to_bits() == best_score.to_bits());
    metrics.regret_sum_cp += f64::from(best_score - teacher_score);
    metrics.qnodes += ai.quiescence_nodes_searched();
    metrics.best_moves.push(best_move);

    let (_, desired) = searched.swap_remove(teacher_index);
    let mut desired = Some(desired);
    let mut pairs = Vec::with_capacity(searched.len());
    for (_, alternative) in searched {
        let desired_leaf = clone_leaf(desired.as_ref().expect("desired"));
        let delta = desired_leaf.fixed_score_cp - alternative.fixed_score_cp;
        metrics.loss_sum += f64::from(mirrored_sigmoid(delta, tau_cp));
        metrics.pairs += 1;
        pairs.push(Pair {
            desired: desired_leaf,
            alternative,
        });
    }
    let _ = desired.take();
    Ok(pairs)
}

fn clone_leaf(leaf: &Leaf) -> Leaf {
    Leaf {
        packed: leaf.packed.clone(),
        root_sign: leaf.root_sign,
        fixed_score_cp: leaf.fixed_score_cp,
        kind: leaf.kind,
    }
}

fn mirrored_sigmoid(delta_cp: f32, tau_cp: f32) -> f32 {
    let x = (delta_cp / tau_cp).clamp(-80.0, 80.0);
    1.0 / (1.0 + x.exp())
}

fn current_leaf_score(weights: &Weights, leaf: &Leaf) -> (f32, Option<Forward>) {
    match leaf.packed.as_ref() {
        Some(position) => {
            let forward = weights.forward(position);
            (leaf.root_sign * forward.raw * TARGET_SCALE, Some(forward))
        }
        None => (leaf.fixed_score_cp, None),
    }
}

fn add_pair_gradient(
    weights: &Weights,
    pair: &Pair,
    tau_cp: f32,
    loss_weight: f32,
    gradient: &mut Gradient,
) -> Result<()> {
    let (desired_score, desired_forward) = current_leaf_score(weights, &pair.desired);
    let (alternative_score, alternative_forward) = current_leaf_score(weights, &pair.alternative);
    let loss = mirrored_sigmoid(desired_score - alternative_score, tau_cp);
    let ratio = ((desired_score - alternative_score).abs() / tau_cp).min(80.0);
    gradient.delta_hist[ratio.floor() as usize] += 1;
    if loss * (1.0 - loss) >= 0.0025 {
        gradient.active_derivative_pairs += 1;
    }
    let d_delta_cp = -loss_weight * loss * (1.0 - loss) / tau_cp;
    if !loss.is_finite() || !d_delta_cp.is_finite() {
        return Err(anyhow!("non-finite MMTO loss or derivative"));
    }
    if let (Some(position), Some(forward)) =
        (pair.desired.packed.as_ref(), desired_forward.as_ref())
    {
        add_position_gradient(
            weights,
            position,
            forward,
            d_delta_cp * pair.desired.root_sign * TARGET_SCALE,
            gradient,
        );
    }
    if let (Some(position), Some(forward)) = (
        pair.alternative.packed.as_ref(),
        alternative_forward.as_ref(),
    ) {
        add_position_gradient(
            weights,
            position,
            forward,
            -d_delta_cp * pair.alternative.root_sign * TARGET_SCALE,
            gradient,
        );
    }
    gradient.pairs += 1;
    gradient.loss_sum += f64::from(loss);
    Ok(())
}

fn add_position_gradient(
    weights: &Weights,
    position: &PackedHalfKpPosition,
    forward: &Forward,
    d_raw: f32,
    gradient: &mut Gradient,
) {
    let (stm, nstm) = if position.side_to_move == Color::Black {
        (&forward.black, &forward.white)
    } else {
        (&forward.white, &forward.black)
    };
    let mut stm_hidden = [0.0; HALFKP_HIDDEN];
    let mut nstm_hidden = [0.0; HALFKP_HIDDEN];
    for h in 0..HALFKP_HIDDEN {
        gradient.out_w[h] += d_raw * stm[h].clamp(0.0, 1.0);
        gradient.out_w[HALFKP_HIDDEN + h] += d_raw * nstm[h].clamp(0.0, 1.0);
        stm_hidden[h] = d_raw * weights.out_w[h] * active_derivative(stm[h]);
        nstm_hidden[h] = d_raw * weights.out_w[HALFKP_HIDDEN + h] * active_derivative(nstm[h]);
    }
    gradient.out_b += d_raw;
    let (black, white) = if position.side_to_move == Color::Black {
        (stm_hidden, nstm_hidden)
    } else {
        (nstm_hidden, stm_hidden)
    };
    for h in 0..HALFKP_HIDDEN {
        gradient.hidden_b[h] += black[h] + white[h];
    }
    for (&feature, hidden) in position
        .features_black
        .iter()
        .map(|feature| (feature, &black))
        .chain(
            position
                .features_white
                .iter()
                .map(|feature| (feature, &white)),
        )
    {
        let row = gradient
            .sparse
            .entry(feature as usize)
            .or_insert([0.0; HALFKP_HIDDEN]);
        for h in 0..HALFKP_HIDDEN {
            row[h] += hidden[h];
        }
    }
}

fn active_derivative(value: f32) -> f32 {
    f32::from(value > 0.0 && value < 1.0)
}

fn apply_batch(
    weights: &mut Weights,
    adagrad: &mut Weights,
    gradient: &mut Gradient,
    args: &Args,
    material_bits: u32,
    diagnostics: &mut UpdateDiagnostics,
) -> Result<()> {
    diagnostics.pairs += gradient.pairs as u64;
    diagnostics.update_batches += 1;
    for (dst, src) in diagnostics.delta_hist.iter_mut().zip(gradient.delta_hist) {
        *dst += src;
    }
    diagnostics.active_derivative_fraction += gradient.active_derivative_pairs as f64;
    project_center_mirror_sparse_gradient(&mut gradient.sparse);
    let scale = 1.0 / gradient.pairs.max(1) as f32;
    gradient.out_w[HALFKP_HIDDEN * 2] = 0.0;
    let mut norm_squared = (gradient.out_b * scale).powi(2);
    gradient.out_b *= scale;
    for value in gradient.hidden_b.iter_mut().chain(&mut gradient.out_w) {
        *value *= scale;
        norm_squared += *value * *value;
    }
    let mut features = gradient.sparse.keys().copied().collect::<Vec<_>>();
    features.sort_unstable();
    for feature in &features {
        for value in gradient.sparse.get_mut(feature).expect("feature") {
            *value *= scale;
            norm_squared += *value * *value;
        }
    }
    if !norm_squared.is_finite() {
        return Err(anyhow!("non-finite gradient norm"));
    }
    let clip = (args.gradient_clip_norm / norm_squared.sqrt()).min(1.0);
    if clip < 1.0 {
        diagnostics.clipped_batches += 1;
    }
    let mut touched_parameter_bound = false;
    for h in 0..HALFKP_HIDDEN {
        touched_parameter_bound |= adagrad_scalar(
            &mut weights.hidden_b[h],
            &mut adagrad.hidden_b[h],
            gradient.hidden_b[h] * clip,
            args.learning_rate,
            args.feature_limit,
        );
    }
    for index in 0..weights.out_w.len() {
        if index == HALFKP_HIDDEN * 2 {
            continue;
        }
        touched_parameter_bound |= adagrad_scalar(
            &mut weights.out_w[index],
            &mut adagrad.out_w[index],
            gradient.out_w[index] * clip,
            args.learning_rate * args.output_learning_rate_ratio,
            args.output_limit,
        );
    }
    touched_parameter_bound |= adagrad_scalar(
        &mut weights.out_b,
        &mut adagrad.out_b,
        gradient.out_b * clip,
        args.learning_rate * args.output_learning_rate_ratio,
        args.output_limit,
    );
    for feature in features {
        let start = feature * HALFKP_HIDDEN;
        for h in 0..HALFKP_HIDDEN {
            touched_parameter_bound |= adagrad_scalar(
                &mut weights.feature_emb[start + h],
                &mut adagrad.feature_emb[start + h],
                gradient.sparse[&feature][h] * clip,
                args.learning_rate,
                args.feature_limit,
            );
        }
    }
    if weights.out_w[HALFKP_HIDDEN * 2].to_bits() != material_bits {
        return Err(anyhow!("material coefficient changed"));
    }
    if touched_parameter_bound {
        return Err(anyhow!(
            "parameter bound touched; refusing to continue with a projected model"
        ));
    }
    *gradient = Gradient::default();
    Ok(())
}

fn adagrad_scalar(
    weight: &mut f32,
    accumulator: &mut f32,
    gradient: f32,
    lr: f32,
    limit: f32,
) -> bool {
    if gradient == 0.0 {
        return false;
    }
    *accumulator += gradient * gradient;
    let updated = *weight - lr * gradient / (accumulator.sqrt() + 1e-8);
    let touched = updated.abs() >= limit;
    *weight = updated.clamp(-limit, limit);
    touched
}

fn validate_outer_delta(
    before: &Weights,
    after: &Weights,
    args: &Args,
    material_bits: u32,
) -> Result<OuterDelta> {
    let feature = before
        .feature_emb
        .iter()
        .zip(&after.feature_emb)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let hidden = before
        .hidden_b
        .iter()
        .zip(&after.hidden_b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let output = before
        .out_w
        .iter()
        .zip(&after.out_w)
        .map(|(a, b)| (a - b).abs())
        .chain(std::iter::once((before.out_b - after.out_b).abs()))
        .fold(0.0f32, f32::max);
    if feature > args.max_feature_delta_per_outer
        || hidden > args.max_feature_delta_per_outer
        || output > args.max_output_delta_per_outer
    {
        return Err(anyhow!(
            "outer delta limit exceeded: feature={feature} hidden={hidden} output={output}"
        ));
    }
    if after.out_w[HALFKP_HIDDEN * 2].to_bits() != material_bits {
        return Err(anyhow!("material coefficient changed"));
    }
    Ok(OuterDelta {
        feature_max_abs: feature,
        hidden_max_abs: hidden,
        output_max_abs: output,
    })
}

fn fresh_evaluate(
    model_path: &Path,
    paths: &[PathBuf],
    split: &str,
    limit: Option<usize>,
    tau_cp: f32,
    search_pool: &rayon::ThreadPool,
    block_roots: usize,
) -> Result<SearchMetrics> {
    let evaluator = Arc::new(EngineEvaluator::new(model_path, 0.0)?);
    let mut metrics = SearchMetrics::default();
    let mut block = Vec::with_capacity(block_roots);
    visit_samples(paths, split, limit, |sample| {
        block.push(sample);
        if block.len() == block_roots {
            evaluate_block(&mut block, search_pool, &evaluator, tau_cp, &mut metrics)?;
        }
        Ok(())
    })?;
    evaluate_block(&mut block, search_pool, &evaluator, tau_cp, &mut metrics)?;
    Ok(metrics)
}

fn evaluate_block(
    block: &mut Vec<Sample>,
    search_pool: &rayon::ThreadPool,
    evaluator: &Arc<EngineEvaluator>,
    tau_cp: f32,
    metrics: &mut SearchMetrics,
) -> Result<()> {
    let searched = search_pool.install(|| {
        block
            .par_iter()
            .map(|sample| search_sample(Arc::clone(evaluator), sample, tau_cp))
            .collect::<Result<Vec<_>>>()
    })?;
    block.clear();
    for (_, sample_metrics) in searched {
        metrics.merge(&sample_metrics);
    }
    Ok(())
}

fn best_move_unchanged(before: &SearchMetrics, after: &SearchMetrics) -> Result<f64> {
    if before.best_moves.is_empty() || after.best_moves.len() < before.best_moves.len() {
        return Err(anyhow!("incomplete fixed PV probe"));
    }
    let unchanged = before
        .best_moves
        .iter()
        .zip(&after.best_moves)
        .filter(|(before, after)| before == after)
        .count();
    Ok(unchanged as f64 / before.best_moves.len() as f64)
}

fn report(metrics: &SearchMetrics) -> MetricReport {
    MetricReport {
        roots: metrics.roots,
        pairs: metrics.pairs,
        mean_loss_per_pair: metrics.loss_sum / metrics.pairs.max(1) as f64,
        agreement: metrics.agreement as f64 / metrics.roots.max(1) as f64,
        mean_regret_cp: metrics.regret_sum_cp / metrics.roots.max(1) as f64,
        exact_leaves: metrics.exact_leaves,
        terminal_leaves: metrics.terminal_leaves,
        rejected_bounds: metrics.rejected_bounds,
        qnodes: metrics.qnodes,
    }
}

fn finalize_diagnostics(d: &mut UpdateDiagnostics) {
    let total = d.pairs.max(1);
    d.delta_over_tau_p50 = histogram_quantile(&d.delta_hist, total, 0.50);
    d.delta_over_tau_p90 = histogram_quantile(&d.delta_hist, total, 0.90);
    d.delta_over_tau_p99 = histogram_quantile(&d.delta_hist, total, 0.99);
    d.active_derivative_fraction /= total as f64;
}

fn histogram_quantile(hist: &[u64; DELTA_HIST_BINS], total: u64, q: f64) -> f64 {
    let target = ((total as f64) * q).ceil().max(1.0) as u64;
    let mut cumulative = 0;
    for (bin, count) in hist.iter().enumerate() {
        cumulative += *count;
        if cumulative >= target {
            return (bin + 1) as f64;
        }
    }
    DELTA_HIST_BINS as f64
}

fn save_optimizer(path: &Path, values: &Weights) -> Result<()> {
    let temporary = path.with_extension("tmp");
    let mut writer = BufWriter::new(File::create(&temporary)?);
    writer.write_all(OPTIMIZER_MAGIC)?;
    writer.write_all(&(values.feature_emb.len() as u64).to_le_bytes())?;
    write_f32s(&mut writer, &values.feature_emb)?;
    write_f32s(&mut writer, &values.hidden_b)?;
    write_f32s(&mut writer, &values.out_w)?;
    write_f32s(&mut writer, &[values.out_b])?;
    writer.flush()?;
    drop(writer);
    fs::rename(temporary, path)?;
    Ok(())
}

fn load_optimizer(path: &Path, shape: &Weights) -> Result<Weights> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != OPTIMIZER_MAGIC {
        return Err(anyhow!("invalid MMTO optimizer magic"));
    }
    let mut len = [0u8; 8];
    reader.read_exact(&mut len)?;
    if u64::from_le_bytes(len) as usize != shape.feature_emb.len() {
        return Err(anyhow!("MMTO optimizer shape mismatch"));
    }
    let mut values = shape.zeros_like();
    read_f32s(&mut reader, &mut values.feature_emb)?;
    read_f32s(&mut reader, &mut values.hidden_b)?;
    read_f32s(&mut reader, &mut values.out_w)?;
    read_f32s(&mut reader, std::slice::from_mut(&mut values.out_b))?;
    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        return Err(anyhow!("trailing MMTO optimizer bytes"));
    }
    Ok(values)
}

fn write_f32s(writer: &mut impl Write, values: &[f32]) -> Result<()> {
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn read_f32s(reader: &mut impl Read, values: &mut [f32]) -> Result<()> {
    let mut bytes = [0u8; 4];
    for value in values {
        reader.read_exact(&mut bytes)?;
        *value = f32::from_le_bytes(bytes);
        if !value.is_finite() || *value < 0.0 {
            return Err(anyhow!("invalid AdaGrad accumulator"));
        }
    }
    Ok(())
}

fn restore_checkpoint_artifacts_if_needed(
    model_path: &Path,
    optimizer_path: &Path,
    expected_model_sha256: &str,
    expected_optimizer_sha256: &str,
) -> Result<()> {
    if sha256_file(model_path).ok().as_deref() == Some(expected_model_sha256)
        && sha256_file(optimizer_path).ok().as_deref() == Some(expected_optimizer_sha256)
    {
        return Ok(());
    }
    let previous_model = model_path.with_extension("previous");
    let previous_optimizer = optimizer_path.with_extension("previous");
    if sha256_file(&previous_model).ok().as_deref() != Some(expected_model_sha256)
        || sha256_file(&previous_optimizer).ok().as_deref() != Some(expected_optimizer_sha256)
    {
        return Err(anyhow!("resume checkpoint artifact hash mismatch"));
    }
    atomic_copy(&previous_model, model_path)?;
    atomic_copy(&previous_optimizer, optimizer_path)?;
    Ok(())
}

fn atomic_copy(source: &Path, destination: &Path) -> Result<()> {
    let temporary = destination.with_extension("tmp");
    fs::copy(source, &temporary)?;
    fs::rename(temporary, destination)?;
    Ok(())
}

fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = path.with_extension("tmp");
    let mut writer = BufWriter::new(File::create(&temporary)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    drop(writer);
    fs::rename(temporary, path)?;
    Ok(())
}

fn write_run_manifest(
    args: &Args,
    completed_outer: usize,
    dataset_manifest_sha256: &str,
    model_path: &Path,
    optimizer_path: &Path,
    train: &SearchMetrics,
    valid: &SearchMetrics,
    pv_probe_best_move_unchanged: f64,
    train_diagnostics: UpdateDiagnostics,
    outer_delta: OuterDelta,
    elapsed_ms: u128,
) -> Result<()> {
    let mut hyperparameters = BTreeMap::new();
    hyperparameters.insert("tau_cp", serde_json::json!(args.tau_cp));
    hyperparameters.insert("loss_weight", serde_json::json!(args.loss_weight));
    hyperparameters.insert(
        "feature_learning_rate",
        serde_json::json!(args.learning_rate),
    );
    hyperparameters.insert(
        "output_learning_rate_ratio",
        serde_json::json!(args.output_learning_rate_ratio),
    );
    hyperparameters.insert("pair_batch_size", serde_json::json!(args.pair_batch_size));
    hyperparameters.insert(
        "search_block_roots",
        serde_json::json!(args.search_block_roots),
    );
    hyperparameters.insert("pv_probe_roots", serde_json::json!(args.pv_probe_roots));
    hyperparameters.insert("jobs", serde_json::json!(args.jobs));
    hyperparameters.insert("leaf_reuse_passes", serde_json::json!(1));
    hyperparameters.insert("l1_lambda", serde_json::json!(0));
    hyperparameters.insert("l2_lambda", serde_json::json!(0));
    hyperparameters.insert("weight_decay", serde_json::json!(0));
    let manifest = RunManifest {
        schema_version: CHECKPOINT_SCHEMA,
        stage: "halfkp_mmto_train",
        command: std::env::args().collect(),
        environment: capture_run_environment(),
        completed_outer,
        dataset_manifest_sha256: dataset_manifest_sha256.to_owned(),
        init_model: artifact_metadata(&args.init, None)?,
        output_model: artifact_metadata(model_path, None)?,
        optimizer: artifact_metadata(optimizer_path, None)?,
        hyperparameters,
        last_train: report(train),
        last_valid_fresh: report(valid),
        pv_probe_best_move_unchanged,
        train_diagnostics,
        outer_delta,
        elapsed_ms,
    };
    atomic_json(&args.output_dir.join("run_manifest.json"), &manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_ai::evaluation::HALFKP_INPUTS;

    fn pair_fixture(side_to_move: Color) -> (Weights, Pair) {
        let mut root = Position::default();
        if side_to_move == Color::White {
            root.do_move(root.legal_moves()[0]);
        }
        let moves = root.legal_moves();
        let mut leaves = Vec::new();
        for mv in [moves[0], moves[1]] {
            let mut child = root.clone();
            child.do_move(mv);
            leaves.push(Leaf {
                packed: Some(PackedHalfKpPosition::from_position(&child).expect("packed child")),
                root_sign: -1.0,
                fixed_score_cp: 0.0,
                kind: QsearchLeafKind::StaticEvaluation,
            });
        }
        let mut out_w = [0.0; HALFKP_HIDDEN * 2 + 1];
        for (index, value) in out_w.iter_mut().enumerate() {
            *value = 0.005 + index as f32 * 0.00001;
        }
        (
            Weights {
                feature_emb: vec![0.0; HALFKP_INPUTS * HALFKP_HIDDEN],
                hidden_b: [0.5; HALFKP_HIDDEN],
                out_w,
                out_b: 0.0,
            },
            Pair {
                desired: leaves.remove(0),
                alternative: leaves.remove(0),
            },
        )
    }

    fn pair_loss(weights: &Weights, pair: &Pair, tau_cp: f32) -> f32 {
        let desired = current_leaf_score(weights, &pair.desired).0;
        let alternative = current_leaf_score(weights, &pair.alternative).0;
        mirrored_sigmoid(desired - alternative, tau_cp)
    }

    #[test]
    fn mirrored_sigmoid_is_symmetric_and_decreasing() {
        for delta in [-1000.0, -100.0, 0.0, 100.0, 1000.0] {
            let a = mirrored_sigmoid(delta, 36.63);
            let b = mirrored_sigmoid(-delta, 36.63);
            assert!((a + b - 1.0).abs() < 1e-6);
        }
        assert!(mirrored_sigmoid(10.0, 36.63) < mirrored_sigmoid(0.0, 36.63));
    }

    #[test]
    fn adagrad_zero_gradient_is_exact_noop() {
        let mut weight = 0.125f32;
        let mut accumulator = 0.0f32;
        assert!(!adagrad_scalar(
            &mut weight,
            &mut accumulator,
            0.0,
            1e-5,
            1.0
        ));
        assert_eq!(weight.to_bits(), 0.125f32.to_bits());
        assert_eq!(accumulator.to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn mirrored_sigmoid_derivative_matches_finite_difference() {
        let tau = 36.63;
        for delta in [-100.0f32, -10.0, 0.0, 10.0, 100.0] {
            let epsilon = 0.01;
            let numerical = (mirrored_sigmoid(delta + epsilon, tau)
                - mirrored_sigmoid(delta - epsilon, tau))
                / (2.0 * epsilon);
            let loss = mirrored_sigmoid(delta, tau);
            let analytical = -loss * (1.0 - loss) / tau;
            assert!(
                (numerical - analytical).abs() < 2e-5,
                "delta={delta} numerical={numerical} analytical={analytical}"
            );
        }
    }

    #[test]
    fn both_leaf_halfkp_gradient_matches_finite_difference_for_both_sides() {
        for side_to_move in [Color::Black, Color::White] {
            let (mut weights, pair) = pair_fixture(side_to_move);
            let mut gradient = Gradient::default();
            add_pair_gradient(&weights, &pair, 36.63, 1.0, &mut gradient).expect("gradient");
            let epsilon = 1e-3;

            let output_index = if side_to_move == Color::Black {
                0
            } else {
                HALFKP_HIDDEN
            };
            let analytical = gradient.out_w[output_index];
            weights.out_w[output_index] += epsilon;
            let plus = pair_loss(&weights, &pair, 36.63);
            weights.out_w[output_index] -= 2.0 * epsilon;
            let minus = pair_loss(&weights, &pair, 36.63);
            weights.out_w[output_index] += epsilon;
            let numerical = (plus - minus) / (2.0 * epsilon);
            assert!(
                (analytical - numerical).abs() < 2e-4,
                "side={side_to_move:?} output analytical={analytical} numerical={numerical}"
            );

            let (&feature, row) = gradient
                .sparse
                .iter()
                .find(|(_, row)| row.iter().any(|value| value.abs() > 1e-7))
                .expect("non-zero sparse gradient");
            let hidden = row
                .iter()
                .position(|value| value.abs() > 1e-7)
                .expect("non-zero hidden component");
            let index = feature * HALFKP_HIDDEN + hidden;
            let analytical = row[hidden];
            weights.feature_emb[index] += epsilon;
            let plus = pair_loss(&weights, &pair, 36.63);
            weights.feature_emb[index] -= 2.0 * epsilon;
            let minus = pair_loss(&weights, &pair, 36.63);
            let numerical = (plus - minus) / (2.0 * epsilon);
            assert!(
                (analytical - numerical).abs() < 2e-4,
                "side={side_to_move:?} sparse analytical={analytical} numerical={numerical}"
            );
        }
    }
}
