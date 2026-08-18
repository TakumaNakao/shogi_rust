use super::halfkp_search_train::{Weights, TARGET_SCALE};
use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use shogi_ai::ai::{QsearchLeafKind, ShogiAI};
use shogi_ai::evaluation::EngineEvaluator;
use shogi_ai::halfkp_training::PackedHalfKpPosition;
use shogi_ai::training_data::sha256_file;
use shogi_ai::utils::{format_move_usi, parse_usi_move_for_color, position_from_sfen_or_usi};
use shogi_core::{Color, Move};
use shogi_lib::Position;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(about = "Fresh-search per-root audit for HalfKP MMTO checkpoints")]
struct Args {
    #[arg(long, required = true)]
    model: PathBuf,
    #[arg(long, required = true)]
    input: PathBuf,
    #[arg(long, required = true)]
    output: PathBuf,
    #[arg(long, default_value_t = 36.63)]
    tau_cp: f32,
    #[arg(long, default_value_t = 1024)]
    block_roots: usize,
    #[arg(long, default_value_t = 8)]
    jobs: usize,
    #[arg(long)]
    max_roots: Option<usize>,
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
}

struct Sample {
    game_key: String,
    opening_family: String,
    sfen: String,
    root: Position,
    teacher: Move,
}

struct Leaf {
    packed: Option<PackedHalfKpPosition>,
    root_sign: f32,
    fixed_score_cp: f32,
}

#[derive(Serialize)]
struct RootAudit {
    root_index: usize,
    game_key: String,
    opening_family: String,
    sfen: String,
    root_hash: String,
    /// Stable PackedHalfKP network-input identity, independent of SFEN text.
    root_key: String,
    teacher_move: String,
    best_move: String,
    teacher_score_cp: f32,
    best_score_cp: f32,
    mean_pair_loss: f64,
    regret_cp: f32,
    agreement: bool,
    pairs: usize,
    exact_leaves: usize,
    terminal_leaves: usize,
    rejected_bounds: usize,
    qnodes: u64,
}

#[derive(Serialize)]
struct AuditManifest {
    schema: &'static str,
    model_path: PathBuf,
    model_sha256: String,
    input_path: PathBuf,
    input_sha256: String,
    tau_cp: f32,
    roots: usize,
    games: usize,
    rejected_bounds: usize,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    if args.tau_cp <= 0.0 || !args.tau_cp.is_finite() || args.block_roots == 0 || args.jobs == 0 {
        return Err(anyhow!("invalid audit hyperparameter"));
    }
    let weights = Arc::new(Weights::load(&args.model)?);
    let evaluator = Arc::new(EngineEvaluator::new(&args.model, 0.0)?);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.jobs)
        .build()
        .context("build audit search pool")?;
    let mut samples = Vec::new();
    for (line_no, line) in BufReader::new(File::open(&args.input)?).lines().enumerate() {
        if args.max_roots.is_some_and(|limit| samples.len() >= limit) {
            break;
        }
        let record: InputRecord = serde_json::from_str(&line?)
            .with_context(|| format!("parse {}:{}", args.input.display(), line_no + 1))?;
        if record.schema != "rank_value_dataset_v3_position" || record.split != "valid" {
            return Err(anyhow!("audit input has invalid schema or split"));
        }
        let position = position_from_sfen_or_usi(&record.sfen)
            .ok_or_else(|| anyhow!("invalid SFEN at line {}", line_no + 1))?;
        let expected_side = if position.side_to_move() == Color::Black {
            "black"
        } else {
            "white"
        };
        if record.side_to_move != expected_side {
            return Err(anyhow!("side-to-move mismatch at line {}", line_no + 1));
        }
        let teacher = parse_usi_move_for_color(&record.teacher_move, position.side_to_move())
            .filter(|mv| position.legal_moves().contains(mv))
            .ok_or_else(|| anyhow!("illegal teacher move at line {}", line_no + 1))?;
        if position.legal_moves().len() < 2 {
            continue;
        }
        samples.push(Sample {
            game_key: record.game_key,
            opening_family: record.opening_family,
            sfen: record.sfen,
            root: position,
            teacher,
        });
    }
    let mut writer = BufWriter::new(File::create(&args.output)?);
    let mut rejected_bounds = 0usize;
    for (block_index, block) in samples.chunks(args.block_roots).enumerate() {
        let offset = block_index * args.block_roots;
        let searched = pool.install(|| {
            block
                .par_iter()
                .map(|sample| audit_sample(Arc::clone(&evaluator), &weights, sample, args.tau_cp))
                .collect::<Result<Vec<_>>>()
        })?;
        for (index, audit) in searched.into_iter().enumerate() {
            rejected_bounds += audit.rejected_bounds;
            serde_json::to_writer(
                &mut writer,
                &RootAudit {
                    root_index: offset + index,
                    game_key: block[index].game_key.clone(),
                    opening_family: block[index].opening_family.clone(),
                    sfen: block[index].sfen.clone(),
                    root_hash: hash_text(&block[index].sfen),
                    root_key: hex_digest(
                        &PackedHalfKpPosition::from_position(&block[index].root)
                            .ok_or_else(|| anyhow!("audit root lacks king"))?
                            .network_input_hash(),
                    ),
                    teacher_move: format_move_usi(block[index].teacher),
                    best_move: format_move_usi(audit.best_move),
                    teacher_score_cp: audit.teacher_score_cp,
                    best_score_cp: audit.best_score_cp,
                    mean_pair_loss: audit.mean_pair_loss,
                    regret_cp: audit.best_score_cp - audit.teacher_score_cp,
                    agreement: audit.best_move == block[index].teacher,
                    pairs: audit.pairs,
                    exact_leaves: audit.exact_leaves,
                    terminal_leaves: audit.terminal_leaves,
                    rejected_bounds: audit.rejected_bounds,
                    qnodes: audit.qnodes,
                },
            )?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
        eprintln!("audit roots={}", (offset + block.len()).min(samples.len()));
    }
    writer.flush()?;
    drop(writer);
    if rejected_bounds != 0 {
        return Err(anyhow!(
            "audit encountered {rejected_bounds} qsearch bound leaves"
        ));
    }
    let manifest = AuditManifest {
        schema: "halfkp_mmto_root_audit_v1",
        model_path: args.model.clone(),
        model_sha256: sha256_file(&args.model)?,
        input_path: args.input.clone(),
        input_sha256: sha256_file(&args.input)?,
        tau_cp: args.tau_cp,
        roots: samples.len(),
        games: samples
            .iter()
            .map(|sample| sample.game_key.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        rejected_bounds,
    };
    atomic_json(&sidecar_path(&args.output), &manifest)?;
    Ok(())
}

struct SampleAudit {
    best_move: Move,
    teacher_score_cp: f32,
    best_score_cp: f32,
    mean_pair_loss: f64,
    pairs: usize,
    exact_leaves: usize,
    terminal_leaves: usize,
    rejected_bounds: usize,
    qnodes: u64,
}

fn audit_sample(
    evaluator: Arc<EngineEvaluator>,
    weights: &Weights,
    sample: &Sample,
    tau_cp: f32,
) -> Result<SampleAudit> {
    let mut ai: ShogiAI<_, 1024> = ShogiAI::new(evaluator);
    ai.set_emit_info(false);
    ai.reset_for_independent_search();
    ai.sennichite_detector.record_initial_position(&sample.root);
    let root_turn = sample.root.side_to_move();
    let mut root = sample.root.clone();
    let mut searched = Vec::new();
    let mut exact_leaves = 0;
    let mut terminal_leaves = 0;
    let rejected_bounds = 0;
    for mv in root.legal_moves() {
        let trace = ai
            .forced_move_qsearch_leaf_trace(&mut root, mv)
            .ok_or_else(|| anyhow!("forced qsearch aborted"))?;
        let leaf = if trace.has_exact_evaluation_leaf() {
            let position = trace.leaf.ok_or_else(|| anyhow!("missing exact leaf"))?;
            let root_sign = if position.side_to_move() == root_turn {
                1.0
            } else {
                -1.0
            };
            exact_leaves += 1;
            Leaf {
                packed: Some(
                    PackedHalfKpPosition::from_position(&position)
                        .ok_or_else(|| anyhow!("qsearch leaf lacks king"))?,
                ),
                root_sign,
                fixed_score_cp: trace.score,
            }
        } else if matches!(
            trace.kind,
            QsearchLeafKind::Mate | QsearchLeafKind::Repetition
        ) {
            terminal_leaves += 1;
            Leaf {
                packed: None,
                root_sign: 0.0,
                fixed_score_cp: trace.score,
            }
        } else {
            return Err(anyhow!("root full-window qsearch returned a bound"));
        };
        searched.push((mv, leaf));
    }
    let teacher_index = searched
        .iter()
        .position(|(mv, _)| *mv == sample.teacher)
        .ok_or_else(|| anyhow!("teacher missing after search"))?;
    let teacher_score_cp = searched[teacher_index].1.fixed_score_cp;
    let (best_move, best_score_cp) = searched
        .iter()
        .max_by(|lhs, rhs| lhs.1.fixed_score_cp.total_cmp(&rhs.1.fixed_score_cp))
        .map(|(mv, leaf)| (*mv, leaf.fixed_score_cp))
        .ok_or_else(|| anyhow!("empty legal move set"))?;
    let desired = &searched[teacher_index].1;
    let desired_score = current_leaf_score(weights, desired);
    let mut loss_sum = 0.0f64;
    let mut pairs = 0usize;
    for (index, (_, alternative)) in searched.iter().enumerate() {
        if index == teacher_index {
            continue;
        }
        let alternative_score = current_leaf_score(weights, alternative);
        loss_sum += f64::from(mirrored_sigmoid(desired_score - alternative_score, tau_cp));
        pairs += 1;
    }
    Ok(SampleAudit {
        best_move,
        teacher_score_cp,
        best_score_cp,
        mean_pair_loss: loss_sum / pairs.max(1) as f64,
        pairs,
        exact_leaves,
        terminal_leaves,
        rejected_bounds,
        qnodes: ai.quiescence_nodes_searched(),
    })
}

fn current_leaf_score(weights: &Weights, leaf: &Leaf) -> f32 {
    match leaf.packed.as_ref() {
        Some(position) => leaf.root_sign * weights.forward(position).raw * TARGET_SCALE,
        None => leaf.fixed_score_cp,
    }
}

fn mirrored_sigmoid(delta_cp: f32, tau_cp: f32) -> f32 {
    let x = (delta_cp / tau_cp).clamp(-80.0, 80.0);
    1.0 / (1.0 + x.exp())
}

fn hash_text(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sidecar_path(path: &Path) -> PathBuf {
    path.with_extension("jsonl.manifest.json")
}

fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("tmp");
    let mut writer = BufWriter::new(File::create(&temporary)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    drop(writer);
    fs::rename(temporary, path)?;
    Ok(())
}
