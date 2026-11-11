use anyhow::{anyhow, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use shogi_ai::ai::{SearchLimits, ShogiAI};
use shogi_ai::evaluation::EngineEvaluator;
use shogi_ai::halfkp_training::{CurriculumReader, ProofMode};
use shogi_ai::mate_proof::{solve_bounded_mate_for, AttackMoveMode, MateLimits, MateProofStatus};
use shogi_ai::training_data::sha256_file;
use shogi_core::PartialPosition;
use shogi_lib::Position;
use shogi_usi_parser::FromUsi;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

const LINEAGE_ID: &str = "halfkp64-scratch-v1";
const HISTORY_CAPACITY: usize = 256;

#[derive(Parser, Debug)]
#[command(about = "Evaluate deterministic fixed-node search on replayable HKCL proofs")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    model_manifest: PathBuf,
    #[arg(long, required = true)]
    dataset: Vec<PathBuf>,
    #[arg(long)]
    nodes: u64,
    #[arg(long, default_value_t = 64)]
    max_depth: u8,
    #[arg(long)]
    json: Option<PathBuf>,
    /// Optional per-position diagnostics (including inconclusive searches).
    #[arg(long)]
    details_json: Option<PathBuf>,
}

#[derive(Deserialize)]
struct ModelManifest {
    lineage_id: String,
    model: ModelArtifact,
}

#[derive(Deserialize)]
struct ModelArtifact {
    sha256: String,
}

#[derive(Default, Serialize)]
struct RoleStats {
    roots: usize,
    conclusive: usize,
    successful: usize,
}

#[derive(Default, Serialize)]
struct Report {
    model_sha256: String,
    node_budget: u64,
    roots: usize,
    conclusive: usize,
    successful: usize,
    completed_depth_zero: usize,
    search_failed: usize,
    nodes_searched: u64,
    success_rate_all_roots: f64,
    success_rate_conclusive: f64,
    mean_nodes: f64,
    attacker: RoleStats,
    defender: RoleStats,
    by_horizon: BTreeMap<u16, RoleStats>,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    if args.details_json.as_ref().is_some_and(|p| p.exists()) {
        return Err(anyhow!("refusing to overwrite position diagnostics"));
    }
    let mut details = Vec::new();
    if args.nodes == 0 || args.max_depth == 0 {
        return Err(anyhow!("--nodes and --max-depth must be positive"));
    }
    let manifest: ModelManifest = serde_json::from_slice(&fs::read(&args.model_manifest)?)?;
    if manifest.lineage_id != LINEAGE_ID || manifest.model.sha256 != sha256_file(&args.model)? {
        return Err(anyhow!(
            "model is not the declared scratch-lineage artifact"
        ));
    }
    let evaluator = Arc::new(EngineEvaluator::new(&args.model, 0.0)?);
    let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
    ai.set_emit_info(false);
    let mut report = Report {
        model_sha256: manifest.model.sha256,
        node_budget: args.nodes,
        ..Report::default()
    };

    for path in &args.dataset {
        let mut reader = CurriculumReader::open(path)?;
        if reader.header().lineage_id != LINEAGE_ID {
            return Err(anyhow!("dataset is outside scratch lineage"));
        }
        while let Some(record) = reader.read_record()? {
            let proof = record
                .proof
                .as_ref()
                .ok_or_else(|| anyhow!("search exact gate requires proof metadata"))?;
            let mut position = Position::new(
                PartialPosition::from_usi(&format!("sfen {}", record.root_sfen))
                    .map_err(|error| anyhow!("parse gate SFEN: {error:?}"))?,
            );
            let root_is_attacker = position.side_to_move() == proof.attacker;
            report.roots += 1;
            role_mut(&mut report, root_is_attacker).roots += 1;
            report
                .by_horizon
                .entry(proof.horizon_plies)
                .or_default()
                .roots += 1;

            ai.reset_for_independent_search();
            ai.sennichite_detector.record_initial_position(&position);
            let outcome = ai.search(
                &mut position,
                SearchLimits::new(args.max_depth, None).with_max_nodes(args.nodes),
            );
            report.nodes_searched += outcome.stats.nodes;
            details.push(serde_json::json!({
                "id": report.roots, "sfen": record.root_sfen,
                "search_failed": outcome.failed,
                "depth": outcome.root.as_ref().map(|r| r.completed_depth).unwrap_or(0),
                "best_move": outcome.root.as_ref().map(|r| format!("{:?}", r.best_move)),
                "successful": false, "proof_status": null,
            }));
            if outcome.failed {
                report.search_failed += 1;
                continue;
            }
            let Some(root_result) = outcome.root else {
                report.completed_depth_zero += 1;
                continue;
            };
            if root_result.completed_depth == 0 {
                report.completed_depth_zero += 1;
                continue;
            }
            report.conclusive += 1;
            role_mut(&mut report, root_is_attacker).conclusive += 1;
            report
                .by_horizon
                .entry(proof.horizon_plies)
                .or_default()
                .conclusive += 1;

            let mut child = position.clone();
            child.do_move(root_result.best_move);
            let selected = solve_bounded_mate_for(
                &child,
                proof.attacker,
                MateLimits {
                    max_plies: proof.horizon_plies.saturating_sub(1),
                    max_nodes: proof.max_nodes,
                    mode: match proof.solver_mode {
                        ProofMode::CheckingOnly => AttackMoveMode::CheckingOnly,
                        ProofMode::AllLegal => AttackMoveMode::AllLegal,
                    },
                },
            )?;
            let successful = if root_is_attacker {
                selected.root.status == MateProofStatus::ProvenMate
            } else {
                selected.root.status == MateProofStatus::ProvenNoMate
            };
            let detail = details.last_mut().expect("position diagnostic");
            detail["successful"] = serde_json::json!(successful);
            detail["proof_status"] = serde_json::json!(format!("{:?}", selected.root.status));
            if successful {
                report.successful += 1;
                role_mut(&mut report, root_is_attacker).successful += 1;
                report
                    .by_horizon
                    .entry(proof.horizon_plies)
                    .or_default()
                    .successful += 1;
            }
        }
    }
    report.success_rate_all_roots = ratio(report.successful, report.roots);
    report.success_rate_conclusive = ratio(report.successful, report.conclusive);
    report.mean_nodes = report.nodes_searched as f64 / report.roots.max(1) as f64;
    let bytes = serde_json::to_vec_pretty(&report)?;
    if let Some(path) = args.details_json {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(&details)?)?;
    }
    println!("{}", String::from_utf8_lossy(&bytes));
    if let Some(path) = args.json {
        if path.exists() {
            return Err(anyhow!("refusing to overwrite {}", path.display()));
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, [&bytes[..], b"\n"].concat())?;
    }
    Ok(())
}

fn role_mut(report: &mut Report, attacker: bool) -> &mut RoleStats {
    if attacker {
        &mut report.attacker
    } else {
        &mut report.defender
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    numerator as f64 / denominator.max(1) as f64
}
