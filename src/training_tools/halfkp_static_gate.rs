use anyhow::{anyhow, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use shogi_ai::evaluation::HalfKpFlatModel;
use shogi_ai::halfkp_training::{CurriculumReader, PackedHalfKpPosition, ProofMode};
use shogi_ai::mate_proof::{solve_bounded_mate_for, AttackMoveMode, MateLimits, MateProofStatus};
use shogi_ai::training_data::sha256_file;
use shogi_core::{Color, PartialPosition};
use shogi_lib::Position;
use shogi_usi_parser::FromUsi;
use std::fs;
use std::path::PathBuf;

const LINEAGE_ID: &str = "halfkp64-scratch-v1";
const TERMINAL_LOSS_CP: f32 = -30_000.0;

#[derive(Parser, Debug)]
#[command(about = "Evaluate static all-legal exact top-1 on replayable HKCL proofs")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    model_manifest: PathBuf,
    #[arg(long, required = true)]
    dataset: Vec<PathBuf>,
    #[arg(long)]
    json: Option<PathBuf>,
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
struct Report {
    model_sha256: String,
    roots: usize,
    roots_with_good: usize,
    covered_roots: usize,
    top1_correct: usize,
    attacker_roots: usize,
    attacker_top1_correct: usize,
    defender_roots: usize,
    defender_top1_correct: usize,
    legal_moves: usize,
    proven_good_moves: usize,
    resource_limit_moves: usize,
    coverage: f64,
    top1_accuracy_all_roots: f64,
    top1_accuracy_covered_roots: f64,
    attacker_top1_accuracy: f64,
    defender_top1_accuracy: f64,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    let manifest: ModelManifest = serde_json::from_slice(&fs::read(&args.model_manifest)?)?;
    if manifest.lineage_id != LINEAGE_ID || manifest.model.sha256 != sha256_file(&args.model)? {
        return Err(anyhow!(
            "model is not the declared scratch-lineage artifact"
        ));
    }
    let model = HalfKpFlatModel::decode(&fs::read(&args.model)?)?;
    let mut report = Report {
        model_sha256: manifest.model.sha256,
        ..Report::default()
    };
    for path in &args.dataset {
        let mut reader = CurriculumReader::open(path)?;
        if reader.header().lineage_id != LINEAGE_ID {
            return Err(anyhow!("dataset is outside scratch lineage"));
        }
        while let Some(record) = reader.read_record()? {
            let expected = record
                .proof
                .as_ref()
                .ok_or_else(|| anyhow!("static exact Gate requires proof metadata"))?;
            let position = Position::new(
                PartialPosition::from_usi(&format!("sfen {}", record.root_sfen))
                    .map_err(|error| anyhow!("parse Gate SFEN: {error:?}"))?,
            );
            let root_is_attacker = position.side_to_move() == expected.attacker;
            report.roots += 1;
            if root_is_attacker {
                report.attacker_roots += 1;
            } else {
                report.defender_roots += 1;
            }
            let mut best: Option<(f32, bool)> = None;
            let mut fully_covered = true;
            let mut root_has_good = false;
            let moves = position.legal_moves();
            report.legal_moves += moves.len();
            for mv in moves {
                let mut child = position.clone();
                child.do_move(mv);
                let proof = solve_bounded_mate_for(
                    &child,
                    expected.attacker,
                    MateLimits {
                        max_plies: expected.horizon_plies.saturating_sub(1),
                        max_nodes: expected.max_nodes,
                        mode: match expected.solver_mode {
                            ProofMode::CheckingOnly => AttackMoveMode::CheckingOnly,
                            ProofMode::AllLegal => AttackMoveMode::AllLegal,
                        },
                    },
                )?;
                let good = if root_is_attacker {
                    proof.root.status == MateProofStatus::ProvenMate
                } else {
                    proof.root.status == MateProofStatus::ProvenNoMate
                };
                if good {
                    report.proven_good_moves += 1;
                    root_has_good = true;
                }
                if proof.root.status == MateProofStatus::ResourceLimit {
                    report.resource_limit_moves += 1;
                    fully_covered = false;
                }
                let score = match PackedHalfKpPosition::from_position(&child) {
                    Some(packed) => score(&model, &packed),
                    None if child.legal_moves().is_empty() && child.in_check() => TERMINAL_LOSS_CP,
                    None => continue,
                };
                if best.as_ref().is_none_or(|(current, _)| score < *current) {
                    best = Some((score, good));
                }
            }
            if root_has_good {
                report.roots_with_good += 1;
            }
            if fully_covered {
                report.covered_roots += 1;
            }
            if best.is_some_and(|(_, good)| good) && fully_covered {
                report.top1_correct += 1;
                if root_is_attacker {
                    report.attacker_top1_correct += 1;
                } else {
                    report.defender_top1_correct += 1;
                }
            }
        }
    }
    report.coverage = ratio(report.covered_roots, report.roots);
    report.top1_accuracy_all_roots = ratio(report.top1_correct, report.roots);
    report.top1_accuracy_covered_roots = ratio(report.top1_correct, report.covered_roots);
    report.attacker_top1_accuracy = ratio(report.attacker_top1_correct, report.attacker_roots);
    report.defender_top1_accuracy = ratio(report.defender_top1_correct, report.defender_roots);
    let bytes = serde_json::to_vec_pretty(&report)?;
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

fn score(model: &HalfKpFlatModel, position: &PackedHalfKpPosition) -> f32 {
    HalfKpFlatModel::forward_parts(
        &model.feature_emb,
        &model.hidden_b,
        &model.out_w,
        model.out_b,
        position.features_black.iter().map(|&value| value as usize),
        position.features_white.iter().map(|&value| value as usize),
        position.material_black,
        position.material_white,
        position.side_to_move == Color::Black,
        model.header.target_scale,
    )
    .raw * model.header.target_scale
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    numerator as f64 / denominator.max(1) as f64
}
