use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::Serialize;
use shogi_ai::halfkp_training::{CurriculumReader, CurriculumRecord, ProofMode, ProofStatus};
use shogi_ai::mate_proof::{
    solve_bounded_mate, solve_bounded_mate_for, verify_mate_proof, AttackMoveMode, MateLimits,
    MateProofStatus,
};
use shogi_ai::training_data::DatasetSplit;
use shogi_core::PartialPosition;
use shogi_lib::Position;
use shogi_usi_parser::FromUsi;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Audit HKCL family isolation and network-input split collisions")]
struct Args {
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    #[arg(long)]
    report: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    revalidate_proofs: bool,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema: &'static str,
    files: usize,
    records: u64,
    families: usize,
    unique_network_inputs: usize,
    duplicate_network_inputs_within_split: u64,
    family_split_collisions: u64,
    network_input_split_collisions: u64,
    records_by_split: [u64; 3],
    proofs: u64,
    proof_revalidation_failures: u64,
    proof_revalidation_first_error: Option<String>,
    passed: bool,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    let mut family_splits = HashMap::<[u8; 32], DatasetSplit>::new();
    let mut input_splits = HashMap::<[u8; 32], DatasetSplit>::new();
    let mut duplicate_inputs = HashSet::<([u8; 32], u8)>::new();
    let mut family_collisions = HashSet::<[u8; 32]>::new();
    let mut input_collisions = HashSet::<[u8; 32]>::new();
    let mut records = 0u64;
    let mut records_by_split = [0u64; 3];
    let mut proofs = 0u64;
    let mut proof_revalidation_failures = 0u64;
    let mut proof_revalidation_first_error = None;

    for path in &args.inputs {
        let mut reader =
            CurriculumReader::open(path).with_context(|| format!("open {}", path.display()))?;
        while let Some(record) = reader.read_record()? {
            records += 1;
            let split_index = split_index(record.split);
            records_by_split[split_index as usize] += 1;
            if let Some(previous) = family_splits.insert(record.family_hash, record.split) {
                if previous != record.split {
                    family_collisions.insert(record.family_hash);
                }
            }
            if let Some(previous) = input_splits.insert(record.network_input_hash, record.split) {
                if previous != record.split {
                    input_collisions.insert(record.network_input_hash);
                } else {
                    duplicate_inputs.insert((record.network_input_hash, split_index));
                }
            }
            for position in record
                .preferences
                .iter()
                .flat_map(|preference| [&preference.good_child, &preference.bad_child])
            {
                let hash = position.network_input_hash();
                if let Some(previous) = input_splits.insert(hash, record.split) {
                    if previous != record.split {
                        input_collisions.insert(hash);
                    } else {
                        duplicate_inputs.insert((hash, split_index));
                    }
                }
            }
            if record.proof.is_some() {
                proofs += 1;
                if args.revalidate_proofs {
                    match revalidate_record(&record) {
                        Ok(true) => {}
                        Ok(false) => {
                            proof_revalidation_failures += 1;
                            proof_revalidation_first_error
                                .get_or_insert_with(|| "proof result mismatch".into());
                        }
                        Err(error) => {
                            proof_revalidation_failures += 1;
                            proof_revalidation_first_error
                                .get_or_insert_with(|| format!("{error:#}"));
                        }
                    }
                }
            }
        }
    }
    let passed = family_collisions.is_empty()
        && input_collisions.is_empty()
        && proof_revalidation_failures == 0;
    let report = AuditReport {
        schema: "halfkp_curriculum_audit_v1",
        files: args.inputs.len(),
        records,
        families: family_splits.len(),
        unique_network_inputs: input_splits.len(),
        duplicate_network_inputs_within_split: duplicate_inputs.len() as u64,
        family_split_collisions: family_collisions.len() as u64,
        network_input_split_collisions: input_collisions.len() as u64,
        records_by_split,
        proofs,
        proof_revalidation_failures,
        proof_revalidation_first_error,
        passed,
    };
    let bytes = serde_json::to_vec_pretty(&report)?;
    println!("{}", String::from_utf8_lossy(&bytes));
    if let Some(path) = args.report {
        if path.exists() {
            return Err(anyhow!("refusing to overwrite {}", path.display()));
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, [&bytes[..], b"\n"].concat())?;
    }
    if !passed {
        return Err(anyhow!("HKCL split isolation audit failed"));
    }
    Ok(())
}

fn revalidate_record(record: &CurriculumRecord) -> Result<bool> {
    let expected = record.proof.as_ref().ok_or_else(|| anyhow!("no proof"))?;
    let position = Position::new(
        PartialPosition::from_usi(&format!("sfen {}", record.root_sfen))
            .map_err(|error| anyhow!("parse proof SFEN: {error:?}"))?,
    );
    let proof = solve_bounded_mate(
        &position,
        MateLimits {
            max_plies: expected.horizon_plies,
            max_nodes: expected.max_nodes,
            mode: match expected.solver_mode {
                ProofMode::CheckingOnly => AttackMoveMode::CheckingOnly,
                ProofMode::AllLegal => AttackMoveMode::AllLegal,
            },
        },
    )?;
    verify_mate_proof(&position, &proof)?;
    let status_matches = matches!(
        (expected.status, proof.root.status),
        (ProofStatus::ProvenMate, MateProofStatus::ProvenMate)
            | (
                ProofStatus::NoForcedMateWithinHorizon,
                MateProofStatus::ProvenNoMate
            )
            | (ProofStatus::ResourceLimit, MateProofStatus::ResourceLimit)
    );
    if !status_matches || expected.mate_distance != proof.root.mate_distance {
        return Ok(false);
    }
    for preference in &record.preferences {
        let (Some(good_text), Some(bad_text)) = (
            preference.good_move_usi.as_deref(),
            preference.bad_move_usi.as_deref(),
        ) else {
            continue;
        };
        let attacker = position.side_to_move();
        let good_move = shogi_ai::utils::parse_usi_move_for_color(good_text, attacker)
            .ok_or_else(|| anyhow!("invalid good move"))?;
        let bad_move = shogi_ai::utils::parse_usi_move_for_color(bad_text, attacker)
            .ok_or_else(|| anyhow!("invalid bad move"))?;
        let limits = MateLimits {
            max_plies: expected.horizon_plies.saturating_sub(1),
            max_nodes: expected.max_nodes,
            mode: proof.mode,
        };
        let mut good = position.clone();
        good.do_move(good_move);
        let good_proof = solve_bounded_mate_for(&good, attacker, limits)?;
        let mut bad = position.clone();
        bad.do_move(bad_move);
        let bad_proof = solve_bounded_mate_for(&bad, attacker, limits)?;
        if good_proof.root.status != MateProofStatus::ProvenMate
            || bad_proof.root.status != MateProofStatus::ProvenNoMate
        {
            return Ok(false);
        }
        // `verify_mate_proof` currently assumes the attacker moves at the
        // proof root, while these child proofs intentionally start with the
        // defender. Re-solving both children and checking their typed status
        // is the independent witness here; the root proof is fully verified.
    }
    Ok(true)
}

fn split_index(split: DatasetSplit) -> u8 {
    match split {
        DatasetSplit::Train => 0,
        DatasetSplit::Valid => 1,
        DatasetSplit::Test => 2,
    }
}
