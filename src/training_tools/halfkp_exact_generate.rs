use anyhow::{anyhow, Result};
use clap::Parser;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use shogi_ai::halfkp_training::{
    CurriculumHeader, CurriculumPreference, CurriculumReader, CurriculumRecord, CurriculumWriter,
    LabelSource, OpeningStratum, PackedHalfKpPosition, PreferenceKind, ProofLabel, ProofStatus,
};
use shogi_ai::mate_proof::{
    solve_bounded_mate, solve_bounded_mate_for, AttackMoveMode, MateLimits, MateProofStatus,
};
use shogi_ai::training_data::{
    capture_run_environment, split_for_game_content, DatasetSplit, TrainingPhase,
};
use shogi_lib::Position;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

const LINEAGE_ID: &str = "halfkp64-scratch-v1";
const RULE_VERSION: u32 = 1;

#[derive(Parser, Debug)]
#[command(about = "Generate evaluator-free bounded mate HKCL records from legal random games")]
struct Args {
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 200)]
    mate1: usize,
    #[arg(long, default_value_t = 200)]
    mate3: usize,
    #[arg(long, default_value_t = 400)]
    no_mate3: usize,
    #[arg(long, default_value_t = 2_000_000)]
    max_positions: usize,
    #[arg(long, default_value_t = 100_000)]
    node_limit: u64,
    #[arg(long, default_value_t = 20260721)]
    seed: u64,
}

struct Writers {
    train: CurriculumWriter,
    valid: CurriculumWriter,
    test: CurriculumWriter,
}

impl Writers {
    fn write(&mut self, record: &CurriculumRecord) -> Result<()> {
        match record.split {
            DatasetSplit::Train => self.train.write_record(record),
            DatasetSplit::Valid => self.valid.write_record(record),
            DatasetSplit::Test => self.test.write_record(record),
        }
    }

    fn finish(self) -> Result<[u64; 3]> {
        Ok([
            self.train.finish()?,
            self.valid.finish()?,
            self.test.finish()?,
        ])
    }
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    if args.mate1 == 0 || args.mate3 == 0 || args.no_mate3 == 0 || args.node_limit == 0 {
        return Err(anyhow!(
            "all category targets and node-limit must be non-zero"
        ));
    }
    std::fs::create_dir_all(&args.output_dir)?;
    let revision = capture_run_environment()
        .git_revision
        .unwrap_or_else(|| "unknown".to_string());
    let header = CurriculumHeader {
        lineage_id: LINEAGE_ID.to_string(),
        feature_profile: "halfkp64-mirror-file-v1".to_string(),
        producer_revision: revision,
        seed: args.seed,
        input_sha256: vec![],
        generator_config: serde_json::json!({
            "generator": "legal_random_play_bounded_mate_v1",
            "mate_plies": [1, 3],
            "node_limit": args.node_limit,
            "rule_version": RULE_VERSION,
            "attack_move_mode": "checking_only",
            "targets": {"mate1": args.mate1, "mate3": args.mate3, "no_mate3": args.no_mate3},
        }),
        teacher_model_sha256: None,
        teacher_lineage_id: None,
    };
    let mut writers = Writers {
        train: CurriculumWriter::create(&args.output_dir.join("train.hkcl"), header.clone())?,
        valid: CurriculumWriter::create(&args.output_dir.join("valid.hkcl"), header.clone())?,
        test: CurriculumWriter::create(&args.output_dir.join("test.hkcl"), header)?,
    };
    let mut rng = ChaCha20Rng::seed_from_u64(args.seed);
    let mut position = Position::default();
    let mut game_ply = 0usize;
    let mut seen_hashes = HashSet::new();
    let mut mate1 = 0usize;
    let mut mate3 = 0usize;
    let mut no_mate3 = 0usize;
    let mut examined = 0usize;
    let mut resource_limits = 0usize;
    while examined < args.max_positions
        && (mate1 < args.mate1 || mate3 < args.mate3 || no_mate3 < args.no_mate3)
    {
        examined += 1;
        let sfen = position.to_sfen_owned();
        let family_hash = Sha256::digest(sfen.as_bytes());
        let mut family = [0u8; 32];
        family.copy_from_slice(&family_hash);
        if seen_hashes.insert(family) {
            let proof = solve_bounded_mate(
                &position,
                MateLimits {
                    max_plies: 3,
                    max_nodes: args.node_limit,
                    mode: AttackMoveMode::CheckingOnly,
                },
            )?;
            let category_available = match (proof.root.status, proof.root.mate_distance) {
                (MateProofStatus::ProvenMate, Some(1)) => mate1 < args.mate1,
                (MateProofStatus::ProvenMate, Some(3)) => mate3 < args.mate3,
                (MateProofStatus::ProvenNoMate, None) => no_mate3 < args.no_mate3,
                (MateProofStatus::ResourceLimit, _) => {
                    resource_limits += 1;
                    false
                }
                _ => false,
            };
            if category_available {
                let record = make_record(&position, family, &proof, args.seed, args.node_limit)?;
                writers.write(&record)?;
                match (proof.root.status, proof.root.mate_distance) {
                    (MateProofStatus::ProvenMate, Some(1)) => mate1 += 1,
                    (MateProofStatus::ProvenMate, Some(3)) => mate3 += 1,
                    (MateProofStatus::ProvenNoMate, None) => no_mate3 += 1,
                    _ => unreachable!(),
                }
            }
        }

        let legal = position.legal_moves();
        game_ply += 1;
        if legal.is_empty() || game_ply >= 300 {
            position = Position::default();
            game_ply = 0;
        } else {
            position.do_move(legal[rng.gen_range(0..legal.len())]);
        }
        if examined % 10_000 == 0 {
            println!("examined={examined} mate1={mate1} mate3={mate3} no_mate3={no_mate3} resource={resource_limits}");
        }
    }
    let records = writers.finish()?;
    println!("complete examined={examined} mate1={mate1} mate3={mate3} no_mate3={no_mate3} resource={resource_limits} train={} valid={} test={}", records[0], records[1], records[2]);
    if mate1 < args.mate1 || mate3 < args.mate3 || no_mate3 < args.no_mate3 {
        return Err(anyhow!("category targets not reached before max-positions"));
    }
    // Re-open all outputs so the declared count and trailing-byte checks run.
    for name in ["train.hkcl", "valid.hkcl", "test.hkcl"] {
        verify_dataset(&args.output_dir.join(name))?;
    }
    Ok(())
}

fn make_record(
    position: &Position,
    family_hash: [u8; 32],
    proof: &shogi_ai::mate_proof::MateProof,
    seed: u64,
    node_limit: u64,
) -> Result<CurriculumRecord> {
    let hex = family_hash
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let split = split_for_game_content(&hex, seed, 10, 10);
    let status = match proof.root.status {
        MateProofStatus::ProvenMate => ProofStatus::ProvenMate,
        MateProofStatus::ProvenNoMate => ProofStatus::ProvenNoMate,
        MateProofStatus::ResourceLimit => ProofStatus::ResourceLimit,
    };
    let preferences = exact_preferences(position, proof, node_limit)?;
    Ok(CurriculumRecord {
        source: LabelSource::ExactMate,
        group_id: u64::from_le_bytes(family_hash[..8].try_into().expect("hash width")),
        position_hash: position.key(),
        family_hash,
        split,
        ply: position.ply(),
        phase: TrainingPhase::for_ply(position.ply() as usize).code(),
        opening_stratum: OpeningStratum::UNKNOWN,
        confidence: 1.0,
        sample_weight: 1.0,
        root: PackedHalfKpPosition::from_position(position)
            .ok_or_else(|| anyhow!("generated legal position lacks a king"))?,
        // Exact mate is a bounded proof objective, not a literal game-result
        // probability of one. The trainer applies its configured bounded target.
        result: None,
        proof: Some(ProofLabel {
            status,
            mate_distance: proof.root.mate_distance,
            nodes: proof.nodes,
            rule_version: RULE_VERSION,
        }),
        preferences,
    })
}

fn exact_preferences(
    position: &Position,
    proof: &shogi_ai::mate_proof::MateProof,
    node_limit: u64,
) -> Result<Vec<CurriculumPreference>> {
    if proof.root.status != MateProofStatus::ProvenMate || proof.root.edges.len() != 1 {
        return Ok(vec![]);
    }
    let good_move = proof.root.edges[0].mv;
    let mut good = position.clone();
    good.do_move(good_move);
    let Some(good_child) = PackedHalfKpPosition::from_position(&good) else {
        return Ok(vec![]);
    };
    for bad_move in position.legal_moves() {
        if bad_move == good_move || !position.is_check_move(bad_move) {
            continue;
        }
        let mut bad = position.clone();
        bad.do_move(bad_move);
        let alternative = solve_bounded_mate_for(
            &bad,
            proof.attacker,
            MateLimits {
                max_plies: proof.max_plies.saturating_sub(1),
                max_nodes: node_limit,
                mode: proof.mode,
            },
        )?;
        if alternative.root.status == MateProofStatus::ProvenNoMate {
            if let Some(bad_child) = PackedHalfKpPosition::from_position(&bad) {
                return Ok(vec![CurriculumPreference {
                    good_child,
                    bad_child,
                    kind: PreferenceKind::ProvenOutcome,
                    margin_class: proof.root.mate_distance.unwrap_or(0).min(u8::MAX as u16) as u8,
                }]);
            }
        }
    }
    Ok(vec![])
}

fn verify_dataset(path: &Path) -> Result<()> {
    let mut reader = CurriculumReader::open(path)?;
    while reader.read_record()?.is_some() {}
    Ok(())
}
