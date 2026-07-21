use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use shogi_ai::halfkp_training::{
    CurriculumHeader, CurriculumPreference, CurriculumReader, CurriculumRecord, CurriculumWriter,
    LabelSource, OpeningStratum, PackedHalfKpPosition, PreferenceKind, ProofLabel, ProofMode,
    ProofStatus,
};
use shogi_ai::mate_proof::{
    solve_bounded_mate, solve_bounded_mate_for, AttackMoveMode, MateLimits, MateProofStatus,
};
use shogi_ai::training_data::{capture_run_environment, DatasetSplit, TrainingPhase};
use shogi_lib::Position;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

const LINEAGE_ID: &str = "halfkp64-scratch-v1";
const RULE_VERSION: u32 = 1;

#[derive(Parser, Debug)]
#[command(about = "Generate evaluator-free bounded mate HKCL records from legal random games")]
struct Args {
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum)]
    split: SplitArg,
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

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SplitArg {
    Train,
    Valid,
    Test,
}

impl SplitArg {
    fn dataset(self) -> DatasetSplit {
        match self {
            Self::Train => DatasetSplit::Train,
            Self::Valid => DatasetSplit::Valid,
            Self::Test => DatasetSplit::Test,
        }
    }
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    if args.mate1 == 0 || args.mate3 == 0 || args.no_mate3 == 0 || args.node_limit == 0 {
        return Err(anyhow!(
            "all category targets and node-limit must be non-zero"
        ));
    }
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
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
            "generator": "legal_random_play_bounded_mate_v2",
            "stream_split": format!("{:?}", args.split).to_ascii_lowercase(),
            "mate_plies": [1, 3],
            "node_limit": args.node_limit,
            "rule_version": RULE_VERSION,
            "attack_move_mode": "checking_only",
            "targets": {"mate1": args.mate1, "mate3": args.mate3, "no_mate3": args.no_mate3},
        }),
        teacher_model_sha256: None,
        teacher_lineage_id: None,
    };
    let mut writer = CurriculumWriter::create(&args.output, header)?;
    let mut rng = ChaCha20Rng::seed_from_u64(args.seed);
    let mut position = Position::default();
    let mut game_ply = 0usize;
    let mut seen_inputs = HashSet::new();
    let mut game_index = 0u64;
    let mut family = family_hash(args.seed, game_index);
    let mut mate1 = 0usize;
    let mut mate3 = 0usize;
    let mut no_mate3 = 0usize;
    let mut examined = 0usize;
    let mut resource_limits = 0usize;
    while examined < args.max_positions
        && (mate1 < args.mate1 || mate3 < args.mate3 || no_mate3 < args.no_mate3)
    {
        examined += 1;
        let packed = PackedHalfKpPosition::from_position(&position)
            .ok_or_else(|| anyhow!("generated legal position lacks a king"))?;
        if seen_inputs.insert(packed.network_input_hash()) {
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
                let record = make_record(
                    &position,
                    family,
                    args.split.dataset(),
                    &proof,
                    args.node_limit,
                )?;
                writer.write_record(&record)?;
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
            game_index += 1;
            family = family_hash(args.seed, game_index);
        } else {
            position.do_move(legal[rng.gen_range(0..legal.len())]);
        }
        if examined % 10_000 == 0 {
            println!("examined={examined} mate1={mate1} mate3={mate3} no_mate3={no_mate3} resource={resource_limits}");
        }
    }
    let records = writer.finish()?;
    println!("complete examined={examined} mate1={mate1} mate3={mate3} no_mate3={no_mate3} resource={resource_limits} records={records}");
    if mate1 < args.mate1 || mate3 < args.mate3 || no_mate3 < args.no_mate3 {
        return Err(anyhow!("category targets not reached before max-positions"));
    }
    // Re-open all outputs so the declared count and trailing-byte checks run.
    verify_dataset(&args.output)?;
    Ok(())
}

fn family_hash(seed: u64, game_index: u64) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"halfkp-exact-random-game-family-v2\0");
    hash.update(seed.to_le_bytes());
    hash.update(game_index.to_le_bytes());
    hash.finalize().into()
}

fn make_record(
    position: &Position,
    family_hash: [u8; 32],
    split: DatasetSplit,
    proof: &shogi_ai::mate_proof::MateProof,
    node_limit: u64,
) -> Result<CurriculumRecord> {
    let status = match proof.root.status {
        MateProofStatus::ProvenMate => ProofStatus::ProvenMate,
        MateProofStatus::ProvenNoMate => ProofStatus::NoForcedMateWithinHorizon,
        MateProofStatus::ResourceLimit => ProofStatus::ResourceLimit,
    };
    let preferences = exact_preferences(position, proof, node_limit)?;
    let root = PackedHalfKpPosition::from_position(position)
        .ok_or_else(|| anyhow!("generated legal position lacks a king"))?;
    Ok(CurriculumRecord {
        source: LabelSource::ExactMate,
        group_id: u64::from_le_bytes(family_hash[..8].try_into().expect("hash width")),
        position_hash: position.key(),
        family_hash,
        network_input_hash: root.network_input_hash(),
        split,
        ply: position.ply(),
        phase: TrainingPhase::for_ply(position.ply() as usize).code(),
        opening_stratum: OpeningStratum::UNKNOWN,
        confidence: 1.0,
        sample_weight: 1.0,
        root,
        // Exact mate is a bounded proof objective, not a literal game-result
        // probability of one. The trainer applies its configured bounded target.
        result: None,
        proof: Some(ProofLabel {
            status,
            mate_distance: proof.root.mate_distance,
            horizon_plies: 3,
            solver_mode: ProofMode::CheckingOnly,
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
