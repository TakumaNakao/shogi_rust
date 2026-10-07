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
use shogi_ai::training_data::sha256_file;
use shogi_ai::training_data::{capture_run_environment, DatasetSplit, TrainingPhase};
use shogi_core::{Move, ToUsi};
use shogi_lib::Position;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

const LINEAGE_ID: &str = "halfkp64-scratch-v1";
const RULE_VERSION: u32 = 3;
const MAX_GOOD_MOVES: usize = 8;

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
    /// Defender-to-move roots with both a refutation and a mate-allowing reply.
    #[arg(long, default_value_t = 0)]
    defense_pairs: usize,
    #[arg(long, default_value_t = 2_000_000)]
    max_positions: usize,
    #[arg(long, default_value_t = 100_000)]
    node_limit: u64,
    #[arg(long, default_value_t = 20260721)]
    seed: u64,
    #[arg(long, default_value_t = false)]
    require_rank_pair: bool,
    #[arg(long, default_value_t = false)]
    one_record_per_family: bool,
    /// HKCL datasets whose root and preference inputs must not appear here.
    #[arg(long)]
    exclude_dataset: Vec<PathBuf>,
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
    if args.mate1 + args.mate3 + args.no_mate3 + args.defense_pairs == 0 || args.node_limit == 0 {
        return Err(anyhow!(
            "at least one target and node-limit must be non-zero"
        ));
    }
    if !args.exclude_dataset.is_empty() && !args.one_record_per_family && !args.require_rank_pair {
        return Err(anyhow!(
            "exclusion-aware generation requires one-record-per-family"
        ));
    }
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let revision = capture_run_environment()
        .git_revision
        .unwrap_or_else(|| "unknown".to_string());
    let excluded = load_excluded_inputs(&args.exclude_dataset)?;
    let header = CurriculumHeader {
        lineage_id: LINEAGE_ID.to_string(),
        feature_profile: "halfkp64-mirror-file-v1".to_string(),
        producer_revision: revision,
        seed: args.seed,
        input_sha256: args
            .exclude_dataset
            .iter()
            .map(|path| sha256_file(path))
            .collect::<Result<_>>()?,
        generator_config: serde_json::json!({
            "generator": "legal_random_play_bounded_mate_v3",
            "stream_split": format!("{:?}", args.split).to_ascii_lowercase(),
            "mate_plies": [1, 3],
            "node_limit": args.node_limit,
            "rule_version": RULE_VERSION,
            "attack_move_mode": "checking_only",
            "targets": {"mate1": args.mate1, "mate3": args.mate3, "no_mate3": args.no_mate3, "defense_pairs": args.defense_pairs},
            "require_rank_pair": args.require_rank_pair,
            "one_record_per_family": args.one_record_per_family || args.require_rank_pair,
            "excluded_network_inputs": excluded.len(),
        }),
        teacher_model_sha256: None,
        teacher_lineage_id: None,
    };
    let mut writer = CurriculumWriter::create(&args.output, header)?;
    let mut rng = ChaCha20Rng::seed_from_u64(args.seed);
    let mut position = Position::default();
    let mut game_ply = 0usize;
    let mut seen_inputs = HashSet::new();
    let mut accepted_inputs = HashSet::new();
    let mut game_index = 0u64;
    let mut family = family_hash(args.seed, game_index);
    let mut mate1 = 0usize;
    let mut mate3 = 0usize;
    let mut no_mate3 = 0usize;
    let mut defense_pairs = 0usize;
    let mut examined = 0usize;
    let mut resource_limits = 0usize;
    while examined < args.max_positions
        && (mate1 < args.mate1
            || mate3 < args.mate3
            || no_mate3 < args.no_mate3
            || defense_pairs < args.defense_pairs)
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
            let mut accepted_record = false;
            if category_available {
                let record = make_record(
                    &position,
                    family,
                    args.split.dataset(),
                    &proof,
                    args.node_limit,
                )?;
                if args.require_rank_pair && record.preferences.is_empty() {
                    continue;
                }
                let candidate_inputs = record_network_inputs(&record).collect::<Vec<_>>();
                if candidate_inputs
                    .iter()
                    .any(|hash| excluded.contains(hash) || accepted_inputs.contains(hash))
                {
                    continue;
                }
                writer.write_record(&record)?;
                accepted_inputs.extend(candidate_inputs);
                match (proof.root.status, proof.root.mate_distance) {
                    (MateProofStatus::ProvenMate, Some(1)) => mate1 += 1,
                    (MateProofStatus::ProvenMate, Some(3)) => mate3 += 1,
                    (MateProofStatus::ProvenNoMate, None) => no_mate3 += 1,
                    _ => unreachable!(),
                }
                accepted_record = true;
            } else if defense_pairs < args.defense_pairs
                && proof.root.status == MateProofStatus::ProvenNoMate
            {
                if let Some(record) =
                    make_defense_record(&position, family, args.split.dataset(), args.node_limit)?
                {
                    let candidate_inputs = record_network_inputs(&record).collect::<Vec<_>>();
                    if !candidate_inputs
                        .iter()
                        .any(|hash| excluded.contains(hash) || accepted_inputs.contains(hash))
                    {
                        writer.write_record(&record)?;
                        accepted_inputs.extend(candidate_inputs);
                        defense_pairs += 1;
                        accepted_record = true;
                    }
                }
            }
            if accepted_record && (args.require_rank_pair || args.one_record_per_family) {
                position = Position::default();
                game_ply = 0;
                game_index += 1;
                family = family_hash(args.seed, game_index);
                continue;
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
            println!("examined={examined} mate1={mate1} mate3={mate3} no_mate3={no_mate3} defense={defense_pairs} resource={resource_limits}");
        }
    }
    let records = writer.finish()?;
    println!("complete examined={examined} mate1={mate1} mate3={mate3} no_mate3={no_mate3} defense={defense_pairs} resource={resource_limits} records={records}");
    if mate1 < args.mate1
        || mate3 < args.mate3
        || no_mate3 < args.no_mate3
        || defense_pairs < args.defense_pairs
    {
        return Err(anyhow!("category targets not reached before max-positions"));
    }
    // Re-open all outputs so the declared count and trailing-byte checks run.
    verify_dataset(&args.output)?;
    Ok(())
}

fn load_excluded_inputs(paths: &[PathBuf]) -> Result<HashSet<[u8; 32]>> {
    let mut excluded = HashSet::new();
    for path in paths {
        let mut reader = CurriculumReader::open(path)?;
        while let Some(record) = reader.read_record()? {
            excluded.extend(record_network_inputs(&record));
        }
    }
    Ok(excluded)
}

fn record_network_inputs(record: &CurriculumRecord) -> impl Iterator<Item = [u8; 32]> + '_ {
    std::iter::once(record.network_input_hash).chain(
        record
            .preferences
            .iter()
            .flat_map(|preference| [&preference.good_child, &preference.bad_child])
            .map(PackedHalfKpPosition::network_input_hash),
    )
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
        position_hash: u64::from_le_bytes(
            root.network_input_hash()[..8]
                .try_into()
                .expect("hash width"),
        ),
        family_hash,
        network_input_hash: root.network_input_hash(),
        root_sfen: position.to_sfen_owned(),
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
            attacker: proof.attacker,
            status,
            mate_distance: proof.root.mate_distance,
            horizon_plies: 3,
            solver_mode: ProofMode::CheckingOnly,
            max_nodes: node_limit,
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
    if proof.root.status != MateProofStatus::ProvenMate
        || proof
            .root
            .mate_distance
            .is_none_or(|distance| distance <= 1)
    {
        return Ok(vec![]);
    }
    let classified = classify_children(
        position,
        proof.attacker,
        proof.max_plies.saturating_sub(1),
        node_limit,
        proof.mode,
    )?;
    let good = select_good_moves(
        classified
            .iter()
            .filter(|child| {
                child.status == MateProofStatus::ProvenMate && position.is_check_move(child.mv)
            })
            .cloned(),
    );
    let bad = select_stratified_moves(
        position,
        classified
            .into_iter()
            .filter(|child| child.status == MateProofStatus::ProvenNoMate),
    );
    Ok(pair_preferences(&good, &bad, PreferenceKind::ProvenOutcome))
}

fn make_defense_record(
    position: &Position,
    family_hash: [u8; 32],
    split: DatasetSplit,
    node_limit: u64,
) -> Result<Option<CurriculumRecord>> {
    let attacker = position.side_to_move();
    for attack_move in position.legal_moves() {
        if !position.is_check_move(attack_move) {
            continue;
        }
        let mut defense_root = position.clone();
        defense_root.do_move(attack_move);
        if defense_root.legal_moves().is_empty() {
            continue;
        }
        let proof = solve_bounded_mate_for(
            &defense_root,
            attacker,
            MateLimits {
                max_plies: 2,
                max_nodes: node_limit,
                mode: AttackMoveMode::CheckingOnly,
            },
        )?;
        if proof.root.status != MateProofStatus::ProvenNoMate {
            continue;
        }
        let classified = classify_children(
            &defense_root,
            attacker,
            1,
            node_limit,
            AttackMoveMode::CheckingOnly,
        )?;
        let good = select_stratified_moves(
            &defense_root,
            classified
                .iter()
                .filter(|child| child.status == MateProofStatus::ProvenNoMate)
                .cloned(),
        );
        let bad = select_stratified_moves(
            &defense_root,
            classified
                .into_iter()
                .filter(|child| child.status == MateProofStatus::ProvenMate),
        );
        let preferences = pair_preferences(&good, &bad, PreferenceKind::ProvenOutcome);
        if preferences.is_empty() {
            continue;
        }
        let root = PackedHalfKpPosition::from_position(&defense_root)
            .ok_or_else(|| anyhow!("defense root lacks a king"))?;
        return Ok(Some(CurriculumRecord {
            source: LabelSource::ExactMate,
            group_id: u64::from_le_bytes(family_hash[..8].try_into().expect("hash width")),
            position_hash: u64::from_le_bytes(
                root.network_input_hash()[..8]
                    .try_into()
                    .expect("hash width"),
            ),
            family_hash,
            network_input_hash: root.network_input_hash(),
            root_sfen: defense_root.to_sfen_owned(),
            split,
            ply: defense_root.ply(),
            phase: TrainingPhase::for_ply(defense_root.ply() as usize).code(),
            opening_stratum: OpeningStratum::UNKNOWN,
            confidence: 1.0,
            sample_weight: 1.0,
            root,
            result: None,
            proof: Some(ProofLabel {
                attacker,
                status: ProofStatus::NoForcedMateWithinHorizon,
                mate_distance: None,
                horizon_plies: 2,
                solver_mode: ProofMode::CheckingOnly,
                max_nodes: node_limit,
                nodes: proof.nodes,
                rule_version: RULE_VERSION,
            }),
            preferences,
        }));
    }
    Ok(None)
}

#[derive(Clone)]
struct ClassifiedChild {
    mv: Move,
    packed: PackedHalfKpPosition,
    status: MateProofStatus,
}

fn classify_children(
    position: &Position,
    attacker: shogi_core::Color,
    remaining: u16,
    node_limit: u64,
    mode: AttackMoveMode,
) -> Result<Vec<ClassifiedChild>> {
    let mut children = Vec::new();
    for mv in position.legal_moves() {
        let mut child = position.clone();
        child.do_move(mv);
        if child.legal_moves().is_empty() {
            continue;
        }
        let proof = solve_bounded_mate_for(
            &child,
            attacker,
            MateLimits {
                max_plies: remaining,
                max_nodes: node_limit,
                mode,
            },
        )?;
        if proof.root.status == MateProofStatus::ResourceLimit {
            continue;
        }
        if let Some(packed) = PackedHalfKpPosition::from_position(&child) {
            children.push(ClassifiedChild {
                mv,
                packed,
                status: proof.root.status,
            });
        }
    }
    Ok(children)
}

fn select_good_moves(children: impl Iterator<Item = ClassifiedChild>) -> Vec<ClassifiedChild> {
    let mut values = children.collect::<Vec<_>>();
    values.sort_by_key(|child| child.mv.to_usi_owned());
    values.truncate(MAX_GOOD_MOVES);
    values
}

fn select_stratified_moves(
    position: &Position,
    children: impl Iterator<Item = ClassifiedChild>,
) -> Vec<ClassifiedChild> {
    let mut strata = BTreeMap::<u8, ClassifiedChild>::new();
    for child in children {
        let key = move_category(position, child.mv);
        let replace = strata
            .get(&key)
            .is_none_or(|current| child.mv.to_usi_owned() < current.mv.to_usi_owned());
        if replace {
            strata.insert(key, child);
        }
    }
    strata.into_values().collect()
}

fn move_category(position: &Position, mv: Move) -> u8 {
    u8::from(position.is_check_move(mv))
        | (u8::from(position.piece_at(mv.to()).is_some()) << 1)
        | (u8::from(mv.is_promoting()) << 2)
        | (u8::from(mv.is_drop()) << 3)
}

fn pair_preferences(
    good: &[ClassifiedChild],
    bad: &[ClassifiedChild],
    kind: PreferenceKind,
) -> Vec<CurriculumPreference> {
    let count = good.len().max(bad.len());
    if good.is_empty() || bad.is_empty() {
        return vec![];
    }
    (0..count)
        .map(|index| {
            let good = &good[index % good.len()];
            let bad = &bad[index % bad.len()];
            CurriculumPreference {
                good_child: good.packed.clone(),
                bad_child: bad.packed.clone(),
                kind,
                margin_class: 0,
                good_move_usi: Some(good.mv.to_usi_owned()),
                bad_move_usi: Some(bad.mv.to_usi_owned()),
            }
        })
        .collect()
}

fn verify_dataset(path: &Path) -> Result<()> {
    let mut reader = CurriculumReader::open(path)?;
    while reader.read_record()?.is_some() {}
    Ok(())
}
