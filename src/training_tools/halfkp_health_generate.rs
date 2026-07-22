use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use shogi_ai::evaluation::HALFKP_PIECE_STATES;
use shogi_ai::halfkp_training::{
    CurriculumHeader, CurriculumPreference, CurriculumReader, CurriculumRecord, CurriculumWriter,
    LabelSource, OpeningStratum, PackedHalfKpPosition, PreferenceKind,
};
use shogi_ai::training_data::{capture_run_environment, sha256_file, DatasetSplit, TrainingPhase};
use shogi_core::{Move, PartialPosition, PieceKind};
use shogi_lib::Position;
use shogi_usi_parser::FromUsi;
use std::collections::HashSet;
use std::path::PathBuf;

const LINEAGE_ID: &str = "halfkp64-scratch-v1";
const KING_BUCKETS: usize = 45;
const PHASES: usize = 3;

#[derive(Parser, Debug)]
#[command(about = "Generate stratified evaluator-free legal health positions")]
struct Args {
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum)]
    split: SplitArg,
    #[arg(long)]
    records: usize,
    #[arg(long)]
    seed: u64,
    #[arg(long, default_value_t = 8)]
    minimum_ply: usize,
    #[arg(long, default_value_t = 400)]
    maximum_ply: usize,
    #[arg(long, default_value_t = 32)]
    king_move_weight: u32,
    #[arg(long, default_value_t = 4)]
    quiet_move_weight: u32,
    #[arg(long, default_value_t = 200)]
    maximum_attempts_per_record: usize,
    #[arg(long, default_value_t = true)]
    require_all_king_buckets: bool,
    #[arg(long)]
    exclude: Vec<PathBuf>,
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
    if args.records == 0
        || args.minimum_ply >= args.maximum_ply
        || args.king_move_weight == 0
        || args.quiet_move_weight == 0
    {
        return Err(anyhow!("invalid generation limits"));
    }
    let (mut seen_inputs, mut seen_families) = load_exclusions(&args.exclude)?;
    let header = CurriculumHeader {
        lineage_id: LINEAGE_ID.into(),
        feature_profile: "halfkp64-health-stratified-v1".into(),
        producer_revision: capture_run_environment()
            .git_revision
            .unwrap_or_else(|| "unknown".into()),
        seed: args.seed,
        input_sha256: args
            .exclude
            .iter()
            .map(|path| sha256_file(path))
            .collect::<Result<Vec<_>>>()?,
        generator_config: serde_json::json!({
            "generator": "legal_health_stratified_v1",
            "records": args.records,
            "minimum_ply": args.minimum_ply,
            "maximum_ply": args.maximum_ply,
            "king_move_weight": args.king_move_weight,
            "quiet_move_weight": args.quiet_move_weight,
            "require_all_king_buckets": args.require_all_king_buckets,
        }),
        teacher_model_sha256: None,
        teacher_lineage_id: None,
    };
    let mut writer = CurriculumWriter::create(&args.output, header)?;
    let mut rng = ChaCha20Rng::seed_from_u64(args.seed);
    let mut position = Position::default();
    let mut game_ply = 0usize;
    let mut bucket_counts = [[0usize; KING_BUCKETS]; 2];
    let mut phase_counts = [0usize; PHASES];
    let bucket_quota = (args.records / (KING_BUCKETS * 2)).max(1);
    let phase_quota = (args.records / (PHASES * 2)).max(1);
    let max_attempts = args
        .records
        .saturating_mul(args.maximum_attempts_per_record);
    let mut attempts = 0usize;
    let mut written = 0usize;

    while written < args.records && attempts < max_attempts {
        attempts += 1;
        let legal = position.legal_moves();
        if legal.is_empty() || game_ply >= args.maximum_ply {
            position = Position::default();
            game_ply = 0;
            continue;
        }
        let mv = choose_move(&position, &legal, &mut rng, &args, &bucket_counts);
        position.do_move(mv);
        game_ply += 1;
        if game_ply < args.minimum_ply {
            continue;
        }
        let root = PackedHalfKpPosition::from_position(&position)
            .ok_or_else(|| anyhow!("legal position lacks a king"))?;
        let network_input_hash = root.network_input_hash();
        if seen_inputs.contains(&network_input_hash) {
            continue;
        }
        let black_bucket = king_bucket(&root.features_black)?;
        let white_bucket = king_bucket(&root.features_white)?;
        let phase = TrainingPhase::for_ply(position.ply() as usize).code() as usize;
        let needs_stratum = bucket_counts[0][black_bucket] < bucket_quota
            || bucket_counts[1][white_bucket] < bucket_quota
            || phase_counts[phase.min(PHASES - 1)] < phase_quota;
        if !needs_stratum && !rng.gen_ratio(1, 8) {
            continue;
        }
        let mirrored = mirror_position(&position)?;
        let mirror = PackedHalfKpPosition::from_position(&mirrored)
            .ok_or_else(|| anyhow!("mirrored position lacks a king"))?;
        let family_hash = symmetry_family_hash(&root, &mirror);
        let mirror_hash = mirror.network_input_hash();
        if seen_inputs.contains(&mirror_hash) || seen_families.contains(&family_hash) {
            continue;
        }
        writer.write_record(&CurriculumRecord {
            source: LabelSource::SymmetryAnchor,
            group_id: u64::from_le_bytes(family_hash[..8].try_into().expect("hash width")),
            position_hash: u64::from_le_bytes(network_input_hash[..8].try_into().expect("hash")),
            family_hash,
            network_input_hash,
            root_sfen: position.to_sfen_owned(),
            split: args.split.dataset(),
            ply: position.ply(),
            phase: phase as u8,
            opening_stratum: OpeningStratum::UNKNOWN,
            confidence: 1.0,
            sample_weight: 1.0,
            root: root.clone(),
            result: None,
            proof: None,
            preferences: vec![CurriculumPreference {
                good_child: root,
                bad_child: mirror,
                kind: PreferenceKind::SymmetryEqual,
                margin_class: 0,
                good_move_usi: None,
                bad_move_usi: None,
            }],
        })?;
        seen_inputs.insert(network_input_hash);
        seen_inputs.insert(mirror_hash);
        seen_families.insert(family_hash);
        bucket_counts[0][black_bucket] += 1;
        bucket_counts[1][white_bucket] += 1;
        phase_counts[phase.min(PHASES - 1)] += 1;
        written += 1;
    }
    if written != args.records {
        return Err(anyhow!(
            "generation exhausted after {attempts} attempts at {written} records"
        ));
    }
    let black_covered = bucket_counts[0].iter().filter(|&&count| count > 0).count();
    let white_covered = bucket_counts[1].iter().filter(|&&count| count > 0).count();
    if args.require_all_king_buckets
        && (black_covered != KING_BUCKETS || white_covered != KING_BUCKETS)
    {
        return Err(anyhow!(
            "king bucket coverage incomplete: black={black_covered}/45 white={white_covered}/45"
        ));
    }
    if phase_counts.iter().any(|&count| count < phase_quota) {
        return Err(anyhow!(
            "phase coverage incomplete: counts={phase_counts:?} minimum={phase_quota}"
        ));
    }
    writer.finish()?;
    println!(
        "complete records={written} attempts={attempts} black_buckets={black_covered}/45 white_buckets={white_covered}/45 phases={phase_counts:?}"
    );
    Ok(())
}

fn load_exclusions(paths: &[PathBuf]) -> Result<(HashSet<[u8; 32]>, HashSet<[u8; 32]>)> {
    let mut inputs = HashSet::new();
    let mut families = HashSet::new();
    for path in paths {
        let mut reader = CurriculumReader::open(path)?;
        while let Some(record) = reader.read_record()? {
            inputs.insert(record.network_input_hash);
            families.insert(record.family_hash);
            for position in record
                .preferences
                .iter()
                .flat_map(|preference| [&preference.good_child, &preference.bad_child])
            {
                inputs.insert(position.network_input_hash());
            }
        }
    }
    Ok((inputs, families))
}

fn choose_move(
    position: &Position,
    moves: &[Move],
    rng: &mut ChaCha20Rng,
    args: &Args,
    bucket_counts: &[[usize; KING_BUCKETS]; 2],
) -> Move {
    let side_index = usize::from(position.side_to_move() != shogi_core::Color::Black);
    let weights = moves
        .iter()
        .map(|&mv| {
            let king = match mv {
                Move::Normal { from, .. } => position
                    .piece_at(from)
                    .is_some_and(|piece| piece.piece_kind() == PieceKind::King),
                Move::Drop { .. } => false,
            };
            let quiet = position.piece_at(mv.to()).is_none();
            let uncovered_multiplier = if king {
                let mut child = position.clone();
                child.do_move(mv);
                PackedHalfKpPosition::from_position(&child)
                    .and_then(|packed| {
                        let features = if side_index == 0 {
                            &packed.features_black
                        } else {
                            &packed.features_white
                        };
                        king_bucket(features).ok()
                    })
                    .map_or(1, |bucket| {
                        if bucket_counts[side_index][bucket] == 0 {
                            128
                        } else {
                            1
                        }
                    })
            } else {
                1
            };
            (if king { args.king_move_weight } else { 1 })
                * if quiet { args.quiet_move_weight } else { 1 }
                * uncovered_multiplier
        })
        .collect::<Vec<_>>();
    let mut pick = rng.gen_range(0..weights.iter().map(|&weight| weight as u64).sum::<u64>());
    for (&mv, &weight) in moves.iter().zip(&weights) {
        if pick < weight as u64 {
            return mv;
        }
        pick -= weight as u64;
    }
    moves[0]
}

fn king_bucket(features: &[u32]) -> Result<usize> {
    features
        .first()
        .map(|&feature| feature as usize / HALFKP_PIECE_STATES)
        .filter(|&bucket| bucket < KING_BUCKETS)
        .ok_or_else(|| anyhow!("missing king bucket"))
}

fn symmetry_family_hash(a: &PackedHalfKpPosition, b: &PackedHalfKpPosition) -> [u8; 32] {
    let mut hashes = [a.network_input_hash(), b.network_input_hash()];
    hashes.sort_unstable();
    let mut hash = Sha256::new();
    hash.update(b"halfkp-health-symmetry-family-v1\0");
    hash.update(hashes[0]);
    hash.update(hashes[1]);
    hash.finalize().into()
}

fn mirror_position(position: &Position) -> Result<Position> {
    let sfen = position.to_sfen_owned();
    let mut fields = sfen.split_whitespace();
    let board = fields.next().ok_or_else(|| anyhow!("missing SFEN board"))?;
    let side = fields.next().ok_or_else(|| anyhow!("missing SFEN side"))?;
    let hands = fields.next().ok_or_else(|| anyhow!("missing SFEN hands"))?;
    let ply = fields.next().ok_or_else(|| anyhow!("missing SFEN ply"))?;
    let mirrored = board
        .split('/')
        .map(mirror_rank)
        .collect::<Result<Vec<_>>>()?
        .join("/");
    let partial = PartialPosition::from_usi(&format!("sfen {mirrored} {side} {hands} {ply}"))
        .map_err(|error| anyhow!("parse mirrored SFEN: {error:?}"))?;
    Ok(Position::new(partial))
}

fn mirror_rank(rank: &str) -> Result<String> {
    let mut cells = Vec::new();
    let mut chars = rank.chars();
    while let Some(ch) = chars.next() {
        if ch.is_ascii_digit() {
            cells.extend(std::iter::repeat_n(
                String::new(),
                ch.to_digit(10).unwrap() as usize,
            ));
        } else if ch == '+' {
            cells.push(format!(
                "+{}",
                chars.next().ok_or_else(|| anyhow!("dangling promotion"))?
            ));
        } else {
            cells.push(ch.to_string());
        }
    }
    if cells.len() != 9 {
        return Err(anyhow!("invalid SFEN rank"));
    }
    cells.reverse();
    let mut output = String::new();
    let mut empty = 0usize;
    for cell in cells {
        if cell.is_empty() {
            empty += 1;
        } else {
            if empty > 0 {
                output.push(char::from_digit(empty as u32, 10).unwrap());
                empty = 0;
            }
            output.push_str(&cell);
        }
    }
    if empty > 0 {
        output.push(char::from_digit(empty as u32, 10).unwrap());
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_move_selection_is_deterministic() {
        let position = Position::default();
        let moves = position.legal_moves();
        let args = Args::parse_from([
            "test",
            "--output",
            "x",
            "--split",
            "valid",
            "--records",
            "1",
            "--seed",
            "7",
        ]);
        let mut a = ChaCha20Rng::seed_from_u64(9);
        let mut b = ChaCha20Rng::seed_from_u64(9);
        assert_eq!(
            choose_move(&position, &moves, &mut a, &args, &[[0; KING_BUCKETS]; 2],),
            choose_move(&position, &moves, &mut b, &args, &[[0; KING_BUCKETS]; 2],)
        );
    }

    #[test]
    fn health_mirror_is_an_involution() {
        let position = Position::default();
        let mirrored = mirror_position(&position).expect("mirror");
        let restored = mirror_position(&mirrored).expect("restore");
        assert_eq!(position.to_sfen_owned(), restored.to_sfen_owned());
    }
}
