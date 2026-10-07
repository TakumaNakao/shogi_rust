use anyhow::{anyhow, Result};
use clap::Parser;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use shogi_ai::halfkp_training::{
    CurriculumHeader, CurriculumPreference, CurriculumReader, CurriculumRecord, CurriculumWriter,
    LabelSource, OpeningStratum, PackedHalfKpPosition, PreferenceKind,
};
use shogi_ai::training_data::{capture_run_environment, sha256_file, DatasetSplit, TrainingPhase};
use shogi_ai::utils::{format_move_usi, parse_usi_move_for_color, position_from_sfen_or_usi};
use shogi_core::Color;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

const LINEAGE_ID: &str = "halfkp64-scratch-v1";

#[derive(Parser, Debug)]
#[command(about = "Convert provenance-checked game JSONL into evaluator-free soft HKCL ranks")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    seed: u64,
    #[arg(long, default_value_t = 0.1)]
    sample_weight: f32,
    #[arg(long, default_value_t = 0.6)]
    confidence: f32,
    #[arg(long)]
    exclude: Vec<PathBuf>,
}

#[derive(Deserialize)]
struct InputRecord {
    schema: String,
    game_key: String,
    split: String,
    ply: usize,
    opening_stratum: u16,
    side_to_move: String,
    winner: Option<String>,
    result_known: bool,
    sfen: String,
    teacher_move: String,
    license_id: String,
    source_uri: String,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    if !(0.0..=1.0).contains(&args.sample_weight)
        || args.sample_weight == 0.0
        || !(0.0..=1.0).contains(&args.confidence)
    {
        return Err(anyhow!("invalid sample weight or confidence"));
    }
    let input_sha256 = args
        .input
        .iter()
        .chain(args.exclude.iter())
        .map(|path| sha256_file(path))
        .collect::<Result<Vec<_>>>()?;
    let header = CurriculumHeader {
        lineage_id: LINEAGE_ID.into(),
        feature_profile: "halfkp64-game-soft-v1".into(),
        producer_revision: capture_run_environment()
            .git_revision
            .unwrap_or_else(|| "unknown".into()),
        seed: args.seed,
        input_sha256,
        generator_config: serde_json::json!({
            "generator": "game_soft_expert_v1",
            "sample_weight": args.sample_weight,
            "confidence": args.confidence,
            "negative": "one deterministic quiet legal move preferred; no evaluator",
            "exclusions": args.exclude,
        }),
        teacher_model_sha256: None,
        teacher_lineage_id: None,
    };
    let mut writer = CurriculumWriter::create(&args.output, header)?;
    let excluded = load_excluded_inputs(&args.exclude)?;
    let mut written = 0usize;
    let mut skipped = 0usize;
    for input in &args.input {
        for line in BufReader::new(File::open(input)?).lines() {
            let record: InputRecord = serde_json::from_str(&line?)?;
            if record.schema != "rank_value_dataset_v3_position"
                || record.license_id.is_empty()
                || record.source_uri.is_empty()
            {
                return Err(anyhow!("invalid game provenance schema"));
            }
            let Some(position) = position_from_sfen_or_usi(&record.sfen) else {
                return Err(anyhow!("invalid game SFEN"));
            };
            let expected_side = if position.side_to_move() == Color::Black {
                "black"
            } else {
                "white"
            };
            if record.side_to_move != expected_side {
                return Err(anyhow!("side-to-move metadata mismatch"));
            }
            let teacher = parse_usi_move_for_color(&record.teacher_move, position.side_to_move())
                .filter(|mv| position.legal_moves().contains(mv))
                .ok_or_else(|| anyhow!("illegal teacher move"))?;
            let root = PackedHalfKpPosition::from_position(&position)
                .ok_or_else(|| anyhow!("game root lacks a king"))?;
            let mut alternatives = position
                .legal_moves()
                .into_iter()
                .filter(|&mv| mv != teacher)
                .collect::<Vec<_>>();
            alternatives.sort_by_key(|&mv| (!is_quiet(&position, mv), format_move_usi(mv)));
            if alternatives.is_empty() {
                skipped += 1;
                continue;
            }
            let quiet_count = alternatives
                .iter()
                .take_while(|&&mv| is_quiet(&position, mv))
                .count();
            let pool_length = if quiet_count > 0 {
                quiet_count
            } else {
                alternatives.len()
            };
            let index = deterministic_index(&root.network_input_hash(), args.seed, pool_length);
            let bad_move = alternatives[index];
            let mut good_child = position.clone();
            good_child.do_move(teacher);
            let mut bad_child = position.clone();
            bad_child.do_move(bad_move);
            let Some(good_child) = PackedHalfKpPosition::from_position(&good_child) else {
                skipped += 1;
                continue;
            };
            let Some(bad_child) = PackedHalfKpPosition::from_position(&bad_child) else {
                skipped += 1;
                continue;
            };
            if [
                root.network_input_hash(),
                good_child.network_input_hash(),
                bad_child.network_input_hash(),
            ]
            .iter()
            .any(|hash| excluded.contains(hash))
            {
                skipped += 1;
                continue;
            }
            let family_hash = hash_text(b"game-family-v1\0", &record.game_key);
            let split = parse_split(&record.split)?;
            let result = if record.result_known {
                record.winner.as_deref().map(|winner| {
                    let winner = match winner {
                        "black" => Color::Black,
                        "white" => Color::White,
                        _ => return f32::NAN,
                    };
                    if winner == position.side_to_move() {
                        1.0
                    } else {
                        0.0
                    }
                })
            } else {
                None
            };
            if result.is_some_and(|value| !value.is_finite()) {
                return Err(anyhow!("invalid winner metadata"));
            }
            writer.write_record(&CurriculumRecord {
                source: LabelSource::ExpertMove,
                group_id: u64::from_le_bytes(family_hash[..8].try_into().unwrap()),
                position_hash: u64::from_le_bytes(
                    root.network_input_hash()[..8].try_into().unwrap(),
                ),
                family_hash,
                network_input_hash: root.network_input_hash(),
                root_sfen: record.sfen,
                split,
                ply: record.ply.min(u16::MAX as usize) as u16,
                phase: TrainingPhase::for_ply(record.ply).code(),
                opening_stratum: OpeningStratum(record.opening_stratum),
                confidence: args.confidence,
                sample_weight: args.sample_weight,
                root,
                result,
                proof: None,
                preferences: vec![CurriculumPreference {
                    good_child,
                    bad_child,
                    kind: PreferenceKind::ExpertMove,
                    margin_class: 0,
                    good_move_usi: Some(format_move_usi(teacher)),
                    bad_move_usi: Some(format_move_usi(bad_move)),
                }],
            })?;
            written += 1;
        }
    }
    writer.finish()?;
    println!("complete written={written} skipped={skipped}");
    Ok(())
}

fn load_excluded_inputs(paths: &[PathBuf]) -> Result<HashSet<[u8; 32]>> {
    let mut inputs = HashSet::new();
    for path in paths {
        let mut reader = CurriculumReader::open(path)?;
        while let Some(record) = reader.read_record()? {
            inputs.insert(record.network_input_hash);
            for child in record
                .preferences
                .iter()
                .flat_map(|preference| [&preference.good_child, &preference.bad_child])
            {
                inputs.insert(child.network_input_hash());
            }
        }
    }
    Ok(inputs)
}

fn is_quiet(position: &shogi_lib::Position, mv: shogi_core::Move) -> bool {
    position.piece_at(mv.to()).is_none()
        && !mv.is_promoting()
        && !mv.is_drop()
        && !position.is_check_move(mv)
}

fn deterministic_index(hash: &[u8; 32], seed: u64, length: usize) -> usize {
    let value = u64::from_le_bytes(hash[..8].try_into().unwrap()) ^ seed;
    value as usize % length
}

fn hash_text(prefix: &[u8], text: &str) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(prefix);
    hash.update(text.as_bytes());
    hash.finalize().into()
}

fn parse_split(value: &str) -> Result<DatasetSplit> {
    match value {
        "train" => Ok(DatasetSplit::Train),
        "valid" => Ok(DatasetSplit::Valid),
        "test" => Ok(DatasetSplit::Test),
        _ => Err(anyhow!("unknown split")),
    }
}
