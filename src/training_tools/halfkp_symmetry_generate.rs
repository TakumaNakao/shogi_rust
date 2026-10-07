use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use shogi_ai::halfkp_training::{
    CurriculumHeader, CurriculumPreference, CurriculumRecord, CurriculumWriter, LabelSource,
    OpeningStratum, PackedHalfKpPosition, PreferenceKind,
};
use shogi_ai::training_data::{capture_run_environment, DatasetSplit, TrainingPhase};
use shogi_core::PartialPosition;
use shogi_lib::Position;
use shogi_usi_parser::FromUsi;
use std::collections::HashSet;
use std::path::PathBuf;

const LINEAGE_ID: &str = "halfkp64-scratch-v1";

#[derive(Parser, Debug)]
#[command(about = "Generate evaluator-free left-right symmetry anchors")]
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
    if args.records == 0 {
        return Err(anyhow!("records must be non-zero"));
    }
    let header = CurriculumHeader {
        lineage_id: LINEAGE_ID.into(),
        feature_profile: "halfkp64-mirror-file-v1".into(),
        producer_revision: capture_run_environment()
            .git_revision
            .unwrap_or_else(|| "unknown".into()),
        seed: args.seed,
        input_sha256: vec![],
        generator_config: serde_json::json!({
            "generator": "legal_random_play_lr_symmetry_v1",
            "stream_split": format!("{:?}", args.split).to_ascii_lowercase(),
            "minimum_ply": args.minimum_ply,
            "records": args.records,
        }),
        teacher_model_sha256: None,
        teacher_lineage_id: None,
    };
    let mut writer = CurriculumWriter::create(&args.output, header)?;
    let mut rng = ChaCha20Rng::seed_from_u64(args.seed);
    let mut position = Position::default();
    let mut game_ply = 0usize;
    let mut written = 0usize;
    let mut seen = HashSet::new();
    while written < args.records {
        let legal = position.legal_moves();
        if legal.is_empty() || game_ply >= 300 {
            position = Position::default();
            game_ply = 0;
            continue;
        }
        position.do_move(legal[rng.gen_range(0..legal.len())]);
        game_ply += 1;
        if game_ply < args.minimum_ply {
            continue;
        }
        let root = PackedHalfKpPosition::from_position(&position)
            .ok_or_else(|| anyhow!("legal position lacks a king"))?;
        let network_input_hash = root.network_input_hash();
        if !seen.insert(network_input_hash) {
            continue;
        }
        let mirrored = mirror_position(&position)?;
        let mirror = PackedHalfKpPosition::from_position(&mirrored)
            .ok_or_else(|| anyhow!("mirrored position lacks a king"))?;
        let family_hash = symmetry_family_hash(&root, &mirror);
        writer.write_record(&CurriculumRecord {
            source: LabelSource::SymmetryAnchor,
            group_id: u64::from_le_bytes(family_hash[..8].try_into().expect("hash width")),
            position_hash: u64::from_le_bytes(
                network_input_hash[..8].try_into().expect("hash width"),
            ),
            family_hash,
            network_input_hash,
            root_sfen: position.to_sfen_owned(),
            split: args.split.dataset(),
            ply: position.ply(),
            phase: TrainingPhase::for_ply(position.ply() as usize).code(),
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
        written += 1;
    }
    writer.finish()?;
    println!("complete records={written} split={:?}", args.split);
    Ok(())
}

fn symmetry_family_hash(a: &PackedHalfKpPosition, b: &PackedHalfKpPosition) -> [u8; 32] {
    let mut hashes = [a.network_input_hash(), b.network_input_hash()];
    hashes.sort_unstable();
    let mut hash = Sha256::new();
    hash.update(b"halfkp-lr-symmetry-family-v1\0");
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
            let piece = chars
                .next()
                .ok_or_else(|| anyhow!("dangling promotion in SFEN"))?;
            cells.push(format!("+{piece}"));
        } else {
            cells.push(ch.to_string());
        }
    }
    if cells.len() != 9 {
        return Err(anyhow!("SFEN rank does not contain nine squares"));
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
    fn mirror_is_an_involution_and_produces_a_distinct_anchor() {
        let position = Position::default();
        let mirrored = mirror_position(&position).expect("mirror");
        let restored = mirror_position(&mirrored).expect("restore");
        assert_eq!(restored.to_sfen_owned(), position.to_sfen_owned());
        assert_ne!(
            PackedHalfKpPosition::from_position(&position)
                .expect("features")
                .network_input_hash(),
            PackedHalfKpPosition::from_position(&mirrored)
                .expect("mirrored features")
                .network_input_hash()
        );
    }
}
