//! Shared conventions for game-record ingestion and training-data partitioning.
//!
//! These definitions are deliberately independent of individual trainer binaries:
//! a position's phase, a CSA result, and a game's split must have one meaning
//! throughout the data pipeline.

use anyhow::{anyhow, Result};
use glob::glob;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use shogi_core::{Color, Move, Piece, PieceKind, Square};
use shogi_lib::Position;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

pub const PHASE_POLICY_VERSION: u32 = 1;
pub const SPLIT_POLICY_VERSION: u32 = 2;
pub const OPENING_MAX_PLY: usize = 40;
pub const MIDDLEGAME_MAX_PLY: usize = 90;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingPhase {
    Opening,
    Middlegame,
    Endgame,
}

impl TrainingPhase {
    pub const fn for_ply(ply: usize) -> Self {
        if ply <= OPENING_MAX_PLY {
            Self::Opening
        } else if ply <= MIDDLEGAME_MAX_PLY {
            Self::Middlegame
        } else {
            Self::Endgame
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Opening => "opening",
            Self::Middlegame => "middlegame",
            Self::Endgame => "endgame",
        }
    }

    pub const fn code(self) -> u8 {
        match self {
            Self::Opening => 0,
            Self::Middlegame => 1,
            Self::Endgame => 2,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "opening" => Some(Self::Opening),
            "middle" | "middlegame" => Some(Self::Middlegame),
            "late" | "endgame" => Some(Self::Endgame),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetSplit {
    Train,
    Valid,
    Test,
}

impl DatasetSplit {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Valid => "valid",
            Self::Test => "test",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CsaMetadata {
    pub black_rate: Option<i32>,
    pub white_rate: Option<i32>,
    pub winner: Option<Color>,
    pub result_known: bool,
    pub termination: CsaTermination,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CsaTermination {
    Decisive,
    IllegalAction,
    Draw,
    #[default]
    Unknown,
}

impl CsaTermination {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Decisive => "decisive",
            Self::IllegalAction => "illegal_action",
            Self::Draw => "draw",
            Self::Unknown => "unknown",
        }
    }
}

impl CsaMetadata {
    pub const fn player_rate(self, color: Color) -> Option<i32> {
        match color {
            Color::Black => self.black_rate,
            Color::White => self.white_rate,
        }
    }
}

pub fn csa_color(color: csa::Color) -> Color {
    if color == csa::Color::Black {
        Color::Black
    } else {
        Color::White
    }
}

pub fn csa_piece_kind(piece_type: csa::PieceType) -> PieceKind {
    match piece_type {
        csa::PieceType::Pawn => PieceKind::Pawn,
        csa::PieceType::Lance => PieceKind::Lance,
        csa::PieceType::Knight => PieceKind::Knight,
        csa::PieceType::Silver => PieceKind::Silver,
        csa::PieceType::Gold => PieceKind::Gold,
        csa::PieceType::Bishop => PieceKind::Bishop,
        csa::PieceType::Rook => PieceKind::Rook,
        csa::PieceType::King => PieceKind::King,
        csa::PieceType::ProPawn => PieceKind::ProPawn,
        csa::PieceType::ProLance => PieceKind::ProLance,
        csa::PieceType::ProKnight => PieceKind::ProKnight,
        csa::PieceType::ProSilver => PieceKind::ProSilver,
        csa::PieceType::Horse => PieceKind::ProBishop,
        csa::PieceType::Dragon => PieceKind::ProRook,
        csa::PieceType::All => unreachable!("CSA All is not a move piece"),
    }
}

pub fn parse_csa_move(position: &Position, action: &csa::Action) -> Option<Move> {
    let csa::Action::Move(color, from, to, piece_after) = action else {
        return None;
    };
    let to = Square::new(to.file, to.rank)?;
    if from.file == 0 && from.rank == 0 {
        Some(Move::Drop {
            piece: Piece::new(csa_piece_kind(*piece_after), csa_color(*color)),
            to,
        })
    } else {
        let from = Square::new(from.file, from.rank)?;
        let piece_before = position.piece_at(from)?;
        Some(Move::Normal {
            from,
            to,
            promote: piece_before.piece_kind() != csa_piece_kind(*piece_after),
        })
    }
}

pub fn parse_csa_metadata(text: &str, record: &csa::GameRecord) -> CsaMetadata {
    let mut metadata = CsaMetadata::default();
    for line in text.lines() {
        if let Some(rate) = parse_rate_line(line, "'black_rate:") {
            metadata.black_rate = Some(rate);
        } else if let Some(rate) = parse_rate_line(line, "'white_rate:") {
            metadata.white_rate = Some(rate);
        }
    }
    let (winner, result_known, termination) = infer_csa_outcome(record);
    metadata.winner = winner;
    metadata.result_known = result_known;
    metadata.termination = termination;
    metadata
}

pub fn infer_csa_outcome(record: &csa::GameRecord) -> (Option<Color>, bool, CsaTermination) {
    let mut last_mover = None;
    for move_record in &record.moves {
        match move_record.action {
            csa::Action::Move(color, ..) => last_mover = Some(csa_color(color)),
            csa::Action::Toryo
            | csa::Action::TimeUp
            | csa::Action::IllegalMove
            | csa::Action::Tsumi
            | csa::Action::Kachi => {
                return (last_mover, last_mover.is_some(), CsaTermination::Decisive);
            }
            csa::Action::IllegalAction(color) => {
                return (
                    Some(csa_color(color).flip()),
                    true,
                    CsaTermination::IllegalAction,
                );
            }
            csa::Action::Sennichite | csa::Action::Jishogi | csa::Action::Hikiwake => {
                return (None, true, CsaTermination::Draw);
            }
            csa::Action::Chudan | csa::Action::Matta | csa::Action::Fuzumi | csa::Action::Error => {
                return (None, false, CsaTermination::Unknown)
            }
        }
    }
    (None, false, CsaTermination::Unknown)
}

pub fn collect_csa_files(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for input in inputs {
        if input.is_dir() {
            let pattern = input.join("**/*.csa");
            let pattern = pattern
                .to_str()
                .ok_or_else(|| anyhow!("path is not valid UTF-8: {}", input.display()))?;
            for entry in glob(pattern)? {
                files.push(entry?);
            }
        } else if is_csa_file(input) {
            files.push(input.clone());
        } else {
            return Err(anyhow!(
                "input path is neither a directory nor a CSA file: {}",
                input.display()
            ));
        }
    }
    files.sort();
    files.dedup();
    if files.is_empty() {
        return Err(anyhow!("no CSA files found"));
    }
    Ok(files)
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn game_content_id(bytes: &[u8]) -> String {
    sha256_hex(bytes)
}

pub fn split_for_game_content(
    game_id: &str,
    seed: u64,
    valid_percent: u8,
    test_percent: u8,
) -> DatasetSplit {
    let mut hasher = Sha256::new();
    hasher.update(b"shogi-ai-split-v2\0");
    hasher.update(seed.to_le_bytes());
    hasher.update(game_id.as_bytes());
    let digest = hasher.finalize();
    let bucket = u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 prefix")) % 100;
    if bucket < u64::from(test_percent) {
        DatasetSplit::Test
    } else if bucket < u64::from(test_percent + valid_percent) {
        DatasetSplit::Valid
    } else {
        DatasetSplit::Train
    }
}

fn parse_rate_line(line: &str, prefix: &str) -> Option<i32> {
    line.strip_prefix(prefix)
        .and_then(|rest| rest.rsplit(':').next())
        .and_then(|rate| rate.parse::<f64>().ok())
        .map(|rate| rate.round() as i32)
}

fn is_csa_file(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("csa"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_boundaries_are_canonical() {
        assert_eq!(TrainingPhase::Opening, TrainingPhase::for_ply(40));
        assert_eq!(TrainingPhase::Middlegame, TrainingPhase::for_ply(41));
        assert_eq!(TrainingPhase::Middlegame, TrainingPhase::for_ply(90));
        assert_eq!(TrainingPhase::Endgame, TrainingPhase::for_ply(91));
        assert_eq!(
            Some(TrainingPhase::Middlegame),
            TrainingPhase::parse("middle")
        );
    }

    #[test]
    fn content_id_and_split_ignore_file_path() {
        let bytes = b"V2.2\nN+black\nN-white\n%CHUDAN\n";
        let id_a = game_content_id(bytes);
        let id_b = game_content_id(bytes);
        assert_eq!(id_a, id_b);
        assert_eq!(64, id_a.len());
        assert_eq!(
            split_for_game_content(&id_a, 9601, 5, 5),
            split_for_game_content(&id_b, 9601, 5, 5)
        );
    }

    #[test]
    fn known_draw_is_distinct_from_unknown_result() {
        let draw = csa::parse_csa("V2.2\nPI\n+\n%SENNICHITE\n").expect("draw CSA");
        assert_eq!((None, true, CsaTermination::Draw), infer_csa_outcome(&draw));
        let unknown = csa::parse_csa("V2.2\nPI\n+\n%CHUDAN\n").expect("unknown CSA");
        assert_eq!(
            (None, false, CsaTermination::Unknown),
            infer_csa_outcome(&unknown)
        );
    }
}
