use anyhow::{anyhow, Context, Result};
use shogi_core::Color;
use shogi_lib::Position;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::evaluation::{extract_halfkp_features_for, HALFKP_HIDDEN, HALFKP_INPUTS};

const DATASET_MAGIC: &[u8; 8] = b"HKST0002";
const DATASET_VERSION: u32 = 2;
const RECORD_MAGIC: u32 = 0x3152_4b48;

pub const CANDIDATE_SEARCH_BEST: u8 = 1;
pub const CANDIDATE_GAME_MOVE: u8 = 2;
pub const CANDIDATE_RANDOM: u8 = 4;
pub const CANDIDATE_TACTICAL: u8 = 8;

#[derive(Clone, Debug)]
pub struct PackedHalfKpPosition {
    pub side_to_move: Color,
    pub features_black: Vec<u32>,
    pub features_white: Vec<u32>,
    pub material_black: f32,
    pub material_white: f32,
}

impl PackedHalfKpPosition {
    pub fn from_position(position: &Position) -> Option<Self> {
        let black = extract_halfkp_features_for(position, Color::Black)?;
        let white = extract_halfkp_features_for(position, Color::White)?;
        Some(Self {
            side_to_move: position.side_to_move(),
            features_black: black
                .features
                .into_iter()
                .map(|value| value as u32)
                .collect(),
            features_white: white
                .features
                .into_iter()
                .map(|value| value as u32)
                .collect(),
            material_black: black.material,
            material_white: white.material,
        })
    }
}

#[derive(Clone, Debug)]
pub struct SearchTeacherCandidate {
    pub flags: u8,
    pub score_cp: f32,
    pub child: PackedHalfKpPosition,
}

#[derive(Clone, Debug)]
pub struct SearchTeacherRecord {
    pub position_hash: u64,
    pub ply: u16,
    pub phase: u8,
    pub teacher_depth: u8,
    pub result: Option<f32>,
    pub root_search_score_cp: f32,
    pub sample_weight: f32,
    pub root: PackedHalfKpPosition,
    pub candidates: Vec<SearchTeacherCandidate>,
}

pub struct SearchTeacherWriter {
    writer: BufWriter<File>,
    records: u64,
}

impl SearchTeacherWriter {
    pub fn create(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        let mut writer = BufWriter::new(
            File::create(path).with_context(|| format!("create {}", path.display()))?,
        );
        writer.write_all(DATASET_MAGIC)?;
        writer.write_all(&DATASET_VERSION.to_le_bytes())?;
        writer.write_all(&(HALFKP_HIDDEN as u32).to_le_bytes())?;
        writer.write_all(&(HALFKP_INPUTS as u32).to_le_bytes())?;
        Ok(Self { writer, records: 0 })
    }

    pub fn write_record(&mut self, record: &SearchTeacherRecord) -> Result<()> {
        if record.candidates.is_empty() || record.candidates.len() > u8::MAX as usize {
            return Err(anyhow!("invalid candidate count"));
        }
        if !record.root_search_score_cp.is_finite()
            || !record.sample_weight.is_finite()
            || record.sample_weight <= 0.0
        {
            return Err(anyhow!("invalid teacher record scalar"));
        }
        self.writer.write_all(&RECORD_MAGIC.to_le_bytes())?;
        self.writer.write_all(&record.position_hash.to_le_bytes())?;
        self.writer.write_all(&record.ply.to_le_bytes())?;
        self.writer.write_all(&[record.phase])?;
        self.writer.write_all(&[encode_result(record.result)?])?;
        self.writer
            .write_all(&[record.candidates.len() as u8, record.teacher_depth])?;
        self.writer
            .write_all(&record.root_search_score_cp.to_le_bytes())?;
        self.writer.write_all(&record.sample_weight.to_le_bytes())?;
        write_position(&mut self.writer, &record.root)?;
        for candidate in &record.candidates {
            if !candidate.score_cp.is_finite() {
                return Err(anyhow!("non-finite candidate score"));
            }
            self.writer.write_all(&[candidate.flags, 0, 0, 0])?;
            self.writer.write_all(&candidate.score_cp.to_le_bytes())?;
            write_position(&mut self.writer, &candidate.child)?;
        }
        self.records += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<u64> {
        self.writer.flush()?;
        Ok(self.records)
    }
}

pub struct SearchTeacherReader {
    reader: BufReader<File>,
}

impl SearchTeacherReader {
    pub fn open(path: &Path) -> Result<Self> {
        let mut reader =
            BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        let version = read_u32(&mut reader)?;
        let hidden = read_u32(&mut reader)? as usize;
        let inputs = read_u32(&mut reader)? as usize;
        if &magic != DATASET_MAGIC
            || version != DATASET_VERSION
            || hidden != HALFKP_HIDDEN
            || inputs != HALFKP_INPUTS
        {
            return Err(anyhow!("incompatible HalfKP search teacher dataset"));
        }
        Ok(Self { reader })
    }

    pub fn read_record(&mut self) -> Result<Option<SearchTeacherRecord>> {
        let Some(record_magic) = read_u32_or_eof(&mut self.reader)? else {
            return Ok(None);
        };
        if record_magic != RECORD_MAGIC {
            return Err(anyhow!("invalid packed record marker"));
        }
        let position_hash = read_u64(&mut self.reader)?;
        let ply = read_u16(&mut self.reader)?;
        let phase = read_u8(&mut self.reader)?;
        let result = decode_result(read_u8(&mut self.reader)?)?;
        let candidate_count = read_u8(&mut self.reader)? as usize;
        let teacher_depth = read_u8(&mut self.reader)?;
        let root_search_score_cp = read_f32(&mut self.reader)?;
        let sample_weight = read_f32(&mut self.reader)?;
        let root = read_position(&mut self.reader)?;
        let mut candidates = Vec::with_capacity(candidate_count);
        for _ in 0..candidate_count {
            let flags = read_u8(&mut self.reader)?;
            let mut reserved = [0u8; 3];
            self.reader.read_exact(&mut reserved)?;
            let score_cp = read_f32(&mut self.reader)?;
            let child = read_position(&mut self.reader)?;
            candidates.push(SearchTeacherCandidate {
                flags,
                score_cp,
                child,
            });
        }
        if candidate_count == 0
            || !root_search_score_cp.is_finite()
            || !sample_weight.is_finite()
            || sample_weight <= 0.0
            || candidates
                .iter()
                .any(|candidate| !candidate.score_cp.is_finite())
        {
            return Err(anyhow!("invalid packed teacher record"));
        }
        Ok(Some(SearchTeacherRecord {
            position_hash,
            ply,
            phase,
            teacher_depth,
            result,
            root_search_score_cp,
            sample_weight,
            root,
            candidates,
        }))
    }
}

fn write_position(writer: &mut impl Write, position: &PackedHalfKpPosition) -> Result<()> {
    if position.features_black.len() > u8::MAX as usize
        || position.features_white.len() > u8::MAX as usize
    {
        return Err(anyhow!("too many HalfKP active features"));
    }
    writer.write_all(&[
        u8::from(position.side_to_move == Color::White),
        position.features_black.len() as u8,
        position.features_white.len() as u8,
        0,
    ])?;
    writer.write_all(&position.material_black.to_le_bytes())?;
    writer.write_all(&position.material_white.to_le_bytes())?;
    for &feature in position
        .features_black
        .iter()
        .chain(position.features_white.iter())
    {
        if feature as usize >= HALFKP_INPUTS {
            return Err(anyhow!("HalfKP feature out of range"));
        }
        writer.write_all(&feature.to_le_bytes())?;
    }
    Ok(())
}

fn read_position(reader: &mut impl Read) -> Result<PackedHalfKpPosition> {
    let side = read_u8(reader)?;
    let black_len = read_u8(reader)? as usize;
    let white_len = read_u8(reader)? as usize;
    let _reserved = read_u8(reader)?;
    let material_black = read_f32(reader)?;
    let material_white = read_f32(reader)?;
    let mut features_black = Vec::with_capacity(black_len);
    let mut features_white = Vec::with_capacity(white_len);
    for index in 0..black_len + white_len {
        let feature = read_u32(reader)?;
        if feature as usize >= HALFKP_INPUTS {
            return Err(anyhow!("packed HalfKP feature out of range"));
        }
        if index < black_len {
            features_black.push(feature);
        } else {
            features_white.push(feature);
        }
    }
    Ok(PackedHalfKpPosition {
        side_to_move: if side == 0 {
            Color::Black
        } else if side == 1 {
            Color::White
        } else {
            return Err(anyhow!("invalid packed side to move"));
        },
        features_black,
        features_white,
        material_black,
        material_white,
    })
}

fn encode_result(result: Option<f32>) -> Result<u8> {
    match result {
        None => Ok(255),
        Some(0.0) => Ok(0),
        Some(0.5) => Ok(1),
        Some(1.0) => Ok(2),
        Some(_) => Err(anyhow!("result must be loss, draw, win, or unknown")),
    }
}

fn decode_result(value: u8) -> Result<Option<f32>> {
    match value {
        0 => Ok(Some(0.0)),
        1 => Ok(Some(0.5)),
        2 => Ok(Some(1.0)),
        255 => Ok(None),
        _ => Err(anyhow!("invalid packed result")),
    }
}

fn read_u8(reader: &mut impl Read) -> Result<u8> {
    let mut bytes = [0u8; 1];
    reader.read_exact(&mut bytes)?;
    Ok(bytes[0])
}

fn read_u16(reader: &mut impl Read) -> Result<u16> {
    let mut bytes = [0u8; 2];
    reader.read_exact(&mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u32_or_eof(reader: &mut impl Read) -> Result<Option<u32>> {
    let mut bytes = [0u8; 4];
    let mut offset = 0;
    while offset < bytes.len() {
        let read = reader.read(&mut bytes[offset..])?;
        if read == 0 {
            return if offset == 0 {
                Ok(None)
            } else {
                Err(anyhow!("truncated packed record marker"))
            };
        }
        offset += read;
    }
    Ok(Some(u32::from_le_bytes(bytes)))
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32(reader: &mut impl Read) -> Result<f32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position_hash::PositionHasher;

    #[test]
    fn packed_position_extracts_valid_halfkp_features() {
        let position = Position::default();
        let packed = PackedHalfKpPosition::from_position(&position).expect("start position");
        assert_eq!(Color::Black, packed.side_to_move);
        assert!(!packed.features_black.is_empty());
        assert!(!packed.features_white.is_empty());
        assert!(packed
            .features_black
            .iter()
            .chain(&packed.features_white)
            .all(|&feature| (feature as usize) < HALFKP_INPUTS));
        assert_ne!(0, PositionHasher::calculate_hash(&position));
    }

    #[test]
    fn teacher_dataset_round_trips() {
        let position = Position::default();
        let packed = PackedHalfKpPosition::from_position(&position).expect("start position");
        let record = SearchTeacherRecord {
            position_hash: PositionHasher::calculate_hash(&position),
            ply: 12,
            phase: 1,
            teacher_depth: 4,
            result: Some(0.5),
            root_search_score_cp: 42.25,
            sample_weight: 0.75,
            root: packed.clone(),
            candidates: vec![SearchTeacherCandidate {
                flags: CANDIDATE_SEARCH_BEST | CANDIDATE_GAME_MOVE,
                score_cp: 37.5,
                child: packed,
            }],
        };
        let path = std::env::temp_dir().join(format!(
            "halfkp-search-teacher-{}-{}.hkst",
            std::process::id(),
            record.position_hash
        ));
        let mut writer = SearchTeacherWriter::create(&path).expect("create dataset");
        writer.write_record(&record).expect("write record");
        assert_eq!(1, writer.finish().expect("finish dataset"));

        let mut reader = SearchTeacherReader::open(&path).expect("open dataset");
        let decoded = reader
            .read_record()
            .expect("read record")
            .expect("record exists");
        assert_eq!(record.position_hash, decoded.position_hash);
        assert_eq!(record.ply, decoded.ply);
        assert_eq!(record.result, decoded.result);
        assert_eq!(record.root.features_black, decoded.root.features_black);
        assert_eq!(record.candidates[0].flags, decoded.candidates[0].flags);
        assert_eq!(
            record.candidates[0].score_cp,
            decoded.candidates[0].score_cp
        );
        assert!(reader.read_record().expect("read eof").is_none());
        std::fs::remove_file(path).expect("remove dataset");
    }
}
