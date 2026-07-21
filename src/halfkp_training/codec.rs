use super::{PackedHalfKpPosition, SearchTeacherCandidate, SearchTeacherRecord};
use crate::evaluation::{HALFKP_HIDDEN, HALFKP_INPUTS};
use anyhow::{anyhow, Result};
use shogi_core::Color;
use std::io::{Read, Write};

const HKST_MAGIC: &[u8; 8] = b"HKST0002";
const HKST_VERSION: u32 = 2;
const HKST_RECORD_MAGIC: u32 = 0x3152_4b48;

pub(super) fn write_header(writer: &mut impl Write) -> Result<()> {
    writer.write_all(HKST_MAGIC)?;
    writer.write_all(&HKST_VERSION.to_le_bytes())?;
    writer.write_all(&(HALFKP_HIDDEN as u32).to_le_bytes())?;
    writer.write_all(&(HALFKP_INPUTS as u32).to_le_bytes())?;
    Ok(())
}

pub(super) fn read_header(reader: &mut impl Read) -> Result<()> {
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    let version = read_u32(reader)?;
    let hidden = read_u32(reader)? as usize;
    let inputs = read_u32(reader)? as usize;
    if &magic != HKST_MAGIC
        || version != HKST_VERSION
        || hidden != HALFKP_HIDDEN
        || inputs != HALFKP_INPUTS
    {
        return Err(anyhow!("incompatible HalfKP search teacher dataset"));
    }
    Ok(())
}

pub(super) fn write_record(writer: &mut impl Write, record: &SearchTeacherRecord) -> Result<()> {
    if record.candidates.is_empty() || record.candidates.len() > u8::MAX as usize {
        return Err(anyhow!("invalid candidate count"));
    }
    if !record.root_search_score_cp.is_finite()
        || !record.sample_weight.is_finite()
        || record.sample_weight <= 0.0
    {
        return Err(anyhow!("invalid teacher record scalar"));
    }
    writer.write_all(&HKST_RECORD_MAGIC.to_le_bytes())?;
    writer.write_all(&record.position_hash.to_le_bytes())?;
    writer.write_all(&record.ply.to_le_bytes())?;
    writer.write_all(&[record.phase])?;
    writer.write_all(&[encode_result(record.result)?])?;
    writer.write_all(&[record.candidates.len() as u8, record.teacher_depth])?;
    writer.write_all(&record.root_search_score_cp.to_le_bytes())?;
    writer.write_all(&record.sample_weight.to_le_bytes())?;
    write_position(writer, &record.root)?;
    for candidate in &record.candidates {
        if !candidate.score_cp.is_finite() {
            return Err(anyhow!("non-finite candidate score"));
        }
        writer.write_all(&[candidate.flags, 0, 0, 0])?;
        writer.write_all(&candidate.score_cp.to_le_bytes())?;
        write_position(writer, &candidate.child)?;
    }
    Ok(())
}

pub(super) fn read_record(reader: &mut impl Read) -> Result<Option<SearchTeacherRecord>> {
    let Some(record_magic) = read_u32_or_eof(reader)? else {
        return Ok(None);
    };
    if record_magic != HKST_RECORD_MAGIC {
        return Err(anyhow!("invalid packed record marker"));
    }
    let position_hash = read_u64(reader)?;
    let ply = read_u16(reader)?;
    let phase = read_u8(reader)?;
    let result = decode_result(read_u8(reader)?)?;
    let candidate_count = read_u8(reader)? as usize;
    let teacher_depth = read_u8(reader)?;
    let root_search_score_cp = read_f32(reader)?;
    let sample_weight = read_f32(reader)?;
    let root = read_position(reader)?;
    let mut candidates = Vec::with_capacity(candidate_count);
    for _ in 0..candidate_count {
        let flags = read_u8(reader)?;
        let mut reserved = [0u8; 3];
        reader.read_exact(&mut reserved)?;
        let score_cp = read_f32(reader)?;
        let child = read_position(reader)?;
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
