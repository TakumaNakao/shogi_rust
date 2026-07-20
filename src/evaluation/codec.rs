use super::{HALFKP_HIDDEN, HALFKP_INPUTS, HALFKP_KING_BUCKETS, HALFKP_PIECE_STATES};
use anyhow::{anyhow, Result};
use std::fs::File;
use std::io::{Read, Write};

pub const HALFKP_HEADER_LEN: usize = 32;
pub const HALFKP_VERSION: u32 = 1;
pub const HALFKP_MAGIC: &[u8; 8] = if cfg!(feature = "halfkp64") {
    b"HKP00064"
} else {
    b"HKP00001"
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HalfKpHeader {
    pub target_scale: f32,
}

#[derive(Debug, Clone)]
pub struct HalfKpFlatModel {
    pub header: HalfKpHeader,
    pub feature_emb: Vec<f32>,
    pub hidden_b: [f32; HALFKP_HIDDEN],
    pub out_w: [f32; HALFKP_HIDDEN * 2 + 1],
    pub out_b: f32,
}

impl HalfKpHeader {
    pub fn current(target_scale: f32) -> Result<Self> {
        let header = Self { target_scale };
        header.validate()?;
        Ok(header)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let bytes = bytes
            .get(..HALFKP_HEADER_LEN)
            .ok_or_else(|| anyhow!("truncated HalfKP header"))?;
        if &bytes[..8] != HALFKP_MAGIC {
            return Err(anyhow!("invalid HalfKP magic"));
        }
        let u32_at = |offset: usize| {
            u32::from_le_bytes(
                bytes[offset..offset + 4]
                    .try_into()
                    .expect("validated HalfKP header width"),
            )
        };
        if u32_at(8) != HALFKP_VERSION
            || u32_at(12) as usize != HALFKP_HIDDEN
            || u32_at(16) as usize != HALFKP_INPUTS
            || u32_at(20) as usize != HALFKP_KING_BUCKETS
            || u32_at(24) as usize != HALFKP_PIECE_STATES
        {
            return Err(anyhow!("invalid HalfKP header"));
        }
        let header = Self {
            target_scale: f32::from_le_bytes(
                bytes[28..32]
                    .try_into()
                    .expect("validated HalfKP header width"),
            ),
        };
        header.validate()?;
        Ok(header)
    }

    pub fn encode(self) -> Result<[u8; HALFKP_HEADER_LEN]> {
        self.validate()?;
        let mut bytes = [0u8; HALFKP_HEADER_LEN];
        bytes[..8].copy_from_slice(HALFKP_MAGIC);
        bytes[8..12].copy_from_slice(&HALFKP_VERSION.to_le_bytes());
        bytes[12..16].copy_from_slice(&(HALFKP_HIDDEN as u32).to_le_bytes());
        bytes[16..20].copy_from_slice(&(HALFKP_INPUTS as u32).to_le_bytes());
        bytes[20..24].copy_from_slice(&(HALFKP_KING_BUCKETS as u32).to_le_bytes());
        bytes[24..28].copy_from_slice(&(HALFKP_PIECE_STATES as u32).to_le_bytes());
        bytes[28..32].copy_from_slice(&self.target_scale.to_le_bytes());
        Ok(bytes)
    }

    pub fn write_to(self, writer: &mut impl Write) -> Result<()> {
        writer.write_all(&self.encode()?)?;
        Ok(())
    }

    fn validate(self) -> Result<()> {
        if !self.target_scale.is_finite() || self.target_scale <= 0.0 {
            return Err(anyhow!("invalid HalfKP header"));
        }
        Ok(())
    }
}

impl HalfKpFlatModel {
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let header = HalfKpHeader::decode(bytes)?;
        let expected_len = Self::encoded_len()?;
        if bytes.len() != expected_len {
            if bytes.len() > expected_len {
                return Err(anyhow!("trailing bytes in HalfKP file"));
            }
            return Err(anyhow!(
                "invalid HalfKP model length: got {}, expected {expected_len}",
                bytes.len()
            ));
        }
        let mut values = bytes[HALFKP_HEADER_LEN..]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        let feature_emb = values
            .by_ref()
            .take(HALFKP_INPUTS * HALFKP_HIDDEN)
            .collect();
        let hidden_b = std::array::from_fn(|_| {
            values
                .next()
                .expect("validated HalfKP model contains hidden bias")
        });
        let out_w = std::array::from_fn(|_| {
            values
                .next()
                .expect("validated HalfKP model contains output weights")
        });
        let out_b = values
            .next()
            .expect("validated HalfKP model contains output bias");
        debug_assert!(values.next().is_none());
        Ok(Self {
            header,
            feature_emb,
            hidden_b,
            out_w,
            out_b,
        })
    }

    pub fn write_to(&self, writer: &mut impl Write) -> Result<()> {
        Self::write_parts(
            writer,
            self.header,
            &self.feature_emb,
            &self.hidden_b,
            &self.out_w,
            self.out_b,
        )
    }

    pub fn write_parts(
        writer: &mut impl Write,
        header: HalfKpHeader,
        feature_emb: &[f32],
        hidden_b: &[f32; HALFKP_HIDDEN],
        out_w: &[f32; HALFKP_HIDDEN * 2 + 1],
        out_b: f32,
    ) -> Result<()> {
        if feature_emb.len() != HALFKP_INPUTS * HALFKP_HIDDEN {
            return Err(anyhow!("invalid HalfKP feature tensor length"));
        }
        header.write_to(writer)?;
        for &value in feature_emb {
            writer.write_all(&value.to_le_bytes())?;
        }
        for &value in hidden_b {
            writer.write_all(&value.to_le_bytes())?;
        }
        for &value in out_w {
            writer.write_all(&value.to_le_bytes())?;
        }
        writer.write_all(&out_b.to_le_bytes())?;
        Ok(())
    }

    pub fn encoded_len() -> Result<usize> {
        let float_count = HALFKP_INPUTS
            .checked_mul(HALFKP_HIDDEN)
            .and_then(|count| count.checked_add(HALFKP_HIDDEN))
            .and_then(|count| count.checked_add(HALFKP_HIDDEN * 2 + 1))
            .and_then(|count| count.checked_add(1))
            .ok_or_else(|| anyhow!("HalfKP model length overflow"))?;
        float_count
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|length| length.checked_add(HALFKP_HEADER_LEN))
            .ok_or_else(|| anyhow!("HalfKP model byte length overflow"))
    }
}

/// One aligned feature-transformer row. Keeping the hidden width in the type
/// lets release builds eliminate per-element bounds checks and keeps rows
/// suitable for the portable and AVX2 backends.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub(super) struct HalfKpFeatureRow(pub(super) [f32; HALFKP_HIDDEN]);

pub(super) fn read_u32_le(file: &mut File) -> Result<u32> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

pub(super) fn read_f32_le(file: &mut File) -> Result<f32> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

pub(super) fn read_f32_vec(file: &mut File, len: usize) -> Result<Vec<f32>> {
    let byte_len = len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow!("f32 vector byte length overflow"))?;
    let mut bytes = vec![0u8; byte_len];
    file.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

pub(super) fn read_f32_array<const N: usize>(file: &mut File) -> Result<[f32; N]> {
    read_f32_vec(file, N)?
        .try_into()
        .map_err(|_| anyhow!("invalid f32 array length"))
}

pub(super) fn feature_rows_from_flat(values: Vec<f32>) -> Result<Box<[HalfKpFeatureRow]>> {
    if values.len() != HALFKP_INPUTS * HALFKP_HIDDEN {
        return Err(anyhow!("invalid HalfKP feature row length"));
    }
    let rows = values
        .chunks_exact(HALFKP_HIDDEN)
        .map(|chunk| {
            let row: [f32; HALFKP_HIDDEN] = chunk
                .try_into()
                .expect("chunks_exact guarantees HalfKP row width");
            HalfKpFeatureRow(row)
        })
        .collect::<Vec<_>>();
    Ok(rows.into_boxed_slice())
}
