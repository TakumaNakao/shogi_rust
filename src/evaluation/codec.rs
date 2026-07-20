use super::{HALFKP_HIDDEN, HALFKP_INPUTS};
use anyhow::{anyhow, Result};
use std::fs::File;
use std::io::Read;

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
