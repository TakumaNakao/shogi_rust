use super::codec::{read_f32_le, read_f32_vec, read_u32_le};
use super::constants::{NNUE_NUM_FEATURES, NNUE_NUM_KING_BUCKETS};
use super::features::extract_nnue_features;
use anyhow::{anyhow, Result};
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TinyNnueModel {
    pub hidden: usize,
    pub target_scale: f32,
    feature_emb: Vec<f32>,
    king_emb: Vec<f32>,
    material_w: Vec<f32>,
    hidden_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: f32,
}

impl TinyNnueModel {
    pub fn load(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != b"TNNUE001" {
            return Err(anyhow!("invalid tiny NNUE magic"));
        }

        let version = read_u32_le(&mut file)? as usize;
        if version != 1 {
            return Err(anyhow!("unsupported tiny NNUE version: {version}"));
        }
        let hidden = read_u32_le(&mut file)? as usize;
        let num_features = read_u32_le(&mut file)? as usize;
        let num_king_buckets = read_u32_le(&mut file)? as usize;
        let target_scale = read_f32_le(&mut file)?;
        if hidden == 0
            || num_features != NNUE_NUM_FEATURES
            || num_king_buckets != NNUE_NUM_KING_BUCKETS
            || !target_scale.is_finite()
            || target_scale <= 0.0
        {
            return Err(anyhow!("invalid tiny NNUE header"));
        }

        let feature_emb = read_f32_vec(&mut file, num_features * hidden)?;
        let king_emb = read_f32_vec(&mut file, num_king_buckets * hidden)?;
        let material_w = read_f32_vec(&mut file, hidden)?;
        let hidden_b = read_f32_vec(&mut file, hidden)?;
        let out_w = read_f32_vec(&mut file, hidden)?;
        let out_b = read_f32_le(&mut file)?;

        let mut trailing = [0u8; 1];
        match file.read(&mut trailing)? {
            0 => {}
            _ => return Err(anyhow!("trailing bytes in tiny NNUE file")),
        }

        Ok(Self {
            hidden,
            target_scale,
            feature_emb,
            king_emb,
            material_w,
            hidden_b,
            out_w,
            out_b,
        })
    }

    pub fn predict_from_position(&self, pos: &shogi_lib::Position) -> f32 {
        let Some(nnue) = extract_nnue_features(pos) else {
            return 0.0;
        };
        let mut hidden = self.hidden_b.clone();
        let king_base = nnue.king_bucket * self.hidden;
        for h in 0..self.hidden {
            hidden[h] += self.king_emb[king_base + h] + self.material_w[h] * nnue.material;
        }
        for feature in nnue.features {
            let feature_base = feature * self.hidden;
            for h in 0..self.hidden {
                hidden[h] += self.feature_emb[feature_base + h];
            }
        }

        let mut score = self.out_b;
        for (value, weight) in hidden.iter().zip(&self.out_w) {
            score += value.clamp(0.0, 1.0) * weight;
        }
        score * self.target_scale
    }
}
