use anyhow::Result;
use shogi_core::{Color, Piece, PieceKind, Position, Square};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use rand::prelude::*;
use rand_distr::Distribution;


const KPP_DIM: usize = 200_000_000;

#[derive(Default)]
pub struct SparseModel {
    pub w: HashMap<usize, f32>,
    pub eta: f32,
}

impl SparseModel {
    pub fn new(eta: f32) -> Self {
        Self {
            w: HashMap::new(),
            eta,
        }
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line?;
            let mut parts = line.split(',');
            if let (Some(k), Some(v)) = (parts.next(), parts.next()) {
                let k: usize = k.parse()?;
                let v: f32 = v.parse()?;
                self.w.insert(k, v);
            }
        }
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path)?;
        for (&k, &v) in &self.w {
            writeln!(file, "{},{}", k, v)?;
        }
        Ok(())
    }

    pub fn initialize_random(&mut self, count: usize, stddev: f32) {
        let mut rng = rand::thread_rng();
        let dist = rand_distr::Normal::new(0.0, stddev).unwrap();
        for _ in 0..count {
            let i = rng.gen_range(0..KPP_DIM);
            let v = dist.sample(&mut rng) as f32;
            self.w.insert(i, v);
        }
    }

    pub fn predict(&self, x: &[usize]) -> f32 {
        x.iter().map(|&i| *self.w.get(&i).unwrap_or(&0.0)).sum()
    }

    pub fn update_batch(&mut self, batch: &[(Vec<usize>, f32)], batch_index: usize) -> f32 {
        let m = batch.len() as f32;
        if m == 0.0 {
            return 0.0;
        }
        let mut total_loss = 0.0;
        for (x, y_true) in batch.iter() {
            let y_pred = self.predict(x);
            let error = y_pred - y_true;
            total_loss += error * error;
            for &i in x {
                let w_i = self.w.entry(i).or_insert(0.0);
                *w_i -= self.eta * error / m;
            }
        }
        let mse = total_loss / m;
        println!("バッチ {}: 平均二乗誤差 = {:.6}", batch_index, mse);
        mse
    }
}

pub fn encode_piece(piece: Piece, sq: Option<Square>, hand_index: usize) -> usize {
    let base = match sq {
        Some(sq) => sq.index() as usize,
        None => 81 + hand_index,
    };
    let kind = piece.piece_kind() as u8 as usize - 1;
    let owner = if piece.color() == Color::Black { 0 } else { 1 };
    owner * 1500 + kind * 100 + base
}

pub fn extract_kpp_features(pos: &Position) -> Vec<usize> {
    let mut pieces = vec![];

    let king_sq = if let Some(king_sq_val) = (0..81).find_map(|i| {
        let file = (i % 9) as u8 + 1;
        let rank = (i / 9) as u8 + 1;
        let sq = Square::new(file, rank).unwrap();
        pos.piece_at(sq).and_then(|p| {
            if p.piece_kind() == PieceKind::King && p.color() == Color::Black {
                Some(sq.index() as usize)
            } else {
                None
            }
        })
    }) {
        king_sq_val
    } else {
        return vec![];
    };

    for i in 0..81 {
        let file = (i % 9) as u8 + 1;
        let rank = (i / 9) as u8 + 1;
        let sq = Square::new(file, rank).unwrap();
        if let Some(piece) = pos.piece_at(sq) {
            pieces.push(encode_piece(piece, Some(sq), 0));
        }
    }

    for color in [Color::Black, Color::White] {
        for kind in PieceKind::all().iter().copied() {
            let count = pos.hand_of_a_player(color).count(kind).unwrap_or(0);
            for i in 0..count {
                pieces.push(encode_piece(Piece::new(kind, color), None, i as usize));
            }
        }
    }

    let mut indices = vec![];

    for i in 0..pieces.len() {
        for j in (i + 1)..pieces.len() {
            // 2つの駒の順序を固定して組み合わせの重複をなくす
            let (p1, p2) = if pieces[i] < pieces[j] { (pieces[i], pieces[j]) } else { (pieces[j], pieces[i]) };

            // 注意: このインデックス計算はKPP_DIMを超える可能性があります。
            // pieceのエンコード値の最大値(P_MAX)を考慮すると、
            // king_sq * P_MAX * P_MAX + p1 * P_MAX + p2 は巨大な数になります。
            // インデックスの設計を見直す必要があります。
            let idx = (king_sq * 1500 * 1500 + p1 * 1500 + p2) % KPP_DIM;
            indices.push(idx);
        }
    }

    indices
}
