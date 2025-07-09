use anyhow::Result;

use shogi_core::{Color, Piece, PieceKind, Position, Square, Move};
use csa;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::fs::File;
use std::io::BufReader;
use std::io::BufRead;
use rand::prelude::*;
use rand_distr::Distribution;

const KPP_DIM: usize = 200_000_000;

const BATCH_SIZE: usize = 32;

#[derive(Default)]
struct SparseModel {
    w: HashMap<usize, f32>,
    eta: f32,
}

impl SparseModel {
    fn new(eta: f32) -> Self {
        Self {
            w: HashMap::new(),
            eta,
        }
    }

    fn load(&mut self, path: &Path) -> Result<()> {
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

    fn save(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path)?;
        for (&k, &v) in &self.w {
            writeln!(file, "{},{}", k, v)?;
        }
        Ok(())
    }

    fn initialize_random(&mut self, count: usize, stddev: f32) {
        let mut rng = rand::thread_rng();
        let dist = rand_distr::Normal::new(0.0, stddev).unwrap();
        for _ in 0..count {
            let i = rng.gen_range(0..KPP_DIM);
            let v = dist.sample(&mut rng) as f32;
            self.w.insert(i, v);
        }
    }

    fn predict(&self, x: &[usize]) -> f32 {
        x.iter().map(|&i| *self.w.get(&i).unwrap_or(&0.0)).sum()
    }

    fn update_batch(&mut self, batch: &[(Vec<usize>, f32)], batch_index: usize) {
        let m = batch.len() as f32;
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
        println!("バッチ {}: 平均二乗誤差 = {:.6}", batch_index, total_loss / m);
    }
}

fn encode_piece(piece: Piece, sq: Option<Square>, hand_index: usize) -> usize {
    let base = match sq {
        Some(sq) => sq.index() as usize,
        None => 81 + hand_index,
    };
    let kind = piece.piece_kind() as u8 as usize;
    let owner = if piece.color() == Color::Black { 0 } else { 1 };
    owner * 750 + kind * 81 + base
}

fn extract_kpp_features(pos: &Position) -> Vec<usize> {
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
            let idx = king_sq * 1500 * 1500 + pieces[i] * 1500 + pieces[j];
            indices.push(idx);
        }
    }

    indices
}

fn load_csa_dataset(dir: &Path) -> Result<Vec<(Vec<usize>, f32)>> {
    let mut data = vec![];
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().map(|e| e == "csa").unwrap_or(false) {
            let text = fs::read_to_string(&path)?;
            let record = csa::parse_csa(&text)?;

            let mut pos = Position::default();
            for mv in &record.moves {
                let shogi_move = match &mv.action {
                    csa::Action::Move(_color, from_csa, to_csa, piece_type_after_csa) => {
                        let from_sq = if let Some(sq) = Square::new(from_csa.file, from_csa.rank) { sq } else { continue; };
                        let to_sq = if let Some(sq) = Square::new(to_csa.file, to_csa.rank) { sq } else { continue; };
                        let piece_before = if let Some(p) = pos.piece_at(from_sq) { p } else { continue; };
                        let promote = piece_before.piece_kind() as u8 != *piece_type_after_csa as u8;
                        Move::Normal { from: from_sq, to: to_sq, promote }
                    },
                    _ => continue, // Skip non-move actions for now
                };

                let features = extract_kpp_features(&pos);
                let label = match &mv.action {
                    csa::Action::Toryo => 0.0,
                    csa::Action::TimeUp => 0.0,
                    csa::Action::IllegalMove => 0.0,
                    _ => 0.0,
                };
                data.push((features, label));
                let _ = pos.make_move(shogi_move);
            }
        }
    }
    Ok(data)
}

fn main() -> Result<()> {
    println!("使用する年を入力してください（例: 2017）: ");
    let mut year = String::new();
    io::stdin().read_line(&mut year)?;
    let year = year.trim();

    let data_dir_str = format!("./csa_files/{}", year);
    let data_dir = Path::new(&data_dir_str);
    let weight_path = Path::new("./weights.csv");

    let dataset = load_csa_dataset(data_dir)?;
    println!("局面数: {}", dataset.len());

    let mut model = SparseModel::new(0.01);

    if weight_path.exists() {
        model.load(weight_path)?;
        println!("重みファイルを読み込みました。");
    } else {
        println!("重みファイルが存在しません。初期化中...");
        model.initialize_random(50_000, 0.01);
        model.save(weight_path)?;
        println!("初期重みを保存しました。");
    }

    for (batch_index, batch) in dataset.chunks(BATCH_SIZE).enumerate() {
        model.update_batch(batch, batch_index);
    }

    println!("学習完了。重み数: {}", model.w.len());
    model.save(weight_path)?;
    println!("重みを保存しました: {:?}", weight_path);
    Ok(())
}
