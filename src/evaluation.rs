use anyhow::Result;
use shogi_core::{Color, Piece, PieceKind, Position, Square, Move};
use std::collections::{HashMap, HashSet};
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

pub struct KppState {
    pub features: HashSet<usize>,
    pieces: Vec<usize>,
    king_sq: usize,
}

impl KppState {
    pub fn new(pos: &Position) -> Option<Self> {
        let king_sq = find_king_sq(pos, Color::Black)?;
        let pieces = get_all_encoded_pieces(pos);
        let features = Self::calculate_features(king_sq, &pieces).into_iter().collect();
        Some(Self {
            features,
            pieces,
            king_sq,
        })
    }

    pub fn update(&mut self, old_pos: &Position, mv: &Move) {
        let mut new_pos = old_pos.clone();
        if new_pos.make_move(*mv).is_none() {
            // 不正な手や、玉が取られる手などはここで弾かれ、状態は更新されない
            return;
        }

        // 玉が動いた場合は、新しい局面で状態を丸ごと再計算する
        if let Move::Normal { from, .. } = *mv {
            if let Some(piece) = old_pos.piece_at(from) {
                if piece.piece_kind() == PieceKind::King {
                    if let Some(new_state) = Self::new(&new_pos) {
                        *self = new_state;
                    }
                    return;
                }
            }
        }

        let mut removed_pieces = Vec::new();
        let mut added_pieces = Vec::new();

        match *mv {
            Move::Drop { piece, to } => {
                let hand_index = old_pos.hand_of_a_player(piece.color()).count(piece.piece_kind()).unwrap_or(1) - 1;
                removed_pieces.push(encode_piece(piece, None, hand_index as usize));
                added_pieces.push(encode_piece(piece, Some(to), 0));
            }
            Move::Normal { from, to, promote } => {
                let moved_piece = if let Some(p) = old_pos.piece_at(from) {
                    p
                } else {
                    // CSAデータが不正で、移動元に駒がない場合は更新をスキップ
                    return;
                };

                removed_pieces.push(encode_piece(moved_piece, Some(from), 0));
                let new_piece_kind = if promote {
                    if let Some(promoted_kind) = moved_piece.piece_kind().promote() {
                        promoted_kind
                    } else {
                        // 不正な成りなので、この手はスキップ
                        return;
                    }
                } else { moved_piece.piece_kind() };
                added_pieces.push(encode_piece(Piece::new(new_piece_kind, moved_piece.color()), Some(to), 0));

                if let Some(captured_piece) = old_pos.piece_at(to) {
                    removed_pieces.push(encode_piece(captured_piece, Some(to), 0));
                    let color = new_pos.side_to_move();
                    let hand_piece = if let Some(unpromoted_kind) = captured_piece.piece_kind().unpromote() {
                        match unpromoted_kind {
                            PieceKind::Pawn | PieceKind::Lance | PieceKind::Knight | PieceKind::Silver | PieceKind::Gold | PieceKind::Bishop | PieceKind::Rook => unpromoted_kind,
                            _ => {
                                eprintln!("Warning: Invalid unpromoted piece kind ({:?}). Skipping move.", unpromoted_kind);
                                return;
                            }
                        }
                    } else {
                        // 不正な不成なので、この手はスキップ
                        return;
                    };
                    let hand_index = if let Some(count) = new_pos.hand_of_a_player(color).count(hand_piece) {
                        count.saturating_sub(1) as usize
                    } else {
                        eprintln!("Warning: Captured piece kind ({:?}) cannot be held in hand. Skipping move.", hand_piece);
                        return;
                    };
                    added_pieces.push(encode_piece(Piece::new(hand_piece, color), None, hand_index as usize));
                }
            }
        }

        let removed_pieces_set: HashSet<_> = removed_pieces.iter().cloned().collect();
        let added_pieces_set: HashSet<_> = added_pieces.iter().cloned().collect();

        let common_pieces: Vec<usize> = self.pieces.iter().filter(|&p| !removed_pieces_set.contains(p)).cloned().collect();

        let removed_features = Self::calculate_delta_features(self.king_sq, &removed_pieces, &common_pieces);
        let added_features = Self::calculate_delta_features(self.king_sq, &added_pieces, &common_pieces);
        let inter_features = Self::calculate_inter_features(self.king_sq, &removed_pieces, &added_pieces);

        for f in removed_features {
            self.features.remove(&f);
        }
        for f in &inter_features.0 {
            self.features.remove(f);
        }
        for f in added_features {
            self.features.insert(f);
        }
        for f in &inter_features.1 {
            self.features.insert(*f);
        }
        
        let mut new_pieces_set: HashSet<_> = self.pieces.iter().cloned().collect();
        for p in removed_pieces {
            new_pieces_set.remove(&p);
        }
        for p in added_pieces {
            new_pieces_set.insert(p);
        }
        self.pieces = new_pieces_set.into_iter().collect();
    }

    fn calculate_features(king_sq: usize, pieces: &[usize]) -> Vec<usize> {
        let mut indices = Vec::new();
        for i in 0..pieces.len() {
            for j in (i + 1)..pieces.len() {
                indices.push(Self::get_feature_index(king_sq, pieces[i], pieces[j]));
            }
        }
        indices
    }
    
    fn calculate_delta_features(king_sq: usize, delta_pieces: &[usize], common_pieces: &[usize]) -> Vec<usize> {
        let mut features = Vec::new();
        // 差分セット内のペア
        for i in 0..delta_pieces.len() {
            for j in (i + 1)..delta_pieces.len() {
                features.push(Self::get_feature_index(king_sq, delta_pieces[i], delta_pieces[j]));
            }
        }
        // 差分セットと共通セットのペア
        for &p1 in delta_pieces {
            for &p2 in common_pieces {
                features.push(Self::get_feature_index(king_sq, p1, p2));
            }
        }
        features
    }

    fn calculate_inter_features(king_sq: usize, removed: &[usize], added: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let mut removed_inter = Vec::new();
        let mut added_inter = Vec::new();
        for &r_piece in removed {
            for &a_piece in added {
                // 仮の古い盤面での特徴（削除されるべき）
                removed_inter.push(Self::get_feature_index(king_sq, r_piece, a_piece));
                 // 仮の新しい盤面での特徴（追加されるべき）
                added_inter.push(Self::get_feature_index(king_sq, r_piece, a_piece));
            }
        }
        (removed_inter, added_inter)
    }


    fn get_feature_index(king_sq: usize, p1: usize, p2: usize) -> usize {
        let (p1, p2) = if p1 < p2 { (p1, p2) } else { (p2, p1) };
        (king_sq * 1500 * 1500 + p1 * 1500 + p2) % KPP_DIM
    }

    fn diff_pieces(old: &[usize], new: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let old_set: std::collections::HashSet<_> = old.iter().cloned().collect();
        let new_set: std::collections::HashSet<_> = new.iter().cloned().collect();
        let removed = old_set.difference(&new_set).cloned().collect();
        let added = new_set.difference(&old_set).cloned().collect();
        (removed, added)
    }
}

fn find_king_sq(pos: &Position, color: Color) -> Option<usize> {
    (0..81).find_map(|i| {
        let sq = if let Some(s) = Square::from_u8(i as u8) { s } else { return None; };
        pos.piece_at(sq).and_then(|p| {
            if p.piece_kind() == PieceKind::King && p.color() == color {
                Some(sq.index() as usize)
            } else {
                None
            }
        })
    })
}

fn get_all_encoded_pieces(pos: &Position) -> Vec<usize> {
    let mut pieces = vec![];
    for i in 0..81 {
        let sq = if let Some(s) = Square::from_u8(i as u8) { s } else { continue; };
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
    pieces.sort_unstable();
    pieces
}

pub fn extract_kpp_features(pos: &Position) -> Vec<usize> {
    KppState::new(pos).map_or(vec![], |s| s.features.into_iter().collect())
}
