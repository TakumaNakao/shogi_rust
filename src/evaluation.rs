use anyhow::Result;
use shogi_core::{Color, Piece, PieceKind, Square};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use rand::prelude::*;
use rand_distr::Distribution;

// --- Evaluator Trait ---
pub trait Evaluator {
    fn evaluate(&self, position: &shogi_core::Position) -> f32;
}

// --- KPP-based Evaluator ---

const NUM_SQUARES: usize = 81;
const NUM_BOARD_PIECE_KINDS: usize = 14;

const MAX_HAND_PAWNS: usize = 18;
const MAX_HAND_LANCES: usize = 4;
const MAX_HAND_KNIGHTS: usize = 4;
const MAX_HAND_SILVERS: usize = 4;
const MAX_HAND_GOLDS: usize = 4;
const MAX_HAND_BISHOPS: usize = 2;
const MAX_HAND_ROOKS: usize = 2;

const NUM_HAND_PIECE_SLOTS_PER_PLAYER: usize = MAX_HAND_PAWNS
    + MAX_HAND_LANCES
    + MAX_HAND_KNIGHTS
    + MAX_HAND_SILVERS
    + MAX_HAND_GOLDS
    + MAX_HAND_BISHOPS
    + MAX_HAND_ROOKS;

const NUM_PIECE_STATES: usize =
    (NUM_BOARD_PIECE_KINDS * NUM_SQUARES * 2) + (NUM_HAND_PIECE_SLOTS_PER_PLAYER * 2);

const NUM_PIECE_PAIRS: usize = NUM_PIECE_STATES * (NUM_PIECE_STATES - 1) / 2;

const MAX_FEATURES: usize = 200_000_000;

const ALL_HAND_PIECES: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

const NUM_BOARD_PIECE_VALUES: usize = 13; // King is not included
const NUM_HAND_PIECE_VALUES: usize = 7;

fn board_kind_to_index(kind: PieceKind) -> Option<usize> {
    match kind {
        PieceKind::Pawn => Some(0),
        PieceKind::Lance => Some(1),
        PieceKind::Knight => Some(2),
        PieceKind::Silver => Some(3),
        PieceKind::Gold => Some(4),
        PieceKind::Bishop => Some(5),
        PieceKind::Rook => Some(6),
        PieceKind::ProPawn => Some(7),
        PieceKind::ProLance => Some(8),
        PieceKind::ProKnight => Some(9),
        PieceKind::ProSilver => Some(10),
        PieceKind::ProBishop => Some(11),
        PieceKind::ProRook => Some(12),
        PieceKind::King => Some(13),
    }
}

fn hand_kind_to_index(kind: PieceKind) -> Option<usize> {
    match kind {
        PieceKind::Pawn => Some(0),
        PieceKind::Lance => Some(1),
        PieceKind::Knight => Some(2),
        PieceKind::Silver => Some(3),
        PieceKind::Gold => Some(4),
        PieceKind::Bishop => Some(5),
        PieceKind::Rook => Some(6),
        _ => None,
    }
}

fn hand_kind_to_offset(kind: PieceKind) -> Option<usize> {
    match kind {
        PieceKind::Pawn => Some(0),
        PieceKind::Lance => Some(MAX_HAND_PAWNS),
        PieceKind::Knight => Some(MAX_HAND_PAWNS + MAX_HAND_LANCES),
        PieceKind::Silver => Some(MAX_HAND_PAWNS + MAX_HAND_LANCES + MAX_HAND_KNIGHTS),
        PieceKind::Gold => Some(MAX_HAND_PAWNS + MAX_HAND_LANCES + MAX_HAND_KNIGHTS + MAX_HAND_SILVERS),
        PieceKind::Bishop => Some(MAX_HAND_PAWNS + MAX_HAND_LANCES + MAX_HAND_KNIGHTS + MAX_HAND_SILVERS + MAX_HAND_GOLDS),
        PieceKind::Rook => Some(MAX_HAND_PAWNS + MAX_HAND_LANCES + MAX_HAND_KNIGHTS + MAX_HAND_SILVERS + MAX_HAND_GOLDS + MAX_HAND_BISHOPS),
        _ => None,
    }
}

#[derive(Default)]
pub struct SparseModel {
    pub w: Vec<f32>,
    pub bias: f32,
    pub board_piece_values: [f32; NUM_BOARD_PIECE_VALUES],
    pub hand_piece_values: [f32; NUM_HAND_PIECE_VALUES],
    pub material_weight: f32,
    pub eta: f32,
}

impl SparseModel {
    pub fn new(eta: f32) -> Self {
        Self {
            w: vec![0.0; MAX_FEATURES],
            bias: 0.0,
            board_piece_values: [
                100.0, 300.0, 350.0, 500.0, 550.0, 800.0, 1000.0, 600.0, 600.0, 600.0, 600.0,
                1000.0, 1200.0,
            ],
            hand_piece_values: [100.0, 300.0, 350.0, 500.0, 550.0, 800.0, 1000.0],
            material_weight: 1.0,
            eta,
        }
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let mut offset = 0;

        self.bias = f32::from_le_bytes(buffer[offset..offset + 4].try_into()?);
        offset += 4;

        for i in 0..NUM_BOARD_PIECE_VALUES {
            self.board_piece_values[i] = f32::from_le_bytes(buffer[offset..offset + 4].try_into()?);
            offset += 4;
        }

        for i in 0..NUM_HAND_PIECE_VALUES {
            self.hand_piece_values[i] = f32::from_le_bytes(buffer[offset..offset + 4].try_into()?);
            offset += 4;
        }

        self.material_weight = f32::from_le_bytes(buffer[offset..offset + 4].try_into()?);
        offset += 4;

        let expected_w_bytes = MAX_FEATURES * 4;
        if buffer.len() - offset != expected_w_bytes {
            return Err(anyhow::anyhow!("File size mismatch for weights"));
        }

        for i in 0..MAX_FEATURES {
            let start = offset + i * 4;
            let end = start + 4;
            self.w[i] = f32::from_le_bytes(buffer[start..end].try_into()?);
        }

        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path)?;
        let mut buffer = Vec::new();

        buffer.extend_from_slice(&self.bias.to_le_bytes());
        for &value in &self.board_piece_values {
            buffer.extend_from_slice(&value.to_le_bytes());
        }
        for &value in &self.hand_piece_values {
            buffer.extend_from_slice(&value.to_le_bytes());
        }
        buffer.extend_from_slice(&self.material_weight.to_le_bytes());

        for &v in self.w.iter() {
            buffer.extend_from_slice(&v.to_le_bytes());
        }

        file.write_all(&buffer)?;

        Ok(())
    }

    pub fn initialize_random(&mut self, count: usize, stddev: f32) {
        let mut rng = rand::thread_rng();
        let dist = rand_distr::Normal::new(0.0, stddev).unwrap();
        for _ in 0..count {
            let i = rng.gen_range(0..MAX_FEATURES);
            let v = dist.sample(&mut rng) as f32;
            self.w[i] = v;
        }
    }

    pub fn predict(&self, pos: &shogi_core::Position, kpp_features: &[usize]) -> f32 {
        let mut prediction = self.bias;
        for &i in kpp_features {
            if i < MAX_FEATURES {
                prediction += self.w[i];
            }
        }
        prediction += self.material_weight * calculate_material_score(pos, &self.board_piece_values, &self.hand_piece_values);
        prediction
    }

    pub fn update_batch(&mut self, batch: &[(shogi_core::Position, Vec<usize>, f32)]) -> f32 {
        let m = batch.len() as f32;
        if m == 0.0 {
            return 0.0;
        }
        let mut total_loss = 0.0;

        let mut bias_grad = 0.0;
        let mut w_grads = vec![0.0; MAX_FEATURES];
        let mut board_piece_values_grads = [0.0; NUM_BOARD_PIECE_VALUES];
        let mut hand_piece_values_grads = [0.0; NUM_HAND_PIECE_VALUES];
        let mut material_weight_grad = 0.0;

        for (pos, kpp_features, y_true) in batch.iter() {
            let y_pred = self.predict(pos, kpp_features);
            let error = y_pred - y_true;
            total_loss += error * error;

            let error_grad = 2.0 * error / m;

            bias_grad += error_grad;

            for &i in kpp_features {
                if i < MAX_FEATURES {
                    w_grads[i] += error_grad;
                }
            }

            let material_score = calculate_material_score(pos, &self.board_piece_values, &self.hand_piece_values);
            material_weight_grad += error_grad * material_score;

            let (board_counts, hand_counts) = get_piece_counts(pos);
            for i in 1..NUM_BOARD_PIECE_VALUES { // Skip Pawn
                board_piece_values_grads[i] += error_grad * self.material_weight * board_counts[i];
            }
            for i in 1..NUM_HAND_PIECE_VALUES { // Skip Pawn
                hand_piece_values_grads[i] += error_grad * self.material_weight * hand_counts[i];
            }
        }

        self.bias -= self.eta * bias_grad;
        for i in 0..MAX_FEATURES {
            self.w[i] -= self.eta * w_grads[i];
        }
        for i in 1..NUM_BOARD_PIECE_VALUES { // Skip Pawn
            self.board_piece_values[i] -= self.eta * board_piece_values_grads[i];
        }
        for i in 1..NUM_HAND_PIECE_VALUES { // Skip Pawn
            self.hand_piece_values[i] -= self.eta * hand_piece_values_grads[i];
        }
        self.material_weight -= self.eta * material_weight_grad;

        let mse = total_loss / m;
        mse
    }
}

fn piece_to_id(piece: Piece, sq: Option<Square>, hand_index: usize) -> Option<usize> {
    let color_offset = if piece.color() == Color::Black { 0 } else { 1 };

    if let Some(sq) = sq {
        if let Some(kind_index) = board_kind_to_index(piece.piece_kind()) {
            let id = (color_offset * NUM_BOARD_PIECE_KINDS * NUM_SQUARES)
                + (kind_index * NUM_SQUARES)
                + (sq.index() as usize);
            Some(id)
        } else {
            None
        }
    } else {
        let board_pieces_total = NUM_BOARD_PIECE_KINDS * NUM_SQUARES * 2;
        if let Some(kind_offset) = hand_kind_to_offset(piece.piece_kind()) {
            let id = board_pieces_total
                + (color_offset * NUM_HAND_PIECE_SLOTS_PER_PLAYER)
                + kind_offset
                + hand_index;
            Some(id)
        } else {
            None
        }
    }
}

pub fn extract_kpp_features(pos: &shogi_core::Position) -> Vec<usize> {
    let king_sq_index = match (0..81).find_map(|i| {
        Square::from_u8(i as u8 + 1).and_then(|sq| {
            pos.piece_at(sq).and_then(|p| {
                if p.piece_kind() == PieceKind::King && p.color() == Color::Black {
                    Some(i as usize)
                } else {
                    None
                }
            })
        })
    }) {
        Some(idx) => idx,
        None => {
            println!("Warning: Black king not found on the board. Skipping this position.");
            return vec![];
        }
    };

    let mut piece_ids = Vec::with_capacity(40);
    for i in 0..81 {
        if let Some(sq) = Square::from_u8(i as u8 + 1) {
            if let Some(piece) = pos.piece_at(sq) {
                if let Some(id) = piece_to_id(piece, Some(sq), 0) {
                    piece_ids.push(id);
                }
            }
        }
    }
    for color in [Color::Black, Color::White] {
        for kind in ALL_HAND_PIECES.iter() {
            let count = pos.hand_of_a_player(color).count(*kind).unwrap_or(0);
            for i in 0..count {
                if let Some(id) = piece_to_id(Piece::new(*kind, color), None, i as usize) {
                    piece_ids.push(id);
                }
            }
        }
    }

    let mut indices = Vec::with_capacity(piece_ids.len() * piece_ids.len() / 2);
    for i in 0..piece_ids.len() {
        for j in (i + 1)..piece_ids.len() {
            let (id1, id2) = if piece_ids[i] < piece_ids[j] {
                (piece_ids[i], piece_ids[j])
            } else {
                (piece_ids[j], piece_ids[i])
            };

            let pair_index = id2 * (id2 - 1) / 2 + id1;

            let final_index = king_sq_index * NUM_PIECE_PAIRS + pair_index;
            indices.push(final_index);
        }
    }

    indices
}

fn get_piece_counts(position: &shogi_core::Position) -> ([f32; NUM_BOARD_PIECE_VALUES], [f32; NUM_HAND_PIECE_VALUES]) {
    let mut board_counts = [0.0; NUM_BOARD_PIECE_VALUES];
    let mut hand_counts = [0.0; NUM_HAND_PIECE_VALUES];

    // 盤上の駒を評価
    for i in 1..=9 {
        for j in 1..=9 {
            if let Some(square) = Square::new(i, j) {
                if let Some(piece) = position.piece_at(square) {
                    if let Some(index) = board_kind_to_index(piece.piece_kind()) {
                        if index < NUM_BOARD_PIECE_VALUES {
                            if piece.color() == Color::Black {
                                board_counts[index] += 1.0;
                            } else {
                                board_counts[index] -= 1.0;
                            }
                        }
                    }
                }
            }
        }
    }
    // 持ち駒を評価
    for color in &[Color::Black, Color::White] {
        let hand = position.hand_of_a_player(*color);
        for piece_kind in ALL_HAND_PIECES.iter() {
            if let Some(count) = hand.count(*piece_kind) {
                if let Some(index) = hand_kind_to_index(*piece_kind) {
                     if *color == Color::Black {
                        hand_counts[index] += count as f32;
                    } else {
                        hand_counts[index] -= count as f32;
                    }
                }
            }
        }
    }
    (board_counts, hand_counts)
}


// 駒得を計算する関数
fn calculate_material_score(
    position: &shogi_core::Position, 
    board_piece_values: &[f32; NUM_BOARD_PIECE_VALUES], 
    hand_piece_values: &[f32; NUM_HAND_PIECE_VALUES]
) -> f32 {
    let mut score = 0.0;
    let (board_counts, hand_counts) = get_piece_counts(position);
    for i in 0..NUM_BOARD_PIECE_VALUES {
        score += board_counts[i] * board_piece_values[i];
    }
    for i in 0..NUM_HAND_PIECE_VALUES {
        score += hand_counts[i] * hand_piece_values[i];
    }
    score
}

pub struct SparseModelEvaluator {
    pub model: SparseModel,
}

impl SparseModelEvaluator {
    pub fn new(weight_path: &Path) -> Result<Self> {
        let mut model = SparseModel::new(0.0);
        model.load(weight_path)?;
        Ok(SparseModelEvaluator { model })
    }
}

impl Evaluator for SparseModelEvaluator {
    fn evaluate(&self, position: &shogi_core::Position) -> f32 {
        let kpp_features = extract_kpp_features(position);
        self.model.predict(position, &kpp_features)
    }
}