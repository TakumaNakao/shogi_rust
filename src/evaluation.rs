#![allow(dead_code)]
use anyhow::{anyhow, Result};
use rand::prelude::*;
use rand_distr::Distribution;
use rayon::prelude::*;
use shogi_core::{Color, Piece, PieceKind, Square};
use shogi_lib;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::LazyLock;

// --- Piece Values ---
const PAWN_VALUE: f32 = 100.0;
const LANCE_VALUE: f32 = 300.0;
const KNIGHT_VALUE: f32 = 350.0;
const SILVER_VALUE: f32 = 500.0;
const GOLD_VALUE: f32 = 550.0;
const BISHOP_VALUE: f32 = 800.0;
const ROOK_VALUE: f32 = 1000.0;
const PRO_PAWN_VALUE: f32 = 400.0;
const PRO_LANCE_VALUE: f32 = 400.0;
const PRO_KNIGHT_VALUE: f32 = 400.0;
const PRO_SILVER_VALUE: f32 = 550.0;
const PRO_BISHOP_VALUE: f32 = 1000.0;
const PRO_ROOK_VALUE: f32 = 1200.0;

fn piece_kind_value(kind: PieceKind) -> f32 {
    match kind {
        PieceKind::Pawn => PAWN_VALUE,
        PieceKind::Lance => LANCE_VALUE,
        PieceKind::Knight => KNIGHT_VALUE,
        PieceKind::Silver => SILVER_VALUE,
        PieceKind::Gold => GOLD_VALUE,
        PieceKind::Bishop => BISHOP_VALUE,
        PieceKind::Rook => ROOK_VALUE,
        PieceKind::ProPawn => PRO_PAWN_VALUE,
        PieceKind::ProLance => PRO_LANCE_VALUE,
        PieceKind::ProKnight => PRO_KNIGHT_VALUE,
        PieceKind::ProSilver => PRO_SILVER_VALUE,
        PieceKind::ProBishop => PRO_BISHOP_VALUE,
        PieceKind::ProRook => PRO_ROOK_VALUE,
        PieceKind::King => 0.0, // King value is effectively infinite
    }
}

// --- Public Constants for KPP Feature Space ---
pub const NUM_SQUARES: usize = 81;
pub const NUM_BOARD_PIECE_KINDS: usize = 14;

pub const MAX_HAND_PAWNS: usize = 18;
pub const MAX_HAND_LANCES: usize = 4;
pub const MAX_HAND_KNIGHTS: usize = 4;
pub const MAX_HAND_SILVERS: usize = 4;
pub const MAX_HAND_GOLDS: usize = 4;
pub const MAX_HAND_BISHOPS: usize = 2;
pub const MAX_HAND_ROOKS: usize = 2;

pub const NUM_HAND_PIECE_SLOTS_PER_PLAYER: usize = MAX_HAND_PAWNS
    + MAX_HAND_LANCES
    + MAX_HAND_KNIGHTS
    + MAX_HAND_SILVERS
    + MAX_HAND_GOLDS
    + MAX_HAND_BISHOPS
    + MAX_HAND_ROOKS;

pub const NUM_PIECE_STATES: usize =
    (NUM_BOARD_PIECE_KINDS * NUM_SQUARES * 2) + (NUM_HAND_PIECE_SLOTS_PER_PLAYER * 2);

pub const NUM_PIECE_PAIRS: usize = NUM_PIECE_STATES * (NUM_PIECE_STATES - 1) / 2;

pub const MAX_FEATURES: usize = NUM_SQUARES * NUM_PIECE_PAIRS;

pub const ALL_HAND_PIECES: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

pub const NNUE_NUM_KING_BUCKETS: usize = NUM_SQUARES * NUM_SQUARES;
pub const NNUE_NUM_BOARD_FEATURES: usize = NUM_BOARD_PIECE_KINDS * NUM_SQUARES * 2;
pub const NNUE_NUM_HAND_FEATURES: usize = NUM_HAND_PIECE_SLOTS_PER_PLAYER * 2;
pub const NNUE_NUM_FEATURES: usize = NNUE_NUM_BOARD_FEATURES + NNUE_NUM_HAND_FEATURES;

#[derive(Debug, Clone)]
pub struct NnueFeatures {
    pub king_bucket: usize,
    pub features: Vec<usize>,
    pub material: f32,
}

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

static BOARD_SQUARES: LazyLock<[Square; NUM_SQUARES]> = LazyLock::new(|| {
    std::array::from_fn(|i| Square::from_u8(i as u8 + 1).expect("valid board square"))
});

// --- Evaluator Trait ---
pub trait Evaluator {
    fn evaluate(&self, position: &shogi_lib::Position) -> f32;
}

// --- KPP-based Evaluator ---

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

fn hand_kind_to_offset(kind: PieceKind) -> Option<usize> {
    match kind {
        PieceKind::Pawn => Some(0),
        PieceKind::Lance => Some(MAX_HAND_PAWNS),
        PieceKind::Knight => Some(MAX_HAND_PAWNS + MAX_HAND_LANCES),
        PieceKind::Silver => Some(MAX_HAND_PAWNS + MAX_HAND_LANCES + MAX_HAND_KNIGHTS),
        PieceKind::Gold => {
            Some(MAX_HAND_PAWNS + MAX_HAND_LANCES + MAX_HAND_KNIGHTS + MAX_HAND_SILVERS)
        }
        PieceKind::Bishop => Some(
            MAX_HAND_PAWNS + MAX_HAND_LANCES + MAX_HAND_KNIGHTS + MAX_HAND_SILVERS + MAX_HAND_GOLDS,
        ),
        PieceKind::Rook => Some(
            MAX_HAND_PAWNS
                + MAX_HAND_LANCES
                + MAX_HAND_KNIGHTS
                + MAX_HAND_SILVERS
                + MAX_HAND_GOLDS
                + MAX_HAND_BISHOPS,
        ),
        _ => None,
    }
}

fn piece_to_id(piece: Piece, sq: Option<Square>, hand_index: usize, turn: Color) -> Option<usize> {
    let normalized_color = if piece.color() == turn {
        Color::Black
    } else {
        Color::White
    };
    let color_offset = if normalized_color == Color::Black {
        0
    } else {
        1
    };

    if let Some(sq) = sq {
        let normalized_sq = if turn == Color::Black { sq } else { sq.flip() };
        if let Some(kind_index) = board_kind_to_index(piece.piece_kind()) {
            let id = (color_offset * NUM_BOARD_PIECE_KINDS * NUM_SQUARES)
                + (kind_index * NUM_SQUARES)
                + ((normalized_sq.index() - 1) as usize);
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

pub fn extract_kpp_features(pos: &shogi_lib::Position) -> Vec<usize> {
    extract_kpp_features_and_material(pos).0
}

pub fn extract_kpp_features_and_material(pos: &shogi_lib::Position) -> (Vec<usize>, f32) {
    let turn = pos.side_to_move();
    let mut material = 0.0;
    let mut piece_ids = Vec::with_capacity(40);
    let mut king_sq = None;

    for &sq in BOARD_SQUARES.iter() {
        if let Some(piece) = pos.piece_at(sq) {
            let value = piece_kind_value(piece.piece_kind());
            if piece.color() == turn {
                material += value;
            } else {
                material -= value;
            }
            if piece.piece_kind() == PieceKind::King && piece.color() == turn {
                king_sq = Some(sq);
                continue;
            }
            if let Some(id) = piece_to_id(piece, Some(sq), 0, turn) {
                piece_ids.push(id);
            }
        }
    }

    let king_sq = match king_sq {
        Some(sq) => sq,
        None => {
            println!(
                "Warning: King not found for side {:?}. Skipping this position.",
                turn
            );
            return (vec![], 0.0);
        }
    };
    let normalized_king_sq = if turn == Color::Black {
        king_sq
    } else {
        king_sq.flip()
    };
    let king_sq_index = (normalized_king_sq.index() - 1) as usize;

    for color in [Color::Black, Color::White] {
        for kind in ALL_HAND_PIECES.iter() {
            let count = pos.hand(color).count(*kind).unwrap_or(0);
            let value = piece_kind_value(*kind);
            if color == turn {
                material += count as f32 * value;
            } else {
                material -= count as f32 * value;
            }
            for i in 0..count {
                if let Some(id) = piece_to_id(Piece::new(*kind, color), None, i as usize, turn) {
                    piece_ids.push(id);
                }
            }
        }
    }

    piece_ids.sort_unstable();

    let mut indices = Vec::with_capacity(piece_ids.len() * piece_ids.len() / 2);
    for i in 0..piece_ids.len() {
        for j in (i + 1)..piece_ids.len() {
            let id1 = piece_ids[i];
            let id2 = piece_ids[j];

            let pair_index = id2 * (id2 - 1) / 2 + id1;

            let final_index = king_sq_index * NUM_PIECE_PAIRS + pair_index;
            indices.push(final_index);
        }
    }

    (indices, material)
}

pub fn extract_nnue_features(pos: &shogi_lib::Position) -> Option<NnueFeatures> {
    let turn = pos.side_to_move();
    let mut material = 0.0;
    let mut features = Vec::with_capacity(40);
    let mut own_king = None;
    let mut opponent_king = None;

    for &sq in BOARD_SQUARES.iter() {
        if let Some(piece) = pos.piece_at(sq) {
            let value = piece_kind_value(piece.piece_kind());
            if piece.color() == turn {
                material += value;
            } else {
                material -= value;
            }

            let normalized_sq = if turn == Color::Black { sq } else { sq.flip() };
            if piece.piece_kind() == PieceKind::King {
                if piece.color() == turn {
                    own_king = Some(normalized_sq);
                } else {
                    opponent_king = Some(normalized_sq);
                }
            }

            let normalized_color = if piece.color() == turn {
                Color::Black
            } else {
                Color::White
            };
            let color_offset = if normalized_color == Color::Black {
                0
            } else {
                NUM_BOARD_PIECE_KINDS * NUM_SQUARES
            };
            let kind_index = board_kind_to_index(piece.piece_kind())?;
            features.push(
                color_offset + kind_index * NUM_SQUARES + (normalized_sq.index() - 1) as usize,
            );
        }
    }

    let hand_base = NNUE_NUM_BOARD_FEATURES;
    for color in [Color::Black, Color::White] {
        let normalized_color = if color == turn {
            Color::Black
        } else {
            Color::White
        };
        let color_offset = if normalized_color == Color::Black {
            0
        } else {
            NUM_HAND_PIECE_SLOTS_PER_PLAYER
        };
        for kind in ALL_HAND_PIECES.iter() {
            let count = pos.hand(color).count(*kind).unwrap_or(0);
            let value = piece_kind_value(*kind);
            if color == turn {
                material += count as f32 * value;
            } else {
                material -= count as f32 * value;
            }
            let kind_offset = hand_kind_to_offset(*kind)?;
            for i in 0..count {
                features.push(hand_base + color_offset + kind_offset + i as usize);
            }
        }
    }

    features.sort_unstable();
    features.dedup();

    let own_king = own_king?;
    let opponent_king = opponent_king?;
    let king_bucket =
        (own_king.index() - 1) as usize * NUM_SQUARES + (opponent_king.index() - 1) as usize;

    Some(NnueFeatures {
        king_bucket,
        features,
        material,
    })
}

fn read_u32_le(file: &mut File) -> Result<u32> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_f32_le(file: &mut File) -> Result<f32> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_f32_vec(file: &mut File, len: usize) -> Result<Vec<f32>> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(read_f32_le(file)?);
    }
    Ok(values)
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

        Ok(TinyNnueModel {
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

pub fn calculate_material_advantage(pos: &shogi_lib::Position) -> f32 {
    let mut material = 0.0;
    let turn = pos.side_to_move();

    // Board pieces
    for &sq in BOARD_SQUARES.iter() {
        if let Some(piece) = pos.piece_at(sq) {
            let value = piece_kind_value(piece.piece_kind());
            if piece.color() == turn {
                material += value;
            } else {
                material -= value;
            }
        }
    }

    // Hand pieces
    for color in [Color::Black, Color::White] {
        for &kind in ALL_HAND_PIECES.iter() {
            let count = pos.hand(color).count(kind).unwrap_or(0) as f32;
            let value = piece_kind_value(kind);
            if color == turn {
                material += count * value;
            } else {
                material -= count * value;
            }
        }
    }
    material
}

#[derive(Default, Clone)]
pub struct SparseModel {
    pub w: Vec<f32>,
    pub bias: f32,
    pub material_coeff: f32, // New coefficient for material advantage
    pub kpp_eta: f32,
    pub l2_lambda: f32,
}

impl SparseModel {
    pub fn new(kpp_eta: f32, l2_lambda: f32) -> Self {
        Self {
            w: vec![0.0; MAX_FEATURES],
            bias: 0.0,
            material_coeff: 1.0, // Initialize to 0
            kpp_eta,
            l2_lambda,
        }
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let mut offset = 0;

        self.bias = f32::from_le_bytes(buffer[offset..offset + 4].try_into()?);
        offset += 4;

        self.material_coeff = f32::from_le_bytes(buffer[offset..offset + 4].try_into()?);
        offset += 4;

        let expected_w_bytes = MAX_FEATURES * 4;
        if buffer.len() - offset != expected_w_bytes {
            return Err(anyhow::anyhow!(
                "File size mismatch for weights. Expected {} bytes, got {}.",
                expected_w_bytes + 8,
                buffer.len()
            ));
        }

        for i in 0..MAX_FEATURES {
            let start = offset + i * 4;
            let end = start + 4;
            if end > buffer.len() {
                return Err(anyhow::anyhow!(
                    "Unexpected end of file while reading weights."
                ));
            }
            self.w[i] = f32::from_le_bytes(buffer[start..end].try_into()?);
        }

        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path)?;
        let mut buffer = Vec::new();

        buffer.extend_from_slice(&self.bias.to_le_bytes());
        buffer.extend_from_slice(&self.material_coeff.to_le_bytes());

        for &v in self.w.iter() {
            buffer.extend_from_slice(&v.to_le_bytes());
        }

        file.write_all(&buffer)?;

        println!(
            "Max W: {:?}, Material Coeff: {}",
            self.w
                .iter()
                .cloned()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
            self.material_coeff
        );

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
        self.material_coeff = dist.sample(&mut rng) as f32;
    }

    pub fn zero_weight_overwrite(&mut self, overwrite_value: f32) {
        for i in 0..MAX_FEATURES {
            if self.w[i] == 0.0 {
                self.w[i] = overwrite_value;
            }
        }
    }

    pub fn predict(&self, pos: &shogi_lib::Position, kpp_features: &[usize]) -> f32 {
        let material = calculate_material_advantage(pos);
        self.predict_with_material(kpp_features, material)
    }

    pub fn predict_with_material(&self, kpp_features: &[usize], material: f32) -> f32 {
        let mut prediction = self.bias;
        for &i in kpp_features {
            if i < MAX_FEATURES {
                prediction += self.w[i];
            }
        }
        prediction += self.material_coeff * material;
        prediction
    }

    pub fn predict_from_position(&self, pos: &shogi_lib::Position) -> f32 {
        let turn = pos.side_to_move();
        let mut material = 0.0;
        let mut piece_ids = Vec::with_capacity(40);
        let mut king_sq = None;

        for &sq in BOARD_SQUARES.iter() {
            if let Some(piece) = pos.piece_at(sq) {
                let value = piece_kind_value(piece.piece_kind());
                if piece.color() == turn {
                    material += value;
                } else {
                    material -= value;
                }
                if piece.piece_kind() == PieceKind::King && piece.color() == turn {
                    king_sq = Some(sq);
                    continue;
                }
                if let Some(id) = piece_to_id(piece, Some(sq), 0, turn) {
                    piece_ids.push(id);
                }
            }
        }

        let king_sq = match king_sq {
            Some(sq) => sq,
            None => {
                println!(
                    "Warning: King not found for side {:?}. Skipping this position.",
                    turn
                );
                return 0.0;
            }
        };
        let normalized_king_sq = if turn == Color::Black {
            king_sq
        } else {
            king_sq.flip()
        };
        let king_sq_index = (normalized_king_sq.index() - 1) as usize;

        for color in [Color::Black, Color::White] {
            for kind in ALL_HAND_PIECES.iter() {
                let count = pos.hand(color).count(*kind).unwrap_or(0);
                let value = piece_kind_value(*kind);
                if color == turn {
                    material += count as f32 * value;
                } else {
                    material -= count as f32 * value;
                }
                for i in 0..count {
                    if let Some(id) = piece_to_id(Piece::new(*kind, color), None, i as usize, turn)
                    {
                        piece_ids.push(id);
                    }
                }
            }
        }

        piece_ids.sort_unstable();

        let mut prediction = self.bias;
        for i in 0..piece_ids.len() {
            for j in (i + 1)..piece_ids.len() {
                let id1 = piece_ids[i];
                let id2 = piece_ids[j];
                let pair_index = id2 * (id2 - 1) / 2 + id1;
                let final_index = king_sq_index * NUM_PIECE_PAIRS + pair_index;
                prediction += self.w[final_index];
            }
        }
        prediction + self.material_coeff * material
    }

    pub fn update_batch_for_moves(
        &mut self,
        batch: &[(shogi_lib::Position, shogi_core::Move)],
    ) -> (usize, usize) {
        let total_samples = batch.len();
        if total_samples == 0 {
            return (0, 0);
        }

        let results: Vec<(bool, HashMap<usize, f32>, f32, f32)> = batch
            .par_iter()
            .map(|(pos, teacher_move)| {
                let legal_moves = pos.legal_moves();
                if legal_moves.is_empty() {
                    return (false, HashMap::new(), 0.0, 0.0);
                }

                let mut best_move_by_model: Option<shogi_core::Move> = None;
                let mut max_score = -f32::INFINITY;
                let mut best_model_features: Vec<usize> = Vec::new();
                let mut best_model_material = 0.0;
                let mut teacher_move_features: Option<Vec<usize>> = None;
                let mut teacher_material = 0.0;

                for &mv in legal_moves.iter() {
                    let mut temp_pos = pos.clone();
                    temp_pos.do_move(mv);
                    temp_pos.switch_turn();
                    let features = extract_kpp_features(&temp_pos);
                    let material = calculate_material_advantage(&temp_pos);
                    let score = self.predict(&temp_pos, &features);

                    if score > max_score {
                        max_score = score;
                        best_move_by_model = Some(mv);
                        best_model_features = features.clone();
                        best_model_material = material;
                    }

                    if mv == *teacher_move {
                        teacher_move_features = Some(features);
                        teacher_material = material;
                    }
                }

                let mut sparse_grads = HashMap::new();
                let mut is_correct = false;

                if let Some(model_move) = best_move_by_model {
                    if model_move == *teacher_move {
                        is_correct = true;
                    } else {
                        if let Some(teacher_features) = teacher_move_features {
                            let teacher_set: HashSet<_> = teacher_features.into_iter().collect();
                            let model_set: HashSet<_> = best_model_features.into_iter().collect();

                            for &idx in teacher_set.difference(&model_set) {
                                *sparse_grads.entry(idx).or_insert(0.0) += self.kpp_eta;
                            }
                            for &idx in model_set.difference(&teacher_set) {
                                *sparse_grads.entry(idx).or_insert(0.0) -= self.kpp_eta;
                            }
                        }
                    }
                }
                (
                    is_correct,
                    sparse_grads,
                    teacher_material,
                    best_model_material,
                )
            })
            .collect();

        let mut correct_predictions = 0;
        let mut w_grads = HashMap::new();
        let mut material_grad = 0.0;

        for (is_correct, sparse_grad, teacher_material, model_material) in results {
            if is_correct {
                correct_predictions += 1;
            } else {
                material_grad += self.kpp_eta * (teacher_material - model_material);
            }
            for (idx, g) in sparse_grad {
                *w_grads.entry(idx).or_insert(0.0) += g;
            }
        }

        let decay_factor = 1.0 - self.kpp_eta * self.l2_lambda;
        for w in self.w.iter_mut() {
            *w *= decay_factor;
        }

        for (i, g) in w_grads {
            self.w[i] += g / total_samples as f32;
        }

        self.material_coeff += material_grad / total_samples as f32
            - self.kpp_eta * self.l2_lambda * self.material_coeff;

        (correct_predictions, total_samples)
    }

    pub fn update_batch_with_cross_entropy(
        &mut self,
        batch: &[(shogi_lib::Position, shogi_core::Move)],
    ) -> (f32, usize) {
        self.update_batch_with_cross_entropy_temperature(batch, 600.0)
    }

    pub fn update_batch_with_cross_entropy_temperature(
        &mut self,
        batch: &[(shogi_lib::Position, shogi_core::Move)],
        softmax_temperature: f32,
    ) -> (f32, usize) {
        let total_samples = batch.len();
        if total_samples == 0 {
            return (0.0, 0);
        }
        if !softmax_temperature.is_finite() || softmax_temperature <= 0.0 {
            return (0.0, 0);
        }

        let results: Vec<(HashMap<usize, f32>, f32, f32, bool)> = batch
            .par_iter()
            .map(|(pos, teacher_move)| {
                let legal_moves = pos.legal_moves();
                if legal_moves.is_empty() {
                    return (HashMap::new(), 0.0, 0.0, false);
                }

                let move_data: Vec<_> = legal_moves
                    .iter()
                    .map(|&mv| {
                        let mut temp_pos = pos.clone();
                        temp_pos.do_move(mv);
                        temp_pos.switch_turn();
                        let features = extract_kpp_features(&temp_pos);
                        let material = calculate_material_advantage(&temp_pos);
                        let score = self.predict(&temp_pos, &features);
                        (mv, features, material, score)
                    })
                    .collect();

                let max_score = move_data
                    .iter()
                    .map(|d| d.3)
                    .fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = move_data
                    .iter()
                    .map(|d| ((d.3 - max_score) / softmax_temperature).exp())
                    .collect();
                let total_score = exp_scores.iter().sum::<f32>();

                let mut sparse_grads = HashMap::new();
                let mut material_grad = 0.0;
                let mut loss = 0.0;

                if move_data.iter().any(|(m, _, _, _)| m == teacher_move) {
                    for (i, (mv, features, material, _)) in move_data.iter().enumerate() {
                        let prob = exp_scores[i] / total_score;
                        if mv == teacher_move {
                            loss = -prob.max(1e-7).ln();
                        }

                        let delta = if mv == teacher_move { prob - 1.0 } else { prob }
                            / softmax_temperature;

                        for &idx in features {
                            *sparse_grads.entry(idx).or_insert(0.0) += delta;
                        }
                        material_grad += delta * *material;
                    }
                    return (sparse_grads, material_grad, loss, true);
                }

                (sparse_grads, material_grad, 0.0, false)
            })
            .collect();

        let mut w_grads = HashMap::new();
        let mut material_grad_total = 0.0;
        let mut loss = 0.0;
        let mut valid_samples = 0;
        for (sparse_grad, material_grad, sample_loss, teacher_found) in results {
            if teacher_found {
                valid_samples += 1;
                loss += sample_loss;
                for (idx, g) in sparse_grad {
                    *w_grads.entry(idx).or_insert(0.0) += g;
                }
                material_grad_total += material_grad;
            }
        }

        if valid_samples == 0 {
            return (0.0, 0);
        }

        let avg_loss = loss / valid_samples as f32;

        for (i, grad) in w_grads {
            self.w[i] -= self.kpp_eta * (grad / valid_samples as f32 + self.l2_lambda * self.w[i]);
        }
        self.material_coeff -= self.kpp_eta
            * (material_grad_total / valid_samples as f32 + self.l2_lambda * self.material_coeff);

        (avg_loss, valid_samples)
    }
}

pub struct SparseModelEvaluator {
    pub model: SparseModel,
}

impl SparseModelEvaluator {
    pub fn new(weight_path: &Path, overwrite_value: f32) -> Result<Self> {
        let mut model = SparseModel::new(0.0, 0.0);
        model.load(weight_path)?;
        model.zero_weight_overwrite(overwrite_value);
        Ok(SparseModelEvaluator { model })
    }
}

impl Evaluator for SparseModelEvaluator {
    fn evaluate(&self, position: &shogi_lib::Position) -> f32 {
        self.model.predict_from_position(position)
    }
}

pub enum EngineEvaluator {
    Sparse(SparseModelEvaluator),
    TinyNnue(TinyNnueModel),
}

impl EngineEvaluator {
    pub fn new(path: &Path, material_override: f32) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 8];
        let read = file.read(&mut magic)?;
        drop(file);

        if read == magic.len() && &magic == b"TNNUE001" {
            Ok(EngineEvaluator::TinyNnue(TinyNnueModel::load(path)?))
        } else {
            Ok(EngineEvaluator::Sparse(SparseModelEvaluator::new(
                path,
                material_override,
            )?))
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            EngineEvaluator::Sparse(_) => "sparse",
            EngineEvaluator::TinyNnue(_) => "tiny-nnue",
        }
    }
}

impl Evaluator for EngineEvaluator {
    fn evaluate(&self, position: &shogi_lib::Position) -> f32 {
        match self {
            EngineEvaluator::Sparse(evaluator) => evaluator.evaluate(position),
            EngineEvaluator::TinyNnue(model) => model.predict_from_position(position),
        }
    }
}

// --- Decoding functions (moved from kpp_weight_check.rs) ---

fn index_to_board_kind(index: usize) -> Option<PieceKind> {
    match index {
        0 => Some(PieceKind::Pawn),
        1 => Some(PieceKind::Lance),
        2 => Some(PieceKind::Knight),
        3 => Some(PieceKind::Silver),
        4 => Some(PieceKind::Gold),
        5 => Some(PieceKind::Bishop),
        6 => Some(PieceKind::Rook),
        7 => Some(PieceKind::ProPawn),
        8 => Some(PieceKind::ProLance),
        9 => Some(PieceKind::ProKnight),
        10 => Some(PieceKind::ProSilver),
        11 => Some(PieceKind::ProBishop),
        12 => Some(PieceKind::ProRook),
        13 => Some(PieceKind::King),
        _ => None,
    }
}

fn index_to_hand_kind_and_offset(index: usize) -> Option<(PieceKind, usize)> {
    let mut current_offset = 0;
    if index < MAX_HAND_PAWNS {
        return Some((PieceKind::Pawn, index));
    }
    current_offset += MAX_HAND_PAWNS;
    if index < current_offset + MAX_HAND_LANCES {
        return Some((PieceKind::Lance, index - current_offset));
    }
    current_offset += MAX_HAND_LANCES;
    if index < current_offset + MAX_HAND_KNIGHTS {
        return Some((PieceKind::Knight, index - current_offset));
    }
    current_offset += MAX_HAND_KNIGHTS;
    if index < current_offset + MAX_HAND_SILVERS {
        return Some((PieceKind::Silver, index - current_offset));
    }
    current_offset += MAX_HAND_SILVERS;
    if index < current_offset + MAX_HAND_GOLDS {
        return Some((PieceKind::Gold, index - current_offset));
    }
    current_offset += MAX_HAND_GOLDS;
    if index < current_offset + MAX_HAND_BISHOPS {
        return Some((PieceKind::Bishop, index - current_offset));
    }
    current_offset += MAX_HAND_BISHOPS;
    if index < current_offset + MAX_HAND_ROOKS {
        return Some((PieceKind::Rook, index - current_offset));
    }
    None
}

fn id_to_piece_info(id: usize) -> Option<(PieceKind, Option<Square>, Option<usize>, Color)> {
    let board_pieces_total = NUM_BOARD_PIECE_KINDS * NUM_SQUARES * 2;
    if id < board_pieces_total {
        let color_offset = id / (NUM_BOARD_PIECE_KINDS * NUM_SQUARES);
        let remaining_id = id % (NUM_BOARD_PIECE_KINDS * NUM_SQUARES);
        let kind_index = remaining_id / NUM_SQUARES;
        let sq_index = remaining_id % NUM_SQUARES;
        let piece_kind = index_to_board_kind(kind_index)?;
        let normalized_sq = Square::from_u8(sq_index as u8 + 1)?;
        let normalized_color = if color_offset == 0 {
            Color::Black
        } else {
            Color::White
        };
        Some((piece_kind, Some(normalized_sq), None, normalized_color))
    } else {
        let hand_id = id - board_pieces_total;
        let color_offset = hand_id / NUM_HAND_PIECE_SLOTS_PER_PLAYER;
        let remaining_hand_id = hand_id % NUM_HAND_PIECE_SLOTS_PER_PLAYER;
        let (piece_kind, hand_index) = index_to_hand_kind_and_offset(remaining_hand_id)?;
        let normalized_color = if color_offset == 0 {
            Color::Black
        } else {
            Color::White
        };
        Some((piece_kind, None, Some(hand_index), normalized_color))
    }
}

fn pair_index_to_ids(pair_index: usize) -> Option<(usize, usize)> {
    let mut id2 = 0;
    while id2 * (id2 - 1) / 2 <= pair_index {
        id2 += 1;
    }
    id2 -= 1;
    let pair_index_base = id2 * (id2 - 1) / 2;
    let id1 = pair_index - pair_index_base;
    if id1 < id2 {
        Some((id1, id2))
    } else {
        None
    }
}

pub type KppInfo = (
    Square,
    PieceKind,
    Option<Square>,
    Option<usize>,
    Color,
    PieceKind,
    Option<Square>,
    Option<usize>,
    Color,
);

pub fn index_to_kpp_info(index: usize) -> Option<KppInfo> {
    let king_sq_index = index / NUM_PIECE_PAIRS;
    let pair_index = index % NUM_PIECE_PAIRS;
    let king_sq = Square::from_u8((king_sq_index + 1) as u8)?;
    let (id1, id2) = pair_index_to_ids(pair_index)?;
    let (p1k, p1sq, p1hi, p1c) = id_to_piece_info(id1)?;
    let (p2k, p2sq, p2hi, p2c) = id_to_piece_info(id2)?;
    Some((king_sq, p1k, p1sq, p1hi, p1c, p2k, p2sq, p2hi, p2c))
}

// --- SFEN Generation for KPP (moved from kpp_weight_check.rs) ---

// Helper for SFEN character conversion
pub fn is_promoted_piece_kind(kind: PieceKind) -> bool {
    matches!(
        kind,
        PieceKind::ProPawn
            | PieceKind::ProLance
            | PieceKind::ProKnight
            | PieceKind::ProSilver
            | PieceKind::ProBishop
            | PieceKind::ProRook
    )
}

// Helper for SFEN character conversion
pub fn piece_kind_to_sfen_char_base(kind: PieceKind, color: Color) -> char {
    match kind {
        PieceKind::Pawn | PieceKind::ProPawn => {
            if color == Color::Black {
                'P'
            } else {
                'p'
            }
        }
        PieceKind::Lance | PieceKind::ProLance => {
            if color == Color::Black {
                'L'
            } else {
                'l'
            }
        }
        PieceKind::Knight | PieceKind::ProKnight => {
            if color == Color::Black {
                'N'
            } else {
                'n'
            }
        }
        PieceKind::Silver | PieceKind::ProSilver => {
            if color == Color::Black {
                'S'
            } else {
                's'
            }
        }
        PieceKind::Gold => {
            if color == Color::Black {
                'G'
            } else {
                'g'
            }
        }
        PieceKind::Bishop | PieceKind::ProBishop => {
            if color == Color::Black {
                'B'
            } else {
                'b'
            }
        }
        PieceKind::Rook | PieceKind::ProRook => {
            if color == Color::Black {
                'R'
            } else {
                'r'
            }
        }
        PieceKind::King => {
            if color == Color::Black {
                'K'
            } else {
                'k'
            }
        }
    }
}

// Function to generate SFEN from KPP info
pub fn generate_sfen(
    king_sq: Square,
    piece1_kind: PieceKind,
    piece1_sq: Option<Square>,
    piece1_hand_idx: Option<usize>,
    piece1_color: Color, // This is the normalized color
    piece2_kind: PieceKind,
    piece2_sq: Option<Square>,
    piece2_hand_idx: Option<usize>,
    piece2_color: Color, // This is the normalized color
    turn: Color,
) -> String {
    let mut sfen_board_pieces: Vec<Vec<Option<Piece>>> = vec![vec![None; 9]; 9];
    let mut black_hand_counts = [0; 7];
    let mut white_hand_counts = [0; 7];

    // Place king (always Black King in SFEN, as features are normalized to king's perspective)
    let file = king_sq.file() as usize - 1;
    let rank = king_sq.rank() as usize - 1;
    sfen_board_pieces[rank][file] = Some(Piece::new(PieceKind::King, Color::Black));

    // Place piece1
    if let Some(sq) = piece1_sq {
        let file = sq.file() as usize - 1;
        let rank = sq.rank() as usize - 1;
        sfen_board_pieces[rank][file] = Some(Piece::new(piece1_kind, piece1_color));
    } else if let Some(_) = piece1_hand_idx {
        if let Some(idx) = ALL_HAND_PIECES.iter().position(|&k| k == piece1_kind) {
            if piece1_color == Color::Black {
                black_hand_counts[idx] += 1;
            } else {
                white_hand_counts[idx] += 1;
            }
        }
    }

    // Place piece2
    if let Some(sq) = piece2_sq {
        let file = sq.file() as usize - 1;
        let rank = sq.rank() as usize - 1;
        sfen_board_pieces[rank][file] = Some(Piece::new(piece2_kind, piece2_color));
    } else if let Some(_) = piece2_hand_idx {
        if let Some(idx) = ALL_HAND_PIECES.iter().position(|&k| k == piece2_kind) {
            if piece2_color == Color::Black {
                black_hand_counts[idx] += 1;
            } else {
                white_hand_counts[idx] += 1;
            }
        }
    }

    // Construct SFEN board string
    let mut sfen_board_str = String::new();
    for rank in 0..9 {
        let mut count = 0;
        for file in 0..9 {
            if let Some(piece) = sfen_board_pieces[rank][file] {
                if count > 0 {
                    sfen_board_str.push_str(&count.to_string());
                    count = 0;
                }
                if is_promoted_piece_kind(piece.piece_kind()) {
                    sfen_board_str.push('+');
                }
                sfen_board_str.push(piece_kind_to_sfen_char_base(
                    piece.piece_kind(),
                    piece.color(),
                ));
            } else {
                count += 1;
            }
        }
        if count > 0 {
            sfen_board_str.push_str(&count.to_string());
        }
        if rank < 8 {
            sfen_board_str.push('/');
        }
    }

    let mut sfen_hand_str = String::new();
    for (i, &kind) in ALL_HAND_PIECES.iter().enumerate() {
        let black_count = black_hand_counts[i];
        let white_count = white_hand_counts[i];
        if black_count > 0 {
            if black_count > 1 {
                sfen_hand_str.push_str(&black_count.to_string());
            }
            sfen_hand_str.push(piece_kind_to_sfen_char_base(kind, Color::Black));
        }
        if white_count > 0 {
            if white_count > 1 {
                sfen_hand_str.push_str(&white_count.to_string());
            }
            sfen_hand_str.push(piece_kind_to_sfen_char_base(kind, Color::White));
        }
    }
    if sfen_hand_str.is_empty() {
        sfen_hand_str.push('-');
    }

    let turn_char = if turn == Color::Black { 'b' } else { 'w' };
    format!("{} {} {} 1", sfen_board_str, turn_char, sfen_hand_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nnue_features_are_in_declared_ranges() {
        let position = shogi_lib::Position::default();
        let features = extract_nnue_features(&position).expect("initial position has both kings");

        assert!(features.king_bucket < NNUE_NUM_KING_BUCKETS);
        assert!(!features.features.is_empty());
        assert!(features
            .features
            .iter()
            .all(|&feature| feature < NNUE_NUM_FEATURES));
        assert!(features.features.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn tiny_nnue_model_loads_binary_and_evaluates() {
        let path = std::env::temp_dir().join(format!(
            "tiny_nnue_model_loads_binary_and_evaluates_{}.bin",
            std::process::id()
        ));
        let mut file = File::create(&path).expect("create tiny nnue test file");
        file.write_all(b"TNNUE001").unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&(NNUE_NUM_FEATURES as u32).to_le_bytes())
            .unwrap();
        file.write_all(&(NNUE_NUM_KING_BUCKETS as u32).to_le_bytes())
            .unwrap();
        file.write_all(&1000.0f32.to_le_bytes()).unwrap();

        for _ in 0..NNUE_NUM_FEATURES {
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
        }
        for _ in 0..NNUE_NUM_KING_BUCKETS {
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
        }
        file.write_all(&0.0f32.to_le_bytes()).unwrap();
        file.write_all(&0.5f32.to_le_bytes()).unwrap();
        file.write_all(&2.0f32.to_le_bytes()).unwrap();
        file.write_all(&0.25f32.to_le_bytes()).unwrap();
        drop(file);

        let model = TinyNnueModel::load(&path).expect("load tiny nnue model");
        let score = model.predict_from_position(&shogi_lib::Position::default());
        std::fs::remove_file(&path).ok();

        assert_eq!(model.hidden, 1);
        assert_eq!(score, 1250.0);
    }
}
