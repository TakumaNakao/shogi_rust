#![allow(dead_code)]
use anyhow::Result;
use shogi_core::{Color, Piece, PieceKind, Square};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use rand::prelude::*;
use rand_distr::Distribution;
use rayon::prelude::*;
use shogi_lib;

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


// --- Evaluator Trait ---
pub trait Evaluator {
    fn evaluate(&self, position: &shogi_lib::Position) -> i32;
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
        PieceKind::Gold => Some(MAX_HAND_PAWNS + MAX_HAND_LANCES + MAX_HAND_KNIGHTS + MAX_HAND_SILVERS),
        PieceKind::Bishop => Some(MAX_HAND_PAWNS + MAX_HAND_LANCES + MAX_HAND_KNIGHTS + MAX_HAND_SILVERS + MAX_HAND_GOLDS),
        PieceKind::Rook => Some(MAX_HAND_PAWNS + MAX_HAND_LANCES + MAX_HAND_KNIGHTS + MAX_HAND_SILVERS + MAX_HAND_GOLDS + MAX_HAND_BISHOPS),
        _ => None,
    }
}

fn piece_to_id(piece: Piece, sq: Option<Square>, hand_index: usize, turn: Color) -> Option<usize> {
    let normalized_color = if piece.color() == turn { Color::Black } else { Color::White };
    let color_offset = if normalized_color == Color::Black { 0 } else { 1 };

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
    let turn = pos.side_to_move();

    let king_sq = match (0..81).find_map(|i| {
        Square::from_u8(i as u8 + 1).and_then(|sq| {
            pos.piece_at(sq).and_then(|p| {
                if p.piece_kind() == PieceKind::King && p.color() == turn {
                    Some(sq)
                } else {
                    None
                }
            })
        })
    }) {
        Some(sq) => sq,
        None => {
            println!("Warning: King not found for side {:?}. Skipping this position.", turn);
            return vec![];
        }
    };

    let normalized_king_sq = if turn == Color::Black { king_sq } else { king_sq.flip() };
    let king_sq_index = (normalized_king_sq.index() - 1) as usize;


    let mut piece_ids = Vec::with_capacity(40);
    for i in 0..81 {
        if let Some(sq) = Square::from_u8(i as u8 + 1) {
            if let Some(piece) = pos.piece_at(sq) {
                if piece.piece_kind() == PieceKind::King && piece.color() == turn {
                    continue; 
                }
                if let Some(id) = piece_to_id(piece, Some(sq), 0, turn) {
                    piece_ids.push(id);
                }
            }
        }
    }
    for color in [Color::Black, Color::White] {
        for kind in ALL_HAND_PIECES.iter() {
            let count = pos.hand(color).count(*kind).unwrap_or(0);
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

    indices
}


const DECAY_SHIFT: u8 = 8; // 256で割るのと同じ
const WEIGHT_LIMIT: i16 = 16384;

#[derive(Default)]
pub struct SparseModel {
    pub w: Vec<i16>,
    pub bias: i16,
    pub kpp_eta: i16,
    // L2 regularisation is not used in Bonanza method
    // pub l2_lambda: f32,
}

impl SparseModel {
    pub fn new(kpp_eta: i16) -> Self {
        Self {
            w: vec![0; MAX_FEATURES],
            bias: 0,
            kpp_eta,
            // l2_lambda,
        }
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let mut offset = 0;

        self.bias = i16::from_le_bytes(buffer[offset..offset + 2].try_into()?);
        offset += 2;

        let expected_w_bytes = MAX_FEATURES * 2;
        if buffer.len() - offset != expected_w_bytes {
            return Err(anyhow::anyhow!("File size mismatch for weights. Expected {} bytes, got {}.", expected_w_bytes + 2, buffer.len()));
        }

        self.w = buffer[offset..]
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path)?;
        let mut buffer = Vec::new();

        buffer.extend_from_slice(&self.bias.to_le_bytes());

        for &v in self.w.iter() {
            buffer.extend_from_slice(&v.to_le_bytes());
        }

        file.write_all(&buffer)?;
        
        println!("Max W: {:?}", self.w.iter().max());
        println!("Min W: {:?}", self.w.iter().min());

        Ok(())
    }

    pub fn initialize_random(&mut self, count: usize, stddev: f32) {
        let mut rng = rand::thread_rng();
        let dist = rand_distr::Normal::new(0.0, stddev).unwrap();
        for _ in 0..count {
            let i = rng.gen_range(0..MAX_FEATURES);
            let v = dist.sample(&mut rng) as i16;
            self.w[i] = v;
        }
    }

    pub fn zero_weight_overwrite(&mut self, overwrite_value: i16) {
        for i in 0..MAX_FEATURES {
            if self.w[i] == 0 {
                self.w[i] = overwrite_value;
            }
        }
    }

    pub fn predict(&self, _pos: &shogi_lib::Position, kpp_features: &[usize]) -> i32 {
        let mut prediction = self.bias as i32;
        for &i in kpp_features {
            if i < MAX_FEATURES {
                prediction += self.w[i] as i32;
            }
        }
        prediction
    }

    pub fn update_batch_for_moves(
        &mut self,
        batch: &[(shogi_lib::Position, shogi_core::Move)],
    ) -> (usize, usize) {
        let total_samples = batch.len();
        if total_samples == 0 {
            return (0, 0);
        }

        let results: Vec<(bool, Vec<usize>, Vec<usize>)> = batch
            .par_iter()
            .map(|(pos, teacher_move)| {
                let legal_moves = pos.legal_moves();
                if legal_moves.is_empty() {
                    return (false, Vec::new(), Vec::new());
                }

                let mut best_move_by_model: Option<shogi_core::Move> = None;
                let mut max_score = i32::MIN;
                let mut best_model_features: Vec<usize> = Vec::new();
                let mut teacher_move_features: Option<Vec<usize>> = None;

                for &mv in legal_moves.iter() {
                    let mut temp_pos = pos.clone();
                    temp_pos.do_move(mv);
                    let features = extract_kpp_features(&temp_pos);
                    let score = self.predict(&temp_pos, &features);

                    if score > max_score {
                        max_score = score;
                        best_move_by_model = Some(mv);
                        best_model_features = features.clone();
                    }

                    if mv == *teacher_move {
                        teacher_move_features = Some(features);
                    }
                }

                let mut is_correct = false;
                let teacher_features = teacher_move_features.unwrap_or_default();

                if let Some(model_move) = best_move_by_model {
                    if model_move == *teacher_move {
                        is_correct = true;
                    }
                }
                (is_correct, teacher_features, best_model_features)
            })
            .collect();

        let mut correct_predictions = 0;
        let mut w_grads: HashMap<usize, i32> = HashMap::new();

        for (is_correct, teacher_features, model_features) in results {
            if is_correct {
                correct_predictions += 1;
            } else {
                for &idx in &teacher_features {
                    if idx < MAX_FEATURES {
                        *w_grads.entry(idx).or_insert(0) += 1;
                    }
                }
                for &idx in &model_features {
                    if idx < MAX_FEATURES {
                        *w_grads.entry(idx).or_insert(0) -= 1;
                    }
                }
            }
        }

        for (idx, total_grad) in w_grads {
            // 1. Gradient-based update
            let update_val = if total_grad > 0 {
                self.kpp_eta
            } else if total_grad < 0 {
                -self.kpp_eta
            } else {
                0
            };
            let mut new_weight = self.w[idx].saturating_add(update_val);

            // 2. More robust proportional decay
            if new_weight != 0 {
                let sign = new_weight.signum();
                let decay = (new_weight.abs() >> DECAY_SHIFT) as i16;
                new_weight -= decay * sign;
            }

            // 3. Weight clipping
            self.w[idx] = new_weight.clamp(-WEIGHT_LIMIT, WEIGHT_LIMIT);
        }

        (correct_predictions, total_samples)
    }
}


pub struct SparseModelEvaluator {
    pub model: SparseModel,
}

impl SparseModelEvaluator {
    pub fn new(weight_path: &Path, overwrite_value: i16) -> Result<Self> {
        let mut model = SparseModel::new(0);
        model.load(weight_path)?;
        model.zero_weight_overwrite(overwrite_value);
        Ok(SparseModelEvaluator { model })
    }
}


impl Evaluator for SparseModelEvaluator {
    fn evaluate(&self, position: &shogi_lib::Position) -> i32 {
        let kpp_features = extract_kpp_features(position);
        self.model.predict(position, &kpp_features)
    }
}

// --- Decoding functions (moved from kpp_weight_check.rs) ---

fn index_to_board_kind(index: usize) -> Option<PieceKind> {
    match index {
        0 => Some(PieceKind::Pawn), 1 => Some(PieceKind::Lance), 2 => Some(PieceKind::Knight),
        3 => Some(PieceKind::Silver), 4 => Some(PieceKind::Gold), 5 => Some(PieceKind::Bishop),
        6 => Some(PieceKind::Rook), 7 => Some(PieceKind::ProPawn), 8 => Some(PieceKind::ProLance),
        9 => Some(PieceKind::ProKnight), 10 => Some(PieceKind::ProSilver), 11 => Some(PieceKind::ProBishop),
        12 => Some(PieceKind::ProRook), 13 => Some(PieceKind::King),
        _ => None,
    }
}

fn index_to_hand_kind_and_offset(index: usize) -> Option<(PieceKind, usize)> {
    let mut current_offset = 0;
    if index < MAX_HAND_PAWNS { return Some((PieceKind::Pawn, index)); }
    current_offset += MAX_HAND_PAWNS;
    if index < current_offset + MAX_HAND_LANCES { return Some((PieceKind::Lance, index - current_offset)); }
    current_offset += MAX_HAND_LANCES;
    if index < current_offset + MAX_HAND_KNIGHTS { return Some((PieceKind::Knight, index - current_offset)); }
    current_offset += MAX_HAND_KNIGHTS;
    if index < current_offset + MAX_HAND_SILVERS { return Some((PieceKind::Silver, index - current_offset)); }
    current_offset += MAX_HAND_SILVERS;
    if index < current_offset + MAX_HAND_GOLDS { return Some((PieceKind::Gold, index - current_offset)); }
    current_offset += MAX_HAND_GOLDS;
    if index < current_offset + MAX_HAND_BISHOPS { return Some((PieceKind::Bishop, index - current_offset)); }
    current_offset += MAX_HAND_BISHOPS;
    if index < current_offset + MAX_HAND_ROOKS { return Some((PieceKind::Rook, index - current_offset)); }
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
        let normalized_color = if color_offset == 0 { Color::Black } else { Color::White };
        Some((piece_kind, Some(normalized_sq), None, normalized_color))
    } else {
        let hand_id = id - board_pieces_total;
        let color_offset = hand_id / NUM_HAND_PIECE_SLOTS_PER_PLAYER;
        let remaining_hand_id = hand_id % NUM_HAND_PIECE_SLOTS_PER_PLAYER;
        let (piece_kind, hand_index) = index_to_hand_kind_and_offset(remaining_hand_id)?;
        let normalized_color = if color_offset == 0 { Color::Black } else { Color::White };
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
    if id1 < id2 { Some((id1, id2)) } else { None }
}

pub type KppInfo = (Square, PieceKind, Option<Square>, Option<usize>, Color, PieceKind, Option<Square>, Option<usize>, Color);

pub fn index_to_kpp_info(index: usize) -> Option<KppInfo> {
    let king_sq_index = index / NUM_PIECE_PAIRS;
    let pair_index = index % NUM_PIECE_PAIRS;
    let king_sq = Square::from_u8((king_sq_index + 1) as u8)?;
    let (id1, id2) = pair_index_to_ids(pair_index)?;
    let (p1k, p1sq, p1hi, p1c) = id_to_piece_info(id1)?;
    let (p2k, p2sq, p2hi, p2c) = id_to_piece_info(id2)?;
    Some((king_sq, p1k, p1sq, p1hi, p1c, p2k, p2sq, p2hi, p2c))
}