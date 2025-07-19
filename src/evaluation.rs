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
const MAX_MATERIAL_WEIGHT: f32 = 10.0;

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

pub fn get_piece_value(piece_kind: PieceKind) -> i32 {
    use shogi_core::PieceKind::*;
    match piece_kind {
        Pawn => 100,
        Lance => 300,
        Knight => 300,
        Silver => 500,
        Gold => 600,
        Bishop => 800,
        Rook => 1000,
        King => 20000,   // 実質的に無限大として扱う
        ProPawn | ProLance | ProKnight => 400,
        ProSilver => 600,
        ProBishop => 1200, // 竜馬
        ProRook => 1500,   // 竜王
    }
}

#[derive(Default)]
pub struct SparseModel {
    pub w: Vec<f32>,
    pub bias: f32,
    pub board_piece_values: [f32; NUM_BOARD_PIECE_VALUES],
    pub hand_value_multiplier_raw: f32,
    pub material_weight_raw: f32,
    pub kpp_eta: f32,
    pub material_eta: f32,
    pub material_loss_ratio: f32,
    pub max_gradient: f32,
}

impl SparseModel {
    pub fn new(kpp_eta: f32, material_eta: f32, material_loss_ratio: f32, max_gradient: f32) -> Self {
        const INITIAL_MATERIAL_WEIGHT_RAW: f32 = 0.0;
        const INITIAL_HAND_VALUE_MULTIPLIER_RAW: f32 = -10.0;

        Self {
            w: vec![0.0; MAX_FEATURES],
            bias: 0.0,
            board_piece_values: [
                100.0, 300.0, 350.0, 500.0, 550.0, 800.0, 1000.0, 600.0, 600.0, 600.0, 600.0,
                1000.0, 1200.0,
            ],
            hand_value_multiplier_raw: INITIAL_HAND_VALUE_MULTIPLIER_RAW,
            material_weight_raw: INITIAL_MATERIAL_WEIGHT_RAW,
            kpp_eta,
            material_eta,
            material_loss_ratio,
            max_gradient,
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

        self.hand_value_multiplier_raw = f32::from_le_bytes(buffer[offset..offset + 4].try_into()?);
        offset += 4;

        self.material_weight_raw = f32::from_le_bytes(buffer[offset..offset + 4].try_into()?);
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

        println!("material_weight: {}, material_weight_raw: {}", self.get_material_weight(), self.material_weight_raw);
        println!("board_piece_values: {:?}", self.board_piece_values);
        println!("hand_value_multiplier: {}", self.get_hand_value_multiplier());

        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path)?;
        let mut buffer = Vec::new();

        buffer.extend_from_slice(&self.bias.to_le_bytes());
        for &value in &self.board_piece_values {
            buffer.extend_from_slice(&value.to_le_bytes());
        }
        buffer.extend_from_slice(&self.hand_value_multiplier_raw.to_le_bytes());
        buffer.extend_from_slice(&self.material_weight_raw.to_le_bytes());

        for &v in self.w.iter() {
            buffer.extend_from_slice(&v.to_le_bytes());
        }

        file.write_all(&buffer)?;
        
        println!("material_weight: {}, material_weight_raw: {}", self.get_material_weight(), self.material_weight_raw);
        println!("board_piece_values: {:?}", self.board_piece_values);
        println!("hand_value_multiplier: {}", self.get_hand_value_multiplier());
        println!("Max W: {:?}", self.w.iter().cloned().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)));

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

    fn get_material_weight(&self) -> f32 {
        if self.max_gradient <= 0.0 {
            return MAX_MATERIAL_WEIGHT * (1.0 / (1.0 + (-self.material_weight_raw).exp()));
        }
        let k = (MAX_MATERIAL_WEIGHT * 0.25) / self.max_gradient;
        MAX_MATERIAL_WEIGHT * (1.0 / (1.0 + (-self.material_weight_raw / k).exp()))
    }

    fn get_hand_value_multiplier(&self) -> f32 {
        1.0 + (self.hand_value_multiplier_raw).exp().ln_1p() // softplus
    }

    pub fn predict(&self, pos: &shogi_core::Position, kpp_features: &[usize]) -> f32 {
        let mut prediction = self.bias;
        for &i in kpp_features {
            if i < MAX_FEATURES {
                prediction += self.w[i];
            }
        }
        let material_weight = self.get_material_weight();
        let hand_value_multiplier = self.get_hand_value_multiplier();
        let material_score = calculate_material_score(pos, &self.board_piece_values, hand_value_multiplier);
        
        let material_term = material_weight * material_score;
        prediction += if pos.side_to_move() == Color::Black { material_term } else { -material_term };

        prediction
    }

    pub fn update_batch(&mut self, batch: &[(shogi_core::Position, Vec<usize>, f32)]) -> (f32, f32, f32) {
        let m = batch.len() as f32;
        if m == 0.0 {
            return (0.0, 0.0, 0.0);
        }
        let mut total_loss = 0.0;
        let mut kpp_loss = 0.0;
        let mut material_loss = 0.0;

        let mut bias_grad = 0.0;
        let mut w_grads = vec![0.0; MAX_FEATURES];

        let mut board_piece_values_grads = [0.0; NUM_BOARD_PIECE_VALUES];
        let mut hand_value_multiplier_raw_grad = 0.0;
        let mut material_weight_raw_grad = 0.0;

        for (pos, kpp_features, y_true) in batch.iter() {
            let mut kpp_score = self.bias;
            for &i in kpp_features {
                if i < MAX_FEATURES {
                    kpp_score += self.w[i];
                }
            }
            
            let material_weight = self.get_material_weight();
            let hand_value_multiplier = self.get_hand_value_multiplier();
            let raw_material_score = calculate_material_score(pos, &self.board_piece_values, hand_value_multiplier);
            let material_score = material_weight * raw_material_score;
            let material_score_for_turn = if pos.side_to_move() == Color::Black { material_score } else { -material_score };

            let y_pred = kpp_score + material_score_for_turn;
            total_loss += (y_pred - y_true) * (y_pred - y_true);

            let y_kpp_true = y_true * (1.0 - self.material_loss_ratio);
            let error_kpp = kpp_score - y_kpp_true;
            kpp_loss += error_kpp * error_kpp;
            let error_grad_kpp = 2.0 * error_kpp / m;
            
            bias_grad += error_grad_kpp;
            for &i in kpp_features {
                if i < MAX_FEATURES {
                    w_grads[i] += error_grad_kpp;
                }
            }

            let y_material_true = y_true * self.material_loss_ratio;
            let error_material = material_score_for_turn - y_material_true;
            material_loss += error_material * error_material;
            let error_grad_material = 2.0 * error_material / m;

            let material_grad_sign = if pos.side_to_move() == Color::Black { 1.0 } else { -1.0 };

            let d_loss_d_weight = error_grad_material * raw_material_score * material_grad_sign;
            let k = if self.max_gradient <= 0.0 { 1.0 } else { (MAX_MATERIAL_WEIGHT * 0.25) / self.max_gradient };
            let sigmoid_input = self.material_weight_raw / k;
            let sigmoid_val = 1.0 / (1.0 + (-sigmoid_input).exp());
            let d_sigmoid_d_input = sigmoid_val * (1.0 - sigmoid_val);
            let d_weight_d_raw = (MAX_MATERIAL_WEIGHT / k) * d_sigmoid_d_input;
            material_weight_raw_grad += d_loss_d_weight * d_weight_d_raw;

            let (board_counts, hand_counts) = get_piece_counts(pos);
            let mut hand_material_for_grad = 0.0;
            for i in 0..NUM_HAND_PIECE_VALUES {
                if let Some(board_idx) = hand_kind_to_board_index(ALL_HAND_PIECES[i]) {
                    hand_material_for_grad += (hand_counts[i] as f32) * self.board_piece_values[board_idx];
                }
            }

            let d_C_d_raw = 1.0 / (1.0 + (-self.hand_value_multiplier_raw).exp()); // sigmoid
            let d_loss_d_C = error_grad_material * material_weight * hand_material_for_grad * material_grad_sign;
            hand_value_multiplier_raw_grad += d_loss_d_C * d_C_d_raw;

            for i in 1..NUM_BOARD_PIECE_VALUES { // Skip Pawn
                let mut hand_count_for_board_piece = 0;
                if let Some(hand_idx) = board_index_to_hand_index(i) {
                    hand_count_for_board_piece = hand_counts[hand_idx];
                }
                let total_count = (board_counts[i] as f32) + hand_value_multiplier * (hand_count_for_board_piece as f32);
                board_piece_values_grads[i] += error_grad_material * material_weight * total_count * material_grad_sign;
            }
        }

        self.bias -= self.kpp_eta * bias_grad;
        for i in 0..MAX_FEATURES {
            self.w[i] -= self.kpp_eta * w_grads[i];
        }
        
        for i in 1..NUM_BOARD_PIECE_VALUES { // Skip Pawn
            self.board_piece_values[i] -= self.material_eta * board_piece_values_grads[i];
        }
        self.hand_value_multiplier_raw -= self.material_eta * hand_value_multiplier_raw_grad;
        self.material_weight_raw -= self.material_eta * material_weight_raw_grad;

        (total_loss / m, kpp_loss / m, material_loss / m)
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
                + (normalized_sq.index() as usize);
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
    let king_sq_index = normalized_king_sq.index() as usize;


    let mut piece_ids = Vec::with_capacity(40);
    for i in 0..81 {
        if let Some(sq) = Square::from_u8(i as u8 + 1) {
            if let Some(piece) = pos.piece_at(sq) {
                if let Some(id) = piece_to_id(piece, Some(sq), 0, turn) {
                    piece_ids.push(id);
                }
            }
        }
    }
    for color in [Color::Black, Color::White] {
        for kind in ALL_HAND_PIECES.iter() {
            let count = pos.hand_of_a_player(color).count(*kind).unwrap_or(0);
            for i in 0..count {
                if let Some(id) = piece_to_id(Piece::new(*kind, color), None, i as usize, turn) {
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

pub fn get_piece_counts(position: &shogi_core::Position) -> ([i8; NUM_BOARD_PIECE_VALUES], [i8; NUM_HAND_PIECE_VALUES]) {
    let mut board_counts: [i8; NUM_BOARD_PIECE_VALUES] = [0; NUM_BOARD_PIECE_VALUES];
    let mut hand_counts: [i8; NUM_HAND_PIECE_VALUES] = [0; NUM_HAND_PIECE_VALUES];

    for piece_kind in PieceKind::all() {
        if piece_kind == PieceKind::King { continue; }
        if let Some(index) = board_kind_to_index(piece_kind) {
            let black_count = position.piece_bitboard(Piece::new(piece_kind, Color::Black)).count();
            let white_count = position.piece_bitboard(Piece::new(piece_kind, Color::White)).count();
            board_counts[index] = (black_count as i8) - (white_count as i8);
        }
    }

    for (index, &piece_kind) in ALL_HAND_PIECES.iter().enumerate() {
        let black_count = position.hand_of_a_player(Color::Black).count(piece_kind).unwrap_or(0);
        let white_count = position.hand_of_a_player(Color::White).count(piece_kind).unwrap_or(0);
        hand_counts[index] = (black_count as i8) - (white_count as i8);
    }
    (board_counts, hand_counts)
}

fn hand_kind_to_board_index(kind: PieceKind) -> Option<usize> {
    board_kind_to_index(kind)
}

fn board_index_to_hand_index(index: usize) -> Option<usize> {
    match index {
        0 => Some(0), // Pawn
        1 => Some(1), // Lance
        2 => Some(2), // Knight
        3 => Some(3), // Silver
        4 => Some(4), // Gold
        5 => Some(5), // Bishop
        6 => Some(6), // Rook
        _ => None,
    }
}

fn calculate_material_score(
    position: &shogi_core::Position, 
    board_piece_values: &[f32; NUM_BOARD_PIECE_VALUES], 
    hand_value_multiplier: f32
) -> f32 {
    let mut score = 0.0;
    let (board_counts, hand_counts) = get_piece_counts(position);
    for i in 0..NUM_BOARD_PIECE_VALUES {
        score += board_counts[i] as f32 * board_piece_values[i];
    }
    for i in 0..NUM_HAND_PIECE_VALUES {
        if let Some(board_idx) = hand_kind_to_board_index(ALL_HAND_PIECES[i]) {
            score += hand_counts[i] as f32 * board_piece_values[board_idx] * hand_value_multiplier;
        }
    }
    score
}

pub struct SparseModelEvaluator {
    pub model: SparseModel,
}

impl SparseModelEvaluator {
    pub fn new(weight_path: &Path) -> Result<Self> {
        let mut model = SparseModel::new(0.0, 0.0, 0.5, 1.0);
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

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_core::{Position, PieceKind, PartialPosition, Color};

    fn create_test_model() -> SparseModel {
        let mut model = SparseModel::new(0.0, 0.0, 0.5, 1.0);
        model.bias = 0.0;
        model.w = vec![0.0; MAX_FEATURES];
        model.board_piece_values = [
            1.0, 3.0, 3.5, 5.0, 5.5, 8.0, 10.0,
            6.0, 6.0, 6.0, 6.0, 10.0, 12.0,
        ];
        model.hand_value_multiplier_raw = 0.0; // multiplier will be > 1.0
        model.material_weight_raw = 10.0;
        model
    }

    #[test]
    fn test_initial_position_evaluation() {
        let model = create_test_model();
        let pos = Position::default();
        let features = extract_kpp_features(&pos);
        let score = model.predict(&pos, &features);
        assert!(score.abs() < 0.1, "Initial score should be close to 0, but was {}", score);
    }

    #[test]
    fn test_material_advantage_evaluation() {
        let model = create_test_model();
        let mut partial_pos = PartialPosition::startpos();

        let black_hand = partial_pos.hand_of_a_player_mut(Color::Black);
        *black_hand = black_hand.added(PieceKind::Rook).unwrap();
        *black_hand = black_hand.added(PieceKind::Bishop).unwrap();
        
        let pos = Position::arbitrary_position(partial_pos);

        let features = extract_kpp_features(&pos);
        let score = model.predict(&pos, &features);

        let material_weight = model.get_material_weight();
        let hand_multiplier = model.get_hand_value_multiplier();
        let rook_val = model.board_piece_values[board_kind_to_index(PieceKind::Rook).unwrap()];
        let bishop_val = model.board_piece_values[board_kind_to_index(PieceKind::Bishop).unwrap()];
        let expected_material_score = material_weight * (rook_val + bishop_val) * hand_multiplier;
        
        assert!(score > expected_material_score * 0.9, "Score ({}) should strongly reflect material advantage ({})", score, expected_material_score);
    }

    #[test]
    fn test_evaluation_is_from_side_to_move_perspective() {
        let model = create_test_model();
        
        let mut partial_pos_black = PartialPosition::startpos();
        let black_hand = partial_pos_black.hand_of_a_player_mut(Color::Black);
        *black_hand = black_hand.added(PieceKind::Pawn).unwrap();
        
        let pos_black_turn = Position::arbitrary_position(partial_pos_black.clone());

        let mut partial_pos_white = partial_pos_black;
        partial_pos_white.side_to_move_set(Color::White);
        let pos_white_turn = Position::arbitrary_position(partial_pos_white);

        let features_black = extract_kpp_features(&pos_black_turn);
        let score_black_turn = model.predict(&pos_black_turn, &features_black);

        let features_white = extract_kpp_features(&pos_white_turn);
        let score_white_turn = model.predict(&pos_white_turn, &features_white);
        
        assert!((score_black_turn + score_white_turn).abs() < 0.1, "White's score ({}) should be close to the negative of Black's score ({})", score_white_turn, score_black_turn);
    }
}