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
const NUM_HAND_PIECE_KINDS: usize = 7;

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

pub const KPP_DIM: usize = 193_000_000;
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

#[derive(Default)]
pub struct SparseModel {
    pub w: Vec<f32>,
    pub bias: f32,
    pub eta: f32,
}

impl SparseModel {
    pub fn new(eta: f32) -> Self {
        Self {
            w: vec![0.0; MAX_FEATURES],
            bias: 0.0,
            eta,
        }
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let mut file = File::open(path)?;
        let mut buffer = [0u8; 4];

        file.read_exact(&mut buffer)?;
        self.bias = f32::from_le_bytes(buffer);

        let mut weights_bytes = Vec::new();
        file.read_to_end(&mut weights_bytes)?;

        if weights_bytes.len() != MAX_FEATURES * 4 {
            return Err(anyhow::anyhow!("File size mismatch for weights"));
        }

        for i in 0..MAX_FEATURES {
            let start = i * 4;
            let end = start + 4;
            let bytes: [u8; 4] = weights_bytes[start..end].try_into()
                .map_err(|_| anyhow::anyhow!("Failed to convert slice to array"))?;
            self.w[i] = f32::from_le_bytes(bytes);
        }
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(&self.bias.to_le_bytes())?;

        let mut buffer = Vec::with_capacity(self.w.len() * 4);
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

    pub fn predict(&self, x: &[usize]) -> f32 {
        let mut prediction = self.bias;
        for &i in x {
            if i < MAX_FEATURES {
                prediction += self.w[i];
            }
        }
        prediction
    }

    pub fn update_batch(&mut self, batch: &[(Vec<usize>, f32)]) -> f32 {
        let m = batch.len() as f32;
        if m == 0.0 {
            return 0.0;
        }
        let mut total_loss = 0.0;
        for (x, y_true) in batch.iter() {
            let y_pred = self.predict(x);
            let error = y_pred - y_true;
            total_loss += error * error;

            self.bias -= self.eta * error / m;

            for &i in x {
                if i < MAX_FEATURES {
                    self.w[i] -= self.eta * error / m;
                }
            }
        }
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
        let features = extract_kpp_features(position);
        self.model.predict(&features)
    }
}