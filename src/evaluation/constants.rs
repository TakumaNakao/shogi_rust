use shogi_core::{PieceKind, Square};
use std::sync::LazyLock;

// Piece values are part of the existing evaluator semantics.
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

pub(super) fn piece_kind_value(kind: PieceKind) -> f32 {
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
        PieceKind::King => 0.0,
    }
}

pub(super) fn unpromoted_kind(kind: PieceKind) -> PieceKind {
    match kind {
        PieceKind::ProPawn => PieceKind::Pawn,
        PieceKind::ProLance => PieceKind::Lance,
        PieceKind::ProKnight => PieceKind::Knight,
        PieceKind::ProSilver => PieceKind::Silver,
        PieceKind::ProBishop => PieceKind::Bishop,
        PieceKind::ProRook => PieceKind::Rook,
        other => other,
    }
}

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

// Compact HalfKP feature space. The king file is mirrored so equivalent
// left/right positions share the same transformer rows.
pub const HALFKP_KING_BUCKETS: usize = 5 * 9;
pub const HALFKP_PIECE_STATES: usize = NUM_PIECE_STATES;
pub const HALFKP_INPUTS: usize = HALFKP_KING_BUCKETS * HALFKP_PIECE_STATES;
#[cfg(feature = "halfkp64")]
pub const HALFKP_HIDDEN: usize = 64;
#[cfg(not(feature = "halfkp64"))]
pub const HALFKP_HIDDEN: usize = 32;

pub(super) static BOARD_SQUARES: LazyLock<[Square; NUM_SQUARES]> = LazyLock::new(|| {
    std::array::from_fn(|i| Square::from_u8(i as u8 + 1).expect("valid board square"))
});
