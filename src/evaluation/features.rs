use super::constants::*;
use shogi_core::{Color, Piece, PieceKind, Square};

#[derive(Debug, Clone)]
pub struct NnueFeatures {
    pub king_bucket: usize,
    pub features: Vec<usize>,
    pub material: f32,
}

pub(super) fn board_kind_to_index(kind: PieceKind) -> Option<usize> {
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

pub(super) fn hand_kind_to_offset(kind: PieceKind) -> Option<usize> {
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

pub(super) fn piece_to_id(
    piece: Piece,
    sq: Option<Square>,
    hand_index: usize,
    turn: Color,
) -> Option<usize> {
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
            indices.push(king_sq_index * NUM_PIECE_PAIRS + pair_index);
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
