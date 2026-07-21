use super::{
    ALL_HAND_PIECES, MAX_HAND_BISHOPS, MAX_HAND_GOLDS, MAX_HAND_KNIGHTS, MAX_HAND_LANCES,
    MAX_HAND_PAWNS, MAX_HAND_ROOKS, MAX_HAND_SILVERS, NUM_BOARD_PIECE_KINDS,
    NUM_HAND_PIECE_SLOTS_PER_PLAYER, NUM_PIECE_PAIRS, NUM_SQUARES,
};
use shogi_core::{Color, Piece, PieceKind, Square};

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
    (id1 < id2).then_some((id1, id2))
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

pub fn piece_kind_to_sfen_char_base(kind: PieceKind, color: Color) -> char {
    let black = color == Color::Black;
    match kind {
        PieceKind::Pawn | PieceKind::ProPawn => {
            if black {
                'P'
            } else {
                'p'
            }
        }
        PieceKind::Lance | PieceKind::ProLance => {
            if black {
                'L'
            } else {
                'l'
            }
        }
        PieceKind::Knight | PieceKind::ProKnight => {
            if black {
                'N'
            } else {
                'n'
            }
        }
        PieceKind::Silver | PieceKind::ProSilver => {
            if black {
                'S'
            } else {
                's'
            }
        }
        PieceKind::Gold => {
            if black {
                'G'
            } else {
                'g'
            }
        }
        PieceKind::Bishop | PieceKind::ProBishop => {
            if black {
                'B'
            } else {
                'b'
            }
        }
        PieceKind::Rook | PieceKind::ProRook => {
            if black {
                'R'
            } else {
                'r'
            }
        }
        PieceKind::King => {
            if black {
                'K'
            } else {
                'k'
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn generate_sfen(
    king_sq: Square,
    piece1_kind: PieceKind,
    piece1_sq: Option<Square>,
    piece1_hand_idx: Option<usize>,
    piece1_color: Color,
    piece2_kind: PieceKind,
    piece2_sq: Option<Square>,
    piece2_hand_idx: Option<usize>,
    piece2_color: Color,
    turn: Color,
) -> String {
    let mut board = vec![vec![None; 9]; 9];
    let mut black_hand_counts = [0; 7];
    let mut white_hand_counts = [0; 7];

    board[king_sq.rank() as usize - 1][king_sq.file() as usize - 1] =
        Some(Piece::new(PieceKind::King, Color::Black));

    for (kind, square, hand_index, color) in [
        (piece1_kind, piece1_sq, piece1_hand_idx, piece1_color),
        (piece2_kind, piece2_sq, piece2_hand_idx, piece2_color),
    ] {
        if let Some(square) = square {
            board[square.rank() as usize - 1][square.file() as usize - 1] =
                Some(Piece::new(kind, color));
        } else if hand_index.is_some() {
            if let Some(index) = ALL_HAND_PIECES
                .iter()
                .position(|&candidate| candidate == kind)
            {
                if color == Color::Black {
                    black_hand_counts[index] += 1;
                } else {
                    white_hand_counts[index] += 1;
                }
            }
        }
    }

    let mut board_text = String::new();
    for (rank, squares) in board.iter().enumerate() {
        let mut empty = 0;
        for piece in squares {
            if let Some(piece) = piece {
                if empty > 0 {
                    board_text.push_str(&empty.to_string());
                    empty = 0;
                }
                if is_promoted_piece_kind(piece.piece_kind()) {
                    board_text.push('+');
                }
                board_text.push(piece_kind_to_sfen_char_base(
                    piece.piece_kind(),
                    piece.color(),
                ));
            } else {
                empty += 1;
            }
        }
        if empty > 0 {
            board_text.push_str(&empty.to_string());
        }
        if rank < 8 {
            board_text.push('/');
        }
    }

    let mut hand_text = String::new();
    for (index, &kind) in ALL_HAND_PIECES.iter().enumerate() {
        for (count, color) in [
            (black_hand_counts[index], Color::Black),
            (white_hand_counts[index], Color::White),
        ] {
            if count > 0 {
                if count > 1 {
                    hand_text.push_str(&count.to_string());
                }
                hand_text.push(piece_kind_to_sfen_char_base(kind, color));
            }
        }
    }
    if hand_text.is_empty() {
        hand_text.push('-');
    }

    let turn = if turn == Color::Black { 'b' } else { 'w' };
    format!("{board_text} {turn} {hand_text} 1")
}
