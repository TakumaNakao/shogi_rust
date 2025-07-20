use std::env;
use std::fs::File;
use std::io::{self, Read};
use shogi_core::{Color, Piece, PieceKind, Square};

// --- Constants from evaluation.rs ---
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

const ALL_HAND_PIECES: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

// --- Helper functions to reverse mappings ---

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

// This function reverses the `piece_to_id` logic
fn id_to_piece_info(id: usize) -> Option<(PieceKind, Option<Square>, Option<usize>, Color)> {
    let board_pieces_total = NUM_BOARD_PIECE_KINDS * NUM_SQUARES * 2;

    if id < board_pieces_total {
        // Board piece
        let color_offset = id / (NUM_BOARD_PIECE_KINDS * NUM_SQUARES);
        let remaining_id = id % (NUM_BOARD_PIECE_KINDS * NUM_SQUARES);
        let kind_index = remaining_id / NUM_SQUARES;
        let sq_index = remaining_id % NUM_SQUARES;

        let piece_kind = index_to_board_kind(kind_index)?;
        let normalized_sq = Square::from_u8(sq_index as u8 + 1)?;
        let normalized_color = if color_offset == 0 { Color::Black } else { Color::White };

        Some((piece_kind, Some(normalized_sq), None, normalized_color))
    } else {
        // Hand piece
        let hand_id = id - board_pieces_total;
        let color_offset = hand_id / NUM_HAND_PIECE_SLOTS_PER_PLAYER;
        let remaining_hand_id = hand_id % NUM_HAND_PIECE_SLOTS_PER_PLAYER;

        let (piece_kind, hand_index) = index_to_hand_kind_and_offset(remaining_hand_id)?;
        let normalized_color = if color_offset == 0 { Color::Black } else { Color::White };

        Some((piece_kind, None, Some(hand_index), normalized_color))
    }
}

// This function reverses the `pair_index` calculation
fn pair_index_to_ids(pair_index: usize) -> Option<(usize, usize)> {
    let mut id2 = 0;
    // Find id2 such that id2 * (id2 - 1) / 2 is the largest triangle number less than or equal to pair_index
    // This is essentially solving for id2 in id2 * (id2 - 1) / 2 = pair_index
    // Approximately id2^2 / 2 = pair_index => id2 = sqrt(2 * pair_index)
    // We can iterate or use a more direct calculation if needed.
    // For now, a simple loop is fine given the constraints.
    while id2 * (id2 - 1) / 2 <= pair_index {
        id2 += 1;
    }
    id2 -= 1; // Adjust to find the correct id2

    let pair_index_base = id2 * (id2 - 1) / 2;
    let id1 = pair_index - pair_index_base;

    if id1 < id2 {
        Some((id1, id2))
    } else {
        None
    }
}

// This function reverses the `final_index` calculation
fn index_to_kpp_info(index: usize) -> Option<(Square, PieceKind, Option<Square>, Option<usize>, Color, PieceKind, Option<Square>, Option<usize>, Color)> {
    let king_sq_index = index / NUM_PIECE_PAIRS;
    let pair_index = index % NUM_PIECE_PAIRS;

    let king_sq = Square::from_u8(king_sq_index as u8 + 1)?;

    let (id1, id2) = pair_index_to_ids(pair_index)?;

    let (piece1_kind, piece1_normalized_sq, piece1_hand_idx, piece1_normalized_color) = id_to_piece_info(id1)?;
    let (piece2_kind, piece2_normalized_sq, piece2_hand_idx, piece2_normalized_color) = id_to_piece_info(id2)?;

    Some((king_sq, piece1_kind, piece1_normalized_sq, piece1_hand_idx, piece1_normalized_color, piece2_kind, piece2_normalized_sq, piece2_hand_idx, piece2_normalized_color))
}

// Helper for SFEN character conversion
fn is_promoted_piece_kind(kind: PieceKind) -> bool {
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
fn piece_kind_to_sfen_char_base(kind: PieceKind, color: Color) -> char {
    match kind {
        PieceKind::Pawn | PieceKind::ProPawn => if color == Color::Black { 'P' } else { 'p' },
        PieceKind::Lance | PieceKind::ProLance => if color == Color::Black { 'L' } else { 'l' },
        PieceKind::Knight | PieceKind::ProKnight => if color == Color::Black { 'N' } else { 'n' },
        PieceKind::Silver | PieceKind::ProSilver => if color == Color::Black { 'S' } else { 's' },
        PieceKind::Gold => if color == Color::Black { 'G' } else { 'g' },
        PieceKind::Bishop | PieceKind::ProBishop => if color == Color::Black { 'B' } else { 'b' },
        PieceKind::Rook | PieceKind::ProRook => if color == Color::Black { 'R' } else { 'r' },
        PieceKind::King => if color == Color::Black { 'K' } else { 'k' },
    }
}

// Function to generate SFEN from KPP info
fn generate_sfen(
    king_sq: Square,
    piece1_kind: PieceKind,
    piece1_sq: Option<Square>,
    piece1_hand_idx: Option<usize>,
    piece1_color: Color, // This is the normalized color
    piece2_kind: PieceKind,
    piece2_sq: Option<Square>,
    piece2_hand_idx: Option<usize>,
    piece2_color: Color // This is the normalized color
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
                sfen_board_str.push(piece_kind_to_sfen_char_base(piece.piece_kind(), piece.color()));
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

    // Always 'b' for Black to move, as we normalized to Black's perspective
    format!("{} {} b 1", sfen_board_str, sfen_hand_str)
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <weight_file_path>", args[0]);
        return Ok(());
    }

    let weight_file_path = &args[1];
    let mut file = File::open(weight_file_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Assuming weights are f32 and stored contiguously
    // Skip the bias (first 4 bytes)
    let weights: Vec<f32> = buffer[4..]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    println!("Loaded {} weights.", weights.len());

    let mut indexed_weights: Vec<(usize, f32)> = weights.iter().enumerate().map(|(i, &w)| (i, w)).collect();

    // Sort by weight value
    indexed_weights.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n--- Top 5 Weights ---");
    for i in (indexed_weights.len().saturating_sub(5)..indexed_weights.len()).rev() {
        let (index, weight) = indexed_weights[i];
        if let Some((king_sq, piece1_kind, piece1_sq, piece1_hand_idx, piece1_color, piece2_kind, piece2_sq, piece2_hand_idx, piece2_color)) = index_to_kpp_info(index) {
            println!("Weight: {:.6}, Index: {}", weight, index);
            println!("  King: {:?} {:?}", Color::Black, king_sq); // King is always Black in normalized SFEN
            println!("  Piece 1: {:?} {:?} at {:?} (Hand: {:?})", piece1_color, piece1_kind, piece1_sq, piece1_hand_idx);
            println!("  Piece 2: {:?} {:?} at {:?} (Hand: {:?})", piece2_color, piece2_kind, piece2_sq, piece2_hand_idx);
            println!("  SFEN: {}", generate_sfen(king_sq, piece1_kind, piece1_sq, piece1_hand_idx, piece1_color, piece2_kind, piece2_sq, piece2_hand_idx, piece2_color));
        } else {
            println!("Failed to decode index: {}", index);
        }
    }

    println!("\n--- Bottom 5 Weights ---");
    for i in 0..std::cmp::min(5, indexed_weights.len()) {
        let (index, weight) = indexed_weights[i];
        if let Some((king_sq, piece1_kind, piece1_sq, piece1_hand_idx, piece1_color, piece2_kind, piece2_sq, piece2_hand_idx, piece2_color)) = index_to_kpp_info(index) {
            println!("Weight: {:.6}, Index: {}", weight, index);
            println!("  King: {:?} {:?}", Color::Black, king_sq); // King is always Black in normalized SFEN
            println!("  Piece 1: {:?} {:?} at {:?} (Hand: {:?})", piece1_color, piece1_kind, piece1_sq, piece1_hand_idx);
            println!("  Piece 2: {:?} {:?} at {:?} (Hand: {:?})", piece2_color, piece2_kind, piece2_sq, piece2_hand_idx);
            println!("  SFEN: {}", generate_sfen(king_sq, piece1_kind, piece1_sq, piece1_hand_idx, piece1_color, piece2_kind, piece2_sq, piece2_hand_idx, piece2_color));
        } else {
            println!("Failed to decode index: {}", index);
        }
    }

    Ok(())
}