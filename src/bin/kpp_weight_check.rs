use std::env;
use std::io::{self};
use shogi_core::{Color, Piece, PieceKind, Square};
use std::path::Path;

// evaluationモジュールから公開された関数と定数を使用する
use shogi_ai::evaluation::{self, index_to_kpp_info, SparseModel};

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
        if let Some(idx) = evaluation::ALL_HAND_PIECES.iter().position(|&k| k == piece1_kind) {
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
        if let Some(idx) = evaluation::ALL_HAND_PIECES.iter().position(|&k| k == piece2_kind) {
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
    for (i, &kind) in evaluation::ALL_HAND_PIECES.iter().enumerate() {
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

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <weight_file_path>", args[0]);
        return Ok(());
    }

    let weight_file_path = &args[1];
    let mut model = SparseModel::new(0.0, 0.0);
    if let Err(e) = model.load(Path::new(weight_file_path)) {
        eprintln!("Error loading weight file: {}", e);
        return Ok(());
    }

    let weights = &model.w;
    println!("Loaded {} weights.", weights.len());

    let mut indices: Vec<usize> = (0..weights.len()).collect();
    indices.sort_by(|&a, &b| weights[a].total_cmp(&weights[b]));

    // --- Display weight statistics ---
    let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_w = weights.iter().cloned().fold(f32::INFINITY, f32::min);
    let non_zero_count = weights.iter().filter(|&&w| w != 0.0).count();
    let total_count = weights.len();
    let sparsity = (non_zero_count as f32 / total_count as f32) * 100.0;

    println!("\n--- 学習完了後の重み統計 ---");
    println!("最大重み: {:.6}", max_w);
    println!("最小重み: {:.6}", min_w);
    println!("非ゼロ要素の割合: {:.4}% ({}/{})", sparsity, non_zero_count, total_count);
    println!("駒得係数: {:.6}", model.material_coeff);
    // --- End of statistics ---

    println!("\n--- Top 10 Weights ---");
    for &index in indices.iter().rev().take(10) {
        let weight = weights[index];
        let turn = if index < evaluation::MAX_FEATURES / 2 { Color::Black } else { Color::White };
        if let Some((king_sq, p1k, p1sq, p1hi, p1c, p2k, p2sq, p2hi, p2c)) = index_to_kpp_info(index) {
            println!("Weight: {:.6}, Index: {} (Turn: {:?})", weight, index, turn);
            println!("  King (Normalized): Black at {:?} (Corresponds to White's King at {:?})", king_sq, king_sq.flip());
            println!("  Piece 1: {:?} {:?} at {:?} (Hand: {:?})", p1c, p1k, p1sq, p1hi);
            println!("  Piece 2: {:?} {:?} at {:?} (Hand: {:?})", p2c, p2k, p2sq, p2hi);
            println!("  SFEN (Normalized): {}", generate_sfen(king_sq, p1k, p1sq, p1hi, p1c, p2k, p2sq, p2hi, p2c, turn));
        } else {
            println!("Failed to decode index: {}", index);
        }
    }

    println!("\n--- Bottom 10 Weights ---");
    for &index in indices.iter().take(10) {
        let weight = weights[index];
        let turn = if index < evaluation::MAX_FEATURES / 2 { Color::Black } else { Color::White };
        if let Some((king_sq, p1k, p1sq, p1hi, p1c, p2k, p2sq, p2hi, p2c)) = index_to_kpp_info(index) {
            println!("Weight: {:.6}, Index: {} (Turn: {:?})", weight, index, turn);
            println!("  King (Normalized): Black at {:?} (Corresponds to White's King at {:?})", king_sq, king_sq.flip());
            println!("  Piece 1: {:?} {:?} at {:?} (Hand: {:?})", p1c, p1k, p1sq, p1hi);
            println!("  Piece 2: {:?} {:?} at {:?} (Hand: {:?})", p2c, p2k, p2sq, p2hi);
            println!("  SFEN (Normalized): {}", generate_sfen(king_sq, p1k, p1sq, p1hi, p1c, p2k, p2sq, p2hi, p2c, turn));
        } else {
            println!("Failed to decode index: {}", index);
        }
    }

    Ok(())
}