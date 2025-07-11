use anyhow::Result;
use shogi_core::{Color, Move, Piece, PieceKind, Position, Square};
use csa;
use std::fs;
use std::env;
use std::path::Path;
use std::time::Instant;

mod evaluation;
mod old_evaluation;
use evaluation::{KppState};

// kpp_learn.rsからコピー
fn csa_to_shogi_piece_kind(csa_piece_type: csa::PieceType) -> PieceKind {
    match csa_piece_type {
        csa::PieceType::Pawn => PieceKind::Pawn,
        csa::PieceType::Lance => PieceKind::Lance,
        csa::PieceType::Knight => PieceKind::Knight,
        csa::PieceType::Silver => PieceKind::Silver,
        csa::PieceType::Gold => PieceKind::Gold,
        csa::PieceType::Bishop => PieceKind::Bishop,
        csa::PieceType::Rook => PieceKind::Rook,
        csa::PieceType::King => PieceKind::King,
        csa::PieceType::ProPawn => PieceKind::ProPawn,
        csa::PieceType::ProLance => PieceKind::ProLance,
        csa::PieceType::ProKnight => PieceKind::ProKnight,
        csa::PieceType::ProSilver => PieceKind::ProSilver,
        csa::PieceType::Horse => PieceKind::ProBishop,
        csa::PieceType::Dragon => PieceKind::ProRook,
        // csa::PieceType::All => unreachable!(), // This variant does not exist
        _ => unreachable!(), // Handle any other unexpected variants
    }
}

fn process_csa_file(path: &Path) -> Result<()> {
    let text = fs::read_to_string(path)?;
    let record = csa::parse_csa(&text)?;

    let mut pos = Position::default();
    let mut kpp_state = if let Some(state) = KppState::new(&pos) {
        state
    } else {
        // 初期局面でKPP状態を生成できなければスキップ
        return Ok(());
    };

    for mv in record.moves.iter() {
        let shogi_move = match &mv.action {
            csa::Action::Move(color, from_csa, to_csa, piece_type_after_csa) => {
                let to_sq = if let Some(sq) = Square::new(to_csa.file, to_csa.rank) {
                    sq
                } else {
                    continue;
                };

                if from_csa.file == 0 && from_csa.rank == 0 {
                    let piece_kind = csa_to_shogi_piece_kind(*piece_type_after_csa);
                    let piece_color = if *color == csa::Color::Black { Color::Black } else { Color::White };
                    Move::Drop {
                        piece: Piece::new(piece_kind, piece_color),
                        to: to_sq,
                    }
                } else {
                    let from_sq = if let Some(sq) = Square::new(from_csa.file, from_csa.rank) {
                        sq
                    } else {
                        eprintln!("Warning: No 'from' square in CSA file. Skipping move.");
                        continue;
                    };
                    let piece_before = if let Some(p) = pos.piece_at(from_sq) {
                        p
                    } else {
                        eprintln!("Warning: No piece at 'from' square in CSA file. Skipping move.");
                        continue;
                    };
                    let promote = piece_before.piece_kind() != csa_to_shogi_piece_kind(*piece_type_after_csa);
                    Move::Normal {
                        from: from_sq,
                        to: to_sq,
                        promote,
                    }
                }
            }
            _ => continue,
        };

        // 特徴ベクトルを実際に使用しないが、生成は行う
        let _features = kpp_state.features.iter().cloned().collect::<Vec<usize>>();

        let old_pos = pos.clone();
        if pos.make_move(shogi_move).is_none() {
            continue;
        }
        kpp_state.update(&old_pos, &shogi_move);
    }
    Ok(())
}

fn process_csa_file_old(path: &Path) -> Result<()> {
    let text = fs::read_to_string(path)?;
    let record = csa::parse_csa(&text)?;

    let mut pos = Position::default();

    for mv in record.moves.iter() {
        let shogi_move = match &mv.action {
            csa::Action::Move(color, from_csa, to_csa, piece_type_after_csa) => {
                let to_sq = if let Some(sq) = Square::new(to_csa.file, to_csa.rank) {
                    sq
                } else {
                    continue;
                };

                if from_csa.file == 0 && from_csa.rank == 0 {
                    let piece_kind = csa_to_shogi_piece_kind(*piece_type_after_csa);
                    let piece_color = if *color == csa::Color::Black { Color::Black } else { Color::White };
                    Move::Drop {
                        piece: Piece::new(piece_kind, piece_color),
                        to: to_sq,
                    }
                } else {
                    let from_sq = if let Some(sq) = Square::new(from_csa.file, from_csa.rank) {
                        sq
                    } else {
                        eprintln!("Warning: No 'from' square in CSA file. Skipping move.");
                        continue;
                    };
                    let piece_before = if let Some(p) = pos.piece_at(from_sq) {
                        p
                    } else {
                        eprintln!("Warning: No piece at 'from' square in CSA file. Skipping move.");
                        continue;
                    };
                    let promote = piece_before.piece_kind() != csa_to_shogi_piece_kind(*piece_type_after_csa);
                    Move::Normal {
                        from: from_sq,
                        to: to_sq,
                        promote,
                    }
                }
            }
            _ => continue,
        };

        // 特徴ベクトルを実際に使用しないが、生成は行う
        let _features = old_evaluation::extract_kpp_features(&pos);

        if pos.make_move(shogi_move).is_none() {
            continue;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("使用法: {} <csa_directory_path>", args[0]);
        return Ok(());
    }
    let csa_dir_path = Path::new(&args[1]);

    if !csa_dir_path.is_dir() {
        eprintln!("エラー: 指定されたパスはディレクトリではありません: {:?}", csa_dir_path);
        return Ok(());
    }

    let csa_files: Vec<_> = fs::read_dir(csa_dir_path)?
        .filter_map(|entry| {
            entry.ok().and_then(|e| {
                let path = e.path();
                if path.extension().map(|s| s == "csa").unwrap_or(false) {
                    Some(path)
                } else {
                    None
                }
            })
        })
        .collect();
    
    println!("{}個のCSAファイルを処理します。", csa_files.len());

    // 新しい評価方法のベンチマーク
    println!("\n--- 新しい評価方法のベンチマーク ---");
    let start_time_new = Instant::now();
    let mut processed_files_new = 0;
    for path in &csa_files {
        if let Err(e) = process_csa_file(&path) {
            eprintln!("ファイル処理エラー (新): {:?} - {}", path, e);
        }
        processed_files_new += 1;
        if processed_files_new % 1000 == 0 {
            println!("処理済みファイル数 (新): {}", processed_files_new);
        }
    }
    let elapsed_time_new = start_time_new.elapsed();
    println!("全ファイルの処理にかかった時間 (新): {:?}", elapsed_time_new);

    // 古い評価方法のベンチマーク
    println!("\n--- 古い評価方法のベンチマーク ---");
    let start_time_old = Instant::now();
    let mut processed_files_old = 0;
    for path in &csa_files {
        if let Err(e) = process_csa_file_old(&path) {
            eprintln!("ファイル処理エラー (旧): {:?} - {}", path, e);
        }
        processed_files_old += 1;
        if processed_files_old % 1000 == 0 {
            println!("処理済みファイル数 (旧): {}", processed_files_old);
        }
    }
    let elapsed_time_old = start_time_old.elapsed();
    println!("全ファイルの処理にかかった時間 (旧): {:?}", elapsed_time_old);

    Ok(())
}