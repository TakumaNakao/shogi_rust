use anyhow::Result;
use shogi_core::{Color, Move, Piece, PieceKind, Position, Square};
use csa;
use std::fs;
use std::env;
use std::path::Path;
use std::time::Instant;

mod evaluation;

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

fn process_csa_file(path: &Path, batch: &mut Vec<(Vec<usize>, f32)>) -> Result<()> {
    let text = fs::read_to_string(path)?;
    let record = csa::parse_csa(&text)?;

    let last_move = if let Some(mv) = record.moves.last() {
        mv
    } else {
        return Ok(()); // Skip games with no moves
    };

    let winner: Option<csa::Color> = match last_move.action {
        csa::Action::Toryo | csa::Action::Tsumi => {
            if let Some(prev) = record.moves.iter().rev().find(|m| matches!(m.action, csa::Action::Move(_, _, _, _))) {
                if let csa::Action::Move(color, _, _, _) = prev.action {
                    Some(color)
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    };

    let final_label = match winner {
        Some(csa::Color::Black) => 1.0,
        Some(csa::Color::White) => -1.0,
        None => return Ok(()), // Skip games with no winner
    };

    let mut pos = Position::default();

    for (index, mv) in record.moves.iter().enumerate() {
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
        let gain = index as f32 / (index as f32 + 25.0); // REWARD_GAINを直接指定
        let label = gain * final_label;

        let features = evaluation::extract_kpp_features(&pos);
        if !features.is_empty() {
            batch.push((features, label));
        }

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

    println!("--- ベンチマーク ---");
    let mut batch = Vec::new(); // バッチを初期化
    let start_time_old = Instant::now();
    let mut processed_files = 0;
    for path in &csa_files {
        if let Err(e) = process_csa_file(&path, &mut batch) {
            eprintln!("ファイル処理エラー: {:?} - {}", path, e);
        }
        processed_files += 1;
        if processed_files % 1000 == 0 {
            println!("処理済みファイル数: {}", processed_files);
        }
    }
    let elapsed_time = start_time_old.elapsed();
    println!("全ファイルの処理にかかった時間: {:?}", elapsed_time);

    Ok(())
}
