use anyhow::Result;
use shogi_core::{Color, Move, Piece, PieceKind, Position, Square};
use csa;
use std::fs;
use std::env;
use std::path::Path;
use std::time::Instant;
use rand::prelude::*;
use plotters::prelude::*;

mod evaluation;
use evaluation::{SparseModel, extract_kpp_features};

const BATCH_SIZE: usize = 16768;

const REWARD_GAIN: f32 = 25.0;

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
        csa::PieceType::All => unreachable!(),
    }
}

fn process_csa_file(path: &Path, _model: &mut SparseModel, batch: &mut Vec<(Vec<usize>, f32)>) -> Result<()> {
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
                        println!{"Error from_sq"};
                        continue;
                    };
                    let piece_before = if let Some(p) = pos.piece_at(from_sq) {
                        p
                    } else {
                        println!{"Error piece_before"};
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

        let gain = index as f32 / (index as f32 + REWARD_GAIN);
        let label = gain * final_label;

        let features = extract_kpp_features(&pos);
        if !features.is_empty() {
            batch.push((features, label));
        }
        if pos.make_move(shogi_move).is_none() {
            break;
        }
    }
    Ok(())
}

fn draw_mse_graph(data: &[(usize, f32)], path: &str) -> Result<()> {
    let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    if data.is_empty() {
        return Ok(());
    }

    let max_mse = data.iter().map(|&(_, mse)| mse).fold(f32::NAN, f32::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("MSE per Batch", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..data.len(), 0f32..max_mse)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().enumerate().map(|(i, &(_, mse))| (i, mse)),
        &RED,
    ))?;

    root.present()?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("使用法: {} <year>", args[0]);
        return Ok(());
    }
    let year = String::from(&args[1]);

    let data_dir_str = format!("./csa_files/{}", year);
    let data_dir = Path::new(&data_dir_str);
    let weight_path = Path::new("./weights.csv");
    let mse_graph_path = "mse_graph.png";

    let mut model = SparseModel::new(0.0000001); // learning_rateを直接渡す

    if weight_path.exists() {
        model.load(weight_path)?;
        println!("重みファイルを読み込みました。");
    } else {
        println!("重みファイルが存在しません。初期化中...");
        model.initialize_random(50_000, 0.01);
        println!("保存中...");
        model.save(weight_path)?;
        println!("初期重みを保存しました。");
    }

    let mut csa_files: Vec<_> = fs::read_dir(data_dir)?
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
    
    let mut rng = thread_rng();
    csa_files.shuffle(&mut rng);

    println!("{}個のCSAファイルを読み込みます。", csa_files.len());

    let mut batch = Vec::with_capacity(BATCH_SIZE);
    let mut batch_count = 0;
    let mut file_count = 0;
    let mut mse_history = Vec::new();

    for path in &csa_files {
        let start_time_file = Instant::now();
        if let Err(e) = process_csa_file(&path, &mut model, &mut batch) {
            eprintln!("ファイル処理エラー: {:?} - {}", path, e);
        }
        file_count += 1;
        let elapsed_time_file = start_time_file.elapsed();
        println!("処理済みファイル数: {} / {} , 処理時間: {:?}", file_count, csa_files.len(), elapsed_time_file);
        
        if batch.len() >= BATCH_SIZE {
            let start_time_batch = Instant::now();
            let mse = model.update_batch(&batch, batch_count);
            batch_count += 1;
            batch.clear();
            let elapsed_time_batch = start_time_batch.elapsed();
            println!("バッチ {}: 平均二乗誤差 = {:.6} , 処理時間: {:?}", batch_count, mse, elapsed_time_batch);
            model.save(weight_path)?;
            mse_history.push((batch_count, mse));
            draw_mse_graph(&mse_history, mse_graph_path)?;
        }
    }

    if !batch.is_empty() {
        let start_time_batch = Instant::now();
        let mse = model.update_batch(&batch, batch_count);
        let elapsed_time_batch = start_time_batch.elapsed();
        println!("バッチ {}: 平均二乗誤差 = {:.6} , 処理時間: {:?}", batch_count, mse, elapsed_time_batch);
        model.save(weight_path)?;
        mse_history.push((batch_count, mse));
        draw_mse_graph(&mse_history, mse_graph_path)?;
    }

    println!("学習完了。重み数: {}", model.w.len());
    Ok(())
}