use anyhow::Result;
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

use csa;
use plotters::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;
use shogi_core::{Color, Move, Piece, PieceKind, Square};
use shogi_lib::Position;

mod evaluation;
use evaluation::SparseModel;

const KPP_LEARNING_RATE: f32 = 0.01;
const L2_LAMBDA: f32 = 1e-4;
const BATCH_SIZE: usize = 4096;

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

/// Processes a single CSA file to extract (position, teacher_move) pairs.
fn process_csa_file(path: &Path) -> Result<Vec<(Position, Move)>> {
    let text = fs::read_to_string(path)?;
    let record = csa::parse_csa(&text)?;

    let mut training_data = Vec::new();
    let mut shogi_lib_pos = Position::default();

    for mv in record.moves.iter() {
        let shogi_move = match mv.action {
            csa::Action::Move(color, from_csa, to_csa, piece_type_after_csa) => {
                let to_sq = if let Some(sq) = Square::new(to_csa.file, to_csa.rank) {
                    sq
                } else {
                    break;
                };

                if from_csa.file == 0 && from_csa.rank == 0 {
                    let piece_kind = csa_to_shogi_piece_kind(piece_type_after_csa);
                    let piece_color = if color == csa::Color::Black {
                        Color::Black
                    } else {
                        Color::White
                    };
                    Some(Move::Drop {
                        piece: Piece::new(piece_kind, piece_color),
                        to: to_sq,
                    })
                } else {
                    let from_sq = if let Some(sq) = Square::new(from_csa.file, from_csa.rank) {
                        sq
                    } else {
                        break;
                    };
                    let piece_before = if let Some(p) = shogi_lib_pos.piece_at(from_sq) {
                        p
                    } else {
                        break;
                    };
                    let promote =
                        piece_before.piece_kind() != csa_to_shogi_piece_kind(piece_type_after_csa);
                    Some(Move::Normal {
                        from: from_sq,
                        to: to_sq,
                        promote,
                    })
                }
            }
            _ => None,
        };

        if let Some(shogi_move) = shogi_move {
            let legal_moves = shogi_lib_pos.legal_moves();
            if legal_moves.contains(&shogi_move) {
                training_data.push((shogi_lib_pos.clone(), shogi_move));
                shogi_lib_pos.do_move(shogi_move);
            } else {
                break;
            }
        } else {
            break;
        }
    }
    Ok(training_data)
}

fn draw_accuracy_graph(data: &[(usize, f32)], path: &str) -> Result<()> {
    let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    if data.is_empty() {
        return Ok(());
    }

    let mut chart = ChartBuilder::on(&root)
        .caption("Move Prediction Accuracy per Batch", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..data.len(), 0f32..100f32)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().enumerate().map(|(i, &(_, acc))| (i, acc)),
        &BLUE,
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
    let weight_path = Path::new("./policy_weights.binary");
    let accuracy_graph_path = "move_accuracy_graph.png";

    let mut model = SparseModel::new(KPP_LEARNING_RATE, L2_LAMBDA);

    if weight_path.exists() {
        println!("重みファイルを読み込んでいます...");
        model.load(weight_path)?;
        println!("重みファイルを読み込みました。");
    } else {
        println!("新しい重みファイルを作成します。");
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

    let mut batch_count = 0;
    let mut accuracy_history = Vec::new();
    let mut remaining_data: Vec<(Position, Move)> = Vec::new();

    let chunk_size = 1024; // Process files in chunks

    for (chunk_index, file_chunk) in csa_files.chunks(chunk_size).enumerate() {
        let start_time_chunk = Instant::now();

        let chunk_results: Vec<Vec<(Position, Move)>> = file_chunk
            .par_iter()
            .map(|path| process_csa_file(path))
            .filter_map(Result::ok)
            .collect();

        let mut new_data: Vec<_> = chunk_results.into_iter().flatten().collect();

        remaining_data.append(&mut new_data);

        let elapsed_time_chunk = start_time_chunk.elapsed();
        println!("チャンク {}/{} ({} ファイル) の処理完了。時間: {:?}. 現在のデータ数: {}",
                 chunk_index + 1, (csa_files.len() + chunk_size - 1) / chunk_size, file_chunk.len(), elapsed_time_chunk, remaining_data.len());

        while remaining_data.len() >= BATCH_SIZE {
            let batch_data: Vec<_> = remaining_data.drain(0..BATCH_SIZE).collect();
            let start_time_batch = Instant::now();

            let (correct_predictions, total_samples) = model.update_batch_for_moves(&batch_data);
            let accuracy = if total_samples > 0 {
                (correct_predictions as f32 / total_samples as f32) * 100.0
            } else {
                0.0
            };

            batch_count += 1;
            let elapsed_time_batch = start_time_batch.elapsed();
            println!("バッチ {}: 正解率: {:.2}%, 時間: {:?}",
                     batch_count, accuracy, elapsed_time_batch);

            accuracy_history.push((batch_count, accuracy));
        }
    }

    // Process any remaining data
    if !remaining_data.is_empty() {
        let start_time_batch = Instant::now();
        let (correct_predictions, total_samples) = model.update_batch_for_moves(&remaining_data);
        let accuracy = if total_samples > 0 {
            (correct_predictions as f32 / total_samples as f32) * 100.0
        } else {
            0.0
        };
        batch_count += 1;
        let elapsed_time_batch = start_time_batch.elapsed();
        println!("最後のバッチ {}: 正解率: {:.2}%, 時間: {:?}",
                 batch_count, accuracy, elapsed_time_batch);
        accuracy_history.push((batch_count, accuracy));
        draw_accuracy_graph(&accuracy_history, accuracy_graph_path)?;
    }

    let start_time_save = Instant::now();
    model.save(weight_path)?;
    let elapsed_time_save = start_time_save.elapsed();
    println!("最終モデル保存完了。処理時間: {:?}", elapsed_time_save);

    println!("学習完了。");
    Ok(())
}