use anyhow::Result;
use csa::PieceType;
use plotters::prelude::*;
use shogi_core::{Move, Piece, PieceKind, Position, Square, Color};
use std::env;
use std::fs;
use std::path::Path;

mod evaluation;
use evaluation::{extract_kpp_features, SparseModel};

fn csa_to_shogi_piece_kind(csa_piece_type: PieceType) -> PieceKind {
    match csa_piece_type {
        PieceType::Pawn => PieceKind::Pawn,
        PieceType::Lance => PieceKind::Lance,
        PieceType::Knight => PieceKind::Knight,
        PieceType::Silver => PieceKind::Silver,
        PieceType::Gold => PieceKind::Gold,
        PieceType::Bishop => PieceKind::Bishop,
        PieceType::Rook => PieceKind::Rook,
        PieceType::King => PieceKind::King,
        PieceType::ProPawn => PieceKind::ProPawn,
        PieceType::ProLance => PieceKind::ProLance,
        PieceType::ProKnight => PieceKind::ProKnight,
        PieceType::ProSilver => PieceKind::ProSilver,
        PieceType::Horse => PieceKind::ProBishop,
                PieceType::Dragon => PieceKind::ProRook,
        PieceType::All => unreachable!(),
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("使用法: {} <csa_file_path>", args[0]);
        return Ok(());
    }

    let csa_file_path = Path::new(&args[1]);
    let weight_path = Path::new("./weights.csv");
    let output_path = "evaluation_graph.png";

    let mut model = SparseModel::new(0.0);
    if weight_path.exists() {
        model.load(weight_path)?;
        println!("重みファイルを読み込みました。");
    } else {
        eprintln!("重みファイルが見つかりません。");
        return Ok(());
    }

    let text = fs::read_to_string(csa_file_path)?;
    let record = csa::parse_csa(&text)?;

    let mut pos = Position::default();
    let mut scores = vec![];

    let features = extract_kpp_features(&pos);
    let score = model.predict(&features);
    scores.push((0, score));

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
                        continue;
                    };
                    let piece_before = if let Some(p) = pos.piece_at(from_sq) {
                        p
                    } else {
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

        if pos.make_move(shogi_move).is_some() {
            let features = extract_kpp_features(&pos);
            let score = model.predict(&features);
            scores.push((index as i32 + 1, score));
        } else {
            eprintln!("不正な手です: {:?}", shogi_move);
            break;
        }
    }

    draw_graph(&scores, output_path)?;
    println!("評価値の推移グラフを {} に保存しました。", output_path);

    Ok(())
}

fn draw_graph(data: &[(i32, f32)], path: &str) -> Result<()> {
    let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let (min_score, max_score) = data
        .iter()
        .fold((f32::MAX, f32::MIN), |(min, max), &(_, score)| {
            (min.min(score), max.max(score))
        });

    let mut chart = ChartBuilder::on(&root)
        .caption("評価値の推移", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..data.len() as i32, min_score..max_score)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().map(|(ply, score)| (*ply, *score)),
        &RED,
    ))?;

    root.present()?;
    Ok(())
}