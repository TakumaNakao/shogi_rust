use shogi_core::{Color, Move, Piece, PieceKind, Position, Square};
use plotters::prelude::*;
use crate::evaluation::get_piece_value;

pub fn flip_color(color: Color) -> Color {
    match color {
        Color::Black => Color::White,
        Color::White => Color::Black,
    }
}

pub fn move_to_kif(mv: &Move, position: &Position, move_number: usize) -> String {
    let mut kif_str = format!("{} ", move_number);

    match mv {
        Move::Normal { from, to, promote } => {
            let piece = if let Some(p) = position.piece_at(*from) { p } else { return "".to_string(); };
            let piece_kind = piece.piece_kind();

            let from_s = format!("({}{})", from.file(), from.rank());
            let to_s = format!("{}{}", to.file(), to.rank());
            let piece_kind_str = match piece_kind {
                PieceKind::Pawn => "歩",
                PieceKind::Lance => "香",
                PieceKind::Knight => "桂",
                PieceKind::Silver => "銀",
                PieceKind::Gold => "金",
                PieceKind::Bishop => "角",
                PieceKind::Rook => "飛",
                PieceKind::King => "玉",
                PieceKind::ProPawn => "と",
                PieceKind::ProLance => "成香",
                PieceKind::ProKnight => "成桂",
                PieceKind::ProSilver => "成銀",
                PieceKind::ProBishop => "馬",
                PieceKind::ProRook => "龍",
            };
            kif_str.push_str(&format!("{}{}{}{}", to_s, piece_kind_str, if *promote { "成" } else { "" }, from_s));
        },
        Move::Drop { to, piece } => {
            let to_s = format!("{}{}", to.file(), to.rank());
            let piece_kind_str = match piece.piece_kind() {
                PieceKind::Pawn => "歩",
                PieceKind::Lance => "香",
                PieceKind::Knight => "桂",
                PieceKind::Silver => "銀",
                PieceKind::Gold => "金",
                PieceKind::Bishop => "角",
                PieceKind::Rook => "飛",
                PieceKind::King => "玉",
                PieceKind::ProPawn => "と",
                PieceKind::ProLance => "成香",
                PieceKind::ProKnight => "成桂",
                PieceKind::ProSilver => "成銀",
                PieceKind::ProBishop => "馬",
                PieceKind::ProRook => "龍",
            };
            kif_str.push_str(&format!("{}{}打", to_s, piece_kind_str));
        },
    }
    kif_str
}

/// 駒の価値を返すヘルパー関数
pub fn piece_value(piece: Piece) -> i32 {
    get_piece_value(piece.piece_kind())
}

pub fn draw_evaluation_graph(sente_data: &[(usize, f32)], gote_data: &[(usize, f32)], path: &str) -> anyhow::Result<()> {
    let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    if sente_data.is_empty() && gote_data.is_empty() {
        return Ok(());
    }

    let max_turn = sente_data.iter().map(|&(turn, _)| turn).max().unwrap_or(0).max(gote_data.iter().map(|&(turn, _)| turn).max().unwrap_or(0));
    let (min_score, max_score) = sente_data
        .iter()
        .chain(gote_data.iter())
        .fold((f32::MAX, f32::MIN), |(min, max), &(_, score)| {
            (min.min(score), max.max(score))
        });

    let mut chart = ChartBuilder::on(&root)
        .caption("評価値の推移", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..max_turn as i32, min_score..max_score)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        sente_data.iter().map(|&(turn, score)| (turn as i32, score)),
        &BLUE,
    ))?.label("先手AI評価値").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.draw_series(LineSeries::new(
        gote_data.iter().map(|&(turn, score)| (turn as i32, score)),
        &RED,
    ))?.label("後手AI評価値").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.configure_series_labels().border_style(&BLACK).draw()?;

    root.present()?;
    Ok(())
}
