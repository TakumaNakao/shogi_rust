use shogi_core::{Color, Move, Piece, PieceKind, Position, Square};
use plotters::prelude::*;

pub fn get_piece_value(piece_kind: PieceKind) -> i32 {
    use shogi_core::PieceKind::*;
    match piece_kind {
        Pawn => 100,
        Lance => 300,
        Knight => 300,
        Silver => 500,
        Gold => 600,
        Bishop => 800,
        Rook => 1000,
        King => 20000,
        ProPawn | ProLance | ProKnight => 400,
        ProSilver => 600,
        ProBishop => 1200,
        ProRook => 1500,
    }
}

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

// --- USI Parsing/Formatting Helpers ---
pub fn parse_usi_move(s: &str) -> Option<Move> {
    if s.len() < 4 || s.len() > 5 { return None; }
    if s.chars().nth(1) == Some('*') {
        let piece_char = s.chars().nth(0)?;
        let piece_kind = match piece_char {
            'P' => PieceKind::Pawn, 'L' => PieceKind::Lance, 'N' => PieceKind::Knight,
            'S' => PieceKind::Silver, 'G' => PieceKind::Gold, 'B' => PieceKind::Bishop,
            'R' => PieceKind::Rook,
            _ => return None,
        };
        let to_sq = parse_square(&s[2..4])?;
        return Some(Move::Drop { piece: Piece::new(piece_kind, Color::Black), to: to_sq });
    }
    let from_sq = parse_square(&s[0..2])?;
    let to_sq = parse_square(&s[2..4])?;
    let promote = s.len() == 5 && s.chars().nth(4) == Some('+');
    Some(Move::Normal { from: from_sq, to: to_sq, promote })
}
pub fn parse_square(s: &str) -> Option<Square> {
    let file = s.chars().nth(0)?.to_digit(10)? as u8;
    let rank_char = s.chars().nth(1)?;
    let rank = match rank_char {
        'a' => 1, 'b' => 2, 'c' => 3, 'd' => 4, 'e' => 5, 'f' => 6, 'g' => 7, 'h' => 8, 'i' => 9,
        _ => return None,
    };
    Square::new(file, rank)
}
pub fn format_move_usi(mv: Move) -> String {
    match mv {
        Move::Normal { from, to, promote } => {
            format!("{}{}{}", format_square(from), format_square(to), if promote { "+" } else { "" })
        }
        Move::Drop { piece, to } => {
            let piece_char = match piece.piece_kind() {
                PieceKind::Pawn => 'P', PieceKind::Lance => 'L', PieceKind::Knight => 'N',
                PieceKind::Silver => 'S', PieceKind::Gold => 'G', PieceKind::Bishop => 'B',
                PieceKind::Rook => 'R',
                _ => ' ',
            };
            format!("{}*{}", piece_char, format_square(to))
        }
    }
}
fn format_square(sq: Square) -> String {
    let file = sq.file();
    let rank = match sq.rank() {
        1 => 'a', 2 => 'b', 3 => 'c', 4 => 'd', 5 => 'e', 6 => 'f', 7 => 'g', 8 => 'h', 9 => 'i',
        _ => ' ',
    };
    format!("{}{}", file, rank)
}