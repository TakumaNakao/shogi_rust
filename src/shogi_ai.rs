use std::fs::File;
use std::io::Write;
use std::path::Path;
use plotters::prelude::*;

use shogi_core::{Color, Move, Piece, PieceKind, Position};
use arrayvec::ArrayVec;

pub mod evaluation;
pub mod search;

use evaluation::{Evaluator, SparseModelEvaluator};
use search::MoveOrdering;


// --- 将棋AIの探索ロジック ---

/// 将棋のアルファベータ探索を管理する構造体
/// ジェネリック型 `E` を持ち、`Evaluator`トレイトを実装する任意の型を評価関数として受け取ります。
pub struct ShogiAI<E: Evaluator> {
    move_ordering: MoveOrdering,
    evaluator: E, // 評価関数のインスタンスを保持
    position_history: Vec<shogi_core::Position>,
}

impl<E: Evaluator> ShogiAI<E> {
    /// 新しい`ShogiAI`インスタンスを作成します。
    /// 評価関数のインスタンスを引数として受け取ります。
    pub fn new(evaluator: E) -> Self {
        ShogiAI {
            move_ordering: MoveOrdering::new(),
            evaluator,
            position_history: Vec::new(),
        }
    }

    /// アルファベータ探索を実行し、局面の最善スコアを返します。
    fn alpha_beta_search(
        &mut self,
        position: &shogi_core::Position,
        depth: u8,
        mut alpha: f32,
        mut beta: f32,
        maximizing_player_color: Color,
    ) -> f32 {
        if depth == 0 {
            return self.evaluator.evaluate(position);
        }
        let yasai_pos: yasai::Position = yasai::Position::new(position.inner().clone());
        let mut moves = yasai_pos.legal_moves();
        self.move_ordering.sort_moves(&mut moves, position);

        if moves.is_empty() {
            return if position.side_to_move() == maximizing_player_color {
                -f32::INFINITY
            } else {
                f32::INFINITY
            };
        }

        if position.side_to_move() == maximizing_player_color {
            let mut max_eval = -f32::INFINITY;
            for mv in moves {
                let mut next_position = position.clone();
                if next_position.make_move(mv).is_none() {
                    continue;
                }

                let eval = self.alpha_beta_search(&next_position, depth - 1, alpha, beta, maximizing_player_color);
                max_eval = max_eval.max(eval);
                alpha = alpha.max(eval);

                if beta <= alpha {
                    self.move_ordering.update_history(&mv, position, depth as i32 * 10);
                    break;
                }
            }
            max_eval
        } else {
            let mut min_eval = f32::INFINITY;
            for mv in moves {
                let mut next_position = position.clone();
                if next_position.make_move(mv).is_none() {
                    continue;
                }

                let eval = self.alpha_beta_search(&next_position, depth - 1, alpha, beta, maximizing_player_color);
                min_eval = min_eval.min(eval);
                beta = beta.min(eval);

                if beta <= alpha {
                    self.move_ordering.update_history(&mv, position, depth as i32 * 10);
                    break;
                }
            }
            min_eval
        }
    }

    pub fn is_sennichite(&self, position: &shogi_core::Position) -> bool {
        println!("{}",self.position_history.iter().filter(|&p| p == position).count());
        self.position_history.iter().filter(|&p| p == position).count() >= 3
    }

    /// 現在の局面で最適な指し手を見つけます。
    pub fn find_best_move(&mut self, position: &shogi_core::Position, depth: u8) -> Option<Move> {
        let mut best_move: Option<Move> = None;
        let mut best_eval = -f32::INFINITY;
        let mut alpha = -f32::INFINITY;
        let beta = f32::INFINITY;

        let current_player = position.side_to_move();
        let yasai_pos: yasai::Position = yasai::Position::new(position.inner().clone());
        let mut moves = yasai_pos.legal_moves();

        self.move_ordering.sort_moves(&mut moves, position);

        if moves.is_empty() {
            return None;
        }

        for mv in moves {
            let mut next_position = position.clone();
            if next_position.make_move(mv).is_none() {
                continue;
            }

            // 千日手チェック
            if self.is_sennichite(&next_position) {
                continue; // この手は千日手になるのでスキップ
            }

            let eval = -self.alpha_beta_search(&next_position, depth - 1, -beta, -alpha, current_player);

            if eval > best_eval {
                best_eval = eval;
                best_move = Some(mv);
            }
            alpha = alpha.max(eval);
        }

        if let Some(bm) = best_move {
            self.move_ordering.update_history(&bm, position, depth as i32 * 20);
        }

        best_move
    }
}

fn move_to_kif(mv: &Move, position: &Position, move_number: usize) -> String {
    let mut kif_str = format!("{} ", move_number);

    match mv {
        Move::Normal { from, to, promote } => {
            let piece = if let Some(p) = position.piece_at(*from) { p } else { println!("err");return "".to_string(); };
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
                _ => "UNKNOWN",
            };
            kif_str.push_str(&format!("{}{}打", to_s, piece_kind_str));
        },
        _ => {
            kif_str.push_str("UNKNOWN_MOVE");
        }
    }
    kif_str
}

fn main() {
    println!("--- ShogiAI 自己対局 ---");

    let evaluator_sente = SparseModelEvaluator::new(Path::new("./weights2017-2018.binary")).expect("Failed to create SparseModelEvaluator for Sente");
    let evaluator_gote = SparseModelEvaluator::new(Path::new("./weights.binary")).expect("Failed to create SparseModelEvaluator for Gote");

    let mut ai_sente = ShogiAI::new(evaluator_sente);
    let mut ai_gote = ShogiAI::new(evaluator_gote);

    let mut position = shogi_core::Position::default();
    let search_depth = 2; // 探索の深さ

    println!("初期局面:{}", position.to_sfen_owned());

    let mut turn = 0;
    let max_turns = 300; // 最大ターン数
    let mut kif_moves: Vec<String> = Vec::new(); // KIF形式の指し手を保存するベクトル
    let mut evaluation_history: Vec<(usize, f32)> = Vec::new(); // 評価値の履歴を保存するベクトル
    let mut sente_evaluation_history: Vec<(usize, f32)> = Vec::new();
    let mut gote_evaluation_history: Vec<(usize, f32)> = Vec::new();

    loop {
        turn += 1;
        if turn > max_turns {
            println!("最大ターン数に達しました。対局終了。");
            break;
        }

        println!("--- ターン {} ---", turn);
        println!("手番: {:?}", position.side_to_move());

        let current_ai = match position.side_to_move() {
            Color::Black => &mut ai_sente,
            Color::White => &mut ai_gote,
        };

        println!("最適な指し手を探しています (深さ: {})", search_depth);
        let best_move = current_ai.find_best_move(&position, search_depth);

        match best_move {
            Some(mv) => {
                kif_moves.push(move_to_kif(&mv, &position, turn));
                println!("見つかった最適な指し手: {:?}", mv);
                    if position.make_move(mv).is_none() {
                        println!("指し手の適用に失敗しました。");
                        break;
                }

                if current_ai.is_sennichite(&position) {
                    println!("千日手により対局終了。");
                    break;
                }
                current_ai.position_history.push(position.clone());

                println!("指し手を適用しました。新しい手番: {:?}", position.side_to_move());
                println!("局面:{}", position.to_sfen_owned());

                // 現在の局面の評価値を記録
                let current_eval = current_ai.evaluator.evaluate(&position);
                evaluation_history.push((turn, current_eval));

                match position.side_to_move() {
                    Color::Black => {
                        sente_evaluation_history.push((turn, current_eval));
                    },
                    Color::White => {
                        gote_evaluation_history.push((turn, current_eval));
                    },
                }
        
                // TODO: 終局判定 (詰み、千日手など) をここに追加
                // 現状は最大ターン数で終了
            }
            None => {
                println!("最適な指し手が見つかりませんでした。対局終了。");
                break;
            }
    }
    }

    // KIF形式で棋譜を出力
    let mut file = File::create("game.kif").expect("ファイル作成失敗");
    file.write_all("手合割：平手\n".as_bytes()).expect("書き込み失敗");
    file.write_all("先手：AI_Sente\n".as_bytes()).expect("書き込み失敗");
    file.write_all("後手：AI_Gote\n".as_bytes()).expect("書き込み失敗");
    file.write_all("開始日時：2025/07/13\n".as_bytes()).expect("書き込み失敗"); // TODO: 日付を動的に
    file.write_all("\n".as_bytes()).expect("書き込み失敗");

    for (i, kif_move) in kif_moves.iter().enumerate() {
        let player_prefix = if (i + 1) % 2 != 0 { "▲" } else { "△" };
        file.write_all(format!("{}{}\n", player_prefix, kif_move).as_bytes()).expect("書き込み失敗");
    }

    println!("\n棋譜を game.kif に出力しました。");

    // 評価値グラフを生成
    if let Err(e) = draw_evaluation_graph(&sente_evaluation_history, &gote_evaluation_history, "evaluation_graph1.png") {
        eprintln!("評価値グラフの生成に失敗しました: {}", e);
    }
}

fn draw_evaluation_graph(sente_data: &[(usize, f32)], gote_data: &[(usize, f32)], path: &str) -> anyhow::Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_core::{Square, Hand, Piece};
    use crate::evaluation::Evaluator;

    // モック評価関数
    struct MockEvaluator;

    impl Evaluator for MockEvaluator {
        fn evaluate(&self, position: &Position) -> f32 {
            // シンプルな評価: 駒の価値の合計
            let mut score = 0.0;
            for i in 1..=9 {
                for j in 1..=9 {
                    if let Some(piece) = position.piece_at(Square::new(i, j).unwrap()) {
                        score += piece_value(piece) * if piece.color() == Color::Black { 1.0 } else { -1.0 };
                    }
                }
            }
            // 持ち駒の評価
            for color in &[Color::Black, Color::White] {
                let hand = position.hand_of_a_player(*color);
                for piece_kind in PieceKind::all() {
                    let count = hand.count(piece_kind).unwrap_or(0);
                    score += piece_kind_value(piece_kind) * count as f32 * if *color == Color::Black { 1.0 } else { -1.0 };
                }
            }
            score
        }
    }

    fn piece_value(piece: Piece) -> f32 {
        piece_kind_value(piece.piece_kind())
    }

    fn piece_kind_value(piece_kind: PieceKind) -> f32 {
        match piece_kind {
            PieceKind::Pawn => 100.0,
            PieceKind::Lance => 300.0,
            PieceKind::Knight => 350.0,
            PieceKind::Silver => 500.0,
            PieceKind::Gold => 550.0,
            PieceKind::Bishop => 800.0,
            PieceKind::Rook => 1000.0,
            PieceKind::King => 10000.0,
            PieceKind::ProPawn | PieceKind::ProLance | PieceKind::ProKnight | PieceKind::ProSilver => 600.0,
            PieceKind::ProBishop => 1000.0,
            PieceKind::ProRook => 1200.0,
        }
    }

    #[test]
    fn test_is_sennichite_detection() {
        let evaluator = MockEvaluator;
        let mut ai = ShogiAI::new(evaluator);
        let mut position = Position::default(); // This will be our repeating position

        for _ in 0..2 {
            position.make_move(Move::Normal { from: Square::new(6, 9).unwrap(), to: Square::new(6, 8).unwrap(), promote: false }).unwrap();
            ai.position_history.push(position.clone());
            position.make_move(Move::Normal { from: Square::new(6, 1).unwrap(), to: Square::new(6, 2).unwrap(), promote: false }).unwrap();
            position.make_move(Move::Normal { from: Square::new(6, 8).unwrap(), to: Square::new(6, 9).unwrap(), promote: false }).unwrap();
            ai.position_history.push(position.clone());
            position.make_move(Move::Normal { from: Square::new(6, 2).unwrap(), to: Square::new(6, 1).unwrap(), promote: false }).unwrap();
        }

        // At this point, the position has appeared 3 times. It should NOT be sennichite.
        position.make_move(Move::Normal { from: Square::new(6, 9).unwrap(), to: Square::new(6, 8).unwrap(), promote: false }).unwrap();
        assert!(!ai.is_sennichite(&position));
        ai.position_history.push(position.clone());

        // Add the position one more time, making it 4 occurrences.
        position.make_move(Move::Normal { from: Square::new(6, 1).unwrap(), to: Square::new(6, 2).unwrap(), promote: false }).unwrap();
        position.make_move(Move::Normal { from: Square::new(6, 8).unwrap(), to: Square::new(6, 9).unwrap(), promote: false }).unwrap();
        ai.position_history.push(position.clone());
        position.make_move(Move::Normal { from: Square::new(6, 2).unwrap(), to: Square::new(6, 1).unwrap(), promote: false }).unwrap();
        position.make_move(Move::Normal { from: Square::new(6, 9).unwrap(), to: Square::new(6, 8).unwrap(), promote: false }).unwrap();

        // Now it should be sennichite.
        assert!(ai.is_sennichite(&position));

        // Test with a different position that is not sennichite
        let mut different_position = Position::default();
        different_position.make_move(Move::Normal { from: Square::new(7, 7).unwrap(), to: Square::new(7, 6).unwrap(), promote: false }).unwrap();
        assert!(!ai.is_sennichite(&different_position));
    }

    #[test]
    fn test_find_best_move_takes_piece() {
        let evaluator = MockEvaluator;
        let mut ai = ShogiAI::new(evaluator);
        let mut position = Position::default();
        // 7六歩、3四歩、2二角成 の局面
        position.make_move(Move::Normal { from: Square::new(7, 7).unwrap(), to: Square::new(7, 6).unwrap(), promote: false }).unwrap();
        position.make_move(Move::Normal { from: Square::new(3, 3).unwrap(), to: Square::new(3, 4).unwrap(), promote: false }).unwrap();
        position.make_move(Move::Normal { from: Square::new(2, 8).unwrap(), to: Square::new(2, 2).unwrap(), promote: true }).unwrap();

        // 次の最善手は同銀のはず
        let best_move = ai.find_best_move(&position, 1).unwrap();
        let expected_move = Move::Normal { from: Square::new(3, 1).unwrap(), to: Square::new(2, 2).unwrap(), promote: false };

        assert_eq!(best_move, expected_move);
    }

    #[test]
    fn test_move_to_kif_normal() {
        let position = Position::default();
        let mv = Move::Normal { from: Square::new(7, 7).unwrap(), to: Square::new(7, 6).unwrap(), promote: false };
        let kif = move_to_kif(&mv, &position, 1);
        assert_eq!(kif, "1 76歩(77)");
    }

    #[test]
    fn test_move_to_kif_promote() {
        let mut position = Position::default();
        position.make_move(Move::Normal { from: Square::new(7, 7).unwrap(), to: Square::new(7, 6).unwrap(), promote: false }).unwrap();
        position.make_move(Move::Normal { from: Square::new(3, 3).unwrap(), to: Square::new(3, 4).unwrap(), promote: false }).unwrap();
        let mv = Move::Normal { from: Square::new(2, 8).unwrap(), to: Square::new(2, 2).unwrap(), promote: true };
        let kif = move_to_kif(&mv, &position, 3);
        assert_eq!(kif, "3 22飛成(28)");
    }

    #[test]
    fn test_move_to_kif_drop() {
        let mut position = Position::default();
        let mut hand = Hand::new();
        hand = hand.added(PieceKind::Pawn).unwrap();
        *position.hand_of_a_player_mut(Color::Black) = hand;
        let mv = Move::Drop { to: Square::new(5, 5).unwrap(), piece: Piece::new(PieceKind::Pawn, Color::Black) };
        let kif = move_to_kif(&mv, &position, 1);
        assert_eq!(kif, "1 55歩打");
    }
}
