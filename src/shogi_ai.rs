use std::fs::File;
use std::io::Write;
use std::path::Path;

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
}

impl<E: Evaluator> ShogiAI<E> {
    /// 新しい`ShogiAI`インスタンスを作成します。
    /// 評価関数のインスタンスを引数として受け取ります。
    pub fn new(evaluator: E) -> Self {
        ShogiAI {
            move_ordering: MoveOrdering::new(),
            evaluator,
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
                _ => "UNKNOWN", // Promoted pieces
            };
            kif_str.push_str(&format!("{}{}{}", to_s, piece_kind_str, from_s));
            if *promote {
                kif_str.push_str("成");
            }
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

    let evaluator_sente = SparseModelEvaluator::new(Path::new("./weights2017-2022.binary")).expect("Failed to create SparseModelEvaluator for Sente");
    let evaluator_gote = SparseModelEvaluator::new(Path::new("./weights.binary")).expect("Failed to create SparseModelEvaluator for Gote");

    let mut ai_sente = ShogiAI::new(evaluator_sente);
    let mut ai_gote = ShogiAI::new(evaluator_gote);

    let mut position = shogi_core::Position::default();
    let search_depth = 2; // 探索の深さ

    println!("初期局面:{}", position.to_sfen_owned());

    let mut turn = 0;
    let max_turns = 100; // 最大ターン数
    let mut kif_moves: Vec<String> = Vec::new(); // KIF形式の指し手を保存するベクトル

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
                println!("指し手を適用しました。新しい手番: {:?}", position.side_to_move());
                println!("局面:{}", position.to_sfen_owned());
        
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
}