use std::path::Path;

use shogi_core::{Color, Move, Piece, PieceKind, Square};

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

fn main() {
    println!("\n--- SparseModelEvaluator を使用した探索 ---");
    let sparse_model_evaluator = SparseModelEvaluator::new(Path::new("./weights.binary"));
    let mut ai_sparse = ShogiAI::new(sparse_model_evaluator);

    let mut position_sparse = shogi_core::Position::default();
    println!("{}", position_sparse.to_sfen_owned());
    let search_depth_sparse = 5;

    println!("最適な指し手を探しています (深さ: {})", search_depth_sparse);
    let best_move_sparse = ai_sparse.find_best_move(&position_sparse, search_depth_sparse);

    match best_move_sparse {
        Some(mv) => {
            println!("見つかった最適な指し手: {:?}", mv);
            if position_sparse.make_move(mv).is_some() {
                println!("指し手を適用しました。新しい手番: {:?}", position_sparse.side_to_move());
                println!("{}", position_sparse.to_sfen_owned());
            }
        }
        None => {
            println!("最適な指し手が見つかりませんでした。");
        }
    }
}