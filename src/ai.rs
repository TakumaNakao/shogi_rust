use arrayvec::ArrayVec;
use shogi_core::{Bitboard, Color, Move, PieceKind, Position, Square};

use crate::evaluation::{get_piece_value, Evaluator};
use crate::move_ordering::MoveOrdering;
use crate::sennichite::{SennichiteDetector, SennichiteStatus};
use crate::utils::flip_color;

// --- 将棋AIの探索ロジック ---

/// 将棋のアルファベータ探索を管理する構造体
/// ジェネリック型 `E` を持ち、`Evaluator`トレイトを実装する任意の型を評価関数として受け取ります。
/// `HISTORY_CAPACITY`は千日手検出のための履歴バッファのサイズです。
pub struct ShogiAI<E: Evaluator, const HISTORY_CAPACITY: usize> {
    move_ordering: MoveOrdering,
    pub evaluator: E, // 評価関数のインスタンスを保持
    pub sennichite_detector: SennichiteDetector<HISTORY_CAPACITY>, // 千日手検出器
}

impl<E: Evaluator, const HISTORY_CAPACITY: usize> ShogiAI<E, HISTORY_CAPACITY> {
    /// 新��い`ShogiAI`インスタンスを作成します。
    pub fn new(evaluator: E) -> Self {
        ShogiAI {
            move_ordering: MoveOrdering::new(),
            evaluator,
            sennichite_detector: SennichiteDetector::new(),
        }
    }

    // --- ここからSEE実装（新規追加メソッド） ---
    /// SEE: 指定されたマスを攻撃している、指定色の最も価値の低い駒を見つけます。
    fn get_least_valuable_attacker(
        &self,
        position: &Position,
        target: Square,
        side: Color,
        _occupied: shogi_core::Bitboard,
    ) -> Option<(Square, PieceKind)> {
        // legal_moves()の呼び出しを一度に抑え、その結果を使い回すことで高速化を図る
        let yasai_pos = yasai::Position::new(position.inner().clone());
        let legal_moves = yasai_pos.legal_moves();

        let mut least_valuable_attacker: Option<(Square, PieceKind)> = None;
        let mut min_value = i32::MAX;

        // 生成された合法手の中から、targetを攻撃しているものを探す
        for mv in legal_moves {
            if let Move::Normal { from, to, .. } = mv {
                if to == target { // targetへの移動である
                    if let Some(p) = position.piece_at(from) {
                        if p.color() == side { // 攻撃している駒の色が正しい
                            let value = get_piece_value(p.piece_kind());
                            if value < min_value {
                                min_value = value;
                                least_valuable_attacker = Some((from, p.piece_kind()));
                            }
                        }
                    }
                }
            }
        }
        least_valuable_attacker
    }

    /// 静的交換評価（SEE）を計算します。指し手を行う側の視点での損得を返します。
    fn calculate_see(&self, position: &Position, mv: Move) -> i32 {
        if let Move::Normal { from, to, .. } = mv {
            if let (Some(attacker), Some(victim)) = (position.piece_at(from), position.piece_at(to)) {
                let occupied = position.occupied_bitboard() & !Bitboard::single(from);
                return get_piece_value(victim.piece_kind())
                    - self.see_recursive(position, to, flip_color(position.side_to_move()), occupied, attacker.piece_kind());
            }
        }
        0
    }

    /// SEEの再帰ヘルパー関数。
    fn see_recursive(
        &self,
        position: &Position,
        target_sq: Square,
        side: Color,
        occupied: shogi_core::Bitboard,
        victim_kind: PieceKind,
    ) -> i32 {
        if let Some((attacker_sq, attacker_kind)) = self.get_least_valuable_attacker(position, target_sq, side, occupied) {
            let new_occupied = occupied & !Bitboard::single(attacker_sq);
            return get_piece_value(victim_kind)
                - self.see_recursive(position, target_sq, flip_color(side), new_occupied, attacker_kind);
        }
        0
    }
    // --- SEE実装ここまで ---

    /// 静止探索（Quiescence Search）を実行し、局面の安定した評価値を返します。
    fn quiescence_search(
        &mut self,
        position: &shogi_core::Position,
        mut alpha: f32,
        beta: f32,
    ) -> f32 {
        // evaluate()は常に手番視��のスコアを返すようになった
        let stand_pat_score = self.evaluator.evaluate(position);

        if stand_pat_score >= beta {
            return beta;
        }
        alpha = alpha.max(stand_pat_score);

        let yasai_pos: yasai::Position = yasai::Position::new(position.inner().clone());
        let mut forcing_moves: ArrayVec<Move, 593> = ArrayVec::new();
        for mv in yasai_pos.legal_moves() {
            if let Move::Normal { to, .. } = mv {
                if position.piece_at(to).is_some() {
                    forcing_moves.push(mv);
                }
            }
        }

        if forcing_moves.is_empty() {
            return stand_pat_score;
        }

        self.move_ordering.sort_moves(&mut forcing_moves, position);

        let mut best_score = stand_pat_score;
        for mv in forcing_moves {
            if self.calculate_see(position, mv) < 0 {
                continue;
            }

            let mut next_position = position.clone();
            if next_position.make_move(mv).is_none() { continue; }

            self.sennichite_detector.record_position(&next_position);
            let sennichite_status = self.is_sennichite_internal(&next_position);
            let score = match sennichite_status {
                SennichiteStatus::Draw => 0.0,
                SennichiteStatus::PerpetualCheckLoss => -f32::INFINITY,
                SennichiteStatus::None => {
                    -self.quiescence_search(&next_position, -beta, -alpha)
                }
            };
            self.sennichite_detector.unrecord_last_position();

            if score > best_score {
                best_score = score;
            }
            alpha = alpha.max(score);

            if alpha >= beta {
                break;
            }
        }
        best_score
    }


    /// アルファベータ探索を実行し、局面の最善スコアを返します。
    fn alpha_beta_search(
        &mut self,
        position: &shogi_core::Position,
        depth: u8,
        mut alpha: f32,
        beta: f32,
    ) -> f32 {
        if depth == 0 {
            return self.quiescence_search(position, alpha, beta);
        }

        let yasai_pos: yasai::Position = yasai::Position::new(position.inner().clone());
        let mut moves = yasai_pos.legal_moves();

        if moves.is_empty() {
            return -f32::INFINITY; // 詰まされた側は負け
        }

        self.move_ordering.sort_moves(&mut moves, position);

        let mut best_score = -f32::INFINITY;
        for mv in moves {
            let mut next_position = position.clone();
            if next_position.make_move(mv).is_none() {
                continue;
            }

            self.sennichite_detector.record_position(&next_position);
            let sennichite_status = self.is_sennichite_internal(&next_position);

            let score = match sennichite_status {
                SennichiteStatus::Draw => 0.0,
                SennichiteStatus::PerpetualCheckLoss => -f32::INFINITY,
                SennichiteStatus::None => {
                    -self.alpha_beta_search(&next_position, depth - 1, -beta, -alpha)
                }
            };

            self.sennichite_detector.unrecord_last_position();

            if score > best_score {
                best_score = score;
            }
            alpha = alpha.max(score);

            if alpha >= beta {
                self.move_ordering.update_history(&mv, position, depth as i32 * 10);
                break;
            }
        }
        best_score
    }

    /// 特定の局面が千日手であるか（4回出現したか）をチェックします。
    /// 連続王手の例外も考慮します。
    /// この関数は、AIの探索中に仮の局面に対して呼び出されます。
    pub fn is_sennichite_internal(&self, position: &shogi_core::Position) -> SennichiteStatus {
        self.sennichite_detector.check_sennichite(position)
    }

    /// 現在の局面で最適な指し��を見つけます。
    pub fn find_best_move(&mut self, position: &shogi_core::Position, depth: u8) -> Option<Move> {
        let mut best_move: Option<Move> = None;
        let mut best_eval = -f32::INFINITY;
        let mut alpha = -f32::INFINITY;
        let beta = f32::INFINITY;

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

            self.sennichite_detector.record_position(&next_position);
            let sennichite_status = self.is_sennichite_internal(&next_position);
            self.sennichite_detector.unrecord_last_position();

            let eval = match sennichite_status {
                SennichiteStatus::Draw => 0.0,
                SennichiteStatus::PerpetualCheckLoss => -f32::INFINITY,
                SennichiteStatus::None => {
                    -self.alpha_beta_search(&next_position, depth - 1, -beta, -alpha)
                }
            };

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
