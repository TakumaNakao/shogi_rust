use std::fs::File;
use std::io::Write;
use std::path::Path;
use plotters::prelude::*;

use shogi_core::{Color, Move, PieceKind, Position, Square, Bitboard};
use arrayvec::ArrayVec;

use circular_buffer::CircularBuffer; // circular-buffer クレートをインポート

pub mod evaluation;
pub mod search;

use evaluation::{Evaluator, SparseModelEvaluator, get_piece_value};
use search::MoveOrdering;

fn flip_color(color: Color) -> Color {
    match color {
        Color::Black => Color::White,
        Color::White => Color::Black,
    }
}

// --- 千日手検出ロジック ---

/// 千日手の判定に使用する、手数を含まない局面情報。
/// shogi_core::PartialPosition には手数(ply)が含まれるため、
/// 同一局面の判定にはこの構造体を使用する。
#[derive(PartialEq, Eq, Clone)]
struct SennichiteKey {
    board: shogi_core::Bitboard,
    hands: [shogi_core::Hand; 2],
    side_to_move: shogi_core::Color,
}

impl From<&shogi_core::PartialPosition> for SennichiteKey {
    fn from(pos: &shogi_core::PartialPosition) -> Self {
        SennichiteKey {
            // PartialPositionから手数(ply)以外のフィールドをクローンする
            board: pos.occupied_bitboard(),
            hands: [pos.hand_of_a_player(Color::Black), pos.hand_of_a_player(Color::White)],
            side_to_move: pos.side_to_move(),
        }
    }
}


/// 千日手の状態を表す列挙型
#[derive(PartialEq, Debug, Eq, Clone, Copy)]
pub enum SennichiteStatus {
    /// 千日手ではない
    None,
    /// 千日手による引き分け（連続王手ではない場合）
    Draw,
    /// 連続王手による負け
    PerpetualCheckLoss,
}

/// 千日手検出器
/// 固定サイズのリングバッファを使用して、過去の局面ハッシュを管理します。
pub struct SennichiteDetector<const CAPACITY: usize> {
    /// 過去の局面ハッシュの履歴（固定サイズのリングバッファ）
    /// circular-bufferは容量に達すると古い要素を自動的に上書きします。
    history: CircularBuffer<CAPACITY, SennichiteKey>,
}

impl<const CAPACITY: usize> SennichiteDetector<CAPACITY> {
    /// 新しい千日手検出器を作成します。
    ///
    /// `CAPACITY`は、履歴に保持する局面の最大数です。
    pub fn new() -> Self {
        SennichiteDetector {
            history: CircularBuffer::new(), // const generics でサイズが指定される
        }
    }

    /// 現在の局面を履歴に記録し、出現回数を更新します。
    ///
    /// `shogi_core::Position`のハッシュ機能を使用します。
    pub fn record_position(&mut self, position: &Position) {
        // PositionからSennichiteKeyを生成して履歴に追加
        let key = SennichiteKey::from(position.inner());
        self.history.push_back(key);
    }

    /// 履歴から最も新しい局面ハッシュを削除し、出現回数を更新します。
    /// 探索中の仮の指し手を元に戻す際に使用します。
    pub fn unrecord_last_position(&mut self) {
        self.history.pop_back(); // 最も新しい要素を削除
    }

    /// 特定の局面の出現回数を取得します。
    pub fn get_position_count(&self, position: &Position) -> u32 {
        // PositionからSennichiteKeyを生成して比較
        let target_key = SennichiteKey::from(position.inner());
        self.history.iter().filter(|key| *key == &target_key).count() as u32
    }

    /// 連続王手チェックのプレースホルダー関数。
    ///
    /// 実際のアプリケーションでは、この関数は、繰り返し発生した局面の
    /// 指し手履歴を分析し、それらがすべて王手であったかを判断する
    /// 複雑なロジックを含む必要があります。
    /// `shogi_core`は合法手判定を提供しないため、このロジックは
    /// 別のクレートや自作のエンジンで実装される必要があります [3, 4]。
    fn is_perpetual_check_placeholder(&self) -> bool {
        // ここに実際の連続王手チェックロジックを実装します。
        // 現時点では常にfalseを返します。
        false
    }
}


// --- 将棋AIの探索ロジック ---

/// 将棋のアルファベータ探索を管理する構造体
/// ジェネリック型 `E` を持ち、`Evaluator`トレイトを実装する任意の型を評価関数として受け取ります。
/// `HISTORY_CAPACITY`は千日手検出のための履歴バッファのサイズです。
pub struct ShogiAI<E: Evaluator, const HISTORY_CAPACITY: usize> {
    move_ordering: MoveOrdering,
    evaluator: E, // 評価関数のインスタンスを保持
    sennichite_detector: SennichiteDetector<HISTORY_CAPACITY>, // 千日手検出器
}

impl<E: Evaluator, const HISTORY_CAPACITY: usize> ShogiAI<E, HISTORY_CAPACITY> {
    /// 新しい`ShogiAI`インスタンスを作成します。
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
        occupied: shogi_core::Bitboard,
    ) -> Option<(Square, PieceKind)> {
        let mut least_valuable_attacker: Option<(Square, PieceKind)> = None;
        let mut min_value = i32::MAX;

        // 盤上の駒を走査して攻撃駒を探します
        for sq_int in 0..81 {
            let Some(sq) = Square::from_u8(sq_int) else { continue; };
            if let Some(piece) = position.piece_at(sq) {
                if piece.color() == side && occupied.contains(sq) {
                    let attacks = shogi_core::Bitboard::empty(); // 暫定的な対応
                    if attacks.contains(target) {
                        let value = get_piece_value(piece.piece_kind());
                        if value < min_value {
                            min_value = value;
                            least_valuable_attacker = Some((sq, piece.piece_kind()));
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
        // evaluate()は常に手番視点のスコアを返すようになった
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
    fn is_sennichite_internal(&self, position: &shogi_core::Position) -> SennichiteStatus {
        let count = self.sennichite_detector.get_position_count(position);
        if count >= 4 { // 将棋の千日手は4回繰り返しで成立 [5]
            // 連続王手チェックは、shogi_coreが合法手判定を提供しないため、プレースホルダーです [3, 4]。
            if self.sennichite_detector.is_perpetual_check_placeholder() {
                SennichiteStatus::PerpetualCheckLoss
            } else {
                SennichiteStatus::Draw
            }
        } else {
            SennichiteStatus::None
        }
    }

    /// 現在の局面で最適な指し手を見つけます。
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
            };
            kif_str.push_str(&format!("{}{}打", to_s, piece_kind_str));
        },
    }
    kif_str
}

fn main() {
    println!("--- ShogiAI 自己対局 ---");

    let evaluator_sente = SparseModelEvaluator::new(Path::new("./weights5times.binary")).expect("Failed to create SparseModelEvaluator for Sente");
    let evaluator_gote = SparseModelEvaluator::new(Path::new("./weights5times.binary")).expect("Failed to create SparseModelEvaluator for Gote");

    // 千日手検出のための履歴バッファの容量を定義
    // 将棋のゲーム履歴は通常数百手なので、256や512程度が妥当です。
    const GAME_HISTORY_CAPACITY: usize = 128;

    let mut ai_sente = ShogiAI::<SparseModelEvaluator, GAME_HISTORY_CAPACITY>::new(evaluator_sente);
    let mut ai_gote = ShogiAI::<SparseModelEvaluator, GAME_HISTORY_CAPACITY>::new(evaluator_gote);

    let mut position = shogi_core::Position::default();
    let search_depth = 2; // 探索の深さ

    println!("初期局面:{}", position.to_sfen_owned());

    let mut turn = 0;
    let max_turns = 150; // 最大ターン数
    let mut kif_moves: Vec<String> = Vec::new(); // KIF形式の指し手を保存するベクトル
    let mut sente_evaluation_history: Vec<(usize, f32)> = Vec::new();
    let mut gote_evaluation_history: Vec<(usize, f32)> = Vec::new();

    // ゲーム開始時の初期局面を履歴に記録
    ai_sente.sennichite_detector.record_position(&position);
    ai_gote.sennichite_detector.record_position(&position);


    loop {
        turn += 1;
        if turn > max_turns {
            println!("最大ターン数に達しました。対局終了。");
            break;
        }

        println!("--- ターン {} ---", turn);
        println!("手番: {:?}", position.side_to_move());

        // 現在の局面の評価値を記録
        let ai_sente_current_eval = ai_sente.evaluator.evaluate(&position);
        let ai_gote_current_eval = ai_gote.evaluator.evaluate(&position);
        match position.side_to_move() {
            Color::Black => {
                sente_evaluation_history.push((turn, ai_sente_current_eval));
                gote_evaluation_history.push((turn, ai_gote_current_eval));
            },
            Color::White => {
                // 先手視点の評価値に統一するため、後手番の評価値は符号を反転させる
                sente_evaluation_history.push((turn, -ai_sente_current_eval));
                gote_evaluation_history.push((turn, -ai_gote_current_eval));
            },
        }

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

                // 実際のゲームの局面を履歴に記録
                // AIは自身の履歴を管理するため、両方のAIの検出器を更新する必要はありません。
                // 現在の手番のAIの検出器のみを更新します。
                current_ai.sennichite_detector.record_position(&position);

                // 実際のゲームの局面で千日手チェック
                let sennichite_status = current_ai.is_sennichite_internal(&position);
                match sennichite_status {
                    SennichiteStatus::Draw => {
                    println!("千日手により対局終了。");
                    break;
                    },
                    SennichiteStatus::PerpetualCheckLoss => {
                        println!("連続王手により対局終了（{:?}の負け）。", position.side_to_move());
                        break;
                    },
                    SennichiteStatus::None => {
                        // ゲーム続行
                    }
                }

                println!("指し手を適用しました。新しい手番: {:?}", position.side_to_move());
                println!("局面:{}", position.to_sfen_owned());
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
        // テスト用に小さな容量のリングバッファを使用
        const TEST_CAPACITY: usize = 10;
        let mut ai = ShogiAI::<MockEvaluator, TEST_CAPACITY>::new(evaluator);

        // 繰り返す局面を作成
        let mut repeating_pos = Position::default();
        for _ in 0..3 {
            repeating_pos.make_move(Move::Normal { from: Square::new(6, 9).unwrap(), to: Square::new(6, 8).unwrap(), promote: false }).unwrap();
            ai.sennichite_detector.record_position(&repeating_pos);
            repeating_pos.make_move(Move::Normal { from: Square::new(6, 1).unwrap(), to: Square::new(6, 2).unwrap(), promote: false }).unwrap();
            repeating_pos.make_move(Move::Normal { from: Square::new(6, 8).unwrap(), to: Square::new(6, 9).unwrap(), promote: false }).unwrap();
            ai.sennichite_detector.record_position(&repeating_pos);
            repeating_pos.make_move(Move::Normal { from: Square::new(6, 2).unwrap(), to: Square::new(6, 1).unwrap(), promote: false }).unwrap();
        }

        repeating_pos.make_move(Move::Normal { from: Square::new(6, 9).unwrap(), to: Square::new(6, 8).unwrap(), promote: false }).unwrap();

        // 3回出現した時点では千日手ではないはず
        assert_eq!(ai.is_sennichite_internal(&repeating_pos), SennichiteStatus::None);

        ai.sennichite_detector.record_position(&repeating_pos);
        repeating_pos.make_move(Move::Normal { from: Square::new(6, 1).unwrap(), to: Square::new(6, 2).unwrap(), promote: false }).unwrap();
    }
}