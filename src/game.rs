use std::fs::File;
use std::io::Write;
use std::path::Path;

use shogi_core::Color;

use crate::ai::ShogiAI;
use crate::evaluation::{Evaluator, SparseModelEvaluator};
use crate::sennichite::SennichiteStatus;
use crate::utils::{draw_evaluation_graph, move_to_kif};

pub fn run() {
    println!("--- ShogiAI 自己対局 ---");

    let evaluator_sente = SparseModelEvaluator::new(Path::new("./weights5times.binary")).expect("Failed to create SparseModelEvaluator for Sente");
    let evaluator_gote = SparseModelEvaluator::new(Path::new("./weights5times.binary")).expect("Failed to create SparseModelEvaluator for Gote");

    // 千日手検出のための履歴バッファの容量を定義
    // 将棋のゲーム履歴は通常数百手なので、256や512程度が妥当です。
    const GAME_HISTORY_CAPACITY: usize = 128;

    let mut ai_sente = ShogiAI::<SparseModelEvaluator, GAME_HISTORY_CAPACITY>::new(evaluator_sente);
    let mut ai_gote = ShogiAI::<SparseModelEvaluator, GAME_HISTORY_CAPACITY>::new(evaluator_gote);

    let mut position = shogi_core::Position::default();
    let search_depth = 3; // 探索の深さ

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
