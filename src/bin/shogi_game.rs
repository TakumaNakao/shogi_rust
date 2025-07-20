use std::fs::File;
use std::io::Write;
use std::path::Path;
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{Evaluator, SparseModelEvaluator};
use shogi_ai::sennichite::SennichiteStatus;
use shogi_ai::utils::{draw_evaluation_graph, move_to_kif};
use shogi_core::Color;
use yasai::Position;

fn main() {
    println!("--- ShogiAI 自己対局 ---");

    let evaluator_sente = SparseModelEvaluator::new(Path::new("./weights5times.binary")).expect("Failed to create SparseModelEvaluator for Sente");
    let evaluator_gote = SparseModelEvaluator::new(Path::new("./weights5times.binary")).expect("Failed to create SparseModelEvaluator for Gote");

    const GAME_HISTORY_CAPACITY: usize = 128;

    let mut ai_sente = ShogiAI::<SparseModelEvaluator, GAME_HISTORY_CAPACITY>::new(evaluator_sente);
    let mut ai_gote = ShogiAI::<SparseModelEvaluator, GAME_HISTORY_CAPACITY>::new(evaluator_gote);

    let mut position = Position::default();
    
    let search_depth_sente = 4;
    let search_depth_gote = 2;

    println!("初期局面:{}", position.to_sfen_owned());
    println!("先手 探索深さ: {}, 後手 探索深さ: {}", search_depth_sente, search_depth_gote);

    let mut turn = 0;
    let max_turns = 150;
    let mut kif_moves: Vec<String> = Vec::new();
    let mut sente_evaluation_history: Vec<(usize, f32)> = Vec::new();
    let mut gote_evaluation_history: Vec<(usize, f32)> = Vec::new();

    ai_sente.sennichite_detector.record_position(&position);
    ai_gote.sennichite_detector.record_position(&position);

    loop {
        turn += 1;
        if turn > max_turns {
            println!("最大ターン数に達しました。対局終了。");
            break;
        }

        println!("--- ターン {} ---", turn);
        
        let ai_sente_current_eval = ai_sente.evaluator.evaluate(&position);
        let ai_gote_current_eval = ai_gote.evaluator.evaluate(&position);

        let (current_ai, max_depth) = match position.side_to_move() {
            Color::Black => {
                sente_evaluation_history.push((turn, ai_sente_current_eval));
                gote_evaluation_history.push((turn, ai_gote_current_eval));
                (&mut ai_sente, search_depth_sente)
            },
            Color::White => {
                sente_evaluation_history.push((turn, -ai_sente_current_eval));
                gote_evaluation_history.push((turn, -ai_gote_current_eval));
                (&mut ai_gote, search_depth_gote)
            },
        };
        
        println!("手番: {:?}, 探索深さ: {}", position.side_to_move(), max_depth);

        let best_move = current_ai.find_best_move(&mut position, max_depth, None);

        match best_move {
            Some(mv) => {
                kif_moves.push(move_to_kif(&mv, &position, turn));
                println!("見つかった最適な指し手: {:?}", mv);
                position.do_move(mv);
                current_ai.sennichite_detector.record_position(&position);
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
                    SennichiteStatus::None => {}
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

    let mut file = File::create("game.kif").expect("ファイル作成失敗");
    file.write_all("手合割：平手\n".as_bytes()).expect("書き込み失敗");
    file.write_all("先手：AI_Sente\n".as_bytes()).expect("書き込み失敗");
    file.write_all("後手：AI_Gote\n".as_bytes()).expect("書き込み失敗");
    file.write_all("開始日時：2025/07/13\n".as_bytes()).expect("書き込み失敗");
    file.write_all("\n".as_bytes()).expect("書き込み失敗");
    for (i, kif_move) in kif_moves.iter().enumerate() {
        let player_prefix = if (i + 1) % 2 != 0 { "▲" } else { "△" };
        file.write_all(format!("{}{}\n", player_prefix, kif_move).as_bytes()).expect("書き込み失敗");
    }
    println!("\n棋譜を game.kif に出力しました。");
    if let Err(e) = draw_evaluation_graph(&sente_evaluation_history, &gote_evaluation_history, "evaluation_graph1.png") {
        eprintln!("評価値グラフの生成に失敗しました: {}", e);
    }
}
