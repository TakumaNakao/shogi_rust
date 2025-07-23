use anyhow::{anyhow, Result};
use rand::prelude::*;
use rayon::prelude::*;
use shogi_lib::Position;
use std::path::Path;
use std::sync::mpsc; // mpsc (multi-producer, single-consumer)チャネルを追加
use std::sync::Arc;
use std::time::Instant;

use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{
    calculate_material_advantage, extract_kpp_features, SparseModel, SparseModelEvaluator,
    MAX_FEATURES,
};

const NUM_GAMES: usize = 10000; // 学習に使用する局面数
const RANDOM_MOVES_AT_START: usize = 10; // 初期局面の多様性を確保するためのランダムな手数
const SEARCH_DEPTH: u8 = 4; // 教師信号を生成するための探索深さ
const LEARNING_RATE: f32 = 0.001; // 学習率
const L2_LAMBDA: f32 = 1e-5; // L2正則化
const BATCH_SIZE: usize = 128; // バッチサイズ
const HISTORY_CAPACITY: usize = 128; // 千日手検出用の履歴サイズ

// 重み更新のロジック（変更なし）
fn update_weights(model: &mut SparseModel, batch: &[(Position, f32)]) {
    if batch.is_empty() {
        return;
    }

    let (w_grad_sum, bias_grad_sum, material_grad_sum) = batch
        .par_iter()
        .map(|(pos, teacher_score)| {
            let kpp_features = extract_kpp_features(pos);
            let predicted_score = model.predict(pos, &kpp_features);
            let error = predicted_score - teacher_score;

            let mut w_grad = vec![0.0; MAX_FEATURES];
            for &i in &kpp_features {
                if i < MAX_FEATURES {
                    w_grad[i] += error;
                }
            }

            let bias_grad = error;
            let material_grad = error * calculate_material_advantage(pos);

            (w_grad, bias_grad, material_grad)
        })
        .reduce(
            || (vec![0.0; MAX_FEATURES], 0.0, 0.0),
            |mut a, b| {
                for i in 0..MAX_FEATURES {
                    a.0[i] += b.0[i];
                }
                a.1 += b.1;
                a.2 += b.2;
                a
            },
        );

    let batch_float_size = batch.len() as f32;

    for i in 0..MAX_FEATURES {
        let avg_grad = w_grad_sum[i] / batch_float_size;
        model.w[i] -= LEARNING_RATE * (avg_grad + L2_LAMBDA * model.w[i]);
    }

    model.bias -= LEARNING_RATE * (bias_grad_sum / batch_float_size);
    model.material_coeff -=
        LEARNING_RATE * (material_grad_sum / batch_float_size + L2_LAMBDA * model.material_coeff);
}

fn generate_random_position() -> Position {
    let mut pos = Position::default();
    let mut rng = thread_rng();
    for _ in 0..RANDOM_MOVES_AT_START {
        let moves = pos.legal_moves();
        if moves.is_empty() {
            break;
        }
        if let Some(mv) = moves.choose(&mut rng) {
            pos.do_move(*mv);
        }
    }
    pos
}

fn main() -> Result<()> {
    let weight_path = Path::new("./policy_weights.binary");

    // 1. Mainスレッドがモデルの唯一の所有者となる
    let mut model = SparseModel::new(LEARNING_RATE, L2_LAMBDA);
    if weight_path.exists() {
        println!("Loading existing weights from {}...", weight_path.display());
        model.load(weight_path)?;
        println!("Weights loaded successfully.");
    } else {
        return Err(anyhow!("Weight file '{}' not found. Please run kpp_learn first.", weight_path.display()));
    }

    let start_time = Instant::now();
    let mut games_done = 0;

    // バッチごとにループ
    while games_done < NUM_GAMES {
        let current_batch_size = (NUM_GAMES - games_done).min(BATCH_SIZE);

        // 2. 現在のモデルをArcで包み、読み取り専用で共有できるようにする
        let model_arc = Arc::new(model.clone());
        let (tx, rx) = mpsc::channel();

        // 3. Rayonを使って並列でデータ生成
        (0..current_batch_size).into_par_iter().for_each(move |_| {
            let thread_model_arc = Arc::clone(&model_arc);
            let thread_tx = tx.clone();

            let mut position = generate_random_position();
            let evaluator = SparseModelEvaluator { model: (*thread_model_arc).clone() };
            let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);

            if let Some((score, _)) = ai.alpha_beta_search(&mut position, SEARCH_DEPTH, -f32::INFINITY, f32::INFINITY) {
                // 無限大の評価値は学習データとして不適切なので除外する
                if !score.is_infinite() {
                    // 計算結果をチャネル経由でMainスレッドに送信
                    if thread_tx.send((position, score)).is_err() {
                        // Mainスレッドが受信をやめた場合は何もしない
                    }
                }
            }
        });

        // 4. Mainスレッドがチャネルから結果を収集
        let training_batch: Vec<(Position, f32)> = rx.iter().take(current_batch_size).collect();

        // 5. Mainスレッドだけがモデルを更新する
        update_weights(&mut model, &training_batch);

        games_done += training_batch.len();

        let elapsed = start_time.elapsed().as_secs_f32();
        let games_per_sec = games_done as f32 / elapsed;
        println!(
            "Game: {}/{}, Batch Trained. Games/sec: {:.2}",
            games_done, NUM_GAMES, games_per_sec
        );

        // 約1000ゲームごとに保存
        if (games_done / 1000) > ((games_done - training_batch.len()) / 1000) {
            println!("Saving model at game {}...", games_done);
            model.save(weight_path)?;
            println!("Model saved.");
        }
    }

    println!("Saving final model...");
    model.save(weight_path)?;
    println!("Final model saved successfully to {}.", weight_path.display());

    println!("Self-play learning finished.");

    Ok(())
}