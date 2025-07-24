use anyhow::{anyhow, Result};
use rand::prelude::*;
use rayon::prelude::*;
use shogi_lib::Position;
use shogi_usi_parser::FromUsi; // FromUsiトレイトをインポート
use std::fs::OpenOptions;
use std::io::{BufRead, Write};
use plotters::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;
use std::path::Path;

use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{
    calculate_material_advantage, extract_kpp_features, SparseModel, SparseModelEvaluator,
    MAX_FEATURES,
};

const NUM_GAMES: usize = 100000; // 学習に使用する局面数
const SEARCH_DEPTH: u8 = 4; // 教師信号を生成するための探索深さ
const LEARNING_RATE: f32 = 0.001; // 学習率
const L2_LAMBDA: f32 = 1e-5; // L2正則化
const BATCH_SIZE: usize = 256; // バッチサイズ
const HISTORY_CAPACITY: usize = 128; // 千日手検出用の履歴サイズ

const SAVE_GAME_NUM: usize = 100;

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

// SFEN文字列から局面を生成する関数
fn position_from_sfen(sfen: &str) -> Option<Position> {
    let partial_pos = shogi_core::PartialPosition::from_usi(&format!("sfen {}", sfen)).expect("failed to parse sfen");
    Some(Position::new(partial_pos))
}

fn main() -> Result<()> {
    let weight_path = Path::new("./policy_weights.binary");

    // 1. モデルの読み込み
    let mut model = SparseModel::new(LEARNING_RATE, L2_LAMBDA);
    if weight_path.exists() {
        println!("Loading existing weights from {}...", weight_path.display());
        model.load(weight_path)?;
        println!("Weights loaded successfully.");
    } else {
        return Err(anyhow!("Weight file '{}' not found. Please run kpp_learn first.", weight_path.display()));
    }

    // 2. SFENファイルの読み込み
    let sfen_content = include_str!("../../converted_records2016_10818.sfen");
    let sfen_list: Vec<String> = sfen_content.lines().map(String::from).collect();
    let sfen_list_arc = Arc::new(sfen_list);
    println!("Loaded {} positions.", sfen_list_arc.len());

    let start_time = Instant::now();
    let mut games_done = 0;

    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("log.txt")?;

    // バッチごとにループ
    while games_done < NUM_GAMES {
        let current_batch_size = (NUM_GAMES - games_done).min(BATCH_SIZE);

        let model_arc = Arc::new(model.clone());
        let sfen_list_clone = Arc::clone(&sfen_list_arc); // ループの外でクローン
        let (tx, rx) = mpsc::channel();

        // 3. Rayonを使って並列でデータ生成
        (0..current_batch_size).into_par_iter().for_each(move |_| {
            let thread_model_arc = Arc::clone(&model_arc);
            let thread_sfen_list_arc = Arc::clone(&sfen_list_clone); // クローンされたArcをムーブ
            let thread_tx = tx.clone();
            let mut rng = thread_rng();

            if let Some(sfen_line) = thread_sfen_list_arc.choose(&mut rng) {
                if let Some(mut position) = position_from_sfen(sfen_line) {
                    // ランダムな1〜3手を進める
                    let num_random_moves = rng.gen_range(1..=3);
                    for _ in 0..num_random_moves {
                        let legal_moves = position.legal_moves();
                        if legal_moves.is_empty() {
                            break; // 合法手がなければ終了
                        }
                        let random_move = *legal_moves.choose(&mut rng).unwrap();
                        position.do_move(random_move);
                    }

                    let evaluator = SparseModelEvaluator { model: (*thread_model_arc).clone() };
                    let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);

                    if let Some((score, _)) = ai.alpha_beta_search(&mut position, SEARCH_DEPTH, -f32::INFINITY, f32::INFINITY) {
                        if !score.is_infinite() {
                            if thread_tx.send((position, score)).is_err() {}
                        }
                    }
                }
            }
        });

        let training_batch: Vec<(Position, f32)> = rx.iter().take(current_batch_size).collect();

        // MSEの計算
        let mut mse_sum = 0.0;
        for (pos, teacher_score) in &training_batch {
            let kpp_features = extract_kpp_features(pos);
            let predicted_score = model.predict(pos, &kpp_features);
            let error = predicted_score - teacher_score;
            mse_sum += error * error;
        }
        let mse = mse_sum / training_batch.len() as f32;

        update_weights(&mut model, &training_batch);

        games_done += training_batch.len();

        let elapsed = start_time.elapsed().as_secs_f32();
        let games_per_sec = if elapsed > 0.0 { games_done as f32 / elapsed } else { 0.0 };
        println!(
            "Game: {}/{}, Batch Trained (size={}). Games/sec: {:.2}",
            games_done, NUM_GAMES, training_batch.len(), games_per_sec
        );

        // ログファイルに追記
        let mut min_w = f32::INFINITY;
        let mut max_w = f32::NEG_INFINITY;
        for &val in model.w.iter() {
            if val < min_w {
                min_w = val;
            }
            if val > max_w {
                max_w = val;
            }
        }
        writeln!(
            log_file,
            "{},{:.4},{:.4},{:.4},{:.4}",
            games_done,
            mse,
            model.material_coeff,
            min_w,
            max_w
        )?;

        // プロットを生成
        plot_metrics(games_done)?;

        if (games_done / SAVE_GAME_NUM) > ((games_done.saturating_sub(training_batch.len())) / SAVE_GAME_NUM) {
            println!("Saving model at game {}...", games_done);
            // 駒得係数とKPP重みの最大値・最小値を出力
            println!("  Material Coeff: {:.4}", model.material_coeff);
            let mut min_w = f32::INFINITY;
            let mut max_w = f32::NEG_INFINITY;
            for &val in model.w.iter() {
                if val < min_w {
                    min_w = val;
                }
                if val > max_w {
                    max_w = val;
                }
            }
            println!("  KPP Weights Min: {:.4}, Max: {:.4}", min_w, max_w);
            model.save(weight_path)?;
            println!("Model saved.");
        }
    }

    println!("Saving final model...");
    // 最終保存時にも出力
    println!("  Material Coeff: {:.4}", model.material_coeff);
    let mut min_w = f32::INFINITY;
    let mut max_w = f32::NEG_INFINITY;
    for &val in model.w.iter() {
        if val < min_w {
            min_w = val;
        }
        if val > max_w {
            max_w = val;
        }
    }
    println!("  KPP Weights Min: {:.4}, Max: {:.4}", min_w, max_w);
    model.save(weight_path)?;
    println!("Final model saved successfully to {}.", weight_path.display());

    println!("Self-play learning finished.");

    // 最終プロットを生成
    plot_metrics(NUM_GAMES)?;

    Ok(())
}

// ログファイルからデータを読み込み、プロットを生成する関数
fn plot_metrics(current_games_done: usize) -> Result<()> {
    let log_path = Path::new("log.txt");
    if !log_path.exists() {
        return Err(anyhow!("log.txt not found for plotting."));
    }

    let file = File::open(log_path)?;
    let reader = BufReader::new(file);

    let mut games_data = Vec::new();
    let mut mse_data = Vec::new();
    let mut material_coeff_data = Vec::new();
    let mut min_w_data = Vec::new();
    let mut max_w_data = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 5 {
            games_data.push(parts[0].parse::<usize>()?);
            mse_data.push(parts[1].parse::<f32>()?);
            material_coeff_data.push(parts[2].parse::<f32>()?);
            min_w_data.push(parts[3].parse::<f32>()?);
            max_w_data.push(parts[4].parse::<f32>()?);
        }
    }

    if games_data.is_empty() {
        return Ok(()); // データがなければプロットしない
    }

    let max_games = *games_data.last().unwrap_or(&0);

    // MSEのプロット
    let mse_file_name = format!("mse_plot_{}.png", current_games_done);
    let root_mse = BitMapBackend::new(&mse_file_name, (800, 600)).into_drawing_area();
    root_mse.fill(&WHITE)?;
    let mut chart_mse = ChartBuilder::on(&root_mse)
        .caption("MSE over Games", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..max_games as i32, 0f32..mse_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) * 1.1)?;

    chart_mse.configure_mesh().draw()?;

    chart_mse.draw_series(LineSeries::new(
        games_data.iter().zip(mse_data.iter()).map(|(&x, &y)| (x as i32, y)),
        &RED,
    ))?.label("MSE").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart_mse.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).draw()?;
    root_mse.present()?;

    // 駒得係数のプロット
    let material_file_name = format!("material_coeff_plot_{}.png", current_games_done);
    let root_material = BitMapBackend::new(&material_file_name, (800, 600)).into_drawing_area();
    root_material.fill(&WHITE)?;
    let mut chart_material = ChartBuilder::on(&root_material)
        .caption("Material Coefficient over Games", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..max_games as i32, material_coeff_data.iter().cloned().fold(f32::INFINITY, f32::min) * 0.9..material_coeff_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) * 1.1)?;

    chart_material.configure_mesh().draw()?;

    chart_material.draw_series(LineSeries::new(
        games_data.iter().zip(material_coeff_data.iter()).map(|(&x, &y)| (x as i32, y)),
        &BLUE,
    ))?.label("Material Coeff").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart_material.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).draw()?;
    root_material.present()?;

    // KPP重みの最大値・最小値のプロット
    let kpp_file_name = format!("kpp_weights_plot_{}.png", current_games_done);
    let root_kpp = BitMapBackend::new(&kpp_file_name, (800, 600)).into_drawing_area();
    root_kpp.fill(&WHITE)?;
    let mut chart_kpp = ChartBuilder::on(&root_kpp)
        .caption("KPP Weights Min/Max over Games", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..max_games as i32, min_w_data.iter().cloned().fold(f32::INFINITY, f32::min) * 0.9..max_w_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) * 1.1)?;

    chart_kpp.configure_mesh().draw()?;

    chart_kpp.draw_series(LineSeries::new(
        games_data.iter().zip(min_w_data.iter()).map(|(&x, &y)| (x as i32, y)),
        &GREEN,
    ))?.label("Min Weight").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart_kpp.draw_series(LineSeries::new(
        games_data.iter().zip(max_w_data.iter()).map(|(&x, &y)| (x as i32, y)),
        &BLUE,
    ))?.label(r#"Max Weight"#).legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart_kpp.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).draw()?;
    root_kpp.present()?;

    Ok(())
}