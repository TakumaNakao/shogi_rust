use anyhow::{anyhow, Result};
use shogi_ai::evaluation::Evaluator;
use rand::prelude::*;
use rayon::prelude::*;
use shogi_lib::Position;
use shogi_usi_parser::FromUsi; // FromUsiトレイトをインポート
use std::collections::HashMap;
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
    extract_kpp_features, SparseModel
};

const SEARCH_DEPTH: u8 = 6; // 教師信号を生成するための探索深さ
const LEARNING_RATE: f32 = 0.001; // 学習率
const L2_LAMBDA: f32 = 1e-4; // L2正則化
const BATCH_SIZE: usize = 512; // バッチサイズ
const HISTORY_CAPACITY: usize = 128; // 千日手検出用の履歴サイズ
const WIN_RATE_SCALING_FACTOR: f32 = 600.0; // 勝率変換のためのスケーリング係数

// modelへの参照を保持する軽量な評価器
struct SharedModelEvaluator<'a> {
    model: &'a SparseModel,
}

impl<'a> Evaluator for SharedModelEvaluator<'a> {
    fn evaluate(&self, position: &Position) -> f32 {
        let kpp_features = extract_kpp_features(position);
        self.model.predict(position, &kpp_features)
    }
}

// Sigmoid function to convert evaluation score to win rate
fn sigmoid(x: f32, k: f32) -> f32 {
    1.0 / (1.0 + (-x / k).exp())
}

// 重みを勝率の差に基づいて更新する
fn update_weights(model: &mut SparseModel, batch: &[(Vec<usize>, f32, f32)]) {
    if batch.is_empty() {
        return;
    }

    // スパースな勾配を計算
    let (w_grad_sum, bias_grad_sum, material_grad_sum) = batch
        .par_iter()
        .map(|(features, p, q)| {
            // error is the difference between predicted win rate (p) and teacher win rate (q)
            let error = p - q;
            let mut w_grad = HashMap::new();
            for &i in features {
                // The gradient for each active feature is the error
                *w_grad.entry(i).or_insert(0.0) += error;
            }

            let bias_grad = error;
            let material_grad = error;

            (w_grad, bias_grad, material_grad)
        })
        .reduce(
            || (HashMap::new(), 0.0, 0.0),
            |mut a, b| {
                for (k, v) in b.0 {
                    *a.0.entry(k).or_insert(0.0) += v;
                }
                a.1 += b.1;
                a.2 += b.2;
                a
            },
        );

    let batch_float_size = batch.len() as f32;

    // L2正則化（Weight Decay）
    let decay_factor = 1.0 - LEARNING_RATE * L2_LAMBDA;
    for w in model.w.iter_mut() {
        *w *= decay_factor;
    }

    // 勾配に基づいて重みを更新
    for (i, grad_sum) in w_grad_sum {
        let avg_grad = grad_sum / batch_float_size;
        model.w[i] -= LEARNING_RATE * avg_grad;
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

    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("log.txt")?;

    let sfen_list_clone = Arc::clone(&sfen_list_arc);
    let (tx, rx) = mpsc::channel();

    println!("Generating training data with {} threads...", BATCH_SIZE);

    // 3. Rayonを使って並列でデータ生成
    // modelへの参照を各スレッドで共有し、クローンを避ける
    (0..BATCH_SIZE)
        .into_par_iter()
        .for_each(|_| {
            let thread_sfen_list_arc = Arc::clone(&sfen_list_clone);
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

                    // modelへの参照を持つ軽量な評価器を使用
                    let evaluator = SharedModelEvaluator { model: &model };
                    let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);

                    // 探索を実行して教師評価値を取得
                    if let Some((teacher_score, _best_move)) = ai.alpha_beta_search(
                        &mut position,
                        SEARCH_DEPTH,
                        -f32::INFINITY,
                        f32::INFINITY,
                    ) {
                        if !teacher_score.is_infinite() {
                            // 1. 現在の局面の特徴量を計算
                            let original_features = extract_kpp_features(&position);

                            // 2. 予測スコアと教師スコアを勝率に変換
                            let predicted_score = model.predict(&position, &original_features);
                            let p = sigmoid(predicted_score, WIN_RATE_SCALING_FACTOR);
                            let q = sigmoid(teacher_score, WIN_RATE_SCALING_FACTOR);

                            // 3. 学習データを送信 (特徴量, 予測勝率, 教師勝率)
                            if thread_tx.send((original_features, p, q)).is_err() {}
                        }
                    }
                }
            }
        });

    // メインスレッドが持つ送信側(tx)を破棄する。
    // これにより、受信側(rx)はこれ以上データが来ないことを知り、ブロックを解除できる。
    drop(tx);

    println!("Collecting generated data...");
    let training_batch: Vec<(Vec<usize>, f32, f32)> = rx.iter().collect();
    println!("Collected {} data points.", training_batch.len());

    // 交差エントロピー損失の計算
    let mut cross_entropy_loss_sum = 0.0;
    if !training_batch.is_empty() {
        for (_, p, q) in &training_batch {
            let epsilon = 1e-7;
            let p_clipped = p.max(epsilon).min(1.0 - epsilon);
            cross_entropy_loss_sum -= q * p_clipped.ln() + (1.0 - q) * (1.0 - p_clipped).ln();
        }
        let avg_loss = cross_entropy_loss_sum / training_batch.len() as f32;

        update_weights(&mut model, &training_batch);

        let games_done = training_batch.len();

        let elapsed = start_time.elapsed().as_secs_f32();
        let games_per_sec = if elapsed > 0.0 { games_done as f32 / elapsed } else { 0.0 };
        println!(
            "Batch Trained (size={}). Cross-Entropy Loss: {:.4}, Games/sec: {:.2}",
            games_done, avg_loss, games_per_sec
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
            avg_loss, // MSEから変更
            model.material_coeff,
            min_w,
            max_w
        )?;

        // プロットを生成
        // plot_metrics(games_done)?;
    }

    println!("Saving model...");
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
    println!("Model saved successfully to {}.", weight_path.display());

    println!("Self-play learning finished.");

    // 最終プロットを生成
    plot_metrics(BATCH_SIZE)?;

    Ok(())
}

// ログファイルからデータを読み込み、プロットを生成する関数
fn plot_metrics(_current_games_done: usize) -> Result<()> {
    let log_path = Path::new("log.txt");
    if !log_path.exists() {
        return Err(anyhow!("log.txt not found for plotting."));
    }

    let file = File::open(log_path)?;
    let reader = BufReader::new(file);

    let mut games_data = Vec::new();
    let mut loss_data = Vec::new();
    let mut material_coeff_data = Vec::new();
    let mut min_w_data = Vec::new();
    let mut max_w_data = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 5 {
            games_data.push(parts[0].parse::<usize>()?);
            loss_data.push(parts[1].parse::<f32>()?);
            material_coeff_data.push(parts[2].parse::<f32>()?);
            min_w_data.push(parts[3].parse::<f32>()?);
            max_w_data.push(parts[4].parse::<f32>()?);
        }
    }

    if games_data.is_empty() {
        return Ok(()); // データがなければプロットしない
    }

    let max_games = *games_data.last().unwrap_or(&0);

    // 交差エントロピー損失のプロット
    let loss_file_name = format!("loss_plot.png");
    let root_loss = BitMapBackend::new(&loss_file_name, (800, 600)).into_drawing_area();
    root_loss.fill(&WHITE)?;
    let mut chart_loss = ChartBuilder::on(&root_loss)
        .caption("Cross-Entropy Loss over Games", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..max_games as i32, 0f32..loss_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) * 1.1)?;

    chart_loss.configure_mesh().draw()?;

    chart_loss.draw_series(LineSeries::new(
        games_data.iter().zip(loss_data.iter()).map(|(&x, &y)| (x as i32, y)),
        &RED,
    ))?.label("Loss").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart_loss.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).draw()?;
    root_loss.present()?;

    // 駒得係数のプロット
    let material_file_name = format!("material_coeff_plot.png");
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
    let kpp_file_name = format!("kpp_weights_plot.png");
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
    ))?.label("Max Weight").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart_kpp.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).draw()?;
    root_kpp.present()?;

    Ok(())
}
