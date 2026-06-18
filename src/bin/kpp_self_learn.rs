use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use plotters::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;
use shogi_ai::evaluation::Evaluator;
use shogi_core::Move;
use shogi_lib::Position;
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::fs::OpenOptions;
use std::io::BufReader;
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{extract_kpp_features, SparseModel};
use shogi_ai::utils::position_from_sfen_or_usi;

const DEFAULT_SEARCH_DEPTH: u8 = 5; // 教師信号を生成するための探索深さ（CPU負荷低減）
const LEARNING_RATE: f32 = 0.0001; // 学習率
const DEFAULT_POLICY_LEARNING_RATE: f32 = 0.1; // 探索手方策蒸留の学習率
const L2_LAMBDA: f32 = 1e-3; // L2正則化
const DEFAULT_BATCH_SIZE: usize = 256; // バッチサイズ
const DEFAULT_NUM_GAMES: usize = 2560; // 自己対局学習の総対局数
const DEFAULT_REPLAY_MULTIPLIER: usize = 8; // 経験再生バッファをバッチ何個分持つか
const DEFAULT_RESIGN_SCORE_THRESHOLD: f32 = 3000.0; // 決着相当局面では深い探索を省略する
const DEFAULT_SAVE_INTERVAL: usize = DEFAULT_BATCH_SIZE * 2; // 中間重みの保存間隔
const HISTORY_CAPACITY: usize = 128; // 千日手検出用の履歴サイズ
const WIN_RATE_SCALING_FACTOR: f32 = 600.0; // 勝率変換のためのスケーリング係数

type ValueTrainingSample = (Vec<usize>, f32, f32, f32);
type PolicyTrainingSample = (Position, Move);

#[derive(Clone)]
enum TrainingSample {
    Value(ValueTrainingSample),
    Policy(PolicyTrainingSample),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum TrainingMode {
    Value,
    Policy,
    PolicyMargin,
    ResultValue,
}

#[derive(Parser, Debug)]
#[command(about = "Self-play training for KPP + material weights")]
struct Args {
    #[arg(long, default_value = "./policy_weights.binary")]
    weight_path: std::path::PathBuf,
    #[arg(long)]
    output_path: Option<std::path::PathBuf>,
    #[arg(long, default_value_t = DEFAULT_NUM_GAMES)]
    games: usize,
    #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
    batch_size: usize,
    #[arg(long, default_value_t = DEFAULT_SEARCH_DEPTH)]
    depth: u8,
    #[arg(long)]
    teacher_time_limit_ms: Option<u64>,
    #[arg(long, default_value_t = DEFAULT_REPLAY_MULTIPLIER)]
    replay_multiplier: usize,
    #[arg(long)]
    replay_sample_size: Option<usize>,
    #[arg(long, default_value_t = DEFAULT_RESIGN_SCORE_THRESHOLD)]
    resign_score_threshold: f32,
    #[arg(long, default_value_t = 80)]
    self_play_plies: usize,
    #[arg(long, default_value_t = false)]
    skip_drawn_result_samples: bool,
    #[arg(long, default_value_t = DEFAULT_SAVE_INTERVAL)]
    save_interval: usize,
    #[arg(long, default_value = "log.txt")]
    log_path: std::path::PathBuf,
    #[arg(long, value_enum, default_value_t = TrainingMode::Policy)]
    training_mode: TrainingMode,
    #[arg(long, default_value_t = DEFAULT_POLICY_LEARNING_RATE)]
    policy_learning_rate: f32,
    #[arg(long, default_value_t = true)]
    freeze_policy_material: bool,
    #[arg(long, default_value_t = LEARNING_RATE)]
    value_learning_rate: f32,
    #[arg(long, default_value_t = false)]
    freeze_value_material: bool,
}

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
fn update_value_weights(
    model: &mut SparseModel,
    batch: &[ValueTrainingSample],
    learning_rate: f32,
    freeze_material: bool,
) {
    if batch.is_empty() {
        return;
    }

    // スパースな勾配を計算
    let (w_grad_sum, bias_grad_sum, material_grad_sum) = batch
        .par_iter()
        .map(|(features, p, q, material)| {
            // error is the difference between predicted win rate (p) and teacher win rate (q)
            let error = p - q;
            let mut w_grad = HashMap::new();
            for &i in features {
                // The gradient for each active feature is the error
                *w_grad.entry(i).or_insert(0.0) += error;
            }

            let bias_grad = error;
            let material_grad = error * material;

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
    let decay_factor = 1.0 - learning_rate * L2_LAMBDA;
    for w in model.w.iter_mut() {
        *w *= decay_factor;
    }

    // 勾配に基づいて重みを更新
    for (i, grad_sum) in w_grad_sum {
        let avg_grad = grad_sum / batch_float_size;
        model.w[i] -= learning_rate * avg_grad;
    }

    model.bias -= learning_rate * (bias_grad_sum / batch_float_size);
    if !freeze_material {
        model.material_coeff -= learning_rate
            * (material_grad_sum / batch_float_size + L2_LAMBDA * model.material_coeff);
    }
}

fn generate_result_value_samples(
    model: &SparseModel,
    mut position: Position,
    depth: u8,
    max_plies: usize,
    resign_score_threshold: f32,
    skip_drawn_samples: bool,
) -> Vec<TrainingSample> {
    use shogi_ai::evaluation::calculate_material_advantage;
    use shogi_core::Color;

    let mut trajectory: Vec<(Vec<usize>, f32, f32, Color)> = Vec::new();
    let mut winner: Option<Color> = None;

    for _ in 0..max_plies {
        let side_to_move = position.side_to_move();
        let features = extract_kpp_features(&position);
        let material = calculate_material_advantage(&position);
        let predicted_score = model.predict(&position, &features);

        if predicted_score >= resign_score_threshold {
            winner = Some(side_to_move);
            break;
        }
        if predicted_score <= -resign_score_threshold {
            winner = Some(side_to_move.flip());
            break;
        }

        let legal_moves = position.legal_moves();
        if legal_moves.is_empty() {
            winner = Some(side_to_move.flip());
            break;
        }

        trajectory.push((
            features,
            sigmoid(predicted_score, WIN_RATE_SCALING_FACTOR),
            material,
            side_to_move,
        ));

        let evaluator = SharedModelEvaluator { model };
        let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
        let best_move = ai
            .alpha_beta_search(&mut position, depth, -f32::INFINITY, f32::INFINITY)
            .and_then(|(_, pv)| pv.first().copied())
            .unwrap_or(legal_moves[0]);
        position.do_move(best_move);
    }

    if winner.is_none() {
        let final_score = model.predict(&position, &extract_kpp_features(&position));
        if final_score >= resign_score_threshold {
            winner = Some(position.side_to_move());
        } else if final_score <= -resign_score_threshold {
            winner = Some(position.side_to_move().flip());
        }
    }

    if winner.is_none() && skip_drawn_samples {
        return Vec::new();
    }

    trajectory
        .into_iter()
        .map(|(features, p, material, side_to_move)| {
            let q = match winner {
                Some(winner) if winner == side_to_move => 1.0,
                Some(_) => 0.0,
                None => 0.5,
            };
            TrainingSample::Value((features, p, q, material))
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.games == 0 {
        return Err(anyhow!("--games must be greater than zero"));
    }
    if args.batch_size == 0 {
        return Err(anyhow!("--batch-size must be greater than zero"));
    }

    let output_path = args.output_path.as_deref().unwrap_or(&args.weight_path);

    // 1. モデルの読み込み
    let mut model = SparseModel::new(LEARNING_RATE, L2_LAMBDA);
    if matches!(
        args.training_mode,
        TrainingMode::Policy | TrainingMode::PolicyMargin
    ) {
        model.kpp_eta = args.policy_learning_rate;
    }
    if args.weight_path.exists() {
        println!(
            "Loading existing weights from {}...",
            args.weight_path.display()
        );
        model.load(&args.weight_path)?;
        println!("Weights loaded successfully.");
    } else {
        return Err(anyhow!(
            "Weight file '{}' not found. Please run kpp_learn first.",
            args.weight_path.display()
        ));
    }

    // 2. SFENファイルの読み込み
    let sfen_content1 = include_str!("../../converted_records2016_10818.sfen");
    let sfen_content2 = include_str!("../../taya36.sfen");
    let mut sfen_list: Vec<String> = sfen_content1.lines().map(String::from).collect();
    sfen_list.extend(sfen_content2.lines().map(String::from));
    let sfen_list_arc = Arc::new(sfen_list);
    println!("Loaded {} positions.", sfen_list_arc.len());

    let start_time = Instant::now();
    let mut games_done = 0;
    let replay_sample_size = args.replay_sample_size.unwrap_or(args.batch_size);
    let replay_buffer_capacity = args.batch_size * args.replay_multiplier;
    let mut next_save_at = args.save_interval;
    let mut replay_buffer: VecDeque<TrainingSample> =
        VecDeque::with_capacity(replay_buffer_capacity);

    let mut log_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&args.log_path)?;

    use shogi_ai::evaluation::calculate_material_advantage;

    while games_done < args.games {
        let current_batch_size = (args.games - games_done).min(args.batch_size);
        let sfen_list_clone = Arc::clone(&sfen_list_arc);
        let (tx, rx) = mpsc::channel();

        println!(
            "Generating training data (Game {}/{})...",
            games_done, args.games
        );

        // 3. Rayonを使って並列でデータ生成
        // modelへの参照を各スレッドで共有し、クローンを避ける
        (0..current_batch_size).into_par_iter().for_each(|_| {
            let thread_sfen_list_arc = Arc::clone(&sfen_list_clone);
            let thread_tx = tx.clone();
            let mut rng = thread_rng();

            if let Some(sfen_line) = thread_sfen_list_arc.choose(&mut rng) {
                if let Some(mut position) = position_from_sfen_or_usi(sfen_line) {
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

                    let original_features = extract_kpp_features(&position);
                    let material = calculate_material_advantage(&position);
                    let predicted_score = model.predict(&position, &original_features);

                    let needs_search = matches!(
                        args.training_mode,
                        TrainingMode::Policy | TrainingMode::PolicyMargin
                    ) || predicted_score.abs() < args.resign_score_threshold;
                    let search_result = if needs_search {
                        let evaluator = SharedModelEvaluator { model: &model };
                        let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
                        ai.set_emit_info(false);
                        if let Some(time_limit_ms) = args.teacher_time_limit_ms.filter(|_| {
                            matches!(
                                args.training_mode,
                                TrainingMode::Policy | TrainingMode::PolicyMargin
                            )
                        }) {
                            ai.find_best_move(&mut position, args.depth, Some(time_limit_ms))
                                .map(|mv| (0.0, vec![mv]))
                        } else {
                            ai.alpha_beta_search(
                                &mut position,
                                args.depth,
                                -f32::INFINITY,
                                f32::INFINITY,
                            )
                        }
                    } else {
                        None
                    };

                    match args.training_mode {
                        TrainingMode::Value => {
                            let teacher_score =
                                if predicted_score.abs() >= args.resign_score_threshold {
                                    predicted_score.signum() * args.resign_score_threshold
                                } else if let Some((teacher_score, _)) = search_result {
                                    teacher_score
                                } else {
                                    return;
                                };
                            let p = sigmoid(predicted_score, WIN_RATE_SCALING_FACTOR);
                            let q = sigmoid(teacher_score, WIN_RATE_SCALING_FACTOR);
                            let sample = TrainingSample::Value((original_features, p, q, material));
                            if thread_tx.send(sample).is_err() {}
                        }
                        TrainingMode::Policy | TrainingMode::PolicyMargin => {
                            let Some((_, pv)) = search_result else {
                                return;
                            };
                            let Some(&teacher_move) = pv.first() else {
                                return;
                            };
                            let sample = TrainingSample::Policy((position, teacher_move));
                            if thread_tx.send(sample).is_err() {}
                        }
                        TrainingMode::ResultValue => {
                            for sample in generate_result_value_samples(
                                &model,
                                position,
                                args.depth,
                                args.self_play_plies,
                                args.resign_score_threshold,
                                args.skip_drawn_result_samples,
                            ) {
                                if thread_tx.send(sample).is_err() {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        });

        // メインスレッドが持つ送信側(tx)を破棄する。
        drop(tx);

        let training_batch: Vec<TrainingSample> = rx.iter().collect();

        if !training_batch.is_empty() {
            for sample in training_batch.iter().cloned() {
                if replay_buffer.len() == replay_buffer_capacity {
                    replay_buffer.pop_front();
                }
                replay_buffer.push_back(sample);
            }

            let mut update_batch = training_batch.clone();
            let mut rng = thread_rng();
            let replay_sample_count = replay_sample_size.min(replay_buffer.len());
            update_batch.extend(
                replay_buffer
                    .iter()
                    .choose_multiple(&mut rng, replay_sample_count)
                    .into_iter()
                    .cloned(),
            );

            let avg_loss = match args.training_mode {
                TrainingMode::Value | TrainingMode::ResultValue => {
                    let value_batch: Vec<_> = update_batch
                        .iter()
                        .filter_map(|sample| match sample {
                            TrainingSample::Value(value_sample) => Some(value_sample.clone()),
                            TrainingSample::Policy(_) => None,
                        })
                        .collect();
                    if value_batch.is_empty() {
                        0.0
                    } else {
                        let mut cross_entropy_loss_sum = 0.0;
                        for (_, p, q, _) in &value_batch {
                            let epsilon = 1e-7;
                            let p_clipped = p.max(epsilon).min(1.0 - epsilon);
                            cross_entropy_loss_sum -=
                                q * p_clipped.ln() + (1.0 - q) * (1.0 - p_clipped).ln();
                        }
                        let avg_loss = cross_entropy_loss_sum / value_batch.len() as f32;
                        update_value_weights(
                            &mut model,
                            &value_batch,
                            args.value_learning_rate,
                            args.freeze_value_material,
                        );
                        avg_loss
                    }
                }
                TrainingMode::Policy => {
                    let policy_batch: Vec<_> = update_batch
                        .iter()
                        .filter_map(|sample| match sample {
                            TrainingSample::Value(_) => None,
                            TrainingSample::Policy(policy_sample) => Some(policy_sample.clone()),
                        })
                        .collect();
                    let material_coeff_before = model.material_coeff;
                    let (avg_loss, _) = model.update_batch_with_cross_entropy(&policy_batch);
                    if args.freeze_policy_material {
                        model.material_coeff = material_coeff_before;
                    }
                    avg_loss
                }
                TrainingMode::PolicyMargin => {
                    let policy_batch: Vec<_> = update_batch
                        .iter()
                        .filter_map(|sample| match sample {
                            TrainingSample::Value(_) => None,
                            TrainingSample::Policy(policy_sample) => Some(policy_sample.clone()),
                        })
                        .collect();
                    let material_coeff_before = model.material_coeff;
                    let (correct, total) = model.update_batch_for_moves(&policy_batch);
                    if args.freeze_policy_material {
                        model.material_coeff = material_coeff_before;
                    }
                    if total == 0 {
                        0.0
                    } else {
                        1.0 - correct as f32 / total as f32
                    }
                }
            };

            games_done += training_batch.len();

            let elapsed = start_time.elapsed().as_secs_f32();
            let games_per_sec = if elapsed > 0.0 {
                games_done as f32 / elapsed
            } else {
                0.0
            };
            println!(
                "Batch Trained (size={}). Training metric: {:.4}, Games/sec: {:.2}",
                training_batch.len(),
                avg_loss,
                games_per_sec
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
                games_done, avg_loss, model.material_coeff, min_w, max_w
            )?;

            if args.save_interval > 0 && games_done >= next_save_at {
                let checkpoint_path = format!("policy_weights_iter{}.binary", games_done);
                model.save(Path::new(&checkpoint_path))?;
                println!("Checkpoint saved to {}.", checkpoint_path);
                next_save_at += args.save_interval;
            }
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
    model.save(output_path)?;
    println!(
        "Final model saved successfully to {}.",
        output_path.display()
    );

    println!("Self-play learning finished.");

    // 最終プロットを生成
    plot_metrics(&args.log_path, args.games)?;

    Ok(())
}

// ログファイルからデータを読み込み、プロットを生成する関数
fn plot_metrics(log_path: &Path, _current_games_done: usize) -> Result<()> {
    if !log_path.exists() {
        return Err(anyhow!("{} not found for plotting.", log_path.display()));
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
        .caption(
            "Cross-Entropy Loss over Games",
            ("sans-serif", 50).into_font(),
        )
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            0..max_games as i32,
            0f32..loss_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) * 1.1,
        )?;

    chart_loss.configure_mesh().draw()?;

    chart_loss
        .draw_series(LineSeries::new(
            games_data
                .iter()
                .zip(loss_data.iter())
                .map(|(&x, &y)| (x as i32, y)),
            &RED,
        ))?
        .label("Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart_loss
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    root_loss.present()?;

    // 駒得係数のプロット
    let material_file_name = format!("material_coeff_plot.png");
    let root_material = BitMapBackend::new(&material_file_name, (800, 600)).into_drawing_area();
    root_material.fill(&WHITE)?;
    let mut chart_material = ChartBuilder::on(&root_material)
        .caption(
            "Material Coefficient over Games",
            ("sans-serif", 50).into_font(),
        )
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            0..max_games as i32,
            material_coeff_data
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min)
                * 0.9
                ..material_coeff_data
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max)
                    * 1.1,
        )?;

    chart_material.configure_mesh().draw()?;

    chart_material
        .draw_series(LineSeries::new(
            games_data
                .iter()
                .zip(material_coeff_data.iter())
                .map(|(&x, &y)| (x as i32, y)),
            &BLUE,
        ))?
        .label("Material Coeff")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart_material
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    root_material.present()?;

    // KPP重みの最大値・最小値のプロット
    let kpp_file_name = format!("kpp_weights_plot.png");
    let root_kpp = BitMapBackend::new(&kpp_file_name, (800, 600)).into_drawing_area();
    root_kpp.fill(&WHITE)?;
    let mut chart_kpp = ChartBuilder::on(&root_kpp)
        .caption(
            "KPP Weights Min/Max over Games",
            ("sans-serif", 50).into_font(),
        )
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            0..max_games as i32,
            min_w_data.iter().cloned().fold(f32::INFINITY, f32::min) * 0.9
                ..max_w_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) * 1.1,
        )?;

    chart_kpp.configure_mesh().draw()?;

    chart_kpp
        .draw_series(LineSeries::new(
            games_data
                .iter()
                .zip(min_w_data.iter())
                .map(|(&x, &y)| (x as i32, y)),
            &GREEN,
        ))?
        .label("Min Weight")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart_kpp
        .draw_series(LineSeries::new(
            games_data
                .iter()
                .zip(max_w_data.iter())
                .map(|(&x, &y)| (x as i32, y)),
            &BLUE,
        ))?
        .label("Max Weight")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart_kpp
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    root_kpp.present()?;

    Ok(())
}
