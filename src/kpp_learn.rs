use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use glob::glob;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use csa;
use plotters::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use shogi_core::{Color, Move, Piece, PieceKind, Square};
use shogi_lib::Position;

// evaluationモジュールから公開されたロジックとモデルを使用する
use shogi_ai::evaluation::SparseModel;

const DEFAULT_LEARNING_RATE: f32 = 0.1;
const DEFAULT_L2_LAMBDA: f32 = 1e-5;
const DEFAULT_BATCH_SIZE: usize = 1024;
const DEFAULT_LOAD_FILE_BATCH_SIZE: usize = 256;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum LossMode {
    Margin,
    Ce,
}

#[derive(Parser, Debug)]
#[command(about = "Supervised KPP training from CSA game records")]
struct Args {
    #[arg(long, required = true)]
    input_dir: Vec<PathBuf>,
    #[arg(long, default_value = "./policy_weights.binary")]
    weights: PathBuf,
    #[arg(long, default_value = "./policy_weights.binary")]
    output: PathBuf,
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
    batch_size: usize,
    #[arg(long, default_value_t = DEFAULT_LEARNING_RATE)]
    learning_rate: f32,
    #[arg(long, default_value_t = DEFAULT_L2_LAMBDA)]
    l2_lambda: f32,
    #[arg(long, value_enum, default_value_t = LossMode::Margin)]
    loss: LossMode,
    #[arg(long, default_value_t = 600.0)]
    softmax_temperature: f32,
    #[arg(long, default_value_t = 1024)]
    chunk_size: usize,
    #[arg(long, default_value_t = DEFAULT_LOAD_FILE_BATCH_SIZE)]
    load_file_batch_size: usize,
    #[arg(long, default_value_t = 0)]
    valid_percent: u8,
    #[arg(long, default_value_t = 512)]
    valid_max_files: usize,
    #[arg(long, default_value_t = 20260620)]
    seed: u64,
    #[arg(long)]
    checkpoint_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    checkpoint_every_batches: usize,
    #[arg(long)]
    log_path: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    freeze_material: bool,
    #[arg(long, default_value_t = false)]
    decisive_only: bool,
    #[arg(long, default_value_t = false)]
    winner_only: bool,
    #[arg(long)]
    min_player_rate: Option<i32>,
    #[arg(long)]
    exclude_loser_after_ply: Option<usize>,
    #[arg(long, default_value_t = 1.0)]
    loser_sample_rate: f64,
    #[arg(long, default_value = "move_accuracy_graph.png")]
    accuracy_graph: PathBuf,
    #[arg(long, default_value_t = false)]
    no_graph: bool,
}

fn csa_to_shogi_piece_kind(csa_piece_type: csa::PieceType) -> PieceKind {
    match csa_piece_type {
        csa::PieceType::Pawn => PieceKind::Pawn,
        csa::PieceType::Lance => PieceKind::Lance,
        csa::PieceType::Knight => PieceKind::Knight,
        csa::PieceType::Silver => PieceKind::Silver,
        csa::PieceType::Gold => PieceKind::Gold,
        csa::PieceType::Bishop => PieceKind::Bishop,
        csa::PieceType::Rook => PieceKind::Rook,
        csa::PieceType::King => PieceKind::King,
        csa::PieceType::ProPawn => PieceKind::ProPawn,
        csa::PieceType::ProLance => PieceKind::ProLance,
        csa::PieceType::ProKnight => PieceKind::ProKnight,
        csa::PieceType::ProSilver => PieceKind::ProSilver,
        csa::PieceType::Horse => PieceKind::ProBishop,
        csa::PieceType::Dragon => PieceKind::ProRook,
        csa::PieceType::All => unreachable!(),
    }
}

fn csa_to_shogi_color(color: csa::Color) -> Color {
    if color == csa::Color::Black {
        Color::Black
    } else {
        Color::White
    }
}

fn infer_winner(record: &csa::GameRecord) -> Option<Color> {
    let mut last_mover = None;
    for move_record in &record.moves {
        match move_record.action {
            csa::Action::Move(color, ..) => {
                last_mover = Some(csa_to_shogi_color(color));
            }
            csa::Action::Toryo | csa::Action::TimeUp | csa::Action::IllegalMove => {
                return last_mover;
            }
            csa::Action::IllegalAction(color) => {
                return Some(csa_to_shogi_color(color).flip());
            }
            csa::Action::Tsumi | csa::Action::Kachi => {
                return last_mover;
            }
            csa::Action::Chudan
            | csa::Action::Sennichite
            | csa::Action::Jishogi
            | csa::Action::Hikiwake
            | csa::Action::Matta
            | csa::Action::Fuzumi
            | csa::Action::Error => return None,
        }
    }
    None
}

#[derive(Clone, Copy, Debug, Default)]
struct CsaMetadata {
    black_rate: Option<i32>,
    white_rate: Option<i32>,
    winner: Option<Color>,
}

fn parse_rate_line(line: &str, prefix: &str) -> Option<i32> {
    line.strip_prefix(prefix)
        .and_then(|rest| rest.rsplit(':').next())
        .and_then(|rate| rate.parse::<f64>().ok())
        .map(|rate| rate.round() as i32)
}

fn parse_csa_metadata(text: &str, record: &csa::GameRecord) -> CsaMetadata {
    let mut metadata = CsaMetadata::default();
    for line in text.lines() {
        if let Some(rate) = parse_rate_line(line, "'black_rate:") {
            metadata.black_rate = Some(rate);
        } else if let Some(rate) = parse_rate_line(line, "'white_rate:") {
            metadata.white_rate = Some(rate);
        }
    }
    metadata.winner = infer_winner(record);
    metadata
}

fn player_rate(metadata: &CsaMetadata, color: Color) -> Option<i32> {
    match color {
        Color::Black => metadata.black_rate,
        Color::White => metadata.white_rate,
    }
}

#[derive(Clone, Copy, Debug)]
struct SampleFilter {
    decisive_only: bool,
    winner_only: bool,
    min_player_rate: Option<i32>,
    exclude_loser_after_ply: Option<usize>,
    loser_sample_rate: f64,
}

impl SampleFilter {
    fn include(
        &self,
        color: Color,
        ply_index: usize,
        metadata: &CsaMetadata,
        rng: &mut ChaCha8Rng,
    ) -> bool {
        if self.decisive_only && metadata.winner.is_none() {
            return false;
        }
        if let Some(min_rate) = self.min_player_rate {
            if !player_rate(metadata, color).is_some_and(|rate| rate >= min_rate) {
                return false;
            }
        }
        if metadata.winner == Some(color) {
            return true;
        }
        if self.winner_only {
            return false;
        }
        if let Some(after_ply) = self.exclude_loser_after_ply {
            if metadata.winner.is_some() && ply_index >= after_ply {
                return false;
            }
        }
        self.loser_sample_rate >= 1.0 || rng.gen_bool(self.loser_sample_rate)
    }
}

fn process_csa_file(
    path: &Path,
    filter: SampleFilter,
    sample_seed: u64,
) -> Result<Vec<(Position, Move)>> {
    let bytes = fs::read(path)?;
    let text = String::from_utf8_lossy(&bytes);
    let record = csa::parse_csa(&text)?;
    let metadata = parse_csa_metadata(&text, &record);

    let mut training_data = Vec::new();
    let mut shogi_lib_pos = Position::default();
    let mut rng = ChaCha8Rng::seed_from_u64(sample_seed);

    for (ply_index, mv) in record.moves.iter().enumerate() {
        let shogi_move = match mv.action {
            csa::Action::Move(color, from_csa, to_csa, piece_type_after_csa) => {
                let to_sq = if let Some(sq) = Square::new(to_csa.file, to_csa.rank) {
                    sq
                } else {
                    break;
                };

                if from_csa.file == 0 && from_csa.rank == 0 {
                    let piece_kind = csa_to_shogi_piece_kind(piece_type_after_csa);
                    let piece_color = if color == csa::Color::Black {
                        Color::Black
                    } else {
                        Color::White
                    };
                    Some(Move::Drop {
                        piece: Piece::new(piece_kind, piece_color),
                        to: to_sq,
                    })
                } else {
                    let from_sq = if let Some(sq) = Square::new(from_csa.file, from_csa.rank) {
                        sq
                    } else {
                        break;
                    };
                    let piece_before = if let Some(p) = shogi_lib_pos.piece_at(from_sq) {
                        p
                    } else {
                        break;
                    };
                    let promote =
                        piece_before.piece_kind() != csa_to_shogi_piece_kind(piece_type_after_csa);
                    Some(Move::Normal {
                        from: from_sq,
                        to: to_sq,
                        promote,
                    })
                }
            }
            _ => None,
        };

        if let Some(shogi_move) = shogi_move {
            let legal_moves = shogi_lib_pos.legal_moves();
            if legal_moves.contains(&shogi_move) {
                let move_color = match mv.action {
                    csa::Action::Move(color, ..) => csa_to_shogi_color(color),
                    _ => break,
                };
                if filter.include(move_color, ply_index, &metadata, &mut rng) {
                    training_data.push((shogi_lib_pos.clone(), shogi_move));
                }
                shogi_lib_pos.do_move(shogi_move);
            } else {
                break;
            }
        } else {
            break;
        }
    }
    Ok(training_data)
}

fn draw_accuracy_graph(data: &[(usize, f32)], path: &Path) -> Result<()> {
    let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    if data.is_empty() {
        return Ok(());
    }

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Move Prediction Accuracy per Batch",
            ("sans-serif", 50).into_font(),
        )
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..data.len(), 0f32..100f32)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().enumerate().map(|(i, &(_, acc))| (i, acc)),
        &BLUE,
    ))?;

    root.present()?;
    Ok(())
}

fn collect_csa_files(input_dirs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for input_dir in input_dirs {
        if input_dir.is_dir() {
            let pattern = input_dir.join("**/*.csa");
            let pattern = pattern
                .to_str()
                .ok_or_else(|| anyhow!("path is not valid UTF-8: {}", input_dir.display()))?;
            for entry in glob(pattern)? {
                files.push(entry?);
            }
        } else if input_dir
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("csa"))
        {
            files.push(input_dir.clone());
        } else {
            return Err(anyhow!(
                "input path is neither a directory nor a CSA file: {}",
                input_dir.display()
            ));
        }
    }
    files.sort();
    files.dedup();
    if files.is_empty() {
        return Err(anyhow!("no CSA files found"));
    }
    Ok(files)
}

#[derive(Clone, Copy, Debug)]
struct PolicyMetrics {
    correct: usize,
    total: usize,
    ce: f32,
}

impl PolicyMetrics {
    fn accuracy_percent(&self) -> f32 {
        if self.total > 0 {
            self.correct as f32 / self.total as f32 * 100.0
        } else {
            0.0
        }
    }
}

fn evaluate_policy_metrics(
    model: &SparseModel,
    samples: &[(Position, Move)],
    softmax_temperature: f32,
) -> PolicyMetrics {
    if !softmax_temperature.is_finite() || softmax_temperature <= 0.0 {
        return PolicyMetrics {
            correct: 0,
            total: 0,
            ce: 0.0,
        };
    }

    let results: Vec<(bool, f32)> = samples
        .par_iter()
        .filter_map(|(position, teacher_move)| {
            let legal_moves = position.legal_moves();
            if legal_moves.is_empty() || !legal_moves.contains(teacher_move) {
                return None;
            }

            let mut best_move = legal_moves[0];
            let mut best_score = f32::NEG_INFINITY;
            let mut teacher_score = None;
            let mut scores = Vec::with_capacity(legal_moves.len());
            for &mv in legal_moves.iter() {
                let mut child = position.clone();
                child.do_move(mv);
                child.switch_turn();
                let score = model.predict_from_position(&child);
                if score > best_score {
                    best_score = score;
                    best_move = mv;
                }
                if mv == *teacher_move {
                    teacher_score = Some(score);
                }
                scores.push(score);
            }

            let teacher_score = teacher_score?;
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let total_exp = scores
                .iter()
                .map(|score| ((*score - max_score) / softmax_temperature).exp())
                .sum::<f32>();
            if total_exp <= 0.0 || !total_exp.is_finite() {
                return None;
            }
            let teacher_prob =
                ((teacher_score - max_score) / softmax_temperature).exp() / total_exp;
            Some((best_move == *teacher_move, -teacher_prob.max(1e-7).ln()))
        })
        .collect();

    let total = results.len();
    if total == 0 {
        return PolicyMetrics {
            correct: 0,
            total: 0,
            ce: 0.0,
        };
    }
    let correct = results.iter().filter(|(is_correct, _)| *is_correct).count();
    let ce = results.iter().map(|(_, loss)| *loss).sum::<f32>() / total as f32;
    PolicyMetrics { correct, total, ce }
}

fn train_batch(
    model: &mut SparseModel,
    batch: &[(Position, Move)],
    loss: LossMode,
    softmax_temperature: f32,
    freeze_material: bool,
) -> (f32, usize, usize) {
    let material_before = model.material_coeff;
    let (train_loss, correct_predictions, total_samples) = match loss {
        LossMode::Margin => {
            let (correct, total) = model.update_batch_for_moves(batch);
            (0.0, correct, total)
        }
        LossMode::Ce => {
            let (loss, total) =
                model.update_batch_with_cross_entropy_temperature(batch, softmax_temperature);
            let metrics = evaluate_policy_metrics(model, batch, softmax_temperature);
            (loss, metrics.correct, total)
        }
    };
    if freeze_material {
        model.material_coeff = material_before;
    }
    (train_loss, correct_predictions, total_samples)
}

fn load_validation_samples(
    paths: &[PathBuf],
    max_files: usize,
    filter: SampleFilter,
    seed: u64,
) -> Vec<(Position, Move)> {
    paths
        .iter()
        .take(max_files)
        .enumerate()
        .collect::<Vec<_>>()
        .par_iter()
        .filter_map(|(idx, path)| {
            process_csa_file(path, filter, seed.wrapping_add(*idx as u64)).ok()
        })
        .flatten()
        .collect()
}

fn save_checkpoint(
    model: &SparseModel,
    checkpoint_dir: Option<&Path>,
    epoch: usize,
    batch_count: usize,
) -> Result<()> {
    let Some(checkpoint_dir) = checkpoint_dir else {
        return Ok(());
    };
    fs::create_dir_all(checkpoint_dir)?;
    let checkpoint_path = checkpoint_dir.join(format!(
        "policy_weights_epoch{:03}_batch{:07}.binary",
        epoch, batch_count
    ));
    model.save(&checkpoint_path)?;
    println!("Checkpoint saved to {}.", checkpoint_path.display());
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.epochs == 0 {
        return Err(anyhow!("--epochs must be greater than zero"));
    }
    if args.batch_size == 0 {
        return Err(anyhow!("--batch-size must be greater than zero"));
    }
    if args.chunk_size == 0 {
        return Err(anyhow!("--chunk-size must be greater than zero"));
    }
    if args.load_file_batch_size == 0 {
        return Err(anyhow!("--load-file-batch-size must be greater than zero"));
    }
    if args.valid_percent > 90 {
        return Err(anyhow!("--valid-percent must be 0..=90"));
    }
    if !args.learning_rate.is_finite() || args.learning_rate <= 0.0 {
        return Err(anyhow!("--learning-rate must be positive"));
    }
    if !args.l2_lambda.is_finite() || args.l2_lambda < 0.0 {
        return Err(anyhow!("--l2-lambda must be non-negative"));
    }
    if !args.softmax_temperature.is_finite() || args.softmax_temperature <= 0.0 {
        return Err(anyhow!("--softmax-temperature must be positive"));
    }
    if !args.loser_sample_rate.is_finite()
        || args.loser_sample_rate < 0.0
        || args.loser_sample_rate > 1.0
    {
        return Err(anyhow!("--loser-sample-rate must be in 0.0..=1.0"));
    }
    if args.winner_only && args.loser_sample_rate < 1.0 {
        return Err(anyhow!(
            "--winner-only and --loser-sample-rate are redundant; use one loser filtering mode"
        ));
    }
    if args.winner_only && args.exclude_loser_after_ply.is_some() {
        return Err(anyhow!(
            "--winner-only and --exclude-loser-after-ply are redundant; use one loser filtering mode"
        ));
    }

    let sample_filter = SampleFilter {
        decisive_only: args.decisive_only,
        winner_only: args.winner_only,
        min_player_rate: args.min_player_rate,
        exclude_loser_after_ply: args.exclude_loser_after_ply,
        loser_sample_rate: args.loser_sample_rate,
    };

    let mut model = SparseModel::new(args.learning_rate, args.l2_lambda);

    if args.weights.exists() {
        println!("重みファイルを読み込んでいます...");
        model.load(&args.weights)?;
        println!("重みファイルを読み込みました。");
    } else {
        println!("新しい重みファイルを作成します。");
        model.save(&args.weights)?;
    }
    model.kpp_eta = args.learning_rate;
    model.l2_lambda = args.l2_lambda;
    let initial_material_coeff = model.material_coeff;

    let mut csa_files = collect_csa_files(&args.input_dir)?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    csa_files.shuffle(&mut rng);

    println!("{}個のCSAファイルを読み込みます。", csa_files.len());
    println!(
        "学習データ読み込み: chunk_size={}、load_file_batch_size={}。メモリ保護のため、chunk内を小分けに処理します。",
        args.chunk_size, args.load_file_batch_size
    );

    let valid_count = if args.valid_percent == 0 {
        0
    } else {
        (csa_files.len() * args.valid_percent as usize / 100).max(1)
    };
    let (valid_files, train_files) = csa_files.split_at(valid_count);
    let validation_samples = if valid_files.is_empty() {
        Vec::new()
    } else {
        let capped = args.valid_max_files.min(valid_files.len());
        println!(
            "検証用CSAファイル: {} / {}、検証サンプルを読み込みます。",
            capped,
            valid_files.len()
        );
        load_validation_samples(valid_files, capped, sample_filter, args.seed ^ 0x9e37_79b9)
    };
    if !validation_samples.is_empty() {
        let metrics =
            evaluate_policy_metrics(&model, &validation_samples, args.softmax_temperature);
        println!(
            "baseline validation accuracy: {:.2}% ({}/{}) ce={:.6}",
            metrics.accuracy_percent(),
            metrics.correct,
            metrics.total,
            metrics.ce
        );
    }

    let mut batch_count = 0;
    let mut accuracy_history = Vec::new();

    let mut log_file = if let Some(path) = &args.log_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = File::create(path)?;
        writeln!(
            file,
            "epoch,batch,loss_mode,train_loss,train_accuracy,train_correct,train_total,valid_ce,valid_accuracy,valid_correct,valid_total,material_coeff,min_w,max_w"
        )?;
        Some(file)
    } else {
        None
    };

    for epoch in 1..=args.epochs {
        let mut epoch_train_files = train_files.to_vec();
        epoch_train_files.shuffle(&mut rng);
        let mut remaining_data: Vec<(Position, Move)> = Vec::new();

        for (chunk_index, file_chunk) in epoch_train_files.chunks(args.chunk_size).enumerate() {
            let start_time_chunk = Instant::now();
            let mut chunk_samples = 0usize;

            for (load_index, file_group) in file_chunk.chunks(args.load_file_batch_size).enumerate()
            {
                let load_results: Vec<Vec<(Position, Move)>> = file_group
                    .par_iter()
                    .enumerate()
                    .map(|(idx, path)| {
                        process_csa_file(
                            path,
                            sample_filter,
                            args.seed
                                .wrapping_add(epoch as u64)
                                .wrapping_mul(1_000_003)
                                .wrapping_add(chunk_index as u64)
                                .wrapping_mul(1_000_003)
                                .wrapping_add(load_index as u64)
                                .wrapping_mul(1_000_003)
                                .wrapping_add(idx as u64),
                        )
                    })
                    .filter_map(Result::ok)
                    .collect();

                let mut new_data: Vec<_> = load_results.into_iter().flatten().collect();
                chunk_samples += new_data.len();
                new_data.shuffle(&mut rng);
                remaining_data.append(&mut new_data);

                while remaining_data.len() >= args.batch_size {
                    let mut tail = remaining_data.split_off(args.batch_size);
                    std::mem::swap(&mut tail, &mut remaining_data);
                    let batch_data = tail;
                    let start_time_batch = Instant::now();

                    let (train_loss, correct_predictions, total_samples) = train_batch(
                        &mut model,
                        &batch_data,
                        args.loss,
                        args.softmax_temperature,
                        args.freeze_material,
                    );
                    let accuracy = if total_samples > 0 {
                        (correct_predictions as f32 / total_samples as f32) * 100.0
                    } else {
                        0.0
                    };

                    batch_count += 1;
                    let elapsed_time_batch = start_time_batch.elapsed();
                    println!(
                        "epoch {} batch {}: 正解率: {:.2}%, 時間: {:?}, pending_samples={}",
                        epoch,
                        batch_count,
                        accuracy,
                        elapsed_time_batch,
                        remaining_data.len()
                    );

                    accuracy_history.push((batch_count, accuracy));

                    if let Some(file) = log_file.as_mut() {
                        let valid_metrics = if validation_samples.is_empty() {
                            PolicyMetrics {
                                correct: 0,
                                total: 0,
                                ce: 0.0,
                            }
                        } else {
                            evaluate_policy_metrics(
                                &model,
                                &validation_samples,
                                args.softmax_temperature,
                            )
                        };
                        let max_w = model.w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let min_w = model.w.iter().cloned().fold(f32::INFINITY, f32::min);
                        writeln!(
                            file,
                            "{},{},{:?},{:.6},{:.4},{},{},{:.6},{:.4},{},{},{:.6},{:.6},{:.6}",
                            epoch,
                            batch_count,
                            args.loss,
                            train_loss,
                            accuracy,
                            correct_predictions,
                            total_samples,
                            valid_metrics.ce,
                            valid_metrics.accuracy_percent(),
                            valid_metrics.correct,
                            valid_metrics.total,
                            model.material_coeff,
                            min_w,
                            max_w
                        )?;
                        file.flush()?;
                    }

                    if args.checkpoint_every_batches > 0
                        && batch_count % args.checkpoint_every_batches == 0
                    {
                        save_checkpoint(
                            &model,
                            args.checkpoint_dir.as_deref(),
                            epoch,
                            batch_count,
                        )?;
                    }
                }
            }

            let elapsed_time_chunk = start_time_chunk.elapsed();
            println!(
                "epoch {} チャンク {}/{} ({} ファイル, {} サンプル) の処理完了。時間: {:?}. pending_samples={}",
                epoch,
                chunk_index + 1,
                (epoch_train_files.len() + args.chunk_size - 1) / args.chunk_size,
                file_chunk.len(),
                chunk_samples,
                elapsed_time_chunk,
                remaining_data.len()
            );
        }

        if !remaining_data.is_empty() {
            let start_time_batch = Instant::now();
            let (train_loss, correct_predictions, total_samples) = train_batch(
                &mut model,
                &remaining_data,
                args.loss,
                args.softmax_temperature,
                args.freeze_material,
            );
            let accuracy = if total_samples > 0 {
                (correct_predictions as f32 / total_samples as f32) * 100.0
            } else {
                0.0
            };
            batch_count += 1;
            let elapsed_time_batch = start_time_batch.elapsed();
            println!(
                "epoch {} 最後のバッチ {}: 正解率: {:.2}%, 時間: {:?}",
                epoch, batch_count, accuracy, elapsed_time_batch
            );
            accuracy_history.push((batch_count, accuracy));

            if let Some(file) = log_file.as_mut() {
                let valid_metrics = if validation_samples.is_empty() {
                    PolicyMetrics {
                        correct: 0,
                        total: 0,
                        ce: 0.0,
                    }
                } else {
                    evaluate_policy_metrics(&model, &validation_samples, args.softmax_temperature)
                };
                let max_w = model.w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min_w = model.w.iter().cloned().fold(f32::INFINITY, f32::min);
                writeln!(
                    file,
                    "{},{},{:?},{:.6},{:.4},{},{},{:.6},{:.4},{},{},{:.6},{:.6},{:.6}",
                    epoch,
                    batch_count,
                    args.loss,
                    train_loss,
                    accuracy,
                    correct_predictions,
                    total_samples,
                    valid_metrics.ce,
                    valid_metrics.accuracy_percent(),
                    valid_metrics.correct,
                    valid_metrics.total,
                    model.material_coeff,
                    min_w,
                    max_w
                )?;
                file.flush()?;
            }
        }

        if args.freeze_material {
            model.material_coeff = initial_material_coeff;
        }
        if !validation_samples.is_empty() {
            let metrics =
                evaluate_policy_metrics(&model, &validation_samples, args.softmax_temperature);
            println!(
                "epoch {} validation accuracy: {:.2}% ({}/{}) ce={:.6}",
                epoch,
                metrics.accuracy_percent(),
                metrics.correct,
                metrics.total,
                metrics.ce
            );
        }
        if args.checkpoint_every_batches > 0 {
            save_checkpoint(&model, args.checkpoint_dir.as_deref(), epoch, batch_count)?;
        }
    }

    if !args.no_graph {
        draw_accuracy_graph(&accuracy_history, &args.accuracy_graph)?;
    }

    let start_time_save = Instant::now();
    model.save(&args.output)?;
    let elapsed_time_save = start_time_save.elapsed();
    println!("最終モデル保存完了。処理時間: {:?}", elapsed_time_save);

    // --- Display weight statistics ---
    let max_w = model.w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_w = model.w.iter().cloned().fold(f32::INFINITY, f32::min);
    let non_zero_count = model.w.iter().filter(|&&w| w != 0.0).count();
    let total_count = model.w.len();
    let sparsity = (non_zero_count as f32 / total_count as f32) * 100.0;

    println!("\n--- 学習完了後の重み統計 ---");
    println!("駒得係数: {:.6}", model.material_coeff);
    println!("最大重み: {:.6}", max_w);
    println!("最小重み: {:.6}", min_w);
    println!(
        "非ゼロ要素の割合: {:.4}% ({}/{})",
        sparsity, non_zero_count, total_count
    );
    // --- End of statistics ---

    println!("学習完了。");
    Ok(())
}
