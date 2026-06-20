use anyhow::{anyhow, Result};
use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Deserialize;
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{Evaluator, SparseModel};
use shogi_ai::utils::{format_move_usi, parse_usi_move, position_from_sfen_or_usi};
use shogi_core::{Move, Piece};
use shogi_lib::Position;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const HISTORY_CAPACITY: usize = 256;

#[derive(Parser, Debug)]
#[command(about = "Measure external policy teacher-move regret against the current search")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long, default_value_t = 4)]
    depth: u8,
    #[arg(long)]
    max_positions: Option<usize>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 0)]
    jobs: usize,
    #[arg(long, default_value_t = 300.0)]
    bad_regret_cp: f32,
    #[arg(long, default_value_t = 10)]
    show_worst: usize,
    #[arg(long)]
    export_accepted: Option<PathBuf>,
    #[arg(long, default_value_t = 100.0)]
    max_accepted_regret_cp: f32,
    #[arg(long, default_value_t = 0)]
    min_accepted_legal_moves: usize,
    #[arg(long)]
    max_accepted_abs_score_cp: Option<f32>,
    #[arg(long, default_value_t = false)]
    exclude_accepted_in_check: bool,
}

#[derive(Deserialize)]
struct PolicyRecord {
    sfen: String,
    teacher_move: String,
}

#[derive(Clone)]
struct Sample {
    sfen: String,
    position: Position,
    teacher_move: Move,
    teacher_move_text: String,
    original_line: String,
}

struct ProbeResult {
    sfen: String,
    legal_moves: usize,
    in_check: bool,
    teacher_move: String,
    search_move: Option<String>,
    search_score: f32,
    teacher_score: f32,
    regret: f32,
    original_line: String,
}

struct SharedModelEvaluator<'a> {
    model: &'a SparseModel,
}

impl Evaluator for SharedModelEvaluator<'_> {
    fn evaluate(&self, position: &Position) -> f32 {
        self.model.predict_from_position(position)
    }
}

fn parse_move_for_position(position: &Position, move_text: &str) -> Option<Move> {
    parse_usi_move(move_text).map(|mv| match mv {
        Move::Drop { piece, to } => Move::Drop {
            piece: Piece::new(piece.piece_kind(), position.side_to_move()),
            to,
        },
        normal => normal,
    })
}

fn load_samples(paths: &[PathBuf]) -> Result<Vec<Sample>> {
    let mut samples = Vec::new();
    for path in paths {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        for (line_index, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let record: PolicyRecord = serde_json::from_str(&line).map_err(|e| {
                anyhow!("{}:{} invalid json: {}", path.display(), line_index + 1, e)
            })?;
            let Some(position) = position_from_sfen_or_usi(&record.sfen) else {
                continue;
            };
            let Some(teacher_move) = parse_move_for_position(&position, &record.teacher_move)
            else {
                continue;
            };
            if !position.legal_moves().contains(&teacher_move) {
                continue;
            }
            samples.push(Sample {
                sfen: record.sfen,
                position,
                teacher_move,
                teacher_move_text: record.teacher_move,
                original_line: line,
            });
        }
    }
    if samples.is_empty() {
        return Err(anyhow!("no usable policy samples loaded"));
    }
    Ok(samples)
}

fn load_model(path: &Path) -> Result<SparseModel> {
    let mut model = SparseModel::new(0.0, 0.0);
    model.load(path)?;
    Ok(model)
}

fn sanitize_score(score: f32) -> f32 {
    const LIMIT: f32 = 100_000.0;
    if score == f32::INFINITY {
        LIMIT
    } else if score == -f32::INFINITY {
        -LIMIT
    } else {
        score.clamp(-LIMIT, LIMIT)
    }
}

fn search_position(
    model: &SparseModel,
    mut position: Position,
    depth: u8,
) -> Option<(f32, Vec<Move>)> {
    let evaluator = SharedModelEvaluator { model };
    let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
    ai.set_emit_info(false);
    ai.sennichite_detector.record_position(&position);
    ai.alpha_beta_search(&mut position, depth, -f32::INFINITY, f32::INFINITY)
}

fn teacher_move_score(model: &SparseModel, sample: &Sample, depth: u8) -> Option<f32> {
    let mut child = sample.position.clone();
    child.do_move(sample.teacher_move);
    let child_depth = depth.saturating_sub(1);
    let (score, _) = search_position(model, child, child_depth)?;
    Some(sanitize_score(-score))
}

fn probe_sample(model: &SparseModel, sample: Sample, depth: u8) -> Option<ProbeResult> {
    let legal_moves = sample.position.legal_moves().len();
    let (search_score, pv) = search_position(model, sample.position.clone(), depth)?;
    let search_score = sanitize_score(search_score);
    let teacher_score = teacher_move_score(model, &sample, depth)?;
    let search_move = pv.first().copied().map(format_move_usi);
    let regret = (search_score - teacher_score).max(0.0);
    Some(ProbeResult {
        sfen: sample.sfen,
        legal_moves,
        in_check: sample.position.in_check(),
        teacher_move: sample.teacher_move_text,
        search_move,
        search_score,
        teacher_score,
        regret,
        original_line: sample.original_line,
    })
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn percentile(mut values: Vec<f32>, percentile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((values.len() - 1) as f32 * percentile).round() as usize;
    values[index]
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.depth == 0 {
        return Err(anyhow!("--depth must be greater than zero"));
    }
    if !args.bad_regret_cp.is_finite() || args.bad_regret_cp < 0.0 {
        return Err(anyhow!("--bad-regret-cp must be non-negative"));
    }
    if !args.max_accepted_regret_cp.is_finite() || args.max_accepted_regret_cp < 0.0 {
        return Err(anyhow!("--max-accepted-regret-cp must be non-negative"));
    }
    if args
        .max_accepted_abs_score_cp
        .is_some_and(|limit| !limit.is_finite() || limit < 0.0)
    {
        return Err(anyhow!("--max-accepted-abs-score-cp must be non-negative"));
    }
    if args.jobs > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build_global()
            .map_err(|e| anyhow!("failed to configure rayon thread pool: {e}"))?;
    }

    let model = load_model(&args.weights)?;
    let mut samples = load_samples(&args.input)?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    samples.shuffle(&mut rng);
    if let Some(max_positions) = args.max_positions {
        samples.truncate(max_positions);
    }

    let mut results = samples
        .into_par_iter()
        .filter_map(|sample| probe_sample(&model, sample, args.depth))
        .collect::<Vec<_>>();
    results.sort_by(|a, b| {
        b.regret
            .partial_cmp(&a.regret)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let regrets = results
        .iter()
        .map(|result| result.regret)
        .collect::<Vec<_>>();
    let bad_count = regrets
        .iter()
        .filter(|&&regret| regret > args.bad_regret_cp)
        .count();
    let move_matches = results
        .iter()
        .filter(|result| {
            result
                .search_move
                .as_ref()
                .is_some_and(|search_move| search_move == &result.teacher_move)
        })
        .count();

    println!("samples: {}", results.len());
    println!("mean_regret_cp: {:.2}", mean(&regrets));
    println!("p50_regret_cp: {:.2}", percentile(regrets.clone(), 0.50));
    println!("p90_regret_cp: {:.2}", percentile(regrets.clone(), 0.90));
    println!("p95_regret_cp: {:.2}", percentile(regrets.clone(), 0.95));
    println!(
        "max_regret_cp: {:.2}",
        regrets.first().copied().unwrap_or(0.0)
    );
    println!(
        "bad_regret_count_gt_{:.0}: {} ({:.2}%)",
        args.bad_regret_cp,
        bad_count,
        if results.is_empty() {
            0.0
        } else {
            bad_count as f32 * 100.0 / results.len() as f32
        }
    );
    println!(
        "teacher_move_match: {} ({:.2}%)",
        move_matches,
        if results.is_empty() {
            0.0
        } else {
            move_matches as f32 * 100.0 / results.len() as f32
        }
    );

    for (idx, result) in results.iter().take(args.show_worst).enumerate() {
        println!(
            "worst[{}] regret={:.2} search_score={:.2} teacher_score={:.2} search_move={} teacher_move={} legal_moves={} sfen={}",
            idx + 1,
            result.regret,
            result.search_score,
            result.teacher_score,
            result.search_move.as_deref().unwrap_or("none"),
            result.teacher_move,
            result.legal_moves,
            result.sfen
        );
    }

    if let Some(path) = &args.export_accepted {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut writer = BufWriter::new(File::create(path)?);
        let mut exported = 0usize;
        let mut seen = HashSet::new();
        for result in results.iter().filter(|result| {
            result.regret <= args.max_accepted_regret_cp
                && result.legal_moves >= args.min_accepted_legal_moves
                && (!args.exclude_accepted_in_check || !result.in_check)
                && args.max_accepted_abs_score_cp.map_or(true, |limit| {
                    result.search_score.abs() <= limit && result.teacher_score.abs() <= limit
                })
        }) {
            if !seen.insert(result.sfen.clone()) {
                continue;
            }
            writeln!(writer, "{}", result.original_line)?;
            exported += 1;
        }
        writer.flush()?;
        println!(
            "exported accepted: {} to {} (max_regret_cp <= {:.2})",
            exported,
            path.display(),
            args.max_accepted_regret_cp
        );
    }

    Ok(())
}
