use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{Evaluator, SparseModel};
use shogi_ai::utils::{format_move_usi, position_from_sfen_or_usi};
use shogi_core::Move;
use shogi_lib::Position;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const HISTORY_CAPACITY: usize = 256;
const SCORE_LIMIT: f32 = 100_000.0;

#[derive(Parser, Debug)]
#[command(
    about = "Dump root position rank records with teacher candidate scores for MMTO-lite listwise training"
)]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long)]
    train_output: PathBuf,
    #[arg(long)]
    valid_output: PathBuf,
    #[arg(long, default_value_t = 4)]
    depth: u8,
    #[arg(long, default_value_t = 8)]
    teacher_score_top: usize,
    #[arg(long, value_enum, default_value_t = TeacherScoreSource::Searched)]
    teacher_score_source: TeacherScoreSource,
    #[arg(long)]
    max_positions: Option<usize>,
    #[arg(long, default_value_t = 10)]
    valid_percent: u8,
    #[arg(long, default_value_t = 0)]
    jobs: usize,
    #[arg(long, default_value_t = 9601)]
    seed: u64,
    #[arg(long, default_value_t = 0)]
    min_legal_moves: usize,
    #[arg(long, default_value_t = false)]
    exclude_in_check: bool,
    #[arg(long)]
    max_abs_root_score: Option<f32>,
    #[arg(long, default_value_t = 0.0)]
    min_teacher_gap: f32,
    #[arg(long)]
    max_teacher_gap: Option<f32>,
    #[arg(long, default_value_t = false)]
    score_all_legal_for_valid: bool,
    #[arg(long, default_value_t = 1)]
    jsonl_version: u8,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum TeacherScoreSource {
    Searched,
    Static,
}

#[derive(Clone, Serialize)]
struct RankCandidateRecord {
    #[serde(rename = "move")]
    move_usi: String,
    teacher_score: f32,
    rank: usize,
    regret: f32,
    searched: bool,
}

#[derive(Serialize)]
struct KppRankRecord {
    schema: String,
    version: u8,
    sfen: String,
    teacher_depth: u8,
    root_score: f32,
    legal_moves: usize,
    candidates: Vec<RankCandidateRecord>,
    teacher_weights: String,
}

struct ProcessedRecord {
    index: usize,
    line: String,
    is_valid: bool,
}

#[derive(Clone)]
struct SharedModelEvaluator<'a> {
    model: &'a SparseModel,
}

impl Evaluator for SharedModelEvaluator<'_> {
    fn evaluate(&self, position: &Position) -> f32 {
        self.model.predict_from_position(position)
    }
}

#[derive(Debug)]
enum SkipReason {
    EmptyLegalMoves,
    InCheck,
    FewLegalMoves,
    RootScoreOutOfRange,
    TeacherGapOutOfRange,
    SearchFailed,
    NoCandidateScores,
}

fn sanitize_score(score: f32) -> f32 {
    if score == f32::INFINITY {
        SCORE_LIMIT
    } else if score == -f32::INFINITY {
        -SCORE_LIMIT
    } else {
        score.clamp(-SCORE_LIMIT, SCORE_LIMIT)
    }
}

fn load_positions(
    paths: &[PathBuf],
    max_positions: Option<usize>,
    seed: u64,
) -> Result<(Vec<Position>, usize)> {
    let mut positions = Vec::new();
    let mut invalid_sfen = 0usize;
    for path in paths {
        let content = fs::read_to_string(path)
            .map_err(|e| anyhow!("failed to read {}: {}", path.display(), e))?;
        for line in content.lines() {
            match position_from_sfen_or_usi(line) {
                Some(position) => positions.push(position),
                None => invalid_sfen += 1,
            }
        }
    }

    if positions.is_empty() {
        return Err(anyhow!("no valid positions loaded"));
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    positions.shuffle(&mut rng);
    if let Some(max_positions) = max_positions {
        positions.truncate(max_positions);
    }
    Ok((positions, invalid_sfen))
}

fn create_writer(path: &Path) -> Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(BufWriter::new(File::create(path)?))
}

fn child_static_score(model: &SparseModel, root: &Position, mv: Move) -> f32 {
    let mut child = root.clone();
    child.do_move(mv);
    child.switch_turn();
    model.predict_from_position(&child)
}

fn child_searched_score(model: &SparseModel, root: &Position, mv: Move, depth: u8) -> Option<f32> {
    let mut child = root.clone();
    child.do_move(mv);
    let evaluator = SharedModelEvaluator { model };
    let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
    ai.set_emit_info(false);
    ai.sennichite_detector.record_position(&child);
    let child_depth = depth.saturating_sub(1);
    ai.alpha_beta_search(&mut child, child_depth, -f32::INFINITY, f32::INFINITY)
        .map(|(score, _)| sanitize_score(-score))
}

fn static_candidates(
    model: &SparseModel,
    root: &Position,
    legal_moves: &[Move],
    limit: Option<usize>,
) -> Vec<(Move, f32)> {
    let mut scores = legal_moves
        .iter()
        .copied()
        .map(|mv| (mv, sanitize_score(child_static_score(model, root, mv))))
        .collect::<Vec<_>>();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    match limit {
        Some(limit) if limit > 0 => scores.into_iter().take(limit).collect(),
        _ => scores,
    }
}

fn searched_candidates(
    model: &SparseModel,
    root: &Position,
    legal_moves: &[Move],
    limit: Option<usize>,
    depth: u8,
) -> Vec<(Move, f32)> {
    let mut scores = legal_moves
        .iter()
        .filter_map(|&mv| {
            let score = child_searched_score(model, root, mv, depth)?;
            Some((mv, score))
        })
        .collect::<Vec<_>>();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    match limit {
        Some(limit) if limit > 0 => scores.into_iter().take(limit).collect(),
        _ => scores,
    }
}

fn make_record(
    model_path: &Path,
    model: &SparseModel,
    args: &Args,
    index: usize,
    position: Position,
    is_valid: bool,
) -> (Option<ProcessedRecord>, Option<SkipReason>) {
    let legal_moves = position.legal_moves();
    if legal_moves.is_empty() {
        return (None, Some(SkipReason::EmptyLegalMoves));
    }
    if args.min_legal_moves > 0 && legal_moves.len() < args.min_legal_moves {
        return (None, Some(SkipReason::FewLegalMoves));
    }
    if args.exclude_in_check && position.in_check() {
        return (None, Some(SkipReason::InCheck));
    }

    let limit = if is_valid && args.score_all_legal_for_valid {
        None
    } else if args.teacher_score_top > 0 {
        Some(args.teacher_score_top)
    } else {
        None
    };

    let candidate_scores = match args.teacher_score_source {
        TeacherScoreSource::Searched => {
            searched_candidates(model, &position, &legal_moves, limit, args.depth)
        }
        TeacherScoreSource::Static => static_candidates(model, &position, &legal_moves, limit),
    };
    if candidate_scores.is_empty() {
        return (None, Some(SkipReason::NoCandidateScores));
    }

    let root_score = candidate_scores[0].1;
    if args
        .max_abs_root_score
        .is_some_and(|limit| root_score.abs() > limit)
    {
        return (None, Some(SkipReason::RootScoreOutOfRange));
    }

    if candidate_scores.len() >= 2 && args.min_teacher_gap > 0.0 {
        let gap = candidate_scores[0].1 - candidate_scores[1].1;
        if gap < args.min_teacher_gap {
            return (None, Some(SkipReason::TeacherGapOutOfRange));
        }
    }
    if let Some(max_teacher_gap) = args.max_teacher_gap {
        if candidate_scores.len() >= 2 && max_teacher_gap >= 0.0 {
            let gap = candidate_scores[0].1 - candidate_scores[1].1;
            if gap > max_teacher_gap {
                return (None, Some(SkipReason::TeacherGapOutOfRange));
            }
        }
    }

    let best_score = candidate_scores[0].1;
    let candidates = candidate_scores
        .iter()
        .enumerate()
        .map(|(rank, (mv, score))| RankCandidateRecord {
            move_usi: format_move_usi(*mv),
            teacher_score: *score,
            rank,
            regret: (best_score - *score).max(0.0),
            searched: matches!(args.teacher_score_source, TeacherScoreSource::Searched),
        })
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        return (None, Some(SkipReason::NoCandidateScores));
    }

    let record = KppRankRecord {
        schema: "kpp_rank_v1".to_string(),
        version: args.jsonl_version,
        sfen: position.to_sfen_owned(),
        teacher_depth: args.depth,
        root_score: root_score,
        legal_moves: legal_moves.len(),
        candidates,
        teacher_weights: model_path.to_string_lossy().to_string(),
    };
    let line = match serde_json::to_string(&record) {
        Ok(line) => line,
        Err(_) => return (None, Some(SkipReason::SearchFailed)),
    };
    (
        Some(ProcessedRecord {
            index,
            line,
            is_valid,
        }),
        None,
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.depth == 0 {
        return Err(anyhow!("--depth must be greater than zero"));
    }
    if args.teacher_score_top == 0 && !args.score_all_legal_for_valid {
        return Err(anyhow!("--teacher-score-top must be greater than zero"));
    }
    if args.valid_percent > 90 {
        return Err(anyhow!("--valid-percent must be 0..=90"));
    }
    if args
        .max_teacher_gap
        .is_some_and(|gap| !gap.is_finite() || gap < 0.0)
    {
        return Err(anyhow!("--max-teacher-gap must be non-negative"));
    }
    if !args.min_teacher_gap.is_finite() || args.min_teacher_gap < 0.0 {
        return Err(anyhow!("--min-teacher-gap must be non-negative"));
    }
    if args
        .max_abs_root_score
        .is_some_and(|limit| !limit.is_finite() || limit < 0.0)
    {
        return Err(anyhow!("--max-abs-root-score must be non-negative"));
    }
    if args.jsonl_version == 0 {
        return Err(anyhow!("--jsonl-version must be >= 1"));
    }
    if args.jobs > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build_global()
            .map_err(|e| anyhow!("failed to configure rayon thread pool: {e}"))?;
    }

    let mut model = SparseModel::new(0.0, 0.0);
    model
        .load(&args.weights)
        .map_err(|e| anyhow!("failed to load {}: {}", args.weights.display(), e))?;

    let (positions, invalid_positions) =
        load_positions(&args.input, args.max_positions, args.seed)?;
    let total_positions = positions.len();
    let valid_stride = if args.valid_percent == 0 {
        usize::MAX
    } else {
        (100 / args.valid_percent as usize).max(1)
    };

    let mut skip_reasons: HashMap<String, usize> = HashMap::new();
    if invalid_positions > 0 {
        skip_reasons.insert("invalid_sfen".to_string(), invalid_positions);
    }

    let mut dumped = positions
        .into_iter()
        .enumerate()
        .filter_map(|(idx, position)| {
            let is_valid = valid_stride != usize::MAX && idx % valid_stride == 0;
            let (record, reason) =
                make_record(&args.weights, &model, &args, idx, position, is_valid);
            if let Some(reason) = reason {
                let key = match reason {
                    SkipReason::EmptyLegalMoves => "empty_legal_moves",
                    SkipReason::InCheck => "exclude_in_check",
                    SkipReason::FewLegalMoves => "min_legal_moves",
                    SkipReason::RootScoreOutOfRange => "max_abs_root_score",
                    SkipReason::TeacherGapOutOfRange => "teacher_gap_filter",
                    SkipReason::SearchFailed => "search_failed",
                    SkipReason::NoCandidateScores => "no_candidates",
                };
                *skip_reasons.entry(key.to_string()).or_insert(0) += 1;
            }
            record
        })
        .collect::<Vec<_>>();

    dumped.sort_unstable_by_key(|record| record.index);

    let mut train_writer = create_writer(&args.train_output)?;
    let mut valid_writer = create_writer(&args.valid_output)?;
    let mut train_count = 0usize;
    let mut valid_count = 0usize;
    for record in dumped {
        if record.is_valid {
            writeln!(valid_writer, "{}", record.line)?;
            valid_count += 1;
        } else {
            writeln!(train_writer, "{}", record.line)?;
            train_count += 1;
        }
    }
    train_writer.flush()?;
    valid_writer.flush()?;

    println!("total positions: {total_positions}");
    println!("train records: {train_count}");
    println!("valid records: {valid_count}");
    println!(
        "skipped positions: {}",
        total_positions.saturating_sub(train_count + valid_count)
    );
    for (reason, count) in skip_reasons.iter() {
        println!("skipped[{reason}]: {count}");
    }
    Ok(())
}
