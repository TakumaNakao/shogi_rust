use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{Evaluator, SparseModel};
use shogi_ai::utils::{format_move_usi, position_from_sfen_or_usi};
use shogi_core::{Color, Move};
use shogi_lib::Position;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const HISTORY_CAPACITY: usize = 256;
const SCORE_LIMIT: f32 = 100_000.0;

#[derive(Parser, Debug)]
#[command(about = "Dump root candidate tree records for MMTO tree regeneration")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    student_weights: PathBuf,
    #[arg(long)]
    teacher_weights: Option<PathBuf>,
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long)]
    train_output: PathBuf,
    #[arg(long)]
    valid_output: PathBuf,
    #[arg(long, default_value_t = 5)]
    teacher_depth: u8,
    #[arg(long, default_value_t = 4)]
    student_depth: u8,
    #[arg(long, default_value_t = 16)]
    teacher_score_top: usize,
    #[arg(long, default_value_t = 16)]
    candidate_top: usize,
    #[arg(long)]
    max_positions: Option<usize>,
    #[arg(long, default_value_t = 0)]
    skip_positions: usize,
    #[arg(long, default_value_t = 1024)]
    position_chunk_size: usize,
    #[arg(long, default_value_t = 10)]
    valid_percent: u8,
    #[arg(long, default_value_t = 9601)]
    seed: u64,
    #[arg(long, default_value_t = 0)]
    jobs: usize,
    #[arg(long, default_value_t = 0)]
    min_legal_moves: usize,
    #[arg(long, default_value_t = false)]
    exclude_in_check: bool,
    #[arg(long)]
    max_abs_root_score: Option<f32>,
    #[arg(long, default_value_t = false)]
    score_all_legal_for_valid: bool,
    #[arg(long, default_value_t = 1)]
    jsonl_version: u8,
}

#[derive(Serialize)]
struct CandidateRecord {
    #[serde(rename = "move")]
    move_usi: String,
    teacher_score: f32,
    teacher_rank: usize,
    regret: f32,
    student_score: f32,
    student_rank: usize,
    teacher_pv: Vec<String>,
    student_pv: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    teacher_leaf_sfen: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    student_leaf_sfen: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    teacher_leaf_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    student_leaf_score: Option<f32>,
    searched_by_teacher: bool,
    selected_by_student: bool,
}

#[derive(Serialize)]
struct TreeRecord {
    schema: String,
    version: u8,
    root_index: usize,
    sfen: String,
    teacher_depth: u8,
    student_depth: u8,
    legal_moves: usize,
    teacher_weights: String,
    student_weights: String,
    teacher_root_score: f32,
    student_root_score: f32,
    teacher_best_move: String,
    student_best_move: String,
    candidates: Vec<CandidateRecord>,
}

#[derive(Deserialize)]
struct PositionLine {
    sfen: String,
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
    NoCandidateScores,
    RootScoreOutOfRange,
    SearchFailed,
}

#[derive(Clone)]
struct ScoredMove {
    mv: Move,
    score: f32,
    pv: Vec<Move>,
}

struct ProcessedRecord {
    index: usize,
    is_valid: bool,
    line: String,
}

struct PerRootResult {
    record: Option<ProcessedRecord>,
    reason: Option<SkipReason>,
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

fn create_writer(path: &Path) -> Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create parent directory {}", parent.display()))?;
    }
    Ok(BufWriter::new(File::create(path).with_context(|| {
        format!("failed to create {}", path.display())
    })?))
}

fn skip_reason_key(reason: SkipReason) -> &'static str {
    match reason {
        SkipReason::EmptyLegalMoves => "empty_legal_moves",
        SkipReason::InCheck => "exclude_in_check",
        SkipReason::FewLegalMoves => "min_legal_moves",
        SkipReason::NoCandidateScores => "no_candidate_scores",
        SkipReason::RootScoreOutOfRange => "max_abs_root_score",
        SkipReason::SearchFailed => "search_failed",
    }
}

fn parse_position_line(line: &str) -> Option<Position> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    if line.starts_with('{') {
        let record = serde_json::from_str::<PositionLine>(line).ok()?;
        position_from_sfen_or_usi(&record.sfen)
    } else {
        position_from_sfen_or_usi(line)
    }
}

fn child_search_candidate(
    model: &SparseModel,
    root: &Position,
    mv: Move,
    depth: u8,
) -> Option<(f32, Vec<Move>)> {
    let mut child = root.clone();
    child.do_move(mv);
    let evaluator = SharedModelEvaluator { model };
    let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
    ai.set_emit_info(false);
    ai.sennichite_detector.record_position(&child);
    let child_depth = depth.saturating_sub(1);
    ai.alpha_beta_search(&mut child, child_depth, -f32::INFINITY, f32::INFINITY)
        .map(|(score, pv)| (sanitize_score(-score), pv))
}

fn search_children(
    model: &SparseModel,
    root: &Position,
    legal_moves: &[Move],
    limit: Option<usize>,
    depth: u8,
) -> Vec<ScoredMove> {
    let mut scored = legal_moves
        .iter()
        .filter_map(|&mv| {
            child_search_candidate(model, root, mv, depth).map(|(score, pv)| ScoredMove {
                mv,
                score,
                pv,
            })
        })
        .collect::<Vec<_>>();

    scored.sort_by(|lhs, rhs| {
        rhs.score
            .partial_cmp(&lhs.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    match limit {
        Some(limit) if limit > 0 => scored.into_iter().take(limit).collect(),
        _ => scored,
    }
}

fn search_root(model: &SparseModel, root: &Position, depth: u8) -> Option<(f32, Vec<Move>)> {
    let evaluator = SharedModelEvaluator { model };
    let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
    ai.set_emit_info(false);
    let mut position = root.clone();
    ai.sennichite_detector.record_position(&position);
    ai.alpha_beta_search(&mut position, depth, -f32::INFINITY, f32::INFINITY)
        .map(|(score, pv)| (sanitize_score(score), pv))
}

fn evaluate_root_oriented(model: &SparseModel, root_turn: Color, position: &Position) -> f32 {
    let score = model.predict_from_position(position);
    let oriented = if position.side_to_move() == root_turn {
        score
    } else {
        -score
    };
    sanitize_score(oriented)
}

fn child_oriented_static_score(
    model: &SparseModel,
    root: &Position,
    root_move: Move,
    root_turn: Color,
) -> Option<f32> {
    let mut child = root.clone();
    if !child.legal_moves().contains(&root_move) {
        return None;
    }
    child.do_move(root_move);
    Some(evaluate_root_oriented(model, root_turn, &child))
}

fn contains_move(candidates: &[ScoredMove], mv: Move) -> bool {
    candidates.iter().any(|candidate| candidate.mv == mv)
}

fn apply_root_plus_pv(
    root: &Position,
    root_move: Move,
    child_pv: &[Move],
    model: &SparseModel,
    root_turn: Color,
) -> (Option<String>, Option<f32>) {
    let mut position = root.clone();
    if !position.legal_moves().contains(&root_move) {
        return (None, None);
    }
    position.do_move(root_move);
    for &mv in child_pv {
        if !position.legal_moves().contains(&mv) {
            return (None, None);
        }
        position.do_move(mv);
    }
    let score = evaluate_root_oriented(model, root_turn, &position);
    (Some(position.to_sfen_owned()), Some(score))
}

fn make_record(
    index: usize,
    position: Position,
    teacher_model: &SparseModel,
    student_model: &SparseModel,
    args: &Args,
    is_valid: bool,
) -> PerRootResult {
    let legal_moves = position.legal_moves();
    if legal_moves.is_empty() {
        return PerRootResult {
            record: None,
            reason: Some(SkipReason::EmptyLegalMoves),
        };
    }
    if args.min_legal_moves > 0 && legal_moves.len() < args.min_legal_moves {
        return PerRootResult {
            record: None,
            reason: Some(SkipReason::FewLegalMoves),
        };
    }
    if args.exclude_in_check && position.in_check() {
        return PerRootResult {
            record: None,
            reason: Some(SkipReason::InCheck),
        };
    }

    let root_turn = position.side_to_move();

    let teacher_limit = if is_valid && args.score_all_legal_for_valid {
        None
    } else if args.teacher_score_top == 0 {
        None
    } else {
        Some(args.teacher_score_top)
    };
    let student_limit = if is_valid && args.score_all_legal_for_valid {
        None
    } else if args.candidate_top == 0 {
        None
    } else {
        Some(args.candidate_top)
    };
    let mut teacher_candidates = search_children(
        teacher_model,
        &position,
        &legal_moves,
        teacher_limit,
        args.teacher_depth,
    );
    if teacher_candidates.is_empty() {
        return PerRootResult {
            record: None,
            reason: Some(SkipReason::NoCandidateScores),
        };
    }
    let student_top_candidates = search_children(
        student_model,
        &position,
        &legal_moves,
        student_limit,
        args.student_depth,
    );

    let (teacher_root_score, teacher_root_pv) =
        match search_root(teacher_model, &position, args.teacher_depth) {
            Some(score) => score,
            None => {
                return PerRootResult {
                    record: None,
                    reason: Some(SkipReason::SearchFailed),
                };
            }
        };

    if args
        .max_abs_root_score
        .is_some_and(|limit| teacher_root_score.abs() > limit)
    {
        return PerRootResult {
            record: None,
            reason: Some(SkipReason::RootScoreOutOfRange),
        };
    }

    for student_candidate in &student_top_candidates {
        if contains_move(&teacher_candidates, student_candidate.mv) {
            continue;
        }
        if let Some((score, pv)) = child_search_candidate(
            teacher_model,
            &position,
            student_candidate.mv,
            args.teacher_depth,
        ) {
            teacher_candidates.push(ScoredMove {
                mv: student_candidate.mv,
                score,
                pv,
            });
        }
    }
    teacher_candidates.sort_by(|lhs, rhs| {
        rhs.score
            .partial_cmp(&lhs.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let (student_root_score, student_root_pv) =
        match search_root(student_model, &position, args.student_depth) {
            Some(score) => score,
            None => {
                return PerRootResult {
                    record: None,
                    reason: Some(SkipReason::SearchFailed),
                };
            }
        };

    let teacher_best_move = teacher_root_pv
        .first()
        .copied()
        .or_else(|| teacher_candidates.first().map(|candidate| candidate.mv))
        .map(format_move_usi)
        .unwrap_or_default();

    let student_best_move_for_selection = student_root_pv
        .first()
        .copied()
        .or_else(|| legal_moves.first().copied())
        .unwrap_or(teacher_candidates[0].mv);

    let mut student_scores = Vec::with_capacity(teacher_candidates.len());
    let mut student_pvs = Vec::with_capacity(teacher_candidates.len());

    for candidate in &teacher_candidates {
        let student_score_pv = student_top_candidates
            .iter()
            .find(|student_candidate| student_candidate.mv == candidate.mv)
            .map(|student_candidate| (student_candidate.score, student_candidate.pv.clone()))
            .or_else(|| {
                child_search_candidate(student_model, &position, candidate.mv, args.student_depth)
            })
            .or_else(|| {
                child_oriented_static_score(student_model, &position, candidate.mv, root_turn)
                    .map(|score| (score, Vec::new()))
            });
        let (student_score, student_pv) = match student_score_pv {
            Some(value) => value,
            None => {
                continue;
            }
        };
        student_scores.push((candidate.mv, student_score));
        student_pvs.push((candidate.mv, student_pv));
    }

    if student_scores.is_empty() {
        return PerRootResult {
            record: None,
            reason: Some(SkipReason::SearchFailed),
        };
    }

    student_scores.sort_by(|lhs, rhs| {
        rhs.1
            .partial_cmp(&lhs.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut candidates = Vec::with_capacity(teacher_candidates.len());
    let teacher_best_score = teacher_candidates[0].score;
    for (rank, candidate) in teacher_candidates.iter().enumerate() {
        let student_pv = student_pvs
            .iter()
            .find(|(mv, _)| *mv == candidate.mv)
            .map(|(_, pv)| pv.clone())
            .unwrap_or_default();
        let student_score = student_scores
            .iter()
            .find(|(mv, _)| *mv == candidate.mv)
            .map(|(_, score)| *score)
            .unwrap_or(0.0);
        let student_rank = student_scores
            .iter()
            .position(|(mv, _)| *mv == candidate.mv)
            .unwrap_or(usize::MAX);
        let teacher_pv = {
            let mut pv = Vec::with_capacity(1 + candidate.pv.len());
            pv.push(candidate.mv);
            pv.extend(candidate.pv.iter().copied());
            pv.into_iter().map(format_move_usi).collect::<Vec<_>>()
        };

        let mut student_pv_text = Vec::with_capacity(1 + student_pv.len());
        student_pv_text.push(candidate.mv);
        student_pv_text.extend(student_pv.iter().copied());
        let student_pv_text = student_pv_text
            .into_iter()
            .map(format_move_usi)
            .collect::<Vec<_>>();

        let (teacher_leaf_sfen, teacher_leaf_score) = apply_root_plus_pv(
            &position,
            candidate.mv,
            &candidate.pv,
            teacher_model,
            root_turn,
        );
        let (student_leaf_sfen, student_leaf_score) = apply_root_plus_pv(
            &position,
            candidate.mv,
            &student_pv,
            student_model,
            root_turn,
        );

        candidates.push(CandidateRecord {
            move_usi: format_move_usi(candidate.mv),
            teacher_score: candidate.score,
            teacher_rank: rank,
            regret: (teacher_best_score - candidate.score).max(0.0),
            student_score,
            student_rank,
            teacher_pv,
            student_pv: student_pv_text,
            teacher_leaf_sfen,
            student_leaf_sfen,
            teacher_leaf_score,
            student_leaf_score,
            searched_by_teacher: true,
            selected_by_student: candidate.mv == student_best_move_for_selection,
        });
    }

    let student_best_move = format_move_usi(student_best_move_for_selection);

    let record = TreeRecord {
        schema: "mmto_tree_v1".to_string(),
        version: args.jsonl_version,
        root_index: index,
        sfen: position.to_sfen_owned(),
        teacher_depth: args.teacher_depth,
        student_depth: args.student_depth,
        legal_moves: legal_moves.len(),
        teacher_weights: args
            .teacher_weights
            .as_deref()
            .unwrap_or(&args.student_weights)
            .to_string_lossy()
            .to_string(),
        student_weights: args.student_weights.to_string_lossy().to_string(),
        teacher_root_score,
        student_root_score,
        teacher_best_move,
        student_best_move,
        candidates,
    };

    let line = match serde_json::to_string(&record) {
        Ok(line) => line,
        Err(_) => {
            return PerRootResult {
                record: None,
                reason: Some(SkipReason::SearchFailed),
            };
        }
    };

    PerRootResult {
        record: Some(ProcessedRecord {
            index,
            is_valid,
            line,
        }),
        reason: None,
    }
}

struct DumpStats {
    total_positions: usize,
    invalid_positions: usize,
    train_count: usize,
    valid_count: usize,
    skip_reasons: BTreeMap<String, usize>,
}

impl DumpStats {
    fn new() -> Self {
        Self {
            total_positions: 0,
            invalid_positions: 0,
            train_count: 0,
            valid_count: 0,
            skip_reasons: BTreeMap::new(),
        }
    }

    fn record_skip(&mut self, reason: SkipReason) {
        *self
            .skip_reasons
            .entry(skip_reason_key(reason).to_string())
            .or_insert(0) += 1;
    }
}

fn is_valid_position(index: usize, valid_percent: u8) -> bool {
    if valid_percent == 0 {
        return false;
    }
    let stride = (100 / valid_percent as usize).max(1);
    index % stride == 0
}

fn process_position_chunk(
    chunk: Vec<(usize, Position)>,
    teacher_model: &SparseModel,
    student_model: &SparseModel,
    args: &Args,
    train_writer: &mut BufWriter<File>,
    valid_writer: &mut BufWriter<File>,
    stats: &mut DumpStats,
) -> Result<()> {
    let mut results = chunk
        .into_par_iter()
        .map(|(index, position)| {
            let is_valid = is_valid_position(index, args.valid_percent);
            make_record(
                index,
                position,
                teacher_model,
                student_model,
                args,
                is_valid,
            )
        })
        .collect::<Vec<_>>();

    let mut dumped = Vec::new();
    for item in results.drain(..) {
        if let Some(reason) = item.reason {
            stats.record_skip(reason);
            continue;
        }
        if let Some(record) = item.record {
            dumped.push(record);
        }
    }
    dumped.sort_unstable_by_key(|record| record.index);

    for record in dumped {
        if record.is_valid {
            writeln!(valid_writer, "{}", record.line)
                .with_context(|| format!("failed to write {}", args.valid_output.display()))?;
            stats.valid_count += 1;
        } else {
            writeln!(train_writer, "{}", record.line)
                .with_context(|| format!("failed to write {}", args.train_output.display()))?;
            stats.train_count += 1;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.teacher_depth == 0 {
        return Err(anyhow!("--teacher-depth must be greater than zero"));
    }
    if args.student_depth == 0 {
        return Err(anyhow!("--student-depth must be greater than zero"));
    }
    if args.teacher_score_top == 0 {
        return Err(anyhow!("--teacher-score-top must be greater than zero"));
    }
    if args.candidate_top == 0 && !args.score_all_legal_for_valid {
        return Err(anyhow!(
            "--candidate-top must be greater than zero unless --score-all-legal-for-valid is set"
        ));
    }
    if args.valid_percent > 90 {
        return Err(anyhow!("--valid-percent must be 0..=90"));
    }
    if args
        .max_abs_root_score
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        return Err(anyhow!("--max-abs-root-score must be non-negative"));
    }
    if args.jsonl_version == 0 {
        return Err(anyhow!("--jsonl-version must be at least 1"));
    }
    if args.position_chunk_size == 0 {
        return Err(anyhow!("--position-chunk-size must be greater than zero"));
    }
    if args.jobs > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build_global()
            .map_err(|e| anyhow!("failed to configure rayon thread pool: {e}"))?;
    }

    let teacher_weights = args
        .teacher_weights
        .clone()
        .unwrap_or_else(|| args.student_weights.clone());

    let mut student_model = SparseModel::new(0.0, 0.0);
    student_model
        .load(&args.student_weights)
        .map_err(|e| anyhow!("failed to load {}: {e}", args.student_weights.display()))?;

    let mut teacher_model = SparseModel::new(0.0, 0.0);
    teacher_model
        .load(&teacher_weights)
        .map_err(|e| anyhow!("failed to load {}: {e}", teacher_weights.display()))?;

    println!("student model: {}", args.student_weights.display());
    println!("teacher model: {}", teacher_weights.display());
    println!("skip positions: {}", args.skip_positions);
    println!(
        "streaming positions with chunk size {}",
        args.position_chunk_size
    );

    let mut train_writer = create_writer(&args.train_output)?;
    let mut valid_writer = create_writer(&args.valid_output)?;
    let mut stats = DumpStats::new();
    let mut chunk = Vec::with_capacity(args.position_chunk_size);
    let mut parsed_positions = 0usize;

    'outer: for path in &args.input {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            if args
                .max_positions
                .is_some_and(|limit| stats.total_positions >= limit)
            {
                break 'outer;
            }
            let line = line.with_context(|| format!("failed to read {}", path.display()))?;
            let Some(position) = parse_position_line(&line) else {
                stats.invalid_positions += 1;
                continue;
            };
            if parsed_positions < args.skip_positions {
                parsed_positions += 1;
                continue;
            }
            parsed_positions += 1;
            let index = stats.total_positions;
            stats.total_positions += 1;
            chunk.push((index, position));
            if chunk.len() >= args.position_chunk_size {
                let current = std::mem::take(&mut chunk);
                process_position_chunk(
                    current,
                    &teacher_model,
                    &student_model,
                    &args,
                    &mut train_writer,
                    &mut valid_writer,
                    &mut stats,
                )?;
            }
        }
    }
    if !chunk.is_empty() {
        process_position_chunk(
            chunk,
            &teacher_model,
            &student_model,
            &args,
            &mut train_writer,
            &mut valid_writer,
            &mut stats,
        )?;
    }
    if stats.total_positions == 0 {
        return Err(anyhow!("no valid positions loaded"));
    }

    train_writer
        .flush()
        .with_context(|| format!("failed to flush {}", args.train_output.display()))?;
    valid_writer
        .flush()
        .with_context(|| format!("failed to flush {}", args.valid_output.display()))?;

    println!("total positions: {}", stats.total_positions);
    println!("train records: {}", stats.train_count);
    println!("valid records: {}", stats.valid_count);
    println!(
        "skipped positions: {}",
        stats
            .total_positions
            .saturating_sub(stats.train_count + stats.valid_count)
    );
    if stats.invalid_positions > 0 {
        println!("skipped[invalid_sfen]: {}", stats.invalid_positions);
    }
    for (reason, count) in stats.skip_reasons {
        println!("skipped[{reason}]: {count}");
    }

    Ok(())
}
