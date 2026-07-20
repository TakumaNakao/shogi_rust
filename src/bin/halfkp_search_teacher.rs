use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Deserialize;
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::EngineEvaluator;
use shogi_ai::halfkp_training::{
    read_search_teacher_manifest, PackedHalfKpPosition, SearchTeacherCandidate,
    SearchTeacherProvenance, SearchTeacherRecord, SearchTeacherWriter, CANDIDATE_GAME_MOVE,
    CANDIDATE_RANDOM, CANDIDATE_SEARCH_BEST, CANDIDATE_TACTICAL,
};
use shogi_ai::position_hash::PositionHasher;
use shogi_ai::training_data::{
    artifact_metadata, capture_run_environment, line_artifact_metadata, sha256_file, TrainingPhase,
    PHASE_POLICY_VERSION, SPLIT_POLICY_VERSION,
};
use shogi_ai::utils::{parse_usi_move_for_color, position_from_sfen_or_usi};
use shogi_core::{Color, Move};
use shogi_lib::Position;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;

const HISTORY_CAPACITY: usize = 256;
const SCORE_CLAMP_CP: f32 = 6000.0;

#[derive(Parser, Debug)]
#[command(about = "Generate packed HalfKP teachers using this engine's own alpha-beta search")]
struct Args {
    #[arg(long)]
    weights: PathBuf,
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long)]
    parent_manifest: Vec<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    reuse_if_matches: bool,
    #[arg(long, default_value_t = 2)]
    selection_depth: u8,
    #[arg(long, default_value_t = 4)]
    teacher_depth: u8,
    #[arg(long, default_value_t = 8)]
    candidate_top: usize,
    #[arg(long, default_value_t = 2)]
    tactical_candidate_limit: usize,
    #[arg(long)]
    hard_teacher_depth: Option<u8>,
    #[arg(long, default_value_t = 20)]
    hard_percent: u8,
    #[arg(long, default_value_t = 100.0)]
    hard_score_gap_cp: f32,
    #[arg(long, default_value_t = false)]
    hard_in_check: bool,
    #[arg(long, default_value_t = false)]
    hard_endgame: bool,
    #[arg(long, default_value_t = 0)]
    randomize_max_plies: u8,
    #[arg(long)]
    max_positions: Option<usize>,
    #[arg(long, default_value_t = 128)]
    chunk_size: usize,
    #[arg(long, default_value_t = 0)]
    jobs: usize,
    #[arg(long, default_value_t = 20260718)]
    seed: u64,
    #[arg(long, default_value_t = false)]
    exclude_in_check: bool,
}

#[derive(Clone, Deserialize)]
struct InputRecord {
    sfen: String,
    #[serde(default)]
    teacher_move: Option<String>,
    #[serde(default)]
    winner: Option<String>,
    #[serde(default)]
    result_known: Option<bool>,
    #[serde(default)]
    ply: Option<u16>,
    #[serde(default)]
    phase: Option<String>,
    #[serde(default)]
    sample_weight: Option<f32>,
}

#[derive(Default)]
struct Stats {
    input: usize,
    written: usize,
    duplicate: usize,
    parse_failed: usize,
    illegal_teacher: usize,
    search_failed: usize,
    excluded_check: usize,
    with_result: usize,
    with_game_move: usize,
    candidates: usize,
    hard_records: usize,
    randomized_records: usize,
}

struct Processed {
    record: Option<SearchTeacherRecord>,
    parse_failed: bool,
    illegal_teacher: bool,
    search_failed: bool,
    excluded_check: bool,
}

struct TeacherSession {
    ai: ShogiAI<Arc<EngineEvaluator>, HISTORY_CAPACITY>,
}

impl TeacherSession {
    fn new(evaluator: Arc<EngineEvaluator>) -> Self {
        let mut ai = ShogiAI::new(evaluator);
        ai.set_emit_info(false);
        Self { ai }
    }

    fn score_move(
        &mut self,
        position: &Position,
        mv: Move,
        depth: u8,
    ) -> Option<(f32, PackedHalfKpPosition)> {
        let mut child = position.clone();
        child.do_move(mv);
        let packed = PackedHalfKpPosition::from_position(&child)?;
        if child.legal_moves().is_empty() {
            return Some((SCORE_CLAMP_CP, packed));
        }
        self.ai.reset_for_independent_search();
        self.ai.sennichite_detector.record_initial_position(&child);
        let (child_score, _) = self.ai.alpha_beta_search(
            &mut child,
            depth.saturating_sub(1),
            f32::NEG_INFINITY,
            f32::INFINITY,
        )?;
        Some((-sanitize_score(child_score), packed))
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.teacher_depth == 0
        || args.selection_depth == 0
        || args.candidate_top == 0
        || args.chunk_size == 0
        || args.hard_percent > 100
        || args.hard_score_gap_cp < 0.0
        || args
            .hard_teacher_depth
            .is_some_and(|depth| depth < args.teacher_depth)
    {
        return Err(anyhow!(
            "depth, candidate-top, and chunk-size must be positive"
        ));
    }
    let evaluator = Arc::new(EngineEvaluator::new(&args.weights, 0.0)?);
    if evaluator.name() != "halfkp" {
        return Err(anyhow!("--weights must be a HalfKP model"));
    }
    let pool = if args.jobs > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(args.jobs)
                .build()?,
        )
    } else {
        None
    };
    let engine_binary = std::env::current_exe()
        .ok()
        .and_then(|path| artifact_metadata(&path, None).ok());
    let provenance = SearchTeacherProvenance {
        environment: Some(capture_run_environment()),
        inputs: args
            .input
            .iter()
            .map(|path| line_artifact_metadata(path))
            .collect::<Result<Vec<_>>>()?,
        model: Some(artifact_metadata(&args.weights, None)?),
        engine_binary,
        feature_profile: if cfg!(feature = "halfkp64") {
            "halfkp64"
        } else {
            "halfkp32"
        }
        .to_string(),
        search_limits: serde_json::json!({
            "selection_depth": args.selection_depth,
            "teacher_depth": args.teacher_depth,
            "candidate_top": args.candidate_top,
            "tactical_candidate_limit": args.tactical_candidate_limit,
            "hard_teacher_depth": args.hard_teacher_depth,
            "hard_percent": args.hard_percent,
            "hard_score_gap_cp": args.hard_score_gap_cp,
            "hard_in_check": args.hard_in_check,
            "hard_endgame": args.hard_endgame,
            "randomize_max_plies": args.randomize_max_plies,
            "exclude_in_check": args.exclude_in_check,
        }),
        jobs: Some(
            pool.as_ref()
                .map_or_else(rayon::current_num_threads, |pool| {
                    pool.current_num_threads()
                }),
        ),
        random_seeds: vec![args.seed],
        phase_policy_version: Some(PHASE_POLICY_VERSION),
        split_policy_version: Some(SPLIT_POLICY_VERSION),
        parent_manifest_sha256: args
            .parent_manifest
            .iter()
            .map(|path| sha256_file(path))
            .collect::<Result<Vec<_>>>()?,
    };
    if args.reuse_if_matches && reusable_output(&args.output, &provenance)? {
        println!("reused output={}", args.output.display());
        return Ok(());
    }
    let mut writer = SearchTeacherWriter::create(&args.output)?;
    writer.set_provenance(provenance);
    let mut stats = Stats::default();
    let mut seen = HashSet::new();
    let mut chunk = Vec::with_capacity(args.chunk_size);
    let max_positions = args.max_positions.unwrap_or(usize::MAX);

    for path in &args.input {
        let reader =
            BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
        for line in reader.lines() {
            if stats.input >= max_positions {
                break;
            }
            stats.input += 1;
            chunk.push((stats.input - 1, line?));
            if chunk.len() >= args.chunk_size {
                process_chunk(
                    &mut chunk,
                    &args,
                    evaluator.clone(),
                    pool.as_ref(),
                    &mut writer,
                    &mut seen,
                    &mut stats,
                )?;
                print_progress(&stats);
            }
        }
        if stats.input >= max_positions {
            break;
        }
    }
    if !chunk.is_empty() {
        process_chunk(
            &mut chunk,
            &args,
            evaluator,
            pool.as_ref(),
            &mut writer,
            &mut seen,
            &mut stats,
        )?;
    }
    let records = writer.finish()?;
    println!(
        "complete input={} written={} records={} duplicates={} parse_failed={} illegal_teacher={} search_failed={} excluded_check={} with_result={} with_game_move={} candidates={} hard_records={} randomized_records={}",
        stats.input,
        stats.written,
        records,
        stats.duplicate,
        stats.parse_failed,
        stats.illegal_teacher,
        stats.search_failed,
        stats.excluded_check,
        stats.with_result,
        stats.with_game_move,
        stats.candidates,
        stats.hard_records,
        stats.randomized_records
    );
    Ok(())
}

fn reusable_output(path: &std::path::Path, provenance: &SearchTeacherProvenance) -> Result<bool> {
    let Some(manifest) = read_search_teacher_manifest(path)? else {
        return Ok(false);
    };
    let Some(output) = manifest.output else {
        return Ok(false);
    };
    Ok(
        manifest.stage_fingerprint == provenance.stage_fingerprint()?
            && manifest.teacher_semantics_version
                == shogi_ai::halfkp_training::SEARCH_TEACHER_SEMANTICS_VERSION
            && output.sha256 == sha256_file(path)?,
    )
}

#[allow(clippy::too_many_arguments)]
fn process_chunk(
    chunk: &mut Vec<(usize, String)>,
    args: &Args,
    evaluator: Arc<EngineEvaluator>,
    pool: Option<&rayon::ThreadPool>,
    writer: &mut SearchTeacherWriter,
    seen: &mut HashSet<u64>,
    stats: &mut Stats,
) -> Result<()> {
    let compute = || {
        chunk
            .par_iter()
            .map_init(
                || TeacherSession::new(evaluator.clone()),
                |session, (index, line)| process_line(*index, line, args, session),
            )
            .collect::<Vec<_>>()
    };
    let processed = pool.map_or_else(compute, |pool| pool.install(compute));
    for item in processed {
        stats.parse_failed += usize::from(item.parse_failed);
        stats.illegal_teacher += usize::from(item.illegal_teacher);
        stats.search_failed += usize::from(item.search_failed);
        stats.excluded_check += usize::from(item.excluded_check);
        let Some(record) = item.record else {
            continue;
        };
        if !seen.insert(record.position_hash) {
            stats.duplicate += 1;
            continue;
        }
        stats.with_result += usize::from(record.result.is_some());
        stats.with_game_move += usize::from(
            record
                .candidates
                .iter()
                .any(|candidate| candidate.flags & CANDIDATE_GAME_MOVE != 0),
        );
        stats.candidates += record.candidates.len();
        stats.hard_records += usize::from(record.teacher_depth > args.teacher_depth);
        stats.randomized_records += usize::from(args.randomize_max_plies > 0);
        writer.write_record(&record)?;
        stats.written += 1;
    }
    chunk.clear();
    Ok(())
}

fn process_line(index: usize, line: &str, args: &Args, session: &mut TeacherSession) -> Processed {
    let Ok(input) = serde_json::from_str::<InputRecord>(line) else {
        return failed(true, false, false, false);
    };
    let Some(mut position) = position_from_sfen_or_usi(&input.sfen) else {
        return failed(true, false, false, false);
    };
    let randomized = args.randomize_max_plies > 0;
    if randomized
        && !randomize_position(
            &mut position,
            args.randomize_max_plies,
            args.seed ^ index as u64,
        )
    {
        return failed(false, false, true, false);
    }
    if args.exclude_in_check && position.in_check() {
        return failed(false, false, false, true);
    }
    let legal_moves = position.legal_moves();
    if legal_moves.len() < 2 {
        return failed(false, false, true, false);
    }
    let game_move = (!randomized)
        .then_some(input.teacher_move.as_deref())
        .flatten()
        .and_then(|text| parse_usi_move_for_color(text, position.side_to_move()));
    let illegal_teacher = !randomized
        && input.teacher_move.is_some()
        && game_move.is_none_or(|mv| !legal_moves.contains(&mv));
    let game_move = game_move.filter(|mv| legal_moves.contains(mv));

    let mut shallow = legal_moves
        .iter()
        .filter_map(|&mv| {
            session
                .score_move(&position, mv, args.selection_depth)
                .map(|(score, _)| (mv, score))
        })
        .collect::<Vec<_>>();
    if shallow.len() < 2 {
        return failed(false, illegal_teacher, true, false);
    }
    shallow.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));

    let mut selected = shallow
        .iter()
        .take(args.candidate_top)
        .map(|&(mv, _)| mv)
        .collect::<Vec<_>>();
    let mut tactical_added = 0usize;
    for &(mv, _) in &shallow {
        if tactical_added >= args.tactical_candidate_limit {
            break;
        }
        if is_tactical_move(&position, mv) && !selected.contains(&mv) {
            selected.push(mv);
            tactical_added += 1;
        }
    }
    if let Some(mv) = game_move {
        if !selected.contains(&mv) {
            selected.push(mv);
        }
    }
    let random_index = ((PositionHasher::calculate_hash(&position) ^ args.seed ^ index as u64)
        % legal_moves.len() as u64) as usize;
    let random_move = legal_moves[random_index];
    if !selected.contains(&random_move) {
        selected.push(random_move);
    }

    let shallow_best = shallow[0].0;
    let phase = if randomized {
        phase_for_ply(
            input
                .ply
                .unwrap_or(0)
                .saturating_add(u16::from(args.randomize_max_plies)),
        )
    } else {
        input
            .phase
            .as_deref()
            .map(phase_code)
            .unwrap_or_else(|| phase_for_ply(input.ply.unwrap_or(0)))
    };
    let shallow_gap = shallow
        .get(1)
        .map(|second| shallow[0].1 - second.1)
        .unwrap_or(f32::INFINITY);
    let hard_eligible = shallow_gap <= args.hard_score_gap_cp
        || (args.hard_in_check && position.in_check())
        || (args.hard_endgame && phase == 2);
    let hard_selected = hard_eligible
        && args.hard_teacher_depth.is_some()
        && (PositionHasher::calculate_hash(&position) ^ args.seed) % 100
            < u64::from(args.hard_percent);
    let teacher_depth = if hard_selected {
        args.hard_teacher_depth.unwrap_or(args.teacher_depth)
    } else {
        args.teacher_depth
    };
    let mut candidates = Vec::with_capacity(selected.len());
    for mv in selected {
        let Some((score_cp, child)) = session.score_move(&position, mv, teacher_depth) else {
            continue;
        };
        let mut flags = 0;
        if mv == shallow_best {
            flags |= CANDIDATE_SEARCH_BEST;
        }
        if game_move == Some(mv) {
            flags |= CANDIDATE_GAME_MOVE;
        }
        if mv == random_move {
            flags |= CANDIDATE_RANDOM;
        }
        if is_tactical_move(&position, mv) {
            flags |= CANDIDATE_TACTICAL;
        }
        candidates.push(SearchTeacherCandidate {
            flags,
            score_cp,
            child,
        });
    }
    if candidates.len() < 2 {
        return failed(false, illegal_teacher, true, false);
    }
    candidates.sort_by(|lhs, rhs| rhs.score_cp.total_cmp(&lhs.score_cp));
    candidates[0].flags |= CANDIDATE_SEARCH_BEST;
    let root_search_score_cp = candidates[0].score_cp;
    let Some(root) = PackedHalfKpPosition::from_position(&position) else {
        return failed(true, illegal_teacher, false, false);
    };
    let result = if randomized {
        None
    } else {
        result_for(&input, position.side_to_move())
    };
    Processed {
        record: Some(SearchTeacherRecord {
            position_hash: PositionHasher::calculate_hash(&position),
            ply: input.ply.unwrap_or(0),
            phase,
            teacher_depth,
            result,
            root_search_score_cp,
            sample_weight: input.sample_weight.unwrap_or(1.0).clamp(0.01, 10.0),
            root,
            candidates,
        }),
        parse_failed: false,
        illegal_teacher,
        search_failed: false,
        excluded_check: false,
    }
}

fn is_tactical_move(position: &Position, mv: Move) -> bool {
    position.is_check_move(mv)
        || match mv {
            Move::Normal { to, .. } => position.piece_at(to).is_some(),
            Move::Drop { .. } => false,
        }
}

fn randomize_position(position: &mut Position, max_plies: u8, seed: u64) -> bool {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ PositionHasher::calculate_hash(position));
    let plies = rng.gen_range(1..=max_plies);
    for _ in 0..plies {
        let legal = position.legal_moves();
        if legal.is_empty() {
            return false;
        }
        position.do_move(legal[rng.gen_range(0..legal.len())]);
    }
    true
}

fn sanitize_score(score: f32) -> f32 {
    if score == f32::INFINITY {
        SCORE_CLAMP_CP
    } else if score == f32::NEG_INFINITY {
        -SCORE_CLAMP_CP
    } else {
        score.clamp(-SCORE_CLAMP_CP, SCORE_CLAMP_CP)
    }
}

fn result_for(input: &InputRecord, side: Color) -> Option<f32> {
    match input.winner.as_deref() {
        Some("black") => Some(if side == Color::Black { 1.0 } else { 0.0 }),
        Some("white") => Some(if side == Color::White { 1.0 } else { 0.0 }),
        None if input.result_known == Some(true) => Some(0.5),
        _ => None,
    }
}

fn phase_code(value: &str) -> u8 {
    TrainingPhase::parse(value).map_or(3, TrainingPhase::code)
}

fn phase_for_ply(ply: u16) -> u8 {
    TrainingPhase::for_ply(usize::from(ply)).code()
}

fn failed(
    parse_failed: bool,
    illegal_teacher: bool,
    search_failed: bool,
    excluded_check: bool,
) -> Processed {
    Processed {
        record: None,
        parse_failed,
        illegal_teacher,
        search_failed,
        excluded_check,
    }
}

fn print_progress(stats: &Stats) {
    if stats.input & 1023 == 0 {
        eprintln!(
            "input={} written={} search_failed={} duplicates={}",
            stats.input, stats.written, stats.search_failed, stats.duplicate
        );
    }
}
