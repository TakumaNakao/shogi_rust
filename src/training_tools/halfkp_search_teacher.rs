use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Deserialize;
use shogi_ai::ai::{SearchLimits, ShogiAI};
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
use shogi_ai::training_tools::scratch_lineage::{
    require_accepted_parent, require_scratch_model_manifest,
};
use shogi_ai::utils::{parse_usi_move_for_color, position_from_sfen_or_usi};
use shogi_core::{Color, Move};
use shogi_lib::Position;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

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
    model_manifest: PathBuf,
    #[arg(long)]
    promotion_registry: PathBuf,
    #[arg(long)]
    gate_report: PathBuf,
    #[arg(long)]
    parent_generation: u32,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    reuse_if_matches: bool,
    /// Search each root once and emit one codec-compatible PV child.
    #[arg(long, default_value_t = false)]
    value_only: bool,
    #[arg(long, default_value_t = 2)]
    selection_depth: u8,
    #[arg(long, default_value_t = 4)]
    teacher_depth: u8,
    /// Minimum fully completed iterative-deepening depth accepted under a node cap.
    #[arg(long, default_value_t = 4)]
    min_completed_depth: u8,
    #[arg(long, default_value_t = 20_000)]
    max_nodes_per_search: u64,
    #[arg(long, default_value_t = 8)]
    candidate_top: usize,
    #[arg(long, default_value_t = 2)]
    tactical_candidate_limit: usize,
    /// Score every legal move at teacher depth and report top-K preselection recall.
    #[arg(long, default_value_t = false)]
    audit_all_legal_recall: bool,
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
    /// Skip this many source lines before filling the immutable output shard.
    #[arg(long, default_value_t = 0)]
    skip_positions: usize,
    /// Deterministically retain every Nth source line after `skip-positions`.
    #[arg(long, default_value_t = 1)]
    sample_stride: usize,
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
    resource_limited: usize,
    search_calls: u64,
    search_nodes: u64,
    qsearch_nodes: u64,
    excluded_check: usize,
    with_result: usize,
    with_game_move: usize,
    candidates: usize,
    hard_records: usize,
    randomized_records: usize,
    root_wall_micros: Vec<u64>,
    recall_roots: usize,
    recall_hits: usize,
}

struct Processed {
    record: Option<SearchTeacherRecord>,
    parse_failed: bool,
    illegal_teacher: bool,
    search_failed: bool,
    resource_limited: bool,
    search_calls: u64,
    search_nodes: u64,
    qsearch_nodes: u64,
    elapsed_micros: u64,
    recall_eligible: bool,
    recall_hit: bool,
    excluded_check: bool,
}

struct TeacherSession {
    ai: ShogiAI<Arc<EngineEvaluator>, HISTORY_CAPACITY>,
}

struct MoveScore {
    score: f32,
    packed: PackedHalfKpPosition,
    best_move: Move,
    completed_depth: u8,
    nodes: u64,
    qnodes: u64,
}

enum ScoreMove {
    Complete(MoveScore),
    ResourceLimited { nodes: u64, qnodes: u64 },
    Failed,
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
        max_nodes: u64,
    ) -> ScoreMove {
        let mut child = position.clone();
        child.do_move(mv);
        let Some(packed) = PackedHalfKpPosition::from_position(&child) else {
            return ScoreMove::Failed;
        };
        if child.legal_moves().is_empty() {
            return ScoreMove::Complete(MoveScore {
                score: SCORE_CLAMP_CP,
                packed,
                best_move: mv,
                completed_depth: depth,
                nodes: 0,
                qnodes: 0,
            });
        }
        self.ai.reset_for_independent_search();
        self.ai.sennichite_detector.record_initial_position(&child);
        let (result, stats, limited) = self.ai.alpha_beta_search_with_max_nodes(
            &mut child,
            depth.saturating_sub(1),
            f32::NEG_INFINITY,
            f32::INFINITY,
            max_nodes,
        );
        match result {
            Some((child_score, _)) => ScoreMove::Complete(MoveScore {
                score: -sanitize_score(child_score),
                packed,
                best_move: mv,
                completed_depth: depth,
                nodes: stats.nodes,
                qnodes: stats.quiescence_nodes,
            }),
            None if limited => ScoreMove::ResourceLimited {
                nodes: stats.nodes,
                qnodes: stats.quiescence_nodes,
            },
            None => ScoreMove::Failed,
        }
    }

    fn score_root(
        &mut self,
        position: &Position,
        depth: u8,
        min_completed_depth: u8,
        max_nodes: u64,
    ) -> ScoreMove {
        let mut root = position.clone();
        self.ai.reset_for_independent_search();
        self.ai.sennichite_detector.record_initial_position(&root);
        let outcome = self.ai.search(
            &mut root,
            SearchLimits::new(depth, None).with_max_nodes(max_nodes),
        );
        let stats = outcome.stats;
        match outcome.root {
            Some(root_result)
                if root_result.completed_depth >= min_completed_depth
                    && root_result.score.is_some() =>
            {
                let best_move = root_result.best_move;
                let mut child = position.clone();
                child.do_move(best_move);
                let Some(packed) = PackedHalfKpPosition::from_position(&child) else {
                    return ScoreMove::Failed;
                };
                ScoreMove::Complete(MoveScore {
                    score: sanitize_score(root_result.score.expect("checked above")),
                    packed,
                    best_move,
                    completed_depth: root_result.completed_depth,
                    nodes: stats.nodes,
                    qnodes: stats.quiescence_nodes,
                })
            }
            _ if stats.nodes >= max_nodes => ScoreMove::ResourceLimited {
                nodes: stats.nodes,
                qnodes: stats.quiescence_nodes,
            },
            _ => ScoreMove::Failed,
        }
    }
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    if args.teacher_depth == 0
        || args.min_completed_depth == 0
        || args.min_completed_depth > args.teacher_depth
        || args.selection_depth == 0
        || args.max_nodes_per_search == 0
        || args.candidate_top == 0
        || args.chunk_size == 0
        || args.sample_stride == 0
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
    require_accepted_parent(
        &args.promotion_registry,
        &args.weights,
        &args.gate_report,
        args.parent_generation,
    )?;
    require_scratch_model_manifest(&args.model_manifest, &args.weights, args.parent_generation)?;
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
            "min_completed_depth": args.min_completed_depth,
            "max_nodes_per_search": args.max_nodes_per_search,
            "value_only": args.value_only,
            "candidate_top": args.candidate_top,
            "tactical_candidate_limit": args.tactical_candidate_limit,
            "audit_all_legal_recall": args.audit_all_legal_recall,
            "hard_teacher_depth": args.hard_teacher_depth,
            "hard_percent": args.hard_percent,
            "hard_score_gap_cp": args.hard_score_gap_cp,
            "hard_in_check": args.hard_in_check,
            "hard_endgame": args.hard_endgame,
            "randomize_max_plies": args.randomize_max_plies,
            "exclude_in_check": args.exclude_in_check,
            "skip_positions": args.skip_positions,
            "max_positions": args.max_positions,
            "sample_stride": args.sample_stride,
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
    let mut source_index = 0usize;

    for path in &args.input {
        let reader =
            BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
        for line in reader.lines() {
            let index = source_index;
            source_index += 1;
            if index < args.skip_positions {
                continue;
            }
            if (index - args.skip_positions) % args.sample_stride != 0 {
                continue;
            }
            if stats.input >= max_positions {
                break;
            }
            stats.input += 1;
            chunk.push((index, line?));
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
        "complete input={} written={} records={} duplicates={} parse_failed={} illegal_teacher={} search_failed={} resource_limited={} search_calls={} search_nodes={} qsearch_nodes={} root_wall_us_p50={} root_wall_us_p95={} root_wall_us_max={} recall_roots={} recall_hits={} recall_rate={:.6} excluded_check={} with_result={} with_game_move={} candidates={} hard_records={} randomized_records={}",
        stats.input,
        stats.written,
        records,
        stats.duplicate,
        stats.parse_failed,
        stats.illegal_teacher,
        stats.search_failed,
        stats.resource_limited,
        stats.search_calls,
        stats.search_nodes,
        stats.qsearch_nodes,
        percentile(&stats.root_wall_micros, 50),
        percentile(&stats.root_wall_micros, 95),
        stats.root_wall_micros.iter().copied().max().unwrap_or(0),
        stats.recall_roots,
        stats.recall_hits,
        stats.recall_hits as f64 / stats.recall_roots.max(1) as f64,
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
        stats.resource_limited += usize::from(item.resource_limited);
        stats.search_calls += item.search_calls;
        stats.search_nodes += item.search_nodes;
        stats.qsearch_nodes += item.qsearch_nodes;
        stats.root_wall_micros.push(item.elapsed_micros);
        stats.recall_roots += usize::from(item.recall_eligible);
        stats.recall_hits += usize::from(item.recall_hit);
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
    let started = Instant::now();
    let mut processed = process_line_inner(index, line, args, session);
    processed.elapsed_micros = started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
    processed
}

fn process_line_inner(
    index: usize,
    line: &str,
    args: &Args,
    session: &mut TeacherSession,
) -> Processed {
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
    if legal_moves.is_empty() || (!args.value_only && legal_moves.len() < 2) {
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

    if args.value_only {
        return process_value_only(
            input,
            position,
            game_move,
            illegal_teacher,
            randomized,
            args,
            session,
        );
    }

    let mut search_calls = 0u64;
    let mut search_nodes = 0u64;
    let mut qsearch_nodes = 0u64;
    let mut shallow = Vec::with_capacity(legal_moves.len());
    for &mv in &legal_moves {
        search_calls += 1;
        match session.score_move(
            &position,
            mv,
            args.selection_depth,
            args.max_nodes_per_search,
        ) {
            ScoreMove::Complete(scored) => {
                search_nodes += scored.nodes;
                qsearch_nodes += scored.qnodes;
                shallow.push((mv, scored));
            }
            ScoreMove::ResourceLimited { nodes, qnodes } => {
                search_nodes += nodes;
                qsearch_nodes += qnodes;
                return failed_with_metrics(
                    false,
                    illegal_teacher,
                    false,
                    true,
                    false,
                    search_calls,
                    search_nodes,
                    qsearch_nodes,
                );
            }
            ScoreMove::Failed => {
                return failed_with_metrics(
                    false,
                    illegal_teacher,
                    true,
                    false,
                    false,
                    search_calls,
                    search_nodes,
                    qsearch_nodes,
                );
            }
        }
    }
    if shallow.len() < 2 {
        return failed(false, illegal_teacher, true, false);
    }
    shallow.sort_by(|lhs, rhs| rhs.1.score.total_cmp(&lhs.1.score));

    let mut selected = shallow
        .iter()
        .take(args.candidate_top)
        .map(|(mv, _)| *mv)
        .collect::<Vec<_>>();
    let mut tactical_added = 0usize;
    for (mv, _) in &shallow {
        let mv = *mv;
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
        .map(|second| shallow[0].1.score - second.1.score)
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
    let mut deep_all = Vec::new();
    let mut recall_eligible = false;
    let mut recall_hit = false;
    if args.audit_all_legal_recall && teacher_depth != args.selection_depth {
        deep_all.reserve(legal_moves.len());
        for &mv in &legal_moves {
            search_calls += 1;
            match session.score_move(&position, mv, teacher_depth, args.max_nodes_per_search) {
                ScoreMove::Complete(scored) => {
                    search_nodes += scored.nodes;
                    qsearch_nodes += scored.qnodes;
                    deep_all.push((mv, scored));
                }
                ScoreMove::ResourceLimited { nodes, qnodes } => {
                    search_nodes += nodes;
                    qsearch_nodes += qnodes;
                    return failed_with_metrics(
                        false,
                        illegal_teacher,
                        false,
                        true,
                        false,
                        search_calls,
                        search_nodes,
                        qsearch_nodes,
                    );
                }
                ScoreMove::Failed => {
                    return failed_with_metrics(
                        false,
                        illegal_teacher,
                        true,
                        false,
                        false,
                        search_calls,
                        search_nodes,
                        qsearch_nodes,
                    );
                }
            }
        }
        deep_all.sort_by(|lhs, rhs| rhs.1.score.total_cmp(&lhs.1.score));
        recall_eligible = true;
        recall_hit = selected.contains(&deep_all[0].0);
    }
    let mut candidates = Vec::with_capacity(selected.len());
    for mv in selected {
        let scored = if teacher_depth == args.selection_depth {
            shallow
                .iter()
                .find(|(shallow_move, _)| *shallow_move == mv)
                .map(|(_, scored)| MoveScore {
                    score: scored.score,
                    packed: scored.packed.clone(),
                    best_move: scored.best_move,
                    completed_depth: scored.completed_depth,
                    nodes: 0,
                    qnodes: 0,
                })
                .expect("selected move was scored during selection")
        } else if args.audit_all_legal_recall {
            deep_all
                .iter()
                .find(|(deep_move, _)| *deep_move == mv)
                .map(|(_, scored)| MoveScore {
                    score: scored.score,
                    packed: scored.packed.clone(),
                    best_move: scored.best_move,
                    completed_depth: scored.completed_depth,
                    nodes: 0,
                    qnodes: 0,
                })
                .expect("selected move was scored by recall audit")
        } else {
            search_calls += 1;
            match session.score_move(&position, mv, teacher_depth, args.max_nodes_per_search) {
                ScoreMove::Complete(scored) => {
                    search_nodes += scored.nodes;
                    qsearch_nodes += scored.qnodes;
                    scored
                }
                ScoreMove::ResourceLimited { nodes, qnodes } => {
                    search_nodes += nodes;
                    qsearch_nodes += qnodes;
                    return failed_with_metrics(
                        false,
                        illegal_teacher,
                        false,
                        true,
                        false,
                        search_calls,
                        search_nodes,
                        qsearch_nodes,
                    );
                }
                ScoreMove::Failed => {
                    return failed_with_metrics(
                        false,
                        illegal_teacher,
                        true,
                        false,
                        false,
                        search_calls,
                        search_nodes,
                        qsearch_nodes,
                    );
                }
            }
        };
        let mut flags = 0;
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
            score_cp: scored.score,
            child: scored.packed,
        });
    }
    if candidates.len() < 2 {
        return failed(false, illegal_teacher, true, false);
    }
    candidates.sort_by(|lhs, rhs| rhs.score_cp.total_cmp(&lhs.score_cp));
    for candidate in &mut candidates {
        candidate.flags &= !CANDIDATE_SEARCH_BEST;
    }
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
        resource_limited: false,
        search_calls,
        search_nodes,
        qsearch_nodes,
        elapsed_micros: 0,
        recall_eligible,
        recall_hit,
        excluded_check: false,
    }
}

fn process_value_only(
    input: InputRecord,
    position: Position,
    game_move: Option<Move>,
    illegal_teacher: bool,
    randomized: bool,
    args: &Args,
    session: &mut TeacherSession,
) -> Processed {
    let scored = match session.score_root(
        &position,
        args.teacher_depth,
        args.min_completed_depth,
        args.max_nodes_per_search,
    ) {
        ScoreMove::Complete(scored) => scored,
        ScoreMove::ResourceLimited { nodes, qnodes } => {
            return failed_with_metrics(
                false,
                illegal_teacher,
                false,
                true,
                false,
                1,
                nodes,
                qnodes,
            );
        }
        ScoreMove::Failed => {
            return failed_with_metrics(false, illegal_teacher, true, false, false, 1, 0, 0);
        }
    };
    let Some(root) = PackedHalfKpPosition::from_position(&position) else {
        return failed_with_metrics(
            true,
            illegal_teacher,
            false,
            false,
            false,
            1,
            scored.nodes,
            scored.qnodes,
        );
    };
    let mut flags = CANDIDATE_SEARCH_BEST;
    if Some(scored.best_move) == game_move {
        flags |= CANDIDATE_GAME_MOVE;
    }
    if is_tactical_move(&position, scored.best_move) {
        flags |= CANDIDATE_TACTICAL;
    }
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
    Processed {
        record: Some(SearchTeacherRecord {
            position_hash: PositionHasher::calculate_hash(&position),
            ply: input.ply.unwrap_or(0),
            phase,
            teacher_depth: scored.completed_depth,
            result: (!randomized)
                .then(|| result_for(&input, position.side_to_move()))
                .flatten(),
            root_search_score_cp: scored.score,
            sample_weight: input.sample_weight.unwrap_or(1.0).clamp(0.01, 10.0),
            root,
            candidates: vec![SearchTeacherCandidate {
                flags,
                score_cp: scored.score,
                child: scored.packed,
            }],
        }),
        parse_failed: false,
        illegal_teacher,
        search_failed: false,
        resource_limited: false,
        search_calls: 1,
        search_nodes: scored.nodes,
        qsearch_nodes: scored.qnodes,
        elapsed_micros: 0,
        recall_eligible: false,
        recall_hit: false,
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
    failed_with_metrics(
        parse_failed,
        illegal_teacher,
        search_failed,
        false,
        excluded_check,
        0,
        0,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
fn failed_with_metrics(
    parse_failed: bool,
    illegal_teacher: bool,
    search_failed: bool,
    resource_limited: bool,
    excluded_check: bool,
    search_calls: u64,
    search_nodes: u64,
    qsearch_nodes: u64,
) -> Processed {
    Processed {
        record: None,
        parse_failed,
        illegal_teacher,
        search_failed,
        resource_limited,
        search_calls,
        search_nodes,
        qsearch_nodes,
        elapsed_micros: 0,
        recall_eligible: false,
        recall_hit: false,
        excluded_check,
    }
}

fn percentile(values: &[u64], percent: usize) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let index = ((sorted.len() - 1) * percent) / 100;
    sorted[index]
}

fn print_progress(stats: &Stats) {
    eprintln!(
        "input={} written={} search_failed={} resource_limited={} calls={} nodes={} qnodes={} duplicates={}",
        stats.input,
        stats.written,
        stats.search_failed,
        stats.resource_limited,
        stats.search_calls,
        stats.search_nodes,
        stats.qsearch_nodes,
        stats.duplicate
    );
}
