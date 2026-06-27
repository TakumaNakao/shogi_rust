use anyhow::{anyhow, Result};
use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{Evaluator, SparseModel};
use shogi_ai::utils::{format_move_usi, position_from_sfen_or_usi};
use shogi_core::Move;
use shogi_lib::Position;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process;

const HISTORY_CAPACITY: usize = 256;
const SCORE_LIMIT: f32 = 100_000.0;

#[derive(Parser, Debug)]
#[command(about = "Gate candidate KPP weights by re-searching selected moves and teacher regret")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    baseline_weights: PathBuf,
    #[arg(long)]
    candidate_weights: PathBuf,
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    teacher_weights: PathBuf,
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long, default_value_t = 5)]
    teacher_depth: u8,
    #[arg(long, default_value_t = 3)]
    candidate_depth: u8,
    #[arg(long)]
    baseline_depth: Option<u8>,
    #[arg(long)]
    max_positions: Option<usize>,
    #[arg(long, default_value_t = false)]
    dedupe_sfen: bool,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 0)]
    jobs: usize,
    #[arg(long, default_value = "50,100,200,300")]
    bad_regret_thresholds_cp: String,
    #[arg(long, default_value_t = 0.0)]
    allow_mean_regret_increase_cp: f32,
    #[arg(long, default_value_t = 0.0)]
    allow_p90_regret_increase_cp: f32,
    #[arg(long, default_value_t = 0.0)]
    allow_p95_regret_increase_cp: f32,
    #[arg(long, default_value_t = 0.0)]
    allow_bad_ratio_increase: f32,
    #[arg(long, default_value_t = 0.0)]
    require_mean_regret_improvement_cp: f32,
    #[arg(long, default_value_t = 0.0)]
    require_p90_regret_improvement_cp: f32,
    #[arg(long, default_value_t = 0.0)]
    require_p95_regret_improvement_cp: f32,
    #[arg(long, default_value_t = 0.0)]
    require_match_rate_improvement_pct: f32,
    #[arg(long)]
    require_bad_regret_improvement: Vec<String>,
    #[arg(long)]
    json_output: Option<PathBuf>,
    #[arg(long, default_value_t = 20)]
    print_worst: usize,
    #[arg(long, default_value_t = 1000)]
    hard_position_limit: usize,
}

#[derive(Deserialize)]
struct SfenRecord {
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

#[derive(Clone)]
struct PositionRecord {
    sfen: String,
    position: Position,
}

#[derive(Clone, Serialize)]
struct ProbeResult {
    index: usize,
    sfen: String,
    teacher_best_move: Option<String>,
    selected_move: Option<String>,
    teacher_score: f32,
    selected_score: f32,
    regret: f32,
    exact_match: bool,
    legal_moves: usize,
}

#[derive(Clone, Serialize)]
struct ComparisonRecord {
    index: usize,
    sfen: String,
    teacher_best_move: Option<String>,
    baseline_move: Option<String>,
    candidate_move: Option<String>,
    baseline_regret: f32,
    candidate_regret: f32,
    regret_delta: f32,
    baseline_score: f32,
    candidate_score: f32,
    teacher_score: f32,
    legal_moves: usize,
}

#[derive(Clone, Serialize)]
struct RegretSummary {
    samples: usize,
    mean_regret_cp: f32,
    p50_regret_cp: f32,
    p90_regret_cp: f32,
    p95_regret_cp: f32,
    max_regret_cp: f32,
    exact_match_ratio: f32,
    bad_ratios: BTreeMap<String, f32>,
}

#[derive(Serialize)]
struct GateReport {
    baseline_weights: String,
    candidate_weights: String,
    teacher_weights: String,
    teacher_depth: u8,
    baseline_depth: u8,
    candidate_depth: u8,
    seed: u64,
    max_positions: Option<usize>,
    dedupe_sfen: bool,
    thresholds_cp: Vec<f32>,
    require_mean_regret_improvement_cp: f32,
    require_p90_regret_improvement_cp: f32,
    require_p95_regret_improvement_cp: f32,
    require_match_rate_improvement_pct: f32,
    require_bad_regret_improvement: Vec<String>,
    baseline: RegretSummary,
    candidate: RegretSummary,
    passed: bool,
    fail_reasons: Vec<String>,
    hard_positions: Vec<ComparisonRecord>,
    worst_candidate: Vec<ComparisonRecord>,
    worst_delta: Vec<ComparisonRecord>,
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

fn parse_thresholds(spec: &str) -> Result<Vec<f32>> {
    let mut thresholds = Vec::new();
    for part in spec.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = trimmed
            .parse::<f32>()
            .map_err(|e| anyhow!("invalid bad regret threshold `{trimmed}`: {e}"))?;
        if !value.is_finite() || value < 0.0 {
            return Err(anyhow!(
                "bad regret thresholds must be finite and non-negative: {value}"
            ));
        }
        thresholds.push(value);
    }
    if thresholds.is_empty() {
        return Err(anyhow!("--bad-regret-thresholds-cp must not be empty"));
    }
    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    thresholds.dedup_by(|a, b| (*a - *b).abs() < f32::EPSILON);
    Ok(thresholds)
}

fn bad_ratio_key(threshold: f32) -> String {
    format!("{threshold:.0}")
}

fn parse_bad_regret_improvements(specs: &[String], thresholds: &[f32]) -> Result<Vec<(f32, f32)>> {
    let mut requirements = Vec::new();
    for spec in specs {
        let (threshold_text, improvement_text) = spec.split_once(':').ok_or_else(|| {
            anyhow!("--require-bad-regret-improvement must be threshold:improvement, got `{spec}`")
        })?;
        let threshold = threshold_text.parse::<f32>().map_err(|error| {
            anyhow!("invalid threshold in --require-bad-regret-improvement `{spec}`: {error}")
        })?;
        let improvement = improvement_text.parse::<f32>().map_err(|error| {
            anyhow!("invalid improvement in --require-bad-regret-improvement `{spec}`: {error}")
        })?;
        if !threshold.is_finite() || threshold < 0.0 {
            return Err(anyhow!(
                "bad-regret threshold in --require-bad-regret-improvement must be finite and non-negative: {spec}"
            ));
        }
        if !improvement.is_finite() || improvement < 0.0 {
            return Err(anyhow!(
                "bad-regret improvement in --require-bad-regret-improvement must be finite and non-negative: {spec}"
            ));
        }
        if !thresholds
            .iter()
            .any(|value| (value - threshold).abs() < f32::EPSILON)
        {
            return Err(anyhow!(
                "--require-bad-regret-improvement references unknown threshold {threshold} (not in --bad-regret-thresholds-cp)"
            ));
        }
        requirements.push((threshold, improvement));
    }
    requirements.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    requirements.dedup_by(|a, b| (a.0 - b.0).abs() < f32::EPSILON);
    Ok(requirements)
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn percentile(mut values: Vec<f32>, p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let idx = ((values.len() - 1) as f32 * p).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn load_model(path: &Path) -> Result<SparseModel> {
    let mut model = SparseModel::new(0.0, 0.0);
    model
        .load(path)
        .map_err(|e| anyhow!("failed to load {}: {}", path.display(), e))?;
    Ok(model)
}

fn load_positions(
    paths: &[PathBuf],
    max_positions: Option<usize>,
    dedupe_sfen: bool,
    seed: u64,
) -> Result<Vec<PositionRecord>> {
    let mut records = Vec::new();
    let mut invalid = 0usize;
    let mut duplicate = 0usize;
    let mut seen = HashSet::new();
    for path in paths {
        let content = fs::read_to_string(path)
            .map_err(|e| anyhow!("failed to read {}: {}", path.display(), e))?;
        for (line_index, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let sfen = if line.starts_with('{') {
                let record: SfenRecord = serde_json::from_str(line).map_err(|e| {
                    anyhow!("{}:{} invalid json: {}", path.display(), line_index + 1, e)
                })?;
                record.sfen
            } else {
                line.to_string()
            };
            if let Some(position) = position_from_sfen_or_usi(&sfen) {
                let canonical_sfen = position.to_sfen_owned();
                if dedupe_sfen && !seen.insert(canonical_sfen.clone()) {
                    duplicate += 1;
                    continue;
                }
                records.push(PositionRecord {
                    sfen: canonical_sfen,
                    position,
                });
            } else {
                invalid += 1;
            }
        }
    }
    if records.is_empty() {
        return Err(anyhow!("no valid positions loaded (invalid={invalid})"));
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    records.shuffle(&mut rng);
    if let Some(max_positions) = max_positions {
        if max_positions == 0 {
            return Err(anyhow!("--max-positions must be greater than zero"));
        }
        records.truncate(max_positions);
    }
    println!(
        "positions: {} (invalid={invalid} duplicate={duplicate})",
        records.len()
    );
    Ok(records)
}

fn searched_root(model: &SparseModel, position: &Position, depth: u8) -> Option<(f32, Vec<Move>)> {
    let evaluator = SharedModelEvaluator { model };
    let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
    ai.set_emit_info(false);
    let mut root = position.clone();
    ai.sennichite_detector.record_position(&root);
    ai.alpha_beta_search(&mut root, depth, -f32::INFINITY, f32::INFINITY)
        .map(|(score, pv)| (sanitize_score(score), pv))
}

fn searched_move_score(
    model: &SparseModel,
    position: &Position,
    mv: Move,
    depth: u8,
) -> Option<f32> {
    let mut child = position.clone();
    child.do_move(mv);
    let evaluator = SharedModelEvaluator { model };
    let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
    ai.set_emit_info(false);
    ai.sennichite_detector.record_position(&child);
    let child_depth = depth.saturating_sub(1);
    ai.alpha_beta_search(&mut child, child_depth, -f32::INFINITY, f32::INFINITY)
        .map(|(score, _)| sanitize_score(-score))
}

fn probe_position(
    teacher: &SparseModel,
    selector: &SparseModel,
    record: &PositionRecord,
    index: usize,
    teacher_depth: u8,
    selector_depth: u8,
) -> Option<ProbeResult> {
    let legal_moves = record.position.legal_moves();
    if legal_moves.is_empty() {
        return None;
    }
    let (teacher_score, teacher_pv) = searched_root(teacher, &record.position, teacher_depth)?;
    let (_, selector_pv) = searched_root(selector, &record.position, selector_depth)?;
    let selected_move = selector_pv.first().copied();
    let selected_score = if let Some(mv) = selected_move {
        searched_move_score(teacher, &record.position, mv, teacher_depth)?
    } else {
        teacher_score
    };
    let teacher_best_move = teacher_pv.first().copied().map(format_move_usi);
    let selected_move_text = selected_move.map(format_move_usi);
    Some(ProbeResult {
        index,
        sfen: record.sfen.clone(),
        teacher_best_move: teacher_best_move.clone(),
        selected_move: selected_move_text.clone(),
        teacher_score,
        selected_score,
        regret: (teacher_score - selected_score).max(0.0),
        exact_match: teacher_best_move == selected_move_text,
        legal_moves: legal_moves.len(),
    })
}

fn summarize(results: &[ProbeResult], thresholds: &[f32]) -> RegretSummary {
    let regrets = results
        .iter()
        .map(|result| result.regret)
        .collect::<Vec<_>>();
    let exact_matches = results.iter().filter(|result| result.exact_match).count();
    let mut bad_ratios = BTreeMap::new();
    for threshold in thresholds {
        let count = regrets
            .iter()
            .filter(|&&regret| regret > *threshold)
            .count();
        bad_ratios.insert(
            bad_ratio_key(*threshold),
            count as f32 / results.len().max(1) as f32,
        );
    }
    RegretSummary {
        samples: results.len(),
        mean_regret_cp: mean(&regrets),
        p50_regret_cp: percentile(regrets.clone(), 0.50),
        p90_regret_cp: percentile(regrets.clone(), 0.90),
        p95_regret_cp: percentile(regrets.clone(), 0.95),
        max_regret_cp: regrets
            .iter()
            .copied()
            .fold(0.0_f32, |acc, regret| acc.max(regret)),
        exact_match_ratio: exact_matches as f32 / results.len().max(1) as f32,
        bad_ratios,
    }
}

fn create_writer(path: &Path) -> Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(BufWriter::new(File::create(path)?))
}

fn print_summary(label: &str, summary: &RegretSummary) {
    println!(
        "{label}: samples={} mean={:.2} p50={:.2} p90={:.2} p95={:.2} max={:.2} match={:.2}%",
        summary.samples,
        summary.mean_regret_cp,
        summary.p50_regret_cp,
        summary.p90_regret_cp,
        summary.p95_regret_cp,
        summary.max_regret_cp,
        summary.exact_match_ratio * 100.0
    );
    let bad = summary
        .bad_ratios
        .iter()
        .map(|(threshold, ratio)| format!("bad{threshold}={:.4}", ratio))
        .collect::<Vec<_>>()
        .join(" ");
    println!("{label}: {bad}");
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.teacher_depth == 0 {
        return Err(anyhow!("--teacher-depth must be greater than zero"));
    }
    if args.candidate_depth == 0 {
        return Err(anyhow!("--candidate-depth must be greater than zero"));
    }
    let baseline_depth = args.baseline_depth.unwrap_or(args.candidate_depth);
    if baseline_depth == 0 {
        return Err(anyhow!("--baseline-depth must be greater than zero"));
    }
    for (name, value) in [
        (
            "--allow-mean-regret-increase-cp",
            args.allow_mean_regret_increase_cp,
        ),
        (
            "--allow-p90-regret-increase-cp",
            args.allow_p90_regret_increase_cp,
        ),
        (
            "--allow-p95-regret-increase-cp",
            args.allow_p95_regret_increase_cp,
        ),
        ("--allow-bad-ratio-increase", args.allow_bad_ratio_increase),
        (
            "--require-mean-regret-improvement-cp",
            args.require_mean_regret_improvement_cp,
        ),
        (
            "--require-p90-regret-improvement-cp",
            args.require_p90_regret_improvement_cp,
        ),
        (
            "--require-p95-regret-improvement-cp",
            args.require_p95_regret_improvement_cp,
        ),
        (
            "--require-match-rate-improvement-pct",
            args.require_match_rate_improvement_pct,
        ),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(anyhow!("{name} must be finite and non-negative"));
        }
    }
    let thresholds = parse_thresholds(&args.bad_regret_thresholds_cp)?;
    let bad_regret_improvements =
        parse_bad_regret_improvements(&args.require_bad_regret_improvement, &thresholds)?;
    let hard_threshold = thresholds.first().copied().unwrap_or(0.0);
    if args.jobs > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build_global()
            .map_err(|e| anyhow!("failed to configure rayon thread pool: {e}"))?;
    }

    let baseline = load_model(&args.baseline_weights)?;
    let candidate = load_model(&args.candidate_weights)?;
    let teacher = load_model(&args.teacher_weights)?;
    let positions = load_positions(&args.input, args.max_positions, args.dedupe_sfen, args.seed)?;

    let mut paired = positions
        .par_iter()
        .enumerate()
        .filter_map(|(idx, record)| {
            let baseline_result = probe_position(
                &teacher,
                &baseline,
                record,
                idx,
                args.teacher_depth,
                baseline_depth,
            )?;
            let candidate_result = probe_position(
                &teacher,
                &candidate,
                record,
                idx,
                args.teacher_depth,
                args.candidate_depth,
            )?;
            Some((baseline_result, candidate_result))
        })
        .collect::<Vec<_>>();
    paired.sort_by_key(|(baseline, _)| baseline.index);

    if paired.is_empty() {
        return Err(anyhow!("no positions could be probed"));
    }

    let baseline_results = paired
        .iter()
        .map(|(baseline, _)| baseline.clone())
        .collect::<Vec<_>>();
    let candidate_results = paired
        .iter()
        .map(|(_, candidate)| candidate.clone())
        .collect::<Vec<_>>();
    let baseline_summary = summarize(&baseline_results, &thresholds);
    let candidate_summary = summarize(&candidate_results, &thresholds);

    print_summary("baseline", &baseline_summary);
    print_summary("candidate", &candidate_summary);

    let mut fail_reasons = Vec::new();
    if candidate_summary.mean_regret_cp
        > baseline_summary.mean_regret_cp + args.allow_mean_regret_increase_cp
    {
        fail_reasons.push(format!(
            "mean regret worsened: {:.2} > {:.2} + {:.2}",
            candidate_summary.mean_regret_cp,
            baseline_summary.mean_regret_cp,
            args.allow_mean_regret_increase_cp
        ));
    }
    if args.require_mean_regret_improvement_cp > 0.0
        && candidate_summary.mean_regret_cp
            > baseline_summary.mean_regret_cp - args.require_mean_regret_improvement_cp
    {
        fail_reasons.push(format!(
            "mean regret failed improvement requirement: {:.2} > {:.2} - {:.2}",
            candidate_summary.mean_regret_cp,
            baseline_summary.mean_regret_cp,
            args.require_mean_regret_improvement_cp
        ));
    }
    if candidate_summary.p90_regret_cp
        > baseline_summary.p90_regret_cp + args.allow_p90_regret_increase_cp
    {
        fail_reasons.push(format!(
            "p90 regret worsened: {:.2} > {:.2} + {:.2}",
            candidate_summary.p90_regret_cp,
            baseline_summary.p90_regret_cp,
            args.allow_p90_regret_increase_cp
        ));
    }
    if args.require_p90_regret_improvement_cp > 0.0
        && candidate_summary.p90_regret_cp
            > baseline_summary.p90_regret_cp - args.require_p90_regret_improvement_cp
    {
        fail_reasons.push(format!(
            "p90 regret failed improvement requirement: {:.2} > {:.2} - {:.2}",
            candidate_summary.p90_regret_cp,
            baseline_summary.p90_regret_cp,
            args.require_p90_regret_improvement_cp
        ));
    }
    if candidate_summary.p95_regret_cp
        > baseline_summary.p95_regret_cp + args.allow_p95_regret_increase_cp
    {
        fail_reasons.push(format!(
            "p95 regret worsened: {:.2} > {:.2} + {:.2}",
            candidate_summary.p95_regret_cp,
            baseline_summary.p95_regret_cp,
            args.allow_p95_regret_increase_cp
        ));
    }
    if args.require_p95_regret_improvement_cp > 0.0
        && candidate_summary.p95_regret_cp
            > baseline_summary.p95_regret_cp - args.require_p95_regret_improvement_cp
    {
        fail_reasons.push(format!(
            "p95 regret failed improvement requirement: {:.2} > {:.2} - {:.2}",
            candidate_summary.p95_regret_cp,
            baseline_summary.p95_regret_cp,
            args.require_p95_regret_improvement_cp
        ));
    }
    for threshold in &thresholds {
        let key = bad_ratio_key(*threshold);
        let baseline_ratio = baseline_summary
            .bad_ratios
            .get(&key)
            .copied()
            .unwrap_or(0.0);
        let candidate_ratio = candidate_summary
            .bad_ratios
            .get(&key)
            .copied()
            .unwrap_or(0.0);
        if candidate_ratio > baseline_ratio + args.allow_bad_ratio_increase {
            fail_reasons.push(format!(
                "bad{key} ratio worsened: {:.4} > {:.4} + {:.4}",
                candidate_ratio, baseline_ratio, args.allow_bad_ratio_increase
            ));
        }
    }
    for (threshold, required_improvement) in &bad_regret_improvements {
        let key = bad_ratio_key(*threshold);
        let baseline_ratio = baseline_summary
            .bad_ratios
            .get(&key)
            .copied()
            .unwrap_or(0.0);
        let candidate_ratio = candidate_summary
            .bad_ratios
            .get(&key)
            .copied()
            .unwrap_or(0.0);
        if candidate_ratio > baseline_ratio - required_improvement {
            fail_reasons.push(format!(
                "bad{key} ratio failed improvement requirement: {:.4} > {:.4} - {:.4}",
                candidate_ratio, baseline_ratio, required_improvement
            ));
        }
    }
    let baseline_match_rate = baseline_summary.exact_match_ratio * 100.0;
    let candidate_match_rate = candidate_summary.exact_match_ratio * 100.0;
    if candidate_match_rate < baseline_match_rate + args.require_match_rate_improvement_pct {
        fail_reasons.push(format!(
            "match rate failed improvement requirement: {:.2}% < {:.2}% + {:.2}%",
            candidate_match_rate, baseline_match_rate, args.require_match_rate_improvement_pct
        ));
    }

    let comparisons = paired
        .iter()
        .map(|(baseline, candidate)| ComparisonRecord {
            index: baseline.index,
            sfen: baseline.sfen.clone(),
            teacher_best_move: baseline.teacher_best_move.clone(),
            baseline_move: baseline.selected_move.clone(),
            candidate_move: candidate.selected_move.clone(),
            baseline_regret: baseline.regret,
            candidate_regret: candidate.regret,
            regret_delta: candidate.regret - baseline.regret,
            baseline_score: baseline.selected_score,
            candidate_score: candidate.selected_score,
            teacher_score: baseline.teacher_score,
            legal_moves: baseline.legal_moves,
        })
        .collect::<Vec<_>>();
    let mut hard_positions = comparisons
        .iter()
        .filter(|record| record.candidate_regret > hard_threshold || record.regret_delta > 0.0)
        .cloned()
        .collect::<Vec<_>>();
    hard_positions.sort_by(|a, b| {
        b.regret_delta
            .partial_cmp(&a.regret_delta)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                b.candidate_regret
                    .partial_cmp(&a.candidate_regret)
                    .unwrap_or(Ordering::Equal)
            })
    });
    hard_positions.truncate(args.hard_position_limit.min(hard_positions.len()));

    let mut worst_candidate = comparisons.clone();
    worst_candidate.sort_by(|a, b| {
        b.candidate_regret
            .partial_cmp(&a.candidate_regret)
            .unwrap_or(Ordering::Equal)
    });
    worst_candidate.truncate(args.print_worst.min(worst_candidate.len()));
    let mut worst_delta = comparisons;
    worst_delta.sort_by(|a, b| {
        b.regret_delta
            .partial_cmp(&a.regret_delta)
            .unwrap_or(Ordering::Equal)
    });
    worst_delta.truncate(args.print_worst.min(worst_delta.len()));

    for (idx, record) in worst_delta.iter().enumerate() {
        println!(
            "worst_delta[{}] delta={:.2} baseline={:.2} candidate={:.2} teacher={} baseline_move={} candidate_move={} sfen={}",
            idx + 1,
            record.regret_delta,
            record.baseline_regret,
            record.candidate_regret,
            record.teacher_best_move.as_deref().unwrap_or("none"),
            record.baseline_move.as_deref().unwrap_or("none"),
            record.candidate_move.as_deref().unwrap_or("none"),
            record.sfen
        );
    }

    let passed = fail_reasons.is_empty();
    if let Some(path) = args.json_output.as_ref() {
        let report = GateReport {
            baseline_weights: args.baseline_weights.display().to_string(),
            candidate_weights: args.candidate_weights.display().to_string(),
            teacher_weights: args.teacher_weights.display().to_string(),
            teacher_depth: args.teacher_depth,
            baseline_depth,
            candidate_depth: args.candidate_depth,
            seed: args.seed,
            max_positions: args.max_positions,
            dedupe_sfen: args.dedupe_sfen,
            require_mean_regret_improvement_cp: args.require_mean_regret_improvement_cp,
            require_p90_regret_improvement_cp: args.require_p90_regret_improvement_cp,
            require_p95_regret_improvement_cp: args.require_p95_regret_improvement_cp,
            require_match_rate_improvement_pct: args.require_match_rate_improvement_pct,
            require_bad_regret_improvement: args.require_bad_regret_improvement,
            thresholds_cp: thresholds,
            baseline: baseline_summary,
            candidate: candidate_summary,
            passed,
            fail_reasons: fail_reasons.clone(),
            hard_positions,
            worst_candidate,
            worst_delta,
        };
        let mut writer = create_writer(path)?;
        serde_json::to_writer_pretty(&mut writer, &report)?;
        writeln!(writer)?;
        println!("json-output: {}", path.display());
    }

    if passed {
        println!("RERANK GATE PASSED");
        Ok(())
    } else {
        println!("RERANK GATE FAILED: {:?}", fail_reasons);
        process::exit(2);
    }
}
