use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use shogi_ai::utils::{parse_usi_move, position_from_sfen_or_usi};
use shogi_core::{Move, Piece};
use shogi_lib::Position;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Filter depth4 rank JSONL by matching depth3 stability checks")]
struct Args {
    #[arg(long, required = true)]
    depth3: Vec<PathBuf>,
    #[arg(long, required = true)]
    depth4: Vec<PathBuf>,
    #[arg(long)]
    output_stable: PathBuf,
    #[arg(long)]
    output_unstable: PathBuf,
    #[arg(long)]
    stats_output: PathBuf,
    #[arg(long, value_enum, default_value_t = KeyMode::Canonical)]
    key_mode: KeyMode,
    #[arg(long, default_value_t = false)]
    keep_duplicates: bool,
    #[arg(long, default_value_t = true)]
    require_best_match: bool,
    #[arg(long, default_value_t = 0)]
    max_d4_best_rank_in_d3: usize,
    #[arg(long, default_value_t = 0.0)]
    max_d3_best_regret_in_d4_cp: f32,
    #[arg(long, default_value_t = 15.0)]
    min_d4_gap_cp: f32,
    #[arg(long, default_value_t = 5.0)]
    min_d3_gap_cp: f32,
    #[arg(long, default_value_t = 8)]
    span_top_k: usize,
    #[arg(long, default_value_t = 50.0)]
    min_d4_topk_span_cp: f32,
    #[arg(long, default_value_t = 200.0)]
    max_root_delta_cp: f32,
    #[arg(long, default_value_t = 8)]
    min_common_candidates: usize,
    #[arg(long, default_value_t = 8)]
    pairwise_top_k: usize,
    #[arg(long, default_value_t = 10.0)]
    pairwise_gap_cp: f32,
    #[arg(long, default_value_t = 0.70)]
    min_pairwise_agreement: f32,
    #[arg(long, default_value_t = 3000.0)]
    max_abs_root_score: f32,
    #[arg(long, default_value_t = 2)]
    min_legal_moves: usize,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum KeyMode {
    #[value(name = "canonical")]
    Canonical,
    #[value(name = "full-sfen")]
    FullSfen,
}

#[derive(Debug, Deserialize)]
struct KppRankRecord {
    sfen: String,
    teacher_depth: u8,
    root_score: f32,
    legal_moves: usize,
    #[serde(default)]
    candidates: Vec<KppRankCandidate>,
    teacher_weights: String,
}

#[derive(Debug, Deserialize)]
struct KppRankCandidate {
    #[serde(rename = "move")]
    move_usi: String,
    teacher_score: f32,
}

#[derive(Clone)]
struct ParsedCandidate {
    move_usi: String,
    teacher_score: f32,
}

#[derive(Clone)]
struct DepthRecord {
    root_score: f32,
    legal_moves: usize,
    teacher_weights: String,
    candidates: Vec<ParsedCandidate>,
    canonical_legal_moves: usize,
}

#[derive(Default)]
struct FilterStats {
    d3_records: usize,
    d4_records: usize,
    matched: usize,
    stable: usize,
    unstable: usize,
    missing_d3: usize,
    duplicates_skipped: usize,
    rejects: BTreeMap<String, usize>,
    d4_best_rank_in_d3: BTreeMap<String, usize>,
    root_delta_values: Vec<f32>,
    d3_gap_values: Vec<f32>,
    d4_gap_values: Vec<f32>,
    d3_best_regret_in_d4_values: Vec<f32>,
    pairwise_agreement_values: Vec<f32>,
    candidate_count_values: Vec<f32>,
    legal_moves_values: Vec<f32>,
}

#[derive(Serialize)]
struct RejectStats {
    #[serde(rename = "missing_d3")]
    missing_d3: usize,
    #[serde(rename = "invalid_depth")]
    invalid_depth: usize,
    #[serde(rename = "weights_mismatch")]
    weights_mismatch: usize,
    #[serde(rename = "legal_mismatch")]
    legal_mismatch: usize,
    #[serde(rename = "few_candidates")]
    few_candidates: usize,
    #[serde(rename = "root_score_out_of_range")]
    root_score_out_of_range: usize,
    #[serde(rename = "root_delta_high")]
    root_delta_high: usize,
    #[serde(rename = "d4_gap_low")]
    d4_gap_low: usize,
    #[serde(rename = "d3_gap_low")]
    d3_gap_low: usize,
    #[serde(rename = "topk_span_low")]
    topk_span_low: usize,
    #[serde(rename = "common_candidates_low")]
    common_candidates_low: usize,
    #[serde(rename = "best_mismatch")]
    best_mismatch: usize,
    #[serde(rename = "pairwise_no_pairs")]
    pairwise_no_pairs: usize,
    #[serde(rename = "pairwise_low")]
    pairwise_low: usize,
}

#[derive(Serialize)]
struct DistributionSummary {
    mean: f32,
    p50: f32,
    p90: Option<f32>,
    p95: Option<f32>,
}

#[derive(Serialize)]
struct PairwiseDistributionSummary {
    mean: f32,
    p50: f32,
}

#[derive(Serialize)]
struct Histograms {
    d4_best_rank_in_d3: BTreeMap<String, usize>,
}

#[derive(Serialize)]
struct DistributionReport {
    root_delta: DistributionSummary,
    d3_gap: DistributionSummary,
    d4_gap: DistributionSummary,
    d3_best_regret_in_d4: DistributionSummary,
    pairwise_agreement: PairwiseDistributionSummary,
    candidate_count: DistributionSummary,
    legal_moves: DistributionSummary,
}

#[derive(Serialize)]
struct StatsOutput {
    d3_records: usize,
    d4_records: usize,
    matched: usize,
    stable: usize,
    unstable: usize,
    missing_d3: usize,
    duplicates_skipped: usize,
    rejects: RejectStats,
    histograms: Histograms,
    distributions: DistributionReport,
}

#[derive(Clone, Copy, Debug)]
enum RejectReason {
    MissingD3,
    InvalidDepth,
    WeightsMismatch,
    LegalMismatch,
    FewCandidates,
    RootScoreOutOfRange,
    RootDeltaHigh,
    D4GapLow,
    D3GapLow,
    TopkSpanLow,
    CommonCandidatesLow,
    BestMismatch,
    PairwiseNoPairs,
    PairwiseLow,
}

fn reject_reason_name(reason: RejectReason) -> &'static str {
    match reason {
        RejectReason::MissingD3 => "missing_d3",
        RejectReason::InvalidDepth => "invalid_depth",
        RejectReason::WeightsMismatch => "weights_mismatch",
        RejectReason::LegalMismatch => "legal_mismatch",
        RejectReason::FewCandidates => "few_candidates",
        RejectReason::RootScoreOutOfRange => "root_score_out_of_range",
        RejectReason::RootDeltaHigh => "root_delta_high",
        RejectReason::D4GapLow => "d4_gap_low",
        RejectReason::D3GapLow => "d3_gap_low",
        RejectReason::TopkSpanLow => "topk_span_low",
        RejectReason::CommonCandidatesLow => "common_candidates_low",
        RejectReason::BestMismatch => "best_mismatch",
        RejectReason::PairwiseNoPairs => "pairwise_no_pairs",
        RejectReason::PairwiseLow => "pairwise_low",
    }
}

fn canonical_key(sfen: &str, key_mode: KeyMode) -> String {
    let sfen = sfen.trim().strip_prefix("sfen ").unwrap_or(sfen).trim();
    match key_mode {
        KeyMode::FullSfen => sfen.to_string(),
        KeyMode::Canonical => {
            let parts: Vec<_> = sfen.split_whitespace().collect();
            if parts.len() >= 3 {
                format!("{} {} {}", parts[0], parts[1], parts[2])
            } else {
                sfen.to_string()
            }
        }
    }
}

fn create_writer(path: &Path) -> Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(BufWriter::new(File::create(path)?))
}

fn sanitize_float(value: f32) -> bool {
    value.is_finite()
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

fn parse_candidates(record: &KppRankRecord, position: &Position) -> (Vec<ParsedCandidate>, bool) {
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();
    let mut legal = true;
    for candidate in &record.candidates {
        let move_usi = candidate.move_usi.trim().to_string();
        if move_usi.is_empty() || !candidate.teacher_score.is_finite() {
            continue;
        }
        let Some(move_) = parse_move_for_position(position, &move_usi) else {
            legal = false;
            continue;
        };
        if !position.legal_moves().contains(&move_) {
            legal = false;
            continue;
        }
        if seen.insert(move_usi.clone()) {
            candidates.push(ParsedCandidate {
                move_usi,
                teacher_score: candidate.teacher_score,
            });
        }
    }
    (candidates, legal)
}

fn parse_position(record: &KppRankRecord) -> Option<Position> {
    position_from_sfen_or_usi(&record.sfen)
}

fn parse_depth_records(
    paths: &[PathBuf],
    key_mode: KeyMode,
    keep_duplicates: bool,
    mut stats: &mut FilterStats,
) -> Result<HashMap<String, Vec<DepthRecord>>> {
    let mut lookup: HashMap<String, Vec<DepthRecord>> = HashMap::new();
    for path in paths {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        for (line_no, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let record: KppRankRecord = match serde_json::from_str(&line) {
                Ok(record) => record,
                Err(err) => {
                    eprintln!("{}:{} invalid json: {}", path.display(), line_no + 1, err);
                    continue;
                }
            };
            if record.teacher_depth != 3 {
                eprintln!(
                    "{}:{} invalid depth: expected 3, got {}",
                    path.display(),
                    line_no + 1,
                    record.teacher_depth
                );
                push_reject(&mut stats, RejectReason::InvalidDepth);
                continue;
            }

            let key = canonical_key(&record.sfen, key_mode);
            let position = match parse_position(&record) {
                Some(position) => position,
                None => {
                    eprintln!("{}:{} invalid sfen", path.display(), line_no + 1);
                    continue;
                }
            };
            let canonical_legal_moves = position.legal_moves().len();
            let (candidates, legal_candidates) = parse_candidates(&record, &position);
            if !legal_candidates {
                eprintln!(
                    "warning: {}:{} legal candidate mismatch in depth3 record",
                    path.display(),
                    line_no + 1
                );
                push_reject(&mut stats, RejectReason::LegalMismatch);
                continue;
            }

            let entry = DepthRecord {
                root_score: record.root_score,
                legal_moves: record.legal_moves,
                teacher_weights: record.teacher_weights.clone(),
                candidates,
                canonical_legal_moves,
            };
            if let Some(existing) = lookup.get_mut(&key) {
                if keep_duplicates {
                    existing.push(entry);
                    stats.d3_records += 1;
                } else {
                    stats.duplicates_skipped += 1;
                }
                continue;
            }
            lookup.insert(key, vec![entry]);
            stats.d3_records += 1;
        }
    }
    Ok(lookup)
}

fn percentiles(mut values: Vec<f32>, percentiles: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return vec![0.0; percentiles.len()];
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let max_idx = values.len() - 1;
    percentiles
        .iter()
        .map(|&p| {
            let idx = ((max_idx as f32) * p).round() as usize;
            values[idx.min(max_idx)]
        })
        .collect()
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn pairwise_top_agreement(
    d4: &[ParsedCandidate],
    d3: &[ParsedCandidate],
    top_k: usize,
    gap_cp: f32,
) -> (usize, usize) {
    if d4.is_empty() || d3.is_empty() || top_k < 2 {
        return (0, 0);
    }

    let d3_rank: HashMap<&str, usize> = d3
        .iter()
        .enumerate()
        .map(|(idx, candidate)| (candidate.move_usi.as_str(), idx))
        .collect();

    let top_k = top_k.min(d4.len());
    let mut total_pairs = 0usize;
    let mut correct_pairs = 0usize;

    for i in 0..top_k {
        for j in (i + 1)..top_k {
            let score_diff = d4[i].teacher_score - d4[j].teacher_score;
            if score_diff < gap_cp {
                continue;
            }
            let left = &d4[i].move_usi;
            let right = &d4[j].move_usi;
            let Some(&left_rank) = d3_rank.get(left.as_str()) else {
                continue;
            };
            let Some(&right_rank) = d3_rank.get(right.as_str()) else {
                continue;
            };
            total_pairs += 1;
            if left_rank < right_rank {
                correct_pairs += 1;
            }
        }
    }

    (correct_pairs, total_pairs)
}

fn push_reject(stats: &mut FilterStats, reason: RejectReason) {
    let key = reject_reason_name(reason).to_string();
    *stats.rejects.entry(key).or_insert(0) += 1;
}

fn reject_stats_from_map(map: &BTreeMap<String, usize>) -> RejectStats {
    let get = |name: &str| map.get(name).copied().unwrap_or(0);
    RejectStats {
        missing_d3: get("missing_d3"),
        invalid_depth: get("invalid_depth"),
        weights_mismatch: get("weights_mismatch"),
        legal_mismatch: get("legal_mismatch"),
        few_candidates: get("few_candidates"),
        root_score_out_of_range: get("root_score_out_of_range"),
        root_delta_high: get("root_delta_high"),
        d4_gap_low: get("d4_gap_low"),
        d3_gap_low: get("d3_gap_low"),
        topk_span_low: get("topk_span_low"),
        common_candidates_low: get("common_candidates_low"),
        best_mismatch: get("best_mismatch"),
        pairwise_no_pairs: get("pairwise_no_pairs"),
        pairwise_low: get("pairwise_low"),
    }
}

fn validate_args(args: &Args) -> Result<()> {
    if args.min_d4_gap_cp.is_sign_negative() {
        return Err(anyhow!("--min-d4-gap-cp must be >= 0"));
    }
    if args.min_d3_gap_cp.is_sign_negative() {
        return Err(anyhow!("--min-d3-gap-cp must be >= 0"));
    }
    if args.span_top_k == 0 {
        return Err(anyhow!("--span-top-k must be > 0"));
    }
    if args.min_d4_topk_span_cp.is_sign_negative() {
        return Err(anyhow!("--min-d4-topk-span-cp must be >= 0"));
    }
    if args.max_root_delta_cp.is_sign_negative() {
        return Err(anyhow!("--max-root-delta-cp must be >= 0"));
    }
    if args.pairwise_top_k == 0 {
        return Err(anyhow!("--pairwise-top-k must be > 0"));
    }
    if args.pairwise_gap_cp.is_sign_negative() {
        return Err(anyhow!("--pairwise-gap-cp must be >= 0"));
    }
    if !(0.0..=1.0).contains(&args.min_pairwise_agreement) {
        return Err(anyhow!("--min-pairwise-agreement must be between 0 and 1"));
    }
    if args.max_abs_root_score.is_sign_negative() {
        return Err(anyhow!("--max-abs-root-score must be >= 0"));
    }
    if args.max_d3_best_regret_in_d4_cp.is_sign_negative() {
        return Err(anyhow!("--max-d3-best-regret-in-d4-cp must be >= 0"));
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;

    let mut stats = FilterStats::default();
    let d3_lookup = parse_depth_records(
        &args.depth3,
        args.key_mode,
        args.keep_duplicates,
        &mut stats,
    )?;

    let mut stable_writer = create_writer(&args.output_stable)?;
    let mut unstable_writer = create_writer(&args.output_unstable)?;
    let mut matched_index: HashMap<String, usize> = HashMap::new();
    let mut seen_d4_keys: HashSet<String> = HashSet::new();

    for path in &args.depth4 {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        for (line_no, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let record: KppRankRecord = match serde_json::from_str(&line) {
                Ok(record) => record,
                Err(err) => {
                    eprintln!("{}:{} invalid json: {}", path.display(), line_no + 1, err);
                    continue;
                }
            };

            let d4_position = match parse_position(&record) {
                Some(value) => value,
                None => {
                    eprintln!("{}:{} invalid sfen", path.display(), line_no + 1);
                    continue;
                }
            };

            let key = canonical_key(&record.sfen, args.key_mode);
            if !args.keep_duplicates {
                if !seen_d4_keys.insert(key.clone()) {
                    stats.duplicates_skipped += 1;
                    continue;
                }
            }

            if record.teacher_depth != 4 {
                stats.d4_records += 1;
                push_reject(&mut stats, RejectReason::InvalidDepth);
                stats.unstable += 1;
                writeln!(unstable_writer, "{}", line)?;
                continue;
            }

            let d4_legal_moves = d4_position.legal_moves().len();
            let (candidates, legal_candidates) = parse_candidates(&record, &d4_position);
            if !legal_candidates {
                stats.d4_records += 1;
                push_reject(&mut stats, RejectReason::LegalMismatch);
                stats.unstable += 1;
                writeln!(unstable_writer, "{}", line)?;
                continue;
            }
            if candidates.len() < 2 {
                stats.d4_records += 1;
                push_reject(&mut stats, RejectReason::FewCandidates);
                stats.unstable += 1;
                writeln!(unstable_writer, "{}", line)?;
                continue;
            }
            let d4_legal_from_position = d4_legal_moves;
            let d4_gap = if candidates.len() >= 2 {
                (candidates[0].teacher_score - candidates[1].teacher_score).max(0.0)
            } else {
                0.0
            };

            if !sanitize_float(record.root_score) || !sanitize_float(d4_gap) {
                stats.d4_records += 1;
                push_reject(&mut stats, RejectReason::RootScoreOutOfRange);
                stats.unstable += 1;
                writeln!(unstable_writer, "{}", line)?;
                continue;
            }

            let top_span = if candidates.len() >= args.span_top_k && args.span_top_k > 1 {
                candidates[0].teacher_score - candidates[args.span_top_k - 1].teacher_score
            } else if candidates.len() >= 2 {
                candidates[0].teacher_score - candidates[candidates.len() - 1].teacher_score
            } else {
                0.0
            };

            stats.d4_records += 1;
            stats.candidate_count_values.push(candidates.len() as f32);
            stats.legal_moves_values.push(d4_legal_from_position as f32);

            let d4_match = match d3_lookup.get(&key) {
                Some(records) => {
                    let next = matched_index.entry(key.clone()).or_insert(0);
                    if *next < records.len() {
                        let matched = records[*next].clone();
                        *next += 1;
                        Some(matched)
                    } else {
                        None
                    }
                }
                None => None,
            };

            let reject_reason = match d4_match {
                Some(d3) => {
                    stats.matched += 1;
                    let d3_legal_from_position = d3.canonical_legal_moves;
                    let d4_set: HashSet<&str> =
                        candidates.iter().map(|c| c.move_usi.as_str()).collect();
                    if d3.legal_moves != d3_legal_from_position
                        || d3.legal_moves != record.legal_moves
                        || record.legal_moves != d4_legal_from_position
                        || d3_legal_from_position != d4_legal_from_position
                        || record.legal_moves < args.min_legal_moves
                        || d3_legal_from_position < args.min_legal_moves
                        || d4_legal_from_position < args.min_legal_moves
                    {
                        Some(RejectReason::LegalMismatch)
                    } else if d3.teacher_weights != record.teacher_weights {
                        Some(RejectReason::WeightsMismatch)
                    } else if d3.candidates.len() < 2 {
                        Some(RejectReason::FewCandidates)
                    } else if record.root_score.abs() > args.max_abs_root_score {
                        Some(RejectReason::RootScoreOutOfRange)
                    } else {
                        let d3_gap = (d3.candidates[0].teacher_score
                            - d3.candidates[1].teacher_score)
                            .max(0.0);
                        let d4_gap =
                            (candidates[0].teacher_score - candidates[1].teacher_score).max(0.0);

                        let root_delta = (record.root_score - d3.root_score).abs();
                        let d4_best_move = &candidates[0].move_usi;
                        let d3_best_move = &d3.candidates[0].move_usi;
                        let d3_best_in_d4 = d4_set.contains(d3_best_move.as_str());
                        let mut d4_best_rank_in_d3: Option<usize> = None;
                        for (rank, candidate) in d3.candidates.iter().enumerate() {
                            if candidate.move_usi == *d4_best_move {
                                d4_best_rank_in_d3 = Some(rank);
                                break;
                            }
                        }
                        let mut d3_best_regret_in_d4_cp: Option<f32> = None;
                        for candidate in candidates.iter() {
                            if candidate.move_usi == *d3_best_move {
                                d3_best_regret_in_d4_cp = Some(
                                    (candidates[0].teacher_score - candidate.teacher_score)
                                        .max(0.0),
                                );
                                break;
                            }
                        }
                        if let Some(best_rank) = d4_best_rank_in_d3 {
                            *stats
                                .d4_best_rank_in_d3
                                .entry(best_rank.to_string())
                                .or_insert(0) += 1;
                        } else {
                            *stats
                                .d4_best_rank_in_d3
                                .entry("missing".to_string())
                                .or_insert(0) += 1;
                        }

                        stats.root_delta_values.push(root_delta);
                        stats.d3_gap_values.push(d3_gap);
                        stats.d4_gap_values.push(d4_gap);

                        let common = d3
                            .candidates
                            .iter()
                            .filter(|c| d4_set.contains(c.move_usi.as_str()))
                            .count();
                        if let Some(regret) = d3_best_regret_in_d4_cp {
                            stats.d3_best_regret_in_d4_values.push(regret);
                        }

                        if common < args.min_common_candidates {
                            Some(RejectReason::CommonCandidatesLow)
                        } else if args.require_best_match && !d3_best_in_d4 {
                            Some(RejectReason::BestMismatch)
                        } else if args.require_best_match && d4_best_rank_in_d3.is_none() {
                            Some(RejectReason::BestMismatch)
                        } else if args.require_best_match
                            && d4_best_rank_in_d3.unwrap() > args.max_d4_best_rank_in_d3
                        {
                            Some(RejectReason::BestMismatch)
                        } else if args.require_best_match
                            && d3_best_regret_in_d4_cp > Some(args.max_d3_best_regret_in_d4_cp)
                        {
                            Some(RejectReason::BestMismatch)
                        } else if top_span < args.min_d4_topk_span_cp {
                            Some(RejectReason::TopkSpanLow)
                        } else if d3_gap < args.min_d3_gap_cp {
                            Some(RejectReason::D3GapLow)
                        } else if d4_gap < args.min_d4_gap_cp {
                            Some(RejectReason::D4GapLow)
                        } else if root_delta > args.max_root_delta_cp {
                            Some(RejectReason::RootDeltaHigh)
                        } else {
                            let (correct, total) = pairwise_top_agreement(
                                &candidates,
                                &d3.candidates,
                                args.pairwise_top_k,
                                args.pairwise_gap_cp,
                            );
                            if total == 0 {
                                Some(RejectReason::PairwiseNoPairs)
                            } else {
                                let agreement = correct as f32 / total as f32;
                                stats.pairwise_agreement_values.push(agreement);
                                if agreement < args.min_pairwise_agreement {
                                    Some(RejectReason::PairwiseLow)
                                } else {
                                    None
                                }
                            }
                        }
                    }
                }
                None => {
                    stats.missing_d3 += 1;
                    Some(RejectReason::MissingD3)
                }
            };

            match reject_reason {
                Some(reason) => {
                    stats.unstable += 1;
                    push_reject(&mut stats, reason);
                    writeln!(unstable_writer, "{}", line)?;
                }
                None => {
                    stats.stable += 1;
                    writeln!(stable_writer, "{}", line)?;
                }
            }
        }
    }

    if stats.d4_records != stats.stable + stats.unstable {
        eprintln!(
            "warning: unstable+stable mismatch ({} + {} != {})",
            stats.stable, stats.unstable, stats.d4_records
        );
    }

    let root_delta = percentiles(stats.root_delta_values.clone(), &[0.5, 0.9, 0.95]);
    let d3_gap = percentiles(stats.d3_gap_values.clone(), &[0.5, 0.9]);
    let d4_gap = percentiles(stats.d4_gap_values.clone(), &[0.5, 0.9]);
    let d3_best_regret_in_d4 =
        percentiles(stats.d3_best_regret_in_d4_values.clone(), &[0.5, 0.9, 0.95]);
    let pairwise = percentiles(stats.pairwise_agreement_values.clone(), &[0.5]);
    let pairwise_mean = mean(&stats.pairwise_agreement_values);
    let candidate_count = percentiles(stats.candidate_count_values.clone(), &[0.5, 0.9]);
    let legal_moves = percentiles(stats.legal_moves_values.clone(), &[0.5, 0.9]);

    let stats_output = StatsOutput {
        d3_records: stats.d3_records,
        d4_records: stats.d4_records,
        matched: stats.matched,
        stable: stats.stable,
        unstable: stats.unstable,
        missing_d3: stats.missing_d3,
        duplicates_skipped: stats.duplicates_skipped,
        rejects: reject_stats_from_map(&stats.rejects),
        histograms: Histograms {
            d4_best_rank_in_d3: stats.d4_best_rank_in_d3,
        },
        distributions: DistributionReport {
            root_delta: DistributionSummary {
                mean: mean(&stats.root_delta_values),
                p50: root_delta[0],
                p90: Some(root_delta[1]),
                p95: Some(root_delta[2]),
            },
            d3_gap: DistributionSummary {
                mean: mean(&stats.d3_gap_values),
                p50: d3_gap[0],
                p90: Some(d3_gap[1]),
                p95: None,
            },
            d4_gap: DistributionSummary {
                mean: mean(&stats.d4_gap_values),
                p50: d4_gap[0],
                p90: Some(d4_gap[1]),
                p95: None,
            },
            d3_best_regret_in_d4: DistributionSummary {
                mean: mean(&stats.d3_best_regret_in_d4_values),
                p50: d3_best_regret_in_d4[0],
                p90: Some(d3_best_regret_in_d4[1]),
                p95: Some(d3_best_regret_in_d4[2]),
            },
            pairwise_agreement: PairwiseDistributionSummary {
                mean: pairwise_mean,
                p50: pairwise[0],
            },
            candidate_count: DistributionSummary {
                mean: mean(&stats.candidate_count_values),
                p50: candidate_count[0],
                p90: Some(candidate_count[1]),
                p95: None,
            },
            legal_moves: DistributionSummary {
                mean: mean(&stats.legal_moves_values),
                p50: legal_moves[0],
                p90: Some(legal_moves[1]),
                p95: None,
            },
        },
    };

    let mut stats_writer = create_writer(&args.stats_output)?;
    serde_json::to_writer_pretty(&mut stats_writer, &stats_output)?;
    writeln!(stats_writer)?;

    Ok(())
}
