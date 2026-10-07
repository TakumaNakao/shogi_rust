use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use shogi_ai::ai::{SearchLimits, ShogiAI};
use shogi_ai::evaluation::EngineEvaluator;
use shogi_ai::sennichite::SennichiteStatus;
use shogi_ai::training_data::{sha256_file, sha256_hex};
use shogi_ai::utils::position_and_history_from_sfen_or_usi;
use shogi_core::{Color, PartialPosition};
use shogi_lib::Position;
use shogi_usi_parser::FromUsi;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const LINEAGE_ID: &str = "halfkp64-scratch-v1";
const HISTORY_CAPACITY: usize = 256;

#[derive(Parser, Debug)]
#[command(about = "Run a deterministic paired-opening HalfKP promotion match")]
struct Args {
    #[arg(long)]
    candidate: PathBuf,
    #[arg(long)]
    candidate_manifest: PathBuf,
    #[arg(long)]
    parent: PathBuf,
    #[arg(long)]
    parent_manifest: PathBuf,
    #[arg(long, conflicts_with = "parent_authorization")]
    registry: Option<PathBuf>,
    #[arg(long, conflicts_with = "registry")]
    parent_authorization: Option<PathBuf>,
    #[arg(long)]
    openings: PathBuf,
    #[arg(long, default_value = "dataset-jsonl")]
    openings_format: String,
    #[arg(long, default_value = "first-unique")]
    opening_selection: String,
    #[arg(long, default_value_t = 0)]
    opening_seed: u64,
    /// Number of opening pairs; the match contains twice this many games.
    #[arg(long)]
    pairs: usize,
    #[arg(long, conflicts_with = "time_limit_ms")]
    nodes: Option<u64>,
    #[arg(long, conflicts_with = "nodes")]
    time_limit_ms: Option<u64>,
    #[arg(long, default_value_t = 64)]
    max_depth: u8,
    #[arg(long, default_value_t = 256)]
    max_plies: usize,
    #[arg(long)]
    json: PathBuf,
    /// Run both colors with the same accepted model. This is an on-policy
    /// self-play smoke and is never promotion-eligible.
    #[arg(long, default_value_t = false)]
    self_play: bool,
}

#[derive(Deserialize)]
struct ModelManifest {
    lineage_id: String,
    model: ModelArtifact,
}
#[derive(Deserialize)]
struct ModelArtifact {
    sha256: String,
}
#[derive(Deserialize)]
struct Registry {
    lineage_id: String,
    accepted: Vec<RegistryEntry>,
}
#[derive(Deserialize)]
struct RegistryEntry {
    model_sha256: String,
}
#[derive(Deserialize)]
struct OpeningRecord {
    split: String,
    opening_family: String,
    sfen: String,
    ply: usize,
}

#[derive(Clone)]
struct Opening {
    family: String,
    ply: usize,
    position_spec: String,
}

#[derive(Default, Serialize)]
struct MatchCounts {
    candidate_wins: usize,
    parent_wins: usize,
    draws: usize,
    candidate_black_wins: usize,
    candidate_white_wins: usize,
    search_failures: usize,
    missing_root_results: usize,
    illegal_moves: usize,
    non_finite_scores: usize,
    max_ply_draws: usize,
    repetition_draws: usize,
    perpetual_check_games: usize,
}

#[derive(Serialize)]
struct GameReport {
    pair: usize,
    opening_family: String,
    opening_ply: usize,
    candidate_color: String,
    result: String,
    termination: String,
    played_plies: usize,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    lineage_id: &'static str,
    candidate_sha256: String,
    parent_sha256: String,
    openings_sha256: String,
    node_budget_per_move: Option<u64>,
    time_limit_ms: Option<u64>,
    max_depth: u8,
    max_plies: usize,
    pairs: usize,
    games: usize,
    score: f64,
    score_wilson_95_lower: f64,
    passes_point_score: bool,
    passes_noninferiority: bool,
    passes_safety: bool,
    promotion_gate_passed: bool,
    self_play: bool,
    counts: MatchCounts,
    game_reports: Vec<GameReport>,
}

enum GameResult {
    CandidateWin,
    ParentWin,
    Draw,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    if args.pairs == 0 || args.max_depth == 0 || args.max_plies == 0 {
        return Err(anyhow!(
            "pairs, nodes, max-depth, and max-plies must be positive"
        ));
    }
    if args.json.exists() {
        return Err(anyhow!("refusing to overwrite {}", args.json.display()));
    }
    let candidate_hash = verify_model(&args.candidate, &args.candidate_manifest, false)?;
    let external_parent = args.parent_authorization.as_ref().is_some_and(|path| {
        serde_json::from_slice::<serde_json::Value>(&fs::read(path).unwrap_or_default())
            .ok()
            .and_then(|v| {
                v.get("authorization_kind")
                    .and_then(|x| x.as_str())
                    .map(|k| k == "evaluation_only_external_baseline")
            })
            .unwrap_or(false)
    });
    let parent_hash = verify_model(&args.parent, &args.parent_manifest, external_parent)?;
    match (&args.registry, &args.parent_authorization) {
        (Some(path), None) => {
            let registry: Registry = serde_json::from_slice(&fs::read(path)?)?;
            if registry.lineage_id != LINEAGE_ID
                || !registry
                    .accepted
                    .iter()
                    .any(|e| e.model_sha256 == parent_hash)
            {
                return Err(anyhow!(
                    "parent is not accepted in the scratch promotion registry"
                ));
            }
        }
        (None, Some(path)) => {
            let authorization: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
            let schema = authorization.get("schema").and_then(|v| v.as_str());
            let model_sha = authorization
                .get("model")
                .and_then(|v| v.get("sha256"))
                .and_then(|v| v.as_str());
            if schema != Some("halfkp_workflow_parent_authorization_v1")
                || model_sha != Some(parent_hash.as_str())
            {
                return Err(anyhow!("parent authorization does not match parent model"));
            }
        }
        _ => {
            return Err(anyhow!(
                "exactly one of --registry and --parent-authorization is required"
            ))
        }
    }
    if args.nodes.is_none() == args.time_limit_ms.is_none() {
        return Err(anyhow!(
            "exactly one of --nodes and --time-limit-ms is required"
        ));
    }
    if args.nodes.is_some_and(|n| n == 0) || args.time_limit_ms.is_some_and(|t| t == 0) {
        return Err(anyhow!("nodes and time-limit-ms must be positive"));
    }
    if candidate_hash == parent_hash && !args.self_play {
        return Err(anyhow!("candidate and parent are identical"));
    }
    let openings = load_openings(
        &args.openings,
        args.pairs,
        &args.openings_format,
        &args.opening_selection,
        args.opening_seed,
    )?;
    let candidate_eval = Arc::new(EngineEvaluator::new(&args.candidate, 0.0)?);
    let parent_eval = Arc::new(EngineEvaluator::new(&args.parent, 0.0)?);
    let mut candidate = ShogiAI::<_, HISTORY_CAPACITY>::new(candidate_eval);
    let mut parent = ShogiAI::<_, HISTORY_CAPACITY>::new(parent_eval);
    candidate.set_emit_info(false);
    parent.set_emit_info(false);
    let mut counts = MatchCounts::default();
    let mut games = Vec::with_capacity(args.pairs * 2);
    for (pair, opening) in openings.iter().enumerate() {
        for &candidate_black in &[true, false] {
            let (result, termination, played_plies) = play_game(
                opening,
                candidate_black,
                &mut candidate,
                &mut parent,
                &args,
                &mut counts,
            )?;
            match result {
                GameResult::CandidateWin => {
                    counts.candidate_wins += 1;
                    if candidate_black {
                        counts.candidate_black_wins += 1
                    } else {
                        counts.candidate_white_wins += 1
                    }
                }
                GameResult::ParentWin => counts.parent_wins += 1,
                GameResult::Draw => counts.draws += 1,
            }
            games.push(GameReport {
                pair,
                opening_family: opening.family.clone(),
                opening_ply: opening.ply,
                candidate_color: if candidate_black { "black" } else { "white" }.into(),
                result: match result {
                    GameResult::CandidateWin => "candidate_win",
                    GameResult::ParentWin => "parent_win",
                    GameResult::Draw => "draw",
                }
                .into(),
                termination,
                played_plies,
            });
        }
    }
    let game_count = games.len();
    let score = (counts.candidate_wins as f64 + 0.5 * counts.draws as f64) / game_count as f64;
    let lower = wilson_lower(
        counts.candidate_wins as f64 + 0.5 * counts.draws as f64,
        game_count,
    );
    let safety = counts.search_failures == 0
        && counts.missing_root_results == 0
        && counts.illegal_moves == 0
        && counts.non_finite_scores == 0;
    let report = Report {
        schema: "halfkp_promotion_match_v1",
        lineage_id: LINEAGE_ID,
        candidate_sha256: candidate_hash,
        parent_sha256: parent_hash,
        openings_sha256: sha256_file(&args.openings)?,
        node_budget_per_move: args.nodes,
        time_limit_ms: args.time_limit_ms,
        max_depth: args.max_depth,
        max_plies: args.max_plies,
        pairs: args.pairs,
        games: game_count,
        score,
        score_wilson_95_lower: lower,
        passes_point_score: score >= 0.5,
        passes_noninferiority: lower >= 0.475,
        passes_safety: safety,
        promotion_gate_passed: !args.self_play && score >= 0.5 && lower >= 0.475 && safety,
        self_play: args.self_play,
        counts,
        game_reports: games,
    };
    let bytes = serde_json::to_vec_pretty(&report)?;
    println!("{}", String::from_utf8_lossy(&bytes));
    if let Some(parent) = args.json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.json, [&bytes[..], b"\n"].concat())?;
    Ok(())
}

fn verify_model(model: &Path, manifest: &Path, allow_external: bool) -> Result<String> {
    let declared: ModelManifest = serde_json::from_slice(&fs::read(manifest)?)?;
    let actual = sha256_file(model)?;
    let lineage_ok = declared.lineage_id == LINEAGE_ID
        || (allow_external && declared.lineage_id == "v2.5.0-kpp-derived-evaluation-only");
    if !lineage_ok || declared.model.sha256 != actual {
        return Err(anyhow!(
            "{} is not its declared scratch-lineage artifact",
            model.display()
        ));
    }
    Ok(actual)
}

fn load_openings(
    path: &Path,
    count: usize,
    format: &str,
    selection: &str,
    seed: u64,
) -> Result<Vec<Opening>> {
    if format == "usi-lines" {
        let mut openings = Vec::new();
        for (line_no, line) in BufReader::new(File::open(path)?).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if position_and_history_from_sfen_or_usi(&line).is_none() {
                return Err(anyhow!("invalid USI opening line {}", line_no + 1));
            }
            openings.push(Opening {
                family: format!("usi-line-{line_no}"),
                ply: line.split_whitespace().skip(2).count(),
                position_spec: line,
            });
        }
        if selection == "sha256-seed" {
            openings.sort_by_key(|opening| {
                let mut bytes = seed.to_le_bytes().to_vec();
                bytes.extend_from_slice(opening.position_spec.as_bytes());
                sha256_hex(&bytes)
            });
        } else if selection != "first-unique" {
            return Err(anyhow!("unknown opening selection {selection}"));
        }
        openings.truncate(count);
        if openings.len() != count {
            return Err(anyhow!(
                "requested {count} USI openings, found {}",
                openings.len()
            ));
        }
        return Ok(openings);
    }
    if format != "dataset-jsonl" || selection != "first-unique" {
        return Err(anyhow!(
            "dataset-jsonl supports only first-unique selection"
        ));
    }
    let mut families = HashSet::new();
    let mut out = Vec::with_capacity(count);
    for (line_no, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let line = line?;
        let record: OpeningRecord = serde_json::from_str(&line)
            .with_context(|| format!("parse opening line {}", line_no + 1))?;
        if record.split != "valid"
            || !(8..=40).contains(&record.ply)
            || !families.insert(record.opening_family.clone())
        {
            continue;
        }
        Position::new(
            PartialPosition::from_usi(&format!("sfen {}", record.sfen))
                .map_err(|e| anyhow!("opening SFEN: {e:?}"))?,
        );
        out.push(Opening {
            family: record.opening_family,
            ply: record.ply,
            position_spec: format!("sfen {}", record.sfen),
        });
        if out.len() == count {
            break;
        }
    }
    if out.len() != count {
        return Err(anyhow!(
            "requested {count} unique valid opening families, found {}",
            out.len()
        ));
    }
    Ok(out)
}

fn play_game(
    opening: &Opening,
    candidate_black: bool,
    candidate: &mut ShogiAI<Arc<EngineEvaluator>, HISTORY_CAPACITY>,
    parent: &mut ShogiAI<Arc<EngineEvaluator>, HISTORY_CAPACITY>,
    args: &Args,
    counts: &mut MatchCounts,
) -> Result<(GameResult, String, usize)> {
    let (mut position, mut history) = position_and_history_from_sfen_or_usi(&opening.position_spec)
        .ok_or_else(|| anyhow!("invalid opening position"))?;
    for played in 0..args.max_plies {
        let legal = position.legal_moves();
        if legal.is_empty() {
            return Ok((
                winner(position.side_to_move().flip(), candidate_black),
                "no_legal_move".into(),
                played,
            ));
        }
        let candidate_turn = (position.side_to_move() == Color::Black) == candidate_black;
        let ai = if candidate_turn {
            &mut *candidate
        } else {
            &mut *parent
        };
        ai.reset_for_independent_search();
        ai.sennichite_detector = history.clone();
        let outcome = ai.search(&mut position, {
            let limits = SearchLimits::from_millis(args.max_depth, args.time_limit_ms);
            match args.nodes {
                Some(nodes) => limits.with_max_nodes(nodes),
                None => limits,
            }
        });
        if outcome.failed {
            counts.search_failures += 1;
            return Ok((GameResult::Draw, "search_failure".into(), played));
        }
        let Some(root) = outcome.root else {
            counts.missing_root_results += 1;
            return Ok((GameResult::Draw, "missing_root".into(), played));
        };
        if root.score.is_some_and(|score| !score.is_finite()) {
            counts.non_finite_scores += 1;
            return Ok((GameResult::Draw, "non_finite_score".into(), played));
        }
        if !legal.contains(&root.best_move) {
            counts.illegal_moves += 1;
            return Ok((GameResult::Draw, "illegal_move".into(), played));
        }
        let moved_by = position.side_to_move();
        position.do_move(root.best_move);
        history.record_position_after_move(&position, moved_by);
        match history.adjudicate(&position) {
            SennichiteStatus::None => {}
            SennichiteStatus::Draw => {
                counts.repetition_draws += 1;
                return Ok((GameResult::Draw, "repetition".into(), played + 1));
            }
            SennichiteStatus::PerpetualCheckLoss { loser } => {
                counts.perpetual_check_games += 1;
                return Ok((
                    winner(loser.flip(), candidate_black),
                    "perpetual_check".into(),
                    played + 1,
                ));
            }
        }
    }
    counts.max_ply_draws += 1;
    Ok((GameResult::Draw, "max_plies".into(), args.max_plies))
}

fn winner(color: Color, candidate_black: bool) -> GameResult {
    if (color == Color::Black) == candidate_black {
        GameResult::CandidateWin
    } else {
        GameResult::ParentWin
    }
}

fn wilson_lower(successes: f64, n: usize) -> f64 {
    let n = n as f64;
    let p = successes / n;
    let z = 1.959_963_984_540_054;
    (p + z * z / (2.0 * n) - z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).sqrt())
        / (1.0 + z * z / n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winner_respects_candidate_color_assignment() {
        assert!(matches!(
            winner(Color::Black, true),
            GameResult::CandidateWin
        ));
        assert!(matches!(
            winner(Color::White, false),
            GameResult::CandidateWin
        ));
        assert!(matches!(winner(Color::Black, false), GameResult::ParentWin));
    }

    #[test]
    fn wilson_lower_is_conservative_and_monotonic() {
        let tied = wilson_lower(50.0, 100);
        let strong = wilson_lower(95.0, 100);
        assert!(tied < 0.5);
        assert!(strong > tied && strong < 0.95);
    }
}
