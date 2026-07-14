use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use shogi_ai::ai::{SearchLimits, SearchReport, ShogiAI};
use shogi_ai::evaluation::{Evaluator, SparseModel};
use shogi_ai::search_quality::{canonical_sfen, MateOracle, MateProof};
use shogi_ai::search_quality::{
    generator_source_sha256, AtomicOutput, DatasetSplit, SuiteKind, SuiteManifest,
};
use shogi_ai::utils::{format_move_usi, parse_usi_move_for_color, position_from_sfen_or_usi};
use shogi_core::{Color, Move, PieceKind};
use shogi_lib::Position;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const HISTORY_CAPACITY: usize = 256;
const SEARCH_DEPTH: u8 = 7;
const SEARCH_NODES: u64 = 20_000;
const MATE_ROOT_NODES: u64 = 8_192;
const MATE_REPLY_NODES: u64 = 128;
const BASELINE_EXPECTED_FIRST: usize = 110;
const BASELINE_MATE_ACCEPTABLE: usize = 199;

#[derive(Parser, Debug)]
#[command(about = "One-shot aggregate-only frozen holdout search gate")]
struct Args {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    weights: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Clone)]
struct GateEvaluator<'a> {
    model: &'a SparseModel,
}

impl Evaluator for GateEvaluator<'_> {
    fn evaluate(&self, position: &Position) -> f32 {
        self.model.predict_from_position(position)
    }
}

#[derive(Deserialize)]
struct MateRecord {
    sfen: String,
    first_move: String,
    mate_horizon: u8,
}

#[derive(Deserialize)]
struct QuietRecord {
    sfen: String,
    legal_evasions: Vec<String>,
    quiet_evasions: Vec<String>,
}

#[derive(Deserialize)]
struct ResourceRecord {
    source_sfen: String,
    cycle_start_sfen: String,
    source_to_cycle_start: Vec<String>,
    proof_line: Vec<String>,
    loser: String,
    start_black_hand: [u8; 7],
    final_black_hand: [u8; 7],
    start_white_hand: [u8; 7],
    final_white_hand: [u8; 7],
}

struct FrozenSuite {
    path: PathBuf,
    entry: Value,
    sidecar: SuiteManifest,
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn fixed_limits() -> SearchLimits {
    SearchLimits {
        max_depth: SEARCH_DEPTH,
        time_limit_ms: None,
        node_limit: Some(SEARCH_NODES),
    }
}

fn run_search(model: &SparseModel, position: &mut Position) -> SearchReport {
    let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(GateEvaluator { model });
    ai.set_emit_info(false);
    ai.set_mate_search_budgets(MATE_ROOT_NODES, MATE_REPLY_NODES);
    ai.set_mate_reply_candidates(3);
    ai.sennichite_detector.record_position(position);
    ai.find_best_move_with_limits(position, fixed_limits())
}

fn normalized_set(moves: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut values = moves.into_iter().collect::<Vec<_>>();
    values.sort();
    values.dedup();
    values
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut rows = Vec::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        rows.push(serde_json::from_str(&line).with_context(|| {
            format!(
                "invalid holdout record at {}:{}",
                path.display(),
                line_no + 1
            )
        })?);
    }
    Ok(rows)
}

fn expected_target(kind: SuiteKind, split: DatasetSplit) -> usize {
    match (kind, split) {
        (SuiteKind::MateSacrifice, DatasetSplit::Dev) => 200,
        (SuiteKind::MateSacrifice, DatasetSplit::Holdout) => 100,
        (SuiteKind::QuietEvasion, DatasetSplit::Dev) => 500,
        (SuiteKind::QuietEvasion, DatasetSplit::Holdout) => 200,
        (SuiteKind::ResourceCycle, DatasetSplit::Dev) => 100,
        (SuiteKind::ResourceCycle, DatasetSplit::Holdout) => 50,
    }
}

fn expected_generator(kind: SuiteKind) -> &'static str {
    match kind {
        SuiteKind::MateSacrifice => "mate_sacrifice_miner",
        SuiteKind::QuietEvasion => "quiet_evasion_miner",
        SuiteKind::ResourceCycle => "resource_cycle_miner",
    }
}

fn manifest_suite_paths(manifest: &Value) -> Result<[FrozenSuite; 3]> {
    let files = manifest["suite_files"]
        .as_array()
        .ok_or_else(|| anyhow!("manifest suite_files missing"))?;
    if files.len() != 6 {
        return Err(anyhow!("manifest must contain exactly six suite files"));
    }
    let mut suites = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut combinations = std::collections::HashSet::new();
    for file in files {
        let path = file["path"]
            .as_str()
            .ok_or_else(|| anyhow!("manifest suite path missing"))?;
        if !seen.insert(path.to_string()) {
            return Err(anyhow!("manifest contains duplicate suite path: {path}"));
        }
        let sidecar_path = file["sidecar_manifest_path"]
            .as_str()
            .ok_or_else(|| anyhow!("manifest sidecar path missing"))?;
        if file["sha256"].as_str().is_none()
            || file["sidecar_manifest_sha256"].as_str().is_none()
            || file["records"].as_u64().is_none()
        {
            return Err(anyhow!("manifest suite entry is incomplete: {path}"));
        }
        let suite_path = PathBuf::from(path);
        let actual_sidecar_path = suite_path.with_extension("manifest.json");
        if Path::new(sidecar_path) != actual_sidecar_path {
            return Err(anyhow!(
                "manifest sidecar path does not match suite path: {path}"
            ));
        }
        let sidecar: SuiteManifest = serde_json::from_reader(
            File::open(&actual_sidecar_path)
                .with_context(|| format!("failed to open sidecar {sidecar_path}"))?,
        )
        .with_context(|| format!("invalid sidecar {sidecar_path}"))?;
        if sidecar.schema_version != 1 || sidecar.generator_worktree_dirty {
            return Err(anyhow!("sidecar is not clean schema v1: {path}"));
        }
        if !combinations.insert((sidecar.suite_kind, sidecar.split)) {
            return Err(anyhow!("duplicate suite kind/split in sidecars: {path}"));
        }
        if sidecar.generator != expected_generator(sidecar.suite_kind)
            || sidecar.input_sha256.is_empty()
            || sidecar.output_sha256 != sha256_file(&suite_path)?
            || file["sha256"].as_str() != Some(sidecar.output_sha256.as_str())
            || file["sidecar_manifest_sha256"].as_str()
                != Some(sha256_file(&actual_sidecar_path)?.as_str())
            || sidecar.records_written != file["records"].as_u64().unwrap() as usize
        {
            return Err(anyhow!("sidecar/output metadata mismatch: {path}"));
        }
        if !Path::new(&sidecar.input_path).is_file()
            || sha256_file(Path::new(&sidecar.input_path))? != sidecar.input_sha256
            || !Path::new(&sidecar.output_path).is_file()
            || fs::canonicalize(&sidecar.output_path)? != fs::canonicalize(&suite_path)?
            || sidecar.generator_commit.is_empty()
            || sidecar.generator_source_sha256.is_empty()
        {
            return Err(anyhow!("sidecar input/path metadata mismatch: {path}"));
        }
        if sidecar.split == DatasetSplit::Holdout {
            suites.push(FrozenSuite {
                path: suite_path,
                entry: file.clone(),
                sidecar,
            });
        }
    }
    if combinations.len() != 6 || suites.len() != 3 {
        return Err(anyhow!(
            "manifest must contain exactly three holdout suites"
        ));
    }
    suites.sort_by_key(|suite| match suite.sidecar.suite_kind {
        SuiteKind::MateSacrifice => 0u8,
        SuiteKind::QuietEvasion => 1,
        SuiteKind::ResourceCycle => 2,
    });
    if suites
        .iter()
        .map(|suite| suite.sidecar.suite_kind)
        .collect::<std::collections::HashSet<_>>()
        .len()
        != 3
    {
        return Err(anyhow!("holdout suite kinds are not exactly one each"));
    }
    Ok([suites.remove(0), suites.remove(0), suites.remove(0)])
}

fn verify_frozen_manifest(path: &Path, weights: &Path) -> Result<(Value, [FrozenSuite; 3])> {
    let manifest: Value = serde_json::from_reader(File::open(path)?)?;
    if manifest["freeze_eligible"] != true
        || manifest["allow_dirty"] != false
        || manifest["allow_incomplete"] != false
        || manifest["source_game_intersection"] != 0
        || manifest["sfen_intersection"] != 0
    {
        return Err(anyhow!("manifest is not an eligible frozen holdout"));
    }
    if manifest["schema_version"] != 1 || manifest["generator"] != "search_suite_manifest" {
        return Err(anyhow!("unsupported aggregate manifest schema"));
    }
    let generator_source = manifest["generator_source_sha256"]
        .as_str()
        .ok_or_else(|| anyhow!("aggregate generator source SHA missing"))?;
    if generator_source_sha256()? != generator_source {
        return Err(anyhow!(
            "aggregate generator source SHA does not match current tree"
        ));
    }
    let generator_commit = manifest["generator_commit"]
        .as_str()
        .ok_or_else(|| anyhow!("aggregate generator commit missing"))?;
    if !Command::new("git")
        .args(["cat-file", "-e", &format!("{generator_commit}^{{commit}}")])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
    {
        return Err(anyhow!("aggregate generator commit is unavailable"));
    }
    let source_files = manifest["source_files"]
        .as_array()
        .ok_or_else(|| anyhow!("aggregate source_files missing"))?;
    if source_files.is_empty() {
        return Err(anyhow!("aggregate source_files is empty"));
    }
    for source in source_files {
        let source_path = Path::new(
            source["path"]
                .as_str()
                .ok_or_else(|| anyhow!("aggregate source path missing"))?,
        );
        let declared_sha = source["sha256"]
            .as_str()
            .ok_or_else(|| anyhow!("aggregate source SHA missing"))?;
        if !source_path.is_file() || sha256_file(source_path)? != declared_sha {
            return Err(anyhow!("aggregate source SHA-256 mismatch"));
        }
    }
    let targets = manifest["targets"]
        .as_array()
        .ok_or_else(|| anyhow!("aggregate targets missing"))?;
    if targets.len() != 6 {
        return Err(anyhow!("aggregate must contain exactly six targets"));
    }
    let mut seen_targets = std::collections::HashSet::new();
    for target in targets {
        let kind: SuiteKind = serde_json::from_value(target["suite_kind"].clone())?;
        let split: DatasetSplit = serde_json::from_value(target["split"].clone())?;
        if !seen_targets.insert((kind, split))
            || target["actual"].as_u64() != Some(expected_target(kind, split) as u64)
            || target["required"].as_u64() != Some(expected_target(kind, split) as u64)
            || target["passed"] != true
        {
            return Err(anyhow!("aggregate target metadata is not frozen/complete"));
        }
    }
    if seen_targets.len() != 6 {
        return Err(anyhow!(
            "aggregate targets are not the exact six combinations"
        ));
    }
    if manifest["all_sidecars_clean"] != true {
        return Err(anyhow!("aggregate reports dirty sidecars"));
    }
    let sources = manifest["source_files"]
        .as_array()
        .ok_or_else(|| anyhow!("aggregate source_files missing"))?;
    if sources.is_empty() {
        return Err(anyhow!("aggregate has no source files"));
    }
    let mut source_hashes = std::collections::HashSet::new();
    for source in sources {
        let source_path = source["path"]
            .as_str()
            .ok_or_else(|| anyhow!("aggregate source path missing"))?;
        let source_sha = source["sha256"]
            .as_str()
            .ok_or_else(|| anyhow!("aggregate source SHA missing"))?;
        let actual_sha = sha256_file(Path::new(source_path))?;
        if actual_sha != source_sha || !source_hashes.insert(source_sha.to_string()) {
            return Err(anyhow!("aggregate source path/SHA mismatch"));
        }
    }
    let weight_entry = &manifest["weight"];
    let manifest_weight_path = weight_entry["path"]
        .as_str()
        .ok_or_else(|| anyhow!("aggregate weight path missing"))?;
    let weight_sha = sha256_file(weights)?;
    if Path::new(manifest_weight_path) != weights
        || weight_entry["sha256"].as_str() != Some(weight_sha.as_str())
    {
        return Err(anyhow!("weight path/SHA does not match frozen manifest"));
    }
    let suites = manifest_suite_paths(&manifest)?;
    for file in manifest["suite_files"].as_array().unwrap() {
        let suite_path = Path::new(file["path"].as_str().unwrap());
        let sidecar_path = suite_path.with_extension("manifest.json");
        let sidecar: SuiteManifest = serde_json::from_reader(File::open(&sidecar_path)?)?;
        if sidecar.generator_source_sha256 != generator_source
            || sidecar.generator_commit != generator_commit
            || !source_hashes.contains(&sidecar.input_sha256)
        {
            return Err(anyhow!("suite sidecar generator/source metadata mismatch"));
        }
    }
    for suite in &suites {
        if !suite.path.is_file() || !suite.path.with_extension("manifest.json").is_file() {
            return Err(anyhow!("holdout suite or sidecar is missing"));
        }
        if suite.sidecar.generator_source_sha256 != generator_source
            || Some(suite.sidecar.generator_commit.as_str())
                != manifest["generator_commit"].as_str()
            || suite.sidecar.generator_worktree_dirty
        {
            return Err(anyhow!("holdout sidecar generator metadata mismatch"));
        }
        if !source_hashes.contains(&suite.sidecar.input_sha256)
            || suite.sidecar.records_written
                != expected_target(suite.sidecar.suite_kind, DatasetSplit::Holdout)
            || fs::canonicalize(&suite.sidecar.output_path)? != fs::canonicalize(&suite.path)?
        {
            return Err(anyhow!(
                "holdout sidecar source/count/path metadata mismatch"
            ));
        }
        let suite_sha = sha256_file(&suite.path)?;
        let sidecar_sha = sha256_file(&suite.path.with_extension("manifest.json"))?;
        if suite.entry["sha256"].as_str() != Some(suite_sha.as_str())
            || suite.entry["sidecar_manifest_sha256"].as_str() != Some(sidecar_sha.as_str())
            || suite.entry["records"].as_u64() != Some(suite.sidecar.records_written as u64)
        {
            return Err(anyhow!("frozen suite SHA/count mismatch"));
        }
    }
    Ok((manifest, suites))
}

fn hand_counts(position: &Position, color: Color) -> [u8; 7] {
    let hand = position.hand(color);
    [
        hand.count(PieceKind::Pawn).unwrap_or(0),
        hand.count(PieceKind::Lance).unwrap_or(0),
        hand.count(PieceKind::Knight).unwrap_or(0),
        hand.count(PieceKind::Silver).unwrap_or(0),
        hand.count(PieceKind::Gold).unwrap_or(0),
        hand.count(PieceKind::Bishop).unwrap_or(0),
        hand.count(PieceKind::Rook).unwrap_or(0),
    ]
}

fn resource_loss(start: &Position, end: &Position, loser: Color) -> bool {
    if start.keys().0 != end.keys().0 {
        return false;
    }
    let own_start = hand_counts(start, loser);
    let own_end = hand_counts(end, loser);
    let opp_start = hand_counts(start, loser.flip());
    let opp_end = hand_counts(end, loser.flip());
    own_end.iter().zip(own_start).all(|(n, o)| *n <= o)
        && opp_end.iter().zip(opp_start).all(|(n, o)| *n >= o)
        && own_end.iter().zip(own_start).any(|(n, o)| *n < o)
        && opp_end.iter().zip(opp_start).any(|(n, o)| *n > o)
}

fn parse_move(position: &Position, text: &str) -> Result<Move> {
    let mv = parse_usi_move_for_color(text, position.side_to_move())
        .ok_or_else(|| anyhow!("invalid holdout move"))?;
    if !position.legal_moves().contains(&mv) {
        return Err(anyhow!("illegal holdout move"));
    }
    Ok(mv)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if fs::symlink_metadata(&args.output).is_ok() {
        return Err(anyhow!(
            "output already exists; the holdout gate is one-shot"
        ));
    }
    let (manifest, suites) = verify_frozen_manifest(&args.manifest, &args.weights)?;
    let model = {
        let mut model = SparseModel::new(0.0, 0.0);
        model.load(&args.weights)?;
        model
    };

    let mut mate_expected = 0usize;
    let mut mate_acceptable = 0usize;
    let mut mate_score = 0usize;
    let mut mate_unknown = 0usize;
    let mut mate_nodes = 0u64;
    let mut mate_depth = 0u64;
    let root_rejected = 0u64;
    let mut reply_rejected = 0u64;
    let mut mate_records = 0usize;

    let mut quiet_records = 0usize;
    let mut quiet_legal_exact = 0usize;
    let mut quiet_set_exact = 0usize;
    let mut quiet_candidate_legal = 0usize;
    let mut quiet_candidate_quiet = 0usize;

    let mut resource_records = 0usize;
    let mut resource_replay_valid = 0usize;
    let mut resource_selection_legal = 0usize;
    let mut resource_selection_matches_proof = 0usize;

    for suite in &suites {
        if suite.sidecar.suite_kind == SuiteKind::MateSacrifice {
            for record in read_jsonl::<MateRecord>(&suite.path)? {
                mate_records += 1;
                let mut position = position_from_sfen_or_usi(&record.sfen)
                    .ok_or_else(|| anyhow!("invalid holdout position"))?;
                let attacker = position.side_to_move();
                let report = run_search(&model, &mut position);
                if report
                    .best_move
                    .is_some_and(|mv| format_move_usi(mv) == record.first_move)
                {
                    mate_expected += 1;
                }
                mate_score += usize::from(report.score == Some(f32::INFINITY));
                mate_unknown += report.mate_unknown as usize;
                mate_nodes += report.nodes;
                mate_depth += u64::from(report.completed_depth);
                reply_rejected += report.mate_rejected;
                if let Some(best) = report.best_move {
                    position.do_move(best);
                    let mut oracle = MateOracle::new(2_000_000);
                    if matches!(
                        oracle.prove(
                            &mut position,
                            attacker,
                            record.mate_horizon.saturating_sub(1)
                        ),
                        MateProof::ProvenMate(_)
                    ) {
                        mate_acceptable += 1;
                    }
                }
            }
        } else if suite.sidecar.suite_kind == SuiteKind::QuietEvasion {
            for record in read_jsonl::<QuietRecord>(&suite.path)? {
                quiet_records += 1;
                let mut position = position_from_sfen_or_usi(&record.sfen)
                    .ok_or_else(|| anyhow!("invalid holdout position"))?;
                let legal =
                    normalized_set(position.legal_moves().iter().copied().map(format_move_usi));
                if legal == normalized_set(record.legal_evasions.clone()) {
                    quiet_legal_exact += 1;
                }
                let quiet = normalized_set(
                    position
                        .legal_moves()
                        .iter()
                        .copied()
                        .filter(|mv| {
                            let capture = match mv {
                                Move::Normal { to, .. } => position.piece_at(*to).is_some(),
                                Move::Drop { .. } => false,
                            };
                            !capture && !position.is_check_move(*mv)
                        })
                        .map(format_move_usi),
                );
                if normalized_set(record.legal_evasions) == legal
                    && normalized_set(record.quiet_evasions.clone()) == quiet
                {
                    quiet_set_exact += 1;
                }
                let report = run_search(&model, &mut position);
                if let Some(best) = report.best_move {
                    let text = format_move_usi(best);
                    quiet_candidate_legal += usize::from(legal.contains(&text));
                    quiet_candidate_quiet += usize::from(record.quiet_evasions.contains(&text));
                }
            }
        } else if suite.sidecar.suite_kind == SuiteKind::ResourceCycle {
            for record in read_jsonl::<ResourceRecord>(&suite.path)? {
                resource_records += 1;
                let mut source = position_from_sfen_or_usi(&record.source_sfen)
                    .ok_or_else(|| anyhow!("invalid holdout position"))?;
                let mut replay = source.clone();
                let mut valid = true;
                for text in &record.source_to_cycle_start {
                    let mv = parse_move(&replay, text)?;
                    replay.do_move(mv);
                }
                let cycle_start = position_from_sfen_or_usi(&record.cycle_start_sfen)
                    .ok_or_else(|| anyhow!("invalid holdout position"))?;
                valid &= canonical_sfen(&replay) == canonical_sfen(&cycle_start);
                valid &= hand_counts(&replay, Color::Black) == record.start_black_hand;
                valid &= hand_counts(&replay, Color::White) == record.start_white_hand;
                let proof_start = replay.clone();
                for text in &record.proof_line {
                    let mv = parse_move(&replay, text)?;
                    replay.do_move(mv);
                }
                valid &= resource_loss(
                    &proof_start,
                    &replay,
                    match record.loser.as_str() {
                        "black" => Color::Black,
                        "white" => Color::White,
                        _ => return Err(anyhow!("invalid holdout loser")),
                    },
                );
                valid &= hand_counts(&replay, Color::Black) == record.final_black_hand;
                valid &= hand_counts(&replay, Color::White) == record.final_white_hand;
                resource_replay_valid += usize::from(valid);
                let report = run_search(&model, &mut source);
                if let Some(best) = report.best_move {
                    resource_selection_legal += 1;
                    if record
                        .source_to_cycle_start
                        .first()
                        .is_some_and(|mv| *mv == format_move_usi(best))
                    {
                        resource_selection_matches_proof += 1;
                    }
                }
            }
        }
    }

    let output = json!({
        "schema_version": 1,
        "gate": "holdout_search_gate",
        "fixed_search": {"depth": SEARCH_DEPTH, "nodes": SEARCH_NODES, "mate_root_nodes": MATE_ROOT_NODES, "mate_reply_nodes": MATE_REPLY_NODES},
        "mate": {"records": mate_records, "expected_first": mate_expected, "mate_acceptable": mate_acceptable, "mate_score": mate_score, "unknown": mate_unknown, "nodes_total": mate_nodes, "completed_depth_total": mate_depth, "root_rejected": root_rejected, "reply_rejected": reply_rejected, "dev_baseline_expected_first": BASELINE_EXPECTED_FIRST, "dev_baseline_mate_acceptable": BASELINE_MATE_ACCEPTABLE},
        "quiet": {"records": quiet_records, "legal_set_exact": quiet_legal_exact, "evasion_set_exact": quiet_set_exact, "candidate_legal": quiet_candidate_legal, "candidate_quiet": quiet_candidate_quiet},
        "resource": {"records": resource_records, "proof_replay_valid": resource_replay_valid, "selection_legal": resource_selection_legal, "selection_matches_proof_first": resource_selection_matches_proof},
        "manifest_sha256": sha256_file(&args.manifest)?,
        "weight_sha256": sha256_file(&args.weights)?,
        "generator_commit": manifest["generator_commit"],
        "git_head": String::from_utf8_lossy(&Command::new("git").args(["rev-parse", "HEAD"]).output()?.stdout).trim().to_string(),
        "holdout_suite_count": suites.len()
    });
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = AtomicOutput::new(&args.output)?;
    serde_json::to_writer_pretty(&mut file, &output)?;
    writeln!(file)?;
    file.commit()?;
    Ok(())
}
