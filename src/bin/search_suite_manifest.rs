use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use shogi_ai::search_quality::{
    canonical_sfen, deduplicate_input_positions, ensure_distinct_paths, generator_source_sha256,
    generator_worktree_dirty, hand_counts, load_input_positions, sha256_file, simple_see,
    AtomicOutput, DatasetSplit, MateOracle, MateProof, SuiteKind, SuiteManifest,
};
use shogi_ai::utils::{format_move_usi, parse_usi_move_for_color, position_from_sfen_or_usi};
use shogi_core::{Color, Move};
use shogi_lib::Position;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Parser, Debug)]
#[command(about = "Validate and freeze generated search-quality suites")]
struct Args {
    #[arg(long, required = true, num_args = 1..)]
    suite_files: Vec<PathBuf>,
    #[arg(long, required = true, num_args = 1..)]
    source_files: Vec<PathBuf>,
    #[arg(long)]
    weight: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_incomplete: bool,
    #[arg(long, default_value_t = false)]
    allow_dirty: bool,
}

#[derive(Serialize)]
struct FileEntry {
    path: String,
    sha256: String,
    records: Option<usize>,
    sidecar_manifest_path: Option<String>,
    sidecar_manifest_sha256: Option<String>,
}

#[derive(Serialize)]
struct TargetResult {
    suite_kind: SuiteKind,
    split: DatasetSplit,
    actual: usize,
    required: usize,
    passed: bool,
}

#[derive(Serialize)]
struct AggregateManifest {
    schema_version: u32,
    generator: &'static str,
    generator_commit: String,
    generator_worktree_dirty: bool,
    generator_source_sha256: String,
    all_sidecars_clean: bool,
    freeze_eligible: bool,
    allow_incomplete: bool,
    allow_dirty: bool,
    source_game_intersection: usize,
    sfen_intersection: usize,
    targets: Vec<TargetResult>,
    weight: FileEntry,
    source_files: Vec<FileEntry>,
    suite_files: Vec<FileEntry>,
}

#[derive(Clone)]
struct SourceRow {
    game_key: String,
    canonical_sfen: String,
}

struct SourcePool {
    path: PathBuf,
    nonempty: usize,
    valid: usize,
    duplicates: usize,
    rows: HashMap<usize, SourceRow>,
    games: HashSet<String>,
    sfens: HashSet<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MateRecord {
    schema_version: u32,
    source_index: usize,
    source_game_key: Option<String>,
    sfen: String,
    first_move: String,
    simple_see: i32,
    mate_horizon: u8,
    proof_line: Vec<String>,
    proof_line_plies: usize,
    root_defense_count: usize,
    proof_nodes: u64,
    proof_status: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QuietRecord {
    schema_version: u32,
    source_index: usize,
    source_game_key: Option<String>,
    sfen: String,
    legal_evasions: Vec<String>,
    quiet_evasions: Vec<String>,
    legal_evasion_count: usize,
    quiet_evasion_count: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResourceRecord {
    schema_version: u32,
    source_index: usize,
    source_game_key: Option<String>,
    source_sfen: String,
    cycle_start_sfen: String,
    source_to_cycle_start: Vec<String>,
    loser: String,
    horizon: u8,
    cycle_length: usize,
    proof_line: Vec<String>,
    gave_check: Vec<bool>,
    start_black_hand: [u8; 7],
    final_black_hand: [u8; 7],
    start_white_hand: [u8; 7],
    final_white_hand: [u8; 7],
    proof_nodes: u64,
    proof_status: String,
}

fn same_file(first: &Path, second: &Path) -> Result<bool> {
    Ok(fs::canonicalize(first)? == fs::canonicalize(second)?)
}

fn expected_generator(kind: SuiteKind) -> &'static str {
    match kind {
        SuiteKind::MateSacrifice => "mate_sacrifice_miner",
        SuiteKind::QuietEvasion => "quiet_evasion_miner",
        SuiteKind::ResourceCycle => "resource_cycle_miner",
    }
}

fn required_filter_keys(kind: SuiteKind) -> &'static [&'static str] {
    match kind {
        SuiteKind::MateSacrifice => &["depths", "proof_node_limit", "candidate", "unknown_policy"],
        SuiteKind::QuietEvasion => &["position_limit", "record_limit", "reference", "quiet"],
        SuiteKind::ResourceCycle => &["depths", "node_limit", "negative_policy", "proof"],
    }
}

fn target(kind: SuiteKind, split: DatasetSplit) -> usize {
    match (kind, split) {
        (SuiteKind::MateSacrifice, DatasetSplit::Dev) => 200,
        (SuiteKind::MateSacrifice, DatasetSplit::Holdout) => 100,
        (SuiteKind::QuietEvasion, DatasetSplit::Dev) => 500,
        (SuiteKind::QuietEvasion, DatasetSplit::Holdout) => 200,
        (SuiteKind::ResourceCycle, DatasetSplit::Dev) => 100,
        (SuiteKind::ResourceCycle, DatasetSplit::Holdout) => 50,
    }
}

fn is_freeze_eligible(
    allow_dirty: bool,
    allow_incomplete: bool,
    current_dirty: bool,
    all_sidecars_clean: bool,
    targets_passed: bool,
    source_game_intersection: usize,
    sfen_intersection: usize,
) -> bool {
    !allow_dirty
        && !allow_incomplete
        && !current_dirty
        && all_sidecars_clean
        && targets_passed
        && source_game_intersection == 0
        && sfen_intersection == 0
}

fn file_entry(path: &Path, records: Option<usize>, sidecar: Option<&Path>) -> Result<FileEntry> {
    Ok(FileEntry {
        path: path.display().to_string(),
        sha256: sha256_file(path)?,
        records,
        sidecar_manifest_path: sidecar.map(|path| path.display().to_string()),
        sidecar_manifest_sha256: sidecar.map(sha256_file).transpose()?,
    })
}

fn load_source_pool(path: &Path) -> Result<(String, SourcePool)> {
    let sha = sha256_file(path)?;
    let (positions, nonempty) = load_input_positions(path)?;
    let valid = positions.len();
    let (_, duplicates) = deduplicate_input_positions(positions.clone());
    let mut rows = HashMap::new();
    let mut games = HashSet::new();
    let mut sfens = HashSet::new();
    for input in positions {
        let game_key = input.source_game_key.ok_or_else(|| {
            anyhow!(
                "every source row must have source_game_key for freezing: {}:{}",
                path.display(),
                input.source_line
            )
        })?;
        let canonical = canonical_sfen(&input.position);
        games.insert(game_key.clone());
        sfens.insert(canonical.clone());
        rows.insert(
            input.source_line,
            SourceRow {
                game_key,
                canonical_sfen: canonical,
            },
        );
    }
    Ok((
        sha,
        SourcePool {
            path: fs::canonicalize(path)?,
            nonempty,
            valid,
            duplicates,
            rows,
            games,
            sfens,
        },
    ))
}

fn record_context(path: &Path, line: usize, message: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("{}:{}: {message}", path.display(), line)
}

fn validate_source_identity(
    pool: &SourcePool,
    source_index: usize,
    game_key: Option<&str>,
    sfen: &str,
    path: &Path,
    line: usize,
) -> Result<Position> {
    let row = pool
        .rows
        .get(&source_index)
        .ok_or_else(|| record_context(path, line, "source_index does not exist in source pool"))?;
    if game_key != Some(row.game_key.as_str()) {
        return Err(record_context(
            path,
            line,
            "source_game_key does not match the exact source row",
        ));
    }
    let position = position_from_sfen_or_usi(sfen)
        .ok_or_else(|| record_context(path, line, "invalid source SFEN"))?;
    if canonical_sfen(&position) != row.canonical_sfen {
        return Err(record_context(
            path,
            line,
            "canonical SFEN does not match the exact source row",
        ));
    }
    Ok(position)
}

fn parse_legal_move(position: &Position, text: &str, path: &Path, line: usize) -> Result<Move> {
    let mv = parse_usi_move_for_color(text, position.side_to_move())
        .ok_or_else(|| record_context(path, line, format!("invalid USI move: {text}")))?;
    if !position.legal_moves().contains(&mv) {
        return Err(record_context(
            path,
            line,
            format!("illegal proof move: {text}"),
        ));
    }
    Ok(mv)
}

fn validate_mate_record(
    record: MateRecord,
    pool: &SourcePool,
    node_limit: u64,
    path: &Path,
    line: usize,
) -> Result<()> {
    if record.schema_version != 1
        || record.proof_status != "proven_mate"
        || record.mate_horizon == 0
        || record.mate_horizon % 2 == 0
        || record.proof_line.is_empty()
        || record.proof_line_plies != record.proof_line.len()
        || record.proof_line.len() > usize::from(record.mate_horizon)
        || record.first_move != record.proof_line[0]
        || record.proof_nodes == 0
        || record.proof_nodes > node_limit
    {
        return Err(record_context(path, line, "invalid mate record fields"));
    }
    let mut position = validate_source_identity(
        pool,
        record.source_index,
        record.source_game_key.as_deref(),
        &record.sfen,
        path,
        line,
    )?;
    let attacker = position.side_to_move();
    let first = parse_legal_move(&position, &record.first_move, path, line)?;
    if !position.is_check_move(first)
        || simple_see(&position, first) >= 0
        || simple_see(&position, first) != record.simple_see
    {
        return Err(record_context(
            path,
            line,
            "first move is not the declared negative-SEE check",
        ));
    }
    position.do_move(first);
    if position.legal_moves().len() != record.root_defense_count {
        return Err(record_context(path, line, "root defense count mismatch"));
    }
    let mut oracle_position = position.clone();
    let mut oracle = MateOracle::new(node_limit);
    if !matches!(
        oracle.prove(
            &mut oracle_position,
            attacker,
            record.mate_horizon.saturating_sub(1)
        ),
        MateProof::ProvenMate(_)
    ) {
        return Err(record_context(
            path,
            line,
            "mate oracle did not re-prove record",
        ));
    }
    if oracle.nodes() != record.proof_nodes {
        return Err(record_context(path, line, "mate proof node count mismatch"));
    }
    for text in record.proof_line.iter().skip(1) {
        let mv = parse_legal_move(&position, text, path, line)?;
        position.do_move(mv);
    }
    if !position.in_check() || !position.legal_moves().is_empty() {
        return Err(record_context(
            path,
            line,
            "mate proof line is not terminal mate",
        ));
    }
    Ok(())
}

fn quiet_move(position: &Position, mv: Move) -> bool {
    let capture = match mv {
        Move::Normal { to, .. } => position.piece_at(to).is_some(),
        Move::Drop { .. } => false,
    };
    !capture && !position.is_check_move(mv)
}

fn exact_move_set(actual: &[String], expected: &[String]) -> bool {
    let actual_set: HashSet<_> = actual.iter().collect();
    let expected_set: HashSet<_> = expected.iter().collect();
    actual.len() == actual_set.len() && actual_set == expected_set
}

fn validate_quiet_record(
    record: QuietRecord,
    pool: &SourcePool,
    path: &Path,
    line: usize,
) -> Result<()> {
    if record.schema_version != 1
        || record.legal_evasion_count != record.legal_evasions.len()
        || record.quiet_evasion_count != record.quiet_evasions.len()
        || record.quiet_evasions.is_empty()
    {
        return Err(record_context(
            path,
            line,
            "invalid quiet-evasion record fields",
        ));
    }
    let position = validate_source_identity(
        pool,
        record.source_index,
        record.source_game_key.as_deref(),
        &record.sfen,
        path,
        line,
    )?;
    if !position.in_check() {
        return Err(record_context(path, line, "evasion source is not in check"));
    }
    let legal: Vec<_> = position
        .legal_moves()
        .iter()
        .copied()
        .map(format_move_usi)
        .collect();
    let quiet: Vec<_> = position
        .legal_moves()
        .iter()
        .copied()
        .filter(|&mv| quiet_move(&position, mv))
        .map(format_move_usi)
        .collect();
    if !exact_move_set(&record.legal_evasions, &legal)
        || !exact_move_set(&record.quiet_evasions, &quiet)
    {
        return Err(record_context(path, line, "evasion move set mismatch"));
    }
    Ok(())
}

fn componentwise_resource_loss(start: &Position, end: &Position) -> Option<Color> {
    if start.keys().0 != end.keys().0 {
        return None;
    }
    for loser in [Color::Black, Color::White] {
        let old_own = hand_counts(start, loser);
        let new_own = hand_counts(end, loser);
        let old_opp = hand_counts(start, loser.flip());
        let new_opp = hand_counts(end, loser.flip());
        let own_subset = new_own.iter().zip(old_own).all(|(new, old)| *new <= old);
        let opp_superset = new_opp.iter().zip(old_opp).all(|(new, old)| *new >= old);
        let own_strict = new_own.iter().zip(old_own).any(|(new, old)| *new < old);
        let opp_strict = new_opp.iter().zip(old_opp).any(|(new, old)| *new > old);
        if own_subset && opp_superset && own_strict && opp_strict {
            return Some(loser);
        }
    }
    None
}

fn validate_resource_record(
    record: ResourceRecord,
    pool: &SourcePool,
    node_limit: u64,
    path: &Path,
    line: usize,
) -> Result<()> {
    let loser = match record.loser.as_str() {
        "black" => Color::Black,
        "white" => Color::White,
        _ => return Err(record_context(path, line, "invalid resource loser")),
    };
    if record.schema_version != 1
        || record.proof_status != "proven_legal_resource_loss_cycle"
        || record.horizon == 0
        || record.proof_line.is_empty()
        || record.cycle_length != record.proof_line.len()
        || record.gave_check.len() != record.proof_line.len()
        || record.source_to_cycle_start.len() + record.cycle_length > usize::from(record.horizon)
        || record.proof_nodes == 0
        || record.proof_nodes > node_limit
    {
        return Err(record_context(
            path,
            line,
            "invalid resource-cycle record fields",
        ));
    }
    let mut source = validate_source_identity(
        pool,
        record.source_index,
        record.source_game_key.as_deref(),
        &record.source_sfen,
        path,
        line,
    )?;
    for text in &record.source_to_cycle_start {
        let mv = parse_legal_move(&source, text, path, line)?;
        source.do_move(mv);
    }
    let declared_start = position_from_sfen_or_usi(&record.cycle_start_sfen)
        .ok_or_else(|| record_context(path, line, "invalid cycle-start SFEN"))?;
    if canonical_sfen(&source) != canonical_sfen(&declared_start) {
        return Err(record_context(
            path,
            line,
            "cycle start is not linked to source",
        ));
    }
    if hand_counts(&source, Color::Black) != record.start_black_hand
        || hand_counts(&source, Color::White) != record.start_white_hand
    {
        return Err(record_context(
            path,
            line,
            "cycle start hand counts mismatch",
        ));
    }
    let cycle_start = source.clone();
    for (text, &declared_check) in record.proof_line.iter().zip(&record.gave_check) {
        let mv = parse_legal_move(&source, text, path, line)?;
        if source.is_check_move(mv) != declared_check {
            return Err(record_context(path, line, "gave_check proof mismatch"));
        }
        source.do_move(mv);
    }
    if hand_counts(&source, Color::Black) != record.final_black_hand
        || hand_counts(&source, Color::White) != record.final_white_hand
        || componentwise_resource_loss(&cycle_start, &source) != Some(loser)
    {
        return Err(record_context(
            path,
            line,
            "resource dominance proof mismatch",
        ));
    }
    Ok(())
}

fn validate_suite(
    suite_path: &Path,
    source_pools: &HashMap<String, SourcePool>,
    generator_source_hash: &str,
    allow_dirty: bool,
) -> Result<(SuiteManifest, usize)> {
    let sidecar_path = suite_path.with_extension("manifest.json");
    if !sidecar_path.is_file() {
        return Err(anyhow!(
            "missing required sidecar manifest: {}",
            sidecar_path.display()
        ));
    }
    let sidecar: SuiteManifest =
        serde_json::from_reader(BufReader::new(File::open(&sidecar_path)?))
            .with_context(|| format!("invalid sidecar schema: {}", sidecar_path.display()))?;
    if sidecar.schema_version != 1
        || sidecar.generator != expected_generator(sidecar.suite_kind)
        || sidecar.weight_sha256.is_some()
        || !sidecar.filters.is_object()
    {
        return Err(anyhow!(
            "invalid sidecar fields: {}",
            sidecar_path.display()
        ));
    }
    if sidecar.generator_worktree_dirty && !allow_dirty {
        return Err(anyhow!(
            "sidecar was generated from a dirty worktree: {}",
            sidecar_path.display()
        ));
    }
    if sidecar.generator_source_sha256 != generator_source_hash {
        return Err(anyhow!(
            "sidecar generator source hash does not match current generator tree"
        ));
    }
    for key in required_filter_keys(sidecar.suite_kind) {
        if sidecar.filters.get(key).is_none() {
            return Err(anyhow!("sidecar filters missing required key {key}"));
        }
    }
    let pool = source_pools.get(&sidecar.input_sha256).ok_or_else(|| {
        anyhow!(
            "sidecar input SHA-256 is not one of --source-files: {}",
            sidecar.input_sha256
        )
    })?;
    if sidecar.input_nonempty_lines != pool.nonempty
        || sidecar.valid_positions != pool.valid
        || sidecar.duplicate_sfens_skipped != pool.duplicates
        || pool.valid == 0
    {
        return Err(anyhow!("sidecar input/dedupe counts disagree with source"));
    }
    if !same_file(&pool.path, Path::new(&sidecar.input_path))? {
        return Err(anyhow!(
            "sidecar input_path does not identify its source file"
        ));
    }
    if !same_file(suite_path, Path::new(&sidecar.output_path))? {
        return Err(anyhow!(
            "sidecar output_path does not identify its suite file"
        ));
    }
    if sha256_file(suite_path)? != sidecar.output_sha256 {
        return Err(anyhow!("suite SHA-256 disagrees with sidecar"));
    }

    let node_limit = match sidecar.suite_kind {
        SuiteKind::MateSacrifice => sidecar.filters["proof_node_limit"].as_u64(),
        SuiteKind::ResourceCycle => sidecar.filters["node_limit"].as_u64(),
        SuiteKind::QuietEvasion => Some(0),
    }
    .ok_or_else(|| anyhow!("sidecar proof node limit must be an unsigned integer"))?;
    let mut records = 0usize;
    let mut suite_sfens = HashSet::new();
    for (index, line) in BufReader::new(File::open(suite_path)?).lines().enumerate() {
        let line_number = index + 1;
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let canonical = match sidecar.suite_kind {
            SuiteKind::MateSacrifice => {
                let record: MateRecord = serde_json::from_str(&line).with_context(|| {
                    format!(
                        "invalid mate record at {}:{line_number}",
                        suite_path.display()
                    )
                })?;
                let canonical = canonical_sfen(
                    &position_from_sfen_or_usi(&record.sfen)
                        .ok_or_else(|| record_context(suite_path, line_number, "invalid SFEN"))?,
                );
                validate_mate_record(record, pool, node_limit, suite_path, line_number)?;
                canonical
            }
            SuiteKind::QuietEvasion => {
                let record: QuietRecord = serde_json::from_str(&line).with_context(|| {
                    format!(
                        "invalid quiet record at {}:{line_number}",
                        suite_path.display()
                    )
                })?;
                let canonical = canonical_sfen(
                    &position_from_sfen_or_usi(&record.sfen)
                        .ok_or_else(|| record_context(suite_path, line_number, "invalid SFEN"))?,
                );
                validate_quiet_record(record, pool, suite_path, line_number)?;
                canonical
            }
            SuiteKind::ResourceCycle => {
                let record: ResourceRecord = serde_json::from_str(&line).with_context(|| {
                    format!(
                        "invalid resource record at {}:{line_number}",
                        suite_path.display()
                    )
                })?;
                let canonical = canonical_sfen(
                    &position_from_sfen_or_usi(&record.source_sfen)
                        .ok_or_else(|| record_context(suite_path, line_number, "invalid SFEN"))?,
                );
                validate_resource_record(record, pool, node_limit, suite_path, line_number)?;
                canonical
            }
        };
        if !suite_sfens.insert(canonical) {
            return Err(anyhow!(
                "duplicate source SFEN inside {}",
                suite_path.display()
            ));
        }
        records += 1;
    }
    if records == 0 || records != sidecar.records_written {
        return Err(anyhow!(
            "suite record count {} disagrees with sidecar {}",
            records,
            sidecar.records_written
        ));
    }
    Ok((sidecar, records))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut labeled_paths: Vec<(String, PathBuf)> = vec![
        ("output".to_string(), args.output.clone()),
        ("weight".to_string(), args.weight.clone()),
    ];
    for (index, path) in args.source_files.iter().enumerate() {
        labeled_paths.push((format!("source[{index}]"), path.clone()));
    }
    for (index, path) in args.suite_files.iter().enumerate() {
        labeled_paths.push((format!("suite[{index}]"), path.clone()));
        labeled_paths.push((
            format!("sidecar[{index}]"),
            path.with_extension("manifest.json"),
        ));
    }
    let path_refs: Vec<_> = labeled_paths
        .iter()
        .map(|(label, path)| (label.as_str(), path.as_path()))
        .collect();
    ensure_distinct_paths(&path_refs)?;

    let dirty = generator_worktree_dirty()?;
    if dirty && !args.allow_dirty {
        return Err(anyhow!(
            "refusing to freeze from a dirty worktree; commit first or use --allow-dirty for smoke validation"
        ));
    }
    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let generator_source_hash = generator_source_sha256()?;

    let mut source_pools = HashMap::new();
    for source in &args.source_files {
        let (sha, pool) = load_source_pool(source)?;
        if source_pools.insert(sha.clone(), pool).is_some() {
            return Err(anyhow!("duplicate source file content SHA-256: {sha}"));
        }
    }

    let mut counts = HashMap::new();
    let mut combinations = HashSet::new();
    let mut source_splits: HashMap<String, DatasetSplit> = HashMap::new();
    let mut suite_entries = Vec::new();
    let mut all_sidecars_clean = true;
    for suite_path in &args.suite_files {
        let (sidecar, records) = validate_suite(
            suite_path,
            &source_pools,
            &generator_source_hash,
            args.allow_dirty,
        )?;
        if !combinations.insert((sidecar.suite_kind, sidecar.split)) {
            return Err(anyhow!("duplicate suite kind/split combination"));
        }
        if let Some(previous) = source_splits.insert(sidecar.input_sha256.clone(), sidecar.split) {
            if previous != sidecar.split {
                return Err(anyhow!(
                    "one source pool is assigned to both dev and holdout"
                ));
            }
        }
        all_sidecars_clean &= !sidecar.generator_worktree_dirty;
        counts.insert((sidecar.suite_kind, sidecar.split), records);
        let sidecar_path = suite_path.with_extension("manifest.json");
        suite_entries.push(file_entry(suite_path, Some(records), Some(&sidecar_path))?);
    }
    if source_splits.len() != source_pools.len() {
        return Err(anyhow!(
            "every supplied source pool must be referenced by a suite sidecar"
        ));
    }

    let mut dev_games = HashSet::new();
    let mut holdout_games = HashSet::new();
    let mut dev_sfens = HashSet::new();
    let mut holdout_sfens = HashSet::new();
    for (sha, pool) in &source_pools {
        match source_splits.get(sha).copied().expect("classified source") {
            DatasetSplit::Dev => {
                dev_games.extend(pool.games.iter().cloned());
                dev_sfens.extend(pool.sfens.iter().cloned());
            }
            DatasetSplit::Holdout => {
                holdout_games.extend(pool.games.iter().cloned());
                holdout_sfens.extend(pool.sfens.iter().cloned());
            }
        }
    }
    let source_game_intersection = dev_games.intersection(&holdout_games).count();
    let sfen_intersection = dev_sfens.intersection(&holdout_sfens).count();
    if source_game_intersection > 0 || sfen_intersection > 0 {
        return Err(anyhow!(
            "dev/holdout full source-pool intersection: source_games={source_game_intersection}, sfens={sfen_intersection}"
        ));
    }

    let mut targets = Vec::new();
    let mut targets_passed = true;
    for kind in [
        SuiteKind::MateSacrifice,
        SuiteKind::QuietEvasion,
        SuiteKind::ResourceCycle,
    ] {
        for split in [DatasetSplit::Dev, DatasetSplit::Holdout] {
            let actual = counts.get(&(kind, split)).copied().unwrap_or(0);
            let required = target(kind, split);
            let passed = actual >= required;
            targets_passed &= passed;
            targets.push(TargetResult {
                suite_kind: kind,
                split,
                actual,
                required,
                passed,
            });
        }
    }
    if !targets_passed && !args.allow_incomplete {
        let failures = targets
            .iter()
            .filter(|target| !target.passed)
            .map(|target| {
                format!(
                    "{:?}/{:?}={}/{}",
                    target.suite_kind, target.split, target.actual, target.required
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Err(anyhow!("suite target counts not met: {failures}"));
    }

    let freeze_eligible = is_freeze_eligible(
        args.allow_dirty,
        args.allow_incomplete,
        dirty,
        all_sidecars_clean,
        targets_passed,
        source_game_intersection,
        sfen_intersection,
    );
    let manifest = AggregateManifest {
        schema_version: 1,
        generator: "search_suite_manifest",
        generator_commit: commit,
        generator_worktree_dirty: dirty,
        generator_source_sha256: generator_source_hash,
        all_sidecars_clean,
        freeze_eligible,
        allow_incomplete: args.allow_incomplete,
        allow_dirty: args.allow_dirty,
        source_game_intersection,
        sfen_intersection,
        targets,
        weight: file_entry(&args.weight, None, None)?,
        source_files: args
            .source_files
            .iter()
            .map(|path| file_entry(path, None, None))
            .collect::<Result<_>>()?,
        suite_files: suite_entries,
    };
    let mut writer = AtomicOutput::new(&args.output)?;
    serde_json::to_writer_pretty(&mut writer, &manifest)?;
    writeln!(writer)?;
    writer.commit()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_pool(sfen: &str) -> SourcePool {
        let position = position_from_sfen_or_usi(sfen).expect("fixture SFEN");
        SourcePool {
            path: PathBuf::new(),
            nonempty: 1,
            valid: 1,
            duplicates: 0,
            rows: HashMap::from([(
                1,
                SourceRow {
                    game_key: "game-1".to_string(),
                    canonical_sfen: canonical_sfen(&position),
                },
            )]),
            games: HashSet::from(["game-1".to_string()]),
            sfens: HashSet::from([canonical_sfen(&position)]),
        }
    }

    #[test]
    fn exact_source_row_rejects_tampered_game_key() {
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        assert!(validate_source_identity(
            &source_pool(sfen),
            1,
            Some("other-game"),
            sfen,
            Path::new("suite"),
            1,
        )
        .is_err());
    }

    #[test]
    fn quiet_validation_rejects_incomplete_move_set() {
        let sfen = "9/7pp/8k/7PP/7G1/9/9/9/K8 w 2r2b3g4s4n4l14p 2";
        let record = QuietRecord {
            schema_version: 1,
            source_index: 1,
            source_game_key: Some("game-1".to_string()),
            sfen: sfen.to_string(),
            legal_evasions: Vec::new(),
            quiet_evasions: vec!["1a1b".to_string()],
            legal_evasion_count: 0,
            quiet_evasion_count: 1,
        };
        assert!(validate_quiet_record(record, &source_pool(sfen), Path::new("suite"), 1).is_err());
    }

    #[test]
    fn mate_validation_rejects_a_tampered_proof_move() {
        let sfen = "6+Bnl/3G2s2/p2p1pkpp/5lg2/3s+R1N2/P4NpPP/1P1PP4/2GSG1+l2/L2K3N1 b RS2Pb4p 133";
        let record = MateRecord {
            schema_version: 1,
            source_index: 1,
            source_game_key: Some("game-1".to_string()),
            sfen: sfen.to_string(),
            first_move: "3e2c+".to_string(),
            simple_see: -200,
            mate_horizon: 5,
            proof_line: ["3e2c+", "3c2c", "R*2b", "2c1d", "1f1e"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            proof_line_plies: 5,
            root_defense_count: 2,
            proof_nodes: 193,
            proof_status: "proven_mate".to_string(),
        };
        validate_mate_record(
            record.clone(),
            &source_pool(sfen),
            10_000,
            Path::new("suite"),
            1,
        )
        .expect("untampered mate proof");
        let mut tampered = record;
        tampered.proof_line[4] = "1f1d".to_string();
        assert!(
            validate_mate_record(tampered, &source_pool(sfen), 10_000, Path::new("suite"), 1,)
                .is_err()
        );
    }

    #[test]
    fn resource_validation_rejects_tampered_dominance() {
        let sfen = "4k4/9/9/4r4/9/9/9/9/4K4 b GP 1";
        let record = ResourceRecord {
            schema_version: 1,
            source_index: 1,
            source_game_key: Some("game-1".to_string()),
            source_sfen: sfen.to_string(),
            cycle_start_sfen: sfen.to_string(),
            source_to_cycle_start: Vec::new(),
            loser: "black".to_string(),
            horizon: 4,
            cycle_length: 4,
            proof_line: ["P*5e", "5d1d", "5e5d", "1d5d"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            gave_check: vec![false, false, false, true],
            start_black_hand: [1, 0, 0, 0, 1, 0, 0],
            final_black_hand: [0, 0, 0, 0, 1, 0, 0],
            start_white_hand: [0; 7],
            final_white_hand: [1, 0, 0, 0, 0, 0, 0],
            proof_nodes: 647,
            proof_status: "proven_legal_resource_loss_cycle".to_string(),
        };
        validate_resource_record(
            record.clone(),
            &source_pool(sfen),
            10_000,
            Path::new("suite"),
            1,
        )
        .expect("untampered resource proof");
        let mut tampered = record;
        tampered.final_white_hand = [0; 7];
        assert!(validate_resource_record(
            tampered,
            &source_pool(sfen),
            10_000,
            Path::new("suite"),
            1,
        )
        .is_err());
    }

    #[test]
    fn freeze_eligibility_rejects_every_relaxation_or_dirty_input() {
        assert!(is_freeze_eligible(false, false, false, true, true, 0, 0));
        assert!(!is_freeze_eligible(true, false, false, true, true, 0, 0));
        assert!(!is_freeze_eligible(false, true, false, true, true, 0, 0));
        assert!(!is_freeze_eligible(false, false, true, true, true, 0, 0));
        assert!(!is_freeze_eligible(false, false, false, false, true, 0, 0));
        assert!(!is_freeze_eligible(false, false, false, true, false, 0, 0));
        assert!(!is_freeze_eligible(false, false, false, true, true, 1, 0));
        assert!(!is_freeze_eligible(false, false, false, true, true, 0, 1));
    }
}
