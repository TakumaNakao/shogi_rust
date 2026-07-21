use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use shogi_ai::training_data::{
    capture_run_environment, collect_csa_files, csa_color, game_content_id, parse_csa_metadata,
    parse_csa_move, sha256_file, sha256_hex, split_for_game_content, CsaMetadata,
    DatasetSplit as Split, RunEnvironment, TrainingPhase as Phase, PHASE_POLICY_VERSION,
    SPLIT_POLICY_VERSION,
};
use shogi_ai::utils::format_move_usi;
use shogi_core::Color;
use shogi_lib::Position;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Parser, Debug)]
#[command(about = "Build streaming rank/value dataset v2 JSONL from CSA games")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = false)]
    reuse_if_matches: bool,
    #[arg(long, default_value_t = 9601)]
    seed: u64,
    #[arg(long, default_value_t = false)]
    shuffle_games: bool,
    #[arg(long, default_value_t = 5)]
    valid_percent: u8,
    #[arg(long, default_value_t = 5)]
    test_percent: u8,
    #[arg(long)]
    max_games: Option<usize>,
    #[arg(long)]
    max_records: Option<usize>,
    #[arg(long, default_value_t = 0)]
    target_opening_records: usize,
    #[arg(long, default_value_t = 0)]
    target_middle_records: usize,
    #[arg(long, default_value_t = 0)]
    target_late_records: usize,
    #[arg(long, default_value_t = false)]
    require_targets: bool,
    #[arg(long, default_value_t = 100)]
    target_minimum_percent: u8,
    #[arg(long)]
    max_records_per_game: Option<usize>,
    #[arg(long, value_delimiter = ',', num_args = 3)]
    phase_records_per_game: Option<Vec<usize>>,
    #[arg(long, default_value_t = 1)]
    sample_every: usize,
    #[arg(long, default_value_t = 0)]
    min_ply: usize,
    #[arg(long)]
    max_ply: Option<usize>,
    #[arg(long)]
    min_player_rate: Option<i32>,
    #[arg(long)]
    min_opponent_rate: Option<i32>,
    #[arg(long, default_value_t = false)]
    known_result_only: bool,
    #[arg(long, default_value_t = false)]
    winner_only: bool,
    #[arg(long, default_value_t = false)]
    decisive_only: bool,
    #[arg(long)]
    exclude_loser_after_ply: Option<usize>,
    #[arg(long, default_value_t = 0)]
    min_legal_moves: usize,
    #[arg(long, default_value_t = false)]
    exclude_in_check: bool,
}

#[derive(Serialize)]
struct DatasetRecord {
    schema: &'static str,
    source: String,
    game_index: usize,
    game_key: String,
    split: &'static str,
    ply: usize,
    phase: &'static str,
    side_to_move: &'static str,
    player_rate: Option<i32>,
    opponent_rate: Option<i32>,
    winner: Option<&'static str>,
    result_known: bool,
    termination: &'static str,
    is_winner_move: Option<bool>,
    legal_moves: usize,
    in_check: bool,
    sfen: String,
    teacher_move: String,
}

#[derive(Default, Serialize)]
struct SplitCounts {
    games: usize,
    records: usize,
}

#[derive(Default, Serialize)]
struct PhaseCounts {
    opening: usize,
    middle: usize,
    late: usize,
}

#[derive(Default, Serialize)]
struct DatasetStats {
    games_seen: usize,
    games_used: usize,
    games_skipped_parse: usize,
    games_duplicate_content: usize,
    games_filtered: usize,
    records_written: usize,
    records_filtered: usize,
    phase: PhaseCounts,
    train: SplitCounts,
    valid: SplitCounts,
    test: SplitCounts,
}

#[derive(Serialize)]
struct DatasetManifest {
    schema_version: u32,
    schema: &'static str,
    stage_fingerprint: String,
    environment: RunEnvironment,
    created_unix_secs: u64,
    git_rev: Option<String>,
    command: Vec<String>,
    input: Vec<String>,
    output_dir: String,
    seed: u64,
    phase_policy_version: u32,
    split_policy_version: u32,
    shuffle_games: bool,
    valid_percent: u8,
    test_percent: u8,
    max_games: Option<usize>,
    max_records: Option<usize>,
    target_opening_records: usize,
    target_middle_records: usize,
    target_late_records: usize,
    require_targets: bool,
    target_minimum_percent: u8,
    max_records_per_game: Option<usize>,
    phase_records_per_game: Option<Vec<usize>>,
    sample_every: usize,
    min_ply: usize,
    max_ply: Option<usize>,
    min_player_rate: Option<i32>,
    min_opponent_rate: Option<i32>,
    known_result_only: bool,
    winner_only: bool,
    decisive_only: bool,
    exclude_loser_after_ply: Option<usize>,
    min_legal_moves: usize,
    exclude_in_check: bool,
    input_files: Vec<InputFileManifest>,
    output_files: Vec<OutputFileManifest>,
    stats: DatasetStats,
}

#[derive(Serialize)]
struct InputFileManifest {
    path: String,
    sha256: String,
    bytes: u64,
}

#[derive(Serialize)]
struct OutputFileManifest {
    path: String,
    sha256: String,
    records: usize,
}

struct GameSource {
    path: PathBuf,
    game_id: String,
    bytes: u64,
}

struct Writers {
    train: BufWriter<File>,
    valid: BufWriter<File>,
    test: BufWriter<File>,
}

fn color_text(color: Color) -> &'static str {
    match color {
        Color::Black => "black",
        Color::White => "white",
    }
}

fn phase_text(ply: usize) -> &'static str {
    Phase::for_ply(ply).as_str()
}

fn phase_for_ply(ply: usize) -> Phase {
    Phase::for_ply(ply)
}

fn phase_target(args: &Args, phase: Phase) -> usize {
    match phase {
        Phase::Opening => args.target_opening_records,
        Phase::Middlegame => args.target_middle_records,
        Phase::Endgame => args.target_late_records,
    }
}

fn phase_count(stats: &DatasetStats, phase: Phase) -> usize {
    match phase {
        Phase::Opening => stats.phase.opening,
        Phase::Middlegame => stats.phase.middle,
        Phase::Endgame => stats.phase.late,
    }
}

fn increment_phase_count(stats: &mut DatasetStats, phase: Phase) {
    match phase {
        Phase::Opening => stats.phase.opening += 1,
        Phase::Middlegame => stats.phase.middle += 1,
        Phase::Endgame => stats.phase.late += 1,
    }
}

fn has_phase_targets(args: &Args) -> bool {
    args.target_opening_records > 0
        || args.target_middle_records > 0
        || args.target_late_records > 0
}

fn phase_target_reached(args: &Args, stats: &DatasetStats, phase: Phase) -> bool {
    let target = phase_target(args, phase);
    target > 0 && phase_count(stats, phase) >= target
}

fn all_phase_targets_reached(args: &Args, stats: &DatasetStats) -> bool {
    if !has_phase_targets(args) {
        return false;
    }
    [Phase::Opening, Phase::Middlegame, Phase::Endgame]
        .into_iter()
        .all(|phase| {
            let target = phase_target(args, phase);
            target == 0 || phase_count(stats, phase) >= target
        })
}

fn phase_targets_meet_minimum(args: &Args, stats: &DatasetStats) -> bool {
    [Phase::Opening, Phase::Middlegame, Phase::Endgame]
        .into_iter()
        .all(|phase| {
            let target = phase_target(args, phase);
            target == 0
                || phase_count(stats, phase) * 100
                    >= target * usize::from(args.target_minimum_percent)
        })
}

fn player_rate(metadata: &CsaMetadata, color: Color) -> Option<i32> {
    metadata.player_rate(color)
}

fn stable_hash(seed: u64, text: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64 ^ seed;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn split_text(split: Split) -> &'static str {
    split.as_str()
}

fn should_include_move(
    ply: usize,
    color: Color,
    metadata: &CsaMetadata,
    args: &Args,
    legal_moves: usize,
    in_check: bool,
) -> bool {
    if ply < args.min_ply {
        return false;
    }
    if args.max_ply.is_some_and(|max_ply| ply > max_ply) {
        return false;
    }
    if args.exclude_in_check && in_check {
        return false;
    }
    if args.min_legal_moves > 0 && legal_moves < args.min_legal_moves {
        return false;
    }
    if args.decisive_only && metadata.winner.is_none() {
        return false;
    }
    if args.known_result_only && !metadata.result_known {
        return false;
    }
    if args.winner_only && metadata.winner != Some(color) {
        return false;
    }
    if args.exclude_loser_after_ply.is_some_and(|min_ply| {
        ply >= min_ply && metadata.winner.is_some_and(|winner| winner != color)
    }) {
        return false;
    }
    if let Some(min_rate) = args.min_player_rate {
        if player_rate(metadata, color).is_none_or(|rate| rate < min_rate) {
            return false;
        }
    }
    if let Some(min_rate) = args.min_opponent_rate {
        if player_rate(metadata, color.flip()).is_none_or(|rate| rate < min_rate) {
            return false;
        }
    }
    true
}

fn create_writers(output_dir: &Path) -> Result<Writers> {
    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    Ok(Writers {
        train: BufWriter::new(File::create(output_dir.join("train.jsonl"))?),
        valid: BufWriter::new(File::create(output_dir.join("valid.jsonl"))?),
        test: BufWriter::new(File::create(output_dir.join("test.jsonl"))?),
    })
}

fn write_record(writers: &mut Writers, split: Split, line: &str) -> Result<()> {
    match split {
        Split::Train => writeln!(writers.train, "{line}")?,
        Split::Valid => writeln!(writers.valid, "{line}")?,
        Split::Test => writeln!(writers.test, "{line}")?,
    }
    Ok(())
}

fn flush_writers(writers: &mut Writers) -> Result<()> {
    writers.train.flush()?;
    writers.valid.flush()?;
    writers.test.flush()?;
    Ok(())
}

fn increment_split_games(stats: &mut DatasetStats, split: Split) {
    match split {
        Split::Train => stats.train.games += 1,
        Split::Valid => stats.valid.games += 1,
        Split::Test => stats.test.games += 1,
    }
}

fn increment_split_records(stats: &mut DatasetStats, split: Split) {
    match split {
        Split::Train => stats.train.records += 1,
        Split::Valid => stats.valid.records += 1,
        Split::Test => stats.test.records += 1,
    }
}

fn process_game(
    game_index: usize,
    bytes: &[u8],
    game_id: &str,
    split: Split,
    args: &Args,
    writers: &mut Writers,
    stats: &mut DatasetStats,
) -> Result<usize> {
    let text = std::str::from_utf8(bytes)?;
    let record = csa::parse_csa(&text)?;
    let metadata = parse_csa_metadata(&text, &record);
    if args.decisive_only && metadata.winner.is_none() {
        return Ok(0);
    }

    let mut position = Position::default();
    let mut eligible_seen = 0usize;
    let game_key = game_id.to_string();
    let mut candidates = Vec::<(Phase, String)>::new();

    for (ply_index, csa_move) in record.moves.iter().enumerate() {
        let ply = ply_index + 1;
        if args.max_ply.is_some_and(|max_ply| ply > max_ply) {
            break;
        }
        let Some(mv) = parse_csa_move(&position, &csa_move.action) else {
            break;
        };
        let legal_moves = position.legal_moves();
        if !legal_moves.contains(&mv) {
            break;
        }
        let move_color = match csa_move.action {
            csa::Action::Move(color, ..) => csa_color(color),
            _ => break,
        };
        let in_check = position.in_check();
        if should_include_move(
            ply,
            move_color,
            &metadata,
            args,
            legal_moves.len(),
            in_check,
        ) {
            eligible_seen += 1;
            if !(eligible_seen - 1).is_multiple_of(args.sample_every) {
                position.do_move(mv);
                continue;
            }
            let phase = phase_for_ply(ply);
            let opponent = move_color.flip();
            let is_winner_move = metadata.winner.map(|winner| winner == move_color);
            let out = DatasetRecord {
                schema: "rank_value_dataset_v2_position",
                source: format!("sha256:{game_id}"),
                game_index,
                game_key: game_key.clone(),
                split: split_text(split),
                ply,
                phase: phase_text(ply),
                side_to_move: color_text(move_color),
                player_rate: player_rate(&metadata, move_color),
                opponent_rate: player_rate(&metadata, opponent),
                winner: metadata.winner.map(color_text),
                result_known: metadata.result_known,
                termination: metadata.termination.as_str(),
                is_winner_move,
                legal_moves: legal_moves.len(),
                in_check,
                sfen: position.to_sfen_owned(),
                teacher_move: format_move_usi(mv),
            };
            let line = serde_json::to_string(&out)?;
            candidates.push((phase, line));
        } else if ply >= args.min_ply {
            stats.records_filtered += 1;
        }
        position.do_move(mv);
    }

    let candidate_count = candidates.len();
    let mut selected = select_game_records(candidates, args, &game_key);
    stats.records_filtered += candidate_count.saturating_sub(selected.len());
    let mut written = 0usize;
    for (phase, line) in selected.drain(..) {
        if all_phase_targets_reached(args, stats)
            || args
                .max_records
                .is_some_and(|limit| stats.records_written >= limit)
        {
            break;
        }
        if phase_target_reached(args, stats, phase) {
            stats.records_filtered += 1;
            continue;
        }
        write_record(writers, split, &line)?;
        increment_split_records(stats, split);
        increment_phase_count(stats, phase);
        stats.records_written += 1;
        written += 1;
    }
    Ok(written)
}

fn select_game_records(
    candidates: Vec<(Phase, String)>,
    args: &Args,
    game_key: &str,
) -> Vec<(Phase, String)> {
    let mut rng = ChaCha8Rng::seed_from_u64(stable_hash(args.seed, game_key));
    if let Some(caps) = args.phase_records_per_game.as_deref() {
        let mut phases = [Vec::new(), Vec::new(), Vec::new()];
        for candidate in candidates {
            let index = match candidate.0 {
                Phase::Opening => 0,
                Phase::Middlegame => 1,
                Phase::Endgame => 2,
            };
            phases[index].push(candidate);
        }
        let mut selected = Vec::new();
        for (records, &cap) in phases.iter_mut().zip(caps) {
            records.shuffle(&mut rng);
            selected.extend(records.drain(..records.len().min(cap)));
        }
        selected.shuffle(&mut rng);
        return selected;
    }
    let mut selected = candidates;
    selected.shuffle(&mut rng);
    if let Some(limit) = args.max_records_per_game {
        selected.truncate(limit);
    }
    selected
}

fn git_rev() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn dataset_stage_fingerprint(args: &Args, sources: &[GameSource]) -> Result<String> {
    let inputs = sources
        .iter()
        .map(|source| (&source.game_id, source.bytes))
        .collect::<Vec<_>>();
    let value = serde_json::json!({
        "schema_version": 1,
        "stage": "dataset_build",
        "inputs": inputs,
        "seed": args.seed,
        "phase_policy_version": PHASE_POLICY_VERSION,
        "split_policy_version": SPLIT_POLICY_VERSION,
        "configuration": {
            "shuffle_games": args.shuffle_games,
            "valid_percent": args.valid_percent,
            "test_percent": args.test_percent,
            "max_games": args.max_games,
            "max_records": args.max_records,
            "target_opening_records": args.target_opening_records,
            "target_middle_records": args.target_middle_records,
            "target_late_records": args.target_late_records,
            "require_targets": args.require_targets,
            "target_minimum_percent": args.target_minimum_percent,
            "max_records_per_game": args.max_records_per_game,
            "phase_records_per_game": args.phase_records_per_game,
            "sample_every": args.sample_every,
            "min_ply": args.min_ply,
            "max_ply": args.max_ply,
            "min_player_rate": args.min_player_rate,
            "min_opponent_rate": args.min_opponent_rate,
            "known_result_only": args.known_result_only,
            "winner_only": args.winner_only,
            "decisive_only": args.decisive_only,
            "exclude_loser_after_ply": args.exclude_loser_after_ply,
            "min_legal_moves": args.min_legal_moves,
            "exclude_in_check": args.exclude_in_check,
        }
    });
    Ok(sha256_hex(&serde_json::to_vec(&value)?))
}

fn reusable_output(output_dir: &Path, expected_fingerprint: &str) -> Result<bool> {
    let manifest_path = output_dir.join("manifest.json");
    let value: serde_json::Value = match fs::read(&manifest_path) {
        Ok(bytes) => serde_json::from_slice(&bytes)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    if value["stage_fingerprint"].as_str() != Some(expected_fingerprint) {
        return Ok(false);
    }
    let Some(outputs) = value["output_files"].as_array() else {
        return Ok(false);
    };
    for output in outputs {
        let (Some(path), Some(expected)) = (output["path"].as_str(), output["sha256"].as_str())
        else {
            return Ok(false);
        };
        if sha256_file(&output_dir.join(path))? != expected {
            return Ok(false);
        }
    }
    Ok(true)
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    if args.valid_percent + args.test_percent > 100 {
        return Err(anyhow!("--valid-percent + --test-percent must be <= 100"));
    }
    if args.sample_every == 0 {
        return Err(anyhow!("--sample-every must be greater than zero"));
    }
    if !(1..=100).contains(&args.target_minimum_percent) {
        return Err(anyhow!("--target-minimum-percent must be in 1..=100"));
    }
    if args
        .phase_records_per_game
        .as_ref()
        .is_some_and(|caps| caps.len() != 3 || caps.iter().all(|&cap| cap == 0))
    {
        return Err(anyhow!(
            "--phase-records-per-game requires three comma-separated values"
        ));
    }
    let files = collect_csa_files(&args.input)?;
    let mut stats = DatasetStats::default();
    let mut sources_by_id = BTreeMap::new();
    for path in files {
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(_) => {
                stats.games_skipped_parse += 1;
                continue;
            }
        };
        let game_id = game_content_id(&bytes);
        let source = GameSource {
            path,
            game_id: game_id.clone(),
            bytes: bytes.len() as u64,
        };
        if sources_by_id.insert(game_id, source).is_some() {
            stats.games_duplicate_content += 1;
        }
    }
    let mut sources = sources_by_id.into_values().collect::<Vec<_>>();
    let stage_fingerprint = dataset_stage_fingerprint(&args, &sources)?;
    if args.reuse_if_matches && reusable_output(&args.output_dir, &stage_fingerprint)? {
        println!("reused output_dir={}", args.output_dir.display());
        return Ok(());
    }
    if args.shuffle_games {
        let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
        sources.shuffle(&mut rng);
    }
    let input_files = sources
        .iter()
        .map(|source| InputFileManifest {
            path: source.path.to_string_lossy().to_string(),
            sha256: source.game_id.clone(),
            bytes: source.bytes,
        })
        .collect();
    let mut writers = create_writers(&args.output_dir)?;

    for (game_index, source) in sources.iter().enumerate() {
        if all_phase_targets_reached(&args, &stats) {
            break;
        }
        if args
            .max_games
            .is_some_and(|limit| stats.games_seen >= limit)
        {
            break;
        }
        if args
            .max_records
            .is_some_and(|limit| stats.records_written >= limit)
        {
            break;
        }
        stats.games_seen += 1;
        let bytes = match fs::read(&source.path) {
            Ok(bytes) => bytes,
            Err(_) => {
                stats.games_skipped_parse += 1;
                continue;
            }
        };
        let split = split_for_game_content(
            &source.game_id,
            args.seed,
            args.valid_percent,
            args.test_percent,
        );
        match process_game(
            game_index,
            &bytes,
            &source.game_id,
            split,
            &args,
            &mut writers,
            &mut stats,
        ) {
            Ok(0) => stats.games_filtered += 1,
            Ok(_) => {
                stats.games_used += 1;
                increment_split_games(&mut stats, split);
            }
            Err(_) => stats.games_skipped_parse += 1,
        }
    }
    flush_writers(&mut writers)?;
    let output_files = [
        ("train.jsonl", stats.train.records),
        ("valid.jsonl", stats.valid.records),
        ("test.jsonl", stats.test.records),
    ]
    .into_iter()
    .map(|(name, records)| {
        let path = args.output_dir.join(name);
        Ok(OutputFileManifest {
            path: name.to_string(),
            sha256: sha256_file(&path)?,
            records,
        })
    })
    .collect::<Result<Vec<_>>>()?;

    let manifest = DatasetManifest {
        schema_version: 3,
        schema: "rank_value_dataset_v3_manifest",
        stage_fingerprint,
        environment: capture_run_environment(),
        created_unix_secs: unix_secs(),
        git_rev: git_rev(),
        command: std::env::args().collect(),
        input: args
            .input
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect(),
        output_dir: args.output_dir.to_string_lossy().to_string(),
        seed: args.seed,
        phase_policy_version: PHASE_POLICY_VERSION,
        split_policy_version: SPLIT_POLICY_VERSION,
        shuffle_games: args.shuffle_games,
        valid_percent: args.valid_percent,
        test_percent: args.test_percent,
        max_games: args.max_games,
        max_records: args.max_records,
        target_opening_records: args.target_opening_records,
        target_middle_records: args.target_middle_records,
        target_late_records: args.target_late_records,
        require_targets: args.require_targets,
        target_minimum_percent: args.target_minimum_percent,
        max_records_per_game: args.max_records_per_game,
        phase_records_per_game: args.phase_records_per_game.clone(),
        sample_every: args.sample_every,
        min_ply: args.min_ply,
        max_ply: args.max_ply,
        min_player_rate: args.min_player_rate,
        min_opponent_rate: args.min_opponent_rate,
        known_result_only: args.known_result_only,
        winner_only: args.winner_only,
        decisive_only: args.decisive_only,
        exclude_loser_after_ply: args.exclude_loser_after_ply,
        min_legal_moves: args.min_legal_moves,
        exclude_in_check: args.exclude_in_check,
        input_files,
        output_files,
        stats,
    };
    let manifest_path = args.output_dir.join("manifest.json");
    let manifest_file = File::create(&manifest_path)
        .with_context(|| format!("failed to create {}", manifest_path.display()))?;
    serde_json::to_writer_pretty(BufWriter::new(manifest_file), &manifest)?;

    println!("manifest: {}", manifest_path.display());
    println!("games seen: {}", manifest.stats.games_seen);
    println!("games used: {}", manifest.stats.games_used);
    println!(
        "games skipped parse: {}",
        manifest.stats.games_skipped_parse
    );
    println!(
        "games duplicate content: {}",
        manifest.stats.games_duplicate_content
    );
    println!("games filtered: {}", manifest.stats.games_filtered);
    println!("records written: {}", manifest.stats.records_written);
    println!("records filtered: {}", manifest.stats.records_filtered);
    println!(
        "phase records: opening={} middle={} late={}",
        manifest.stats.phase.opening, manifest.stats.phase.middle, manifest.stats.phase.late
    );
    println!(
        "train: games={} records={}",
        manifest.stats.train.games, manifest.stats.train.records
    );
    println!(
        "valid: games={} records={}",
        manifest.stats.valid.games, manifest.stats.valid.records
    );
    println!(
        "test: games={} records={}",
        manifest.stats.test.games, manifest.stats.test.records
    );

    if args.require_targets && !phase_targets_meet_minimum(&args, &manifest.stats) {
        return Err(anyhow!(
            "phase targets did not reach {}%: opening={}/{} middle={}/{} late={}/{}",
            args.target_minimum_percent,
            manifest.stats.phase.opening,
            args.target_opening_records,
            manifest.stats.phase.middle,
            args.target_middle_records,
            manifest.stats.phase.late,
            args.target_late_records
        ));
    }
    Ok(())
}
