use anyhow::{anyhow, Context, Result};
use clap::{Args, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use shogi_ai::halfkp_training::PackedHalfKpPosition;
use shogi_ai::training_data::sha256_file;
use shogi_ai::utils::position_from_sfen_or_usi;
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Prepare deterministic WDL and dense HalfKP datasets")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    WdlPool(WdlPoolArgs),
    WdlValidSplit(WdlValidSplitArgs),
    ComposeInterleave(ComposeArgs),
    UnseenPilot(UnseenArgs),
}

#[derive(Args, Debug)]
struct WdlPoolArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    input_manifest: PathBuf,
    #[arg(long)]
    frozen_dataset_manifest: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long)]
    seed: u64,
    #[arg(long, default_value_t = 100_000)]
    max_roots: usize,
    #[arg(long, default_value_t = 1)]
    max_roots_per_game_phase: usize,
    #[arg(long, default_value_t = 0.25)]
    max_year_game_fraction: f64,
}

#[derive(Args, Debug)]
struct WdlValidSplitArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    input_manifest: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    calibration_games: usize,
    #[arg(long)]
    audit_games: usize,
}

#[derive(Args, Debug)]
struct ComposeArgs {
    #[arg(long)]
    membership_manifest: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long)]
    seed: u64,
    #[arg(long, default_value = "game_round_robin_weighted_fair_v1")]
    algorithm: String,
    #[arg(long, default_value_t = 3_000_000)]
    hard_max_train_roots: usize,
}

#[derive(Args, Debug)]
struct UnseenArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    input_manifest: PathBuf,
    #[arg(long)]
    exclude_network_inputs: PathBuf,
    #[arg(long)]
    frozen_valid: PathBuf,
    #[arg(long)]
    frozen_test: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long)]
    roots: usize,
    #[arg(long)]
    seed: u64,
    #[arg(long, default_value = "game,phase")]
    balance: String,
}

#[derive(Deserialize)]
struct LineRecord {
    game_key: String,
    #[serde(default)]
    phase: String,
    sfen: String,
}

struct Line {
    raw: String,
    game: String,
    phase: String,
    input_hash: [u8; 32],
    digest: [u8; 32],
}

#[derive(Serialize)]
struct PrepareManifest {
    schema: &'static str,
    algorithm: String,
    seed: u64,
    input_files: Vec<Artifact>,
    output_files: Vec<Artifact>,
    records: usize,
    games: usize,
}

#[derive(Serialize)]
struct Artifact {
    path: String,
    sha256: String,
    bytes: u64,
}

pub fn run() -> Result<()> {
    match Cli::parse().command {
        Command::WdlPool(args) => wdl_pool(args),
        Command::WdlValidSplit(args) => wdl_valid_split(args),
        Command::ComposeInterleave(args) => compose_interleave(args),
        Command::UnseenPilot(args) => unseen_pilot(args),
    }
}

fn load_lines(path: &Path, seed: u64) -> Result<Vec<Line>> {
    let mut result = Vec::new();
    for (line_no, raw) in BufReader::new(File::open(path)?).lines().enumerate() {
        let raw = raw.with_context(|| format!("read {}:{}", path.display(), line_no + 1))?;
        if raw.trim().is_empty() {
            continue;
        }
        let record: LineRecord = serde_json::from_str(&raw)
            .with_context(|| format!("parse {}:{}", path.display(), line_no + 1))?;
        let position = position_from_sfen_or_usi(&record.sfen)
            .ok_or_else(|| anyhow!("invalid SFEN at {}:{}", path.display(), line_no + 1))?;
        let packed = PackedHalfKpPosition::from_position(&position)
            .ok_or_else(|| anyhow!("position lacks king at {}:{}", path.display(), line_no + 1))?;
        let input_hash = packed.network_input_hash();
        let digest = digest(seed, &[&record.game_key, &record.phase, &record.sfen]);
        result.push(Line {
            raw,
            game: record.game_key,
            phase: record.phase,
            input_hash,
            digest,
        });
    }
    Ok(result)
}

fn wdl_pool(args: WdlPoolArgs) -> Result<()> {
    if args.max_roots == 0 || args.max_roots_per_game_phase == 0 {
        return Err(anyhow!("WDL pool limits must be positive"));
    }
    if args.max_roots_per_game_phase > 1 || !(0.0..=1.0).contains(&args.max_year_game_fraction) {
        return Err(anyhow!("unsupported WDL pool quota"));
    }
    let _ = sha256_file(&args.input_manifest)?;
    let _ = sha256_file(&args.frozen_dataset_manifest)?;
    let lines = load_lines(&args.input, args.seed)?;
    let mut groups: BTreeMap<String, Vec<Line>> = BTreeMap::new();
    for line in lines {
        groups.entry(line.game.clone()).or_default().push(line);
    }
    let mut selected = Vec::new();
    for (game, mut roots) in groups {
        roots.sort_by_key(|line| line.digest);
        let mut phases = HashSet::new();
        for line in roots {
            if phases.contains(&line.phase) || phases.len() >= 3 {
                continue;
            }
            phases.insert(line.phase.clone());
            selected.push((digest(args.seed, &[&game]), line));
        }
    }
    selected.sort_by_key(|(game_digest, line)| (*game_digest, line.digest));
    let mut seen = HashSet::new();
    selected.retain(|(_, line)| seen.insert(line.input_hash));
    selected.truncate(args.max_roots);
    let output = args.output_dir.join("train.jsonl");
    write_lines(&output, selected.iter().map(|(_, line)| line))?;
    write_manifest(
        &args.output_dir,
        "halfkp_wdl_pool_v1",
        "game_balanced_hash_stratified_v1",
        args.seed,
        &[(&args.input, &args.input_manifest)],
        &[&output],
        selected.len(),
        selected
            .iter()
            .map(|(_, line)| line.game.as_str())
            .collect::<HashSet<_>>()
            .len(),
    )
}

fn wdl_valid_split(args: WdlValidSplitArgs) -> Result<()> {
    let _ = sha256_file(&args.input_manifest)?;
    let mut lines = load_lines(&args.input, args.seed)?;
    let mut games = BTreeMap::<String, Vec<Line>>::new();
    for line in lines.drain(..) {
        games.entry(line.game.clone()).or_default().push(line);
    }
    let mut ordered = games.into_iter().collect::<Vec<_>>();
    ordered.sort_by_key(|(game, _)| digest(args.seed, &[game]));
    let calibration = ordered
        .iter()
        .take(args.calibration_games)
        .flat_map(|(_, lines)| lines);
    let audit = ordered
        .iter()
        .skip(args.calibration_games)
        .flat_map(|(_, lines)| lines);
    let calibration_path = args.output_dir.join("calibration.jsonl");
    let audit_path = args.output_dir.join("audit.jsonl");
    write_lines(&calibration_path, calibration)?;
    write_lines(&audit_path, audit)?;
    write_manifest(
        &args.output_dir,
        "halfkp_wdl_valid_split_v1",
        "game_hash_split_v1",
        args.seed,
        &[(&args.input, &args.input_manifest)],
        &[&calibration_path, &audit_path],
        args.calibration_games + args.audit_games,
        ordered.len(),
    )
}

fn compose_interleave(args: ComposeArgs) -> Result<()> {
    if args.algorithm != "game_round_robin_weighted_fair_v1" {
        return Err(anyhow!("unsupported compose algorithm"));
    }
    let value: serde_json::Value = serde_json::from_reader(File::open(&args.membership_manifest)?)?;
    let membership_train = manifest_path(&args.membership_manifest, &value, "train.jsonl")?;
    let membership_valid = manifest_path(&args.membership_manifest, &value, "valid.jsonl")?;
    let membership_test = manifest_path(&args.membership_manifest, &value, "test.jsonl")?;
    let mut supplement = load_lines(&membership_train, args.seed)?;
    let base_manifest = value["base_dataset_manifest"].as_str().map(PathBuf::from);
    let (base, valid, test) = if let Some(path) = base_manifest {
        let path = if path.is_absolute() {
            path
        } else {
            args.membership_manifest
                .parent()
                .unwrap_or(Path::new("."))
                .join(path)
        };
        let base_train = manifest_output_path(&path, "train.jsonl")?;
        let base_valid = manifest_output_path(&path, "valid.jsonl")?;
        let base_test = manifest_output_path(&path, "test.jsonl")?;
        (load_lines(&base_train, args.seed)?, base_valid, base_test)
    } else {
        (Vec::new(), membership_valid, membership_test)
    };
    let supplement_limit = args.hard_max_train_roots.saturating_sub(base.len());
    if supplement.len() > supplement_limit {
        supplement = phase_select(supplement, supplement_limit);
    }
    let base_stream = game_round_robin(base, args.seed);
    let supplement_stream = game_round_robin(supplement, args.seed);
    let merged = weighted_merge(base_stream, supplement_stream);
    let output_train = args.output_dir.join("train.jsonl");
    write_lines(&output_train, merged.iter())?;
    fs::create_dir_all(&args.output_dir)?;
    fs::copy(valid, args.output_dir.join("valid.jsonl"))?;
    fs::copy(test, args.output_dir.join("test.jsonl"))?;
    write_manifest(
        &args.output_dir,
        "rank_value_dataset_v6_composed_manifest",
        &args.algorithm,
        args.seed,
        &[(&args.membership_manifest, &args.membership_manifest)],
        &[
            &output_train,
            &args.output_dir.join("valid.jsonl"),
            &args.output_dir.join("test.jsonl"),
        ],
        merged.len(),
        merged
            .iter()
            .map(|line| line.game.as_str())
            .collect::<HashSet<_>>()
            .len(),
    )
}

fn unseen_pilot(args: UnseenArgs) -> Result<()> {
    if args.roots == 0 || args.balance != "game,phase" {
        return Err(anyhow!(
            "unseen pilot requires positive roots and game,phase balance"
        ));
    }
    let _ = sha256_file(&args.input_manifest)?;
    let excluded = load_input_hashes(&args.exclude_network_inputs, args.seed)?;
    let mut seen = HashSet::new();
    let mut lines = load_lines(&args.input, args.seed)?
        .into_iter()
        .filter(|line| !excluded.contains(&line.input_hash) && seen.insert(line.input_hash))
        .collect::<Vec<_>>();
    lines.sort_by_key(|line| line.digest);
    lines.truncate(args.roots);
    if lines.len() != args.roots {
        return Err(anyhow!("unseen pilot has only {} roots", lines.len()));
    }
    let train = args.output_dir.join("train.jsonl");
    write_lines(&train, lines.iter())?;
    fs::create_dir_all(&args.output_dir)?;
    fs::copy(&args.frozen_valid, args.output_dir.join("valid.jsonl"))?;
    fs::copy(&args.frozen_test, args.output_dir.join("test.jsonl"))?;
    write_manifest(
        &args.output_dir,
        "halfkp_unseen_pilot_v1",
        "game_hash_phase_v1",
        args.seed,
        &[
            (&args.input, &args.input_manifest),
            (&args.exclude_network_inputs, &args.exclude_network_inputs),
        ],
        &[
            &train,
            &args.output_dir.join("valid.jsonl"),
            &args.output_dir.join("test.jsonl"),
        ],
        lines.len(),
        lines
            .iter()
            .map(|line| line.game.as_str())
            .collect::<HashSet<_>>()
            .len(),
    )
}

fn phase_select(mut lines: Vec<Line>, limit: usize) -> Vec<Line> {
    let weights = [("opening", 2usize), ("middle", 3), ("late", 1)];
    let mut selected = Vec::new();
    for (phase, weight) in weights {
        let quota = limit.saturating_mul(weight) / 6;
        let mut candidates = lines
            .iter()
            .filter(|line| line.phase == phase)
            .map(|line| line.digest)
            .collect::<Vec<_>>();
        candidates.sort();
        for digest in candidates.into_iter().take(quota) {
            if let Some(index) = lines.iter().position(|line| line.digest == digest) {
                selected.push(lines.swap_remove(index));
            }
        }
    }
    selected.sort_by_key(|line| line.digest);
    if selected.len() < limit {
        lines.sort_by_key(|line| line.digest);
        selected.extend(lines.into_iter().take(limit - selected.len()));
    }
    selected
}

fn game_round_robin(lines: Vec<Line>, seed: u64) -> Vec<Line> {
    let mut games = BTreeMap::<String, Vec<Line>>::new();
    for line in lines {
        games.entry(line.game.clone()).or_default().push(line);
    }
    let mut games = games.into_iter().collect::<Vec<_>>();
    games.sort_by_key(|(game, _)| digest(seed, &[game]));
    for (_, lines) in &mut games {
        lines.sort_by_key(|line| line.digest);
    }
    let mut output = Vec::new();
    for round in 0.. {
        let mut emitted = false;
        for (_, lines) in &mut games {
            if let Some(line) = lines.get(round) {
                output.push(Line {
                    raw: line.raw.clone(),
                    game: line.game.clone(),
                    phase: line.phase.clone(),
                    input_hash: line.input_hash,
                    digest: line.digest,
                });
                emitted = true;
            }
        }
        if !emitted {
            break;
        }
    }
    output
}

fn weighted_merge(base: Vec<Line>, supplement: Vec<Line>) -> Vec<Line> {
    let total = base.len() + supplement.len();
    let mut output = Vec::with_capacity(total);
    let mut left = base.into_iter();
    let mut right = supplement.into_iter();
    let base_total = output_capacity(&left);
    let supplement_total = output_capacity(&right);
    let mut base_emitted = 0usize;
    let mut supplement_emitted = 0usize;
    let mut base_next = left.next();
    let mut supplement_next = right.next();
    for n in 0..total {
        let base_deficit = if base_next.is_some() {
            (n + 1) * base_total - base_emitted * total
        } else {
            usize::MIN
        };
        let supplement_deficit = if supplement_next.is_some() {
            (n + 1) * supplement_total - supplement_emitted * total
        } else {
            usize::MIN
        };
        if base_deficit >= supplement_deficit {
            output.push(base_next.take().expect("base stream"));
            base_emitted += 1;
            base_next = left.next();
        } else {
            output.push(supplement_next.take().expect("supplement stream"));
            supplement_emitted += 1;
            supplement_next = right.next();
        }
    }
    output
}

fn output_capacity(iter: &std::vec::IntoIter<Line>) -> usize {
    iter.len()
}

fn load_input_hashes(path: &Path, seed: u64) -> Result<HashSet<[u8; 32]>> {
    Ok(load_lines(path, seed)?
        .into_iter()
        .map(|line| line.input_hash)
        .collect())
}

fn digest(seed: u64, fields: &[&str]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"halfkp-dataset-prepare-v1\0");
    hasher.update(seed.to_le_bytes());
    for field in fields {
        hasher.update((*field).len().to_le_bytes());
        hasher.update(field.as_bytes());
    }
    hasher.finalize().into()
}

fn write_lines<'a>(path: &Path, lines: impl Iterator<Item = &'a Line>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(path)?);
    for line in lines {
        writeln!(writer, "{}", line.raw)?;
    }
    writer.flush()?;
    Ok(())
}

fn manifest_path(manifest_path: &Path, value: &serde_json::Value, name: &str) -> Result<PathBuf> {
    let path = value["output_files"]
        .as_array()
        .and_then(|files| files.iter().find(|file| file["path"] == name))
        .and_then(|file| file["path"].as_str())
        .ok_or_else(|| anyhow!("manifest lacks {name}"))?;
    let resolved = manifest_path.parent().unwrap_or(Path::new(".")).join(path);
    if !resolved.exists() {
        return Err(anyhow!("missing dataset output {}", resolved.display()));
    }
    Ok(resolved)
}

fn manifest_output_path(manifest_file: &Path, name: &str) -> Result<PathBuf> {
    let value: serde_json::Value = serde_json::from_reader(File::open(manifest_file)?)?;
    manifest_path(manifest_file, &value, name)
}

fn write_manifest(
    output_dir: &Path,
    schema: &'static str,
    algorithm: &str,
    seed: u64,
    inputs: &[(&Path, &Path)],
    outputs: &[&Path],
    records: usize,
    games: usize,
) -> Result<()> {
    let input_files = inputs
        .iter()
        .map(|(path, _manifest)| artifact(path))
        .collect::<Result<Vec<_>>>()?;
    let output_files = outputs
        .iter()
        .map(|path| artifact(path))
        .collect::<Result<Vec<_>>>()?;
    let manifest = PrepareManifest {
        schema,
        algorithm: algorithm.to_owned(),
        seed,
        input_files,
        output_files,
        records,
        games,
    };
    fs::create_dir_all(output_dir)?;
    let path = output_dir.join("manifest.json");
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut writer, &manifest)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn artifact(path: &Path) -> Result<Artifact> {
    Ok(Artifact {
        path: path.to_string_lossy().to_string(),
        sha256: sha256_file(path)?,
        bytes: fs::metadata(path)?.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(game: &str, phase: &str, value: u8) -> Line {
        Line {
            raw: format!("{{\"game_key\":\"{game}\",\"phase\":\"{phase}\"}}"),
            game: game.to_owned(),
            phase: phase.to_owned(),
            input_hash: [value; 32],
            digest: [value; 32],
        }
    }

    #[test]
    fn weighted_merge_preserves_all_layers() {
        let merged = weighted_merge(
            vec![line("base-a", "opening", 1), line("base-b", "middle", 2)],
            vec![line("new-a", "late", 3)],
        );
        assert_eq!(merged.len(), 3);
        assert_eq!(
            merged
                .iter()
                .filter(|line| line.game.starts_with("base"))
                .count(),
            2
        );
        assert_eq!(
            merged
                .iter()
                .filter(|line| line.game.starts_with("new"))
                .count(),
            1
        );
    }

    #[test]
    fn digest_is_stable_and_domain_separated() {
        assert_eq!(digest(7, &["g", "opening"]), digest(7, &["g", "opening"]));
        assert_ne!(digest(7, &["g", "opening"]), digest(8, &["g", "opening"]));
        assert_ne!(digest(7, &["g", "opening"]), digest(7, &["g", "late"]));
    }
}
