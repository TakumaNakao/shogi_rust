use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use shogi_core::{Color, Move, PieceKind};
use shogi_lib::Position;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::utils::get_piece_value;
use crate::utils::position_from_sfen_or_usi;

pub const HAND_KINDS: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
pub struct InputPosition {
    pub source_line: usize,
    pub source_game_key: Option<String>,
    pub position: Position,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SuiteKind {
    MateSacrifice,
    QuietEvasion,
    ResourceCycle,
}

impl FromStr for SuiteKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "mate_sacrifice" => Ok(Self::MateSacrifice),
            "quiet_evasion" => Ok(Self::QuietEvasion),
            "resource_cycle" => Ok(Self::ResourceCycle),
            _ => Err(anyhow!("invalid suite kind: {value}")),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetSplit {
    Dev,
    Holdout,
}

impl FromStr for DatasetSplit {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "dev" => Ok(Self::Dev),
            "holdout" => Ok(Self::Holdout),
            _ => Err(anyhow!("invalid split: {value}; expected dev or holdout")),
        }
    }
}

pub struct AtomicOutput {
    final_path: PathBuf,
    temp_path: PathBuf,
    writer: Option<BufWriter<File>>,
}

impl AtomicOutput {
    pub fn new(path: &Path) -> Result<Self> {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                anyhow!(
                    "output path must have a UTF-8 file name: {}",
                    path.display()
                )
            })?;
        for _ in 0..100 {
            let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let temp_path = parent.join(format!(".{name}.{}.{}.tmp", std::process::id(), sequence));
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_path)
            {
                Ok(file) => {
                    return Ok(Self {
                        final_path: path.to_path_buf(),
                        temp_path,
                        writer: Some(BufWriter::new(file)),
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(anyhow!(
            "failed to allocate temporary output for {}",
            path.display()
        ))
    }

    pub fn commit(mut self) -> Result<()> {
        self.prepare()?;
        if let Err(error) = fs::rename(&self.temp_path, &self.final_path) {
            let _ = fs::remove_file(&self.temp_path);
            return Err(error).with_context(|| {
                format!(
                    "failed to atomically rename {} to {}",
                    self.temp_path.display(),
                    self.final_path.display()
                )
            });
        }
        Ok(())
    }

    fn prepare(&mut self) -> Result<()> {
        let Some(writer) = self.writer.as_mut() else {
            return Ok(());
        };
        writer.flush()?;
        writer.get_ref().sync_all()?;
        let writer = self.writer.take().expect("atomic writer");
        drop(writer);
        Ok(())
    }

    fn backup_path(&self) -> PathBuf {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let name = self
            .final_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("output");
        self.final_path.with_file_name(format!(
            ".{name}.{}.{}.backup",
            std::process::id(),
            sequence
        ))
    }

    pub fn commit_pair(self, other: Self) -> Result<()> {
        Self::commit_pair_internal(self, other, false, false)
    }

    pub fn final_path(&self) -> &Path {
        &self.final_path
    }

    pub fn temp_path(&self) -> &Path {
        &self.temp_path
    }

    fn commit_pair_internal(
        mut first: Self,
        mut second: Self,
        fail_after_first: bool,
        inject_backup_cleanup_failure: bool,
    ) -> Result<()> {
        let had_first = Self::validate_pair_destination(&first.final_path)?;
        let had_second = Self::validate_pair_destination(&second.final_path)?;
        first.prepare()?;
        second.prepare()?;
        let first_backup = first.backup_path();
        let second_backup = second.backup_path();

        if had_first {
            fs::rename(&first.final_path, &first_backup)?;
        }
        if had_second {
            if let Err(error) = fs::rename(&second.final_path, &second_backup) {
                if had_first {
                    let _ = fs::rename(&first_backup, &first.final_path);
                }
                return Err(error.into());
            }
        }

        let publish_result = (|| -> Result<()> {
            fs::rename(&first.temp_path, &first.final_path)?;
            if fail_after_first {
                return Err(anyhow!("injected pair publication failure"));
            }
            fs::rename(&second.temp_path, &second.final_path)?;
            Ok(())
        })();

        if let Err(error) = publish_result {
            let _ = fs::remove_file(&first.final_path);
            let _ = fs::remove_file(&second.final_path);
            if had_first {
                fs::rename(&first_backup, &first.final_path)
                    .context("failed to restore first output after pair publication failure")?;
            }
            if had_second {
                fs::rename(&second_backup, &second.final_path)
                    .context("failed to restore second output after pair publication failure")?;
            }
            return Err(error);
        }

        // Publication is committed once both new files are in place. Old-backup cleanup is
        // best-effort maintenance: reporting failure here would misrepresent a valid new pair.
        if had_first {
            let cleanup = if inject_backup_cleanup_failure {
                Err(std::io::Error::other("injected backup cleanup failure"))
            } else {
                fs::remove_file(&first_backup)
            };
            if let Err(error) = cleanup {
                eprintln!(
                    "warning: published output pair but could not remove backup {}: {error}",
                    first_backup.display()
                );
            }
        }
        if had_second {
            let cleanup = if inject_backup_cleanup_failure {
                Err(std::io::Error::other("injected backup cleanup failure"))
            } else {
                fs::remove_file(&second_backup)
            };
            if let Err(error) = cleanup {
                eprintln!(
                    "warning: published output pair but could not remove backup {}: {error}",
                    second_backup.display()
                );
            }
        }
        Ok(())
    }

    fn validate_pair_destination(path: &Path) -> Result<bool> {
        match fs::symlink_metadata(path) {
            Ok(metadata) if metadata.file_type().is_file() => Ok(true),
            Ok(_) => Err(anyhow!(
                "pair output destination must be a regular file or not exist: {}",
                path.display()
            )),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error).with_context(|| {
                format!(
                    "failed to inspect pair output destination {}",
                    path.display()
                )
            }),
        }
    }
}

impl Write for AtomicOutput {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.writer.as_mut().expect("atomic writer").write(buffer)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.as_mut().expect("atomic writer").flush()
    }
}

impl Drop for AtomicOutput {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.temp_path);
    }
}

fn comparable_path(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return Ok(fs::canonicalize(path)?);
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    Ok(normalized)
}

pub fn ensure_distinct_paths(paths: &[(&str, &Path)]) -> Result<()> {
    #[cfg(unix)]
    fn file_identity(path: &Path) -> Option<(u64, u64)> {
        use std::os::unix::fs::MetadataExt;

        fs::metadata(path)
            .ok()
            .map(|metadata| (metadata.dev(), metadata.ino()))
    }

    #[cfg(not(unix))]
    fn file_identity(_path: &Path) -> Option<(u64, u64)> {
        None
    }

    let mut checked: Vec<(&str, &Path, PathBuf, Option<(u64, u64)>)> =
        Vec::with_capacity(paths.len());
    for &(label, path) in paths {
        let comparable = comparable_path(path)?;
        let identity = file_identity(path);
        for (other_label, other_path, other_comparable, other_inode) in &checked {
            if &comparable == other_comparable
                || identity.is_some() && identity == *other_inode
            {
                return Err(anyhow!(
                    "path collision: {label}={} and {other_label}={} refer to the same file",
                    path.display(),
                    other_path.display()
                ));
            }
        }
        checked.push((label, path, comparable, identity));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MateProof {
    ProvenMate(Vec<Move>),
    ProvenNoMateWithinHorizon,
    Unknown,
}

pub struct MateOracle {
    node_limit: u64,
    nodes: u64,
}

impl MateOracle {
    pub fn new(node_limit: u64) -> Self {
        Self {
            node_limit,
            nodes: 0,
        }
    }

    pub fn nodes(&self) -> u64 {
        self.nodes
    }

    pub fn prove(&mut self, position: &mut Position, attacker: Color, depth: u8) -> MateProof {
        if self.nodes >= self.node_limit {
            return MateProof::Unknown;
        }
        self.nodes += 1;

        let moves = position.legal_moves();
        if moves.is_empty() {
            return if position.in_check() && position.side_to_move() != attacker {
                MateProof::ProvenMate(Vec::new())
            } else {
                MateProof::ProvenNoMateWithinHorizon
            };
        }
        if depth == 0 {
            return MateProof::ProvenNoMateWithinHorizon;
        }

        if position.side_to_move() == attacker {
            let mut saw_unknown = false;
            let checking_moves: Vec<_> = moves
                .iter()
                .copied()
                .filter(|&mv| position.is_check_move(mv))
                .collect();
            for mv in checking_moves {
                position.do_move(mv);
                let result = self.prove(position, attacker, depth - 1);
                position.undo_move(mv);
                match result {
                    MateProof::ProvenMate(mut line) => {
                        line.insert(0, mv);
                        return MateProof::ProvenMate(line);
                    }
                    MateProof::Unknown => saw_unknown = true,
                    MateProof::ProvenNoMateWithinHorizon => {}
                }
            }
            if saw_unknown {
                MateProof::Unknown
            } else {
                MateProof::ProvenNoMateWithinHorizon
            }
        } else {
            let defense_count = moves.len();
            let mut longest_line = Vec::new();
            let mut saw_unknown = false;
            for mv in moves {
                position.do_move(mv);
                let result = self.prove(position, attacker, depth - 1);
                position.undo_move(mv);
                match result {
                    MateProof::ProvenMate(mut line) => {
                        line.insert(0, mv);
                        if line.len() > longest_line.len() {
                            longest_line = line;
                        }
                    }
                    MateProof::ProvenNoMateWithinHorizon => {
                        return MateProof::ProvenNoMateWithinHorizon;
                    }
                    MateProof::Unknown => saw_unknown = true,
                }
            }
            debug_assert!(defense_count > 0);
            if saw_unknown {
                MateProof::Unknown
            } else {
                MateProof::ProvenMate(longest_line)
            }
        }
    }
}

pub fn simple_see(position: &Position, mv: Move) -> i32 {
    if let Move::Normal { from, to, .. } = mv {
        if let (Some(attacker), Some(victim)) = (position.piece_at(from), position.piece_at(to)) {
            return get_piece_value(victim.piece_kind()) - get_piece_value(attacker.piece_kind());
        }
    }
    0
}

pub fn hand_counts(position: &Position, color: Color) -> [u8; 7] {
    std::array::from_fn(|i| position.hand(color).count(HAND_KINDS[i]).unwrap_or(0))
}

pub fn color_name(color: Color) -> &'static str {
    match color {
        Color::Black => "black",
        Color::White => "white",
    }
}

pub fn canonical_sfen(position: &Position) -> String {
    position
        .to_sfen_owned()
        .split_whitespace()
        .take(3)
        .collect::<Vec<_>>()
        .join(" ")
}

fn parse_dataset_line(line: &str, source_line: usize) -> Result<InputPosition> {
    if line.trim_start().starts_with('{') {
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("invalid JSON at input line {source_line}"))?;
        let sfen = value
            .get("sfen")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("missing string sfen at input line {source_line}"))?;
        let position = position_from_sfen_or_usi(sfen)
            .ok_or_else(|| anyhow!("invalid sfen at input line {source_line}: {sfen}"))?;
        let source_game_key = value
            .get("source_game_key")
            .or_else(|| value.get("game_key"))
            .and_then(Value::as_str)
            .filter(|key| !key.is_empty())
            .map(str::to_owned);
        Ok(InputPosition {
            source_line,
            source_game_key,
            position,
        })
    } else {
        let position = position_from_sfen_or_usi(line)
            .ok_or_else(|| anyhow!("invalid SFEN/USI at input line {source_line}: {line}"))?;
        Ok(InputPosition {
            source_line,
            source_game_key: None,
            position,
        })
    }
}

pub fn load_input_positions(path: &Path) -> Result<(Vec<InputPosition>, usize)> {
    let reader = BufReader::new(
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?,
    );
    let mut positions = Vec::new();
    let mut nonempty_lines = 0usize;
    for (index, line) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line = line.with_context(|| format!("failed to read input line {line_number}"))?;
        if line.trim().is_empty() {
            continue;
        }
        nonempty_lines += 1;
        positions.push(parse_dataset_line(&line, line_number)?);
    }
    if positions.is_empty() {
        return Err(anyhow!(
            "no valid non-empty positions in {}",
            path.display()
        ));
    }
    Ok((positions, nonempty_lines))
}

pub fn deduplicate_input_positions(positions: Vec<InputPosition>) -> (Vec<InputPosition>, usize) {
    let mut seen = std::collections::HashSet::new();
    let mut unique = Vec::with_capacity(positions.len());
    let mut duplicates = 0usize;
    for input in positions {
        if seen.insert(canonical_sfen(&input.position)) {
            unique.push(input);
        } else {
            duplicates += 1;
        }
    }
    (unique, duplicates)
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut reader = BufReader::new(File::open(path)?);
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn collect_source_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            collect_source_files(&entry?.path(), files)?;
        }
    } else if path.is_file() {
        files.push(path.to_path_buf());
    }
    Ok(())
}

pub fn generator_source_sha256() -> Result<String> {
    let root_output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()?;
    if !root_output.status.success() {
        return Err(anyhow!("not inside a git worktree"));
    }
    let root = PathBuf::from(String::from_utf8_lossy(&root_output.stdout).trim());
    let mut files = Vec::new();
    for relative in ["Cargo.toml", "Cargo.lock", "src", "shogi_lib"] {
        collect_source_files(&root.join(relative), &mut files)?;
    }
    files.sort();
    let mut hasher = Sha256::new();
    for path in files {
        let relative = path.strip_prefix(&root).unwrap_or(&path);
        hasher.update(relative.to_string_lossy().as_bytes());
        hasher.update([0]);
        let mut reader = BufReader::new(File::open(&path)?);
        let mut buffer = [0u8; 1024 * 1024];
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        hasher.update([0xff]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn generator_worktree_dirty() -> Result<bool> {
    let output = Command::new("git")
        .args([
            "status",
            "--porcelain",
            "--",
            "Cargo.toml",
            "Cargo.lock",
            "src",
            "shogi_lib",
        ])
        .output()?;
    if !output.status.success() {
        return Err(anyhow!("failed to inspect generator worktree status"));
    }
    Ok(!output.stdout.is_empty())
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SuiteManifest {
    pub schema_version: u32,
    pub generator: String,
    pub generator_commit: String,
    pub generator_worktree_dirty: bool,
    pub generator_source_sha256: String,
    pub suite_kind: SuiteKind,
    pub split: DatasetSplit,
    pub weight_sha256: Option<String>,
    pub input_path: String,
    pub input_sha256: String,
    pub input_nonempty_lines: usize,
    pub valid_positions: usize,
    pub output_path: String,
    pub output_sha256: String,
    pub seed: u64,
    pub records_written: usize,
    pub duplicate_sfens_skipped: usize,
    pub filters: Value,
}

pub fn commit_suite_with_manifest(
    mut output_writer: AtomicOutput,
    generator: &str,
    suite_kind: SuiteKind,
    split: DatasetSplit,
    input: &Path,
    seed: u64,
    input_nonempty_lines: usize,
    valid_positions: usize,
    records_written: usize,
    duplicate_sfens_skipped: usize,
    filters: Value,
) -> Result<()> {
    output_writer.prepare()?;
    let output = output_writer.final_path();
    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let manifest = SuiteManifest {
        schema_version: 1,
        generator: generator.to_string(),
        generator_commit: commit,
        generator_worktree_dirty: generator_worktree_dirty()?,
        generator_source_sha256: generator_source_sha256()?,
        weight_sha256: None,
        suite_kind,
        split,
        input_path: fs::canonicalize(input)?.display().to_string(),
        input_sha256: sha256_file(input)?,
        input_nonempty_lines,
        valid_positions,
        output_path: comparable_path(output)?.display().to_string(),
        output_sha256: sha256_file(output_writer.temp_path())?,
        seed,
        records_written,
        duplicate_sfens_skipped,
        filters,
    };
    let manifest_path = output.with_extension("manifest.json");
    let mut writer = AtomicOutput::new(&manifest_path)?;
    serde_json::to_writer_pretty(&mut writer, &manifest)?;
    writeln!(writer)?;
    output_writer.commit_pair(writer)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{parse_usi_move_for_color, position_from_sfen_or_usi};
    use std::os::unix::fs::symlink;

    fn temp_test_dir(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "shogi_search_quality_{name}_{}_{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).expect("create temp test directory");
        path
    }

    #[test]
    fn mate_oracle_recognizes_terminal_mate_without_evaluation() {
        let mut position =
            position_from_sfen_or_usi("9/7pp/8k/7PP/7G1/9/9/9/K8 w 2r2b3g4s4n4l14p 2")
                .expect("valid position");
        assert!(position.in_check());
        assert!(position.legal_moves().is_empty());
        let attacker = position.side_to_move().flip();
        let mut oracle = MateOracle::new(100_000);
        assert!(matches!(
            oracle.prove(&mut position, attacker, 0),
            MateProof::ProvenMate(_)
        ));
    }

    #[test]
    fn zero_node_limit_is_unknown() {
        let mut position = Position::default();
        let attacker = position.side_to_move();
        assert_eq!(
            MateProof::Unknown,
            MateOracle::new(0).prove(&mut position, attacker, 3)
        );
    }

    #[test]
    fn mate_oracle_proves_all_branches_of_a_two_defense_mate() {
        let mut position = position_from_sfen_or_usi(
            "6+Bnl/3G2s2/p2p1pkpp/5lg2/3s+R1N2/P4NpPP/1P1PP4/2GSG1+l2/L2K3N1 b RS2Pb4p 133",
        )
        .expect("valid branching mate");
        let attacker = position.side_to_move();
        let first = parse_usi_move_for_color("3e2c+", attacker).expect("valid first move");
        assert!(position.legal_moves().contains(&first));
        assert!(position.is_check_move(first));
        position.do_move(first);
        assert_eq!(
            2,
            position.legal_moves().len(),
            "fixture must exercise AND branching"
        );
        let mut oracle = MateOracle::new(10_000);
        assert!(matches!(
            oracle.prove(&mut position, attacker, 4),
            MateProof::ProvenMate(_)
        ));
    }

    #[test]
    fn parses_dataset_jsonl_sfen() {
        let record = parse_dataset_line(
            r#"{"source_game_key":"g1","sfen":"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"}"#,
            1,
        )
        .expect("dataset JSONL position");
        assert_eq!(Color::Black, record.position.side_to_move());
        assert_eq!(Some("g1"), record.source_game_key.as_deref());
        let record = parse_dataset_line(
            r#"{"game_key":"legacy-g2","sfen":"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"}"#,
            2,
        )
        .expect("dataset_build JSONL position");
        assert_eq!(Some("legacy-g2"), record.source_game_key.as_deref());
    }

    #[test]
    fn strict_loader_rejects_a_malformed_nonempty_line() {
        let directory = temp_test_dir("invalid_input");
        let input = directory.join("input.sfen");
        fs::write(&input, "startpos\ninvalid\n").expect("write input");
        let error = load_input_positions(&input)
            .err()
            .expect("must reject invalid line");
        assert!(error.to_string().contains("input line 2"));
        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[test]
    fn atomic_output_replaces_only_on_commit() {
        let directory = temp_test_dir("atomic");
        let output = directory.join("output.txt");
        fs::write(&output, "old").expect("write old output");
        let mut atomic = AtomicOutput::new(&output).expect("atomic output");
        write!(atomic, "new").expect("write temp output");
        assert_eq!("old", fs::read_to_string(&output).expect("read old output"));
        atomic.commit().expect("commit");
        assert_eq!("new", fs::read_to_string(&output).expect("read new output"));
        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[test]
    fn atomic_output_publishes_a_pair() {
        let directory = temp_test_dir("atomic_pair");
        let first_path = directory.join("suite.jsonl");
        let second_path = directory.join("suite.manifest.json");
        fs::write(&first_path, "old suite").expect("write old suite");
        fs::write(&second_path, "old manifest").expect("write old manifest");
        let mut first = AtomicOutput::new(&first_path).expect("first output");
        let mut second = AtomicOutput::new(&second_path).expect("second output");
        write!(first, "new suite").expect("write suite");
        write!(second, "new manifest").expect("write manifest");
        first.commit_pair(second).expect("publish pair");
        assert_eq!("new suite", fs::read_to_string(&first_path).unwrap());
        assert_eq!("new manifest", fs::read_to_string(&second_path).unwrap());
        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[test]
    fn atomic_output_pair_rolls_back_both_files() {
        let directory = temp_test_dir("atomic_pair_rollback");
        let first_path = directory.join("suite.jsonl");
        let second_path = directory.join("suite.manifest.json");
        fs::write(&first_path, "old suite").expect("write old suite");
        fs::write(&second_path, "old manifest").expect("write old manifest");
        let mut first = AtomicOutput::new(&first_path).expect("first output");
        let mut second = AtomicOutput::new(&second_path).expect("second output");
        write!(first, "new suite").expect("write suite");
        write!(second, "new manifest").expect("write manifest");
        assert!(AtomicOutput::commit_pair_internal(first, second, true, false).is_err());
        assert_eq!("old suite", fs::read_to_string(&first_path).unwrap());
        assert_eq!("old manifest", fs::read_to_string(&second_path).unwrap());
        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[test]
    fn atomic_output_pair_rejects_a_sidecar_directory_without_touching_either_destination() {
        let directory = temp_test_dir("atomic_pair_sidecar_directory");
        let first_path = directory.join("suite.jsonl");
        let second_path = directory.join("suite.manifest.json");
        fs::write(&first_path, "old suite").expect("write old suite");
        fs::create_dir(&second_path).expect("create sidecar directory");
        let mut first = AtomicOutput::new(&first_path).expect("first output");
        let mut second = AtomicOutput::new(&second_path).expect("second output");
        write!(first, "new suite").expect("write suite");
        write!(second, "new manifest").expect("write manifest");

        assert!(first.commit_pair(second).is_err());
        assert_eq!("old suite", fs::read_to_string(&first_path).unwrap());
        assert!(second_path.is_dir());
        let residue: Vec<_> = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp") || name.contains(".backup"))
            .collect();
        assert!(
            residue.is_empty(),
            "unexpected transaction residue: {residue:?}"
        );
        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[test]
    fn atomic_output_pair_rejects_a_sidecar_symlink_without_touching_destinations() {
        let directory = temp_test_dir("atomic_pair_sidecar_symlink");
        let first_path = directory.join("suite.jsonl");
        let second_path = directory.join("suite.manifest.json");
        let symlink_target = directory.join("real-manifest.json");
        fs::write(&first_path, "old suite").expect("write old suite");
        fs::write(&symlink_target, "old manifest").expect("write symlink target");
        symlink(&symlink_target, &second_path).expect("create sidecar symlink");
        let mut first = AtomicOutput::new(&first_path).expect("first output");
        let mut second = AtomicOutput::new(&second_path).expect("second output");
        write!(first, "new suite").expect("write suite");
        write!(second, "new manifest").expect("write manifest");

        assert!(first.commit_pair(second).is_err());
        assert_eq!("old suite", fs::read_to_string(&first_path).unwrap());
        assert!(fs::symlink_metadata(&second_path)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!("old manifest", fs::read_to_string(&symlink_target).unwrap());
        let residue: Vec<_> = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp") || name.contains(".backup"))
            .collect();
        assert!(
            residue.is_empty(),
            "unexpected transaction residue: {residue:?}"
        );
        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[test]
    fn backup_cleanup_failure_does_not_invalidate_a_published_pair() {
        let directory = temp_test_dir("atomic_pair_cleanup_failure");
        let first_path = directory.join("suite.jsonl");
        let second_path = directory.join("suite.manifest.json");
        fs::write(&first_path, "old suite").expect("write old suite");
        fs::write(&second_path, "old manifest").expect("write old manifest");
        let mut first = AtomicOutput::new(&first_path).expect("first output");
        let mut second = AtomicOutput::new(&second_path).expect("second output");
        write!(first, "new suite").expect("write suite");
        write!(second, "new manifest").expect("write manifest");

        AtomicOutput::commit_pair_internal(first, second, false, true)
            .expect("cleanup failure must not invalidate publication");
        assert_eq!("new suite", fs::read_to_string(&first_path).unwrap());
        assert_eq!("new manifest", fs::read_to_string(&second_path).unwrap());
        let backups: Vec<_> = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .is_some_and(|name| name.to_string_lossy().contains(".backup"))
            })
            .collect();
        assert_eq!(
            2,
            backups.len(),
            "old backups remain available for recovery"
        );
        fs::remove_dir_all(directory).expect("cleanup");
    }

    #[test]
    fn distinct_path_check_detects_hard_links() {
        let directory = temp_test_dir("hardlink");
        let input = directory.join("input");
        let output = directory.join("output");
        fs::write(&input, "data").expect("write input");
        fs::hard_link(&input, &output).expect("create hardlink");
        assert!(ensure_distinct_paths(&[("input", &input), ("output", &output)]).is_err());
        fs::remove_dir_all(directory).expect("cleanup");
    }
}
