use anyhow::{anyhow, Context, Result};
use clap::{ArgAction, Parser};
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Build phase-balanced MMTO input JSONL from dataset_build records")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 9601)]
    seed: u64,
    #[arg(long, default_value_t = 0)]
    opening_limit: usize,
    #[arg(long, default_value_t = 0)]
    middle_limit: usize,
    #[arg(long, default_value_t = 0)]
    late_limit: usize,
    #[arg(long, default_value_t = 0)]
    max_input_records: usize,
    #[arg(long, default_value_t = 1.0)]
    opening_weight: f32,
    #[arg(long, default_value_t = 1.0)]
    middle_weight: f32,
    #[arg(long, default_value_t = 1.0)]
    late_weight: f32,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    dedupe_sfen: bool,
    #[arg(long, default_value_t = false)]
    exclude_in_check: bool,
    #[arg(long, default_value_t = 0)]
    min_legal_moves: usize,
    #[arg(long)]
    max_legal_moves: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Opening,
    Middle,
    Late,
}

impl Phase {
    fn label(self) -> &'static str {
        match self {
            Phase::Opening => "opening",
            Phase::Middle => "middle",
            Phase::Late => "late",
        }
    }

    fn index(self) -> usize {
        match self {
            Phase::Opening => 0,
            Phase::Middle => 1,
            Phase::Late => 2,
        }
    }
}

#[derive(Debug, Deserialize)]
struct InputRecord {
    sfen: String,
    teacher_move: String,
    #[serde(default)]
    ply: Option<usize>,
    #[serde(default)]
    phase: Option<String>,
    #[serde(default)]
    legal_moves: Option<usize>,
    #[serde(default)]
    in_check: Option<bool>,
}

#[derive(Clone, Debug, Serialize)]
struct OutputRecord {
    sfen: String,
    teacher_move: String,
    sample_weight: f32,
    phase: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    ply: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    legal_moves: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    in_check: Option<bool>,
}

#[derive(Default)]
struct Bucket {
    seen: usize,
    selected: Vec<OutputRecord>,
}

#[derive(Default)]
struct Stats {
    input_lines: usize,
    invalid_json: usize,
    filtered: usize,
    duplicate_sfen: usize,
    bucket_seen: [usize; 3],
    bucket_selected: [usize; 3],
}

fn normalize_phase(record: &InputRecord) -> Phase {
    if let Some(phase) = record.phase.as_deref() {
        match phase {
            "opening" => return Phase::Opening,
            "middle" | "middlegame" => return Phase::Middle,
            "late" | "endgame" => return Phase::Late,
            _ => {}
        }
    }
    match record.ply.unwrap_or(1) {
        0..=40 => Phase::Opening,
        41..=90 => Phase::Middle,
        _ => Phase::Late,
    }
}

fn sample_weight(phase: Phase, args: &Args) -> f32 {
    match phase {
        Phase::Opening => args.opening_weight,
        Phase::Middle => args.middle_weight,
        Phase::Late => args.late_weight,
    }
}

fn phase_limit(phase: Phase, args: &Args) -> usize {
    match phase {
        Phase::Opening => args.opening_limit,
        Phase::Middle => args.middle_limit,
        Phase::Late => args.late_limit,
    }
}

fn should_keep(record: &InputRecord, args: &Args) -> bool {
    if record.sfen.trim().is_empty() || record.teacher_move.trim().is_empty() {
        return false;
    }
    if args.exclude_in_check && record.in_check.unwrap_or(false) {
        return false;
    }
    if args.min_legal_moves > 0
        && record
            .legal_moves
            .is_some_and(|legal_moves| legal_moves < args.min_legal_moves)
    {
        return false;
    }
    if args.max_legal_moves.is_some_and(|max| {
        record
            .legal_moves
            .is_some_and(|legal_moves| legal_moves > max)
    }) {
        return false;
    }
    true
}

fn push_reservoir(bucket: &mut Bucket, record: OutputRecord, limit: usize, rng: &mut ChaCha8Rng) {
    bucket.seen += 1;
    if limit == 0 {
        bucket.selected.push(record);
        return;
    }
    if bucket.selected.len() < limit {
        bucket.selected.push(record);
        return;
    }
    let replacement = rng.gen_range(0..bucket.seen);
    if replacement < limit {
        bucket.selected[replacement] = record;
    }
}

fn create_writer(path: &Path) -> Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }
    Ok(BufWriter::new(File::create(path).with_context(|| {
        format!("failed to create {}", path.display())
    })?))
}

fn main() -> Result<()> {
    let args = Args::parse();
    for (name, value) in [
        ("--opening-weight", args.opening_weight),
        ("--middle-weight", args.middle_weight),
        ("--late-weight", args.late_weight),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(anyhow!("{name} must be finite and positive"));
        }
    }

    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let mut buckets = [Bucket::default(), Bucket::default(), Bucket::default()];
    let mut seen_sfens = HashSet::new();
    let mut stats = Stats::default();

    'inputs: for path in &args.input {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let reader = BufReader::new(file);
        for (line_index, line) in reader.lines().enumerate() {
            if args.max_input_records > 0 && stats.input_lines >= args.max_input_records {
                break 'inputs;
            }
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            stats.input_lines += 1;
            let record: InputRecord = match serde_json::from_str(&line) {
                Ok(record) => record,
                Err(_) => {
                    stats.invalid_json += 1;
                    eprintln!(
                        "{}:{} invalid json; skipped",
                        path.display(),
                        line_index + 1
                    );
                    continue;
                }
            };
            if !should_keep(&record, &args) {
                stats.filtered += 1;
                continue;
            }
            let sfen = record.sfen.trim().to_string();
            if args.dedupe_sfen && !seen_sfens.insert(sfen.clone()) {
                stats.duplicate_sfen += 1;
                continue;
            }

            let phase = normalize_phase(&record);
            let phase_index = phase.index();
            let output = OutputRecord {
                sfen,
                teacher_move: record.teacher_move.trim().to_string(),
                sample_weight: sample_weight(phase, &args),
                phase: phase.label(),
                ply: record.ply,
                legal_moves: record.legal_moves,
                in_check: record.in_check,
            };
            let limit = phase_limit(phase, &args);
            push_reservoir(&mut buckets[phase_index], output, limit, &mut rng);
            stats.bucket_seen[phase_index] += 1;
            stats.bucket_selected[phase_index] = buckets[phase_index].selected.len();
        }
    }

    let mut records = buckets
        .into_iter()
        .flat_map(|bucket| bucket.selected)
        .collect::<Vec<_>>();
    records.shuffle(&mut rng);

    let mut writer = create_writer(&args.output)?;
    for record in &records {
        serde_json::to_writer(&mut writer, record)?;
        writeln!(writer)?;
    }
    writer.flush()?;

    println!("input_lines={}", stats.input_lines);
    println!("invalid_json={}", stats.invalid_json);
    println!("filtered={}", stats.filtered);
    println!("duplicate_sfen={}", stats.duplicate_sfen);
    for phase in [Phase::Opening, Phase::Middle, Phase::Late] {
        let index = phase.index();
        println!(
            "{}: seen={} selected={} limit={} weight={:.3}",
            phase.label(),
            stats.bucket_seen[index],
            stats.bucket_selected[index],
            phase_limit(phase, &args),
            sample_weight(phase, &args)
        );
    }
    println!("written={}", records.len());
    println!("output={}", args.output.display());
    Ok(())
}
