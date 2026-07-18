use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Shard and split HalfKP KPP-distillation datasets")]
struct Args {
    #[arg(long)]
    generated_raw: PathBuf,
    #[arg(long)]
    generated_features: PathBuf,
    #[arg(long, required = true)]
    mainline: Vec<PathBuf>,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 256)]
    shards: usize,
    #[arg(long, default_value_t = 10)]
    holdout_permille: u16,
    #[arg(long, default_value_t = 20260716)]
    seed: u64,
}

#[derive(Deserialize)]
struct RawRecord {
    generation: String,
    sfen: String,
}

#[derive(Deserialize)]
struct FeatureRecord {
    sfen: String,
    static_eval: Option<f32>,
}

#[derive(Default, Serialize)]
struct Counts {
    search_train: u64,
    search_holdout: u64,
    random_train: u64,
    random_holdout: u64,
    mainline_train: u64,
    skipped: u64,
}

#[derive(Serialize)]
struct Manifest<'a> {
    schema: &'static str,
    generated_raw: String,
    generated_features: String,
    mainline: Vec<String>,
    shards: usize,
    holdout_permille: u16,
    seed: u64,
    counts: &'a Counts,
}

fn stable_hash(seed: u64, text: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64 ^ seed;
    for byte in text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn create_writer(path: &Path) -> Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(BufWriter::new(File::create(path)?))
}

fn shard_writers(root: &Path, category: &str, shards: usize) -> Result<Vec<BufWriter<File>>> {
    (0..shards)
        .map(|index| {
            create_writer(
                &root
                    .join("train")
                    .join(category)
                    .join(format!("part-{index:04}.jsonl")),
            )
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.shards == 0 || args.holdout_permille >= 1000 {
        return Err(anyhow!("invalid shard/split parameters"));
    }
    fs::create_dir_all(&args.output_dir)?;
    let mut search = shard_writers(&args.output_dir, "search", args.shards)?;
    let mut random = shard_writers(&args.output_dir, "random", args.shards)?;
    let mut mainline = shard_writers(&args.output_dir, "mainline", args.shards)?;
    let mut search_holdout = create_writer(&args.output_dir.join("holdout/search.jsonl"))?;
    let mut random_holdout = create_writer(&args.output_dir.join("holdout/random.jsonl"))?;
    let raw = BufReader::new(File::open(&args.generated_raw)?);
    let features = BufReader::new(File::open(&args.generated_features)?);
    let mut counts = Counts::default();

    let mut raw_lines = raw.lines();
    let mut feature_lines = features.lines();
    loop {
        let raw_line = raw_lines.next().transpose()?;
        let feature_line = feature_lines.next().transpose()?;
        match (raw_line, feature_line) {
            (None, None) => break,
            (Some(raw_line), Some(feature_line)) => {
                let raw_record: RawRecord = serde_json::from_str(&raw_line)?;
                let feature_record: FeatureRecord = serde_json::from_str(&feature_line)?;
                if raw_record.sfen != feature_record.sfen {
                    return Err(anyhow!("raw/feature SFEN mismatch"));
                }
                if !feature_record.static_eval.is_some_and(f32::is_finite) {
                    counts.skipped += 1;
                    continue;
                }
                let hash = stable_hash(args.seed, &raw_record.sfen);
                let holdout = hash % 1000 < args.holdout_permille as u64;
                let shard = ((hash / 1000) % args.shards as u64) as usize;
                match (raw_record.generation.as_str(), holdout) {
                    ("search", true) => {
                        writeln!(search_holdout, "{feature_line}")?;
                        counts.search_holdout += 1;
                    }
                    ("search", false) => {
                        writeln!(search[shard], "{feature_line}")?;
                        counts.search_train += 1;
                    }
                    ("random", true) => {
                        writeln!(random_holdout, "{feature_line}")?;
                        counts.random_holdout += 1;
                    }
                    ("random", false) => {
                        writeln!(random[shard], "{feature_line}")?;
                        counts.random_train += 1;
                    }
                    _ => counts.skipped += 1,
                }
            }
            _ => return Err(anyhow!("raw/feature line count mismatch")),
        }
    }

    for path in &args.mainline {
        let reader = BufReader::new(
            File::open(path).with_context(|| format!("open mainline {}", path.display()))?,
        );
        for line in reader.lines() {
            let line = line?;
            let record: FeatureRecord = match serde_json::from_str(&line) {
                Ok(record) => record,
                Err(_) => {
                    counts.skipped += 1;
                    continue;
                }
            };
            if !record.static_eval.is_some_and(f32::is_finite) {
                counts.skipped += 1;
                continue;
            }
            let hash = stable_hash(args.seed ^ 0x9e3779b97f4a7c15, &record.sfen);
            let shard = (hash % args.shards as u64) as usize;
            writeln!(mainline[shard], "{line}")?;
            counts.mainline_train += 1;
        }
    }

    for writer in search
        .iter_mut()
        .chain(random.iter_mut())
        .chain(mainline.iter_mut())
    {
        writer.flush()?;
    }
    search_holdout.flush()?;
    random_holdout.flush()?;
    let manifest = Manifest {
        schema: "halfkp_distill_shards_v1",
        generated_raw: args.generated_raw.display().to_string(),
        generated_features: args.generated_features.display().to_string(),
        mainline: args
            .mainline
            .iter()
            .map(|p| p.display().to_string())
            .collect(),
        shards: args.shards,
        holdout_permille: args.holdout_permille,
        seed: args.seed,
        counts: &counts,
    };
    let mut manifest_writer = create_writer(&args.output_dir.join("manifest.json"))?;
    serde_json::to_writer_pretty(&mut manifest_writer, &manifest)?;
    manifest_writer.write_all(b"\n")?;
    manifest_writer.flush()?;
    println!("{}", serde_json::to_string(&counts)?);
    Ok(())
}
