use anyhow::{anyhow, Context, Result};
use clap::Parser;
use glob::glob;
use serde::{Deserialize, Serialize};
use shogi_ai::training_data::{
    artifact_metadata, capture_run_environment, line_artifact_metadata, sha256_file, sha256_hex,
    ArtifactMetadata, RunEnvironment,
};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Split JSONL into reproducible, manifest-verified shards")]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output_prefix: PathBuf,
    #[arg(long)]
    lines: usize,
    #[arg(long, default_value_t = 4)]
    suffix_width: usize,
    #[arg(long)]
    parent_manifest: Vec<PathBuf>,
    #[arg(long, default_value_t = false)]
    reuse_if_matches: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct ShardManifest {
    schema_version: u32,
    stage: String,
    stage_fingerprint: String,
    environment: RunEnvironment,
    input: ArtifactMetadata,
    lines_per_shard: usize,
    suffix_width: usize,
    parent_manifest_sha256: Vec<String>,
    outputs: Vec<ArtifactMetadata>,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    if args.lines == 0 || args.suffix_width == 0 {
        return Err(anyhow!("--lines and --suffix-width must be positive"));
    }
    let input = line_artifact_metadata(&args.input)?;
    let parent_manifest_sha256 = args
        .parent_manifest
        .iter()
        .map(|path| sha256_file(path))
        .collect::<Result<Vec<_>>>()?;
    let fingerprint = stage_fingerprint(
        &input,
        args.lines,
        args.suffix_width,
        &parent_manifest_sha256,
    )?;
    let manifest_path = manifest_path(&args.output_prefix);
    if args.reuse_if_matches && reusable(&manifest_path, &fingerprint)? {
        println!("reused output_prefix={}", args.output_prefix.display());
        return Ok(());
    }

    remove_previous_shards(&args.output_prefix)?;
    if let Some(parent) = args.output_prefix.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut reader = BufReader::new(File::open(&args.input)?);
    let mut outputs = Vec::new();
    let mut buffer = Vec::new();
    let mut shard_index = 0;
    loop {
        let path = shard_path(&args.output_prefix, shard_index, args.suffix_width);
        let temporary = append_suffix(&path, ".tmp");
        let mut writer = BufWriter::new(File::create(&temporary)?);
        let mut records = 0_u64;
        while records < args.lines as u64 {
            buffer.clear();
            if reader.read_until(b'\n', &mut buffer)? == 0 {
                break;
            }
            writer.write_all(&buffer)?;
            records += 1;
        }
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        if records == 0 && shard_index > 0 {
            fs::remove_file(temporary)?;
            break;
        }
        fs::rename(&temporary, &path)?;
        outputs.push(artifact_metadata(&path, Some(records))?);
        shard_index += 1;
        if records < args.lines as u64 {
            break;
        }
    }

    let manifest = ShardManifest {
        schema_version: 1,
        stage: "jsonl_shard".to_string(),
        stage_fingerprint: fingerprint,
        environment: capture_run_environment(),
        input,
        lines_per_shard: args.lines,
        suffix_width: args.suffix_width,
        parent_manifest_sha256,
        outputs,
    };
    let temporary_manifest = append_suffix(&manifest_path, ".tmp");
    let mut manifest_writer = BufWriter::new(File::create(&temporary_manifest)?);
    serde_json::to_writer_pretty(&mut manifest_writer, &manifest)?;
    manifest_writer.flush()?;
    manifest_writer.get_ref().sync_all()?;
    drop(manifest_writer);
    fs::rename(temporary_manifest, &manifest_path)?;
    println!(
        "complete shards={} manifest={}",
        manifest.outputs.len(),
        manifest_path.display()
    );
    Ok(())
}

fn stage_fingerprint(
    input: &ArtifactMetadata,
    lines: usize,
    suffix_width: usize,
    parents: &[String],
) -> Result<String> {
    let value = serde_json::json!({
        "schema_version": 1,
        "stage": "jsonl_shard",
        "input_sha256": input.sha256,
        "input_records": input.records,
        "lines_per_shard": lines,
        "suffix_width": suffix_width,
        "parent_manifest_sha256": parents,
    });
    Ok(sha256_hex(&serde_json::to_vec(&value)?))
}

fn reusable(manifest_path: &Path, expected_fingerprint: &str) -> Result<bool> {
    let manifest: ShardManifest = match File::open(manifest_path) {
        Ok(file) => serde_json::from_reader(BufReader::new(file))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    if manifest.stage_fingerprint != expected_fingerprint {
        return Ok(false);
    }
    for output in manifest.outputs {
        if sha256_file(Path::new(&output.path))
            .with_context(|| format!("verify shard {}", output.path))?
            != output.sha256
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn remove_previous_shards(prefix: &Path) -> Result<()> {
    let pattern = format!("{}*.jsonl", prefix.to_string_lossy());
    for entry in glob(&pattern)? {
        fs::remove_file(entry?)?;
    }
    Ok(())
}

fn shard_path(prefix: &Path, index: usize, width: usize) -> PathBuf {
    PathBuf::from(format!("{}{index:0width$}.jsonl", prefix.to_string_lossy()))
}

fn manifest_path(prefix: &Path) -> PathBuf {
    append_suffix(prefix, "manifest.json")
}

fn append_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}
