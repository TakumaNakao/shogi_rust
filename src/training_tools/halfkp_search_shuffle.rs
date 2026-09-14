use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::Serialize;
use shogi_ai::halfkp_training::{
    read_search_teacher_manifest, search_teacher_manifest_path, SearchTeacherProvenance,
    SearchTeacherReader, SearchTeacherWriter, SEARCH_TEACHER_SEMANTICS_ID,
    SEARCH_TEACHER_SEMANTICS_VERSION,
};
use shogi_ai::training_data::{artifact_metadata, capture_run_environment, sha256_file};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Deterministically distribute HKST records across globally mixed shards")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long, required = true)]
    output_dir: PathBuf,
    #[arg(long, default_value = "train")]
    prefix: String,
    #[arg(long, default_value_t = 32)]
    shards: usize,
    #[arg(long)]
    max_records: Option<u64>,
    #[arg(long, default_value_t = 20260915)]
    seed: u64,
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    seed: u64,
    shards: usize,
    input_files: Vec<InputReport>,
    records_read: u64,
    records_written: u64,
    shard_records: Vec<u64>,
    outputs: Vec<OutputReport>,
    passed: bool,
}

#[derive(Serialize)]
struct InputReport {
    path: PathBuf,
    sha256: String,
    records_declared: u64,
}

#[derive(Serialize)]
struct OutputReport {
    path: PathBuf,
    sha256: String,
    manifest_sha256: String,
    records: u64,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    if args.input.is_empty() || args.shards == 0 || args.shards > 1024 {
        return Err(anyhow!("input is required and shards must be in 1..=1024"));
    }
    if args.output_dir.exists() {
        if std::fs::read_dir(&args.output_dir)?.next().is_some() {
            return Err(anyhow!(
                "refusing to write into non-empty output directory {}",
                args.output_dir.display()
            ));
        }
    } else {
        std::fs::create_dir_all(&args.output_dir)?;
    }

    let manifests = args
        .input
        .iter()
        .map(|path| {
            let manifest = read_search_teacher_manifest(path)?
                .ok_or_else(|| anyhow!("{} has no HKST manifest", path.display()))?;
            let output = manifest
                .output
                .as_ref()
                .ok_or_else(|| anyhow!("{} manifest has no output", path.display()))?;
            if manifest.format != "HKST0002"
                || manifest.teacher_semantics_version != SEARCH_TEACHER_SEMANTICS_VERSION
                || manifest.teacher_semantics_id != SEARCH_TEACHER_SEMANTICS_ID
                || output.sha256 != sha256_file(path)?
                || output.records != Some(manifest.records)
            {
                return Err(anyhow!("invalid or incompatible HKST input {}", path.display()));
            }
            Ok(manifest)
        })
        .collect::<Result<Vec<_>>>()?;
    let model_sha = manifests
        .first()
        .and_then(|manifest| manifest.model.as_ref())
        .map(|model| model.sha256.clone())
        .ok_or_else(|| anyhow!("HKST input has no teacher model"))?;
    if manifests.iter().any(|manifest| {
        manifest.model.as_ref().map(|model| model.sha256.as_str()) != Some(model_sha.as_str())
    }) {
        return Err(anyhow!("all inputs must use the same teacher model"));
    }

    let input_reports = args
        .input
        .iter()
        .zip(&manifests)
        .map(|(path, manifest)| {
            Ok(InputReport {
                path: path.clone(),
                sha256: sha256_file(path)?,
                records_declared: manifest.records,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let provenance = SearchTeacherProvenance {
        environment: Some(capture_run_environment()),
        inputs: args
            .input
            .iter()
            .zip(&manifests)
            .map(|(path, manifest)| artifact_metadata(path, Some(manifest.records)))
            .collect::<Result<Vec<_>>>()?,
        model: manifests.first().and_then(|manifest| manifest.model.clone()),
        engine_binary: std::env::current_exe()
            .ok()
            .and_then(|path| artifact_metadata(&path, None).ok()),
        feature_profile: manifests
            .first()
            .map(|manifest| manifest.feature_profile.clone())
            .unwrap_or_default(),
        search_limits: serde_json::json!({
            "operation": "global_hkst_shard_assignment",
            "assignment": "splitmix64(position_hash ^ seed) % shards",
            "seed": args.seed,
            "shards": args.shards,
            "max_records": args.max_records,
            "input_order_is_priority": false,
        }),
        jobs: Some(1),
        random_seeds: vec![args.seed],
        phase_policy_version: manifests.first().and_then(|manifest| manifest.phase_policy_version),
        split_policy_version: manifests.first().and_then(|manifest| manifest.split_policy_version),
        parent_manifest_sha256: args
            .input
            .iter()
            .map(|path| sha256_file(&search_teacher_manifest_path(path)))
            .collect::<Result<Vec<_>>>()?,
    };

    let mut writers = Vec::with_capacity(args.shards);
    let mut output_paths = Vec::with_capacity(args.shards);
    for index in 0..args.shards {
        let path = args
            .output_dir
            .join(format!("{}-{index:04}.hkst", args.prefix));
        let mut writer = SearchTeacherWriter::create(&path)?;
        writer.set_provenance(provenance.clone());
        output_paths.push(path);
        writers.push(writer);
    }

    let mut records_read = 0_u64;
    let mut shard_records = vec![0_u64; args.shards];
    'inputs: for (path, manifest) in args.input.iter().zip(&manifests) {
        let mut reader = SearchTeacherReader::open(path)
            .with_context(|| format!("open {}", path.display()))?;
        let mut input_records = 0_u64;
        while let Some(record) = reader.read_record()? {
            if args.max_records.is_some_and(|limit| records_read >= limit) {
                break 'inputs;
            }
            let shard = (splitmix64(record.position_hash ^ args.seed) % args.shards as u64)
                as usize;
            writers[shard].write_record(&record)?;
            records_read += 1;
            input_records += 1;
            shard_records[shard] += 1;
        }
        if input_records != manifest.records && args.max_records.is_none() {
            return Err(anyhow!(
                "{} manifest declares {} records but read {}",
                path.display(),
                manifest.records,
                input_records
            ));
        }
    }
    let mut outputs = Vec::with_capacity(args.shards);
    for (path, writer) in output_paths.iter().zip(writers) {
        let records = writer.finish()?;
        if records == 0 {
            return Err(anyhow!("empty output shard {}", path.display()));
        }
        outputs.push(OutputReport {
            path: path.clone(),
            sha256: sha256_file(path)?,
            manifest_sha256: sha256_file(&search_teacher_manifest_path(path))?,
            records,
        });
    }
    let report = Report {
        schema: "halfkp_search_shuffle_v1",
        seed: args.seed,
        shards: args.shards,
        input_files: input_reports,
        records_read,
        records_written: shard_records.iter().sum(),
        shard_records,
        outputs,
        passed: true,
    };
    let bytes = serde_json::to_vec_pretty(&report)?;
    println!("{}", String::from_utf8_lossy(&bytes));
    if let Some(path) = args.report {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, [&bytes[..], b"\n"].concat())?;
    }
    Ok(())
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
