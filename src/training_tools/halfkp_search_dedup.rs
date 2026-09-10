use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::Serialize;
use shogi_ai::halfkp_training::{
    read_search_teacher_manifest, search_teacher_manifest_path, PackedHalfKpPosition,
    SearchTeacherProvenance, SearchTeacherReader, SearchTeacherWriter, SEARCH_TEACHER_SEMANTICS_ID,
    SEARCH_TEACHER_SEMANTICS_VERSION,
};
use shogi_ai::training_data::{
    artifact_metadata, capture_run_environment, sha256_file, PHASE_POLICY_VERSION,
    SPLIT_POLICY_VERSION,
};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Deduplicate HKST records by root and candidate network inputs")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Serialize)]
struct DedupReport {
    schema: &'static str,
    input_files: usize,
    input_records: u64,
    retained_records: u64,
    rejected_records: u64,
    unique_network_inputs: usize,
    output_sha256: String,
    output_manifest_sha256: String,
    passed: bool,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    if args.input.is_empty() {
        return Err(anyhow!("at least one --input is required"));
    }
    if args.output.exists() || search_teacher_manifest_path(&args.output).exists() {
        return Err(anyhow!(
            "refusing to overwrite deduplicated output {}",
            args.output.display()
        ));
    }
    if let Some(report) = &args.report {
        if report.exists() {
            return Err(anyhow!("refusing to overwrite {}", report.display()));
        }
    }

    let manifests = args
        .input
        .iter()
        .map(|path| verified_manifest(path))
        .collect::<Result<Vec<_>>>()?;
    let model_sha = manifests
        .first()
        .and_then(|manifest| manifest.model.as_ref())
        .map(|artifact| artifact.sha256.clone())
        .ok_or_else(|| anyhow!("input manifest has no teacher model"))?;
    if manifests.iter().any(|manifest| {
        manifest.format != "HKST0002"
            || manifest.teacher_semantics_version != SEARCH_TEACHER_SEMANTICS_VERSION
            || manifest.teacher_semantics_id != SEARCH_TEACHER_SEMANTICS_ID
            || manifest
                .model
                .as_ref()
                .map(|artifact| artifact.sha256.as_str())
                != Some(model_sha.as_str())
    }) {
        return Err(anyhow!(
            "all HKST inputs must use the same v4 semantics and teacher model"
        ));
    }

    let first = &manifests[0];
    let provenance = SearchTeacherProvenance {
        environment: Some(capture_run_environment()),
        inputs: args
            .input
            .iter()
            .zip(&manifests)
            .map(|(path, manifest)| artifact_metadata(path, Some(manifest.records)))
            .collect::<Result<Vec<_>>>()?,
        model: first.model.clone(),
        engine_binary: std::env::current_exe()
            .ok()
            .and_then(|path| artifact_metadata(&path, None).ok()),
        feature_profile: first.feature_profile.clone(),
        search_limits: serde_json::json!({
            "operation": "deduplicate_hkst_network_inputs",
            "hash_scope": "root_and_all_candidate_children",
            "input_order_is_priority": true,
            "inputs": args.input,
        }),
        jobs: Some(1),
        random_seeds: Vec::new(),
        phase_policy_version: Some(PHASE_POLICY_VERSION),
        split_policy_version: Some(SPLIT_POLICY_VERSION),
        parent_manifest_sha256: args
            .input
            .iter()
            .map(|path| sha256_file(&search_teacher_manifest_path(path)))
            .collect::<Result<Vec<_>>>()?,
    };
    let mut writer = SearchTeacherWriter::create(&args.output)?;
    writer.set_provenance(provenance);
    let mut seen = HashSet::<[u8; 32]>::new();
    let mut input_records = 0u64;
    let mut retained_records = 0u64;
    for (path, manifest) in args.input.iter().zip(&manifests) {
        let mut reader =
            SearchTeacherReader::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut records = 0u64;
        while let Some(record) = reader.read_record()? {
            input_records += 1;
            records += 1;
            let hashes = record_inputs(&record);
            if hashes.iter().any(|hash| seen.contains(hash)) {
                continue;
            }
            seen.extend(hashes);
            writer.write_record(&record)?;
            retained_records += 1;
        }
        if records != manifest.records {
            return Err(anyhow!(
                "{} manifest declares {} records but read {}",
                path.display(),
                manifest.records,
                records
            ));
        }
    }
    writer.finish()?;

    let report = DedupReport {
        schema: "halfkp_search_dedup_v1",
        input_files: args.input.len(),
        input_records,
        retained_records,
        rejected_records: input_records - retained_records,
        unique_network_inputs: seen.len(),
        output_sha256: sha256_file(&args.output)?,
        output_manifest_sha256: sha256_file(&search_teacher_manifest_path(&args.output))?,
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

fn verified_manifest(path: &Path) -> Result<shogi_ai::halfkp_training::SearchTeacherManifest> {
    let manifest = read_search_teacher_manifest(path)?
        .ok_or_else(|| anyhow!("{} has no HKST manifest", path.display()))?;
    let output = manifest
        .output
        .as_ref()
        .ok_or_else(|| anyhow!("{} manifest has no output artifact", path.display()))?;
    if output.sha256 != sha256_file(path)? || output.records != Some(manifest.records) {
        return Err(anyhow!(
            "{} manifest output does not match dataset",
            path.display()
        ));
    }
    Ok(manifest)
}

fn record_inputs(record: &shogi_ai::halfkp_training::SearchTeacherRecord) -> Vec<[u8; 32]> {
    std::iter::once(&record.root)
        .chain(record.candidates.iter().map(|candidate| &candidate.child))
        .map(PackedHalfKpPosition::network_input_hash)
        .collect()
}
