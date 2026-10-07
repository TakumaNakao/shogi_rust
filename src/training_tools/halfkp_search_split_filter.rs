use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::Serialize;
use shogi_ai::halfkp_training::{
    read_search_teacher_manifest, search_teacher_manifest_path, PackedHalfKpPosition,
    SearchTeacherManifest, SearchTeacherProvenance, SearchTeacherReader, SearchTeacherRecord,
    SearchTeacherWriter, SEARCH_TEACHER_SEMANTICS_ID, SEARCH_TEACHER_SEMANTICS_VERSION,
};
use shogi_ai::training_data::{
    artifact_metadata, capture_run_environment, sha256_file, PHASE_POLICY_VERSION,
    SPLIT_POLICY_VERSION,
};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Remove HKST validation records overlapping any training network input")]
struct Args {
    #[arg(long, required = true)]
    train: Vec<PathBuf>,
    #[arg(long, required = true)]
    valid: Vec<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Serialize)]
struct FilterReport {
    schema: &'static str,
    train_files: usize,
    valid_files: usize,
    train_records: u64,
    raw_valid_records: u64,
    retained_valid_records: u64,
    excluded_valid_records: u64,
    unique_train_network_inputs: usize,
    colliding_network_inputs: usize,
    output_sha256: String,
    output_manifest_sha256: String,
    passed: bool,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    if args.output.exists() || search_teacher_manifest_path(&args.output).exists() {
        return Err(anyhow!(
            "refusing to overwrite filtered output {}",
            args.output.display()
        ));
    }
    if let Some(report) = &args.report {
        if report.exists() {
            return Err(anyhow!("refusing to overwrite {}", report.display()));
        }
    }

    let manifests = args
        .train
        .iter()
        .chain(&args.valid)
        .map(|path| verified_manifest(path))
        .collect::<Result<Vec<_>>>()?;
    let teacher_model_sha = manifests
        .first()
        .and_then(|manifest| manifest.model.as_ref())
        .map(|artifact| artifact.sha256.as_str())
        .ok_or_else(|| anyhow!("input manifest has no teacher model"))?;
    if manifests.iter().any(|manifest| {
        manifest
            .model
            .as_ref()
            .map(|artifact| artifact.sha256.as_str())
            != Some(teacher_model_sha)
    }) {
        return Err(anyhow!("HKST inputs use different teacher models"));
    }

    let mut train_inputs = HashSet::new();
    let mut train_records = 0u64;
    for path in &args.train {
        visit_records(path, |record| {
            train_records += 1;
            insert_record_inputs(record, &mut train_inputs);
            Ok(())
        })?;
    }

    let valid_manifest = &manifests[args.train.len()];
    let provenance = SearchTeacherProvenance {
        environment: Some(capture_run_environment()),
        inputs: args
            .train
            .iter()
            .chain(&args.valid)
            .map(|path| {
                let records = read_search_teacher_manifest(path)?.map(|manifest| manifest.records);
                artifact_metadata(path, records)
            })
            .collect::<Result<Vec<_>>>()?,
        model: valid_manifest.model.clone(),
        engine_binary: std::env::current_exe()
            .ok()
            .and_then(|path| artifact_metadata(&path, None).ok()),
        feature_profile: valid_manifest.feature_profile.clone(),
        search_limits: serde_json::json!({
            "operation": "remove_train_valid_network_input_overlap",
            "hash_scope": "root_and_all_candidate_children",
            "train_files": args.train,
            "valid_files": args.valid,
        }),
        jobs: Some(1),
        random_seeds: Vec::new(),
        phase_policy_version: Some(PHASE_POLICY_VERSION),
        split_policy_version: Some(SPLIT_POLICY_VERSION),
        parent_manifest_sha256: args
            .train
            .iter()
            .chain(&args.valid)
            .map(|path| sha256_file(&search_teacher_manifest_path(path)))
            .collect::<Result<Vec<_>>>()?,
    };
    let mut writer = SearchTeacherWriter::create(&args.output)?;
    writer.set_provenance(provenance);
    let mut raw_valid_records = 0u64;
    let mut retained_valid_records = 0u64;
    let mut colliding_inputs = HashSet::new();
    for path in &args.valid {
        visit_records(path, |record| {
            raw_valid_records += 1;
            let record_inputs = record_inputs(record);
            let collisions = record_inputs
                .iter()
                .filter(|hash| train_inputs.contains(*hash))
                .copied()
                .collect::<Vec<_>>();
            if collisions.is_empty() {
                writer.write_record(record)?;
                retained_valid_records += 1;
            } else {
                colliding_inputs.extend(collisions);
            }
            Ok(())
        })?;
    }
    writer.finish()?;

    let mut retained_inputs = HashSet::new();
    visit_records(&args.output, |record| {
        insert_record_inputs(record, &mut retained_inputs);
        Ok(())
    })?;
    let residual_overlap = retained_inputs
        .iter()
        .filter(|hash| train_inputs.contains(*hash))
        .count();
    if residual_overlap != 0 {
        return Err(anyhow!(
            "filtered validation still has {residual_overlap} overlapping network inputs"
        ));
    }

    let report = FilterReport {
        schema: "halfkp_search_split_filter_v1",
        train_files: args.train.len(),
        valid_files: args.valid.len(),
        train_records,
        raw_valid_records,
        retained_valid_records,
        excluded_valid_records: raw_valid_records - retained_valid_records,
        unique_train_network_inputs: train_inputs.len(),
        colliding_network_inputs: colliding_inputs.len(),
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

fn verified_manifest(path: &Path) -> Result<SearchTeacherManifest> {
    let manifest = read_search_teacher_manifest(path)?
        .ok_or_else(|| anyhow!("{} has no HKST manifest", path.display()))?;
    if manifest.format != "HKST0002"
        || manifest.teacher_semantics_version != SEARCH_TEACHER_SEMANTICS_VERSION
        || manifest.teacher_semantics_id != SEARCH_TEACHER_SEMANTICS_ID
    {
        return Err(anyhow!(
            "{} has incompatible teacher semantics",
            path.display()
        ));
    }
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
    if manifest.model.is_none() {
        return Err(anyhow!("{} manifest has no teacher model", path.display()));
    }
    Ok(manifest)
}

fn visit_records(
    path: &Path,
    mut visit: impl FnMut(&SearchTeacherRecord) -> Result<()>,
) -> Result<()> {
    let manifest = verified_manifest(path)?;
    let mut reader =
        SearchTeacherReader::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut records = 0u64;
    while let Some(record) = reader.read_record()? {
        visit(&record)?;
        records += 1;
    }
    if records != manifest.records {
        return Err(anyhow!(
            "{} manifest declares {} records but read {}",
            path.display(),
            manifest.records,
            records
        ));
    }
    Ok(())
}

fn record_inputs(record: &SearchTeacherRecord) -> Vec<[u8; 32]> {
    std::iter::once(&record.root)
        .chain(record.candidates.iter().map(|candidate| &candidate.child))
        .map(PackedHalfKpPosition::network_input_hash)
        .collect()
}

fn insert_record_inputs(record: &SearchTeacherRecord, inputs: &mut HashSet<[u8; 32]>) {
    inputs.extend(record_inputs(record));
}
