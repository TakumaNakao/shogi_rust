use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::Serialize;
use shogi_ai::halfkp_training::CurriculumReader;
use shogi_ai::training_data::DatasetSplit;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Audit HKCL family isolation and network-input split collisions")]
struct Args {
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema: &'static str,
    files: usize,
    records: u64,
    families: usize,
    unique_network_inputs: usize,
    duplicate_network_inputs_within_split: u64,
    family_split_collisions: u64,
    network_input_split_collisions: u64,
    records_by_split: [u64; 3],
    passed: bool,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    let mut family_splits = HashMap::<[u8; 32], DatasetSplit>::new();
    let mut input_splits = HashMap::<[u8; 32], DatasetSplit>::new();
    let mut duplicate_inputs = HashSet::<([u8; 32], u8)>::new();
    let mut family_collisions = HashSet::<[u8; 32]>::new();
    let mut input_collisions = HashSet::<[u8; 32]>::new();
    let mut records = 0u64;
    let mut records_by_split = [0u64; 3];

    for path in &args.inputs {
        let mut reader =
            CurriculumReader::open(path).with_context(|| format!("open {}", path.display()))?;
        while let Some(record) = reader.read_record()? {
            records += 1;
            let split_index = split_index(record.split);
            records_by_split[split_index as usize] += 1;
            if let Some(previous) = family_splits.insert(record.family_hash, record.split) {
                if previous != record.split {
                    family_collisions.insert(record.family_hash);
                }
            }
            if let Some(previous) = input_splits.insert(record.network_input_hash, record.split) {
                if previous != record.split {
                    input_collisions.insert(record.network_input_hash);
                } else {
                    duplicate_inputs.insert((record.network_input_hash, split_index));
                }
            }
            for position in record
                .preferences
                .iter()
                .flat_map(|preference| [&preference.good_child, &preference.bad_child])
            {
                let hash = position.network_input_hash();
                if let Some(previous) = input_splits.insert(hash, record.split) {
                    if previous != record.split {
                        input_collisions.insert(hash);
                    } else {
                        duplicate_inputs.insert((hash, split_index));
                    }
                }
            }
        }
    }
    let passed = family_collisions.is_empty() && input_collisions.is_empty();
    let report = AuditReport {
        schema: "halfkp_curriculum_audit_v1",
        files: args.inputs.len(),
        records,
        families: family_splits.len(),
        unique_network_inputs: input_splits.len(),
        duplicate_network_inputs_within_split: duplicate_inputs.len() as u64,
        family_split_collisions: family_collisions.len() as u64,
        network_input_split_collisions: input_collisions.len() as u64,
        records_by_split,
        passed,
    };
    let bytes = serde_json::to_vec_pretty(&report)?;
    println!("{}", String::from_utf8_lossy(&bytes));
    if let Some(path) = args.report {
        if path.exists() {
            return Err(anyhow!("refusing to overwrite {}", path.display()));
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, [&bytes[..], b"\n"].concat())?;
    }
    if !passed {
        return Err(anyhow!("HKCL split isolation audit failed"));
    }
    Ok(())
}

fn split_index(split: DatasetSplit) -> u8 {
    match split {
        DatasetSplit::Train => 0,
        DatasetSplit::Valid => 1,
        DatasetSplit::Test => 2,
    }
}
