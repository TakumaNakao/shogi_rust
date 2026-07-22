use anyhow::{anyhow, Result};
use clap::Parser;
use serde::Serialize;
use shogi_ai::halfkp_training::CurriculumReader;
use shogi_ai::opening::classify_opening;
use shogi_core::PartialPosition;
use shogi_lib::Position;
use shogi_usi_parser::FromUsi;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Report deterministic opening strata for HKCL roots")]
struct Args {
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    #[arg(long)]
    json: Option<PathBuf>,
}

#[derive(Serialize)]
struct StratumCount {
    id: u16,
    alias: String,
    records: usize,
}

#[derive(Serialize)]
struct Report {
    records: usize,
    strata: Vec<StratumCount>,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    let mut counts = BTreeMap::<u16, (String, usize)>::new();
    let mut records = 0usize;
    for input in &args.inputs {
        let mut reader = CurriculumReader::open(input)?;
        while let Some(record) = reader.read_record()? {
            let position = Position::new(
                PartialPosition::from_usi(&format!("sfen {}", record.root_sfen))
                    .map_err(|error| anyhow!("parse root SFEN: {error:?}"))?,
            );
            let features = classify_opening(&position);
            let entry = counts
                .entry(features.stratum_id())
                .or_insert_with(|| (features.alias(), 0));
            entry.1 += 1;
            records += 1;
        }
    }
    let report = Report {
        records,
        strata: counts
            .into_iter()
            .map(|(id, (alias, records))| StratumCount { id, alias, records })
            .collect(),
    };
    let bytes = serde_json::to_vec_pretty(&report)?;
    println!("{}", String::from_utf8_lossy(&bytes));
    if let Some(path) = args.json {
        if path.exists() {
            return Err(anyhow!("refusing to overwrite {}", path.display()));
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, [&bytes[..], b"\n"].concat())?;
    }
    Ok(())
}
