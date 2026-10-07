use anyhow::{anyhow, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use shogi_ai::halfkp_training::PackedHalfKpPosition;
use shogi_ai::utils::{parse_usi_move_for_color, position_from_sfen_or_usi};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Audit provenance, legality, and split isolation of game JSONL")]
struct Args {
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    #[arg(long)]
    expected_license_id: String,
    #[arg(long)]
    expected_source_uri: String,
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Deserialize)]
struct Record {
    schema: String,
    license_id: String,
    source_uri: String,
    game_key: String,
    opening_family: String,
    split: String,
    result_known: bool,
    termination: String,
    sfen: String,
    teacher_move: String,
}

#[derive(Default, Serialize)]
struct Report {
    records: usize,
    records_by_split: HashMap<String, usize>,
    games: usize,
    opening_families: usize,
    unique_network_inputs: usize,
    game_split_collisions: usize,
    opening_family_split_collisions: usize,
    network_input_split_collisions: usize,
    duplicate_network_inputs_within_split: usize,
    illegal_teacher_moves: usize,
    unknown_results: usize,
    non_decisive_results: usize,
    provenance_mismatches: usize,
    schema_mismatches: usize,
    passed: bool,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    let mut report = Report::default();
    let mut games = HashMap::<String, String>::new();
    let mut families = HashMap::<String, String>::new();
    let mut inputs = HashMap::<[u8; 32], String>::new();
    let mut game_collisions = HashSet::new();
    let mut family_collisions = HashSet::new();
    let mut input_collisions = HashSet::new();
    let mut duplicate_inputs = HashSet::new();

    for path in &args.inputs {
        for line in BufReader::new(File::open(path)?).lines() {
            let record: Record = serde_json::from_str(&line?)?;
            report.records += 1;
            *report
                .records_by_split
                .entry(record.split.clone())
                .or_default() += 1;
            report.provenance_mismatches += usize::from(
                record.license_id != args.expected_license_id
                    || record.source_uri != args.expected_source_uri,
            );
            report.schema_mismatches +=
                usize::from(record.schema != "rank_value_dataset_v3_position");
            report.unknown_results += usize::from(!record.result_known);
            report.non_decisive_results += usize::from(record.termination != "decisive");
            insert_split(
                &mut games,
                &mut game_collisions,
                record.game_key,
                &record.split,
            );
            insert_split(
                &mut families,
                &mut family_collisions,
                record.opening_family,
                &record.split,
            );

            let position = match position_from_sfen_or_usi(&record.sfen) {
                Some(position) => position,
                None => {
                    report.illegal_teacher_moves += 1;
                    continue;
                }
            };
            let legal = parse_usi_move_for_color(&record.teacher_move, position.side_to_move())
                .is_some_and(|mv| position.legal_moves().contains(&mv));
            report.illegal_teacher_moves += usize::from(!legal);
            let Some(packed) = PackedHalfKpPosition::from_position(&position) else {
                report.illegal_teacher_moves += 1;
                continue;
            };
            let hash = packed.network_input_hash();
            if let Some(previous) = inputs.insert(hash, record.split.clone()) {
                if previous == record.split {
                    duplicate_inputs.insert((hash, record.split));
                } else {
                    input_collisions.insert(hash);
                }
            }
        }
    }
    report.games = games.len();
    report.opening_families = families.len();
    report.unique_network_inputs = inputs.len();
    report.game_split_collisions = game_collisions.len();
    report.opening_family_split_collisions = family_collisions.len();
    report.network_input_split_collisions = input_collisions.len();
    report.duplicate_network_inputs_within_split = duplicate_inputs.len();
    report.passed = report.game_split_collisions == 0
        && report.opening_family_split_collisions == 0
        && report.network_input_split_collisions == 0
        && report.duplicate_network_inputs_within_split == 0
        && report.illegal_teacher_moves == 0
        && report.unknown_results == 0
        && report.non_decisive_results == 0
        && report.provenance_mismatches == 0
        && report.schema_mismatches == 0;
    let bytes = serde_json::to_vec_pretty(&report)?;
    println!("{}", String::from_utf8_lossy(&bytes));
    if let Some(path) = args.report {
        if path.exists() {
            return Err(anyhow!("refusing to overwrite {}", path.display()));
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, [&bytes[..], b"\n"].concat())?;
    }
    if !report.passed {
        return Err(anyhow!("game dataset audit failed"));
    }
    Ok(())
}

fn insert_split(
    values: &mut HashMap<String, String>,
    collisions: &mut HashSet<String>,
    key: String,
    split: &str,
) {
    if values
        .insert(key.clone(), split.to_string())
        .is_some_and(|old| old != split)
    {
        collisions.insert(key);
    }
}
