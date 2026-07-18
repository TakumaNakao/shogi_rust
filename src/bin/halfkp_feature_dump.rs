use anyhow::{anyhow, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use shogi_ai::evaluation::{extract_halfkp_features_for, SparseModel};
use shogi_ai::utils::position_from_sfen_or_usi;
use shogi_core::Color;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Dump compact HalfKP features from dataset_build JSONL")]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    weights: Option<PathBuf>,
    #[arg(long)]
    max_records: Option<usize>,
}

#[derive(Deserialize)]
struct InputRecord {
    sfen: String,
    side_to_move: String,
    winner: Option<String>,
    result_known: Option<bool>,
    termination: Option<String>,
}

#[derive(Serialize)]
struct OutputRecord {
    schema: &'static str,
    sfen: String,
    side_to_move: &'static str,
    features_black: Vec<usize>,
    features_white: Vec<usize>,
    material_black: f32,
    material_white: f32,
    result: f32,
    result_known: bool,
    termination: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    static_eval: Option<f32>,
}

fn parse_color(text: &str) -> Result<Color> {
    match text {
        "black" | "b" => Ok(Color::Black),
        "white" | "w" => Ok(Color::White),
        _ => Err(anyhow!("invalid color: {text}")),
    }
}

fn result_for(winner: Option<&str>, side_to_move: Color) -> f32 {
    let Some(winner) = winner else { return 0.5 };
    match winner {
        "black" => (side_to_move == Color::Black) as u8 as f32,
        "white" => (side_to_move == Color::White) as u8 as f32,
        _ => 0.5,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let sparse = if let Some(path) = args.weights.as_ref() {
        let mut model = SparseModel::new(0.0, 0.0);
        model.load(path)?;
        Some(model)
    } else {
        None
    };
    let input = BufReader::new(File::open(&args.input)?);
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut output = BufWriter::new(File::create(&args.output)?);
    let mut written = 0usize;
    let mut skipped = 0usize;

    for line in input.lines() {
        if args.max_records.is_some_and(|limit| written >= limit) {
            break;
        }
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let input_record: InputRecord = match serde_json::from_str(&line) {
            Ok(record) => record,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let Some(position) = position_from_sfen_or_usi(&input_record.sfen) else {
            skipped += 1;
            continue;
        };
        let Some(black) = extract_halfkp_features_for(&position, Color::Black) else {
            skipped += 1;
            continue;
        };
        let Some(white) = extract_halfkp_features_for(&position, Color::White) else {
            skipped += 1;
            continue;
        };
        let side = parse_color(&input_record.side_to_move)?;
        let static_eval = sparse
            .as_ref()
            .map(|model| model.predict_from_position(&position));
        let record = OutputRecord {
            schema: "halfkp-v1",
            sfen: input_record.sfen,
            side_to_move: if side == Color::Black {
                "black"
            } else {
                "white"
            },
            features_black: black.features,
            features_white: white.features,
            material_black: black.material,
            material_white: white.material,
            result: result_for(input_record.winner.as_deref(), side),
            result_known: input_record
                .result_known
                .unwrap_or_else(|| input_record.winner.is_some()),
            termination: input_record
                .termination
                .unwrap_or_else(|| "legacy".to_string()),
            static_eval,
        };
        serde_json::to_writer(&mut output, &record)?;
        output.write_all(b"\n")?;
        written += 1;
    }
    output.flush()?;
    println!("records: {written}");
    println!("skipped: {skipped}");
    Ok(())
}
