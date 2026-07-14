use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::Value;
use shogi_ai::utils::position_from_sfen_or_usi;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Generate deterministic legal synthetic SFENs from a source pool")]
struct Args {
    #[arg(long)]
    positions: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 80_000)]
    count: usize,
    #[arg(long, default_value_t = 20260714)]
    seed: u64,
    #[arg(long, default_value_t = 3)]
    max_plies: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.count == 0 || args.max_plies == 0 {
        return Err(anyhow!("--count and --max-plies must be positive"));
    }
    let input = BufReader::new(
        File::open(&args.positions)
            .with_context(|| format!("failed to open {}", args.positions.display()))?,
    );
    let mut source = Vec::new();
    for (line_no, line) in input.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)
            .with_context(|| format!("invalid JSON at source line {}", line_no + 1))?;
        let sfen = row["sfen"]
            .as_str()
            .ok_or_else(|| anyhow!("missing sfen at source line {}", line_no + 1))?;
        let game_key = row["game_key"]
            .as_str()
            .or_else(|| row["source_game_key"].as_str())
            .ok_or_else(|| anyhow!("missing game key at source line {}", line_no + 1))?;
        let position = position_from_sfen_or_usi(sfen)
            .ok_or_else(|| anyhow!("invalid SFEN at source line {}", line_no + 1))?;
        source.push((line_no + 1, game_key.to_owned(), position));
    }
    if source.is_empty() {
        return Err(anyhow!("source pool is empty"));
    }

    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let mut seen = HashSet::new();
    let mut writer = BufWriter::new(File::create(&args.output)?);
    let mut written = 0usize;
    let mut attempts = 0usize;
    while written < args.count && attempts < args.count.saturating_mul(20) {
        attempts += 1;
        let (source_line, game_key, mut position) = source[rng.gen_range(0..source.len())].clone();
        let plies = rng.gen_range(1..=args.max_plies);
        let mut applied = 0usize;
        for _ in 0..plies {
            let moves = position.legal_moves();
            if moves.is_empty() {
                break;
            }
            let mv = moves[rng.gen_range(0..moves.len())];
            position.do_move(mv);
            applied += 1;
        }
        if applied == 0 {
            continue;
        }
        let sfen = position.to_sfen_owned();
        if !seen.insert(sfen.clone()) {
            continue;
        }
        let synthetic_id = 1_000_000usize + written;
        let row = serde_json::json!({
            "schema": "synthetic_mate_source_v1",
            "source_index": synthetic_id,
            "source_game_key": format!("{}#synthetic-v2-{}", game_key, synthetic_id),
            "game_key": format!("{}#synthetic-v2-{}", game_key, synthetic_id),
            "sfen": sfen,
            "synthetic": true,
            "synthetic_from_source_line": source_line,
            "synthetic_plies": applied,
            "synthetic_seed": args.seed
        });
        serde_json::to_writer(&mut writer, &row)?;
        writeln!(writer)?;
        written += 1;
    }
    writer.flush()?;
    if written < args.count {
        return Err(anyhow!(
            "generated only {written} unique synthetic positions after {attempts} attempts"
        ));
    }
    eprintln!("synthetic records written: {written}");
    Ok(())
}
