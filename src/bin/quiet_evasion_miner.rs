use anyhow::Result;
use clap::Parser;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use serde_json::json;
use shogi_ai::search_quality::{
    commit_suite_with_manifest, deduplicate_input_positions, ensure_distinct_paths,
    load_input_positions, AtomicOutput, DatasetSplit, SuiteKind,
};
use shogi_ai::utils::format_move_usi;
use shogi_core::Move;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Mine in-check positions containing quiet legal evasions")]
struct Args {
    #[arg(long)]
    positions: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    split: DatasetSplit,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 0)]
    position_limit: usize,
    #[arg(long, default_value_t = 0)]
    record_limit: usize,
}

#[derive(Serialize)]
struct EvasionRecord {
    schema_version: u32,
    source_index: usize,
    source_game_key: Option<String>,
    sfen: String,
    legal_evasions: Vec<String>,
    quiet_evasions: Vec<String>,
    legal_evasion_count: usize,
    quiet_evasion_count: usize,
}

fn is_quiet(position: &shogi_lib::Position, mv: Move) -> bool {
    let capture = match mv {
        Move::Normal { to, .. } => position.piece_at(to).is_some(),
        Move::Drop { .. } => false,
    };
    !capture && !position.is_check_move(mv)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let sidecar = args.output.with_extension("manifest.json");
    ensure_distinct_paths(&[
        ("positions", &args.positions),
        ("output", &args.output),
        ("sidecar", &sidecar),
    ])?;
    let (positions, input_nonempty_lines) = load_input_positions(&args.positions)?;
    let valid_positions = positions.len();
    let (mut positions, duplicates) = deduplicate_input_positions(positions);
    positions.shuffle(&mut ChaCha8Rng::seed_from_u64(args.seed));
    let mut writer = AtomicOutput::new(&args.output)?;
    let mut written = 0usize;
    let position_limit = if args.position_limit == 0 {
        usize::MAX
    } else {
        args.position_limit
    };

    for input in positions.into_iter().take(position_limit) {
        let source_index = input.source_line;
        let source_game_key = input.source_game_key;
        let position = input.position;
        if !position.in_check() {
            continue;
        }
        let sfen = position.to_sfen_owned();
        let legal = position.legal_moves();
        let quiet: Vec<_> = legal
            .iter()
            .copied()
            .filter(|&mv| is_quiet(&position, mv))
            .collect();
        if quiet.is_empty() {
            continue;
        }
        let record = EvasionRecord {
            schema_version: 1,
            source_index,
            source_game_key,
            sfen,
            legal_evasion_count: legal.len(),
            quiet_evasion_count: quiet.len(),
            legal_evasions: legal.iter().copied().map(format_move_usi).collect(),
            quiet_evasions: quiet.into_iter().map(format_move_usi).collect(),
        };
        serde_json::to_writer(&mut writer, &record)?;
        writeln!(writer)?;
        written += 1;
        if args.record_limit > 0 && written >= args.record_limit {
            break;
        }
    }
    if written == 0 {
        return Err(anyhow::anyhow!("no quiet-evasion records were generated"));
    }
    commit_suite_with_manifest(
        writer,
        "quiet_evasion_miner",
        SuiteKind::QuietEvasion,
        args.split,
        &args.positions,
        args.seed,
        input_nonempty_lines,
        valid_positions,
        written,
        duplicates,
        json!({
            "position_limit": args.position_limit,
            "record_limit": args.record_limit,
            "reference": "complete legal_moves set while in check",
            "quiet": "non-capture and non-check"
        }),
    )?;
    eprintln!("records written: {written}");
    Ok(())
}
