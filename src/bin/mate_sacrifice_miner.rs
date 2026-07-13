use anyhow::{anyhow, Result};
use clap::Parser;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use serde_json::json;
use shogi_ai::search_quality::{
    commit_suite_with_manifest, deduplicate_input_positions, ensure_distinct_paths,
    load_input_positions, simple_see, AtomicOutput, DatasetSplit, MateOracle, MateProof, SuiteKind,
};
use shogi_ai::utils::format_move_usi;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Mine negative-SEE checking moves with rule-only mate proofs")]
struct Args {
    #[arg(long)]
    positions: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    split: DatasetSplit,
    #[arg(long, value_delimiter = ',', default_value = "1,3,5,7")]
    depths: Vec<u8>,
    #[arg(long, default_value_t = 2_000_000)]
    proof_node_limit: u64,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 0)]
    position_limit: usize,
    #[arg(long, default_value_t = 0)]
    record_limit: usize,
}

#[derive(Serialize)]
struct MateRecord {
    schema_version: u32,
    source_index: usize,
    source_game_key: Option<String>,
    sfen: String,
    first_move: String,
    simple_see: i32,
    mate_horizon: u8,
    proof_line: Vec<String>,
    proof_line_plies: usize,
    root_defense_count: usize,
    proof_nodes: u64,
    proof_status: &'static str,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.depths.is_empty()
        || args.depths.iter().any(|depth| depth % 2 == 0)
        || args.proof_node_limit == 0
    {
        return Err(anyhow!(
            "--depths must contain odd horizons and --proof-node-limit must be positive"
        ));
    }
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

    'positions: for input in positions.into_iter().take(position_limit) {
        let source_index = input.source_line;
        let source_game_key = input.source_game_key;
        let position = input.position;
        let sfen = position.to_sfen_owned();
        let attacker = position.side_to_move();
        let candidates: Vec<_> = position
            .legal_moves()
            .iter()
            .copied()
            .filter(|&mv| position.is_check_move(mv) && simple_see(&position, mv) < 0)
            .collect();
        for first_move in candidates {
            for &horizon in &args.depths {
                let mut child = position.clone();
                child.do_move(first_move);
                let root_defense_count = child.legal_moves().len();
                let mut oracle = MateOracle::new(args.proof_node_limit);
                let result = oracle.prove(&mut child, attacker, horizon.saturating_sub(1));
                if let MateProof::ProvenMate(mut line) = result {
                    line.insert(0, first_move);
                    let record = MateRecord {
                        schema_version: 1,
                        source_index,
                        source_game_key: source_game_key.clone(),
                        sfen: sfen.clone(),
                        first_move: format_move_usi(first_move),
                        simple_see: simple_see(&position, first_move),
                        mate_horizon: horizon,
                        proof_line_plies: line.len(),
                        proof_line: line.into_iter().map(format_move_usi).collect(),
                        root_defense_count,
                        proof_nodes: oracle.nodes(),
                        proof_status: "proven_mate",
                    };
                    serde_json::to_writer(&mut writer, &record)?;
                    writeln!(writer)?;
                    written += 1;
                    if args.record_limit > 0 && written >= args.record_limit {
                        break 'positions;
                    }
                    continue 'positions;
                }
            }
        }
    }
    if written == 0 {
        return Err(anyhow!("no proven mate-sacrifice records were generated"));
    }
    commit_suite_with_manifest(
        writer,
        "mate_sacrifice_miner",
        SuiteKind::MateSacrifice,
        args.split,
        &args.positions,
        args.seed,
        input_nonempty_lines,
        valid_positions,
        written,
        duplicates,
        json!({
            "depths": args.depths,
            "proof_node_limit": args.proof_node_limit,
            "position_limit": args.position_limit,
            "record_limit": args.record_limit,
            "candidate": "legal checking move with current simple SEE < 0",
            "unknown_policy": "excluded"
        }),
    )?;
    eprintln!("records written: {written}");
    Ok(())
}
