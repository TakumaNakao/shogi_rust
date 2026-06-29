use anyhow::{anyhow, Result};
use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use shogi_ai::utils::position_from_sfen_or_usi;
use shogi_lib::Position;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Profile legal move generation speed")]
struct Args {
    #[arg(long, default_value = "./taya36.sfen")]
    positions: PathBuf,
    #[arg(long, default_value_t = 1024)]
    samples: usize,
    #[arg(long, default_value_t = 1)]
    repeat: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = false)]
    do_undo: bool,
    #[arg(long, default_value_t = false)]
    quiescence: bool,
    #[arg(long, default_value_t = false)]
    captures: bool,
}

fn load_positions(path: &PathBuf) -> Result<Vec<Position>> {
    let content = fs::read_to_string(path)?;
    let positions = content
        .lines()
        .filter_map(position_from_sfen_or_usi)
        .collect::<Vec<_>>();
    if positions.is_empty() {
        return Err(anyhow!("no valid positions loaded from {}", path.display()));
    }
    Ok(positions)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.samples == 0 {
        return Err(anyhow!("--samples must be greater than zero"));
    }
    if args.repeat == 0 {
        return Err(anyhow!("--repeat must be greater than zero"));
    }
    if args.quiescence && args.captures {
        return Err(anyhow!(
            "--quiescence and --captures are mutually exclusive"
        ));
    }

    let mut positions = load_positions(&args.positions)?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    positions.shuffle(&mut rng);

    let start = Instant::now();
    let mut total_positions = 0usize;
    let mut total_moves = 0usize;
    let mut total_generated = 0usize;
    let mut total_do_undo = 0usize;
    let mut max_moves = 0usize;

    for _ in 0..args.repeat {
        for i in 0..args.samples {
            let mut position = positions[i % positions.len()].clone();
            let (moves, generated) = if args.quiescence {
                position.legal_quiescence_moves_with_generated_count()
            } else if args.captures {
                position.legal_capture_moves_with_generated_count()
            } else {
                let moves = position.legal_moves();
                let generated = moves.len();
                (moves, generated)
            };
            total_positions += 1;
            total_moves += moves.len();
            total_generated += generated;
            max_moves = max_moves.max(moves.len());
            if args.do_undo {
                for mv in moves {
                    position.do_move(mv);
                    position.undo_move(mv);
                    total_do_undo += 1;
                }
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!("positions: {}", total_positions);
    println!("legal moves: {}", total_moves);
    println!("generated moves: {}", total_generated);
    println!(
        "discarded moves: {}",
        total_generated.saturating_sub(total_moves)
    );
    println!("max moves: {}", max_moves);
    println!("do/undo moves: {}", total_do_undo);
    println!("elapsed ms: {:.2}", elapsed * 1000.0);
    if elapsed > 0.0 {
        println!("positions/sec: {:.2}", total_positions as f64 / elapsed);
        println!("moves/sec: {:.2}", total_moves as f64 / elapsed);
        println!("generated/sec: {:.2}", total_generated as f64 / elapsed);
        if total_do_undo > 0 {
            println!("do-undo/sec: {:.2}", total_do_undo as f64 / elapsed);
        }
    }
    println!(
        "avg moves/position: {:.2}",
        total_moves as f64 / total_positions.max(1) as f64
    );
    println!(
        "avg generated/position: {:.2}",
        total_generated as f64 / total_positions.max(1) as f64
    );
    println!(
        "discard rate: {:.2}%",
        total_generated.saturating_sub(total_moves) as f64 / total_generated.max(1) as f64 * 100.0
    );

    Ok(())
}
