use anyhow::{anyhow, Result};
use clap::Parser;
use shogi_ai::utils::{format_move_usi, position_from_sfen_or_usi};
use shogi_lib::Position;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Count legal move tree nodes from a position")]
struct Args {
    #[arg(long, default_value = "startpos")]
    position: String,
    #[arg(long)]
    depth: u8,
    #[arg(long, default_value_t = false)]
    divide: bool,
}

fn perft(position: &mut Position, depth: u8) -> u64 {
    if depth == 0 {
        return 1;
    }

    let moves = position.legal_moves();
    if depth == 1 {
        return moves.len() as u64;
    }

    let mut nodes = 0;
    for mv in moves {
        position.do_move(mv);
        nodes += perft(position, depth - 1);
        position.undo_move(mv);
    }
    nodes
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut position = position_from_sfen_or_usi(&args.position)
        .ok_or_else(|| anyhow!("invalid position: {}", args.position))?;

    let start = Instant::now();
    if args.divide && args.depth > 0 {
        let moves = position.legal_moves();
        let mut total = 0;
        for mv in moves {
            position.do_move(mv);
            let nodes = perft(&mut position, args.depth - 1);
            position.undo_move(mv);
            total += nodes;
            println!("{} {}", format_move_usi(mv), nodes);
        }
        let elapsed = start.elapsed().as_secs_f64();
        println!("nodes: {}", total);
        println!("elapsed ms: {:.2}", elapsed * 1000.0);
        if elapsed > 0.0 {
            println!("nodes/sec: {:.2}", total as f64 / elapsed);
        }
        return Ok(());
    }

    let nodes = perft(&mut position, args.depth);
    let elapsed = start.elapsed().as_secs_f64();
    println!("nodes: {}", nodes);
    println!("elapsed ms: {:.2}", elapsed * 1000.0);
    if elapsed > 0.0 {
        println!("nodes/sec: {:.2}", nodes as f64 / elapsed);
    }

    Ok(())
}
