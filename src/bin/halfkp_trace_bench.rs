use anyhow::{anyhow, Result};
use clap::Parser;
use shogi_ai::evaluation::{HalfKpModel, SparseModel};
use shogi_ai::utils::position_from_sfen_or_usi;
use shogi_core::Move;
use shogi_lib::Position;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Replay identical move traces through KPP and incremental HalfKP")]
struct Args {
    #[arg(long, default_value = "policy_weights_v2.1.0.binary")]
    kpp_weights: PathBuf,
    #[arg(long)]
    halfkp_weights: PathBuf,
    #[arg(long, default_value = "taya36.sfen")]
    positions: PathBuf,
    #[arg(long, default_value_t = 256)]
    traces: usize,
    #[arg(long, default_value_t = 80)]
    plies: usize,
}

fn load_positions(path: &PathBuf) -> Result<Vec<Position>> {
    let content = fs::read_to_string(path)?;
    let positions = content
        .lines()
        .filter_map(position_from_sfen_or_usi)
        .collect::<Vec<_>>();
    if positions.is_empty() {
        return Err(anyhow!("no valid positions in {}", path.display()));
    }
    Ok(positions)
}

fn make_traces(starts: &[Position], count: usize, plies: usize) -> Vec<(Position, Vec<Move>)> {
    (0..count)
        .map(|i| {
            let root = starts[i % starts.len()].clone();
            let mut pos = root.clone();
            let mut moves = Vec::with_capacity(plies);
            for ply in 0..plies {
                let legal = pos.legal_moves();
                if legal.is_empty() {
                    break;
                }
                let mv = legal[(i.wrapping_mul(31) + ply.wrapping_mul(17)) % legal.len()];
                moves.push(mv);
                pos.do_move(mv);
            }
            (root, moves)
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let starts = load_positions(&args.positions)?;
    let traces = make_traces(&starts, args.traces, args.plies);
    let mut kpp = SparseModel::new(0.0, 0.0);
    kpp.load(&args.kpp_weights)?;
    let halfkp = HalfKpModel::load(&args.halfkp_weights)?;

    let start = Instant::now();
    let mut kpp_sum = 0.0f32;
    let mut kpp_evals = 0usize;
    for (root, moves) in &traces {
        let mut pos = root.clone();
        for &mv in moves {
            pos.do_move(mv);
            kpp_sum += kpp.predict_from_position(&pos);
            kpp_evals += 1;
        }
    }
    let kpp_elapsed = start.elapsed();

    let start = Instant::now();
    let mut halfkp_sum = 0.0f32;
    let mut halfkp_evals = 0usize;
    let mut max_context_error = 0.0f32;
    for (root, moves) in &traces {
        let mut pos = root.clone();
        let mut ctx = halfkp
            .begin_search_context(&pos)
            .ok_or_else(|| anyhow!("invalid root"))?;
        for &mv in moves {
            halfkp.prepare_search_context(&mut ctx, &pos, mv);
            pos.do_move(mv);
            halfkp.commit_search_context_move(&mut ctx, &pos);
            let context_score = halfkp.evaluate_search_context(&pos, &ctx);
            if halfkp_evals < 1024 {
                let full_score = halfkp.predict_from_position(&pos);
                let error = (context_score - full_score).abs();
                max_context_error = max_context_error.max(error);
            }
            halfkp_sum += context_score;
            halfkp_evals += 1;
        }
    }
    let halfkp_elapsed = start.elapsed();
    black_box(kpp_sum);
    black_box(halfkp_sum);
    let kpp_rate = kpp_evals as f64 / kpp_elapsed.as_secs_f64();
    let halfkp_rate = halfkp_evals as f64 / halfkp_elapsed.as_secs_f64();
    println!(
        "traces={} plies={} evals={}",
        traces.len(),
        args.plies,
        kpp_evals
    );
    println!(
        "kpp: {:.2} eval/s elapsed_ms={:.2} checksum={:.3}",
        kpp_rate,
        kpp_elapsed.as_secs_f64() * 1000.0,
        kpp_sum
    );
    println!(
        "halfkp_incremental: {:.2} eval/s elapsed_ms={:.2} checksum={:.3}",
        halfkp_rate,
        halfkp_elapsed.as_secs_f64() * 1000.0,
        halfkp_sum
    );
    println!("speedup: {:.3}x", halfkp_rate / kpp_rate);
    println!("max_incremental_vs_full_error: {:.6}", max_context_error);
    Ok(())
}
