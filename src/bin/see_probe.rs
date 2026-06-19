use anyhow::{anyhow, Result};
use clap::Parser;
use shogi_ai::utils::{format_move_usi, get_piece_value, position_from_sfen_or_usi};
use shogi_core::Move;
use shogi_lib::Position;
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Compare one-ply SEE with recursive capture SEE on SFEN positions")]
struct Args {
    #[arg(long)]
    positions: PathBuf,
    #[arg(long, default_value_t = 16)]
    max_depth: u8,
    #[arg(long, default_value_t = 20)]
    top: usize,
    #[arg(long, default_value_t = 0)]
    limit: usize,
}

#[derive(Debug)]
struct DiffRecord {
    delta: i32,
    old_see: i32,
    new_see: i32,
    idx: usize,
    move_text: String,
    sfen: String,
}

fn old_see(position: &Position, mv: Move) -> Option<i32> {
    if let Move::Normal { from, to, .. } = mv {
        let attacker = position.piece_at(from)?;
        let victim = position.piece_at(to)?;
        Some(get_piece_value(victim.piece_kind()) - get_piece_value(attacker.piece_kind()))
    } else {
        None
    }
}

fn best_capture_gain(position: &mut Position, target: shogi_core::Square, depth: u8) -> i32 {
    if depth == 0 {
        return 0;
    }

    let mut best = 0;
    for mv in position.legal_moves() {
        let Move::Normal { to, .. } = mv else {
            continue;
        };
        if to != target {
            continue;
        }
        let Some(victim) = position.piece_at(target) else {
            continue;
        };
        let captured_value = get_piece_value(victim.piece_kind());
        position.do_move(mv);
        let reply_gain = best_capture_gain(position, target, depth - 1).max(0);
        position.undo_move(mv);
        best = best.max(captured_value - reply_gain);
    }
    best
}

fn recursive_see(position: &Position, mv: Move, max_depth: u8) -> Option<i32> {
    let Move::Normal { to, .. } = mv else {
        return None;
    };
    let victim = position.piece_at(to)?;
    let captured_value = get_piece_value(victim.piece_kind());
    let mut child = position.clone();
    child.do_move(mv);
    let reply_gain = best_capture_gain(&mut child, to, max_depth.saturating_sub(1)).max(0);
    Some(captured_value - reply_gain)
}

fn is_qsearch_candidate(position: &Position, mv: Move) -> bool {
    match mv {
        Move::Normal { to, .. } => position.piece_at(to).is_some() || position.is_check_move(mv),
        Move::Drop { .. } => position.is_check_move(mv),
    }
}

fn sign(value: i32) -> i32 {
    value.signum()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let content = fs::read_to_string(&args.positions)
        .map_err(|e| anyhow!("failed to read {}: {}", args.positions.display(), e))?;

    let mut positions = 0usize;
    let mut qsearch_candidates = 0usize;
    let mut captures = 0usize;
    let mut sign_flips = 0usize;
    let mut old_negative_new_nonnegative = 0usize;
    let mut old_nonnegative_new_negative = 0usize;
    let mut diff_records = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if args.limit > 0 && positions >= args.limit {
            break;
        }
        let Some(position) = position_from_sfen_or_usi(line) else {
            eprintln!("skip invalid position: {line}");
            continue;
        };
        positions += 1;

        for mv in position.legal_moves() {
            if !is_qsearch_candidate(&position, mv) {
                continue;
            }
            qsearch_candidates += 1;
            let Some(old_value) = old_see(&position, mv) else {
                continue;
            };
            captures += 1;
            let Some(new_value) = recursive_see(&position, mv, args.max_depth) else {
                continue;
            };
            if sign(old_value) != sign(new_value) {
                sign_flips += 1;
            }
            if old_value < 0 && new_value >= 0 {
                old_negative_new_nonnegative += 1;
            }
            if old_value >= 0 && new_value < 0 {
                old_nonnegative_new_negative += 1;
            }
            let delta = (new_value - old_value).abs();
            if delta > 0 {
                diff_records.push(DiffRecord {
                    delta,
                    old_see: old_value,
                    new_see: new_value,
                    idx: positions,
                    move_text: format_move_usi(mv),
                    sfen: position.to_sfen_owned(),
                });
            }
        }
    }

    diff_records.sort_by(|a, b| b.delta.cmp(&a.delta).then_with(|| a.idx.cmp(&b.idx)));

    println!("positions: {}", positions);
    println!("qsearch candidates: {}", qsearch_candidates);
    println!("capture candidates: {}", captures);
    println!("see sign flips: {}", sign_flips);
    println!(
        "old negative -> new nonnegative: {}",
        old_negative_new_nonnegative
    );
    println!(
        "old nonnegative -> new negative: {}",
        old_nonnegative_new_negative
    );
    println!("top see differences:");
    for record in diff_records.iter().take(args.top) {
        println!(
            "  idx={} move={} old={} new={} delta={}",
            record.idx, record.move_text, record.old_see, record.new_see, record.delta
        );
        println!("    sfen {}", record.sfen);
    }

    Ok(())
}
