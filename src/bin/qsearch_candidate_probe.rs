use anyhow::{anyhow, Result};
use clap::Parser;
use shogi_ai::utils::{format_move_usi, position_from_sfen_or_usi};
use shogi_core::Move;
use shogi_lib::Position;
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Compare qsearch candidate move sets with the reference filter")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long, default_value_t = 0)]
    max_positions: usize,
    #[arg(long, default_value_t = 20)]
    show: usize,
}

fn reference_quiescence_moves(position: &Position) -> Vec<String> {
    let mut moves = position.legal_moves();
    moves.retain(|m| {
        if let Move::Normal { to, .. } = *m {
            position.piece_at(to).is_some() || position.is_check_move(*m)
        } else {
            position.is_check_move(*m)
        }
    });
    sorted_move_texts(moves.into_iter())
}

fn current_quiescence_moves(position: &Position) -> Vec<String> {
    sorted_move_texts(position.legal_quiescence_moves().into_iter())
}

fn sorted_move_texts(moves: impl Iterator<Item = Move>) -> Vec<String> {
    let mut moves = moves.map(format_move_usi).collect::<Vec<_>>();
    moves.sort();
    moves
}

fn load_positions(
    paths: &[PathBuf],
    max_positions: usize,
) -> Result<Vec<(PathBuf, usize, Position)>> {
    let mut positions = Vec::new();
    for path in paths {
        let content = fs::read_to_string(path)
            .map_err(|e| anyhow!("failed to read {}: {}", path.display(), e))?;
        for (line_index, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Some(position) = position_from_sfen_or_usi(line) else {
                return Err(anyhow!(
                    "{}:{} invalid position",
                    path.display(),
                    line_index + 1
                ));
            };
            positions.push((path.clone(), line_index + 1, position));
            if max_positions > 0 && positions.len() >= max_positions {
                return Ok(positions);
            }
        }
    }
    if positions.is_empty() {
        return Err(anyhow!("no positions loaded"));
    }
    Ok(positions)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let positions = load_positions(&args.input, args.max_positions)?;
    let mut mismatches = 0usize;
    let mut shown = 0usize;
    let mut total_reference = 0usize;
    let mut total_current = 0usize;

    for (path, line, position) in &positions {
        let reference = reference_quiescence_moves(position);
        let current = current_quiescence_moves(position);
        total_reference += reference.len();
        total_current += current.len();
        if reference != current {
            mismatches += 1;
            if shown < args.show {
                shown += 1;
                let missing = reference
                    .iter()
                    .filter(|mv| !current.contains(mv))
                    .cloned()
                    .collect::<Vec<_>>();
                let extra = current
                    .iter()
                    .filter(|mv| !reference.contains(mv))
                    .cloned()
                    .collect::<Vec<_>>();
                println!(
                    "mismatch {}:{} ref={} current={} in_check={} side={:?}",
                    path.display(),
                    line,
                    reference.len(),
                    current.len(),
                    position.in_check(),
                    position.side_to_move()
                );
                println!("  missing={}", missing.join(","));
                println!("  extra={}", extra.join(","));
                println!("  sfen {}", position.to_sfen_owned());
            }
        }
    }

    println!("positions: {}", positions.len());
    println!("mismatches: {}", mismatches);
    println!("reference candidates: {}", total_reference);
    println!("current candidates: {}", total_current);
    if mismatches > 0 {
        return Err(anyhow!("qsearch candidate mismatches found"));
    }
    Ok(())
}
