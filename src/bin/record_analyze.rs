use anyhow::{anyhow, Context, Result};
use clap::Parser;
use glob::glob;
use shogi_ai::evaluation::SparseModel;
use shogi_ai::utils::{parse_usi_move, position_from_sfen_or_usi};
use shogi_core::{Color, Move, Piece};
use shogi_lib::Position;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Analyze saved benchmark game records")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long)]
    record_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 8)]
    tail_plies: usize,
    #[arg(long, default_value_t = 0)]
    top_drops: usize,
    #[arg(required_unless_present = "record_dir")]
    records: Vec<PathBuf>,
}

#[derive(Debug)]
struct Record {
    path: PathBuf,
    result: String,
    reason: Option<String>,
    new_as: Option<Color>,
    start_sfen: Option<String>,
    final_position: Position,
    positions: Vec<Position>,
    plies: usize,
}

#[derive(Debug)]
struct TailSummary {
    text: String,
    worst_drop: Option<(usize, f32)>,
}

#[derive(Debug)]
struct DropRecord {
    drop: f32,
    ply: usize,
    path: PathBuf,
    result: String,
    reason: String,
    final_score: Option<f32>,
}

fn load_model(path: &Path) -> Result<SparseModel> {
    let mut model = SparseModel::new(0.0, 0.0);
    model
        .load(path)
        .map_err(|e| anyhow!("failed to load {}: {}", path.display(), e))?;
    Ok(model)
}

fn parse_side(side: &str) -> Option<Color> {
    match side {
        "black" => Some(Color::Black),
        "white" => Some(Color::White),
        _ => None,
    }
}

fn parse_move_for_position(position: &Position, move_text: &str) -> Option<Move> {
    parse_usi_move(move_text).map(|mv| match mv {
        Move::Drop { piece, to } => Move::Drop {
            piece: Piece::new(piece.piece_kind(), position.side_to_move()),
            to,
        },
        normal => normal,
    })
}

fn parse_position_command(command: &str) -> Result<Vec<Position>> {
    let rest = command
        .trim()
        .strip_prefix("position ")
        .ok_or_else(|| anyhow!("record position line must start with 'position '"))?;
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    if tokens.is_empty() {
        return Err(anyhow!("empty position command"));
    }

    let (start_text, move_tokens): (String, &[&str]) = if tokens[0] == "startpos" {
        let moves_idx = tokens.iter().position(|&token| token == "moves");
        let move_tokens = moves_idx.map(|idx| &tokens[idx + 1..]).unwrap_or(&[]);
        ("startpos".to_string(), move_tokens)
    } else if tokens[0] == "sfen" {
        let moves_idx = tokens
            .iter()
            .position(|&token| token == "moves")
            .unwrap_or(tokens.len());
        let move_tokens = if moves_idx < tokens.len() {
            &tokens[moves_idx + 1..]
        } else {
            &[]
        };
        (tokens[..moves_idx].join(" "), move_tokens)
    } else {
        return Err(anyhow!("unsupported position command: {}", command));
    };

    let mut position = position_from_sfen_or_usi(&start_text)
        .ok_or_else(|| anyhow!("invalid start position: {}", start_text))?;
    let mut positions = vec![position.clone()];
    for move_text in move_tokens {
        let mv = parse_move_for_position(&position, move_text)
            .ok_or_else(|| anyhow!("invalid move '{}' in {}", move_text, command))?;
        if !position.legal_moves().contains(&mv) {
            return Err(anyhow!("illegal move '{}' in {}", move_text, command));
        }
        position.do_move(mv);
        positions.push(position.clone());
    }

    Ok(positions)
}

fn load_record(path: &Path) -> Result<Record> {
    let content =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut result = None;
    let mut reason = None;
    let mut new_as = None;
    let mut start_sfen = None;
    let mut final_position = None;
    let mut positions = None;
    let mut plies = 0;

    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("result ") {
            result = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("reason ") {
            reason = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("new_as ") {
            new_as = parse_side(rest.trim());
        } else if let Some(rest) = line.strip_prefix("start_sfen ") {
            start_sfen = Some(rest.trim().to_string());
        } else if line.starts_with("position ") {
            let parsed_positions = parse_position_command(line)?;
            plies = parsed_positions.len().saturating_sub(1);
            final_position = parsed_positions.last().cloned();
            positions = Some(parsed_positions);
        }
    }

    Ok(Record {
        path: path.to_path_buf(),
        result: result.unwrap_or_else(|| "Unknown".to_string()),
        reason,
        new_as,
        start_sfen,
        final_position: final_position
            .ok_or_else(|| anyhow!("missing position line in {}", path.display()))?,
        positions: positions.ok_or_else(|| anyhow!("missing position line in {}", path.display()))?,
        plies,
    })
}

fn collect_records(args: &Args) -> Result<Vec<PathBuf>> {
    let mut paths = args.records.clone();
    if let Some(record_dir) = &args.record_dir {
        let pattern = record_dir.join("*.usi");
        let pattern = pattern
            .to_str()
            .ok_or_else(|| anyhow!("record path is not valid UTF-8: {}", pattern.display()))?;
        for entry in glob(pattern)? {
            paths.push(entry?);
        }
    }
    paths.sort();
    paths.dedup();
    if paths.is_empty() {
        return Err(anyhow!("no record files found"));
    }
    Ok(paths)
}

fn score_for_new_position(model: &SparseModel, position: &Position, new_as: Color) -> f32 {
    let score = model.predict_from_position(position);
    let new_to_move = match position.side_to_move() {
        Color::Black => new_as == Color::Black,
        Color::White => new_as == Color::White,
    };
    if new_to_move { score } else { -score }
}

fn score_for_new(model: &SparseModel, record: &Record) -> Option<f32> {
    let new_as = record.new_as?;
    Some(score_for_new_position(model, &record.final_position, new_as))
}

fn tail_score_summary(
    model: &SparseModel,
    record: &Record,
    tail_plies: usize,
) -> Option<TailSummary> {
    let new_as = record.new_as?;
    if tail_plies == 0 || record.positions.is_empty() {
        return None;
    }

    let start = record.positions.len().saturating_sub(tail_plies + 1);
    let mut scores = Vec::new();
    for (ply, position) in record.positions.iter().enumerate().skip(start) {
        scores.push((ply, score_for_new_position(model, position, new_as)));
    }

    let mut worst_drop: Option<(usize, f32)> = None;
    for window in scores.windows(2) {
        let drop = window[0].1 - window[1].1;
        if drop > 0.0 && worst_drop.map(|(_, best)| drop > best).unwrap_or(true) {
            worst_drop = Some((window[1].0, drop));
        }
    }

    let compact_scores = scores
        .iter()
        .map(|(ply, score)| format!("{ply}:{score:.0}"))
        .collect::<Vec<_>>()
        .join(",");
    let worst_drop_text = worst_drop
        .map(|(ply, drop)| format!(" worst_drop=ply{ply}:{drop:.0}"))
        .unwrap_or_default();
    Some(TailSummary {
        text: format!("tail_scores=[{}]{}", compact_scores, worst_drop_text),
        worst_drop,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let model = load_model(&args.weights)?;
    let paths = collect_records(&args)?;

    let mut new_wins = 0usize;
    let mut baseline_wins = 0usize;
    let mut draws = 0usize;
    let mut scored_records = 0usize;
    let mut score_sum = 0.0f32;
    let mut new_win_score_sum = 0.0f32;
    let mut new_win_scored = 0usize;
    let mut baseline_win_score_sum = 0.0f32;
    let mut baseline_win_scored = 0usize;
    let mut score_result_mismatches = 0usize;
    let mut reason_counts = BTreeMap::<String, usize>::new();
    let mut paired_results = BTreeMap::<String, (usize, usize, usize)>::new();
    let mut drop_records = Vec::new();

    for path in paths {
        let record = load_record(&path)?;
        let reason = record
            .reason
            .clone()
            .unwrap_or_else(|| "Unknown".to_string());
        *reason_counts.entry(reason.clone()).or_insert(0) += 1;
        match record.result.as_str() {
            "NewWin" => new_wins += 1,
            "BaselineWin" => baseline_wins += 1,
            "Draw" => draws += 1,
            _ => {}
        }
        if let Some(start_sfen) = &record.start_sfen {
            let entry = paired_results.entry(start_sfen.clone()).or_insert((0, 0, 0));
            match record.result.as_str() {
                "NewWin" => entry.0 += 1,
                "BaselineWin" => entry.1 += 1,
                "Draw" => entry.2 += 1,
                _ => {}
            }
        }
        let raw_score = score_for_new(&model, &record);
        if let Some(score) = raw_score {
            scored_records += 1;
            score_sum += score;
            match record.result.as_str() {
                "NewWin" => {
                    new_win_scored += 1;
                    new_win_score_sum += score;
                    if score < 0.0 {
                        score_result_mismatches += 1;
                    }
                }
                "BaselineWin" => {
                    baseline_win_scored += 1;
                    baseline_win_score_sum += score;
                    if score > 0.0 {
                        score_result_mismatches += 1;
                    }
                }
                _ => {}
            }
        }
        let score = raw_score
            .map(|score| format!("{score:.1}"))
            .unwrap_or_else(|| "n/a".to_string());
        let tail_summary = tail_score_summary(&model, &record, args.tail_plies);
        if let Some(summary) = &tail_summary {
            if let Some((ply, drop)) = summary.worst_drop {
                drop_records.push(DropRecord {
                    drop,
                    ply,
                    path: record.path.clone(),
                    result: record.result.clone(),
                    reason: reason.clone(),
                    final_score: raw_score,
                });
            }
        }
        let tail_text = tail_summary
            .map(|summary| summary.text)
            .unwrap_or_else(|| "tail_scores=n/a".to_string());
        println!(
            "{} result={} reason={} new_as={} plies={} final_score_for_new={} {}",
            record
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("<unknown>"),
            record.result,
            reason,
            record
                .new_as
                .map(|side| if side == Color::Black { "black" } else { "white" })
                .unwrap_or("unknown"),
            record.plies,
            score,
            tail_text
        );
    }

    println!("new wins: {}", new_wins);
    println!("baseline wins: {}", baseline_wins);
    println!("draws: {}", draws);
    if !reason_counts.is_empty() {
        println!("end reasons:");
        for (reason, count) in reason_counts {
            println!("  {}: {}", reason, count);
        }
    }
    if !paired_results.is_empty() {
        let mut new_sweeps = 0usize;
        let mut baseline_sweeps = 0usize;
        let mut splits = 0usize;
        let mut draw_pairs = 0usize;
        for (_, (new_pair_wins, baseline_pair_wins, pair_draws)) in paired_results {
            if new_pair_wins > 0 && baseline_pair_wins == 0 && pair_draws == 0 {
                new_sweeps += 1;
            } else if baseline_pair_wins > 0 && new_pair_wins == 0 && pair_draws == 0 {
                baseline_sweeps += 1;
            } else if new_pair_wins > 0 && baseline_pair_wins > 0 {
                splits += 1;
            } else {
                draw_pairs += 1;
            }
        }
        println!("paired starts:");
        println!("  new sweeps: {}", new_sweeps);
        println!("  baseline sweeps: {}", baseline_sweeps);
        println!("  splits: {}", splits);
        println!("  draw/mixed pairs: {}", draw_pairs);
    }
    if scored_records > 0 {
        println!(
            "average final score for new: {:.1}",
            score_sum / scored_records as f32
        );
    }
    if new_win_scored > 0 {
        println!(
            "average final score for NewWin: {:.1}",
            new_win_score_sum / new_win_scored as f32
        );
    }
    if baseline_win_scored > 0 {
        println!(
            "average final score for BaselineWin: {:.1}",
            baseline_win_score_sum / baseline_win_scored as f32
        );
    }
    if args.top_drops > 0 && !drop_records.is_empty() {
        drop_records.sort_by(|a, b| {
            b.drop
                .partial_cmp(&a.drop)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        println!("largest tail drops:");
        for record in drop_records.iter().take(args.top_drops) {
            let final_score = record
                .final_score
                .map(|score| format!("{score:.1}"))
                .unwrap_or_else(|| "n/a".to_string());
            println!(
                "  {} drop=ply{}:{:.0} result={} reason={} final_score_for_new={}",
                record
                    .path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("<unknown>"),
                record.ply,
                record.drop,
                record.result,
                record.reason,
                final_score
            );
        }
    }
    println!("score/result sign mismatches: {}", score_result_mismatches);

    Ok(())
}
