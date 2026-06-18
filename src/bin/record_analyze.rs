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
    #[arg(required_unless_present = "record_dir")]
    records: Vec<PathBuf>,
}

#[derive(Debug)]
struct Record {
    path: PathBuf,
    result: String,
    reason: Option<String>,
    new_as: Option<Color>,
    final_position: Position,
    plies: usize,
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

fn parse_position_command(command: &str) -> Result<(Position, usize)> {
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
    for move_text in move_tokens {
        let mv = parse_move_for_position(&position, move_text)
            .ok_or_else(|| anyhow!("invalid move '{}' in {}", move_text, command))?;
        if !position.legal_moves().contains(&mv) {
            return Err(anyhow!("illegal move '{}' in {}", move_text, command));
        }
        position.do_move(mv);
    }

    Ok((position, move_tokens.len()))
}

fn load_record(path: &Path) -> Result<Record> {
    let content =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut result = None;
    let mut reason = None;
    let mut new_as = None;
    let mut final_position = None;
    let mut plies = 0;

    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("result ") {
            result = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("reason ") {
            reason = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("new_as ") {
            new_as = parse_side(rest.trim());
        } else if line.starts_with("position ") {
            let (position, move_count) = parse_position_command(line)?;
            final_position = Some(position);
            plies = move_count;
        }
    }

    Ok(Record {
        path: path.to_path_buf(),
        result: result.unwrap_or_else(|| "Unknown".to_string()),
        reason,
        new_as,
        final_position: final_position
            .ok_or_else(|| anyhow!("missing position line in {}", path.display()))?,
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

fn score_for_new(model: &SparseModel, record: &Record) -> Option<f32> {
    let new_as = record.new_as?;
    let score = model.predict_from_position(&record.final_position);
    let new_to_move = match record.final_position.side_to_move() {
        Color::Black => new_as == Color::Black,
        Color::White => new_as == Color::White,
    };
    Some(if new_to_move { score } else { -score })
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
        println!(
            "{} result={} reason={} new_as={} plies={} final_score_for_new={}",
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
            score
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
    println!("score/result sign mismatches: {}", score_result_mismatches);

    Ok(())
}
