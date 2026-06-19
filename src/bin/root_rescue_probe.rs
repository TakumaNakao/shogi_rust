use anyhow::{anyhow, Context, Result};
use clap::Parser;
use glob::glob;
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{Evaluator, SparseModel};
use shogi_ai::utils::{format_move_usi, parse_usi_move, position_from_sfen_or_usi};
use shogi_core::{Color, Move, Piece};
use shogi_lib::Position;
use std::path::{Path, PathBuf};
use std::{cmp::Ordering, fs};

const HISTORY_CAPACITY: usize = 256;

#[derive(Parser, Debug)]
#[command(about = "Compare actual tail moves with root search alternatives")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long)]
    record_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 12)]
    tail_plies: usize,
    #[arg(long, default_value_t = 5)]
    depth: u8,
    #[arg(long, default_value_t = 30)]
    top: usize,
    #[arg(long, default_value_t = 0)]
    max_records: usize,
    #[arg(long, default_value_t = true)]
    only_new_losses: bool,
    #[arg(required_unless_present = "record_dir")]
    records: Vec<PathBuf>,
}

#[derive(Debug)]
struct Record {
    path: PathBuf,
    result: String,
    new_as: Option<Color>,
    positions: Vec<Position>,
    moves: Vec<String>,
}

#[derive(Clone)]
struct SharedModelEvaluator<'a> {
    model: &'a SparseModel,
}

impl Evaluator for SharedModelEvaluator<'_> {
    fn evaluate(&self, position: &Position) -> f32 {
        self.model.predict_from_position(position)
    }
}

#[derive(Debug)]
struct ProbeRecord {
    improvement: f32,
    actual_score: f32,
    best_score: f32,
    strong: bool,
    actual_forced_loss: bool,
    best_avoids_forced_loss: bool,
    ply: usize,
    actual_move: String,
    best_move: String,
    best_pv: String,
    legal_moves: usize,
    checking_moves: usize,
    actual_reply_moves: usize,
    best_reply_moves: Option<usize>,
    path: PathBuf,
    sfen: String,
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

fn parse_position_command(command: &str) -> Result<(Vec<Position>, Vec<String>)> {
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
    let mut moves = Vec::new();
    for move_text in move_tokens {
        let mv = parse_move_for_position(&position, move_text)
            .ok_or_else(|| anyhow!("invalid move '{}' in {}", move_text, command))?;
        if !position.legal_moves().contains(&mv) {
            return Err(anyhow!("illegal move '{}' in {}", move_text, command));
        }
        position.do_move(mv);
        moves.push((*move_text).to_string());
        positions.push(position.clone());
    }

    Ok((positions, moves))
}

fn load_record(path: &Path) -> Result<Record> {
    let content =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut result = None;
    let mut new_as = None;
    let mut positions = None;
    let mut moves = None;

    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("result ") {
            result = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("new_as ") {
            new_as = parse_side(rest.trim());
        } else if line.starts_with("position ") {
            let (parsed_positions, parsed_moves) = parse_position_command(line)?;
            positions = Some(parsed_positions);
            moves = Some(parsed_moves);
        }
    }

    Ok(Record {
        path: path.to_path_buf(),
        result: result.unwrap_or_else(|| "Unknown".to_string()),
        new_as,
        positions: positions
            .ok_or_else(|| anyhow!("missing position line in {}", path.display()))?,
        moves: moves.ok_or_else(|| anyhow!("missing position line in {}", path.display()))?,
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
    if args.max_records > 0 {
        paths.truncate(args.max_records);
    }
    if paths.is_empty() {
        return Err(anyhow!("no record files found"));
    }
    Ok(paths)
}

fn pv_text(pv: &[Move]) -> String {
    if pv.is_empty() {
        "none".to_string()
    } else {
        pv.iter()
            .map(|&mv| format_move_usi(mv))
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn score_text(score: f32) -> String {
    if score == f32::INFINITY {
        "inf".to_string()
    } else if score == -f32::INFINITY {
        "-inf".to_string()
    } else {
        format!("{score:.1}")
    }
}

fn is_forced_loss(score: f32) -> bool {
    score == -f32::INFINITY || score <= -1000.0
}

fn avoids_forced_loss(score: f32) -> bool {
    score > -1000.0
}

fn is_strong_rescue(
    actual_score: f32,
    best_score: f32,
    improvement: f32,
    best_move: Option<Move>,
) -> bool {
    best_move.is_some()
        && is_forced_loss(actual_score)
        && avoids_forced_loss(best_score)
        && improvement >= 300.0
}

fn search_position(
    model: &SparseModel,
    position: &Position,
    depth: u8,
) -> Option<(f32, Vec<Move>)> {
    let evaluator = SharedModelEvaluator { model };
    let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
    ai.set_emit_info(false);
    ai.sennichite_detector.record_position(position);
    ai.alpha_beta_search(&mut position.clone(), depth, -f32::INFINITY, f32::INFINITY)
}

fn actual_move_score(
    model: &SparseModel,
    position: &Position,
    actual_move: Move,
    depth: u8,
) -> Option<(f32, usize)> {
    let mut child = position.clone();
    child.do_move(actual_move);
    let reply_moves = child.legal_moves().len();
    let child_depth = depth.saturating_sub(1);
    if child_depth == 0 {
        return Some((-model.predict_from_position(&child), reply_moves));
    }
    search_position(model, &child, child_depth).map(|(score, _)| (-score, reply_moves))
}

fn checking_moves(position: &Position, moves: &[Move]) -> usize {
    moves
        .iter()
        .filter(|&&mv| position.is_check_move(mv))
        .count()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let model = load_model(&args.weights)?;
    let paths = collect_records(&args)?;

    let mut probed_positions = 0usize;
    let mut same_best = 0usize;
    let mut rescue_candidates = Vec::new();

    for path in paths {
        let record = load_record(&path)?;
        if args.only_new_losses && record.result != "BaselineWin" {
            continue;
        }
        let Some(new_as) = record.new_as else {
            continue;
        };
        let start = record.moves.len().saturating_sub(args.tail_plies);
        for ply in start..record.moves.len() {
            let position = &record.positions[ply];
            if position.side_to_move() != new_as {
                continue;
            }
            let Some(actual_move) = parse_move_for_position(position, &record.moves[ply]) else {
                continue;
            };
            let legal_moves = position.legal_moves();
            if !legal_moves.contains(&actual_move) {
                continue;
            }
            probed_positions += 1;
            let Some((best_score, best_pv)) = search_position(&model, position, args.depth) else {
                continue;
            };
            let Some((actual_score, actual_reply_moves)) =
                actual_move_score(&model, position, actual_move, args.depth)
            else {
                continue;
            };
            let best_move = best_pv.first().copied();
            if best_move == Some(actual_move) {
                same_best += 1;
            }
            let best_reply_moves = best_move.map(|mv| {
                let mut child = position.clone();
                child.do_move(mv);
                child.legal_moves().len()
            });
            let improvement = best_score - actual_score;
            let strong = is_strong_rescue(actual_score, best_score, improvement, best_move);
            rescue_candidates.push(ProbeRecord {
                improvement,
                actual_score,
                best_score,
                strong,
                actual_forced_loss: is_forced_loss(actual_score),
                best_avoids_forced_loss: avoids_forced_loss(best_score),
                ply,
                actual_move: record.moves[ply].clone(),
                best_move: best_move
                    .map(format_move_usi)
                    .unwrap_or_else(|| "none".to_string()),
                best_pv: pv_text(&best_pv),
                legal_moves: legal_moves.len(),
                checking_moves: checking_moves(position, &legal_moves),
                actual_reply_moves,
                best_reply_moves,
                path: record.path.clone(),
                sfen: position.to_sfen_owned(),
            });
        }
    }

    rescue_candidates.sort_by(|a, b| {
        b.improvement
            .partial_cmp(&a.improvement)
            .unwrap_or(Ordering::Equal)
    });

    let actual_forced_losses = rescue_candidates
        .iter()
        .filter(|record| record.actual_forced_loss)
        .count();
    let strong_candidates = rescue_candidates
        .iter()
        .filter(|record| record.strong)
        .count();

    println!("records probed positions: {}", probed_positions);
    println!("same as root best: {}", same_best);
    println!("candidate positions: {}", rescue_candidates.len());
    println!("actual forced-loss moves: {}", actual_forced_losses);
    println!("strong root-rescuable candidates: {}", strong_candidates);
    println!("top root rescue candidates:");
    for record in rescue_candidates.iter().take(args.top) {
        println!(
            "  {} ply={} strong={} actual_forced_loss={} best_avoids_forced_loss={} improvement={} actual={} score={} best={} best_score={} legal={} checking={} actual_replies={} best_replies={} pv={}",
            record
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("<unknown>"),
            record.ply,
            record.strong,
            record.actual_forced_loss,
            record.best_avoids_forced_loss,
            score_text(record.improvement),
            record.actual_move,
            score_text(record.actual_score),
            record.best_move,
            score_text(record.best_score),
            record.legal_moves,
            record.checking_moves,
            record.actual_reply_moves,
            record
                .best_reply_moves
                .map(|moves| moves.to_string())
                .unwrap_or_else(|| "n/a".to_string()),
            record.best_pv
        );
        println!("    sfen {}", record.sfen);
    }

    Ok(())
}
