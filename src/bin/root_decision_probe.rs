use anyhow::{anyhow, Context, Result};
use clap::Parser;
use glob::glob;
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{Evaluator, SparseModel};
use shogi_ai::utils::{format_move_usi, parse_usi_move, position_from_sfen_or_usi};
use shogi_core::{Color, Move, Piece};
use shogi_lib::Position;
use std::fs;
use std::path::{Path, PathBuf};

const HISTORY_CAPACITY: usize = 256;
const SCORE_LIMIT: f32 = 100_000.0;

#[derive(Parser, Debug)]
#[command(about = "Classify actual root decisions against timed and no-time searches")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long)]
    record_dir: Option<PathBuf>,
    #[arg(long, default_value_t = true)]
    only_new_losses: bool,
    #[arg(long, default_value_t = 16)]
    tail_plies: usize,
    #[arg(long, default_value_t = 5)]
    depth: u8,
    #[arg(long, default_value_t = 100)]
    time_limit_ms: u64,
    #[arg(long, default_value_t = 6)]
    teacher_depth: u8,
    #[arg(long, default_value_t = 300.0)]
    bad_regret_cp: f32,
    #[arg(long, default_value_t = 30)]
    top: usize,
    #[arg(long, default_value_t = 0)]
    max_records: usize,
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
    timed_regret: f32,
    actual_regret: f32,
    timed_score: f32,
    actual_score: f32,
    teacher_score: f32,
    ply: usize,
    actual_move: String,
    timed_move: String,
    teacher_move: String,
    timed_matches_actual: bool,
    timed_matches_teacher: bool,
    actual_matches_teacher: bool,
    legal_moves: usize,
    checking_moves: usize,
    timed_nodes: u64,
    timed_qnodes: u64,
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

fn sanitize_score(score: f32) -> f32 {
    if score == f32::INFINITY {
        SCORE_LIMIT
    } else if score == -f32::INFINITY {
        -SCORE_LIMIT
    } else {
        score.clamp(-SCORE_LIMIT, SCORE_LIMIT)
    }
}

fn score_delta(best_score: f32, move_score: f32) -> f32 {
    (sanitize_score(best_score) - sanitize_score(move_score)).max(0.0)
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

fn move_text(mv: Option<Move>) -> String {
    mv.map(format_move_usi)
        .unwrap_or_else(|| "none".to_string())
}

fn checking_moves(position: &Position, moves: &[Move]) -> usize {
    moves
        .iter()
        .filter(|&&mv| position.is_check_move(mv))
        .count()
}

fn teacher_root(
    model: &SparseModel,
    position: &Position,
    depth: u8,
) -> Option<(f32, Option<Move>)> {
    let evaluator = SharedModelEvaluator { model };
    let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
    let mut root = position.clone();
    ai.set_emit_info(false);
    ai.sennichite_detector.record_position(&root);
    ai.alpha_beta_search(&mut root, depth, -f32::INFINITY, f32::INFINITY)
        .map(|(score, pv)| (score, pv.first().copied()))
}

fn teacher_move_score(
    model: &SparseModel,
    position: &Position,
    mv: Move,
    depth: u8,
) -> Option<f32> {
    let mut child = position.clone();
    child.do_move(mv);
    let evaluator = SharedModelEvaluator { model };
    let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
    ai.set_emit_info(false);
    ai.sennichite_detector.record_position(&child);
    let child_depth = depth.saturating_sub(1);
    ai.alpha_beta_search(&mut child, child_depth, -f32::INFINITY, f32::INFINITY)
        .map(|(score, _)| -score)
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn percentile(mut values: Vec<f32>, pct: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let idx = ((values.len() - 1) as f32 * pct).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.depth == 0 {
        return Err(anyhow!("--depth must be greater than zero"));
    }
    if args.teacher_depth == 0 {
        return Err(anyhow!("--teacher-depth must be greater than zero"));
    }
    if !args.bad_regret_cp.is_finite() || args.bad_regret_cp < 0.0 {
        return Err(anyhow!("--bad-regret-cp must be non-negative"));
    }

    let model = load_model(&args.weights)?;
    let paths = collect_records(&args)?;
    let evaluator = SharedModelEvaluator { model: &model };
    let mut timed_ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
    timed_ai.set_emit_info(false);

    let mut probed_positions = 0usize;
    let mut timed_teacher_mismatches = 0usize;
    let mut actual_teacher_mismatches = 0usize;
    let mut bad_timed_regrets = 0usize;
    let mut bad_actual_regrets = 0usize;
    let mut records = Vec::new();

    for path in paths {
        let record = load_record(&path)?;
        if args.only_new_losses && record.result != "BaselineWin" {
            continue;
        }
        let Some(new_as) = record.new_as else {
            continue;
        };

        timed_ai.clear();
        let start = record.moves.len().saturating_sub(args.tail_plies);
        for ply in 0..record.moves.len() {
            let position = &record.positions[ply];
            timed_ai.sennichite_detector.record_position(position);

            if ply >= start && position.side_to_move() == new_as {
                let Some(actual_move) = parse_move_for_position(position, &record.moves[ply])
                else {
                    continue;
                };
                let legal_moves = position.legal_moves();
                if !legal_moves.contains(&actual_move) {
                    continue;
                }

                probed_positions += 1;
                let mut timed_position = position.clone();
                let timed_move = timed_ai.find_best_move(
                    &mut timed_position,
                    args.depth,
                    Some(args.time_limit_ms),
                );
                let timed_nodes = timed_ai.nodes_searched();
                let timed_qnodes = timed_ai.quiescence_nodes_searched();

                let Some((teacher_score, teacher_move)) =
                    teacher_root(&model, position, args.teacher_depth)
                else {
                    continue;
                };
                let actual_score =
                    teacher_move_score(&model, position, actual_move, args.teacher_depth)
                        .unwrap_or(teacher_score);
                let timed_score = timed_move
                    .and_then(|mv| teacher_move_score(&model, position, mv, args.teacher_depth))
                    .unwrap_or(teacher_score);
                let timed_regret = score_delta(teacher_score, timed_score);
                let actual_regret = score_delta(teacher_score, actual_score);

                let timed_matches_teacher = timed_move == teacher_move;
                let actual_matches_teacher = Some(actual_move) == teacher_move;
                if !timed_matches_teacher {
                    timed_teacher_mismatches += 1;
                }
                if !actual_matches_teacher {
                    actual_teacher_mismatches += 1;
                }
                if timed_regret > args.bad_regret_cp {
                    bad_timed_regrets += 1;
                }
                if actual_regret > args.bad_regret_cp {
                    bad_actual_regrets += 1;
                }

                records.push(ProbeRecord {
                    timed_regret,
                    actual_regret,
                    timed_score,
                    actual_score,
                    teacher_score,
                    ply,
                    actual_move: format_move_usi(actual_move),
                    timed_move: move_text(timed_move),
                    teacher_move: move_text(teacher_move),
                    timed_matches_actual: timed_move == Some(actual_move),
                    timed_matches_teacher,
                    actual_matches_teacher,
                    legal_moves: legal_moves.len(),
                    checking_moves: checking_moves(position, &legal_moves),
                    timed_nodes,
                    timed_qnodes,
                    path: record.path.clone(),
                    sfen: position.to_sfen_owned(),
                });
            }
        }
    }

    records.sort_by(|a, b| {
        b.timed_regret
            .total_cmp(&a.timed_regret)
            .then_with(|| b.actual_regret.total_cmp(&a.actual_regret))
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.ply.cmp(&b.ply))
    });

    let timed_regrets = records
        .iter()
        .map(|record| record.timed_regret)
        .collect::<Vec<_>>();
    let actual_regrets = records
        .iter()
        .map(|record| record.actual_regret)
        .collect::<Vec<_>>();

    println!("records probed positions: {}", probed_positions);
    println!("candidate positions: {}", records.len());
    println!(
        "timed teacher mismatches: {} ({:.2}%)",
        timed_teacher_mismatches,
        percent(timed_teacher_mismatches, records.len())
    );
    println!(
        "actual teacher mismatches: {} ({:.2}%)",
        actual_teacher_mismatches,
        percent(actual_teacher_mismatches, records.len())
    );
    println!(
        "timed bad_regret_gt_{:.0}: {} ({:.2}%)",
        args.bad_regret_cp,
        bad_timed_regrets,
        percent(bad_timed_regrets, records.len())
    );
    println!(
        "actual bad_regret_gt_{:.0}: {} ({:.2}%)",
        args.bad_regret_cp,
        bad_actual_regrets,
        percent(bad_actual_regrets, records.len())
    );
    println!("timed mean_regret_cp: {:.2}", mean(&timed_regrets));
    println!(
        "timed p90_regret_cp: {:.2}",
        percentile(timed_regrets.clone(), 0.90)
    );
    println!(
        "timed p95_regret_cp: {:.2}",
        percentile(timed_regrets.clone(), 0.95)
    );
    println!("actual mean_regret_cp: {:.2}", mean(&actual_regrets));
    println!(
        "actual p90_regret_cp: {:.2}",
        percentile(actual_regrets.clone(), 0.90)
    );
    println!(
        "actual p95_regret_cp: {:.2}",
        percentile(actual_regrets.clone(), 0.95)
    );
    println!("top timed decision regrets:");
    for record in records.iter().take(args.top) {
        println!(
            "  {} ply={} timed_regret={} actual_regret={} actual={} timed={} teacher={} timed_matches_actual={} timed_matches_teacher={} actual_matches_teacher={} teacher_score={} timed_score={} actual_score={} legal={} checking={} timed_nodes={} timed_qnodes={}",
            record
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("<unknown>"),
            record.ply,
            score_text(record.timed_regret),
            score_text(record.actual_regret),
            record.actual_move,
            record.timed_move,
            record.teacher_move,
            record.timed_matches_actual,
            record.timed_matches_teacher,
            record.actual_matches_teacher,
            score_text(record.teacher_score),
            score_text(record.timed_score),
            score_text(record.actual_score),
            record.legal_moves,
            record.checking_moves,
            record.timed_nodes,
            record.timed_qnodes
        );
        println!("    sfen {}", record.sfen);
    }

    Ok(())
}

fn percent(count: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        count as f32 * 100.0 / total as f32
    }
}
