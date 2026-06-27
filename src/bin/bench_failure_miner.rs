use anyhow::{anyhow, Context, Result};
use clap::Parser;
use glob::glob;
use serde::Serialize;
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{Evaluator, SparseModel};
use shogi_ai::utils::{format_move_usi, parse_usi_move, position_from_sfen_or_usi};
use shogi_core::{Color, Move, Piece};
use shogi_lib::Position;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const HISTORY_CAPACITY: usize = 256;
const SCORE_LIMIT: f32 = 100_000.0;

#[derive(Parser, Debug)]
#[command(about = "Mine benchmark losses for root-decision and teacher-regret counterexamples")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long)]
    teacher_weights: Option<PathBuf>,
    #[arg(long)]
    record_dir: Option<PathBuf>,
    #[arg(long, default_value_t = true)]
    only_new_losses: bool,
    #[arg(long, action = clap::ArgAction::SetTrue)]
    all_records: bool,
    #[arg(long, default_value_t = 16)]
    tail_plies: usize,
    #[arg(long, default_value_t = 5)]
    timed_depth: u8,
    #[arg(long, default_value_t = 100)]
    time_limit_ms: u64,
    #[arg(long, default_value_t = 6)]
    teacher_depth: u8,
    #[arg(long, default_value_t = 300.0)]
    bad_regret_cp: f32,
    #[arg(long, default_value_t = 100.0)]
    root_rescue_good_regret_cp: f32,
    #[arg(long, default_value_t = 300.0)]
    root_rescue_min_improvement_cp: f32,
    #[arg(long, default_value_t = 20)]
    top: usize,
    #[arg(long, default_value_t = 0)]
    max_records: usize,
    #[arg(long)]
    jsonl_output: Option<PathBuf>,
    #[arg(long)]
    export_timed_bad_sfens: Option<PathBuf>,
    #[arg(long)]
    export_root_rescue_sfens: Option<PathBuf>,
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

#[derive(Clone, Serialize)]
struct FailureSample {
    record: String,
    result: String,
    reason: String,
    new_as: String,
    start_sfen: Option<String>,
    baseline_sweep_start: bool,
    ply: usize,
    tail_index_from_end: usize,
    sfen: String,
    in_check: bool,
    legal_moves: usize,
    checking_moves: usize,
    actual_move: String,
    timed_move: Option<String>,
    teacher_move: Option<String>,
    actual_matches_teacher: bool,
    timed_matches_teacher: bool,
    timed_matches_actual: bool,
    teacher_score_cp: f32,
    actual_score_cp: f32,
    timed_score_cp: f32,
    actual_regret_cp: f32,
    timed_regret_cp: f32,
    actual_minus_timed_regret_cp: f32,
    static_tail_drop_cp: Option<f32>,
    timed_nodes: u64,
    timed_qnodes: u64,
    actual_bad: bool,
    timed_bad: bool,
    root_rescuable: bool,
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

fn side_text(side: Color) -> &'static str {
    match side {
        Color::Black => "black",
        Color::White => "white",
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
    let mut reason = None;
    let mut new_as = None;
    let mut start_sfen = None;
    let mut positions = None;
    let mut moves = None;

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
            let (parsed_positions, parsed_moves) = parse_position_command(line)?;
            positions = Some(parsed_positions);
            moves = Some(parsed_moves);
        }
    }

    Ok(Record {
        path: path.to_path_buf(),
        result: result.unwrap_or_else(|| "Unknown".to_string()),
        reason,
        new_as,
        start_sfen,
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

fn move_text(mv: Option<Move>) -> Option<String> {
    mv.map(format_move_usi)
}

fn checking_moves(position: &Position, moves: &[Move]) -> usize {
    moves
        .iter()
        .filter(|&&mv| position.is_check_move(mv))
        .count()
}

fn score_for_new_position(model: &SparseModel, position: &Position, new_as: Color) -> f32 {
    let score = model.predict_from_position(position);
    let new_to_move = position.side_to_move() == new_as;
    if new_to_move {
        score
    } else {
        -score
    }
}

fn static_tail_drop(
    model: &SparseModel,
    record: &Record,
    new_as: Color,
    ply: usize,
) -> Option<f32> {
    if ply == 0 {
        return None;
    }
    let prev = record.positions.get(ply - 1)?;
    let current = record.positions.get(ply)?;
    Some(
        score_for_new_position(model, prev, new_as)
            - score_for_new_position(model, current, new_as),
    )
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

fn create_writer(path: &Path) -> Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    Ok(BufWriter::new(file))
}

fn write_sfen_file(path: &Path, positions: &[String]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }
    let content = if positions.is_empty() {
        String::new()
    } else {
        format!("{}\n", positions.join("\n"))
    };
    fs::write(path, content).with_context(|| format!("failed to write {}", path.display()))
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

fn percent(count: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        count as f32 * 100.0 / total as f32
    }
}

fn record_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("<unknown>")
        .to_string()
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.timed_depth == 0 {
        return Err(anyhow!("--timed-depth must be greater than zero"));
    }
    if args.teacher_depth == 0 {
        return Err(anyhow!("--teacher-depth must be greater than zero"));
    }
    if !args.bad_regret_cp.is_finite() || args.bad_regret_cp < 0.0 {
        return Err(anyhow!("--bad-regret-cp must be non-negative"));
    }

    let model = load_model(&args.weights)?;
    let teacher_path = args.teacher_weights.as_ref().unwrap_or(&args.weights);
    let teacher_model = if teacher_path == &args.weights {
        None
    } else {
        Some(load_model(teacher_path)?)
    };
    let teacher_model = teacher_model.as_ref().unwrap_or(&model);

    let paths = collect_records(&args)?;
    let mut records = Vec::new();
    let mut paired_results = BTreeMap::<String, (usize, usize, usize)>::new();
    for path in paths {
        let record = load_record(&path)?;
        if let Some(start_sfen) = &record.start_sfen {
            let entry = paired_results
                .entry(start_sfen.clone())
                .or_insert((0, 0, 0));
            match record.result.as_str() {
                "NewWin" => entry.0 += 1,
                "BaselineWin" => entry.1 += 1,
                "Draw" => entry.2 += 1,
                _ => {}
            }
        }
        records.push(record);
    }
    let baseline_sweep_starts = paired_results
        .iter()
        .filter_map(|(start_sfen, (new_wins, baseline_wins, draws))| {
            if *baseline_wins > 0 && *new_wins == 0 && *draws == 0 {
                Some(start_sfen.clone())
            } else {
                None
            }
        })
        .collect::<BTreeSet<_>>();

    let evaluator = SharedModelEvaluator { model: &model };
    let mut timed_ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
    timed_ai.set_emit_info(false);

    let mut jsonl_writer = match &args.jsonl_output {
        Some(path) => Some(create_writer(path)?),
        None => None,
    };
    let mut samples = Vec::<FailureSample>::new();
    let mut timed_bad_sfens = Vec::new();
    let mut root_rescue_sfens = Vec::new();
    let mut skipped_records = 0usize;
    let mut probed_positions = 0usize;

    for record in &records {
        if args.only_new_losses && !args.all_records && record.result != "BaselineWin" {
            continue;
        }
        let Some(new_as) = record.new_as else {
            skipped_records += 1;
            continue;
        };

        timed_ai.clear();
        let reason = record
            .reason
            .clone()
            .unwrap_or_else(|| "Unknown".to_string());
        let start = record.moves.len().saturating_sub(args.tail_plies);
        for ply in 0..record.moves.len() {
            let position = &record.positions[ply];
            timed_ai.sennichite_detector.record_position(position);

            if ply < start || position.side_to_move() != new_as {
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

            let mut timed_position = position.clone();
            let timed_move = timed_ai.find_best_move(
                &mut timed_position,
                args.timed_depth,
                Some(args.time_limit_ms),
            );
            let timed_nodes = timed_ai.nodes_searched();
            let timed_qnodes = timed_ai.quiescence_nodes_searched();

            let Some((teacher_score, teacher_move)) =
                teacher_root(teacher_model, position, args.teacher_depth)
            else {
                continue;
            };
            let actual_score =
                teacher_move_score(teacher_model, position, actual_move, args.teacher_depth)
                    .unwrap_or(teacher_score);
            let timed_score = timed_move
                .and_then(|mv| teacher_move_score(teacher_model, position, mv, args.teacher_depth))
                .unwrap_or(teacher_score);

            let actual_regret = score_delta(teacher_score, actual_score);
            let timed_regret = score_delta(teacher_score, timed_score);
            let actual_bad = actual_regret > args.bad_regret_cp;
            let timed_bad = timed_regret > args.bad_regret_cp;
            let actual_minus_timed = actual_regret - timed_regret;
            let root_rescuable = actual_bad
                && timed_regret <= args.root_rescue_good_regret_cp
                && actual_minus_timed >= args.root_rescue_min_improvement_cp;
            let sfen = position.to_sfen_owned();
            let baseline_sweep_start = record
                .start_sfen
                .as_ref()
                .is_some_and(|start_sfen| baseline_sweep_starts.contains(start_sfen));

            let sample = FailureSample {
                record: record_name(&record.path),
                result: record.result.clone(),
                reason: reason.clone(),
                new_as: side_text(new_as).to_string(),
                start_sfen: record.start_sfen.clone(),
                baseline_sweep_start,
                ply,
                tail_index_from_end: record.moves.len().saturating_sub(ply),
                sfen: sfen.clone(),
                in_check: position.in_check(),
                legal_moves: legal_moves.len(),
                checking_moves: checking_moves(position, &legal_moves),
                actual_move: format_move_usi(actual_move),
                timed_move: move_text(timed_move),
                teacher_move: move_text(teacher_move),
                actual_matches_teacher: Some(actual_move) == teacher_move,
                timed_matches_teacher: timed_move == teacher_move,
                timed_matches_actual: timed_move == Some(actual_move),
                teacher_score_cp: sanitize_score(teacher_score),
                actual_score_cp: sanitize_score(actual_score),
                timed_score_cp: sanitize_score(timed_score),
                actual_regret_cp: actual_regret,
                timed_regret_cp: timed_regret,
                actual_minus_timed_regret_cp: actual_minus_timed,
                static_tail_drop_cp: static_tail_drop(&model, record, new_as, ply),
                timed_nodes,
                timed_qnodes,
                actual_bad,
                timed_bad,
                root_rescuable,
            };
            if timed_bad {
                timed_bad_sfens.push(sfen.clone());
            }
            if root_rescuable {
                root_rescue_sfens.push(sfen);
            }
            if let Some(writer) = jsonl_writer.as_mut() {
                serde_json::to_writer(&mut *writer, &sample)?;
                writeln!(writer)?;
            }
            samples.push(sample);
        }
    }
    if let Some(writer) = jsonl_writer.as_mut() {
        writer.flush()?;
    }

    let timed_regrets = samples
        .iter()
        .map(|sample| sample.timed_regret_cp)
        .collect::<Vec<_>>();
    let actual_regrets = samples
        .iter()
        .map(|sample| sample.actual_regret_cp)
        .collect::<Vec<_>>();
    let mut top_timed_bad = samples.clone();
    top_timed_bad.sort_by(|a, b| {
        b.timed_regret_cp
            .total_cmp(&a.timed_regret_cp)
            .then_with(|| b.actual_regret_cp.total_cmp(&a.actual_regret_cp))
            .then_with(|| a.record.cmp(&b.record))
            .then_with(|| a.ply.cmp(&b.ply))
    });
    let mut top_root_rescue = samples
        .iter()
        .filter(|sample| sample.root_rescuable)
        .cloned()
        .collect::<Vec<_>>();
    top_root_rescue.sort_by(|a, b| {
        b.actual_minus_timed_regret_cp
            .total_cmp(&a.actual_minus_timed_regret_cp)
            .then_with(|| b.actual_regret_cp.total_cmp(&a.actual_regret_cp))
            .then_with(|| a.record.cmp(&b.record))
            .then_with(|| a.ply.cmp(&b.ply))
    });

    let timed_bad = samples.iter().filter(|sample| sample.timed_bad).count();
    let actual_bad = samples.iter().filter(|sample| sample.actual_bad).count();
    let both_bad = samples
        .iter()
        .filter(|sample| sample.actual_bad && sample.timed_bad)
        .count();
    let actual_bad_timed_ok = samples
        .iter()
        .filter(|sample| sample.actual_bad && !sample.timed_bad)
        .count();
    let root_rescuable = samples
        .iter()
        .filter(|sample| sample.root_rescuable)
        .count();
    let timed_teacher_mismatch = samples
        .iter()
        .filter(|sample| !sample.timed_matches_teacher)
        .count();
    let actual_teacher_mismatch = samples
        .iter()
        .filter(|sample| !sample.actual_matches_teacher)
        .count();
    let in_check = samples.iter().filter(|sample| sample.in_check).count();
    let low_legal = samples
        .iter()
        .filter(|sample| sample.legal_moves <= 3)
        .count();
    let baseline_sweep_samples = samples
        .iter()
        .filter(|sample| sample.baseline_sweep_start)
        .count();

    if let Some(path) = &args.export_timed_bad_sfens {
        timed_bad_sfens.sort();
        timed_bad_sfens.dedup();
        write_sfen_file(path, &timed_bad_sfens)?;
        println!(
            "exported timed bad sfens: {} to {}",
            timed_bad_sfens.len(),
            path.display()
        );
    }
    if let Some(path) = &args.export_root_rescue_sfens {
        root_rescue_sfens.sort();
        root_rescue_sfens.dedup();
        write_sfen_file(path, &root_rescue_sfens)?;
        println!(
            "exported root-rescue sfens: {} to {}",
            root_rescue_sfens.len(),
            path.display()
        );
    }

    println!("records loaded: {}", records.len());
    println!("records skipped missing new_as: {}", skipped_records);
    println!("positions probed: {}", probed_positions);
    println!("samples mined: {}", samples.len());
    println!("paired starts: {}", paired_results.len());
    println!("baseline sweep starts: {}", baseline_sweep_starts.len());
    println!(
        "baseline sweep samples: {} ({:.2}%)",
        baseline_sweep_samples,
        percent(baseline_sweep_samples, samples.len())
    );
    println!(
        "actual teacher mismatches: {} ({:.2}%)",
        actual_teacher_mismatch,
        percent(actual_teacher_mismatch, samples.len())
    );
    println!(
        "timed teacher mismatches: {} ({:.2}%)",
        timed_teacher_mismatch,
        percent(timed_teacher_mismatch, samples.len())
    );
    println!(
        "actual bad_regret_gt_{:.0}: {} ({:.2}%)",
        args.bad_regret_cp,
        actual_bad,
        percent(actual_bad, samples.len())
    );
    println!(
        "timed bad_regret_gt_{:.0}: {} ({:.2}%)",
        args.bad_regret_cp,
        timed_bad,
        percent(timed_bad, samples.len())
    );
    println!("both actual_and_timed_bad: {}", both_bad);
    println!("actual_bad_timed_not_bad: {}", actual_bad_timed_ok);
    println!("root_rescuable: {}", root_rescuable);
    println!(
        "in_check: {} ({:.2}%)",
        in_check,
        percent(in_check, samples.len())
    );
    println!(
        "legal_moves_le_3: {} ({:.2}%)",
        low_legal,
        percent(low_legal, samples.len())
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

    println!("top timed-bad samples:");
    for sample in top_timed_bad.iter().take(args.top) {
        println!(
            "  {} ply={} timed_regret={} actual_regret={} actual_minus_timed={} actual={} timed={} teacher={} in_check={} legal={} checking={} baseline_sweep={} teacher_score={} timed_score={} actual_score={} nodes={} qnodes={}",
            sample.record,
            sample.ply,
            score_text(sample.timed_regret_cp),
            score_text(sample.actual_regret_cp),
            score_text(sample.actual_minus_timed_regret_cp),
            sample.actual_move,
            sample.timed_move.as_deref().unwrap_or("none"),
            sample.teacher_move.as_deref().unwrap_or("none"),
            sample.in_check,
            sample.legal_moves,
            sample.checking_moves,
            sample.baseline_sweep_start,
            score_text(sample.teacher_score_cp),
            score_text(sample.timed_score_cp),
            score_text(sample.actual_score_cp),
            sample.timed_nodes,
            sample.timed_qnodes
        );
        println!("    sfen {}", sample.sfen);
    }

    println!("top root-rescuable samples:");
    for sample in top_root_rescue.iter().take(args.top) {
        println!(
            "  {} ply={} improvement={} actual_regret={} timed_regret={} actual={} timed={} teacher={} in_check={} legal={} checking={} baseline_sweep={}",
            sample.record,
            sample.ply,
            score_text(sample.actual_minus_timed_regret_cp),
            score_text(sample.actual_regret_cp),
            score_text(sample.timed_regret_cp),
            sample.actual_move,
            sample.timed_move.as_deref().unwrap_or("none"),
            sample.teacher_move.as_deref().unwrap_or("none"),
            sample.in_check,
            sample.legal_moves,
            sample.checking_moves,
            sample.baseline_sweep_start
        );
        println!("    sfen {}", sample.sfen);
    }

    Ok(())
}
