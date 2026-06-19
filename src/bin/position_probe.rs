use anyhow::{anyhow, Result};
use clap::Parser;
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{Evaluator, SparseModel};
use shogi_ai::utils::{format_move_usi, position_from_sfen_or_usi};
use shogi_core::{Color, Move};
use shogi_lib::Position;
use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};

const HISTORY_CAPACITY: usize = 256;

#[derive(Parser, Debug)]
#[command(about = "Probe evaluation and search behavior on SFEN positions")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long)]
    positions: PathBuf,
    #[arg(long, default_value_t = 5)]
    depth: u8,
    #[arg(long)]
    time_limit_ms: Option<u64>,
    #[arg(long, default_value_t = 0)]
    limit: usize,
    #[arg(long, default_value_t = false)]
    show_legal: bool,
    #[arg(long, default_value_t = 0)]
    root_top: usize,
    #[arg(long, default_value_t = 0)]
    root_search_limit: usize,
    #[arg(long, default_value_t = false)]
    summary: bool,
}

struct SharedModelEvaluator<'a> {
    model: &'a SparseModel,
}

impl Evaluator for SharedModelEvaluator<'_> {
    fn evaluate(&self, position: &Position) -> f32 {
        self.model.predict_from_position(position)
    }
}

fn load_model(path: &Path) -> Result<SparseModel> {
    let mut model = SparseModel::new(0.0, 0.0);
    model
        .load(path)
        .map_err(|e| anyhow!("failed to load {}: {}", path.display(), e))?;
    Ok(model)
}

fn side_text(side: Color) -> &'static str {
    match side {
        Color::Black => "black",
        Color::White => "white",
    }
}

fn checking_moves(position: &Position, moves: &[Move]) -> usize {
    moves
        .iter()
        .filter(|&&mv| position.is_check_move(mv))
        .count()
}

fn move_list_text(moves: &[Move]) -> String {
    moves
        .iter()
        .map(|&mv| format_move_usi(mv))
        .collect::<Vec<_>>()
        .join(",")
}

fn pv_text(pv: &[Move]) -> String {
    if pv.is_empty() {
        "none".to_string()
    } else {
        move_list_text(pv)
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

#[derive(Debug)]
struct RootMoveProbe {
    mv: Move,
    score: f32,
    gives_check: bool,
    child_in_check: bool,
    child_legal_moves: usize,
    child_checking_moves: usize,
    child_static_eval_for_opponent: f32,
    nodes: u64,
    qnodes: u64,
    check_evasion_extensions: u64,
    pv: Vec<Move>,
}

fn probe_root_moves(
    model: &SparseModel,
    position: &Position,
    depth: u8,
    search_limit: usize,
) -> Vec<RootMoveProbe> {
    let mut probes = Vec::new();
    let legal_moves = position.legal_moves();
    let root_moves = if search_limit > 0 {
        &legal_moves[..legal_moves.len().min(search_limit)]
    } else {
        &legal_moves[..]
    };
    for &mv in root_moves {
        let mut child = position.clone();
        child.do_move(mv);
        let child_legal_moves = child.legal_moves();
        let child_static_eval_for_opponent = model.predict_from_position(&child);

        let evaluator = SharedModelEvaluator { model };
        let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
        ai.set_emit_info(false);
        ai.sennichite_detector.record_position(&child);
        let child_depth = depth.saturating_sub(1);
        let result = ai.alpha_beta_search(
            &mut child.clone(),
            child_depth,
            -f32::INFINITY,
            f32::INFINITY,
        );

        if let Some((child_score, child_pv)) = result {
            let mut pv = vec![mv];
            pv.extend(child_pv);
            probes.push(RootMoveProbe {
                mv,
                score: -child_score,
                gives_check: position.is_check_move(mv),
                child_in_check: child.in_check(),
                child_legal_moves: child_legal_moves.len(),
                child_checking_moves: checking_moves(&child, &child_legal_moves),
                child_static_eval_for_opponent,
                nodes: ai.nodes_searched(),
                qnodes: ai.quiescence_nodes_searched(),
                check_evasion_extensions: ai.check_evasion_extensions(),
                pv,
            });
        }
    }
    probes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    probes
}

#[derive(Default)]
struct ProbeSummary {
    total: usize,
    in_check: usize,
    low_legal_in_check: usize,
    terminal: usize,
    search_win: usize,
    search_loss: usize,
    legal_without_bestmove: usize,
    bestmove_gives_check: usize,
    bestmove_limits_reply: usize,
}

impl ProbeSummary {
    fn record(
        &mut self,
        position: &Position,
        legal_count: usize,
        search_score: Option<f32>,
        best_move: Option<Move>,
        after_legal_count: Option<usize>,
    ) {
        self.total += 1;
        if position.in_check() {
            self.in_check += 1;
            if (1..=3).contains(&legal_count) {
                self.low_legal_in_check += 1;
            }
        }
        if legal_count == 0 {
            self.terminal += 1;
        }
        match search_score {
            Some(score) if score == f32::INFINITY => self.search_win += 1,
            Some(score) if score == -f32::INFINITY => self.search_loss += 1,
            _ => {}
        }
        if legal_count > 0 && best_move.is_none() {
            self.legal_without_bestmove += 1;
        }
        if best_move.is_some_and(|mv| position.is_check_move(mv)) {
            self.bestmove_gives_check += 1;
        }
        if after_legal_count.is_some_and(|count| count <= 3) {
            self.bestmove_limits_reply += 1;
        }
    }

    fn print(&self) {
        println!("summary:");
        println!("  total: {}", self.total);
        println!("  in_check: {}", self.in_check);
        println!("  low_legal_in_check: {}", self.low_legal_in_check);
        println!("  terminal: {}", self.terminal);
        println!("  search_win: {}", self.search_win);
        println!("  search_loss: {}", self.search_loss);
        println!("  legal_without_bestmove: {}", self.legal_without_bestmove);
        println!("  bestmove_gives_check: {}", self.bestmove_gives_check);
        println!("  bestmove_limits_reply: {}", self.bestmove_limits_reply);
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let model = load_model(&args.weights)?;
    let content = fs::read_to_string(&args.positions)
        .map_err(|e| anyhow!("failed to read {}: {}", args.positions.display(), e))?;

    let mut probed = 0usize;
    let mut summary = ProbeSummary::default();
    for (line_index, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if args.limit > 0 && probed >= args.limit {
            break;
        }
        let Some(mut position) = position_from_sfen_or_usi(line) else {
            eprintln!("skip invalid line {}: {}", line_index + 1, line);
            continue;
        };
        probed += 1;

        let static_eval = model.predict_from_position(&position);
        let legal_moves = position.legal_moves();
        let legal_count = legal_moves.len();
        let checking_count = checking_moves(&position, &legal_moves);

        let evaluator = SharedModelEvaluator { model: &model };
        let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
        ai.set_emit_info(false);
        ai.sennichite_detector.record_position(&position);
        let best_move = ai.find_best_move(&mut position, args.depth, args.time_limit_ms);

        let search_evaluator = SharedModelEvaluator { model: &model };
        let mut search_ai = ShogiAI::<_, HISTORY_CAPACITY>::new(search_evaluator);
        search_ai.set_emit_info(false);
        search_ai.sennichite_detector.record_position(&position);
        let search_result = search_ai.alpha_beta_search(
            &mut position.clone(),
            args.depth,
            -f32::INFINITY,
            f32::INFINITY,
        );
        let search_score_value = search_result.as_ref().map(|(score, _)| *score);
        let (search_score, search_pv) = search_result
            .map(|(score, pv)| (format!("{score:.1}"), pv_text(&pv)))
            .unwrap_or_else(|| ("interrupted".to_string(), "none".to_string()));

        let mut after_text = "after_best=n/a".to_string();
        let mut after_legal_count = None;
        if let Some(best_move) = best_move {
            position.do_move(best_move);
            let after_legal = position.legal_moves();
            after_legal_count = Some(after_legal.len());
            let after_eval = model.predict_from_position(&position);
            after_text = format!(
                "after_best_in_check={} after_best_legal_moves={} after_best_checking_moves={} after_best_static_eval_for_opponent={:.1}",
                position.in_check(),
                after_legal.len(),
                checking_moves(&position, &after_legal),
                after_eval
            );
            position.undo_move(best_move);
        }

        summary.record(
            &position,
            legal_count,
            search_score_value,
            best_move,
            after_legal_count,
        );

        println!(
            "idx={} side={} in_check={} legal_moves={} checking_moves={} static_eval={:.1} search_score={} search_pv={} bestmove={} nodes={} qnodes={} search_nodes={} search_qnodes={} {}",
            probed,
            side_text(position.side_to_move()),
            position.in_check(),
            legal_count,
            checking_count,
            static_eval,
            search_score,
            search_pv,
            best_move.map(format_move_usi).unwrap_or_else(|| "none".to_string()),
            ai.nodes_searched(),
            ai.quiescence_nodes_searched(),
            search_ai.nodes_searched(),
            search_ai.quiescence_nodes_searched(),
            after_text
        );
        if args.show_legal {
            println!("  legal {}", move_list_text(&legal_moves));
        }
        if args.root_top > 0 {
            let searched_root_moves = if args.root_search_limit > 0 {
                legal_count.min(args.root_search_limit)
            } else {
                legal_count
            };
            println!(
                "  root_probe searched_moves={} total_legal_moves={} depth={}",
                searched_root_moves, legal_count, args.depth
            );
            for (rank, probe) in
                probe_root_moves(&model, &position, args.depth, args.root_search_limit)
                    .into_iter()
                    .take(args.root_top)
                    .enumerate()
            {
                println!(
                    "  root rank={} move={} score={} gives_check={} child_in_check={} child_legal_moves={} child_checking_moves={} child_static_eval_for_opponent={:.1} nodes={} qnodes={} check_evasion_extensions={} pv={}",
                    rank + 1,
                    format_move_usi(probe.mv),
                    score_text(probe.score),
                    probe.gives_check,
                    probe.child_in_check,
                    probe.child_legal_moves,
                    probe.child_checking_moves,
                    probe.child_static_eval_for_opponent,
                    probe.nodes,
                    probe.qnodes,
                    probe.check_evasion_extensions,
                    pv_text(&probe.pv)
                );
            }
        }
        println!("  sfen {}", position.to_sfen_owned());
    }

    if args.summary {
        summary.print();
    }

    Ok(())
}
