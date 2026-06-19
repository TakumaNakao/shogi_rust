use anyhow::{anyhow, Result};
use clap::Parser;
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{Evaluator, SparseModel};
use shogi_ai::utils::{format_move_usi, position_from_sfen_or_usi};
use shogi_core::{Color, Move};
use shogi_lib::Position;
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
}

struct SharedModelEvaluator<'a> {
    model: &'a SparseModel,
    use_search_eval: bool,
}

impl Evaluator for SharedModelEvaluator<'_> {
    fn evaluate(&self, position: &Position) -> f32 {
        if self.use_search_eval {
            self.model.predict_search_from_position(position)
        } else {
            self.model.predict_from_position(position)
        }
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

fn main() -> Result<()> {
    let args = Args::parse();
    let model = load_model(&args.weights)?;
    let content = fs::read_to_string(&args.positions)
        .map_err(|e| anyhow!("failed to read {}: {}", args.positions.display(), e))?;

    let mut probed = 0usize;
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
        let search_static_eval = model.predict_search_from_position(&position);
        let legal_moves = position.legal_moves();
        let legal_count = legal_moves.len();
        let checking_count = checking_moves(&position, &legal_moves);

        let evaluator = SharedModelEvaluator {
            model: &model,
            use_search_eval: true,
        };
        let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
        ai.set_emit_info(false);
        ai.sennichite_detector.record_position(&position);
        let best_move = ai.find_best_move(&mut position, args.depth, args.time_limit_ms);

        let search_evaluator = SharedModelEvaluator {
            model: &model,
            use_search_eval: true,
        };
        let mut search_ai = ShogiAI::<_, HISTORY_CAPACITY>::new(search_evaluator);
        search_ai.set_emit_info(false);
        search_ai.sennichite_detector.record_position(&position);
        let search_result = search_ai.alpha_beta_search(
            &mut position.clone(),
            args.depth,
            -f32::INFINITY,
            f32::INFINITY,
        );
        let (search_score, search_pv) = search_result
            .map(|(score, pv)| (format!("{score:.1}"), pv_text(&pv)))
            .unwrap_or_else(|| ("interrupted".to_string(), "none".to_string()));

        let mut after_text = "after_best=n/a".to_string();
        if let Some(best_move) = best_move {
            position.do_move(best_move);
            let after_legal = position.legal_moves();
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

        println!(
            "idx={} side={} in_check={} legal_moves={} checking_moves={} static_eval={:.1} search_static_eval={:.1} search_score={} search_pv={} bestmove={} nodes={} qnodes={} search_nodes={} search_qnodes={} {}",
            probed,
            side_text(position.side_to_move()),
            position.in_check(),
            legal_count,
            checking_count,
            static_eval,
            search_static_eval,
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
        println!("  sfen {}", position.to_sfen_owned());
    }

    Ok(())
}
