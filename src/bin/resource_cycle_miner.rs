use anyhow::{anyhow, Result};
use clap::Parser;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use serde_json::json;
use shogi_ai::search_quality::{
    color_name, commit_suite_with_manifest, deduplicate_input_positions, ensure_distinct_paths,
    hand_counts, load_input_positions, AtomicOutput, DatasetSplit, SuiteKind,
};
use shogi_ai::utils::{format_move_usi, position_from_sfen_or_usi};
use shogi_core::{Color, Move};
use shogi_lib::Position;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Mine legal same-board resource-loss cycles without evaluation")]
struct Args {
    #[arg(long)]
    positions: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    split: DatasetSplit,
    #[arg(long, value_delimiter = ',', default_value = "4,6,8")]
    depths: Vec<u8>,
    #[arg(long, default_value_t = 250_000)]
    node_limit: u64,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 0)]
    position_limit: usize,
    #[arg(long, default_value_t = 0)]
    record_limit: usize,
}

#[derive(Clone)]
struct PathState {
    board_key_with_side: u64,
    black_hand: [u8; 7],
    white_hand: [u8; 7],
    sfen: String,
}

impl PathState {
    fn new(position: &Position) -> Self {
        Self {
            board_key_with_side: position.keys().0,
            black_hand: hand_counts(position, Color::Black),
            white_hand: hand_counts(position, Color::White),
            sfen: position.to_sfen_owned(),
        }
    }
}

struct Witness {
    ancestor: PathState,
    loser: Color,
    source_to_cycle_start: Vec<Move>,
    moves: Vec<Move>,
    checks: Vec<bool>,
    final_black_hand: [u8; 7],
    final_white_hand: [u8; 7],
}

enum DfsResult {
    Found(Witness),
    Exhausted,
    Unknown,
}

fn componentwise_loss(ancestor: &PathState, current: &PathState) -> Option<Color> {
    for loser in [Color::Black, Color::White] {
        let (old_own, new_own, old_opp, new_opp) = match loser {
            Color::Black => (
                &ancestor.black_hand,
                &current.black_hand,
                &ancestor.white_hand,
                &current.white_hand,
            ),
            Color::White => (
                &ancestor.white_hand,
                &current.white_hand,
                &ancestor.black_hand,
                &current.black_hand,
            ),
        };
        let own_subset = new_own.iter().zip(old_own).all(|(new, old)| new <= old);
        let opp_superset = new_opp.iter().zip(old_opp).all(|(new, old)| new >= old);
        let strict = new_own.iter().zip(old_own).any(|(new, old)| new < old);
        let opponent_strict = new_opp.iter().zip(old_opp).any(|(new, old)| new > old);
        if own_subset && opp_superset && strict && opponent_strict {
            return Some(loser);
        }
    }
    None
}

fn ordered_moves(position: &Position, previous_board_key: Option<u64>) -> Vec<Move> {
    let mut moves: Vec<_> = position.legal_moves().iter().copied().collect();
    moves.sort_unstable_by_key(|mv| match *mv {
        Move::Drop { piece, .. } if piece.piece_kind() == shogi_core::PieceKind::Pawn => 0,
        Move::Normal { to, .. } if position.piece_at(to).is_some() => 1,
        _ if previous_board_key.is_some_and(|key| {
            let mut child = position.clone();
            child.do_move(*mv);
            child.keys().0 == key
        }) =>
        {
            2
        }
        _ => 3,
    });
    moves
}

fn find_cycle(
    position: &mut Position,
    remaining: u8,
    node_limit: u64,
    nodes: &mut u64,
    states: &mut Vec<PathState>,
    moves: &mut Vec<Move>,
    checks: &mut Vec<bool>,
) -> DfsResult {
    if *nodes >= node_limit {
        return DfsResult::Unknown;
    }
    *nodes += 1;
    let current = states.last().expect("current path state");
    if moves.len() >= 2 {
        for (index, ancestor) in states[..states.len() - 1].iter().enumerate().rev() {
            if ancestor.board_key_with_side != current.board_key_with_side {
                continue;
            }
            if let Some(loser) = componentwise_loss(ancestor, current) {
                return DfsResult::Found(Witness {
                    ancestor: ancestor.clone(),
                    loser,
                    source_to_cycle_start: moves[..index].to_vec(),
                    moves: moves[index..].to_vec(),
                    checks: checks[index..].to_vec(),
                    final_black_hand: current.black_hand,
                    final_white_hand: current.white_hand,
                });
            }
        }
    }
    if remaining == 0 {
        return DfsResult::Exhausted;
    }

    let mut saw_unknown = false;
    let previous_board_key = states
        .get(states.len().saturating_sub(2))
        .map(|state| state.board_key_with_side);
    for mv in ordered_moves(position, previous_board_key) {
        let gave_check = position.is_check_move(mv);
        position.do_move(mv);
        states.push(PathState::new(position));
        moves.push(mv);
        checks.push(gave_check);
        let result = find_cycle(
            position,
            remaining - 1,
            node_limit,
            nodes,
            states,
            moves,
            checks,
        );
        checks.pop();
        moves.pop();
        states.pop();
        position.undo_move(mv);
        match result {
            DfsResult::Found(witness) => return DfsResult::Found(witness),
            DfsResult::Unknown => saw_unknown = true,
            DfsResult::Exhausted => {}
        }
    }
    if saw_unknown {
        DfsResult::Unknown
    } else {
        DfsResult::Exhausted
    }
}

fn validate_witness(witness: &Witness) -> bool {
    let Some(mut position) = position_from_sfen_or_usi(&witness.ancestor.sfen) else {
        return false;
    };
    let replay_start = PathState::new(&position);
    if replay_start.black_hand != witness.ancestor.black_hand
        || replay_start.white_hand != witness.ancestor.white_hand
    {
        return false;
    }
    for &mv in &witness.moves {
        if !position.legal_moves().contains(&mv) {
            return false;
        }
        position.do_move(mv);
    }
    let final_state = PathState::new(&position);
    final_state.board_key_with_side == replay_start.board_key_with_side
        && final_state.black_hand == witness.final_black_hand
        && final_state.white_hand == witness.final_white_hand
        && componentwise_loss(&replay_start, &final_state) == Some(witness.loser)
}

#[derive(Serialize)]
struct CycleRecord {
    schema_version: u32,
    source_index: usize,
    source_game_key: Option<String>,
    source_sfen: String,
    cycle_start_sfen: String,
    source_to_cycle_start: Vec<String>,
    loser: &'static str,
    horizon: u8,
    cycle_length: usize,
    proof_line: Vec<String>,
    gave_check: Vec<bool>,
    start_black_hand: [u8; 7],
    final_black_hand: [u8; 7],
    start_white_hand: [u8; 7],
    final_white_hand: [u8; 7],
    proof_nodes: u64,
    proof_status: &'static str,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.depths.is_empty() || args.depths.contains(&0) || args.node_limit == 0 {
        return Err(anyhow!("positive --depths and --node-limit are required"));
    }
    let sidecar = args.output.with_extension("manifest.json");
    ensure_distinct_paths(&[
        ("positions", &args.positions),
        ("output", &args.output),
        ("sidecar", &sidecar),
    ])?;
    let (positions, input_nonempty_lines) = load_input_positions(&args.positions)?;
    let valid_positions = positions.len();
    let (mut positions, duplicates) = deduplicate_input_positions(positions);
    positions.shuffle(&mut ChaCha8Rng::seed_from_u64(args.seed));
    let mut writer = AtomicOutput::new(&args.output)?;
    let mut written = 0usize;
    let position_limit = if args.position_limit == 0 {
        usize::MAX
    } else {
        args.position_limit
    };

    'positions: for input in positions.into_iter().take(position_limit) {
        let source_index = input.source_line;
        let source_game_key = input.source_game_key;
        let mut position = input.position;
        let source_sfen = position.to_sfen_owned();
        for &depth in &args.depths {
            let mut nodes = 0;
            let mut states = vec![PathState::new(&position)];
            let mut moves = Vec::new();
            let mut checks = Vec::new();
            let DfsResult::Found(witness) = find_cycle(
                &mut position,
                depth,
                args.node_limit,
                &mut nodes,
                &mut states,
                &mut moves,
                &mut checks,
            ) else {
                continue;
            };
            if !validate_witness(&witness) {
                return Err(anyhow!(
                    "internal error: generated cycle proof is not legal: source_index={source_index} depth={depth} source_sfen={source_sfen:?} cycle_start={:?} prefix={:?} proof={:?}",
                    witness.ancestor.sfen,
                    witness
                        .source_to_cycle_start
                        .iter()
                        .copied()
                        .map(format_move_usi)
                        .collect::<Vec<_>>(),
                    witness
                        .moves
                        .iter()
                        .copied()
                        .map(format_move_usi)
                        .collect::<Vec<_>>()
                ));
            }
            let record = CycleRecord {
                schema_version: 1,
                source_index,
                source_game_key,
                source_sfen: source_sfen.clone(),
                cycle_start_sfen: witness.ancestor.sfen.clone(),
                source_to_cycle_start: witness
                    .source_to_cycle_start
                    .iter()
                    .copied()
                    .map(format_move_usi)
                    .collect(),
                loser: color_name(witness.loser),
                horizon: depth,
                cycle_length: witness.moves.len(),
                proof_line: witness.moves.iter().copied().map(format_move_usi).collect(),
                gave_check: witness.checks,
                start_black_hand: witness.ancestor.black_hand,
                final_black_hand: witness.final_black_hand,
                start_white_hand: witness.ancestor.white_hand,
                final_white_hand: witness.final_white_hand,
                proof_nodes: nodes,
                proof_status: "proven_legal_resource_loss_cycle",
            };
            serde_json::to_writer(&mut writer, &record)?;
            writeln!(writer)?;
            written += 1;
            if args.record_limit > 0 && written >= args.record_limit {
                break 'positions;
            }
            break;
        }
    }
    if written == 0 {
        return Err(anyhow!("no proven resource-cycle records were generated"));
    }
    commit_suite_with_manifest(
        writer,
        "resource_cycle_miner",
        SuiteKind::ResourceCycle,
        args.split,
        &args.positions,
        args.seed,
        input_nonempty_lines,
        valid_positions,
        written,
        duplicates,
        json!({
            "depths": args.depths,
            "node_limit": args.node_limit,
            "position_limit": args.position_limit,
            "record_limit": args.record_limit,
            "negative_policy": "absence is never labeled; node-limited searches are excluded",
            "proof": "legal replay, same board key and side, component-wise hand dominance"
        }),
    )?;
    eprintln!("records written: {written}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_ai::utils::{parse_usi_move_for_color, position_from_sfen_or_usi};

    #[test]
    fn detects_componentwise_hand_transfer_on_same_board() {
        let before = position_from_sfen_or_usi("4k4/9/9/9/9/9/9/9/4K4 b P 1")
            .expect("valid before position");
        let after =
            position_from_sfen_or_usi("4k4/9/9/9/9/9/9/9/4K4 b p 3").expect("valid after position");
        let before = PathState::new(&before);
        let after = PathState::new(&after);
        assert_eq!(before.board_key_with_side, after.board_key_with_side);
        assert_eq!(Some(Color::Black), componentwise_loss(&before, &after));
    }

    #[test]
    fn rejects_trade_when_board_does_not_return() {
        let before = PathState {
            board_key_with_side: 1,
            black_hand: [1, 0, 0, 0, 0, 0, 0],
            white_hand: [0; 7],
            sfen: String::new(),
        };
        let after = PathState {
            board_key_with_side: 2,
            black_hand: [0; 7],
            white_hand: [1, 0, 0, 0, 0, 0, 0],
            sfen: String::new(),
        };
        assert_ne!(before.board_key_with_side, after.board_key_with_side);
    }

    #[test]
    fn validates_a_legal_four_ply_drop_capture_cycle() {
        let mut position =
            position_from_sfen_or_usi("4k4/9/9/4r4/9/9/9/9/4K4 b GP 1").expect("valid cycle start");
        let ancestor = PathState::new(&position);
        let mut moves = Vec::new();
        let mut checks = Vec::new();
        for text in ["P*5e", "5d1d", "5e5d", "1d5d"] {
            let mv = parse_usi_move_for_color(text, position.side_to_move()).expect("valid move");
            assert!(
                position.legal_moves().contains(&mv),
                "illegal fixture move {text}"
            );
            checks.push(position.is_check_move(mv));
            moves.push(mv);
            position.do_move(mv);
        }
        let final_state = PathState::new(&position);
        let witness = Witness {
            ancestor,
            loser: Color::Black,
            source_to_cycle_start: Vec::new(),
            moves,
            checks,
            final_black_hand: final_state.black_hand,
            final_white_hand: final_state.white_hand,
        };
        assert!(validate_witness(&witness));
        assert_eq!(
            Some(Color::Black),
            componentwise_loss(&witness.ancestor, &final_state)
        );
    }

    #[test]
    fn replays_real_depth_six_cycle_candidate() {
        let mut position = position_from_sfen_or_usi(
            "ln1g3nl/1k7/1spsprbpp/p2p2p2/1Pg1Pp1P1/P1PSG1P2/1KNP1P2P/2SGR4/L6NL w bp 74",
        )
        .expect("valid cycle start");
        let ancestor = PathState::new(&position);
        for text in ["P*8h", "8g8h", "B*8g", "8h8g"] {
            let mv = parse_usi_move_for_color(text, position.side_to_move()).expect("valid USI");
            assert!(
                position.legal_moves().contains(&mv),
                "illegal fixture move {text}; in_check={}",
                position.in_check()
            );
            position.do_move(mv);
        }
        let final_state = PathState::new(&position);
        assert_eq!(
            ancestor.board_key_with_side,
            final_state.board_key_with_side
        );
        assert_eq!(
            Some(Color::White),
            componentwise_loss(&ancestor, &final_state)
        );
    }

    #[test]
    fn real_depth_six_search_only_returns_a_replayable_witness() {
        let mut position = position_from_sfen_or_usi(
            "ln1g3nl/1k7/1spsprbpp/p2p2p2/1Pg1Pp1P1/P1PSG1P2/1KN2P2P/2SGR4/L6NL b Pbp 73",
        )
        .expect("valid source");
        let mut nodes = 0;
        let mut states = vec![PathState::new(&position)];
        let mut moves = Vec::new();
        let mut checks = Vec::new();
        let DfsResult::Found(witness) = find_cycle(
            &mut position,
            6,
            250_000,
            &mut nodes,
            &mut states,
            &mut moves,
            &mut checks,
        ) else {
            panic!("fixture must find a witness");
        };
        assert!(
            validate_witness(&witness),
            "loser={:?} moves={:?}",
            witness.loser,
            witness
                .moves
                .iter()
                .copied()
                .map(format_move_usi)
                .collect::<Vec<_>>(),
        );
    }
}
