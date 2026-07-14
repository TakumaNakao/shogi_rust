use crate::position_hash::PositionHasher;
use crate::utils::get_piece_value;
use shogi_core::{Color, Move, PieceKind};
use shogi_lib::Position;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub const MAX_MATE_HORIZON: u8 = 7;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MateSearchResult {
    ProvenMate {
        first_move: Option<Move>,
        ply: u8,
        proof: Vec<Move>,
    },
    ProvenNoMateWithinHorizon,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MateSearchStopReason {
    NodeLimit,
    TimeLimit,
    ExternalStop,
}

#[derive(Clone, Debug)]
pub struct MateSearchLimits {
    pub node_limit: u64,
    pub deadline: Option<Instant>,
    pub stop_signal: Option<Arc<AtomicBool>>,
    pub prior_position_hashes: Vec<u64>,
}

impl MateSearchLimits {
    pub fn nodes(node_limit: u64) -> Self {
        Self {
            node_limit,
            deadline: None,
            stop_signal: None,
            prior_position_hashes: Vec::new(),
        }
    }
}

#[derive(Clone, Copy)]
enum CachedResult {
    // Non-terminal mate proofs are path-sensitive because a new ancestor can turn a
    // defense into repetition. Only terminal mate and path-independent no-mate are cached.
    TerminalMate,
    NoMate,
}

struct InternalResult {
    result: MateSearchResult,
    path_dependent: bool,
}

pub struct MateSearcher {
    limits: MateSearchLimits,
    nodes: u64,
    stopped: bool,
    stop_reason: Option<MateSearchStopReason>,
    transposition: HashMap<(u64, u8), CachedResult>,
    path: HashSet<u64>,
}

impl MateSearcher {
    pub fn new(limits: MateSearchLimits) -> Self {
        Self {
            limits,
            nodes: 0,
            stopped: false,
            stop_reason: None,
            transposition: HashMap::new(),
            path: HashSet::new(),
        }
    }

    pub fn nodes(&self) -> u64 {
        self.nodes
    }

    pub fn stopped(&self) -> bool {
        self.stopped
    }

    pub fn stop_reason(&self) -> Option<MateSearchStopReason> {
        self.stop_reason
    }

    pub fn search(
        &mut self,
        position: &mut Position,
        attacker: Color,
        horizon: u8,
    ) -> MateSearchResult {
        assert!(
            horizon <= MAX_MATE_HORIZON,
            "mate horizon must be at most {MAX_MATE_HORIZON}"
        );
        self.reset();
        self.remove_root_from_prior_path(position);
        let result = self.search_node(position, attacker, horizon).result;
        if self.stopped {
            MateSearchResult::Unknown
        } else {
            result
        }
    }

    pub fn search_shortest(
        &mut self,
        position: &mut Position,
        attacker: Color,
        max_horizon: u8,
    ) -> MateSearchResult {
        assert!(
            max_horizon <= MAX_MATE_HORIZON,
            "mate horizon must be at most {MAX_MATE_HORIZON}"
        );
        self.reset();
        self.remove_root_from_prior_path(position);
        let start = if max_horizon % 2 == 0 { 0 } else { 1 };
        for horizon in (start..=max_horizon).step_by(2) {
            match self.search_node(position, attacker, horizon).result {
                result @ MateSearchResult::ProvenMate { .. } => {
                    return if self.stopped {
                        MateSearchResult::Unknown
                    } else {
                        result
                    };
                }
                MateSearchResult::Unknown => return MateSearchResult::Unknown,
                MateSearchResult::ProvenNoMateWithinHorizon => {}
            }
        }
        if self.stopped {
            MateSearchResult::Unknown
        } else {
            MateSearchResult::ProvenNoMateWithinHorizon
        }
    }

    fn reset(&mut self) {
        self.nodes = 0;
        self.stopped = false;
        self.stop_reason = None;
        self.transposition.clear();
        self.path.clear();
        self.path
            .extend(self.limits.prior_position_hashes.iter().copied());
    }

    fn remove_root_from_prior_path(&mut self, position: &Position) {
        // The root is allowed once. If it is already in the game history, only
        // a root-to-child-to-root re-entry is a path cycle during this probe.
        let root_hash = PositionHasher::calculate_hash(position);
        self.path.remove(&root_hash);
    }

    fn limit_reached(&mut self) -> bool {
        if self.nodes >= self.limits.node_limit {
            self.stopped = true;
            self.stop_reason = Some(MateSearchStopReason::NodeLimit);
            return true;
        }
        if self
            .limits
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            self.stopped = true;
            self.stop_reason = Some(MateSearchStopReason::TimeLimit);
            return true;
        }
        if self
            .limits
            .stop_signal
            .as_ref()
            .is_some_and(|signal| signal.load(Ordering::Relaxed))
        {
            self.stopped = true;
            self.stop_reason = Some(MateSearchStopReason::ExternalStop);
            return true;
        }
        false
    }

    fn search_node(
        &mut self,
        position: &mut Position,
        attacker: Color,
        depth: u8,
    ) -> InternalResult {
        let hash = PositionHasher::calculate_hash(position);
        if self.path.contains(&hash) {
            return InternalResult {
                result: MateSearchResult::ProvenNoMateWithinHorizon,
                path_dependent: true,
            };
        }
        if self.limit_reached() {
            return InternalResult {
                result: MateSearchResult::Unknown,
                path_dependent: false,
            };
        }
        self.nodes += 1;

        if let Some(cached) = self.transposition.get(&(hash, depth)).copied() {
            let result = match cached {
                CachedResult::TerminalMate => MateSearchResult::ProvenMate {
                    first_move: None,
                    ply: 0,
                    proof: Vec::new(),
                },
                CachedResult::NoMate => MateSearchResult::ProvenNoMateWithinHorizon,
            };
            return InternalResult {
                result,
                path_dependent: false,
            };
        }

        let moves = position.legal_moves();
        if moves.is_empty() {
            let is_mate = position.in_check() && position.side_to_move() != attacker;
            let result = if is_mate {
                self.transposition
                    .insert((hash, depth), CachedResult::TerminalMate);
                MateSearchResult::ProvenMate {
                    first_move: None,
                    ply: 0,
                    proof: Vec::new(),
                }
            } else {
                self.transposition
                    .insert((hash, depth), CachedResult::NoMate);
                MateSearchResult::ProvenNoMateWithinHorizon
            };
            return InternalResult {
                result,
                path_dependent: false,
            };
        }
        if depth == 0 {
            self.transposition
                .insert((hash, depth), CachedResult::NoMate);
            return InternalResult {
                result: MateSearchResult::ProvenNoMateWithinHorizon,
                path_dependent: false,
            };
        }

        self.path.insert(hash);
        let result = if position.side_to_move() == attacker {
            self.search_attacker(position, attacker, depth, moves)
        } else {
            self.search_defender(position, attacker, depth, moves)
        };
        self.path.remove(&hash);

        if matches!(result.result, MateSearchResult::ProvenNoMateWithinHorizon)
            && !result.path_dependent
        {
            self.transposition
                .insert((hash, depth), CachedResult::NoMate);
        }
        result
    }

    fn search_attacker(
        &mut self,
        position: &mut Position,
        attacker: Color,
        depth: u8,
        moves: impl IntoIterator<Item = Move>,
    ) -> InternalResult {
        let mut checking_moves = moves
            .into_iter()
            .filter(|&mv| position.is_check_move(mv))
            .collect::<Vec<_>>();
        checking_moves.sort_unstable_by_key(|&mv| (simple_see(position, mv), move_order_key(mv)));

        let mut saw_unknown = false;
        let mut path_dependent = false;
        for mv in checking_moves {
            position.do_move(mv);
            let child = self.search_node(position, attacker, depth - 1);
            position.undo_move(mv);
            match child.result {
                MateSearchResult::ProvenMate { mut proof, .. } => {
                    proof.insert(0, mv);
                    return InternalResult {
                        result: MateSearchResult::ProvenMate {
                            first_move: Some(mv),
                            ply: proof.len() as u8,
                            proof,
                        },
                        path_dependent: child.path_dependent,
                    };
                }
                MateSearchResult::Unknown => saw_unknown = true,
                MateSearchResult::ProvenNoMateWithinHorizon => {
                    path_dependent |= child.path_dependent;
                }
            }
        }
        InternalResult {
            result: if saw_unknown {
                MateSearchResult::Unknown
            } else {
                MateSearchResult::ProvenNoMateWithinHorizon
            },
            path_dependent,
        }
    }

    fn search_defender(
        &mut self,
        position: &mut Position,
        attacker: Color,
        depth: u8,
        moves: impl IntoIterator<Item = Move>,
    ) -> InternalResult {
        let mut moves = moves.into_iter().collect::<Vec<_>>();
        moves.sort_unstable_by_key(|&mv| move_order_key(mv));
        let mut longest_proof = Vec::new();
        let mut saw_unknown = false;
        let mut path_dependent = false;
        for mv in moves {
            position.do_move(mv);
            let child = self.search_node(position, attacker, depth - 1);
            position.undo_move(mv);
            match child.result {
                MateSearchResult::ProvenMate { mut proof, .. } => {
                    proof.insert(0, mv);
                    if proof.len() > longest_proof.len() {
                        longest_proof = proof;
                    }
                    path_dependent |= child.path_dependent;
                }
                MateSearchResult::ProvenNoMateWithinHorizon => {
                    return InternalResult {
                        result: MateSearchResult::ProvenNoMateWithinHorizon,
                        path_dependent: child.path_dependent,
                    };
                }
                MateSearchResult::Unknown => saw_unknown = true,
            }
        }
        InternalResult {
            result: if saw_unknown {
                MateSearchResult::Unknown
            } else {
                MateSearchResult::ProvenMate {
                    first_move: longest_proof.first().copied(),
                    ply: longest_proof.len() as u8,
                    proof: longest_proof,
                }
            },
            path_dependent,
        }
    }
}

fn move_order_key(mv: Move) -> (u8, u8, u8, u8) {
    match mv {
        Move::Normal { from, to, promote } => (0, from.index(), to.index(), u8::from(promote)),
        Move::Drop { piece, to } => {
            let piece = match piece.piece_kind() {
                PieceKind::Pawn => 0,
                PieceKind::Lance => 1,
                PieceKind::Knight => 2,
                PieceKind::Silver => 3,
                PieceKind::Gold => 4,
                PieceKind::Bishop => 5,
                PieceKind::Rook => 6,
                _ => 7,
            };
            (1, piece, to.index(), 0)
        }
    }
}

fn simple_see(position: &Position, mv: Move) -> i32 {
    match mv {
        Move::Normal { from, to, .. } => position
            .piece_at(from)
            .zip(position.piece_at(to))
            .map(|(attacker, victim)| {
                get_piece_value(victim.piece_kind()) - get_piece_value(attacker.piece_kind())
            })
            .unwrap_or(0),
        Move::Drop { .. } => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{format_move_usi, position_from_sfen_or_usi};
    use serde_json::Value;

    fn dev_record(horizon: u8) -> Value {
        include_str!("../data/search_quality/generated/dev_mate_sacrifice.jsonl")
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).expect("valid dev mate record"))
            .find(|record| record["mate_horizon"].as_u64() == Some(u64::from(horizon)))
            .expect("horizon fixture")
    }

    #[test]
    fn proves_one_three_five_and_seven_ply_mates_and_restores_positions() {
        let fixtures = [
            (
                1,
                "+B2gklR2/2s4+Rl/np1+Bpgp1p/p1P2p3/9/9/PP1PPPPPP/3S2K2/LN1G1GSNL b 3Psn 45",
            ),
            (
                3,
                "ln5nl/2B1g1k2/p1s1ppspp/6p2/9/PP2P3P/2PP+bPPP1/L1G3GR1/1NSKGrSNL w 3P 42",
            ),
            (
                5,
                "1n5nl/5s+Bp1/1pp1p3k/5pp1p/3p5/2P5N/1PlPPPPPP/2bG5/L2RKGSNL b 2GS2Prs 65",
            ),
            (
                7,
                "7n1/4+R2s+L/p1p+Bppkp1/1p4p2/3g5/PP7/2P1PPPP1/1B2G2R1/LN1GK1SN1 b G2L2P2sn2p 55",
            ),
        ];
        for (horizon, sfen) in fixtures {
            let mut position = position_from_sfen_or_usi(sfen).unwrap();
            let original = position.to_sfen_owned();
            let attacker = position.side_to_move();
            let mut searcher = MateSearcher::new(MateSearchLimits::nodes(100_000));
            let result = searcher.search_shortest(&mut position, attacker, horizon);
            let MateSearchResult::ProvenMate {
                first_move,
                ply,
                proof,
            } = result
            else {
                panic!("horizon {horizon} was not proven");
            };
            assert_eq!(Some(ply), u8::try_from(proof.len()).ok());
            assert_eq!(horizon, ply);
            assert_eq!(original, position.to_sfen_owned());
            let first_move = first_move.expect("root proof has a first move");
            assert!(position.legal_moves().contains(&first_move));
            assert!(position.is_check_move(first_move));
            let mut proof_position = position.clone();
            for mv in proof {
                assert!(proof_position.legal_moves().contains(&mv));
                proof_position.do_move(mv);
            }
            assert!(proof_position.in_check());
            assert!(proof_position.legal_moves().is_empty());
        }
    }

    #[test]
    fn proves_every_branch_of_a_two_defense_mate() {
        let mut position = position_from_sfen_or_usi(
            "6+Bnl/3G2s2/p2p1pkpp/5lg2/3s+R1N2/P4NpPP/1P1PP4/2GSG1+l2/L2K3N1 b RS2Pb4p 133",
        )
        .unwrap();
        let attacker = position.side_to_move();
        let mut searcher = MateSearcher::new(MateSearchLimits::nodes(10_000));
        assert!(matches!(
            searcher.search(&mut position, attacker, 5),
            MateSearchResult::ProvenMate { .. }
        ));
    }

    #[test]
    fn escape_is_proven_no_mate() {
        let mut position = Position::default();
        let attacker = position.side_to_move();
        let mut searcher = MateSearcher::new(MateSearchLimits::nodes(100_000));
        assert_eq!(
            MateSearchResult::ProvenNoMateWithinHorizon,
            searcher.search(&mut position, attacker, 3)
        );
    }

    #[test]
    fn repetition_is_not_treated_as_mate() {
        let mut position = Position::default();
        let hash = PositionHasher::calculate_hash(&position);
        let attacker = position.side_to_move();
        let mut searcher = MateSearcher::new(MateSearchLimits::nodes(100));
        searcher.path.insert(hash);
        let result = searcher.search_node(&mut position, attacker, 3);
        assert_eq!(MateSearchResult::ProvenNoMateWithinHorizon, result.result);
        assert!(result.path_dependent);
    }

    #[test]
    fn root_child_root_cycle_is_path_dependent() {
        let mut position = Position::default();
        let root_hash = PositionHasher::calculate_hash(&position);
        let first = crate::utils::parse_usi_move_for_color("7g7f", position.side_to_move())
            .expect("legal first move");
        position.do_move(first);
        let child_hash = PositionHasher::calculate_hash(&position);
        let second = crate::utils::parse_usi_move_for_color("3c3d", position.side_to_move())
            .expect("legal second move");
        position.do_move(second);
        let third = crate::utils::parse_usi_move_for_color("7f7g", position.side_to_move())
            .expect("legal third move");
        position.do_move(third);
        let fourth = crate::utils::parse_usi_move_for_color("3d3c", position.side_to_move())
            .expect("legal fourth move");
        position.do_move(fourth);
        assert_eq!(root_hash, PositionHasher::calculate_hash(&position));

        let mut searcher = MateSearcher::new(MateSearchLimits::nodes(100));
        searcher.path.insert(root_hash);
        searcher.path.insert(child_hash);
        let result = searcher.search_node(&mut position, Color::Black, 3);
        assert_eq!(MateSearchResult::ProvenNoMateWithinHorizon, result.result);
        assert!(result.path_dependent);
    }

    #[test]
    fn illegal_pawn_drop_mate_is_not_a_proof_move() {
        let mut position =
            position_from_sfen_or_usi("9/7pp/8k/7P1/7G1/9/9/9/9 b P2r2b3g4s4n4l14p 1").unwrap();
        assert!(position
            .legal_moves()
            .iter()
            .all(|mv| format_move_usi(*mv) != "P*1d"));
        let attacker = position.side_to_move();
        let mut searcher = MateSearcher::new(MateSearchLimits::nodes(10_000));
        let result = searcher.search(&mut position, attacker, 1);
        assert!(!matches!(
            result,
            MateSearchResult::ProvenMate {
                first_move: Some(mv),
                ..
            } if format_move_usi(mv) == "P*1d"
        ));
    }

    #[test]
    fn exhausted_node_budget_is_unknown() {
        let record = dev_record(7);
        let mut position = position_from_sfen_or_usi(record["sfen"].as_str().unwrap()).unwrap();
        let attacker = position.side_to_move();
        let mut searcher = MateSearcher::new(MateSearchLimits::nodes(1));
        assert_eq!(
            MateSearchResult::Unknown,
            searcher.search(&mut position, attacker, 7)
        );
        assert!(searcher.stopped());
        assert_eq!(
            Some(MateSearchStopReason::NodeLimit),
            searcher.stop_reason()
        );
    }

    #[test]
    fn root_hash_in_prior_history_is_allowed_once() {
        let record = dev_record(1);
        let mut position = position_from_sfen_or_usi(record["sfen"].as_str().unwrap()).unwrap();
        let root_hash = PositionHasher::calculate_hash(&position);
        let attacker = position.side_to_move();
        let mut searcher = MateSearcher::new(MateSearchLimits {
            node_limit: 10_000,
            deadline: None,
            stop_signal: None,
            prior_position_hashes: vec![root_hash, root_hash],
        });
        assert!(matches!(
            searcher.search(&mut position, attacker, 1),
            MateSearchResult::ProvenMate { .. }
        ));
    }

    #[test]
    fn deadline_and_external_stop_are_reported_separately() {
        let mut position = Position::default();
        let attacker = position.side_to_move();
        let mut deadline_searcher = MateSearcher::new(MateSearchLimits {
            node_limit: 100,
            deadline: Some(Instant::now()),
            stop_signal: None,
            prior_position_hashes: Vec::new(),
        });
        assert_eq!(
            MateSearchResult::Unknown,
            deadline_searcher.search(&mut position, attacker, 1)
        );
        assert_eq!(
            Some(MateSearchStopReason::TimeLimit),
            deadline_searcher.stop_reason()
        );

        let signal = Arc::new(AtomicBool::new(true));
        let mut external_searcher = MateSearcher::new(MateSearchLimits {
            node_limit: 100,
            deadline: None,
            stop_signal: Some(signal),
            prior_position_hashes: Vec::new(),
        });
        assert_eq!(
            MateSearchResult::Unknown,
            external_searcher.search(&mut position, attacker, 1)
        );
        assert_eq!(
            Some(MateSearchStopReason::ExternalStop),
            external_searcher.stop_reason()
        );
    }

    #[test]
    fn increasing_budget_never_turns_a_proof_into_a_disproof() {
        let record = dev_record(7);
        let position = position_from_sfen_or_usi(record["sfen"].as_str().unwrap()).unwrap();
        let attacker = position.side_to_move();
        let budgets = [1, 64, 512, 2_048, 20_000];
        let mut saw_proof = false;
        for budget in budgets {
            let mut candidate = position.clone();
            let mut searcher = MateSearcher::new(MateSearchLimits::nodes(budget));
            let result = searcher.search(&mut candidate, attacker, 7);
            if saw_proof {
                assert!(matches!(result, MateSearchResult::ProvenMate { .. }));
            }
            saw_proof |= matches!(result, MateSearchResult::ProvenMate { .. });
        }
        assert!(saw_proof);
    }
}
