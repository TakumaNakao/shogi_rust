//! Bounded, exact mate proofs used before any learned evaluator is available.

use anyhow::{anyhow, Result};
use shogi_core::{Color, Move};
use shogi_lib::Position;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttackMoveMode {
    CheckingOnly,
    AllLegal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MateProofStatus {
    ProvenMate,
    ProvenNoMate,
    ResourceLimit,
}

#[derive(Clone, Debug)]
pub struct MateProofEdge {
    pub mv: Move,
    pub child: Box<MateProofNode>,
}

#[derive(Clone, Debug)]
pub struct MateProofNode {
    pub position_key: u64,
    pub side_to_move: Color,
    pub attacker_to_move: bool,
    pub remaining_plies: u16,
    pub status: MateProofStatus,
    pub mate_distance: Option<u16>,
    pub edges: Vec<MateProofEdge>,
}

#[derive(Clone, Debug)]
pub struct MateProof {
    pub attacker: Color,
    pub mode: AttackMoveMode,
    pub max_plies: u16,
    pub nodes: u64,
    pub root: MateProofNode,
}

#[derive(Clone, Copy, Debug)]
pub struct MateLimits {
    pub max_plies: u16,
    pub max_nodes: u64,
    pub mode: AttackMoveMode,
}

pub fn solve_bounded_mate(position: &Position, limits: MateLimits) -> Result<MateProof> {
    let attacker = position.side_to_move();
    solve_bounded_mate_for(position, attacker, limits)
}

pub fn solve_bounded_mate_for(
    position: &Position,
    attacker: Color,
    limits: MateLimits,
) -> Result<MateProof> {
    if limits.max_nodes == 0 {
        return Err(anyhow!("mate node limit must be non-zero"));
    }
    let mut context = Context {
        attacker,
        mode: limits.mode,
        max_nodes: limits.max_nodes,
        nodes: 0,
        path: HashSet::new(),
    };
    let root = context.solve(position, limits.max_plies);
    Ok(MateProof {
        attacker,
        mode: limits.mode,
        max_plies: limits.max_plies,
        nodes: context.nodes,
        root,
    })
}

struct Context {
    attacker: Color,
    mode: AttackMoveMode,
    max_nodes: u64,
    nodes: u64,
    path: HashSet<u64>,
}

impl Context {
    fn solve(&mut self, position: &Position, remaining: u16) -> MateProofNode {
        let key = position.key();
        let attacker_to_move = position.side_to_move() == self.attacker;
        let base = |status, mate_distance, edges| MateProofNode {
            position_key: key,
            side_to_move: position.side_to_move(),
            attacker_to_move,
            remaining_plies: remaining,
            status,
            mate_distance,
            edges,
        };
        if self.nodes >= self.max_nodes {
            return base(MateProofStatus::ResourceLimit, None, vec![]);
        }
        self.nodes += 1;
        if !self.path.insert(key) {
            return base(MateProofStatus::ProvenNoMate, None, vec![]);
        }
        let legal = position.legal_moves();
        if legal.is_empty() {
            self.path.remove(&key);
            return if position.in_check() && !attacker_to_move {
                base(MateProofStatus::ProvenMate, Some(0), vec![])
            } else {
                base(MateProofStatus::ProvenNoMate, None, vec![])
            };
        }
        if remaining == 0 {
            self.path.remove(&key);
            return base(MateProofStatus::ProvenNoMate, None, vec![]);
        }

        let moves = if attacker_to_move && self.mode == AttackMoveMode::CheckingOnly {
            legal
                .into_iter()
                .filter(|&mv| position.is_check_move(mv))
                .collect::<Vec<_>>()
        } else {
            legal.into_iter().collect::<Vec<_>>()
        };
        let node = if attacker_to_move {
            self.solve_or(position, remaining, moves, base)
        } else {
            self.solve_and(position, remaining, moves, base)
        };
        self.path.remove(&key);
        node
    }

    fn solve_or<F>(
        &mut self,
        position: &Position,
        remaining: u16,
        moves: Vec<Move>,
        base: F,
    ) -> MateProofNode
    where
        F: Fn(MateProofStatus, Option<u16>, Vec<MateProofEdge>) -> MateProofNode,
    {
        let mut all_edges = Vec::new();
        let mut resource = false;
        let mut best: Option<(u16, MateProofEdge)> = None;
        for mv in moves {
            let mut child_position = position.clone();
            child_position.do_move(mv);
            let child = self.solve(&child_position, remaining - 1);
            match child.status {
                MateProofStatus::ProvenMate => {
                    let distance = child.mate_distance.expect("proven mate distance") + 1;
                    let edge = MateProofEdge {
                        mv,
                        child: Box::new(child),
                    };
                    if best.as_ref().is_none_or(|(best, _)| distance < *best) {
                        best = Some((distance, edge));
                    }
                }
                MateProofStatus::ResourceLimit => {
                    resource = true;
                    all_edges.push(MateProofEdge {
                        mv,
                        child: Box::new(child),
                    });
                }
                MateProofStatus::ProvenNoMate => {
                    all_edges.push(MateProofEdge {
                        mv,
                        child: Box::new(child),
                    });
                }
            }
        }
        if let Some((distance, edge)) = best {
            base(MateProofStatus::ProvenMate, Some(distance), vec![edge])
        } else if resource {
            base(MateProofStatus::ResourceLimit, None, all_edges)
        } else {
            base(MateProofStatus::ProvenNoMate, None, all_edges)
        }
    }

    fn solve_and<F>(
        &mut self,
        position: &Position,
        remaining: u16,
        moves: Vec<Move>,
        base: F,
    ) -> MateProofNode
    where
        F: Fn(MateProofStatus, Option<u16>, Vec<MateProofEdge>) -> MateProofNode,
    {
        let mut edges = Vec::new();
        let mut resource = false;
        let mut longest = 0;
        for mv in moves {
            let mut child_position = position.clone();
            child_position.do_move(mv);
            let child = self.solve(&child_position, remaining - 1);
            let status = child.status;
            if status == MateProofStatus::ProvenMate {
                longest = longest.max(child.mate_distance.expect("proven mate distance") + 1);
            }
            edges.push(MateProofEdge {
                mv,
                child: Box::new(child),
            });
            match status {
                MateProofStatus::ProvenNoMate => {
                    return base(MateProofStatus::ProvenNoMate, None, edges)
                }
                MateProofStatus::ResourceLimit => resource = true,
                MateProofStatus::ProvenMate => {}
            }
        }
        if resource {
            base(MateProofStatus::ResourceLimit, None, edges)
        } else {
            base(MateProofStatus::ProvenMate, Some(longest), edges)
        }
    }
}

pub fn verify_mate_proof(position: &Position, proof: &MateProof) -> Result<()> {
    if proof.root.remaining_plies != proof.max_plies {
        return Err(anyhow!("mate proof root metadata mismatch"));
    }
    verify_node(
        position,
        proof.attacker,
        proof.mode,
        &proof.root,
        &mut HashSet::new(),
    )
}

fn verify_node(
    position: &Position,
    attacker: Color,
    mode: AttackMoveMode,
    node: &MateProofNode,
    path: &mut HashSet<u64>,
) -> Result<()> {
    if node.position_key != position.key()
        || node.side_to_move != position.side_to_move()
        || node.attacker_to_move != (position.side_to_move() == attacker)
    {
        return Err(anyhow!("mate proof node does not match position"));
    }
    if node.status == MateProofStatus::ResourceLimit {
        return Ok(());
    }
    if !path.insert(position.key()) {
        if node.status != MateProofStatus::ProvenNoMate || !node.edges.is_empty() {
            return Err(anyhow!("invalid repetition refutation"));
        }
        return Ok(());
    }
    let legal = position.legal_moves();
    if legal.is_empty() {
        let expected = if position.in_check() && !node.attacker_to_move {
            MateProofStatus::ProvenMate
        } else {
            MateProofStatus::ProvenNoMate
        };
        if node.status != expected || !node.edges.is_empty() {
            return Err(anyhow!("invalid terminal mate proof"));
        }
        path.remove(&position.key());
        return Ok(());
    }
    if node.remaining_plies == 0 {
        if node.status != MateProofStatus::ProvenNoMate
            || node.mate_distance.is_some()
            || !node.edges.is_empty()
        {
            return Err(anyhow!("invalid horizon refutation"));
        }
        path.remove(&position.key());
        return Ok(());
    }
    let admissible = if node.attacker_to_move && mode == AttackMoveMode::CheckingOnly {
        legal
            .into_iter()
            .filter(|&mv| position.is_check_move(mv))
            .collect::<Vec<_>>()
    } else {
        legal.into_iter().collect::<Vec<_>>()
    };
    for edge in &node.edges {
        if !admissible.contains(&edge.mv) {
            return Err(anyhow!("proof edge is not admissible"));
        }
        let mut child = position.clone();
        child.do_move(edge.mv);
        verify_node(&child, attacker, mode, &edge.child, path)?;
    }
    let universal = (node.attacker_to_move && node.status == MateProofStatus::ProvenNoMate)
        || (!node.attacker_to_move && node.status == MateProofStatus::ProvenMate);
    if universal
        && (node.edges.len() != admissible.len()
            || admissible
                .iter()
                .any(|mv| !node.edges.iter().any(|edge| edge.mv == *mv)))
    {
        return Err(anyhow!("universal proof omits a legal branch"));
    }
    match (node.attacker_to_move, node.status) {
        (true, MateProofStatus::ProvenMate) => {
            if node.edges.len() != 1 || node.edges[0].child.status != MateProofStatus::ProvenMate {
                return Err(anyhow!("invalid OR mate proof"));
            }
        }
        (true, MateProofStatus::ProvenNoMate) => {
            if node
                .edges
                .iter()
                .any(|edge| edge.child.status != MateProofStatus::ProvenNoMate)
            {
                return Err(anyhow!("invalid OR no-mate proof"));
            }
        }
        (false, MateProofStatus::ProvenMate) => {
            if node
                .edges
                .iter()
                .any(|edge| edge.child.status != MateProofStatus::ProvenMate)
            {
                return Err(anyhow!("invalid AND mate proof"));
            }
        }
        (false, MateProofStatus::ProvenNoMate) => {
            if !node
                .edges
                .iter()
                .any(|edge| edge.child.status == MateProofStatus::ProvenNoMate)
            {
                return Err(anyhow!("invalid AND refutation"));
            }
        }
        (_, MateProofStatus::ResourceLimit) => unreachable!(),
    }
    path.remove(&position.key());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::position_from_sfen_or_usi;

    const ONE_MOVE_MATE: &str = "sfen 8k/9/6NG1/9/9/9/9/9/K8 b R 1";

    #[test]
    fn proves_and_revalidates_known_one_move_mate() {
        let position = position_from_sfen_or_usi(ONE_MOVE_MATE).expect("position");
        let proof = solve_bounded_mate(
            &position,
            MateLimits {
                max_plies: 1,
                max_nodes: 10_000,
                mode: AttackMoveMode::CheckingOnly,
            },
        )
        .expect("solve");
        assert_eq!(proof.root.status, MateProofStatus::ProvenMate);
        assert_eq!(proof.root.mate_distance, Some(1));
        verify_mate_proof(&position, &proof).expect("verify");
    }

    #[test]
    fn node_cap_is_not_mislabeled_as_no_mate() {
        let position = Position::default();
        let proof = solve_bounded_mate(
            &position,
            MateLimits {
                max_plies: 3,
                max_nodes: 1,
                mode: AttackMoveMode::AllLegal,
            },
        )
        .expect("solve");
        assert_eq!(proof.root.status, MateProofStatus::ResourceLimit);
    }

    #[test]
    fn revalidates_no_mate_children_at_the_horizon() {
        let position = Position::default();
        let proof = solve_bounded_mate(
            &position,
            MateLimits {
                max_plies: 1,
                max_nodes: 10_000,
                mode: AttackMoveMode::AllLegal,
            },
        )
        .expect("solve");
        assert_eq!(proof.root.status, MateProofStatus::ProvenNoMate);
        verify_mate_proof(&position, &proof).expect("verify");
    }
}
