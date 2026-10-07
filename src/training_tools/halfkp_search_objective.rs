//! Objective-aligned metrics for the recorded-move versus search-best pair.

use shogi_ai::halfkp_training::{SearchTeacherRecord, CANDIDATE_GAME_MOVE};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GameVsSearchPair {
    pub game_index: usize,
    pub search_best_index: usize,
    pub teacher_regret_cp: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GameVsSearchObservation {
    /// `utility(game move) - utility(search best)`. Larger is better.
    pub signed_margin_cp: f32,
    /// The unit-weight softplus loss used by the training objective. Smaller is better.
    pub objective_loss: f32,
    pub correct: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GameRankPair {
    pub game_index: usize,
    pub bad_index: usize,
    pub teacher_regret_cp: f32,
    pub teacher_gap_cp: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GameRankObservation {
    /// `utility(game move) - utility(hard negative)`. Larger is better.
    pub signed_margin_cp: f32,
    /// The unit-weight softplus loss used by the training objective. Smaller is better.
    pub objective_loss: f32,
    pub correct: bool,
}

pub fn eligible_game_vs_search_pair(
    record: &SearchTeacherRecord,
    regret_cap_cp: f32,
) -> Option<GameVsSearchPair> {
    let search_best_index = record
        .candidates
        .iter()
        .enumerate()
        .max_by(|lhs, rhs| lhs.1.score_cp.total_cmp(&rhs.1.score_cp))
        .map(|(index, _)| index)?;
    let game_index = record
        .candidates
        .iter()
        .position(|candidate| candidate.flags & CANDIDATE_GAME_MOVE != 0)?;
    if game_index == search_best_index {
        return None;
    }
    let teacher_regret_cp =
        record.candidates[search_best_index].score_cp - record.candidates[game_index].score_cp;
    if !teacher_regret_cp.is_finite()
        || teacher_regret_cp < 0.0
        || teacher_regret_cp > regret_cap_cp
    {
        return None;
    }
    Some(GameVsSearchPair {
        game_index,
        search_best_index,
        teacher_regret_cp,
    })
}

pub fn observe_game_vs_search_pair(
    utilities: &[f32],
    pair: GameVsSearchPair,
    rank_margin_cp: f32,
    rank_temperature_cp: f32,
) -> GameVsSearchObservation {
    let signed_margin_cp = utilities[pair.game_index] - utilities[pair.search_best_index];
    let normalized_violation = (rank_margin_cp - signed_margin_cp) / rank_temperature_cp;
    GameVsSearchObservation {
        signed_margin_cp,
        objective_loss: softplus(normalized_violation),
        correct: signed_margin_cp > 0.0,
    }
}

/// Select the currently most overvalued candidate that the frozen search teacher
/// scores at least `minimum_teacher_gap_cp` below the recorded game move.
///
/// Passing the frozen parent utilities fixes the pair for model comparison.
/// Passing the current training utilities performs online hard-negative mining.
pub fn select_game_rank_pair(
    record: &SearchTeacherRecord,
    selection_utilities: &[f32],
    regret_cap_cp: f32,
    minimum_teacher_gap_cp: f32,
) -> Option<GameRankPair> {
    if selection_utilities.len() != record.candidates.len()
        || !regret_cap_cp.is_finite()
        || regret_cap_cp < 0.0
        || !minimum_teacher_gap_cp.is_finite()
        || minimum_teacher_gap_cp < 0.0
    {
        return None;
    }
    let teacher_best_index = record
        .candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| candidate.score_cp.is_finite())
        .max_by(|lhs, rhs| lhs.1.score_cp.total_cmp(&rhs.1.score_cp))
        .map(|(index, _)| index)?;
    let game_index = record
        .candidates
        .iter()
        .position(|candidate| candidate.flags & CANDIDATE_GAME_MOVE != 0)?;
    let game_score = record.candidates[game_index].score_cp;
    let teacher_regret_cp = record.candidates[teacher_best_index].score_cp - game_score;
    if !game_score.is_finite()
        || !teacher_regret_cp.is_finite()
        || teacher_regret_cp < 0.0
        || teacher_regret_cp > regret_cap_cp
    {
        return None;
    }
    let bad_index = record
        .candidates
        .iter()
        .enumerate()
        .filter(|(index, candidate)| {
            *index != game_index
                && candidate.score_cp.is_finite()
                && selection_utilities[*index].is_finite()
                && game_score - candidate.score_cp >= minimum_teacher_gap_cp
        })
        .max_by(|(lhs, _), (rhs, _)| {
            selection_utilities[*lhs].total_cmp(&selection_utilities[*rhs])
        })
        .map(|(index, _)| index)?;
    Some(GameRankPair {
        game_index,
        bad_index,
        teacher_regret_cp,
        teacher_gap_cp: game_score - record.candidates[bad_index].score_cp,
    })
}

pub fn observe_game_rank_pair(
    utilities: &[f32],
    pair: GameRankPair,
    rank_margin_cp: f32,
    rank_temperature_cp: f32,
) -> GameRankObservation {
    let signed_margin_cp = utilities[pair.game_index] - utilities[pair.bad_index];
    let normalized_violation = (rank_margin_cp - signed_margin_cp) / rank_temperature_cp;
    GameRankObservation {
        signed_margin_cp,
        objective_loss: softplus(normalized_violation),
        correct: signed_margin_cp > 0.0,
    }
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        (1.0 + value.exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_ai::halfkp_training::{
        PackedHalfKpPosition, SearchTeacherCandidate, CANDIDATE_SEARCH_BEST,
    };
    use shogi_core::Color;

    fn position() -> PackedHalfKpPosition {
        PackedHalfKpPosition {
            side_to_move: Color::Black,
            features_black: vec![],
            features_white: vec![],
            material_black: 0.0,
            material_white: 0.0,
        }
    }

    fn record(game_score: f32, search_score: f32) -> SearchTeacherRecord {
        SearchTeacherRecord {
            position_hash: 0,
            ply: 0,
            root: position(),
            root_search_score_cp: 0.0,
            result: None,
            phase: 1,
            teacher_depth: 4,
            sample_weight: 1.0,
            candidates: vec![
                SearchTeacherCandidate {
                    flags: CANDIDATE_GAME_MOVE,
                    score_cp: game_score,
                    child: position(),
                },
                SearchTeacherCandidate {
                    flags: CANDIDATE_SEARCH_BEST,
                    score_cp: search_score,
                    child: position(),
                },
            ],
        }
    }

    #[test]
    fn eligibility_uses_teacher_regret_cap() {
        let record = record(100.0, 250.0);
        assert!(eligible_game_vs_search_pair(&record, 149.0).is_none());
        let pair = eligible_game_vs_search_pair(&record, 150.0).unwrap();
        assert_eq!(pair.game_index, 0);
        assert_eq!(pair.search_best_index, 1);
        assert_eq!(pair.teacher_regret_cp, 150.0);
    }

    #[test]
    fn objective_loss_matches_training_pair_direction() {
        let pair = eligible_game_vs_search_pair(&record(100.0, 250.0), 300.0).unwrap();
        let losing = observe_game_vs_search_pair(&[-30.0, 40.0], pair, 20.0, 100.0);
        let winning = observe_game_vs_search_pair(&[60.0, 40.0], pair, 20.0, 100.0);
        assert_eq!(losing.signed_margin_cp, -70.0);
        assert_eq!(winning.signed_margin_cp, 20.0);
        assert!(!losing.correct);
        assert!(winning.correct);
        assert!(winning.objective_loss < losing.objective_loss);
        assert!((winning.objective_loss - std::f32::consts::LN_2).abs() < 1e-6);
    }

    #[test]
    fn game_rank_selects_model_hardest_search_confirmed_negative() {
        let mut record = record(100.0, 250.0);
        record.candidates.extend([
            SearchTeacherCandidate {
                flags: 0,
                score_cp: 80.0,
                child: position(),
            },
            SearchTeacherCandidate {
                flags: 0,
                score_cp: 40.0,
                child: position(),
            },
        ]);
        let pair = select_game_rank_pair(&record, &[0.0, 5.0, 30.0, 20.0], 300.0, 15.0).unwrap();
        assert_eq!(pair.game_index, 0);
        assert_eq!(pair.bad_index, 2);
        assert_eq!(pair.teacher_regret_cp, 150.0);
        assert_eq!(pair.teacher_gap_cp, 20.0);
    }

    #[test]
    fn game_rank_enforces_regret_gap_and_finite_inputs() {
        let mut record = record(100.0, 250.0);
        record.candidates.push(SearchTeacherCandidate {
            flags: 0,
            score_cp: 90.0,
            child: position(),
        });
        assert!(select_game_rank_pair(&record, &[0.0, 1.0, 2.0], 149.0, 5.0).is_none());
        assert!(select_game_rank_pair(&record, &[0.0, 1.0, 2.0], 150.0, 15.0).is_none());
        assert!(select_game_rank_pair(&record, &[0.0, 1.0], 300.0, 5.0).is_none());
    }

    #[test]
    fn game_rank_objective_matches_training_direction() {
        let pair = GameRankPair {
            game_index: 0,
            bad_index: 1,
            teacher_regret_cp: 0.0,
            teacher_gap_cp: 20.0,
        };
        let losing = observe_game_rank_pair(&[-30.0, 40.0], pair, 20.0, 100.0);
        let winning = observe_game_rank_pair(&[60.0, 40.0], pair, 20.0, 100.0);
        assert_eq!(losing.signed_margin_cp, -70.0);
        assert_eq!(winning.signed_margin_cp, 20.0);
        assert!(!losing.correct);
        assert!(winning.correct);
        assert!(winning.objective_loss < losing.objective_loss);
        assert!((winning.objective_loss - std::f32::consts::LN_2).abs() < 1e-6);
    }
}
