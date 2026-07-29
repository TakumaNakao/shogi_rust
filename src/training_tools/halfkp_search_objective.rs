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
}
