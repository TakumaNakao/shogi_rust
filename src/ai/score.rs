pub(super) const MATE_SCORE: f32 = 1_000_000.0;
const MATE_THRESHOLD: f32 = MATE_SCORE - 1_000.0;
pub(super) const REPETITION_WIN_SCORE: f32 = 500_000.0;

#[inline]
pub(super) fn mate_loss_score(ply_from_root: u16) -> f32 {
    -MATE_SCORE + f32::from(ply_from_root)
}

#[inline]
pub(super) fn score_to_tt(score: f32, ply_from_root: u16) -> f32 {
    if score >= MATE_THRESHOLD {
        score + f32::from(ply_from_root)
    } else if score <= -MATE_THRESHOLD {
        score - f32::from(ply_from_root)
    } else {
        score
    }
}

#[inline]
pub(super) fn score_from_tt(score: f32, ply_from_root: u16) -> f32 {
    if score >= MATE_THRESHOLD {
        score - f32::from(ply_from_root)
    } else if score <= -MATE_THRESHOLD {
        score + f32::from(ply_from_root)
    } else {
        score
    }
}

#[inline]
pub(super) fn is_history_dependent_score(score: f32) -> bool {
    score == 0.0 || (score.abs() >= REPETITION_WIN_SCORE - 1.0 && score.abs() < MATE_THRESHOLD)
}
