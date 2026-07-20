pub(super) const USI_SCORE_CP_LIMIT: i32 = 2_000;
const USI_SCORE_CP_SOFT_START: i32 = 1_000;
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

pub(super) fn usi_display_score_cp(score: f32) -> i32 {
    if !score.is_finite() {
        return if score.is_sign_negative() {
            -USI_SCORE_CP_LIMIT
        } else {
            USI_SCORE_CP_LIMIT
        };
    }

    let sign = if score < 0.0 { -1 } else { 1 };
    let abs_score = score.abs();
    let soft_start = USI_SCORE_CP_SOFT_START as f32;
    let limit = USI_SCORE_CP_LIMIT as f32;
    let displayed = if abs_score <= soft_start {
        abs_score
    } else {
        let tail = limit - soft_start;
        soft_start + tail * (1.0 - (-(abs_score - soft_start) / tail).exp())
    };

    sign * (displayed.round() as i32).min(USI_SCORE_CP_LIMIT)
}
