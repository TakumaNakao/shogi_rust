use super::codec::HalfKpFeatureRow;
use super::HALFKP_HIDDEN;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps, _mm256_sub_ps};

#[inline]
pub(super) fn halfkp_avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn apply_feature_rows_avx2(
    hidden: &mut [f32; HALFKP_HIDDEN],
    rows: &[HalfKpFeatureRow],
    removed: &[usize],
    added: &[usize],
) {
    for lane in 0..(HALFKP_HIDDEN / 8) {
        let offset = lane * 8;
        let mut value = unsafe { _mm256_loadu_ps(hidden.as_ptr().add(offset)) };
        for &feature in removed {
            let row = &rows[feature].0;
            let delta = unsafe { std::arch::x86_64::_mm256_load_ps(row.as_ptr().add(offset)) };
            value = _mm256_sub_ps(value, delta);
        }
        for &feature in added {
            let row = &rows[feature].0;
            let delta = unsafe { std::arch::x86_64::_mm256_load_ps(row.as_ptr().add(offset)) };
            value = _mm256_add_ps(value, delta);
        }
        unsafe { _mm256_storeu_ps(hidden.as_mut_ptr().add(offset), value) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn accumulate_rows_avx2(
    hidden: &mut [f32; HALFKP_HIDDEN],
    bias: &[f32; HALFKP_HIDDEN],
    rows: &[HalfKpFeatureRow],
    features: &[usize],
) {
    for lane in 0..(HALFKP_HIDDEN / 8) {
        let offset = lane * 8;
        let mut value = unsafe { _mm256_loadu_ps(bias.as_ptr().add(offset)) };
        for &feature in features {
            let row = &rows[feature].0;
            let delta = unsafe { std::arch::x86_64::_mm256_load_ps(row.as_ptr().add(offset)) };
            value = _mm256_add_ps(value, delta);
        }
        unsafe { _mm256_storeu_ps(hidden.as_mut_ptr().add(offset), value) };
    }
}
