//! Training-only HalfKP virtual features. Runtime files remain ordinary HalfKP.

use anyhow::{anyhow, Result};
use shogi_ai::evaluation::{
    HALFKP_HIDDEN, HALFKP_INPUTS, HALFKP_KING_BUCKETS, HALFKP_PIECE_STATES, NUM_BOARD_PIECE_KINDS,
    NUM_SQUARES,
};

const RELATIVE_AXIS: usize = 17;
pub const HALFKP_RELATIVE_FACTORS: usize =
    2 * NUM_BOARD_PIECE_KINDS * RELATIVE_AXIS * RELATIVE_AXIS;
const BOARD_STATES: usize = 2 * NUM_BOARD_PIECE_KINDS * NUM_SQUARES;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FactorIndices {
    pub piece: usize,
    pub relative: Option<usize>,
}

#[derive(Clone)]
pub struct HalfKpFactorization {
    pub piece: Vec<f32>,
    pub relative: Vec<f32>,
}

impl HalfKpFactorization {
    pub fn zeros() -> Self {
        Self {
            piece: vec![0.0; HALFKP_PIECE_STATES * HALFKP_HIDDEN],
            relative: vec![0.0; HALFKP_RELATIVE_FACTORS * HALFKP_HIDDEN],
        }
    }

    pub fn add_feature(&self, feature: usize, hidden: &mut [f32; HALFKP_HIDDEN]) -> Result<()> {
        let factors = factor_indices(feature)?;
        add_row(hidden, &self.piece, factors.piece);
        if let Some(relative) = factors.relative {
            add_row(hidden, &self.relative, relative);
        }
        Ok(())
    }

    /// Fold virtual factors into direct rows before writing a runtime model.
    pub fn fold_into(&self, direct: &mut [f32]) -> Result<()> {
        if direct.len() != HALFKP_INPUTS * HALFKP_HIDDEN {
            return Err(anyhow!("invalid HalfKP direct tensor length"));
        }
        for feature in 0..HALFKP_INPUTS {
            let factors = factor_indices(feature)?;
            add_row_to_direct(direct, feature, &self.piece, factors.piece);
            if let Some(relative) = factors.relative {
                add_row_to_direct(direct, feature, &self.relative, relative);
            }
        }
        Ok(())
    }

    pub fn project_lr_symmetry(&mut self) {
        for color_kind in 0..2 * NUM_BOARD_PIECE_KINDS {
            for file_delta in 0..RELATIVE_AXIS / 2 {
                for rank_delta in 0..RELATIVE_AXIS {
                    let a = (color_kind * RELATIVE_AXIS + file_delta) * RELATIVE_AXIS + rank_delta;
                    let b = (color_kind * RELATIVE_AXIS + (RELATIVE_AXIS - 1 - file_delta))
                        * RELATIVE_AXIS
                        + rank_delta;
                    average_rows(&mut self.relative, a, b);
                }
            }
        }
    }
}

pub fn factor_indices(feature: usize) -> Result<FactorIndices> {
    if feature >= HALFKP_INPUTS {
        return Err(anyhow!("HalfKP feature out of range"));
    }
    let king = feature / HALFKP_PIECE_STATES;
    let piece = feature % HALFKP_PIECE_STATES;
    let relative = (piece < BOARD_STATES).then(|| relative_index(king, piece));
    Ok(FactorIndices { piece, relative })
}

fn relative_index(king: usize, piece: usize) -> usize {
    debug_assert!(king < HALFKP_KING_BUCKETS);
    debug_assert!(piece < BOARD_STATES);
    let king_file = king / 9;
    let king_rank = king % 9;
    let color_kind = piece / NUM_SQUARES;
    let square = piece % NUM_SQUARES;
    let piece_file = square / 9;
    let piece_rank = square % 9;
    let file_delta = (piece_file as isize - king_file as isize + 8) as usize;
    let rank_delta = (piece_rank as isize - king_rank as isize + 8) as usize;
    (color_kind * RELATIVE_AXIS + file_delta) * RELATIVE_AXIS + rank_delta
}

fn add_row(target: &mut [f32; HALFKP_HIDDEN], factors: &[f32], row: usize) {
    let start = row * HALFKP_HIDDEN;
    for h in 0..HALFKP_HIDDEN {
        target[h] += factors[start + h];
    }
}

fn add_row_to_direct(direct: &mut [f32], direct_row: usize, factors: &[f32], factor_row: usize) {
    let direct_start = direct_row * HALFKP_HIDDEN;
    let factor_start = factor_row * HALFKP_HIDDEN;
    for h in 0..HALFKP_HIDDEN {
        direct[direct_start + h] += factors[factor_start + h];
    }
}

fn average_rows(values: &mut [f32], a: usize, b: usize) {
    let a = a * HALFKP_HIDDEN;
    let b = b * HALFKP_HIDDEN;
    for h in 0..HALFKP_HIDDEN {
        let mean = 0.5 * (values[a + h] + values[b + h]);
        values[a + h] = mean;
        values[b + h] = mean;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_features_have_only_piece_factor() {
        let feature = 7 * HALFKP_PIECE_STATES + BOARD_STATES;
        let indices = factor_indices(feature).expect("indices");
        assert_eq!(indices.piece, BOARD_STATES);
        assert_eq!(indices.relative, None);
    }

    #[test]
    fn relative_factor_shares_equal_geometry_across_king_buckets() {
        // Own pawn one file right and one rank down from the king.
        let first_king = 1 * 9 + 2;
        let first_square = 2 * 9 + 3;
        let second_king = 3 * 9 + 5;
        let second_square = 4 * 9 + 6;
        let first = factor_indices(first_king * HALFKP_PIECE_STATES + first_square)
            .expect("first")
            .relative;
        let second = factor_indices(second_king * HALFKP_PIECE_STATES + second_square)
            .expect("second")
            .relative;
        assert_eq!(first, second);
    }

    #[test]
    fn folding_matches_training_time_accumulation() {
        let feature = 4 * HALFKP_PIECE_STATES + 123;
        let indices = factor_indices(feature).expect("indices");
        let mut factors = HalfKpFactorization::zeros();
        for h in 0..HALFKP_HIDDEN {
            factors.piece[indices.piece * HALFKP_HIDDEN + h] = h as f32 * 0.01;
            factors.relative[indices.relative.expect("relative") * HALFKP_HIDDEN + h] =
                h as f32 * -0.002;
        }
        let mut expected = [0.0; HALFKP_HIDDEN];
        factors.add_feature(feature, &mut expected).expect("add");
        let mut direct = vec![0.0; HALFKP_INPUTS * HALFKP_HIDDEN];
        factors.fold_into(&mut direct).expect("fold");
        assert_eq!(
            &direct[feature * HALFKP_HIDDEN..(feature + 1) * HALFKP_HIDDEN],
            &expected
        );
    }
}
