#![allow(dead_code)]
mod codec;
mod constants;
mod debug;
mod evaluator;
mod facade;
mod features;
mod kernels;
mod sparse;
mod tiny_nnue;

use anyhow::{anyhow, Result};
use codec::{
    feature_rows_from_flat, read_f32_array, read_f32_le, read_f32_vec, read_u32_le,
    HalfKpFeatureRow,
};
pub use constants::*;
use constants::{piece_kind_value, unpromoted_kind};
pub use debug::{
    generate_sfen, index_to_kpp_info, is_promoted_piece_kind, piece_kind_to_sfen_char_base, KppInfo,
};
pub use evaluator::Evaluator;
pub use facade::{EngineEvaluator, HybridNnueEvaluator, SparseModelEvaluator};
use features::{extract_halfkp_features_fixed, halfkp_piece_state, HalfKpFixedFeatures};
pub use features::{
    extract_halfkp_features_for, extract_kpp_features, extract_kpp_features_and_material,
    extract_nnue_features, HalfKpFeatures, NnueFeatures,
};
use kernels::halfkp_avx2_available;
#[cfg(target_arch = "x86_64")]
use kernels::{accumulate_rows_avx2, apply_feature_rows_avx2};
use shogi_core::{Color, Move, Piece, PieceKind, Square};
use shogi_lib;
pub use sparse::{calculate_material_advantage, SparseModel};
use std::fs::File;
use std::io::Read;
use std::path::Path;
pub use tiny_nnue::TinyNnueModel;

#[derive(Debug, Clone)]
pub struct HalfKpModel {
    pub target_scale: f32,
    feature_emb: Box<[HalfKpFeatureRow]>,
    hidden_b: [f32; HALFKP_HIDDEN],
    // STM accumulator, non-STM accumulator, material, tempo.
    out_w: [f32; HALFKP_HIDDEN * 2 + 1],
    out_b: f32,
    use_avx2: bool,
}

/// Reusable feature-transformer state for one side's HalfKP perspective.
///
/// Search can retain one of these per side and refresh it after make/undo;
/// the move-delta update is intentionally kept separate from the model so it
/// can be tested without changing the Evaluator trait.
#[derive(Debug, Clone, Copy)]
pub struct HalfKpAccumulator {
    pub perspective: Color,
    pub king_bucket: usize,
    pub mirror: bool,
    pub hidden: [f32; HALFKP_HIDDEN],
    pub material: f32,
}

#[derive(Clone)]
pub struct HalfKpSearchContext {
    frames: Vec<HalfKpFrame>,
    ply: usize,
    pending: Option<HalfKpPendingMove>,
}

#[derive(Debug, Clone, Copy)]
struct HalfKpFrame {
    black: HalfKpAccumulator,
    white: HalfKpAccumulator,
}

#[derive(Clone, Copy)]
struct HalfKpPendingMove {
    mv: Move,
    moved: Option<Piece>,
    captured: Option<Piece>,
    hand_index: usize,
}

impl HalfKpModel {
    pub const MAGIC: &'static [u8; 8] = if cfg!(feature = "halfkp64") {
        b"HKP00064"
    } else {
        b"HKP00001"
    };

    pub fn load(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != Self::MAGIC {
            return Err(anyhow!("invalid HalfKP magic"));
        }
        let version = read_u32_le(&mut file)? as usize;
        let hidden = read_u32_le(&mut file)? as usize;
        let inputs = read_u32_le(&mut file)? as usize;
        let buckets = read_u32_le(&mut file)? as usize;
        let piece_states = read_u32_le(&mut file)? as usize;
        let target_scale = read_f32_le(&mut file)?;
        if version != 1
            || hidden != HALFKP_HIDDEN
            || inputs != HALFKP_INPUTS
            || buckets != HALFKP_KING_BUCKETS
            || piece_states != HALFKP_PIECE_STATES
            || !target_scale.is_finite()
            || target_scale <= 0.0
        {
            return Err(anyhow!("invalid HalfKP header"));
        }
        let feature_emb = feature_rows_from_flat(read_f32_vec(&mut file, inputs * hidden)?)?;
        let hidden_b = read_f32_array::<HALFKP_HIDDEN>(&mut file)?;
        let out_w = read_f32_array::<{ HALFKP_HIDDEN * 2 + 1 }>(&mut file)?;
        let out_b = read_f32_le(&mut file)?;
        let mut trailing = [0u8; 1];
        if file.read(&mut trailing)? != 0 {
            return Err(anyhow!("trailing bytes in HalfKP file"));
        }
        Ok(Self {
            target_scale,
            feature_emb,
            hidden_b,
            out_w,
            out_b,
            use_avx2: halfkp_avx2_available(),
        })
    }

    fn accumulator(&self, features: &HalfKpFeatures) -> [f32; HALFKP_HIDDEN] {
        let mut acc = self.hidden_b;
        #[cfg(target_arch = "x86_64")]
        if self.use_avx2 {
            unsafe {
                accumulate_rows_avx2(
                    &mut acc,
                    &self.hidden_b,
                    &self.feature_emb,
                    &features.features,
                );
            }
            return acc;
        }
        for &feature in &features.features {
            let row = &self.feature_emb[feature].0;
            for h in 0..HALFKP_HIDDEN {
                acc[h] += row[h];
            }
        }
        acc
    }

    pub fn accumulator_for_position(
        &self,
        pos: &shogi_lib::Position,
        perspective: Color,
    ) -> Option<HalfKpAccumulator> {
        let features = extract_halfkp_features_fixed(pos, perspective)?;
        Some(self.accumulator_from_fixed(&features, perspective))
    }

    pub fn begin_search_context(&self, pos: &shogi_lib::Position) -> Option<HalfKpSearchContext> {
        let black_features = extract_halfkp_features_fixed(pos, Color::Black)?;
        let white_features = extract_halfkp_features_fixed(pos, Color::White)?;
        let black = self.accumulator_from_fixed(&black_features, Color::Black);
        let white = self.accumulator_from_fixed(&white_features, Color::White);
        let mut frames = Vec::with_capacity(128);
        frames.push(HalfKpFrame { black, white });
        Some(HalfKpSearchContext {
            frames,
            ply: 0,
            pending: None,
        })
    }

    fn accumulator_from_features(&self, features: &HalfKpFeatures) -> HalfKpAccumulator {
        let hidden = self.accumulator(features);
        HalfKpAccumulator {
            perspective: if features.king_bucket < HALFKP_KING_BUCKETS {
                Color::Black
            } else {
                Color::White
            },
            king_bucket: features.king_bucket,
            mirror: false,
            hidden,
            material: features.material,
        }
    }

    fn accumulator_from_fixed(
        &self,
        features: &HalfKpFixedFeatures,
        perspective: Color,
    ) -> HalfKpAccumulator {
        let mut hidden = self.hidden_b;
        #[cfg(target_arch = "x86_64")]
        if self.use_avx2 {
            unsafe {
                accumulate_rows_avx2(
                    &mut hidden,
                    &self.hidden_b,
                    &self.feature_emb,
                    &features.features[..features.len],
                );
            }
        } else {
            for &feature in &features.features[..features.len] {
                let row = &self.feature_emb[feature].0;
                for h in 0..HALFKP_HIDDEN {
                    hidden[h] += row[h];
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        for &feature in &features.features[..features.len] {
            let row = &self.feature_emb[feature].0;
            for h in 0..HALFKP_HIDDEN {
                hidden[h] += row[h];
            }
        }
        HalfKpAccumulator {
            perspective,
            king_bucket: features.king_bucket,
            mirror: features.mirror,
            hidden,
            material: features.material,
        }
    }

    pub fn prepare_search_context(
        &self,
        ctx: &mut HalfKpSearchContext,
        pos: &shogi_lib::Position,
        mv: Move,
    ) {
        let (moved, captured, hand_index) = match mv {
            Move::Normal { from, to, .. } => {
                let moved = pos.piece_at(from);
                let captured = pos.piece_at(to);
                let hand_index = captured
                    .map(|piece| unpromoted_kind(piece.piece_kind()))
                    .and_then(|kind| pos.hand(pos.side_to_move()).count(kind))
                    .unwrap_or(0) as usize;
                (moved, captured, hand_index)
            }
            Move::Drop { piece, to: _ } => {
                let count = pos
                    .hand(piece.color())
                    .count(piece.piece_kind())
                    .unwrap_or(1);
                (Some(piece), None, count.saturating_sub(1) as usize)
            }
        };
        ctx.pending = Some(HalfKpPendingMove {
            mv,
            moved,
            captured,
            hand_index,
        });
    }

    pub fn commit_search_context_move(
        &self,
        ctx: &mut HalfKpSearchContext,
        pos: &shogi_lib::Position,
    ) {
        let Some(pending) = ctx.pending.take() else {
            return;
        };
        let parent = ctx.frames[ctx.ply];
        let mut child = parent;
        Self::apply_move_to_accumulator(self, &mut child.black, &pending, pos);
        Self::apply_move_to_accumulator(self, &mut child.white, &pending, pos);
        ctx.ply += 1;
        if ctx.ply == ctx.frames.len() {
            ctx.frames.push(child);
        } else {
            ctx.frames[ctx.ply] = child;
        }
    }

    fn apply_move_to_accumulator(
        &self,
        acc: &mut HalfKpAccumulator,
        pending: &HalfKpPendingMove,
        after: &shogi_lib::Position,
    ) {
        let Some(moved) = pending.moved else { return };
        if let Move::Normal { from, to, .. } = pending.mv {
            if moved.piece_kind() == PieceKind::King && moved.color() == acc.perspective {
                if let Some(refreshed) = self.accumulator_for_position(after, acc.perspective) {
                    *acc = refreshed;
                }
                return;
            }
            let mut removed = [0usize; 2];
            let mut removed_len = 0;
            if let Some(row) = self.piece_row(acc, moved, Some(from), 0) {
                removed[removed_len] = row;
                removed_len += 1;
            }
            if let Some(captured) = pending.captured {
                if let Some(row) = self.piece_row(acc, captured, Some(to), 0) {
                    removed[removed_len] = row;
                    removed_len += 1;
                }
            }
            let mut added = [0usize; 2];
            let mut added_len = 0;
            let after_piece = after.piece_at(to);
            if let Some(after_piece) = after_piece {
                if let Some(row) = self.piece_row(acc, after_piece, Some(to), 0) {
                    added[added_len] = row;
                    added_len += 1;
                }
            }
            if let Some(captured) = pending.captured {
                let hand_piece = Piece::new(unpromoted_kind(captured.piece_kind()), moved.color());
                if let Some(row) = self.piece_row(acc, hand_piece, None, pending.hand_index) {
                    added[added_len] = row;
                    added_len += 1;
                }
            }
            self.apply_feature_rows(acc, &removed[..removed_len], &added[..added_len]);
            let old_value = piece_kind_value(moved.piece_kind());
            let new_value = after_piece
                .map(|p| piece_kind_value(p.piece_kind()))
                .unwrap_or(old_value);
            let sign = if moved.color() == acc.perspective {
                1.0
            } else {
                -1.0
            };
            acc.material += sign * (new_value - old_value);
            if let Some(captured) = pending.captured {
                acc.material += if moved.color() == acc.perspective {
                    1.0
                } else {
                    -1.0
                } * (piece_kind_value(captured.piece_kind())
                    + piece_kind_value(unpromoted_kind(captured.piece_kind())));
            }
        } else if let Move::Drop { piece, to } = pending.mv {
            let mut removed = [0usize; 1];
            let removed_len = self
                .piece_row(acc, piece, None, pending.hand_index)
                .map(|row| {
                    removed[0] = row;
                    1
                })
                .unwrap_or(0);
            let mut added = [0usize; 1];
            let added_len = after
                .piece_at(to)
                .and_then(|after_piece| self.piece_row(acc, after_piece, Some(to), 0))
                .map(|row| {
                    added[0] = row;
                    1
                })
                .unwrap_or(0);
            self.apply_feature_rows(acc, &removed[..removed_len], &added[..added_len]);
        }
    }

    fn piece_row(
        &self,
        acc: &HalfKpAccumulator,
        piece: Piece,
        square: Option<Square>,
        hand_index: usize,
    ) -> Option<usize> {
        if piece.piece_kind() == PieceKind::King
            && piece.color() == acc.perspective
            && square.is_some()
        {
            return None;
        }
        halfkp_piece_state(piece, square, hand_index, acc.perspective, acc.mirror)
            .map(|state| acc.king_bucket * HALFKP_PIECE_STATES + state)
    }

    fn apply_feature_rows(&self, acc: &mut HalfKpAccumulator, removed: &[usize], added: &[usize]) {
        let mut hidden = acc.hidden;
        #[cfg(target_arch = "x86_64")]
        if self.use_avx2 {
            unsafe {
                apply_feature_rows_avx2(&mut hidden, &self.feature_emb, removed, added);
            }
            acc.hidden = hidden;
            return;
        }
        for &feature in removed {
            let row = &self.feature_emb[feature].0;
            for h in 0..HALFKP_HIDDEN {
                hidden[h] -= row[h];
            }
        }
        for &feature in added {
            let row = &self.feature_emb[feature].0;
            for h in 0..HALFKP_HIDDEN {
                hidden[h] += row[h];
            }
        }
        acc.hidden = hidden;
    }

    pub fn undo_search_context(&self, ctx: &mut HalfKpSearchContext) {
        if ctx.ply > 0 {
            ctx.ply -= 1;
        }
    }

    pub fn evaluate_search_context(
        &self,
        pos: &shogi_lib::Position,
        ctx: &HalfKpSearchContext,
    ) -> f32 {
        let frame = &ctx.frames[ctx.ply];
        self.evaluate_accumulators(pos, &frame.black, &frame.white)
    }

    /// Apply the feature-row delta between two adjacent positions.
    ///
    /// This is the correctness-first bridge to search integration: callers
    /// keep the previous accumulator and pass the pre/post move positions.
    /// Only rows whose occupancy changed are touched; no network output is
    /// recomputed here.
    pub fn update_accumulator_from_positions(
        &self,
        accumulator: &mut HalfKpAccumulator,
        before: &shogi_lib::Position,
        after: &shogi_lib::Position,
    ) -> bool {
        let Some(old) = extract_halfkp_features_for(before, accumulator.perspective) else {
            return false;
        };
        let Some(new) = extract_halfkp_features_for(after, accumulator.perspective) else {
            return false;
        };
        if accumulator.king_bucket != old.king_bucket || old.king_bucket != new.king_bucket {
            return false;
        }
        let mut i = 0;
        let mut j = 0;
        while i < old.features.len() || j < new.features.len() {
            match (old.features.get(i), new.features.get(j)) {
                (Some(&a), Some(&b)) if a == b => {
                    i += 1;
                    j += 1;
                }
                (Some(&a), Some(&b)) if a < b => {
                    self.add_feature_row(accumulator, a, -1.0);
                    i += 1;
                }
                (Some(_), Some(&b)) => {
                    self.add_feature_row(accumulator, b, 1.0);
                    j += 1;
                }
                (Some(&a), None) => {
                    self.add_feature_row(accumulator, a, -1.0);
                    i += 1;
                }
                (None, Some(&b)) => {
                    self.add_feature_row(accumulator, b, 1.0);
                    j += 1;
                }
                (None, None) => break,
            }
        }
        accumulator.material += new.material - old.material;
        true
    }

    fn add_feature_row(&self, accumulator: &mut HalfKpAccumulator, feature: usize, sign: f32) {
        let row = &self.feature_emb[feature].0;
        for h in 0..HALFKP_HIDDEN {
            accumulator.hidden[h] += sign * row[h];
        }
    }

    pub fn evaluate_accumulators(
        &self,
        pos: &shogi_lib::Position,
        black: &HalfKpAccumulator,
        white: &HalfKpAccumulator,
    ) -> f32 {
        let (stm, nstm) = if pos.side_to_move() == Color::Black {
            (black, white)
        } else {
            (white, black)
        };
        let mut score = self.out_b;
        for h in 0..HALFKP_HIDDEN {
            score += stm.hidden[h].clamp(0.0, 1.0) * self.out_w[h];
            score += nstm.hidden[h].clamp(0.0, 1.0) * self.out_w[HALFKP_HIDDEN + h];
        }
        score += (stm.material / 1000.0) * self.out_w[HALFKP_HIDDEN * 2];
        score * self.target_scale
    }

    pub fn predict_from_position(&self, pos: &shogi_lib::Position) -> f32 {
        let Some(black) = self.accumulator_for_position(pos, Color::Black) else {
            return 0.0;
        };
        let Some(white) = self.accumulator_for_position(pos, Color::White) else {
            return 0.0;
        };
        self.evaluate_accumulators(pos, &black, &white)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn decode_hex_fixture(contents: &str) -> Vec<u8> {
        contents
            .split_whitespace()
            .map(|byte| u8::from_str_radix(byte, 16).expect("valid fixture byte"))
            .collect()
    }

    fn write_test_bytes(label: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "shogi-ai-{label}-{}-{}.bin",
            std::process::id(),
            HALFKP_HIDDEN
        ));
        std::fs::write(&path, bytes).expect("write test fixture");
        path
    }

    fn assert_halfkp_load_error(label: &str, bytes: &[u8], expected: Option<&str>) {
        let path = write_test_bytes(label, bytes);
        let error = HalfKpModel::load(&path).expect_err("damaged HalfKP file must be rejected");
        std::fs::remove_file(path).ok();
        if let Some(expected) = expected {
            assert!(
                error.to_string().contains(expected),
                "expected error containing {expected:?}, got {error:#}"
            );
        }
    }

    fn test_feature_rows<F>(mut value: F) -> Box<[HalfKpFeatureRow]>
    where
        F: FnMut(usize) -> f32,
    {
        (0..HALFKP_INPUTS)
            .map(|row_index| {
                HalfKpFeatureRow(std::array::from_fn(|h| {
                    value(row_index * HALFKP_HIDDEN + h)
                }))
            })
            .collect()
    }

    #[test]
    fn nnue_features_are_in_declared_ranges() {
        let position = shogi_lib::Position::default();
        let features = extract_nnue_features(&position).expect("initial position has both kings");

        assert!(features.king_bucket < NNUE_NUM_KING_BUCKETS);
        assert!(!features.features.is_empty());
        assert!(features
            .features
            .iter()
            .all(|&feature| feature < NNUE_NUM_FEATURES));
        assert!(features.features.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn tiny_nnue_model_loads_binary_and_evaluates() {
        let path = std::env::temp_dir().join(format!(
            "tiny_nnue_model_loads_binary_and_evaluates_{}.bin",
            std::process::id()
        ));
        let mut file = File::create(&path).expect("create tiny nnue test file");
        file.write_all(b"TNNUE001").unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&(NNUE_NUM_FEATURES as u32).to_le_bytes())
            .unwrap();
        file.write_all(&(NNUE_NUM_KING_BUCKETS as u32).to_le_bytes())
            .unwrap();
        file.write_all(&1000.0f32.to_le_bytes()).unwrap();

        for _ in 0..NNUE_NUM_FEATURES {
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
        }
        for _ in 0..NNUE_NUM_KING_BUCKETS {
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
        }
        file.write_all(&0.0f32.to_le_bytes()).unwrap();
        file.write_all(&0.5f32.to_le_bytes()).unwrap();
        file.write_all(&2.0f32.to_le_bytes()).unwrap();
        file.write_all(&0.25f32.to_le_bytes()).unwrap();
        drop(file);

        let model = TinyNnueModel::load(&path).expect("load tiny nnue model");
        let score = model.predict_from_position(&shogi_lib::Position::default());
        std::fs::remove_file(&path).ok();

        assert_eq!(model.hidden, 1);
        assert_eq!(score, 1250.0);
    }

    #[test]
    fn halfkp_features_are_compact_and_sorted() {
        let position = shogi_lib::Position::default();
        for perspective in [Color::Black, Color::White] {
            let features = extract_halfkp_features_for(&position, perspective)
                .expect("initial position has both kings");
            assert!(features.king_bucket < HALFKP_KING_BUCKETS);
            assert!(!features.features.is_empty());
            assert!(features
                .features
                .iter()
                .all(|&feature| feature < HALFKP_INPUTS));
            assert!(features.features.windows(2).all(|pair| pair[0] < pair[1]));
        }
    }

    #[test]
    fn halfkp_v1_header_matches_golden_fixture_and_rejects_damage() {
        let fixture = if cfg!(feature = "halfkp64") {
            include_str!("../tests/fixtures/halfkp/header_v1_halfkp64.hex")
        } else {
            include_str!("../tests/fixtures/halfkp/header_v1_halfkp32.hex")
        };
        let golden = decode_hex_fixture(fixture);
        let mut expected = Vec::with_capacity(32);
        expected.extend_from_slice(HalfKpModel::MAGIC);
        expected.extend_from_slice(&1u32.to_le_bytes());
        expected.extend_from_slice(&(HALFKP_HIDDEN as u32).to_le_bytes());
        expected.extend_from_slice(&(HALFKP_INPUTS as u32).to_le_bytes());
        expected.extend_from_slice(&(HALFKP_KING_BUCKETS as u32).to_le_bytes());
        expected.extend_from_slice(&(HALFKP_PIECE_STATES as u32).to_le_bytes());
        expected.extend_from_slice(&1000.0f32.to_le_bytes());
        assert_eq!(32, golden.len());
        assert_eq!(expected, golden);

        assert_halfkp_load_error("halfkp-header-only", &golden, None);
        for &length in &[0, 7, 8, 11, 12, 27, 31] {
            assert_halfkp_load_error(
                &format!("halfkp-truncated-header-{length}"),
                &golden[..length],
                None,
            );
        }

        let mut invalid_magic = golden.clone();
        invalid_magic[0] ^= 0xff;
        assert_halfkp_load_error(
            "halfkp-invalid-magic",
            &invalid_magic,
            Some("invalid HalfKP magic"),
        );

        let mut invalid_version = golden.clone();
        invalid_version[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert_halfkp_load_error(
            "halfkp-invalid-version",
            &invalid_version,
            Some("invalid HalfKP header"),
        );

        let mut invalid_hidden = golden.clone();
        invalid_hidden[12..16].copy_from_slice(&0u32.to_le_bytes());
        assert_halfkp_load_error(
            "halfkp-invalid-hidden",
            &invalid_hidden,
            Some("invalid HalfKP header"),
        );

        let mut invalid_inputs = golden.clone();
        invalid_inputs[16..20].copy_from_slice(&0u32.to_le_bytes());
        assert_halfkp_load_error(
            "halfkp-invalid-inputs",
            &invalid_inputs,
            Some("invalid HalfKP header"),
        );

        let mut invalid_scale = golden;
        invalid_scale[28..32].copy_from_slice(&f32::NAN.to_le_bytes());
        assert_halfkp_load_error(
            "halfkp-invalid-scale",
            &invalid_scale,
            Some("invalid HalfKP header"),
        );
    }

    #[test]
    fn halfkp_model_loads_binary_and_evaluates() {
        let path = std::env::temp_dir().join(format!(
            "halfkp_model_loads_binary_and_evaluates_{}.bin",
            std::process::id()
        ));
        let mut file = File::create(&path).expect("create HalfKP test file");
        file.write_all(HalfKpModel::MAGIC).unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&(HALFKP_HIDDEN as u32).to_le_bytes())
            .unwrap();
        file.write_all(&(HALFKP_INPUTS as u32).to_le_bytes())
            .unwrap();
        file.write_all(&(HALFKP_KING_BUCKETS as u32).to_le_bytes())
            .unwrap();
        file.write_all(&(HALFKP_PIECE_STATES as u32).to_le_bytes())
            .unwrap();
        file.write_all(&1000.0f32.to_le_bytes()).unwrap();
        for _ in 0..HALFKP_INPUTS * HALFKP_HIDDEN {
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
        }
        for _ in 0..HALFKP_HIDDEN {
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
        }
        for _ in 0..(HALFKP_HIDDEN * 2 + 1) {
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
        }
        file.write_all(&0.0f32.to_le_bytes()).unwrap();
        drop(file);

        let model = HalfKpModel::load(&path).expect("load HalfKP model");
        let position = shogi_lib::Position::default();
        let score = model.predict_from_position(&position);
        let black = model
            .accumulator_for_position(&position, Color::Black)
            .expect("black accumulator");
        let white = model
            .accumulator_for_position(&position, Color::White)
            .expect("white accumulator");
        let accumulated_score = model.evaluate_accumulators(&position, &black, &white);

        let mut trailing_file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("reopen HalfKP model");
        trailing_file
            .write_all(&[0xff])
            .expect("append trailing byte");
        drop(trailing_file);
        let trailing_error =
            HalfKpModel::load(&path).expect_err("trailing model byte must be rejected");
        assert!(
            trailing_error
                .to_string()
                .contains("trailing bytes in HalfKP file"),
            "unexpected error: {trailing_error:#}"
        );

        std::fs::remove_file(&path).ok();
        assert_eq!(model.feature_emb.len(), HALFKP_INPUTS);
        assert_eq!(score, 0.0);
        assert_eq!(score, accumulated_score);
    }

    #[test]
    fn halfkp_accumulator_feature_delta_matches_refresh() {
        let position = shogi_lib::Position::default();
        let mv = position
            .legal_moves()
            .into_iter()
            .find(|mv| !matches!(mv, shogi_core::Move::Normal { from, .. } if from.index() == 0))
            .expect("initial position has a legal move");
        let mut after = position.clone();
        after.do_move(mv);

        let model = HalfKpModel {
            target_scale: 1000.0,
            feature_emb: test_feature_rows(|i| (i as f32 % 13.0) * 0.001),
            hidden_b: [0.25; HALFKP_HIDDEN],
            out_w: [0.1; HALFKP_HIDDEN * 2 + 1],
            out_b: 0.0,
            use_avx2: false,
        };
        for perspective in [Color::Black, Color::White] {
            let mut delta = model
                .accumulator_for_position(&position, perspective)
                .unwrap();
            let refreshed = model.accumulator_for_position(&after, perspective).unwrap();
            if delta.king_bucket == refreshed.king_bucket {
                assert!(model.update_accumulator_from_positions(&mut delta, &position, &after));
                for (actual, expected) in delta.hidden.iter().zip(refreshed.hidden.iter()) {
                    assert!((actual - expected).abs() < 1e-5);
                }
                assert!((delta.material - refreshed.material).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn halfkp_search_context_move_matches_refresh() {
        let position = shogi_lib::Position::default();
        let mv = position.legal_moves()[0];
        let mut after = position.clone();
        after.do_move(mv);
        let model = HalfKpModel {
            target_scale: 1000.0,
            feature_emb: test_feature_rows(|i| ((i % 97) as f32) * 0.0001),
            hidden_b: std::array::from_fn(|i| i as f32 * 0.001),
            out_w: std::array::from_fn(|i| i as f32 * 0.01),
            out_b: 0.0,
            use_avx2: false,
        };
        let mut ctx = model.begin_search_context(&position).unwrap();
        model.prepare_search_context(&mut ctx, &position, mv);
        model.commit_search_context_move(&mut ctx, &after);
        let black = model
            .accumulator_for_position(&after, Color::Black)
            .unwrap();
        let white = model
            .accumulator_for_position(&after, Color::White)
            .unwrap();
        let frame = &ctx.frames[ctx.ply];
        for (a, b) in frame.black.hidden.iter().zip(black.hidden.iter()) {
            assert!((a - b).abs() < 1e-5, "black {} {}", a, b);
        }
        for (a, b) in frame.white.hidden.iter().zip(white.hidden.iter()) {
            assert!((a - b).abs() < 1e-5, "white {} {}", a, b);
        }
        model.undo_search_context(&mut ctx);
        let root_black = model
            .accumulator_for_position(&position, Color::Black)
            .unwrap();
        let frame = &ctx.frames[ctx.ply];
        assert!(frame
            .black
            .hidden
            .iter()
            .zip(root_black.hidden.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5));
    }

    #[test]
    fn halfkp_avx2_kernel_matches_portable_kernel() {
        if !halfkp_avx2_available() {
            return;
        }
        let position = shogi_lib::Position::default();
        let mv = position.legal_moves()[0];
        let mut after = position.clone();
        after.do_move(mv);
        let portable = HalfKpModel {
            target_scale: 1000.0,
            feature_emb: test_feature_rows(|i| ((i % 53) as f32 - 20.0) * 0.0003),
            hidden_b: std::array::from_fn(|i| i as f32 * 0.002),
            out_w: std::array::from_fn(|i| i as f32 * 0.01),
            out_b: -0.25,
            use_avx2: false,
        };
        let mut avx2 = portable.clone();
        avx2.use_avx2 = true;
        for perspective in [Color::Black, Color::White] {
            let a = portable
                .accumulator_for_position(&after, perspective)
                .unwrap();
            let b = avx2.accumulator_for_position(&after, perspective).unwrap();
            assert!(a
                .hidden
                .iter()
                .zip(b.hidden.iter())
                .all(|(x, y)| (x - y).abs() < 1e-5));
        }
        let mut portable_ctx = portable.begin_search_context(&position).unwrap();
        let mut avx2_ctx = avx2.begin_search_context(&position).unwrap();
        portable.prepare_search_context(&mut portable_ctx, &position, mv);
        avx2.prepare_search_context(&mut avx2_ctx, &position, mv);
        portable.commit_search_context_move(&mut portable_ctx, &after);
        avx2.commit_search_context_move(&mut avx2_ctx, &after);
        let portable_frame = &portable_ctx.frames[portable_ctx.ply];
        let avx2_frame = &avx2_ctx.frames[avx2_ctx.ply];
        assert!(portable_frame
            .black
            .hidden
            .iter()
            .zip(avx2_frame.black.hidden.iter())
            .all(|(x, y)| (x - y).abs() < 1e-5));
        assert!(portable_frame
            .white
            .hidden
            .iter()
            .zip(avx2_frame.white.hidden.iter())
            .all(|(x, y)| (x - y).abs() < 1e-5));
        assert!(
            (portable.predict_from_position(&after) - avx2.predict_from_position(&after)).abs()
                < 1e-4
        );
    }
}
