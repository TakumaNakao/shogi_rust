use super::codec::{
    feature_rows_from_flat, read_f32_array, read_f32_le, read_f32_vec, HalfKpFeatureRow,
    HalfKpHeader, HALFKP_HEADER_LEN, HALFKP_MAGIC,
};
use super::constants::{
    piece_kind_value, unpromoted_kind, HALFKP_HIDDEN, HALFKP_INPUTS, HALFKP_KING_BUCKETS,
    HALFKP_PIECE_STATES,
};
use super::features::{
    extract_halfkp_features_fixed, extract_halfkp_features_for, halfkp_piece_state, HalfKpFeatures,
    HalfKpFixedFeatures,
};
use super::kernels::halfkp_avx2_available;
#[cfg(target_arch = "x86_64")]
use super::kernels::{accumulate_rows_avx2, apply_feature_rows_avx2};
use anyhow::{anyhow, Result};
use shogi_core::{Color, Move, Piece, PieceKind, Square};
use shogi_lib::Position;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct HalfKpModel {
    pub target_scale: f32,
    pub(super) feature_emb: Box<[HalfKpFeatureRow]>,
    pub(super) hidden_b: [f32; HALFKP_HIDDEN],
    pub(super) out_w: [f32; HALFKP_HIDDEN * 2 + 1],
    pub(super) out_b: f32,
    pub(super) use_avx2: bool,
}

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
    pub(super) frames: Vec<HalfKpFrame>,
    pub(super) ply: usize,
    pending: Option<HalfKpPendingMove>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct HalfKpFrame {
    pub(super) black: HalfKpAccumulator,
    pub(super) white: HalfKpAccumulator,
}

#[derive(Clone, Copy)]
struct HalfKpPendingMove {
    mv: Move,
    moved: Option<Piece>,
    captured: Option<Piece>,
    hand_index: usize,
}

impl HalfKpModel {
    pub const MAGIC: &'static [u8; 8] = HALFKP_MAGIC;

    pub fn load(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut header_bytes = [0u8; HALFKP_HEADER_LEN];
        file.read_exact(&mut header_bytes)?;
        let header = HalfKpHeader::decode(&header_bytes)?;
        let feature_emb =
            feature_rows_from_flat(read_f32_vec(&mut file, HALFKP_INPUTS * HALFKP_HIDDEN)?)?;
        let hidden_b = read_f32_array::<HALFKP_HIDDEN>(&mut file)?;
        let out_w = read_f32_array::<{ HALFKP_HIDDEN * 2 + 1 }>(&mut file)?;
        let out_b = read_f32_le(&mut file)?;
        let mut trailing = [0u8; 1];
        if file.read(&mut trailing)? != 0 {
            return Err(anyhow!("trailing bytes in HalfKP file"));
        }
        Ok(Self {
            target_scale: header.target_scale,
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
        pos: &Position,
        perspective: Color,
    ) -> Option<HalfKpAccumulator> {
        let features = extract_halfkp_features_fixed(pos, perspective)?;
        Some(self.accumulator_from_fixed(&features, perspective))
    }

    pub fn begin_search_context(&self, pos: &Position) -> Option<HalfKpSearchContext> {
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

    pub fn prepare_search_context(&self, ctx: &mut HalfKpSearchContext, pos: &Position, mv: Move) {
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

    pub fn commit_search_context_move(&self, ctx: &mut HalfKpSearchContext, pos: &Position) {
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
        after: &Position,
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
                .map(|piece| piece_kind_value(piece.piece_kind()))
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

    pub fn evaluate_search_context(&self, pos: &Position, ctx: &HalfKpSearchContext) -> f32 {
        let frame = &ctx.frames[ctx.ply];
        self.evaluate_accumulators(pos, &frame.black, &frame.white)
    }

    pub fn update_accumulator_from_positions(
        &self,
        accumulator: &mut HalfKpAccumulator,
        before: &Position,
        after: &Position,
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
        pos: &Position,
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

    pub fn predict_from_position(&self, pos: &Position) -> f32 {
        let Some(black) = self.accumulator_for_position(pos, Color::Black) else {
            return 0.0;
        };
        let Some(white) = self.accumulator_for_position(pos, Color::White) else {
            return 0.0;
        };
        self.evaluate_accumulators(pos, &black, &white)
    }
}
