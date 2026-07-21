use super::{EvaluationContext, Evaluator, HalfKpModel, SparseModel, TinyNnueModel};
use anyhow::{anyhow, Result};
use shogi_core::Move;
use shogi_lib::Position;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub struct SparseModelEvaluator {
    pub model: SparseModel,
}

impl SparseModelEvaluator {
    pub fn new(weight_path: &Path, overwrite_value: f32) -> Result<Self> {
        let mut model = SparseModel::new(0.0, 0.0);
        model.load(weight_path)?;
        model.zero_weight_overwrite(overwrite_value);
        Ok(Self { model })
    }
}

impl Evaluator for SparseModelEvaluator {
    fn evaluate(&self, position: &Position) -> f32 {
        self.model.predict_from_position(position)
    }
}

pub struct HybridNnueEvaluator {
    pub sparse: SparseModel,
    pub residual: TinyNnueModel,
    pub residual_scale: f32,
}

impl HybridNnueEvaluator {
    pub fn new(sparse_path: &Path, residual_path: &Path, residual_scale: f32) -> Result<Self> {
        if !residual_scale.is_finite() {
            return Err(anyhow!("residual scale must be finite"));
        }
        let mut sparse = SparseModel::new(0.0, 0.0);
        sparse.load(sparse_path)?;
        let residual = TinyNnueModel::load(residual_path)?;
        Ok(Self {
            sparse,
            residual,
            residual_scale,
        })
    }

    pub fn predict_from_position(&self, position: &Position) -> f32 {
        self.sparse.predict_from_position(position)
            + self.residual_scale * self.residual.predict_from_position(position)
    }
}

impl Evaluator for HybridNnueEvaluator {
    fn evaluate(&self, position: &Position) -> f32 {
        self.predict_from_position(position)
    }
}

pub enum EngineEvaluator {
    Sparse(SparseModelEvaluator),
    TinyNnue(TinyNnueModel),
    HalfKp(HalfKpModel),
    HybridNnue(HybridNnueEvaluator),
}

impl EngineEvaluator {
    pub fn new(path: &Path, material_override: f32) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 8];
        let read = file.read(&mut magic)?;
        drop(file);

        if read == magic.len() && &magic == b"TNNUE001" {
            Ok(Self::TinyNnue(TinyNnueModel::load(path)?))
        } else if read == magic.len() && &magic == HalfKpModel::MAGIC {
            Ok(Self::HalfKp(HalfKpModel::load(path)?))
        } else {
            Ok(Self::Sparse(SparseModelEvaluator::new(
                path,
                material_override,
            )?))
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Sparse(_) => "sparse",
            Self::TinyNnue(_) => "tiny-nnue",
            Self::HalfKp(_) => "halfkp",
            Self::HybridNnue(_) => "hybrid-nnue",
        }
    }
}

impl Evaluator for EngineEvaluator {
    fn evaluate(&self, position: &Position) -> f32 {
        match self {
            Self::Sparse(evaluator) => evaluator.evaluate(position),
            Self::TinyNnue(model) => model.predict_from_position(position),
            Self::HalfKp(model) => model.predict_from_position(position),
            Self::HybridNnue(evaluator) => evaluator.evaluate(position),
        }
    }

    fn begin_context(&self, position: &Position) -> Option<EvaluationContext> {
        match self {
            Self::HalfKp(model) => model
                .begin_search_context(position)
                .map(EvaluationContext::HalfKp),
            _ => None,
        }
    }

    fn evaluate_context(&self, position: &Position, context: &EvaluationContext) -> Option<f32> {
        match (self, context) {
            (Self::HalfKp(model), EvaluationContext::HalfKp(ctx)) => {
                Some(model.evaluate_search_context(position, ctx))
            }
            _ => None,
        }
    }

    fn commit_context_move(&self, context: &mut EvaluationContext, position: &Position) {
        if let (Self::HalfKp(model), EvaluationContext::HalfKp(ctx)) = (self, context) {
            model.commit_search_context_move(ctx, position);
        }
    }

    fn prepare_context_move(&self, context: &mut EvaluationContext, position: &Position, mv: Move) {
        if let (Self::HalfKp(model), EvaluationContext::HalfKp(ctx)) = (self, context) {
            model.prepare_search_context(ctx, position, mv);
        }
    }

    fn undo_context_move(&self, context: &mut EvaluationContext) {
        if let (Self::HalfKp(model), EvaluationContext::HalfKp(ctx)) = (self, context) {
            model.undo_search_context(ctx);
        }
    }
}
