use super::HalfKpSearchContext;
use shogi_core::Move;
use shogi_lib::Position;
use std::sync::Arc;

/// Typed incremental state owned by a search. Add a variant when another
/// evaluator gains an incremental representation.
pub enum EvaluationContext {
    HalfKp(HalfKpSearchContext),
}

pub trait Evaluator {
    fn evaluate(&self, position: &Position) -> f32;

    fn begin_context(&self, _position: &Position) -> Option<EvaluationContext> {
        None
    }

    fn evaluate_context(&self, _position: &Position, _context: &EvaluationContext) -> Option<f32> {
        None
    }

    fn prepare_context_move(
        &self,
        _context: &mut EvaluationContext,
        _position: &Position,
        _mv: Move,
    ) {
    }

    fn commit_context_move(&self, _context: &mut EvaluationContext, _position: &Position) {}

    fn undo_context_move(&self, _context: &mut EvaluationContext) {}
}

impl<T: Evaluator + ?Sized> Evaluator for Arc<T> {
    fn evaluate(&self, position: &Position) -> f32 {
        (**self).evaluate(position)
    }

    fn begin_context(&self, position: &Position) -> Option<EvaluationContext> {
        (**self).begin_context(position)
    }

    fn evaluate_context(&self, position: &Position, context: &EvaluationContext) -> Option<f32> {
        (**self).evaluate_context(position, context)
    }

    fn prepare_context_move(&self, context: &mut EvaluationContext, position: &Position, mv: Move) {
        (**self).prepare_context_move(context, position, mv);
    }

    fn commit_context_move(&self, context: &mut EvaluationContext, position: &Position) {
        (**self).commit_context_move(context, position);
    }

    fn undo_context_move(&self, context: &mut EvaluationContext) {
        (**self).undo_context_move(context);
    }
}
