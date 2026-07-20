use shogi_core::Move;
use shogi_lib::Position;
use std::any::Any;
use std::sync::Arc;

pub trait Evaluator {
    fn evaluate(&self, position: &Position) -> f32;

    fn begin_context(&self, _position: &Position) -> Option<Box<dyn Any + Send>> {
        None
    }

    fn evaluate_context(&self, _position: &Position, _context: &(dyn Any + Send)) -> Option<f32> {
        None
    }

    fn prepare_context_move(
        &self,
        _context: &mut (dyn Any + Send),
        _position: &Position,
        _mv: Move,
    ) {
    }

    fn commit_context_move(&self, _context: &mut (dyn Any + Send), _position: &Position) {}

    fn undo_context_move(&self, _context: &mut (dyn Any + Send)) {}
}

impl<T: Evaluator + ?Sized> Evaluator for Arc<T> {
    fn evaluate(&self, position: &Position) -> f32 {
        (**self).evaluate(position)
    }

    fn begin_context(&self, position: &Position) -> Option<Box<dyn Any + Send>> {
        (**self).begin_context(position)
    }

    fn evaluate_context(&self, position: &Position, context: &(dyn Any + Send)) -> Option<f32> {
        (**self).evaluate_context(position, context)
    }

    fn prepare_context_move(&self, context: &mut (dyn Any + Send), position: &Position, mv: Move) {
        (**self).prepare_context_move(context, position, mv);
    }

    fn commit_context_move(&self, context: &mut (dyn Any + Send), position: &Position) {
        (**self).commit_context_move(context, position);
    }

    fn undo_context_move(&self, context: &mut (dyn Any + Send)) {
        (**self).undo_context_move(context);
    }
}
