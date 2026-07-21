use shogi_core::Move;
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchLimits {
    pub max_depth: u8,
    pub time_limit: Option<Duration>,
}

impl SearchLimits {
    pub const fn new(max_depth: u8, time_limit: Option<Duration>) -> Self {
        Self {
            max_depth,
            time_limit,
        }
    }

    pub const fn from_millis(max_depth: u8, time_limit_ms: Option<u64>) -> Self {
        Self::new(
            max_depth,
            match time_limit_ms {
                Some(milliseconds) => Some(Duration::from_millis(milliseconds)),
                None => None,
            },
        )
    }

    pub fn time_limit_ms(self) -> Option<u64> {
        self.time_limit
            .map(|limit| limit.as_millis().min(u64::MAX as u128) as u64)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SearchStats {
    pub nodes: u64,
    pub quiescence_nodes: u64,
    pub quiescence_moves_considered: u64,
    pub quiescence_moves_generated: u64,
    pub quiescence_moves_discarded: u64,
    pub quiescence_moves_searched: u64,
    pub quiescence_see_skips: u64,
    pub quiescence_terminal_mates: u64,
    pub check_evasion_extensions: u64,
    pub aspiration_fail_lows: u64,
    pub aspiration_fail_highs: u64,
    pub aspiration_researches: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchInfo {
    pub depth: u8,
    pub root_score: f32,
    pub elapsed: Duration,
    pub stats: SearchStats,
    pub pv: Vec<Move>,
}

pub trait SearchObserver: Send + Sync {
    fn on_info(&self, info: &SearchInfo);
}

impl<F> SearchObserver for F
where
    F: Fn(&SearchInfo) + Send + Sync,
{
    fn on_info(&self, info: &SearchInfo) {
        self(info);
    }
}

pub type SharedSearchObserver = Arc<dyn SearchObserver>;

#[derive(Debug, Clone, PartialEq)]
pub struct RootResult {
    pub best_move: Move,
    pub score: Option<f32>,
    pub completed_depth: u8,
    pub pv: Vec<Move>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchOutcome {
    pub root: Option<RootResult>,
    pub stats: SearchStats,
    pub failed: bool,
}

impl SearchOutcome {
    pub fn best_move(&self) -> Option<Move> {
        self.root.as_ref().map(|root| root.best_move)
    }
}
