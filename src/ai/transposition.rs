use shogi_core::Move;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub(super) const TRANSPOSITION_TABLE_MAX_ENTRIES: usize = 1_000_000;
const SHARED_TT_SHARDS: usize = 4096;
const SHARED_TT_ENTRIES_PER_SHARD: usize = TRANSPOSITION_TABLE_MAX_ENTRIES / SHARED_TT_SHARDS + 1;

/// トランスポジションテーブルに格納する評価値の種類
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum NodeType {
    Exact,
    LowerBound,
    UpperBound,
}

/// トランスポジションテーブルのエントリ
#[derive(Clone, Copy, Debug)]
pub(super) struct TranspositionEntry {
    pub(super) score: f32,
    pub(super) depth: u8,
    pub(super) node_type: NodeType,
    pub(super) best_move: Option<Move>,
    pub(super) generation: u32,
}

pub(super) struct SharedTranspositionTable {
    shards: Box<[RwLock<HashMap<u64, TranspositionEntry>>]>,
}

impl SharedTranspositionTable {
    pub(super) fn new() -> Self {
        let shards = (0..SHARED_TT_SHARDS)
            .map(|_| RwLock::new(HashMap::with_capacity(SHARED_TT_ENTRIES_PER_SHARD)))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self { shards }
    }

    #[inline]
    fn shard_index(hash: u64) -> usize {
        (hash as usize) & (SHARED_TT_SHARDS - 1)
    }

    pub(super) fn get(&self, hash: u64) -> Option<TranspositionEntry> {
        let shard = self.shards[Self::shard_index(hash)]
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        shard.get(&hash).copied()
    }

    pub(super) fn insert(&self, hash: u64, entry: TranspositionEntry) {
        let mut shard = self.shards[Self::shard_index(hash)]
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !shard.contains_key(&hash) && shard.len() >= SHARED_TT_ENTRIES_PER_SHARD {
            let replacement = shard
                .iter()
                .take(8)
                .min_by_key(|(_, candidate)| {
                    (candidate.generation == entry.generation, candidate.depth)
                })
                .map(|(&key, _)| key);
            if let Some(key) = replacement {
                shard.remove(&key);
            }
        }
        shard.insert(hash, entry);
    }

    pub(super) fn clear(&self) {
        for shard in &self.shards {
            shard
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clear();
        }
    }
}

pub(super) enum TranspositionTable {
    Local(HashMap<u64, TranspositionEntry>),
    Shared(Arc<SharedTranspositionTable>),
}

impl TranspositionTable {
    pub(super) fn get(&self, hash: u64) -> Option<TranspositionEntry> {
        match self {
            Self::Local(table) => table.get(&hash).copied(),
            Self::Shared(table) => table.get(hash),
        }
    }

    pub(super) fn insert(&mut self, hash: u64, entry: TranspositionEntry) {
        match self {
            Self::Local(table) => {
                table.insert(hash, entry);
            }
            Self::Shared(table) => table.insert(hash, entry),
        }
    }

    pub(super) fn clear(&mut self) {
        match self {
            Self::Local(table) => table.clear(),
            Self::Shared(table) => table.clear(),
        }
    }

    pub(super) fn local_len(&self) -> Option<usize> {
        match self {
            Self::Local(table) => Some(table.len()),
            Self::Shared(_) => None,
        }
    }
}
