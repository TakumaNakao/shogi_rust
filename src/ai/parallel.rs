use super::*;

struct StopOnDrop(Arc<AtomicBool>);

impl Drop for StopOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

impl<E, const HISTORY_CAPACITY: usize> ShogiAI<E, HISTORY_CAPACITY>
where
    E: Evaluator + Clone + Send + Sync + 'static,
{
    pub fn search_parallel(
        &mut self,
        position: &mut Position,
        limits: SearchLimits,
        threads: usize,
    ) -> SearchOutcome {
        let best_move = self.find_best_move_parallel_internal(
            position,
            limits.max_depth,
            limits.time_limit_ms(),
            threads,
        );
        self.search_outcome(best_move)
    }

    pub fn find_best_move_parallel(
        &mut self,
        position: &mut Position,
        max_depth: u8,
        time_limit_ms: Option<u64>,
        threads: usize,
    ) -> Option<Move> {
        self.search_parallel(
            position,
            SearchLimits::from_millis(max_depth, time_limit_ms),
            threads,
        )
        .best_move()
    }

    fn find_best_move_parallel_internal(
        &mut self,
        position: &mut Position,
        max_depth: u8,
        time_limit_ms: Option<u64>,
        threads: usize,
    ) -> Option<Move> {
        let threads = resolve_search_threads(threads);
        let root_position = position.clone();
        let fallback_move = position.legal_moves().first().copied();
        self.last_search_failed = false;

        if threads == 1 {
            if matches!(&self.transposition_table, TranspositionTable::Shared(_)) {
                self.transposition_table = TranspositionTable::Local(HashMap::new());
            }
            return match catch_unwind(AssertUnwindSafe(|| {
                self.find_best_move(position, max_depth, time_limit_ms)
            })) {
                Ok(best_move) => best_move,
                Err(_) => {
                    *position = root_position;
                    self.recover_from_search_failure(position);
                    fallback_move
                }
            };
        }

        let shared_tt = match &self.transposition_table {
            TranspositionTable::Shared(table) => table.clone(),
            TranspositionTable::Local(_) => {
                let table = Arc::new(SharedTranspositionTable::new());
                self.transposition_table = TranspositionTable::Shared(table.clone());
                table
            }
        };
        let owns_stop_signal = self.stop_signal.is_none();
        let stop_signal = self
            .stop_signal
            .clone()
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        if owns_stop_signal {
            self.stop_signal = Some(stop_signal.clone());
        }

        let evaluator = self.evaluator.clone();
        let generation = self.search_generation;
        let observer = self.observer.clone();
        let game_history = self.sennichite_detector.clone();

        let parallel_result = catch_unwind(AssertUnwindSafe(|| {
            thread::scope(|scope| {
                let _stop_on_drop = StopOnDrop(stop_signal.clone());
                let mut handles = Vec::with_capacity(threads - 1);
                let mut worker_failed = false;
                for worker_id in 1..threads {
                    let evaluator = evaluator.clone();
                    let table = shared_tt.clone();
                    let worker_stop_signal = stop_signal.clone();
                    let worker_history = game_history.clone();
                    let mut worker_position = root_position.clone();
                    match thread::Builder::new()
                        .stack_size(SEARCH_THREAD_STACK_BYTES)
                        .spawn_scoped(scope, move || {
                            let mut worker =
                                ShogiAI::new_with_shared_tt(evaluator, table, generation);
                            worker.sennichite_detector = worker_history;
                            worker.set_emit_info(false);
                            worker.set_stop_signal(Some(worker_stop_signal.clone()));
                            let worker_result = catch_unwind(AssertUnwindSafe(|| {
                                worker.find_best_move_with_root_offset(
                                    &mut worker_position,
                                    max_depth,
                                    time_limit_ms,
                                    worker_id,
                                    false,
                                )
                            }));
                            match worker_result {
                                Ok(best_move) => Ok((best_move, worker)),
                                Err(_) => {
                                    worker_stop_signal.store(true, Ordering::Relaxed);
                                    Err(())
                                }
                            }
                        }) {
                        Ok(handle) => handles.push(handle),
                        Err(_) => {
                            worker_failed = true;
                            stop_signal.store(true, Ordering::Relaxed);
                            break;
                        }
                    }
                }

                self.set_search_observer(observer);
                let main_result = catch_unwind(AssertUnwindSafe(|| {
                    self.find_best_move_with_root_offset(
                        position,
                        max_depth,
                        time_limit_ms,
                        0,
                        true,
                    )
                }));
                stop_signal.store(true, Ordering::Relaxed);

                for handle in handles {
                    match handle.join() {
                        Ok(Ok((_worker_move, worker))) => self.absorb_statistics(&worker),
                        Ok(Err(())) | Err(_) => worker_failed = true,
                    }
                }

                if worker_failed {
                    self.transposition_table.clear();
                    self.last_search_failed = true;
                }

                match main_result {
                    Ok(main_move) => main_move,
                    Err(_) => {
                        *position = root_position.clone();
                        self.recover_from_search_failure(position);
                        fallback_move
                    }
                }
            })
        }));

        if owns_stop_signal {
            self.stop_signal = None;
        }

        match parallel_result {
            Ok(best_move) => best_move,
            Err(_) => {
                *position = root_position;
                self.recover_from_search_failure(position);
                fallback_move
            }
        }
    }
}

pub fn resolve_search_threads(requested: usize) -> usize {
    if requested == 0 {
        thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(256)
    } else {
        requested.clamp(1, 256)
    }
}
