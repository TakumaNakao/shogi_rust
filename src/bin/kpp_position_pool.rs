use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::{Evaluator, SparseModel};
use shogi_ai::utils::position_from_sfen_or_usi;
use shogi_lib::Position;
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::rc::Rc;

const HISTORY_CAPACITY: usize = 256;

#[derive(Parser, Debug)]
#[command(about = "Collect KPP-search and random-playout positions for HalfKP distillation")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long, required = true)]
    output: PathBuf,
    #[arg(long, default_value = "policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long, default_value_t = 0)]
    max_roots: usize,
    #[arg(long, default_value_t = 4)]
    random_branches: usize,
    #[arg(long, default_value_t = 80)]
    random_plies: usize,
    #[arg(long, default_value_t = 0)]
    search_roots: usize,
    #[arg(long, default_value_t = 6)]
    search_depth: u8,
    #[arg(long)]
    search_time_limit_ms: Option<u64>,
    #[arg(long, default_value_t = 20_000)]
    search_position_cap: usize,
    #[arg(long, default_value_t = 9601)]
    seed: u64,
    #[arg(long)]
    max_output: Option<usize>,
}

#[derive(Serialize)]
struct PoolRecord<'a> {
    schema: &'static str,
    source: &'a str,
    generation: &'static str,
    sfen: String,
    side_to_move: &'static str,
    winner: Option<&'static str>,
    result_known: bool,
    termination: &'static str,
}

struct UniqueWriter {
    writer: BufWriter<File>,
    seen: HashSet<u64>,
    max_output: Option<usize>,
    written: usize,
}

impl UniqueWriter {
    fn new(path: &PathBuf, max_output: Option<usize>) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(Self {
            writer: BufWriter::new(File::create(path)?),
            seen: HashSet::new(),
            max_output,
            written: 0,
        })
    }

    fn full(&self) -> bool {
        self.max_output.is_some_and(|limit| self.written >= limit)
    }

    fn write_position(
        &mut self,
        position: &Position,
        source: &str,
        generation: &'static str,
    ) -> Result<bool> {
        if self.full() {
            return Ok(false);
        }
        let sfen = position.to_sfen_owned();
        let mut hasher = DefaultHasher::new();
        sfen.hash(&mut hasher);
        if !self.seen.insert(hasher.finish()) {
            return Ok(false);
        }
        let side_to_move = if position.side_to_move() == shogi_core::Color::Black {
            "black"
        } else {
            "white"
        };
        let record = PoolRecord {
            schema: "kpp_position_pool_v1",
            source,
            generation,
            sfen,
            side_to_move,
            winner: None,
            result_known: false,
            termination: "generated",
        };
        serde_json::to_writer(&mut self.writer, &record)?;
        self.writer.write_all(b"\n")?;
        self.written += 1;
        Ok(true)
    }

    fn finish(mut self) -> Result<usize> {
        self.writer.flush()?;
        Ok(self.written)
    }
}

struct RecordingEvaluator<'a> {
    model: &'a SparseModel,
    positions: Rc<RefCell<Vec<String>>>,
    cap: usize,
}

impl Evaluator for RecordingEvaluator<'_> {
    fn evaluate(&self, position: &Position) -> f32 {
        let mut positions = self.positions.borrow_mut();
        if positions.len() < self.cap {
            positions.push(position.to_sfen_owned());
        }
        self.model.predict_from_position(position)
    }
}

fn load_positions(paths: &[PathBuf]) -> Result<Vec<(String, Position)>> {
    let mut positions = Vec::new();
    for path in paths {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        for line in BufReader::new(file).lines() {
            let line = line?;
            let text = line.trim();
            if text.is_empty() {
                continue;
            }
            if let Some(position) = position_from_sfen_or_usi(text) {
                positions.push((path.to_string_lossy().into_owned(), position));
            }
        }
    }
    if positions.is_empty() {
        return Err(anyhow!("no valid positions loaded"));
    }
    Ok(positions)
}

fn random_playouts(
    root: &Position,
    branches: usize,
    plies: usize,
    rng: &mut ChaCha8Rng,
    output: &mut UniqueWriter,
    source: &str,
) -> Result<()> {
    for _ in 0..branches {
        let mut position = root.clone();
        output.write_position(&position, source, "random")?;
        for _ in 0..plies {
            let legal = position.legal_moves();
            if legal.is_empty() || output.full() {
                break;
            }
            let Some(&mv) = legal.choose(rng) else { break };
            position.do_move(mv);
            output.write_position(&position, source, "random")?;
        }
        if output.full() {
            break;
        }
    }
    Ok(())
}

fn search_positions(
    root: &Position,
    model: &SparseModel,
    args: &Args,
    output: &mut UniqueWriter,
    source: &str,
) -> Result<()> {
    let evaluator = RecordingEvaluator {
        model,
        positions: Rc::new(RefCell::new(Vec::with_capacity(args.search_position_cap))),
        cap: args.search_position_cap,
    };
    let recorded = evaluator.positions.clone();
    let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(evaluator);
    ai.set_emit_info(false);
    ai.sennichite_detector.record_position(root);
    let mut position = root.clone();
    let _ = ai.find_best_move(&mut position, args.search_depth, args.search_time_limit_ms);
    drop(ai);
    for sfen in recorded.borrow_mut().drain(..) {
        if let Some(position) = position_from_sfen_or_usi(&sfen) {
            output.write_position(&position, source, "search")?;
        }
        if output.full() {
            break;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.search_roots > 0 && args.search_depth == 0 {
        return Err(anyhow!("--search-depth must be greater than zero"));
    }
    let mut roots = load_positions(&args.input)?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    roots.shuffle(&mut rng);
    if args.max_roots > 0 {
        roots.truncate(args.max_roots);
    }
    let search_limit = if args.search_roots == 0 {
        0
    } else {
        args.search_roots.min(roots.len())
    };
    let model = if search_limit > 0 {
        let mut model = SparseModel::new(0.0, 0.0);
        model.load(&args.weights)?;
        Some(model)
    } else {
        None
    };
    let mut output = UniqueWriter::new(&args.output, args.max_output)?;
    for (index, (source, root)) in roots.iter().enumerate() {
        if output.full() {
            break;
        }
        random_playouts(
            root,
            args.random_branches,
            args.random_plies,
            &mut rng,
            &mut output,
            source,
        )?;
        if index < search_limit {
            search_positions(root, model.as_ref().unwrap(), &args, &mut output, source)?;
        }
        if (index + 1) % 1000 == 0 {
            eprintln!("roots={} unique_positions={}", index + 1, output.written);
        }
    }
    let count = output.finish()?;
    println!(
        "roots={} search_roots={} unique_positions={}",
        roots.len(),
        search_limit,
        count
    );
    Ok(())
}
