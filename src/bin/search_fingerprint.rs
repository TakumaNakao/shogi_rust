use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use shogi_ai::ai::ShogiAI;
use shogi_ai::evaluation::Evaluator;
use shogi_ai::position_hash::PositionHasher;
use shogi_ai::utils::{format_move_usi, position_from_sfen_or_usi};
use shogi_lib::Position;
use std::fs;
use std::path::PathBuf;

const HISTORY_CAPACITY: usize = 256;
const OUTPUT_SCHEMA_VERSION: u32 = 1;
const EVALUATOR_ID: &str = "position-hash-v1";

#[derive(Parser, Debug)]
#[command(about = "Generate a deterministic Threads=1 search fingerprint")]
struct Args {
    #[arg(long, default_value = "tests/fixtures/search/phase0_positions.json")]
    positions: PathBuf,

    #[arg(long, default_value_t = 3)]
    depth: u8,

    #[arg(long)]
    expected: Option<PathBuf>,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct FixtureFile {
    schema_version: u32,
    cases: Vec<FixtureCase>,
}

#[derive(Debug, Deserialize)]
struct FixtureCase {
    id: String,
    position: String,
    expect_terminal: bool,
}

#[derive(Clone, Copy)]
struct HashEvaluator;

impl Evaluator for HashEvaluator {
    fn evaluate(&self, position: &Position) -> f32 {
        (PositionHasher::calculate_hash(position) % 2_001) as f32 - 1_000.0
    }
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
struct FingerprintOutput {
    schema_version: u32,
    fixture_schema_version: u32,
    evaluator: String,
    depth: u8,
    threads: usize,
    cases: Vec<CaseFingerprint>,
    totals: SearchCounters,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
struct CaseFingerprint {
    id: String,
    input: String,
    normalized_sfen: String,
    position_key: u64,
    terminal: bool,
    legal_moves: usize,
    bestmove: Option<String>,
    root_score: Option<f32>,
    root_score_bits: Option<u32>,
    pv: Vec<String>,
    completed_depth: u8,
    search_failed: bool,
    position_restored: bool,
    counters: SearchCounters,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
struct SearchCounters {
    nodes: u64,
    quiescence_nodes: u64,
    quiescence_moves_considered: u64,
    quiescence_moves_generated: u64,
    quiescence_moves_discarded: u64,
    quiescence_moves_searched: u64,
    quiescence_see_skips: u64,
    quiescence_terminal_mates: u64,
    check_evasion_extensions: u64,
    aspiration_fail_lows: u64,
    aspiration_fail_highs: u64,
    aspiration_researches: u64,
}

impl SearchCounters {
    fn capture(ai: &ShogiAI<HashEvaluator, HISTORY_CAPACITY>) -> Self {
        Self {
            nodes: ai.nodes_searched(),
            quiescence_nodes: ai.quiescence_nodes_searched(),
            quiescence_moves_considered: ai.quiescence_moves_considered(),
            quiescence_moves_generated: ai.quiescence_moves_generated(),
            quiescence_moves_discarded: ai.quiescence_moves_discarded(),
            quiescence_moves_searched: ai.quiescence_moves_searched(),
            quiescence_see_skips: ai.quiescence_see_skips(),
            quiescence_terminal_mates: ai.quiescence_terminal_mates(),
            check_evasion_extensions: ai.check_evasion_extensions(),
            aspiration_fail_lows: ai.aspiration_fail_lows(),
            aspiration_fail_highs: ai.aspiration_fail_highs(),
            aspiration_researches: ai.aspiration_researches(),
        }
    }

    fn add_assign(&mut self, other: Self) {
        self.nodes += other.nodes;
        self.quiescence_nodes += other.quiescence_nodes;
        self.quiescence_moves_considered += other.quiescence_moves_considered;
        self.quiescence_moves_generated += other.quiescence_moves_generated;
        self.quiescence_moves_discarded += other.quiescence_moves_discarded;
        self.quiescence_moves_searched += other.quiescence_moves_searched;
        self.quiescence_see_skips += other.quiescence_see_skips;
        self.quiescence_terminal_mates += other.quiescence_terminal_mates;
        self.check_evasion_extensions += other.check_evasion_extensions;
        self.aspiration_fail_lows += other.aspiration_fail_lows;
        self.aspiration_fail_highs += other.aspiration_fail_highs;
        self.aspiration_researches += other.aspiration_researches;
    }
}

fn load_fixtures(path: &PathBuf) -> Result<FixtureFile> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read fixture {}", path.display()))?;
    let fixtures: FixtureFile = serde_json::from_str(&content)
        .with_context(|| format!("failed to parse fixture {}", path.display()))?;
    if fixtures.schema_version != 1 {
        bail!(
            "unsupported fixture schema version {} in {}",
            fixtures.schema_version,
            path.display()
        );
    }
    if fixtures.cases.is_empty() {
        bail!("fixture contains no cases: {}", path.display());
    }
    Ok(fixtures)
}

fn generate_fingerprint(fixtures: FixtureFile, depth: u8) -> Result<FingerprintOutput> {
    if depth == 0 {
        bail!("--depth must be greater than zero");
    }

    let mut cases = Vec::with_capacity(fixtures.cases.len());
    let mut totals = SearchCounters::default();

    for fixture in fixtures.cases {
        let mut position = position_from_sfen_or_usi(&fixture.position)
            .ok_or_else(|| anyhow!("invalid position in fixture {}", fixture.id))?;
        let original_sfen = position.to_sfen_owned();
        let position_key = PositionHasher::calculate_hash(&position);
        let legal_moves = position.legal_moves();
        let terminal = legal_moves.is_empty();
        if terminal != fixture.expect_terminal {
            bail!(
                "fixture {} expected terminal={} but legal move count is {}",
                fixture.id,
                fixture.expect_terminal,
                legal_moves.len()
            );
        }

        let mut ai = ShogiAI::<_, HISTORY_CAPACITY>::new(HashEvaluator);
        ai.set_emit_info(false);
        ai.sennichite_detector.record_position(&position);
        let best_move = ai.find_best_move_parallel(&mut position, depth, None, 1);
        let restored_sfen = position.to_sfen_owned();
        let position_restored = restored_sfen == original_sfen;
        if !position_restored {
            bail!(
                "search did not restore fixture {}: expected {}, actual {}",
                fixture.id,
                original_sfen,
                restored_sfen
            );
        }

        if terminal {
            if best_move.is_some() {
                bail!("terminal fixture {} returned a best move", fixture.id);
            }
        } else {
            let best_move =
                best_move.ok_or_else(|| anyhow!("fixture {} returned no best move", fixture.id))?;
            if !legal_moves.contains(&best_move) {
                bail!("fixture {} returned an illegal best move", fixture.id);
            }
            if ai.last_pv().first().copied() != Some(best_move) {
                bail!(
                    "fixture {} best move does not match the first PV move",
                    fixture.id
                );
            }
        }

        let root_score = ai.last_root_score();
        let counters = SearchCounters::capture(&ai);
        totals.add_assign(counters);
        cases.push(CaseFingerprint {
            id: fixture.id,
            input: fixture.position,
            normalized_sfen: original_sfen,
            position_key,
            terminal,
            legal_moves: legal_moves.len(),
            bestmove: best_move.map(format_move_usi),
            root_score,
            root_score_bits: root_score.map(f32::to_bits),
            pv: ai.last_pv().iter().copied().map(format_move_usi).collect(),
            completed_depth: ai.last_completed_depth(),
            search_failed: ai.last_search_failed(),
            position_restored,
            counters,
        });
    }

    Ok(FingerprintOutput {
        schema_version: OUTPUT_SCHEMA_VERSION,
        fixture_schema_version: fixtures.schema_version,
        evaluator: EVALUATOR_ID.to_string(),
        depth,
        threads: 1,
        cases,
        totals,
    })
}

fn verify_expected(path: &PathBuf, actual: &FingerprintOutput) -> Result<()> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read expected fingerprint {}", path.display()))?;
    let expected: FingerprintOutput = serde_json::from_str(&content)
        .with_context(|| format!("failed to parse expected fingerprint {}", path.display()))?;

    if expected.schema_version != actual.schema_version
        || expected.fixture_schema_version != actual.fixture_schema_version
        || expected.evaluator != actual.evaluator
        || expected.depth != actual.depth
        || expected.threads != actual.threads
    {
        bail!(
            "fingerprint metadata mismatch\nexpected: schema={} fixture_schema={} evaluator={} depth={} threads={}\nactual:   schema={} fixture_schema={} evaluator={} depth={} threads={}",
            expected.schema_version,
            expected.fixture_schema_version,
            expected.evaluator,
            expected.depth,
            expected.threads,
            actual.schema_version,
            actual.fixture_schema_version,
            actual.evaluator,
            actual.depth,
            actual.threads
        );
    }

    if expected.cases.len() != actual.cases.len() {
        bail!(
            "fingerprint case count mismatch: expected {}, actual {}",
            expected.cases.len(),
            actual.cases.len()
        );
    }
    for (expected_case, actual_case) in expected.cases.iter().zip(&actual.cases) {
        if expected_case != actual_case {
            bail!(
                "fingerprint mismatch for case {}\nexpected: {:#?}\nactual: {:#?}",
                actual_case.id,
                expected_case,
                actual_case
            );
        }
    }
    if expected.totals != actual.totals {
        bail!(
            "fingerprint totals mismatch\nexpected: {:#?}\nactual: {:#?}",
            expected.totals,
            actual.totals
        );
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let fixtures = load_fixtures(&args.positions)?;
    let fingerprint = generate_fingerprint(fixtures, args.depth)?;
    let json = serde_json::to_string_pretty(&fingerprint)? + "\n";

    if let Some(path) = &args.expected {
        verify_expected(path, &fingerprint)?;
        eprintln!("search fingerprint matches {}", path.display());
    }
    if let Some(path) = &args.output {
        fs::write(path, json.as_bytes())
            .with_context(|| format!("failed to write fingerprint {}", path.display()))?;
    } else if args.expected.is_none() {
        print!("{json}");
    }

    Ok(())
}
