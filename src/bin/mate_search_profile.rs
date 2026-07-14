use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::Deserialize;
use shogi_ai::mate_search::{MateSearchLimits, MateSearchResult, MateSearcher};
use shogi_ai::utils::{format_move_usi, position_from_sfen_or_usi};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Profile rule-only mate search budgets on the development suite")]
struct Args {
    #[arg(
        long,
        default_value = "data/search_quality/generated/dev_mate_sacrifice.jsonl",
        help = "Development-suite JSONL only; holdout input is prohibited"
    )]
    input: PathBuf,
    #[arg(long, value_delimiter = ',', default_value = "256,512,1024,2048,4096")]
    budgets: Vec<u64>,
    #[arg(long, default_value_t = 0)]
    limit: usize,
    #[arg(long)]
    source_index: Option<usize>,
}

#[derive(Deserialize)]
struct Record {
    source_index: usize,
    sfen: String,
    first_move: String,
    mate_horizon: u8,
}

fn reject_holdout(path: &Path) -> Result<()> {
    if path
        .components()
        .any(|part| part.as_os_str().to_string_lossy().contains("holdout"))
    {
        return Err(anyhow!(
            "holdout input is prohibited for mate search profiling"
        ));
    }
    Ok(())
}

fn percentile(values: &mut [u64], percentile: usize) -> u64 {
    values.sort_unstable();
    values[(values.len() - 1) * percentile / 100]
}

fn main() -> Result<()> {
    let args = Args::parse();
    reject_holdout(&args.input)?;
    if args.budgets.is_empty() || args.budgets.contains(&0) {
        return Err(anyhow!("budgets must be non-empty and positive"));
    }
    let content = fs::read_to_string(&args.input)
        .with_context(|| format!("failed to read {}", args.input.display()))?;
    let mut records = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str::<Record>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if let Some(source_index) = args.source_index {
        records.retain(|record| record.source_index == source_index);
    }
    if args.limit > 0 {
        records.truncate(args.limit);
    }
    if records.is_empty() {
        return Err(anyhow!("development mate suite is empty"));
    }

    for budget in args.budgets {
        let mut proven = 0usize;
        let mut expected = 0usize;
        let mut unknown = 0usize;
        let mut nodes = Vec::with_capacity(records.len());
        let mut exact_fixtures = [None; 4];
        for record in &records {
            let mut position = position_from_sfen_or_usi(&record.sfen)
                .ok_or_else(|| anyhow!("invalid sfen: {}", record.sfen))?;
            let attacker = position.side_to_move();
            let mut searcher = MateSearcher::new(MateSearchLimits::nodes(budget));
            match searcher.search_shortest(&mut position, attacker, record.mate_horizon) {
                MateSearchResult::ProvenMate {
                    first_move, ply, ..
                } => {
                    proven += 1;
                    expected += usize::from(
                        first_move.map(format_move_usi).as_deref() == Some(&record.first_move),
                    );
                    if ply == record.mate_horizon {
                        exact_fixtures[((ply - 1) / 2) as usize].get_or_insert(record.source_index);
                    }
                }
                MateSearchResult::Unknown => unknown += 1,
                MateSearchResult::ProvenNoMateWithinHorizon => {}
            }
            nodes.push(searcher.nodes());
        }
        let mut p50_nodes = nodes.clone();
        let mut p95_nodes = nodes.clone();
        println!(
            "budget={budget} samples={} proven={} expected={} unknown={} nodes_total={} nodes_p50={} nodes_p95={} nodes_max={}",
            records.len(),
            proven,
            expected,
            unknown,
            nodes.iter().sum::<u64>(),
            percentile(&mut p50_nodes, 50),
            percentile(&mut p95_nodes, 95),
            nodes.iter().copied().max().unwrap_or(0)
        );
        println!(
            "exact_fixtures={}",
            [1u8, 3, 5, 7]
                .into_iter()
                .zip(exact_fixtures)
                .filter_map(|(ply, source)| source.map(|source| format!("{ply}:{source}")))
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    Ok(())
}
