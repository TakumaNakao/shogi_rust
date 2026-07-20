use anyhow::Result;
use clap::Parser;
use shogi_ai::training_data::collect_csa_files;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Summarize wdoor/floodgate CSA player-rate distribution")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long, value_delimiter = ',', default_value = "3000,3500,4000,4500")]
    thresholds: Vec<i32>,
    #[arg(long, default_value_t = 20)]
    top_players: usize,
}

#[derive(Clone, Debug, Default)]
struct GameRates {
    black_name: Option<String>,
    white_name: Option<String>,
    black_rate: Option<i32>,
    white_rate: Option<i32>,
}

#[derive(Clone, Debug, Default)]
struct PlayerStats {
    games: usize,
    max_rate: i32,
    rate_sum: i64,
}

impl PlayerStats {
    fn add(&mut self, rate: i32) {
        self.games += 1;
        self.max_rate = self.max_rate.max(rate);
        self.rate_sum += i64::from(rate);
    }

    fn avg_rate(&self) -> f64 {
        if self.games == 0 {
            0.0
        } else {
            self.rate_sum as f64 / self.games as f64
        }
    }
}

fn parse_player_name(line: &str, prefix: &str) -> Option<String> {
    line.strip_prefix(prefix)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
}

fn parse_rate_line(line: &str, prefix: &str) -> Option<i32> {
    line.strip_prefix(prefix)
        .and_then(|rest| rest.rsplit(':').next())
        .and_then(|rate| rate.parse::<f64>().ok())
        .map(|rate| rate.round() as i32)
}

fn parse_game_rates(path: &Path) -> Result<GameRates> {
    let bytes = fs::read(path)?;
    let text = String::from_utf8_lossy(&bytes);
    let mut rates = GameRates::default();
    for line in text.lines() {
        if let Some(name) = parse_player_name(line, "N+") {
            rates.black_name = Some(name);
        } else if let Some(name) = parse_player_name(line, "N-") {
            rates.white_name = Some(name);
        } else if let Some(rate) = parse_rate_line(line, "'black_rate:") {
            rates.black_rate = Some(rate);
        } else if let Some(rate) = parse_rate_line(line, "'white_rate:") {
            rates.white_rate = Some(rate);
        }
    }
    Ok(rates)
}

fn percentile(sorted: &[i32], pct: usize) -> Option<i32> {
    if sorted.is_empty() {
        return None;
    }
    let idx = ((sorted.len() - 1) * pct + 50) / 100;
    sorted.get(idx).copied()
}

fn add_player(
    players: &mut BTreeMap<String, PlayerStats>,
    name: Option<String>,
    rate: Option<i32>,
) {
    let (Some(name), Some(rate)) = (name, rate) else {
        return;
    };
    players.entry(name).or_default().add(rate);
}

fn main() -> Result<()> {
    let args = Args::parse();
    let files = collect_csa_files(&args.input)?;

    let mut all_rates = Vec::new();
    let mut both_rates = Vec::new();
    let mut missing_games = 0usize;
    let mut players = BTreeMap::<String, PlayerStats>::new();

    for path in &files {
        let rates = parse_game_rates(path)?;
        add_player(&mut players, rates.black_name.clone(), rates.black_rate);
        add_player(&mut players, rates.white_name.clone(), rates.white_rate);
        match (rates.black_rate, rates.white_rate) {
            (Some(black), Some(white)) => {
                all_rates.push(black);
                all_rates.push(white);
                both_rates.push((black, white));
            }
            (Some(rate), None) | (None, Some(rate)) => {
                all_rates.push(rate);
                missing_games += 1;
            }
            (None, None) => {
                missing_games += 1;
            }
        }
    }

    all_rates.sort_unstable();
    both_rates.sort_unstable();

    println!("CSA files: {}", files.len());
    println!("games with both rates: {}", both_rates.len());
    println!("games missing at least one rate: {missing_games}");
    println!("player-rate entries: {}", all_rates.len());
    if let (Some(min), Some(max)) = (all_rates.first(), all_rates.last()) {
        println!("rate min/max: {min}/{max}");
        println!(
            "rate p50/p75/p90/p95/p99: {}/{}/{}/{}/{}",
            percentile(&all_rates, 50).unwrap_or_default(),
            percentile(&all_rates, 75).unwrap_or_default(),
            percentile(&all_rates, 90).unwrap_or_default(),
            percentile(&all_rates, 95).unwrap_or_default(),
            percentile(&all_rates, 99).unwrap_or_default()
        );
    }

    println!();
    for threshold in args.thresholds {
        let player_entries = all_rates.iter().filter(|rate| **rate >= threshold).count();
        let either_side = both_rates
            .iter()
            .filter(|(black, white)| *black >= threshold || *white >= threshold)
            .count();
        let both_sides = both_rates
            .iter()
            .filter(|(black, white)| *black >= threshold && *white >= threshold)
            .count();
        println!(
            "rate >= {threshold}: player entries {player_entries}, games either side {either_side}, games both sides {both_sides}"
        );
    }

    let mut top_players: Vec<_> = players.into_iter().collect();
    top_players.sort_by(|(_, lhs), (_, rhs)| {
        rhs.max_rate
            .cmp(&lhs.max_rate)
            .then_with(|| rhs.games.cmp(&lhs.games))
    });
    let top_names = top_players
        .iter()
        .take(args.top_players)
        .map(|(name, _)| name.as_str())
        .collect::<BTreeSet<_>>();
    if !top_names.is_empty() {
        println!();
        println!("top players by max rate:");
        for (name, stats) in top_players
            .iter()
            .filter(|(name, _)| top_names.contains(name.as_str()))
        {
            println!(
                "  {name}: games {}, max {}, avg {:.1}",
                stats.games,
                stats.max_rate,
                stats.avg_rate()
            );
        }
    }

    Ok(())
}
