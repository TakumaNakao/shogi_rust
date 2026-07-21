use anyhow::{anyhow, Result};
use clap::Parser;
use serde::Serialize;
use shogi_ai::evaluation::{extract_halfkp_features_for, SparseModel};
use shogi_ai::utils::position_from_sfen_or_usi;
use shogi_core::Color;
use shogi_lib::Position;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Build KPP legal-child rank data for HalfKP distillation")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long, default_value = "policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long)]
    flat_output: Option<PathBuf>,
    #[arg(long)]
    groups_output: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    max_roots: usize,
}

#[derive(Clone, Serialize)]
struct Candidate {
    features_black: Vec<usize>,
    features_white: Vec<usize>,
    material_black: f32,
    material_white: f32,
    side_to_move: &'static str,
    static_eval: f32,
}

#[derive(Serialize)]
struct FlatRecord<'a> {
    schema: &'static str,
    sfen: String,
    features_black: &'a [usize],
    features_white: &'a [usize],
    material_black: f32,
    material_white: f32,
    side_to_move: &'static str,
    static_eval: f32,
}

#[derive(Serialize)]
struct RankGroup {
    schema: &'static str,
    root_sfen: String,
    candidates: Vec<Candidate>,
}

fn writer(path: Option<&PathBuf>) -> Result<Option<BufWriter<File>>> {
    path.map(|path| {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(BufWriter::new(File::create(path)?))
    })
    .transpose()
}

fn roots(paths: &[PathBuf]) -> Result<Vec<Position>> {
    let mut roots = Vec::new();
    for path in paths {
        for line in BufReader::new(File::open(path)?).lines() {
            if let Some(position) = position_from_sfen_or_usi(line?.trim()) {
                roots.push(position);
            }
        }
    }
    if roots.is_empty() {
        return Err(anyhow!("no valid roots"));
    }
    Ok(roots)
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    if args.flat_output.is_none() && args.groups_output.is_none() {
        return Err(anyhow!("at least one output is required"));
    }
    let mut model = SparseModel::new(0.0, 0.0);
    model.load(&args.weights)?;
    let mut roots = roots(&args.input)?;
    if args.max_roots > 0 {
        roots.truncate(args.max_roots);
    }
    let mut flat = writer(args.flat_output.as_ref())?;
    let mut groups = writer(args.groups_output.as_ref())?;
    let mut root_count = 0usize;
    let mut candidate_count = 0usize;
    for root in roots {
        let mut candidates = Vec::new();
        for mv in root.legal_moves() {
            let mut child = root.clone();
            child.do_move(mv);
            if child.legal_moves().is_empty() {
                continue;
            }
            let Some(black) = extract_halfkp_features_for(&child, Color::Black) else {
                continue;
            };
            let Some(white) = extract_halfkp_features_for(&child, Color::White) else {
                continue;
            };
            candidates.push((
                child.to_sfen_owned(),
                Candidate {
                    features_black: black.features,
                    features_white: white.features,
                    material_black: black.material,
                    material_white: white.material,
                    side_to_move: if child.side_to_move() == Color::Black {
                        "black"
                    } else {
                        "white"
                    },
                    static_eval: model.predict_from_position(&child),
                },
            ));
        }
        if candidates.len() < 2 {
            continue;
        }
        candidates.sort_by(|a, b| a.1.static_eval.total_cmp(&b.1.static_eval));
        if let Some(output) = flat.as_mut() {
            for (sfen, candidate) in &candidates {
                serde_json::to_writer(
                    &mut *output,
                    &FlatRecord {
                        schema: "halfkp-v1",
                        sfen: sfen.clone(),
                        features_black: &candidate.features_black,
                        features_white: &candidate.features_white,
                        material_black: candidate.material_black,
                        material_white: candidate.material_white,
                        side_to_move: candidate.side_to_move,
                        static_eval: candidate.static_eval,
                    },
                )?;
                output.write_all(b"\n")?;
                candidate_count += 1;
            }
        } else {
            candidate_count += candidates.len();
        }
        if let Some(output) = groups.as_mut() {
            serde_json::to_writer(
                &mut *output,
                &RankGroup {
                    schema: "halfkp_rank_v1",
                    root_sfen: root.to_sfen_owned(),
                    candidates: candidates
                        .into_iter()
                        .map(|(_, candidate)| candidate)
                        .collect(),
                },
            )?;
            output.write_all(b"\n")?;
        }
        root_count += 1;
    }
    if let Some(output) = flat.as_mut() {
        output.flush()?;
    }
    if let Some(output) = groups.as_mut() {
        output.flush()?;
    }
    println!("roots={root_count} candidates={candidate_count}");
    Ok(())
}
