use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use shogi_ai::evaluation::{extract_kpp_features_and_material, SparseModel};
use shogi_ai::utils::{format_move_usi, parse_usi_move, position_from_sfen_or_usi};
use shogi_core::{Move, Piece};
use shogi_lib::Position;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const SCORE_LIMIT: f32 = 100_000.0;

#[derive(Parser, Debug)]
#[command(about = "Probe MMTO-lite full-legal JSONL for model/teacher rank mismatch")]
struct Args {
    #[arg(long, default_value = "./policy_weights_v2.1.0.binary")]
    weights: PathBuf,
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 0.0)]
    min_regret: f32,
    #[arg(long, default_value_t = 0)]
    top: usize,
    #[arg(long, value_enum, default_value_t = OutputFormat::Jsonl)]
    format: OutputFormat,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutputFormat {
    Csv,
    Jsonl,
}

#[derive(Debug, Deserialize)]
struct KppRankRecord {
    sfen: String,
    #[serde(rename = "teacher_scores", default)]
    teacher_scores: Vec<LegacyTeacherScore>,
    #[serde(default)]
    candidates: Vec<RankCandidateRecord>,
    #[serde(default)]
    schema: Option<String>,
    #[serde(default)]
    version: Option<u8>,
}

#[derive(Debug, Deserialize)]
struct RankCandidateRecord {
    #[serde(rename = "move")]
    move_usi: String,
    teacher_score: f32,
}

#[derive(Debug, Deserialize)]
struct LegacyTeacherScore {
    #[serde(rename = "move_usi")]
    move_usi: String,
    score: f32,
}

#[derive(Clone)]
struct Candidate {
    mv: Move,
    teacher_score: f32,
    teacher_rank: usize,
}

#[derive(Serialize)]
struct ProbeOutput {
    sfen: String,
    model_move: String,
    // 0-based rank in the teacher-sorted candidate list.
    model_rank_by_teacher: usize,
    model_teacher_score: f32,
    teacher_best_move: String,
    teacher_best_score: f32,
    selected_regret: f32,
    model_score: f32,
    teacher_gap: f32,
    candidate_count: usize,
}

fn sanitize_score(score: f32) -> f32 {
    if score == f32::INFINITY {
        SCORE_LIMIT
    } else if score == -f32::INFINITY {
        -SCORE_LIMIT
    } else {
        score.clamp(-SCORE_LIMIT, SCORE_LIMIT)
    }
}

fn parse_move_for_position(position: &Position, move_text: &str) -> Option<Move> {
    parse_usi_move(move_text).map(|mv| match mv {
        Move::Drop { piece, to } => Move::Drop {
            piece: Piece::new(piece.piece_kind(), position.side_to_move()),
            to,
        },
        normal => normal,
    })
}

fn load_model(path: &Path) -> Result<SparseModel> {
    let mut model = SparseModel::new(0.0, 0.0);
    model
        .load(path)
        .map_err(|e| anyhow!("failed to load {}: {}", path.display(), e))?;
    Ok(model)
}

fn create_writer(path: &Path) -> Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(BufWriter::new(File::create(path)?))
}

fn parse_candidates(record: &KppRankRecord, position: &Position) -> Vec<Candidate> {
    if !record.candidates.is_empty() {
        let mut candidates = record
            .candidates
            .iter()
            .enumerate()
            .filter_map(|(idx, candidate)| {
                let mv = parse_move_for_position(position, &candidate.move_usi)?;
                if !position.legal_moves().contains(&mv) {
                    return None;
                }
                if !candidate.teacher_score.is_finite() {
                    return None;
                }
                Some(Candidate {
                    mv,
                    teacher_score: candidate.teacher_score,
                    teacher_rank: idx,
                })
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| {
            b.teacher_score
                .partial_cmp(&a.teacher_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut deduped = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if !deduped
                .iter()
                .any(|item: &Candidate| item.mv == candidate.mv)
            {
                deduped.push(candidate);
            }
        }
        for (rank, candidate) in deduped.iter_mut().enumerate() {
            candidate.teacher_rank = rank;
        }
        return deduped;
    }

    let mut candidates = record
        .teacher_scores
        .iter()
        .filter_map(|teacher_score| {
            let mv = parse_move_for_position(position, &teacher_score.move_usi)?;
            if !position.legal_moves().contains(&mv) {
                return None;
            }
            if !teacher_score.score.is_finite() {
                return None;
            }
            Some(Candidate {
                mv,
                teacher_score: teacher_score.score,
                teacher_rank: 0,
            })
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| {
        b.teacher_score
            .partial_cmp(&a.teacher_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut deduped = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if !deduped
            .iter()
            .any(|item: &Candidate| item.mv == candidate.mv)
        {
            deduped.push(candidate);
        }
    }
    for (rank, candidate) in deduped.iter_mut().enumerate() {
        candidate.teacher_rank = rank;
    }
    deduped
}

fn child_model_score(model: &SparseModel, root: &Position, mv: Move) -> f32 {
    let mut child = root.clone();
    child.do_move(mv);
    child.switch_turn();
    let (features, material) = extract_kpp_features_and_material(&child);
    sanitize_score(model.predict_with_material(&features, material))
}

fn probe_record(
    model: &SparseModel,
    position: Position,
    candidates: Vec<Candidate>,
) -> Option<ProbeOutput> {
    if candidates.is_empty() {
        return None;
    }

    let mut model_scores = Vec::with_capacity(candidates.len());
    let mut best_score = f32::NEG_INFINITY;
    let mut best_index = 0usize;

    for (idx, candidate) in candidates.iter().enumerate() {
        let model_score = child_model_score(model, &position, candidate.mv);
        if !model_score.is_finite() {
            continue;
        }
        if model_score > best_score {
            best_score = model_score;
            best_index = idx;
        }
        model_scores.push((idx, model_score, candidate.teacher_score));
    }

    if model_scores.is_empty() {
        return None;
    }

    let best_model_candidate = &candidates[best_index];
    let teacher_best = &candidates[0];
    let selected_regret =
        (teacher_best.teacher_score - best_model_candidate.teacher_score).max(0.0);

    Some(ProbeOutput {
        sfen: position.to_sfen_owned(),
        model_move: format_move_usi(best_model_candidate.mv),
        model_rank_by_teacher: best_model_candidate.teacher_rank,
        model_teacher_score: best_model_candidate.teacher_score,
        teacher_best_move: format_move_usi(teacher_best.mv),
        teacher_best_score: teacher_best.teacher_score,
        selected_regret,
        model_score: best_score,
        teacher_gap: if candidates.len() >= 2 {
            (candidates[0].teacher_score - candidates[1].teacher_score).max(0.0)
        } else {
            0.0
        },
        candidate_count: candidates.len(),
    })
}

fn csv_escape(field: &str) -> String {
    if field.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", field.replace('\"', "\"\""))
    } else {
        field.to_string()
    }
}

fn write_csv_output<W: Write>(writer: &mut W, records: &[ProbeOutput]) -> Result<()> {
    writeln!(
        writer,
        "sfen,model_move,model_rank_by_teacher,model_teacher_score,teacher_best_move,teacher_best_score,selected_regret,model_score,teacher_gap,candidate_count"
    )?;
    for record in records {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{}",
            csv_escape(&record.sfen),
            csv_escape(&record.model_move),
            record.model_rank_by_teacher,
            record.model_teacher_score,
            csv_escape(&record.teacher_best_move),
            record.teacher_best_score,
            record.selected_regret,
            record.model_score,
            record.teacher_gap,
            record.candidate_count
        )?;
    }
    Ok(())
}

fn write_jsonl_output<W: Write>(writer: &mut W, records: &[ProbeOutput]) -> Result<()> {
    for record in records {
        let line = serde_json::to_string(record)
            .map_err(|e| anyhow!("failed to serialize output record: {e}"))?;
        writeln!(writer, "{line}")?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.min_regret.is_finite() || args.min_regret < 0.0 {
        return Err(anyhow!("--min-regret must be non-negative"));
    }

    let model = load_model(&args.weights)?;
    let mut records = Vec::new();

    for path in &args.input {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        for (line_index, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let record: KppRankRecord = serde_json::from_str(&line)
                .map_err(|e| anyhow!("{path:?}:{:?} invalid json: {e}", line_index + 1))?;
            if record
                .schema
                .as_deref()
                .is_some_and(|schema| schema != "kpp_rank_v1")
            {
                continue;
            }
            if record.version.is_some_and(|version| version == 0) {
                continue;
            }
            let Some(position) = position_from_sfen_or_usi(&record.sfen) else {
                continue;
            };
            let candidates = parse_candidates(&record, &position);
            if candidates.is_empty() {
                continue;
            }
            let Some(result) = probe_record(&model, position, candidates) else {
                continue;
            };
            if result.selected_regret >= args.min_regret {
                records.push(result);
            }
        }
    }

    records.sort_by(|lhs, rhs| {
        rhs.selected_regret
            .partial_cmp(&lhs.selected_regret)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if args.top > 0 && records.len() > args.top {
        records.truncate(args.top);
    }

    let mut writer = create_writer(&args.output)?;
    match args.format {
        OutputFormat::Csv => write_csv_output(&mut writer, &records)?,
        OutputFormat::Jsonl => write_jsonl_output(&mut writer, &records)?,
    }
    Ok(())
}
