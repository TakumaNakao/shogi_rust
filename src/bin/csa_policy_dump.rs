use anyhow::{anyhow, Result};
use clap::Parser;
use glob::glob;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use shogi_ai::utils::format_move_usi;
use shogi_core::{Color, Move, Piece, PieceKind, Square};
use shogi_lib::Position;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Convert CSA games into SFEN+teacher-move JSONL for policy training")]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    #[arg(long)]
    train_output: PathBuf,
    #[arg(long)]
    valid_output: PathBuf,
    #[arg(long, default_value_t = 9601)]
    seed: u64,
    #[arg(long, default_value_t = 10)]
    valid_percent: u8,
    #[arg(long)]
    max_records: Option<usize>,
    #[arg(long, default_value_t = 0)]
    min_ply: usize,
    #[arg(long)]
    max_ply: Option<usize>,
}

#[derive(Serialize)]
struct PolicyRecord {
    sfen: String,
    teacher_move: String,
}

fn csa_to_shogi_piece_kind(csa_piece_type: csa::PieceType) -> PieceKind {
    match csa_piece_type {
        csa::PieceType::Pawn => PieceKind::Pawn,
        csa::PieceType::Lance => PieceKind::Lance,
        csa::PieceType::Knight => PieceKind::Knight,
        csa::PieceType::Silver => PieceKind::Silver,
        csa::PieceType::Gold => PieceKind::Gold,
        csa::PieceType::Bishop => PieceKind::Bishop,
        csa::PieceType::Rook => PieceKind::Rook,
        csa::PieceType::King => PieceKind::King,
        csa::PieceType::ProPawn => PieceKind::ProPawn,
        csa::PieceType::ProLance => PieceKind::ProLance,
        csa::PieceType::ProKnight => PieceKind::ProKnight,
        csa::PieceType::ProSilver => PieceKind::ProSilver,
        csa::PieceType::Horse => PieceKind::ProBishop,
        csa::PieceType::Dragon => PieceKind::ProRook,
        csa::PieceType::All => unreachable!(),
    }
}

fn collect_csa_files(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for input in inputs {
        if input.is_dir() {
            let pattern = input.join("**/*.csa");
            let pattern = pattern
                .to_str()
                .ok_or_else(|| anyhow!("path is not valid UTF-8: {}", input.display()))?;
            for entry in glob(pattern)? {
                files.push(entry?);
            }
        } else if input
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("csa"))
        {
            files.push(input.clone());
        }
    }
    files.sort();
    files.dedup();
    if files.is_empty() {
        return Err(anyhow!("no CSA files found"));
    }
    Ok(files)
}

fn parse_csa_move(position: &Position, action: &csa::Action) -> Option<Move> {
    let csa::Action::Move(color, from_csa, to_csa, piece_type_after_csa) = action else {
        return None;
    };
    let to = Square::new(to_csa.file, to_csa.rank)?;
    if from_csa.file == 0 && from_csa.rank == 0 {
        let piece_kind = csa_to_shogi_piece_kind(*piece_type_after_csa);
        let piece_color = if *color == csa::Color::Black {
            Color::Black
        } else {
            Color::White
        };
        Some(Move::Drop {
            piece: Piece::new(piece_kind, piece_color),
            to,
        })
    } else {
        let from = Square::new(from_csa.file, from_csa.rank)?;
        let piece_before = position.piece_at(from)?;
        let promote = piece_before.piece_kind() != csa_to_shogi_piece_kind(*piece_type_after_csa);
        Some(Move::Normal { from, to, promote })
    }
}

fn records_from_csa(
    path: &Path,
    min_ply: usize,
    max_ply: Option<usize>,
) -> Result<Vec<PolicyRecord>> {
    let text = fs::read_to_string(path)?;
    let record = csa::parse_csa(&text)?;
    let mut position = Position::default();
    let mut records = Vec::new();

    for (ply_index, csa_move) in record.moves.iter().enumerate() {
        if let Some(max_ply) = max_ply {
            if ply_index >= max_ply {
                break;
            }
        }
        let Some(mv) = parse_csa_move(&position, &csa_move.action) else {
            break;
        };
        if !position.legal_moves().contains(&mv) {
            break;
        }
        if ply_index >= min_ply {
            records.push(PolicyRecord {
                sfen: position.to_sfen_owned(),
                teacher_move: format_move_usi(mv),
            });
        }
        position.do_move(mv);
    }

    Ok(records)
}

fn create_writer(path: &Path) -> Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(BufWriter::new(File::create(path)?))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.valid_percent > 90 {
        return Err(anyhow!("--valid-percent must be 0..=90"));
    }
    let mut files = collect_csa_files(&args.input)?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    files.shuffle(&mut rng);

    let mut train_writer = create_writer(&args.train_output)?;
    let mut valid_writer = create_writer(&args.valid_output)?;
    let valid_stride = if args.valid_percent == 0 {
        usize::MAX
    } else {
        (100 / args.valid_percent as usize).max(1)
    };
    let mut train_count = 0usize;
    let mut valid_count = 0usize;
    let mut games = 0usize;
    let mut skipped_games = 0usize;

    for path in files {
        if args
            .max_records
            .is_some_and(|limit| train_count + valid_count >= limit)
        {
            break;
        }
        let records = match records_from_csa(&path, args.min_ply, args.max_ply) {
            Ok(records) => records,
            Err(_) => {
                skipped_games += 1;
                continue;
            }
        };
        if records.is_empty() {
            skipped_games += 1;
            continue;
        }
        games += 1;
        for record in records {
            if args
                .max_records
                .is_some_and(|limit| train_count + valid_count >= limit)
            {
                break;
            }
            let line = serde_json::to_string(&record)?;
            let record_index = train_count + valid_count;
            if valid_stride != usize::MAX && record_index % valid_stride == 0 {
                writeln!(valid_writer, "{line}")?;
                valid_count += 1;
            } else {
                writeln!(train_writer, "{line}")?;
                train_count += 1;
            }
        }
    }

    train_writer.flush()?;
    valid_writer.flush()?;
    println!("games used: {games}");
    println!("games skipped: {skipped_games}");
    println!("train records: {train_count}");
    println!("valid records: {valid_count}");
    Ok(())
}
