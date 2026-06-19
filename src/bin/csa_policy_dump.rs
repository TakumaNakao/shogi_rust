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
    #[arg(long)]
    min_player_rate: Option<i32>,
    #[arg(long, default_value_t = false)]
    winner_only: bool,
    #[arg(long, default_value_t = false)]
    decisive_only: bool,
}

#[derive(Serialize)]
struct PolicyRecord {
    sfen: String,
    teacher_move: String,
}

#[derive(Clone, Copy, Debug, Default)]
struct CsaMetadata {
    black_rate: Option<i32>,
    white_rate: Option<i32>,
    winner: Option<Color>,
}

#[derive(Default)]
struct GameDump {
    records: Vec<PolicyRecord>,
    filtered_records: usize,
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

fn csa_to_shogi_color(color: csa::Color) -> Color {
    if color == csa::Color::Black {
        Color::Black
    } else {
        Color::White
    }
}

fn parse_rate_line(line: &str, prefix: &str) -> Option<i32> {
    line.strip_prefix(prefix)
        .and_then(|rest| rest.rsplit(':').next())
        .and_then(|rate| rate.parse::<i32>().ok())
}

fn parse_csa_metadata(text: &str, record: &csa::GameRecord) -> CsaMetadata {
    let mut metadata = CsaMetadata::default();
    for line in text.lines() {
        if let Some(rate) = parse_rate_line(line, "'black_rate:") {
            metadata.black_rate = Some(rate);
        } else if let Some(rate) = parse_rate_line(line, "'white_rate:") {
            metadata.white_rate = Some(rate);
        }
    }
    metadata.winner = infer_winner(record);
    metadata
}

fn infer_winner(record: &csa::GameRecord) -> Option<Color> {
    let mut last_mover = None;
    for move_record in &record.moves {
        match move_record.action {
            csa::Action::Move(color, ..) => {
                last_mover = Some(csa_to_shogi_color(color));
            }
            csa::Action::Toryo | csa::Action::TimeUp | csa::Action::IllegalMove => {
                return last_mover;
            }
            csa::Action::IllegalAction(color) => {
                return Some(csa_to_shogi_color(color).flip());
            }
            csa::Action::Tsumi => {
                return last_mover;
            }
            csa::Action::Kachi => {
                return last_mover;
            }
            csa::Action::Chudan
            | csa::Action::Sennichite
            | csa::Action::Jishogi
            | csa::Action::Hikiwake
            | csa::Action::Matta
            | csa::Action::Fuzumi
            | csa::Action::Error => return None,
        }
    }
    None
}

fn player_rate(metadata: &CsaMetadata, color: Color) -> Option<i32> {
    match color {
        Color::Black => metadata.black_rate,
        Color::White => metadata.white_rate,
    }
}

fn should_include_move(
    color: Color,
    metadata: &CsaMetadata,
    min_player_rate: Option<i32>,
    winner_only: bool,
) -> bool {
    if winner_only && metadata.winner != Some(color) {
        return false;
    }
    if let Some(min_rate) = min_player_rate {
        if !player_rate(metadata, color).is_some_and(|rate| rate >= min_rate) {
            return false;
        }
    }
    true
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
    min_player_rate: Option<i32>,
    winner_only: bool,
    decisive_only: bool,
) -> Result<GameDump> {
    let text = fs::read_to_string(path)?;
    let record = csa::parse_csa(&text)?;
    let metadata = parse_csa_metadata(&text, &record);
    if decisive_only && metadata.winner.is_none() {
        return Ok(GameDump::default());
    }
    let mut position = Position::default();
    let mut records = Vec::new();
    let mut filtered_records = 0usize;

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
        let move_color = match csa_move.action {
            csa::Action::Move(color, ..) => csa_to_shogi_color(color),
            _ => break,
        };
        if ply_index >= min_ply
            && should_include_move(move_color, &metadata, min_player_rate, winner_only)
        {
            records.push(PolicyRecord {
                sfen: position.to_sfen_owned(),
                teacher_move: format_move_usi(mv),
            });
        } else if ply_index >= min_ply {
            filtered_records += 1;
        }
        position.do_move(mv);
    }

    Ok(GameDump {
        records,
        filtered_records,
    })
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
    let mut filtered_games = 0usize;
    let mut filtered_records = 0usize;

    for path in files {
        if args
            .max_records
            .is_some_and(|limit| train_count + valid_count >= limit)
        {
            break;
        }
        let dump = match records_from_csa(
            &path,
            args.min_ply,
            args.max_ply,
            args.min_player_rate,
            args.winner_only,
            args.decisive_only,
        ) {
            Ok(records) => records,
            Err(_) => {
                skipped_games += 1;
                continue;
            }
        };
        filtered_records += dump.filtered_records;
        if dump.records.is_empty() {
            filtered_games += 1;
            continue;
        }
        games += 1;
        for record in dump.records {
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
    println!("games filtered: {filtered_games}");
    println!("records filtered: {filtered_records}");
    println!("train records: {train_count}");
    println!("valid records: {valid_count}");
    Ok(())
}
