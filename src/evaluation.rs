use anyhow::Result;
use shogi_core::{Color, Piece, PieceKind, Position, Square};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use rand::prelude::*;
use rand_distr::Distribution;


const NUM_SQUARES: usize = 81;
// KPPで扱う駒の種類（盤上）。玉を除いた13種（歩,香,桂,銀,金,角,飛 + それらの成駒）。
const NUM_BOARD_PIECE_KINDS: usize = 13;
// KPPで扱う駒の種類（持ち駒）。成駒はないので7種。
const NUM_HAND_PIECE_KINDS: usize = 7;

// 各持ち駒の最大枚数
const MAX_HAND_PAWNS: usize = 18;
const MAX_HAND_LANCES: usize = 4;
const MAX_HAND_KNIGHTS: usize = 4;
const MAX_HAND_SILVERS: usize = 4;
const MAX_HAND_GOLDS: usize = 4;
const MAX_HAND_BISHOPS: usize = 2;
const MAX_HAND_ROOKS: usize = 2;

// 片方のプレイヤーが持ちうる持ち駒の総スロット数
const NUM_HAND_PIECE_SLOTS_PER_PLAYER: usize = MAX_HAND_PAWNS
    + MAX_HAND_LANCES
    + MAX_HAND_KNIGHTS
    + MAX_HAND_SILVERS
    + MAX_HAND_GOLDS
    + MAX_HAND_BISHOPS
    + MAX_HAND_ROOKS; // = 38

// 全ての駒の状態（駒ID）の総数
// = (盤上の駒状態) + (持ち駒の状態)
// = (13種 * 81マス * 2色) + (38スロット * 2色) = 2106 + 76 = 2182
const NUM_PIECE_STATES: usize =
    (NUM_BOARD_PIECE_KINDS * NUM_SQUARES * 2) + (NUM_HAND_PIECE_SLOTS_PER_PLAYER * 2);

// 2つの駒の組み合わせの総数
const NUM_PIECE_PAIRS: usize = NUM_PIECE_STATES * (NUM_PIECE_STATES - 1) / 2; // = 2,379,591

// KPP特徴量の総次元数
// = 玉の位置(81) * 駒のペアの組み合わせ総数
// 81 * 2,379,591 = 192,746,871
// 少し余裕を持たせて1億9300万に設定
pub const KPP_DIM: usize = 193_000_000;

const ALL_HAND_PIECES: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

// --- ヘルパー関数 ---

/// 盤上の駒の種類を0-12のインデックスに変換する
fn board_kind_to_index(kind: PieceKind) -> Option<usize> {
    match kind {
        PieceKind::Pawn => Some(0),
        PieceKind::Lance => Some(1),
        PieceKind::Knight => Some(2),
        PieceKind::Silver => Some(3),
        PieceKind::Gold => Some(4),
        PieceKind::Bishop => Some(5),
        PieceKind::Rook => Some(6),
        PieceKind::ProPawn => Some(7),
        PieceKind::ProLance => Some(8),
        PieceKind::ProKnight => Some(9),
        PieceKind::ProSilver => Some(10),
        PieceKind::ProBishop => Some(11),
        PieceKind::ProRook => Some(12),
        PieceKind::King => None, // 玉は対象外
    }
}

/// 持ち駒の種類からID計算用のオフセットを返す
fn hand_kind_to_offset(kind: PieceKind) -> Option<usize> {
    match kind {
        PieceKind::Pawn => Some(0),
        PieceKind::Lance => Some(MAX_HAND_PAWNS),
        PieceKind::Knight => Some(MAX_HAND_PAWNS + MAX_HAND_LANCES),
        PieceKind::Silver => Some(MAX_HAND_PAWNS + MAX_HAND_LANCES + MAX_HAND_KNIGHTS),
        PieceKind::Gold => Some(MAX_HAND_PAWNS + MAX_HAND_LANCES + MAX_HAND_KNIGHTS + MAX_HAND_SILVERS),
        PieceKind::Bishop => Some(MAX_HAND_PAWNS + MAX_HAND_LANCES + MAX_HAND_KNIGHTS + MAX_HAND_SILVERS + MAX_HAND_GOLDS),
        PieceKind::Rook => Some(MAX_HAND_PAWNS + MAX_HAND_LANCES + MAX_HAND_KNIGHTS + MAX_HAND_SILVERS + MAX_HAND_GOLDS + MAX_HAND_BISHOPS),
        _ => None, // 持ち駒になりえない駒種
    }
}

#[derive(Default)]
pub struct SparseModel {
    pub w: HashMap<usize, f32>,
    pub eta: f32,
}

impl SparseModel {
    pub fn new(eta: f32) -> Self {
        Self {
            w: HashMap::new(),
            eta,
        }
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line?;
            let mut parts = line.split(',');
            if let (Some(k), Some(v)) = (parts.next(), parts.next()) {
                let k: usize = k.parse()?;
                let v: f32 = v.parse()?;
                self.w.insert(k, v);
            }
        }
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path)?;
        for (&k, &v) in &self.w {
            writeln!(file, "{},{}", k, v)?;
        }
        Ok(())
    }

    pub fn initialize_random(&mut self, count: usize, stddev: f32) {
        let mut rng = rand::thread_rng();
        let dist = rand_distr::Normal::new(0.0, stddev).unwrap();
        for _ in 0..count {
            let i = rng.gen_range(0..KPP_DIM);
            let v = dist.sample(&mut rng) as f32;
            self.w.insert(i, v);
        }
    }

    pub fn predict(&self, x: &[usize]) -> f32 {
        x.iter().map(|&i| *self.w.get(&i).unwrap_or(&0.0)).sum()
    }

    pub fn update_batch(&mut self, batch: &[(Vec<usize>, f32)], batch_index: usize) -> f32 {
        let m = batch.len() as f32;
        if m == 0.0 {
            return 0.0;
        }
        let mut total_loss = 0.0;
        for (x, y_true) in batch.iter() {
            let y_pred = self.predict(x);
            let error = y_pred - y_true;
            total_loss += error * error;
            for &i in x {
                let w_i = self.w.entry(i).or_insert(0.0);
                *w_i -= self.eta * error / m;
            }
        }
        let mse = total_loss / m;
        println!("バッチ {}: 平均二乗誤差 = {:.6}", batch_index, mse);
        mse
    }
}

// --- ID生成・特徴量抽出関数 ---

/// 駒の状態（種類、位置、先後など）からユニークなID（0から2181）を生成する
fn piece_to_id(piece: Piece, sq: Option<Square>, hand_index: usize) -> Option<usize> {
    let color_offset = if piece.color() == Color::Black { 0 } else { 1 };

    if let Some(sq) = sq {
        // --- 盤上の駒のID ---
        if let Some(kind_index) = board_kind_to_index(piece.piece_kind()) {
            // ID = (色による大きなオフセット) + (駒種によるオフセット) + マス目
            let id = (color_offset * NUM_BOARD_PIECE_KINDS * NUM_SQUARES)
                + (kind_index * NUM_SQUARES)
                + (sq.index() as usize);
            Some(id)
        } else {
            None // 玉はIDをつけない
        }
    } else {
        // --- 持ち駒のID ---
        // 盤上の駒IDの総数分オフセットを足す
        let board_pieces_total = NUM_BOARD_PIECE_KINDS * NUM_SQUARES * 2;
        if let Some(kind_offset) = hand_kind_to_offset(piece.piece_kind()) {
            // ID = (盤上駒の総数) + (色によるオフセット) + (駒種オフセット) + 持ち駒インデックス
            let id = board_pieces_total
                + (color_offset * NUM_HAND_PIECE_SLOTS_PER_PLAYER)
                + kind_offset
                + hand_index;
            Some(id)
        } else {
            None // 持ち駒になりえない駒種
        }
    }
}

/// KPP (King, Piece, Piece) の特徴量インデックスを抽出する
pub fn extract_kpp_features(pos: &Position) -> Vec<usize> {
    // 0. 自玉（黒玉）の位置を探す
    let king_sq_index = match (1..=81).find_map(|i| {
        let sq = Square::from_u8(i as u8).unwrap(); // This line is safe, as 0..80 are all valid squares
        pos.piece_at(sq).and_then(|p| {
            if p.piece_kind() == PieceKind::King && p.color() == Color::Black {
                Some(i)
            } else {
                None
            }
        })
    }) {
        Some(idx) => idx,
        None => {
            // If the black king is not found, we cannot generate KPP features.
            // Return an empty vector as intended.
            println!("Warning: Black king not found on the board. Skipping this position.");
            return vec![];
        }
    };

    // 1. 全ての駒（玉以外）の状態からIDをリストアップする
    let mut piece_ids = Vec::with_capacity(40);
    // 盤上の駒
    for i in 1..=81 {
        let sq = Square::from_u8(i as u8).unwrap();
        if let Some(piece) = pos.piece_at(sq) {
            if let Some(id) = piece_to_id(piece, Some(sq), 0) {
                piece_ids.push(id);
            }
        }
    }
    // 持ち駒
    for color in [Color::Black, Color::White] {
        for kind in ALL_HAND_PIECES.iter() {
            let count = pos.hand_of_a_player(color).count(*kind).unwrap_or(0);
            for i in 0..count {
                if let Some(id) = piece_to_id(Piece::new(*kind, color), None, i as usize) {
                    piece_ids.push(id);
                }
            }
        }
    }

    // 2. 駒のペアを作り、最終的な特徴量インデックスを計算する
    let mut indices = Vec::with_capacity(piece_ids.len() * piece_ids.len() / 2);
    for i in 0..piece_ids.len() {
        for j in (i + 1)..piece_ids.len() {
            // 2つの駒IDをソートし、組み合わせを一意にする（(歩,金)と(金,歩)を同じと扱う）
            let (id1, id2) = if piece_ids[i] < piece_ids[j] {
                (piece_ids[i], piece_ids[j])
            } else {
                (piece_ids[j], piece_ids[i])
            };

            // 三角行列インデックスを使って、ペアのインデックスを密に計算する
            // これにより、特徴量空間を効率的に使える
            let pair_index = id2 * (id2 - 1) / 2 + id1;

            // 最終的なインデックス = (玉の位置によるオフセット) + ペアのインデックス
            let final_index = king_sq_index as usize * NUM_PIECE_PAIRS + pair_index;
            indices.push(final_index);
        }
    }

    indices
}
