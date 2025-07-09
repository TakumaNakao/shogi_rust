use anyhow::Result;
use rayon::prelude::*;
use shogi::{Color, Piece, PieceKind, Position, Square, Move};
use shogi_csa::Record;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::fs::File;
use std::io::BufReader;
use std::io::BufRead;

const KPP_DIM: usize = 200_000_000;
const NNZ: usize = 1482;
const BATCH_SIZE: usize = 32;

#[derive(Default)]
struct SparseModel {
    w: HashMap<usize, f32>,
    eta: f32,
}

impl SparseModel {
    fn new(eta: f32) -> Self {
        Self {
            w: HashMap::new(),
            eta,
        }
    }

    fn initialize_random(&mut self, count: usize, stddev: f32) {
        let mut rng = rand::thread_rng();
        let dist = rand_distr::Normal::new(0.0, stddev).unwrap();
        for _ in 0..count {
            let i = rng.gen_range(0..KPP_DIM);
            let v = dist.sample(&mut rng) as f32;
            self.w.insert(i, v);
        }
    }

    fn load(&mut self, path: &Path) -> Result<()> {
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

    fn save(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path)?;
        for (&k, &v) in &self.w {
            writeln!(file, "{},{}", k, v)?;
        }
        Ok(())
    }

    fn predict(&self, x: &[usize]) -> f32 {
        x.iter().map(|&i| *self.w.get(&i).unwrap_or(&0.0)).sum()
    }

    fn update_batch(&mut self, batch: &[(Vec<usize>, f32)]) {
        let m = batch.len() as f32;
        for (x, &y_true) in batch.iter() {
            let y_pred = self.predict(x);
            let error = y_pred - y_true;
            for &i in x {
                let w_i = self.w.entry(i).or_insert(0.0);
                *w_i -= self.eta * error / m;
            }
        }
    }
}

fn encode_piece(piece: Piece, sq: Option<Square>, hand_index: usize) -> usize {
    let base = match sq {
        Some(sq) => sq.to_u8() as usize,
        None => 81 + hand_index,
    };
    let kind = piece.piece_kind().to_u8() as usize;
    let owner = if piece.color().is_black() { 0 } else { 1 };
    owner * 750 + kind * 81 + base
}

fn extract_kpp_features(pos: &Position) -> Vec<usize> {
    let mut pieces = vec![];

    let king_sq = if let Some(sq) = pos.king_square(Color::Black) {
        sq.to_u8() as usize
    } else {
        return vec![];
    };

    for sq in Square::all() {
        if let Some(piece) = pos.piece_at(sq) {
            pieces.push(encode_piece(piece, Some(sq), 0));
        }
    }

    for color in [Color::Black, Color::White] {
        for kind in PieceKind::iter() {
            let count = pos.hand_of(color).num_kind(kind);
            for i in 0..count {
                pieces.push(encode_piece(Piece::new(color, kind), None, i as usize));
            }
        }
    }

    let mut indices = vec![];
    for i in 0..pieces.len() {
        for j in (i + 1)..pieces.len() {
            let idx = king_sq * 1500 * 1500 + pieces[i] * 1500 + pieces[j];
            indices.push(idx);
        }
    }

    indices
}

fn load_csa_dataset(dir: &Path) -> Result<Vec<(Vec<usize>, f32)>> {
    let mut data = vec![];
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().map(|e| e == "csa").unwrap_or(false) {
            let text = fs::read_to_string(&path)?;
            let record = Record::from_str(&text)?;

            let mut pos = Position::startpos();
            for mv in &record.moves {
                if let Move::Normal(m) = mv {
                    let features = extract_kpp_features(&pos);
                    let label = match record.win {
                        Some(Color::Black) => 1.0,
                        Some(Color::White) => -1.0,
                        None => 0.0,
                    };
                    data.push((features, label));
                    pos.make_move(m);
                }
            }
        }
    }
    Ok(data)
}

fn main() -> Result<()> {
    println!("使用する年を入力してください（例: 2017）: ");
    let mut year = String::new();
    io::stdin().read_line(&mut year)?;
    let year = year.trim();

    let data_dir = Path::new(&format!('./csa_files/{}', year));
    let weight_path = Path::new("./weights.csv");

    let dataset = load_csa_dataset(data_dir)?;
    println!("局面数: {}", dataset.len());

    let mut model = SparseModel::new(0.01);

    if weight_path.exists() {
        model.load(weight_path)?;
        println!("重みファイルを読み込みました。");
    } else {
        println!("重みファイルが存在しません。初期化中...");
        model.initialize_random(50_000, 0.01); // 例：50,000点を平均0・標準偏差0.01で初期化
        model.save(weight_path)?;
        println!("初期重みを保存しました。");
    }

    for batch in dataset.chunks(BATCH_SIZE) {
        model.update_batch(batch);
    }

    println!("学習完了。重み数: {}", model.w.len());
    model.save(weight_path)?;
    println!("重みを保存しました: {:?}", weight_path);
    Ok(())
}
