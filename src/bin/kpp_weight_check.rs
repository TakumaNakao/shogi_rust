use std::env;
use std::io::{self};
use shogi_core::{Color};
use std::path::Path;

// evaluationモジュールから公開された関数と定数を使用する
use shogi_ai::evaluation::{self, index_to_kpp_info, SparseModel};

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <weight_file_path>", args[0]);
        return Ok(());
    }

    let weight_file_path = &args[1];
    let mut model = SparseModel::new(0.0, 0.0);
    if let Err(e) = model.load(Path::new(weight_file_path)) {
        eprintln!("Error loading weight file: {}", e);
        return Ok(());
    }

    let weights = &model.w;
    println!("Loaded {} weights.", weights.len());

    let mut indices: Vec<usize> = (0..weights.len()).collect();
    indices.sort_by(|&a, &b| weights[a].total_cmp(&weights[b]));

    // --- Display weight statistics ---
    let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_w = weights.iter().cloned().fold(f32::INFINITY, f32::min);
    let non_zero_count = weights.iter().filter(|&&w| w != 0.0).count();
    let total_count = weights.len();
    let sparsity = (non_zero_count as f32 / total_count as f32) * 100.0;

    println!("
--- 学習完了後の重み統計 ---");
    println!("最大重み: {:.6}", max_w);
    println!("最小重み: {:.6}", min_w);
    println!("非ゼロ要素の割合: {:.4}% ({}/{})", sparsity, non_zero_count, total_count);
    println!("駒得係数: {:.6}", model.material_coeff);
    // --- End of statistics ---

    println!("
--- Top 10 Weights ---");
    for &index in indices.iter().rev().take(10) {
        let weight = weights[index];
        let turn = if index < evaluation::MAX_FEATURES / 2 { Color::Black } else { Color::White };
        if let Some((king_sq, p1k, p1sq, p1hi, p1c, p2k, p2sq, p2hi, p2c)) = index_to_kpp_info(index) {
            println!("Weight: {:.6}, Index: {} (Turn: {:?})", weight, index, turn);
            println!("  King (Normalized): Black at {:?} (Corresponds to White's King at {:?})", king_sq, king_sq.flip());
            println!("  Piece 1: {:?} {:?} at {:?} (Hand: {:?})", p1c, p1k, p1sq, p1hi);
            println!("  Piece 2: {:?} {:?} at {:?} (Hand: {:?})", p2c, p2k, p2sq, p2hi);
            println!("  SFEN (Normalized): {}", evaluation::generate_sfen(king_sq, p1k, p1sq, p1hi, p1c, p2k, p2sq, p2hi, p2c, turn));
        } else {
            println!("Failed to decode index: {}", index);
        }
    }

    println!("
--- Bottom 10 Weights ---");
    for &index in indices.iter().take(10) {
        let weight = weights[index];
        let turn = if index < evaluation::MAX_FEATURES / 2 { Color::Black } else { Color::White };
        if let Some((king_sq, p1k, p1sq, p1hi, p1c, p2k, p2sq, p2hi, p2c)) = index_to_kpp_info(index) {
            println!("Weight: {:.6}, Index: {} (Turn: {:?})", weight, index, turn);
            println!("  King (Normalized): Black at {:?} (Corresponds to White's King at {:?})", king_sq, king_sq.flip());
            println!("  Piece 1: {:?} {:?} at {:?} (Hand: {:?})", p1c, p1k, p1sq, p1hi);
            println!("  Piece 2: {:?} {:?} at {:?} (Hand: {:?})", p2c, p2k, p2sq, p2hi);
            println!("  SFEN (Normalized): {}", evaluation::generate_sfen(king_sq, p1k, p1sq, p1hi, p1c, p2k, p2sq, p2hi, p2c, turn));
        } else {
            println!("Failed to decode index: {}", index);
        }
    }

    Ok(())
}