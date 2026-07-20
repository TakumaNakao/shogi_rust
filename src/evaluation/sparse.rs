use super::constants::{
    piece_kind_value, ALL_HAND_PIECES, BOARD_SQUARES, MAX_FEATURES, NUM_PIECE_PAIRS,
};
use super::features::{extract_kpp_features, piece_to_id};
use anyhow::Result;
use rand::prelude::*;
use rand_distr::Distribution;
use rayon::prelude::*;
use shogi_core::{Color, Move, Piece, PieceKind};
use shogi_lib::Position;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub fn calculate_material_advantage(pos: &Position) -> f32 {
    let mut material = 0.0;
    let turn = pos.side_to_move();

    for &sq in BOARD_SQUARES.iter() {
        if let Some(piece) = pos.piece_at(sq) {
            let value = piece_kind_value(piece.piece_kind());
            if piece.color() == turn {
                material += value;
            } else {
                material -= value;
            }
        }
    }

    for color in [Color::Black, Color::White] {
        for &kind in ALL_HAND_PIECES.iter() {
            let count = pos.hand(color).count(kind).unwrap_or(0) as f32;
            let value = piece_kind_value(kind);
            if color == turn {
                material += count * value;
            } else {
                material -= count * value;
            }
        }
    }
    material
}

#[derive(Default, Clone)]
pub struct SparseModel {
    pub w: Vec<f32>,
    pub bias: f32,
    pub material_coeff: f32,
    pub kpp_eta: f32,
    pub l2_lambda: f32,
}

impl SparseModel {
    pub fn new(kpp_eta: f32, l2_lambda: f32) -> Self {
        Self {
            w: vec![0.0; MAX_FEATURES],
            bias: 0.0,
            material_coeff: 1.0,
            kpp_eta,
            l2_lambda,
        }
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let mut offset = 0;
        self.bias = f32::from_le_bytes(buffer[offset..offset + 4].try_into()?);
        offset += 4;
        self.material_coeff = f32::from_le_bytes(buffer[offset..offset + 4].try_into()?);
        offset += 4;

        let expected_w_bytes = MAX_FEATURES * 4;
        if buffer.len() - offset != expected_w_bytes {
            return Err(anyhow::anyhow!(
                "File size mismatch for weights. Expected {} bytes, got {}.",
                expected_w_bytes + 8,
                buffer.len()
            ));
        }

        for i in 0..MAX_FEATURES {
            let start = offset + i * 4;
            let end = start + 4;
            if end > buffer.len() {
                return Err(anyhow::anyhow!(
                    "Unexpected end of file while reading weights."
                ));
            }
            self.w[i] = f32::from_le_bytes(buffer[start..end].try_into()?);
        }
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path)?;
        let mut buffer = Vec::new();
        buffer.extend_from_slice(&self.bias.to_le_bytes());
        buffer.extend_from_slice(&self.material_coeff.to_le_bytes());
        for &value in &self.w {
            buffer.extend_from_slice(&value.to_le_bytes());
        }
        file.write_all(&buffer)?;

        println!(
            "Max W: {:?}, Material Coeff: {}",
            self.w
                .iter()
                .cloned()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
            self.material_coeff
        );
        Ok(())
    }

    pub fn initialize_random(&mut self, count: usize, stddev: f32) {
        let mut rng = rand::thread_rng();
        let dist = rand_distr::Normal::new(0.0, stddev).unwrap();
        for _ in 0..count {
            let i = rng.gen_range(0..MAX_FEATURES);
            self.w[i] = dist.sample(&mut rng) as f32;
        }
        self.material_coeff = dist.sample(&mut rng) as f32;
    }

    pub fn zero_weight_overwrite(&mut self, overwrite_value: f32) {
        for i in 0..MAX_FEATURES {
            if self.w[i] == 0.0 {
                self.w[i] = overwrite_value;
            }
        }
    }

    pub fn predict(&self, pos: &Position, kpp_features: &[usize]) -> f32 {
        let material = calculate_material_advantage(pos);
        self.predict_with_material(kpp_features, material)
    }

    pub fn predict_with_material(&self, kpp_features: &[usize], material: f32) -> f32 {
        let mut prediction = self.bias;
        for &index in kpp_features {
            if index < MAX_FEATURES {
                prediction += self.w[index];
            }
        }
        prediction + self.material_coeff * material
    }

    pub fn predict_from_position(&self, pos: &Position) -> f32 {
        let turn = pos.side_to_move();
        let mut material = 0.0;
        let mut piece_ids = Vec::with_capacity(40);
        let mut king_sq = None;

        for &sq in BOARD_SQUARES.iter() {
            if let Some(piece) = pos.piece_at(sq) {
                let value = piece_kind_value(piece.piece_kind());
                if piece.color() == turn {
                    material += value;
                } else {
                    material -= value;
                }
                if piece.piece_kind() == PieceKind::King && piece.color() == turn {
                    king_sq = Some(sq);
                    continue;
                }
                if let Some(id) = piece_to_id(piece, Some(sq), 0, turn) {
                    piece_ids.push(id);
                }
            }
        }

        let king_sq = match king_sq {
            Some(sq) => sq,
            None => {
                println!(
                    "Warning: King not found for side {:?}. Skipping this position.",
                    turn
                );
                return 0.0;
            }
        };
        let normalized_king_sq = if turn == Color::Black {
            king_sq
        } else {
            king_sq.flip()
        };
        let king_sq_index = (normalized_king_sq.index() - 1) as usize;

        for color in [Color::Black, Color::White] {
            for kind in ALL_HAND_PIECES.iter() {
                let count = pos.hand(color).count(*kind).unwrap_or(0);
                let value = piece_kind_value(*kind);
                if color == turn {
                    material += count as f32 * value;
                } else {
                    material -= count as f32 * value;
                }
                for i in 0..count {
                    if let Some(id) = piece_to_id(Piece::new(*kind, color), None, i as usize, turn)
                    {
                        piece_ids.push(id);
                    }
                }
            }
        }

        piece_ids.sort_unstable();
        let mut prediction = self.bias;
        for i in 0..piece_ids.len() {
            for j in (i + 1)..piece_ids.len() {
                let id1 = piece_ids[i];
                let id2 = piece_ids[j];
                let pair_index = id2 * (id2 - 1) / 2 + id1;
                prediction += self.w[king_sq_index * NUM_PIECE_PAIRS + pair_index];
            }
        }
        prediction + self.material_coeff * material
    }

    pub fn update_batch_for_moves(&mut self, batch: &[(Position, Move)]) -> (usize, usize) {
        let total_samples = batch.len();
        if total_samples == 0 {
            return (0, 0);
        }

        let results: Vec<(bool, HashMap<usize, f32>, f32, f32)> = batch
            .par_iter()
            .map(|(pos, teacher_move)| {
                let legal_moves = pos.legal_moves();
                if legal_moves.is_empty() {
                    return (false, HashMap::new(), 0.0, 0.0);
                }

                let mut best_move_by_model = None;
                let mut max_score = -f32::INFINITY;
                let mut best_model_features = Vec::new();
                let mut best_model_material = 0.0;
                let mut teacher_move_features = None;
                let mut teacher_material = 0.0;

                for &mv in legal_moves.iter() {
                    let mut temp_pos = pos.clone();
                    temp_pos.do_move(mv);
                    temp_pos.switch_turn();
                    let features = extract_kpp_features(&temp_pos);
                    let material = calculate_material_advantage(&temp_pos);
                    let score = self.predict(&temp_pos, &features);

                    if score > max_score {
                        max_score = score;
                        best_move_by_model = Some(mv);
                        best_model_features = features.clone();
                        best_model_material = material;
                    }
                    if mv == *teacher_move {
                        teacher_move_features = Some(features);
                        teacher_material = material;
                    }
                }

                let mut sparse_grads = HashMap::new();
                let mut is_correct = false;
                if let Some(model_move) = best_move_by_model {
                    if model_move == *teacher_move {
                        is_correct = true;
                    } else if let Some(teacher_features) = teacher_move_features {
                        let teacher_set: HashSet<_> = teacher_features.into_iter().collect();
                        let model_set: HashSet<_> = best_model_features.into_iter().collect();
                        for &idx in teacher_set.difference(&model_set) {
                            *sparse_grads.entry(idx).or_insert(0.0) += self.kpp_eta;
                        }
                        for &idx in model_set.difference(&teacher_set) {
                            *sparse_grads.entry(idx).or_insert(0.0) -= self.kpp_eta;
                        }
                    }
                }
                (
                    is_correct,
                    sparse_grads,
                    teacher_material,
                    best_model_material,
                )
            })
            .collect();

        let mut correct_predictions = 0;
        let mut w_grads = HashMap::new();
        let mut material_grad = 0.0;
        for (is_correct, sparse_grad, teacher_material, model_material) in results {
            if is_correct {
                correct_predictions += 1;
            } else {
                material_grad += self.kpp_eta * (teacher_material - model_material);
            }
            for (idx, gradient) in sparse_grad {
                *w_grads.entry(idx).or_insert(0.0) += gradient;
            }
        }

        let decay_factor = 1.0 - self.kpp_eta * self.l2_lambda;
        for weight in &mut self.w {
            *weight *= decay_factor;
        }
        for (index, gradient) in w_grads {
            self.w[index] += gradient / total_samples as f32;
        }
        self.material_coeff += material_grad / total_samples as f32
            - self.kpp_eta * self.l2_lambda * self.material_coeff;

        (correct_predictions, total_samples)
    }

    pub fn update_batch_with_cross_entropy(&mut self, batch: &[(Position, Move)]) -> (f32, usize) {
        self.update_batch_with_cross_entropy_temperature(batch, 600.0)
    }

    pub fn update_batch_with_cross_entropy_temperature(
        &mut self,
        batch: &[(Position, Move)],
        softmax_temperature: f32,
    ) -> (f32, usize) {
        if batch.is_empty() || !softmax_temperature.is_finite() || softmax_temperature <= 0.0 {
            return (0.0, 0);
        }

        let results: Vec<(HashMap<usize, f32>, f32, f32, bool)> = batch
            .par_iter()
            .map(|(pos, teacher_move)| {
                let legal_moves = pos.legal_moves();
                if legal_moves.is_empty() {
                    return (HashMap::new(), 0.0, 0.0, false);
                }

                let move_data: Vec<_> = legal_moves
                    .iter()
                    .map(|&mv| {
                        let mut temp_pos = pos.clone();
                        temp_pos.do_move(mv);
                        temp_pos.switch_turn();
                        let features = extract_kpp_features(&temp_pos);
                        let material = calculate_material_advantage(&temp_pos);
                        let score = self.predict(&temp_pos, &features);
                        (mv, features, material, score)
                    })
                    .collect();

                let max_score = move_data
                    .iter()
                    .map(|data| data.3)
                    .fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = move_data
                    .iter()
                    .map(|data| ((data.3 - max_score) / softmax_temperature).exp())
                    .collect();
                let total_score = exp_scores.iter().sum::<f32>();
                let mut sparse_grads = HashMap::new();
                let mut material_grad = 0.0;
                let mut loss = 0.0;

                if move_data.iter().any(|(mv, _, _, _)| mv == teacher_move) {
                    for (index, (mv, features, material, _)) in move_data.iter().enumerate() {
                        let probability = exp_scores[index] / total_score;
                        if mv == teacher_move {
                            loss = -probability.max(1e-7).ln();
                        }
                        let delta = if mv == teacher_move {
                            probability - 1.0
                        } else {
                            probability
                        } / softmax_temperature;
                        for &feature in features {
                            *sparse_grads.entry(feature).or_insert(0.0) += delta;
                        }
                        material_grad += delta * *material;
                    }
                    return (sparse_grads, material_grad, loss, true);
                }
                (sparse_grads, material_grad, 0.0, false)
            })
            .collect();

        let mut w_grads = HashMap::new();
        let mut material_grad_total = 0.0;
        let mut loss = 0.0;
        let mut valid_samples = 0;
        for (sparse_grad, material_grad, sample_loss, teacher_found) in results {
            if teacher_found {
                valid_samples += 1;
                loss += sample_loss;
                for (index, gradient) in sparse_grad {
                    *w_grads.entry(index).or_insert(0.0) += gradient;
                }
                material_grad_total += material_grad;
            }
        }
        if valid_samples == 0 {
            return (0.0, 0);
        }

        let avg_loss = loss / valid_samples as f32;
        for (index, gradient) in w_grads {
            self.w[index] -=
                self.kpp_eta * (gradient / valid_samples as f32 + self.l2_lambda * self.w[index]);
        }
        self.material_coeff -= self.kpp_eta
            * (material_grad_total / valid_samples as f32 + self.l2_lambda * self.material_coeff);
        (avg_loss, valid_samples)
    }
}
