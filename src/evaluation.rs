#![allow(dead_code)]
mod codec;
mod constants;
mod debug;
mod evaluator;
mod facade;
mod features;
mod halfkp;
mod kernels;
mod sparse;
mod tiny_nnue;

pub use codec::{HalfKpHeader, HALFKP_HEADER_LEN};
pub use constants::*;
pub use debug::{
    generate_sfen, index_to_kpp_info, is_promoted_piece_kind, piece_kind_to_sfen_char_base, KppInfo,
};
pub use evaluator::Evaluator;
pub use facade::{EngineEvaluator, HybridNnueEvaluator, SparseModelEvaluator};
pub use features::{
    extract_halfkp_features_for, extract_kpp_features, extract_kpp_features_and_material,
    extract_nnue_features, HalfKpFeatures, NnueFeatures,
};
pub use halfkp::{HalfKpAccumulator, HalfKpModel, HalfKpSearchContext};
pub use sparse::{calculate_material_advantage, SparseModel};
pub use tiny_nnue::TinyNnueModel;

#[cfg(test)]
mod tests {
    use super::codec::HalfKpFeatureRow;
    use super::kernels::halfkp_avx2_available;
    use super::*;
    use shogi_core::Color;
    use std::fs::File;
    use std::io::Write;

    fn decode_hex_fixture(contents: &str) -> Vec<u8> {
        contents
            .split_whitespace()
            .map(|byte| u8::from_str_radix(byte, 16).expect("valid fixture byte"))
            .collect()
    }

    fn write_test_bytes(label: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "shogi-ai-{label}-{}-{}.bin",
            std::process::id(),
            HALFKP_HIDDEN
        ));
        std::fs::write(&path, bytes).expect("write test fixture");
        path
    }

    fn assert_halfkp_load_error(label: &str, bytes: &[u8], expected: Option<&str>) {
        let path = write_test_bytes(label, bytes);
        let error = HalfKpModel::load(&path).expect_err("damaged HalfKP file must be rejected");
        std::fs::remove_file(path).ok();
        if let Some(expected) = expected {
            assert!(
                error.to_string().contains(expected),
                "expected error containing {expected:?}, got {error:#}"
            );
        }
    }

    fn test_feature_rows<F>(mut value: F) -> Box<[HalfKpFeatureRow]>
    where
        F: FnMut(usize) -> f32,
    {
        (0..HALFKP_INPUTS)
            .map(|row_index| {
                HalfKpFeatureRow(std::array::from_fn(|h| {
                    value(row_index * HALFKP_HIDDEN + h)
                }))
            })
            .collect()
    }

    #[test]
    fn nnue_features_are_in_declared_ranges() {
        let position = shogi_lib::Position::default();
        let features = extract_nnue_features(&position).expect("initial position has both kings");

        assert!(features.king_bucket < NNUE_NUM_KING_BUCKETS);
        assert!(!features.features.is_empty());
        assert!(features
            .features
            .iter()
            .all(|&feature| feature < NNUE_NUM_FEATURES));
        assert!(features.features.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn tiny_nnue_model_loads_binary_and_evaluates() {
        let path = std::env::temp_dir().join(format!(
            "tiny_nnue_model_loads_binary_and_evaluates_{}.bin",
            std::process::id()
        ));
        let mut file = File::create(&path).expect("create tiny nnue test file");
        file.write_all(b"TNNUE001").unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&(NNUE_NUM_FEATURES as u32).to_le_bytes())
            .unwrap();
        file.write_all(&(NNUE_NUM_KING_BUCKETS as u32).to_le_bytes())
            .unwrap();
        file.write_all(&1000.0f32.to_le_bytes()).unwrap();

        for _ in 0..NNUE_NUM_FEATURES {
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
        }
        for _ in 0..NNUE_NUM_KING_BUCKETS {
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
        }
        file.write_all(&0.0f32.to_le_bytes()).unwrap();
        file.write_all(&0.5f32.to_le_bytes()).unwrap();
        file.write_all(&2.0f32.to_le_bytes()).unwrap();
        file.write_all(&0.25f32.to_le_bytes()).unwrap();
        drop(file);

        let model = TinyNnueModel::load(&path).expect("load tiny nnue model");
        let score = model.predict_from_position(&shogi_lib::Position::default());
        std::fs::remove_file(&path).ok();

        assert_eq!(model.hidden, 1);
        assert_eq!(score, 1250.0);
    }

    #[test]
    fn halfkp_features_are_compact_and_sorted() {
        let position = shogi_lib::Position::default();
        for perspective in [Color::Black, Color::White] {
            let features = extract_halfkp_features_for(&position, perspective)
                .expect("initial position has both kings");
            assert!(features.king_bucket < HALFKP_KING_BUCKETS);
            assert!(!features.features.is_empty());
            assert!(features
                .features
                .iter()
                .all(|&feature| feature < HALFKP_INPUTS));
            assert!(features.features.windows(2).all(|pair| pair[0] < pair[1]));
        }
    }

    #[test]
    fn halfkp_v1_header_matches_golden_fixture_and_rejects_damage() {
        let fixture = if cfg!(feature = "halfkp64") {
            include_str!("../tests/fixtures/halfkp/header_v1_halfkp64.hex")
        } else {
            include_str!("../tests/fixtures/halfkp/header_v1_halfkp32.hex")
        };
        let golden = decode_hex_fixture(fixture);
        let header = HalfKpHeader::current(1000.0).expect("valid current header");
        assert_eq!(HALFKP_HEADER_LEN, golden.len());
        assert_eq!(header.encode().expect("encode header"), golden.as_slice());
        assert_eq!(
            header,
            HalfKpHeader::decode(&golden).expect("decode header")
        );

        assert_halfkp_load_error("halfkp-header-only", &golden, None);
        for &length in &[0, 7, 8, 11, 12, 27, 31] {
            assert_halfkp_load_error(
                &format!("halfkp-truncated-header-{length}"),
                &golden[..length],
                None,
            );
        }

        let mut invalid_magic = golden.clone();
        invalid_magic[0] ^= 0xff;
        assert_halfkp_load_error(
            "halfkp-invalid-magic",
            &invalid_magic,
            Some("invalid HalfKP magic"),
        );

        let mut invalid_version = golden.clone();
        invalid_version[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert_halfkp_load_error(
            "halfkp-invalid-version",
            &invalid_version,
            Some("invalid HalfKP header"),
        );

        let mut invalid_hidden = golden.clone();
        invalid_hidden[12..16].copy_from_slice(&0u32.to_le_bytes());
        assert_halfkp_load_error(
            "halfkp-invalid-hidden",
            &invalid_hidden,
            Some("invalid HalfKP header"),
        );

        let mut invalid_inputs = golden.clone();
        invalid_inputs[16..20].copy_from_slice(&0u32.to_le_bytes());
        assert_halfkp_load_error(
            "halfkp-invalid-inputs",
            &invalid_inputs,
            Some("invalid HalfKP header"),
        );

        let mut invalid_scale = golden;
        invalid_scale[28..32].copy_from_slice(&f32::NAN.to_le_bytes());
        assert_halfkp_load_error(
            "halfkp-invalid-scale",
            &invalid_scale,
            Some("invalid HalfKP header"),
        );
    }

    #[test]
    fn halfkp_model_loads_binary_and_evaluates() {
        let path = std::env::temp_dir().join(format!(
            "halfkp_model_loads_binary_and_evaluates_{}.bin",
            std::process::id()
        ));
        let mut file = File::create(&path).expect("create HalfKP test file");
        file.write_all(HalfKpModel::MAGIC).unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&(HALFKP_HIDDEN as u32).to_le_bytes())
            .unwrap();
        file.write_all(&(HALFKP_INPUTS as u32).to_le_bytes())
            .unwrap();
        file.write_all(&(HALFKP_KING_BUCKETS as u32).to_le_bytes())
            .unwrap();
        file.write_all(&(HALFKP_PIECE_STATES as u32).to_le_bytes())
            .unwrap();
        file.write_all(&1000.0f32.to_le_bytes()).unwrap();
        for _ in 0..HALFKP_INPUTS * HALFKP_HIDDEN {
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
        }
        for _ in 0..HALFKP_HIDDEN {
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
        }
        for _ in 0..(HALFKP_HIDDEN * 2 + 1) {
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
        }
        file.write_all(&0.0f32.to_le_bytes()).unwrap();
        drop(file);

        let model = HalfKpModel::load(&path).expect("load HalfKP model");
        let position = shogi_lib::Position::default();
        let score = model.predict_from_position(&position);
        let black = model
            .accumulator_for_position(&position, Color::Black)
            .expect("black accumulator");
        let white = model
            .accumulator_for_position(&position, Color::White)
            .expect("white accumulator");
        let accumulated_score = model.evaluate_accumulators(&position, &black, &white);

        let mut trailing_file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("reopen HalfKP model");
        trailing_file
            .write_all(&[0xff])
            .expect("append trailing byte");
        drop(trailing_file);
        let trailing_error =
            HalfKpModel::load(&path).expect_err("trailing model byte must be rejected");
        assert!(
            trailing_error
                .to_string()
                .contains("trailing bytes in HalfKP file"),
            "unexpected error: {trailing_error:#}"
        );

        std::fs::remove_file(&path).ok();
        assert_eq!(model.feature_emb.len(), HALFKP_INPUTS);
        assert_eq!(score, 0.0);
        assert_eq!(score, accumulated_score);
    }

    #[test]
    fn halfkp_accumulator_feature_delta_matches_refresh() {
        let position = shogi_lib::Position::default();
        let mv = position
            .legal_moves()
            .into_iter()
            .find(|mv| !matches!(mv, shogi_core::Move::Normal { from, .. } if from.index() == 0))
            .expect("initial position has a legal move");
        let mut after = position.clone();
        after.do_move(mv);

        let model = HalfKpModel {
            target_scale: 1000.0,
            feature_emb: test_feature_rows(|i| (i as f32 % 13.0) * 0.001),
            hidden_b: [0.25; HALFKP_HIDDEN],
            out_w: [0.1; HALFKP_HIDDEN * 2 + 1],
            out_b: 0.0,
            use_avx2: false,
        };
        for perspective in [Color::Black, Color::White] {
            let mut delta = model
                .accumulator_for_position(&position, perspective)
                .unwrap();
            let refreshed = model.accumulator_for_position(&after, perspective).unwrap();
            if delta.king_bucket == refreshed.king_bucket {
                assert!(model.update_accumulator_from_positions(&mut delta, &position, &after));
                for (actual, expected) in delta.hidden.iter().zip(refreshed.hidden.iter()) {
                    assert!((actual - expected).abs() < 1e-5);
                }
                assert!((delta.material - refreshed.material).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn halfkp_search_context_move_matches_refresh() {
        let position = shogi_lib::Position::default();
        let mv = position.legal_moves()[0];
        let mut after = position.clone();
        after.do_move(mv);
        let model = HalfKpModel {
            target_scale: 1000.0,
            feature_emb: test_feature_rows(|i| ((i % 97) as f32) * 0.0001),
            hidden_b: std::array::from_fn(|i| i as f32 * 0.001),
            out_w: std::array::from_fn(|i| i as f32 * 0.01),
            out_b: 0.0,
            use_avx2: false,
        };
        let mut ctx = model.begin_search_context(&position).unwrap();
        model.prepare_search_context(&mut ctx, &position, mv);
        model.commit_search_context_move(&mut ctx, &after);
        let black = model
            .accumulator_for_position(&after, Color::Black)
            .unwrap();
        let white = model
            .accumulator_for_position(&after, Color::White)
            .unwrap();
        let frame = &ctx.frames[ctx.ply];
        for (a, b) in frame.black.hidden.iter().zip(black.hidden.iter()) {
            assert!((a - b).abs() < 1e-5, "black {} {}", a, b);
        }
        for (a, b) in frame.white.hidden.iter().zip(white.hidden.iter()) {
            assert!((a - b).abs() < 1e-5, "white {} {}", a, b);
        }
        model.undo_search_context(&mut ctx);
        let root_black = model
            .accumulator_for_position(&position, Color::Black)
            .unwrap();
        let frame = &ctx.frames[ctx.ply];
        assert!(frame
            .black
            .hidden
            .iter()
            .zip(root_black.hidden.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5));
    }

    #[test]
    fn halfkp_avx2_kernel_matches_portable_kernel() {
        if !halfkp_avx2_available() {
            return;
        }
        let position = shogi_lib::Position::default();
        let mv = position.legal_moves()[0];
        let mut after = position.clone();
        after.do_move(mv);
        let portable = HalfKpModel {
            target_scale: 1000.0,
            feature_emb: test_feature_rows(|i| ((i % 53) as f32 - 20.0) * 0.0003),
            hidden_b: std::array::from_fn(|i| i as f32 * 0.002),
            out_w: std::array::from_fn(|i| i as f32 * 0.01),
            out_b: -0.25,
            use_avx2: false,
        };
        let mut avx2 = portable.clone();
        avx2.use_avx2 = true;
        for perspective in [Color::Black, Color::White] {
            let a = portable
                .accumulator_for_position(&after, perspective)
                .unwrap();
            let b = avx2.accumulator_for_position(&after, perspective).unwrap();
            assert!(a
                .hidden
                .iter()
                .zip(b.hidden.iter())
                .all(|(x, y)| (x - y).abs() < 1e-5));
        }
        let mut portable_ctx = portable.begin_search_context(&position).unwrap();
        let mut avx2_ctx = avx2.begin_search_context(&position).unwrap();
        portable.prepare_search_context(&mut portable_ctx, &position, mv);
        avx2.prepare_search_context(&mut avx2_ctx, &position, mv);
        portable.commit_search_context_move(&mut portable_ctx, &after);
        avx2.commit_search_context_move(&mut avx2_ctx, &after);
        let portable_frame = &portable_ctx.frames[portable_ctx.ply];
        let avx2_frame = &avx2_ctx.frames[avx2_ctx.ply];
        assert!(portable_frame
            .black
            .hidden
            .iter()
            .zip(avx2_frame.black.hidden.iter())
            .all(|(x, y)| (x - y).abs() < 1e-5));
        assert!(portable_frame
            .white
            .hidden
            .iter()
            .zip(avx2_frame.white.hidden.iter())
            .all(|(x, y)| (x - y).abs() < 1e-5));
        assert!(
            (portable.predict_from_position(&after) - avx2.predict_from_position(&after)).abs()
                < 1e-4
        );
    }
}
