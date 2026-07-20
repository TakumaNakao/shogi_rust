mod codec;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use shogi_core::Color;
use shogi_lib::Position;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::evaluation::extract_halfkp_features_for;
use crate::training_data::{artifact_metadata, sha256_hex, ArtifactMetadata, RunEnvironment};

pub const CANDIDATE_SEARCH_BEST: u8 = 1;
pub const CANDIDATE_GAME_MOVE: u8 = 2;
pub const CANDIDATE_RANDOM: u8 = 4;
pub const CANDIDATE_TACTICAL: u8 = 8;
pub const SEARCH_TEACHER_SEMANTICS_VERSION: u32 = 3;
pub const SEARCH_TEACHER_SEMANTICS_ID: &str = "jsa-complete-check-interval-v1";

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct SearchTeacherManifest {
    pub schema_version: u32,
    #[serde(default)]
    pub stage: String,
    pub format: String,
    pub teacher_semantics_version: u32,
    pub teacher_semantics_id: String,
    pub records: u64,
    #[serde(default)]
    pub stage_fingerprint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<RunEnvironment>,
    #[serde(default)]
    pub inputs: Vec<ArtifactMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ArtifactMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_binary: Option<ArtifactMetadata>,
    #[serde(default)]
    pub feature_profile: String,
    #[serde(default)]
    pub search_limits: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jobs: Option<usize>,
    #[serde(default)]
    pub random_seeds: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_policy_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split_policy_version: Option<u32>,
    #[serde(default)]
    pub parent_manifest_sha256: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<ArtifactMetadata>,
}

#[derive(Clone, Debug, Default)]
pub struct SearchTeacherProvenance {
    pub environment: Option<RunEnvironment>,
    pub inputs: Vec<ArtifactMetadata>,
    pub model: Option<ArtifactMetadata>,
    pub engine_binary: Option<ArtifactMetadata>,
    pub feature_profile: String,
    pub search_limits: serde_json::Value,
    pub jobs: Option<usize>,
    pub random_seeds: Vec<u64>,
    pub phase_policy_version: Option<u32>,
    pub split_policy_version: Option<u32>,
    pub parent_manifest_sha256: Vec<String>,
}

impl SearchTeacherProvenance {
    pub fn stage_fingerprint(&self) -> Result<String> {
        let input_content = self
            .inputs
            .iter()
            .map(|artifact| (&artifact.sha256, artifact.records))
            .collect::<Vec<_>>();
        let value = serde_json::json!({
            "schema_version": 1,
            "stage": "halfkp_search_teacher",
            "inputs": input_content,
            "model_sha256": self.model.as_ref().map(|artifact| &artifact.sha256),
            "engine_binary_sha256": self.engine_binary.as_ref().map(|artifact| &artifact.sha256),
            "feature_profile": self.feature_profile,
            "search_limits": self.search_limits,
            "jobs": self.jobs,
            "random_seeds": self.random_seeds,
            "phase_policy_version": self.phase_policy_version,
            "split_policy_version": self.split_policy_version,
            "teacher_semantics_version": SEARCH_TEACHER_SEMANTICS_VERSION,
            "teacher_semantics_id": SEARCH_TEACHER_SEMANTICS_ID,
            "parent_manifest_sha256": self.parent_manifest_sha256,
        });
        Ok(sha256_hex(&serde_json::to_vec(&value)?))
    }
}

pub fn search_teacher_manifest_path(dataset: &Path) -> PathBuf {
    let mut path = dataset.as_os_str().to_os_string();
    path.push(".manifest.json");
    PathBuf::from(path)
}

pub fn read_search_teacher_manifest(dataset: &Path) -> Result<Option<SearchTeacherManifest>> {
    let path = search_teacher_manifest_path(dataset);
    match std::fs::read(&path) {
        Ok(bytes) => Ok(Some(
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?,
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("read {}", path.display())),
    }
}

#[derive(Clone, Debug)]
pub struct PackedHalfKpPosition {
    pub side_to_move: Color,
    pub features_black: Vec<u32>,
    pub features_white: Vec<u32>,
    pub material_black: f32,
    pub material_white: f32,
}

impl PackedHalfKpPosition {
    pub fn from_position(position: &Position) -> Option<Self> {
        let black = extract_halfkp_features_for(position, Color::Black)?;
        let white = extract_halfkp_features_for(position, Color::White)?;
        Some(Self {
            side_to_move: position.side_to_move(),
            features_black: black
                .features
                .into_iter()
                .map(|value| value as u32)
                .collect(),
            features_white: white
                .features
                .into_iter()
                .map(|value| value as u32)
                .collect(),
            material_black: black.material,
            material_white: white.material,
        })
    }
}

#[derive(Clone, Debug)]
pub struct SearchTeacherCandidate {
    pub flags: u8,
    pub score_cp: f32,
    pub child: PackedHalfKpPosition,
}

#[derive(Clone, Debug)]
pub struct SearchTeacherRecord {
    pub position_hash: u64,
    pub ply: u16,
    pub phase: u8,
    pub teacher_depth: u8,
    pub result: Option<f32>,
    pub root_search_score_cp: f32,
    pub sample_weight: f32,
    pub root: PackedHalfKpPosition,
    pub candidates: Vec<SearchTeacherCandidate>,
}

pub struct SearchTeacherWriter {
    writer: BufWriter<File>,
    dataset_path: PathBuf,
    temporary_path: PathBuf,
    records: u64,
    provenance: SearchTeacherProvenance,
}

impl SearchTeacherWriter {
    pub fn create(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        let temporary_path = append_suffix(path, ".tmp");
        let mut writer = BufWriter::new(
            File::create(&temporary_path)
                .with_context(|| format!("create {}", temporary_path.display()))?,
        );
        codec::write_header(&mut writer)?;
        Ok(Self {
            writer,
            dataset_path: path.to_path_buf(),
            temporary_path,
            records: 0,
            provenance: SearchTeacherProvenance::default(),
        })
    }

    pub fn set_provenance(&mut self, provenance: SearchTeacherProvenance) {
        self.provenance = provenance;
    }

    pub fn write_record(&mut self, record: &SearchTeacherRecord) -> Result<()> {
        codec::write_record(&mut self.writer, record)?;
        self.records += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<u64> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        let records = self.records;
        let dataset_path = self.dataset_path;
        let temporary_path = self.temporary_path;
        let provenance = self.provenance;
        drop(self.writer);
        replace_file(&temporary_path, &dataset_path)?;
        let output = artifact_metadata(&dataset_path, Some(records))?;
        let stage_fingerprint = provenance.stage_fingerprint()?;
        let manifest = SearchTeacherManifest {
            schema_version: 2,
            stage: "halfkp_search_teacher".to_string(),
            format: "HKST0002".to_string(),
            teacher_semantics_version: SEARCH_TEACHER_SEMANTICS_VERSION,
            teacher_semantics_id: SEARCH_TEACHER_SEMANTICS_ID.to_string(),
            records,
            stage_fingerprint,
            environment: provenance.environment,
            inputs: provenance.inputs,
            model: provenance.model,
            engine_binary: provenance.engine_binary,
            feature_profile: provenance.feature_profile,
            search_limits: provenance.search_limits,
            jobs: provenance.jobs,
            random_seeds: provenance.random_seeds,
            phase_policy_version: provenance.phase_policy_version,
            split_policy_version: provenance.split_policy_version,
            parent_manifest_sha256: provenance.parent_manifest_sha256,
            output: Some(output),
        };
        let path = search_teacher_manifest_path(&dataset_path);
        let temporary_manifest = append_suffix(&path, ".tmp");
        let bytes = serde_json::to_vec_pretty(&manifest)?;
        std::fs::write(&temporary_manifest, bytes)
            .with_context(|| format!("write {}", temporary_manifest.display()))?;
        replace_file(&temporary_manifest, &path)?;
        Ok(records)
    }
}

fn append_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

fn replace_file(source: &Path, destination: &Path) -> Result<()> {
    match std::fs::rename(source, destination) {
        Ok(()) => Ok(()),
        Err(error)
            if destination.exists()
                && matches!(
                    error.kind(),
                    std::io::ErrorKind::AlreadyExists | std::io::ErrorKind::PermissionDenied
                ) =>
        {
            std::fs::remove_file(destination)?;
            std::fs::rename(source, destination)?;
            Ok(())
        }
        Err(error) => Err(error).with_context(|| {
            format!(
                "replace {} with {}",
                destination.display(),
                source.display()
            )
        }),
    }
}

pub struct SearchTeacherReader {
    reader: BufReader<File>,
}

impl SearchTeacherReader {
    pub fn open(path: &Path) -> Result<Self> {
        let mut reader =
            BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
        codec::read_header(&mut reader)?;
        Ok(Self { reader })
    }

    pub fn read_record(&mut self) -> Result<Option<SearchTeacherRecord>> {
        codec::read_record(&mut self.reader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::{HALFKP_HIDDEN, HALFKP_INPUTS};
    use crate::position_hash::PositionHasher;

    fn decode_hex_fixture(contents: &str) -> Vec<u8> {
        contents
            .split_whitespace()
            .map(|byte| u8::from_str_radix(byte, 16).expect("valid fixture byte"))
            .collect()
    }

    fn golden_teacher_bytes() -> Vec<u8> {
        let fixture = if cfg!(feature = "halfkp64") {
            include_str!("../tests/fixtures/teacher/hkst_v2_halfkp64.hex")
        } else {
            include_str!("../tests/fixtures/teacher/hkst_v2_halfkp32.hex")
        };
        decode_hex_fixture(fixture)
    }

    fn teacher_fixture_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "shogi-ai-{label}-{}-{}.hkst",
            std::process::id(),
            HALFKP_HIDDEN
        ))
    }

    fn write_teacher_fixture(label: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = teacher_fixture_path(label);
        std::fs::write(&path, bytes).expect("write teacher fixture");
        path
    }

    fn golden_teacher_record() -> SearchTeacherRecord {
        SearchTeacherRecord {
            position_hash: 0x0123_4567_89ab_cdef,
            ply: 12,
            phase: 1,
            teacher_depth: 4,
            result: Some(0.5),
            root_search_score_cp: 42.25,
            sample_weight: 0.75,
            root: PackedHalfKpPosition {
                side_to_move: Color::Black,
                features_black: vec![0],
                features_white: vec![(HALFKP_INPUTS - 1) as u32],
                material_black: 100.0,
                material_white: -100.0,
            },
            candidates: vec![SearchTeacherCandidate {
                flags: CANDIDATE_SEARCH_BEST | CANDIDATE_GAME_MOVE,
                score_cp: 37.5,
                child: PackedHalfKpPosition {
                    side_to_move: Color::White,
                    features_black: vec![1],
                    features_white: vec![2],
                    material_black: 50.0,
                    material_white: -50.0,
                },
            }],
        }
    }

    #[test]
    fn packed_position_extracts_valid_halfkp_features() {
        let position = Position::default();
        let packed = PackedHalfKpPosition::from_position(&position).expect("start position");
        assert_eq!(Color::Black, packed.side_to_move);
        assert!(!packed.features_black.is_empty());
        assert!(!packed.features_white.is_empty());
        assert!(packed
            .features_black
            .iter()
            .chain(&packed.features_white)
            .all(|&feature| (feature as usize) < HALFKP_INPUTS));
        assert_ne!(0, PositionHasher::calculate_hash(&position));
    }

    #[test]
    fn teacher_dataset_round_trips() {
        let position = Position::default();
        let packed = PackedHalfKpPosition::from_position(&position).expect("start position");
        let record = SearchTeacherRecord {
            position_hash: PositionHasher::calculate_hash(&position),
            ply: 12,
            phase: 1,
            teacher_depth: 4,
            result: Some(0.5),
            root_search_score_cp: 42.25,
            sample_weight: 0.75,
            root: packed.clone(),
            candidates: vec![SearchTeacherCandidate {
                flags: CANDIDATE_SEARCH_BEST | CANDIDATE_GAME_MOVE,
                score_cp: 37.5,
                child: packed,
            }],
        };
        let path = std::env::temp_dir().join(format!(
            "halfkp-search-teacher-{}-{}.hkst",
            std::process::id(),
            record.position_hash
        ));
        let mut writer = SearchTeacherWriter::create(&path).expect("create dataset");
        writer.write_record(&record).expect("write record");
        assert_eq!(1, writer.finish().expect("finish dataset"));
        let manifest = read_search_teacher_manifest(&path)
            .expect("read manifest")
            .expect("manifest exists");
        assert_eq!(2, manifest.schema_version);
        assert_eq!("HKST0002", manifest.format);
        assert_eq!(
            SEARCH_TEACHER_SEMANTICS_VERSION,
            manifest.teacher_semantics_version
        );
        assert_eq!(SEARCH_TEACHER_SEMANTICS_ID, manifest.teacher_semantics_id);
        assert_eq!(1, manifest.records);

        let mut reader = SearchTeacherReader::open(&path).expect("open dataset");
        let decoded = reader
            .read_record()
            .expect("read record")
            .expect("record exists");
        assert_eq!(record.position_hash, decoded.position_hash);
        assert_eq!(record.ply, decoded.ply);
        assert_eq!(record.result, decoded.result);
        assert_eq!(record.root.features_black, decoded.root.features_black);
        assert_eq!(record.candidates[0].flags, decoded.candidates[0].flags);
        assert_eq!(
            record.candidates[0].score_cp,
            decoded.candidates[0].score_cp
        );
        assert!(reader.read_record().expect("read eof").is_none());
        std::fs::remove_file(search_teacher_manifest_path(&path)).expect("remove manifest");
        std::fs::remove_file(path).expect("remove dataset");
    }

    #[test]
    fn teacher_v2_reader_and_writer_match_golden_fixture() {
        let golden = golden_teacher_bytes();
        assert_eq!(94, golden.len());

        let fixture_path = write_teacher_fixture("teacher-v2-golden-read", &golden);
        let mut reader = SearchTeacherReader::open(&fixture_path).expect("open golden dataset");
        let decoded = reader
            .read_record()
            .expect("read golden record")
            .expect("golden record exists");
        assert_eq!(0x0123_4567_89ab_cdef, decoded.position_hash);
        assert_eq!(12, decoded.ply);
        assert_eq!(1, decoded.phase);
        assert_eq!(4, decoded.teacher_depth);
        assert_eq!(Some(0.5), decoded.result);
        assert_eq!(42.25, decoded.root_search_score_cp);
        assert_eq!(0.75, decoded.sample_weight);
        assert_eq!(Color::Black, decoded.root.side_to_move);
        assert_eq!(vec![0], decoded.root.features_black);
        assert_eq!(
            vec![(HALFKP_INPUTS - 1) as u32],
            decoded.root.features_white
        );
        assert_eq!(100.0, decoded.root.material_black);
        assert_eq!(-100.0, decoded.root.material_white);
        assert_eq!(1, decoded.candidates.len());
        assert_eq!(
            CANDIDATE_SEARCH_BEST | CANDIDATE_GAME_MOVE,
            decoded.candidates[0].flags
        );
        assert_eq!(37.5, decoded.candidates[0].score_cp);
        assert_eq!(Color::White, decoded.candidates[0].child.side_to_move);
        assert_eq!(vec![1], decoded.candidates[0].child.features_black);
        assert_eq!(vec![2], decoded.candidates[0].child.features_white);
        assert!(reader.read_record().expect("read golden eof").is_none());
        std::fs::remove_file(fixture_path).expect("remove golden fixture copy");

        let output_path = teacher_fixture_path("teacher-v2-golden-write");
        let mut writer = SearchTeacherWriter::create(&output_path).expect("create output dataset");
        writer
            .write_record(&golden_teacher_record())
            .expect("write golden record");
        assert_eq!(1, writer.finish().expect("finish golden output"));
        let actual = std::fs::read(&output_path).expect("read written golden dataset");
        let manifest = read_search_teacher_manifest(&output_path)
            .expect("read golden manifest")
            .expect("golden manifest exists");
        assert_eq!(
            SEARCH_TEACHER_SEMANTICS_VERSION,
            manifest.teacher_semantics_version
        );
        assert_eq!(1, manifest.records);
        std::fs::remove_file(search_teacher_manifest_path(&output_path))
            .expect("remove golden manifest");
        std::fs::remove_file(output_path).expect("remove golden output");
        assert_eq!(golden, actual);
    }

    #[test]
    fn teacher_v2_rejects_truncated_and_corrupt_data() {
        let golden = golden_teacher_bytes();

        for &length in &[0, 7, 8, 11, 12, 19] {
            let path = write_teacher_fixture(
                &format!("teacher-v2-truncated-header-{length}"),
                &golden[..length],
            );
            assert!(
                SearchTeacherReader::open(&path).is_err(),
                "header length {length} must be rejected"
            );
            std::fs::remove_file(path).ok();
        }

        for &length in &[21, 23, 32, 45, 65, 93] {
            let path = write_teacher_fixture(
                &format!("teacher-v2-truncated-record-{length}"),
                &golden[..length],
            );
            let mut reader = SearchTeacherReader::open(&path).expect("complete header");
            assert!(
                reader.read_record().is_err(),
                "record length {length} must be rejected"
            );
            std::fs::remove_file(path).ok();
        }

        let mut invalid_magic = golden.clone();
        invalid_magic[0] ^= 0xff;
        let path = write_teacher_fixture("teacher-v2-invalid-magic", &invalid_magic);
        assert!(SearchTeacherReader::open(&path).is_err());
        std::fs::remove_file(path).ok();

        let mut invalid_version = golden.clone();
        invalid_version[8..12].copy_from_slice(&3u32.to_le_bytes());
        let path = write_teacher_fixture("teacher-v2-invalid-version", &invalid_version);
        assert!(SearchTeacherReader::open(&path).is_err());
        std::fs::remove_file(path).ok();

        let mut invalid_record_magic = golden.clone();
        invalid_record_magic[20] ^= 0xff;
        let path = write_teacher_fixture("teacher-v2-invalid-record-magic", &invalid_record_magic);
        let mut reader = SearchTeacherReader::open(&path).expect("valid header");
        assert!(reader.read_record().is_err());
        std::fs::remove_file(path).ok();

        let mut invalid_result = golden.clone();
        invalid_result[35] = 3;
        let path = write_teacher_fixture("teacher-v2-invalid-result", &invalid_result);
        let mut reader = SearchTeacherReader::open(&path).expect("valid header");
        assert!(reader.read_record().is_err());
        std::fs::remove_file(path).ok();

        let mut zero_candidates = golden.clone();
        zero_candidates[36] = 0;
        let path = write_teacher_fixture("teacher-v2-zero-candidates", &zero_candidates);
        let mut reader = SearchTeacherReader::open(&path).expect("valid header");
        assert!(reader.read_record().is_err());
        std::fs::remove_file(path).ok();

        let mut invalid_side = golden.clone();
        invalid_side[46] = 2;
        let path = write_teacher_fixture("teacher-v2-invalid-side", &invalid_side);
        let mut reader = SearchTeacherReader::open(&path).expect("valid header");
        assert!(reader.read_record().is_err());
        std::fs::remove_file(path).ok();

        let mut invalid_feature = golden;
        invalid_feature[58..62].copy_from_slice(&(HALFKP_INPUTS as u32).to_le_bytes());
        let path = write_teacher_fixture("teacher-v2-invalid-feature", &invalid_feature);
        let mut reader = SearchTeacherReader::open(&path).expect("valid header");
        assert!(reader.read_record().is_err());
        std::fs::remove_file(path).ok();
    }
}
