use super::PackedHalfKpPosition;
use crate::evaluation::{HALFKP_HIDDEN, HALFKP_INPUTS};
use crate::training_data::DatasetSplit;
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use shogi_core::PartialPosition;
use shogi_lib::Position;
use shogi_usi_parser::FromUsi;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"HKCL0003";
const VERSION: u32 = 3;
const RECORD_MAGIC: u32 = 0x334c_4348;
const RECORD_COUNT_OFFSET: u64 = 20;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CurriculumHeader {
    pub lineage_id: String,
    pub feature_profile: String,
    pub producer_revision: String,
    pub seed: u64,
    #[serde(default)]
    pub input_sha256: Vec<String>,
    #[serde(default)]
    pub generator_config: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher_model_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher_lineage_id: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum LabelSource {
    ExactMate = 0,
    ExactTactic = 1,
    ExpertMove = 2,
    GameResult = 3,
    SelfSearch = 4,
    SelfPlay = 5,
    SymmetryAnchor = 6,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ProofStatus {
    Unknown = 0,
    NotProven = 1,
    ProvenMate = 2,
    NoForcedMateWithinHorizon = 3,
    ResourceLimit = 4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ProofMode {
    CheckingOnly = 0,
    AllLegal = 1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum PreferenceKind {
    ProvenOutcome = 0,
    ShorterMate = 1,
    LongerDefense = 2,
    ExpertMove = 3,
    Search = 4,
    SymmetryEqual = 5,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpeningStratum(pub u16);

impl OpeningStratum {
    pub const UNKNOWN: Self = Self(u16::MAX);
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProofLabel {
    pub status: ProofStatus,
    pub mate_distance: Option<u16>,
    pub horizon_plies: u16,
    pub solver_mode: ProofMode,
    pub max_nodes: u64,
    pub nodes: u64,
    pub rule_version: u32,
}

#[derive(Clone, Debug)]
pub struct CurriculumPreference {
    pub good_child: PackedHalfKpPosition,
    pub bad_child: PackedHalfKpPosition,
    pub kind: PreferenceKind,
    pub margin_class: u8,
    pub good_move_usi: Option<String>,
    pub bad_move_usi: Option<String>,
}

#[derive(Clone, Debug)]
pub struct CurriculumRecord {
    pub source: LabelSource,
    pub group_id: u64,
    pub position_hash: u64,
    pub family_hash: [u8; 32],
    pub network_input_hash: [u8; 32],
    /// Reconstructable root position for proof replay and search Gates.
    pub root_sfen: String,
    pub split: DatasetSplit,
    pub ply: u16,
    pub phase: u8,
    pub opening_stratum: OpeningStratum,
    pub confidence: f32,
    pub sample_weight: f32,
    pub root: PackedHalfKpPosition,
    /// Loss/draw/win from the root side-to-move perspective.
    pub result: Option<f32>,
    pub proof: Option<ProofLabel>,
    pub preferences: Vec<CurriculumPreference>,
}

pub struct CurriculumWriter {
    writer: BufWriter<File>,
    path: PathBuf,
    temporary: PathBuf,
    records: u64,
    header: CurriculumHeader,
}

impl CurriculumWriter {
    pub fn create(path: &Path, header: CurriculumHeader) -> Result<Self> {
        validate_header(&header)?;
        if path.exists() {
            return Err(anyhow!("refusing to overwrite {}", path.display()));
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let temporary = append_suffix(path, ".tmp");
        if temporary.exists() {
            return Err(anyhow!(
                "temporary output already exists: {}",
                temporary.display()
            ));
        }
        let mut writer = BufWriter::new(File::create(&temporary)?);
        write_header(&mut writer, &header, 0)?;
        Ok(Self {
            writer,
            path: path.to_path_buf(),
            temporary,
            records: 0,
            header,
        })
    }

    pub fn write_record(&mut self, record: &CurriculumRecord) -> Result<()> {
        validate_record(&self.header, record)?;
        write_record(&mut self.writer, record)?;
        self.records += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<u64> {
        self.writer.flush()?;
        self.writer.seek(SeekFrom::Start(RECORD_COUNT_OFFSET))?;
        self.writer.write_all(&self.records.to_le_bytes())?;
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        drop(self.writer);
        std::fs::rename(&self.temporary, &self.path)
            .with_context(|| format!("publish {}", self.path.display()))?;
        Ok(self.records)
    }
}

pub struct CurriculumReader {
    reader: BufReader<File>,
    header: CurriculumHeader,
    remaining: u64,
}

impl CurriculumReader {
    pub fn open(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let (header, records) = read_header(&mut reader)?;
        Ok(Self {
            reader,
            header,
            remaining: records,
        })
    }

    pub fn header(&self) -> &CurriculumHeader {
        &self.header
    }

    pub fn remaining(&self) -> u64 {
        self.remaining
    }

    pub fn read_record(&mut self) -> Result<Option<CurriculumRecord>> {
        if self.remaining == 0 {
            let mut trailing = [0u8; 1];
            return match self.reader.read(&mut trailing)? {
                0 => Ok(None),
                _ => Err(anyhow!("trailing bytes after declared HKCL records")),
            };
        }
        let record = read_record(&mut self.reader)?;
        validate_record(&self.header, &record)?;
        self.remaining -= 1;
        Ok(Some(record))
    }
}

fn validate_header(header: &CurriculumHeader) -> Result<()> {
    if header.lineage_id.is_empty()
        || header.feature_profile.is_empty()
        || header.producer_revision.is_empty()
    {
        return Err(anyhow!("HKCL provenance fields must not be empty"));
    }
    for hash in header
        .input_sha256
        .iter()
        .chain(header.teacher_model_sha256.iter())
    {
        if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(anyhow!("invalid SHA-256 in HKCL provenance"));
        }
    }
    if header.teacher_model_sha256.is_some() != header.teacher_lineage_id.is_some() {
        return Err(anyhow!(
            "teacher model hash and lineage must appear together"
        ));
    }
    if header
        .teacher_lineage_id
        .as_ref()
        .is_some_and(|lineage| lineage != &header.lineage_id)
    {
        return Err(anyhow!(
            "teacher model is outside the dataset scratch lineage"
        ));
    }
    Ok(())
}

fn validate_record(header: &CurriculumHeader, record: &CurriculumRecord) -> Result<()> {
    if record.family_hash == [0; 32] || record.network_input_hash == [0; 32] {
        return Err(anyhow!(
            "HKCL family and network-input hashes must be non-zero"
        ));
    }
    if record.network_input_hash != record.root.network_input_hash() {
        return Err(anyhow!(
            "HKCL network-input hash does not match root features"
        ));
    }
    let parsed = PartialPosition::from_usi(&format!("sfen {}", record.root_sfen))
        .map(Position::new)
        .map_err(|_| anyhow!("invalid HKCL root SFEN"))?;
    let parsed_root = PackedHalfKpPosition::from_position(&parsed)
        .ok_or_else(|| anyhow!("HKCL root SFEN lacks a king"))?;
    if parsed_root != record.root {
        return Err(anyhow!("HKCL root SFEN does not match packed root"));
    }
    let canonical_position_hash = u64::from_le_bytes(
        record.network_input_hash[..8]
            .try_into()
            .expect("SHA-256 prefix width"),
    );
    if record.position_hash != canonical_position_hash {
        return Err(anyhow!("HKCL position hash is not canonical"));
    }
    if !record.confidence.is_finite()
        || !(0.0..=1.0).contains(&record.confidence)
        || !record.sample_weight.is_finite()
        || record.sample_weight <= 0.0
    {
        return Err(anyhow!("invalid HKCL confidence or sample weight"));
    }
    encode_result(record.result)?;
    if record.preferences.len() > u16::MAX as usize {
        return Err(anyhow!("too many HKCL preferences"));
    }
    match record.source {
        LabelSource::ExactMate | LabelSource::ExactTactic => {
            if header.teacher_model_sha256.is_some() {
                return Err(anyhow!("exact labels must not carry a teacher model"));
            }
        }
        LabelSource::SelfSearch if header.teacher_model_sha256.is_none() => {
            return Err(anyhow!(
                "self-search labels require a same-lineage teacher model"
            ));
        }
        _ => {}
    }
    if let Some(proof) = &record.proof {
        if matches!(proof.status, ProofStatus::ProvenMate) != proof.mate_distance.is_some() {
            return Err(anyhow!("mate distance must appear only for proven mate"));
        }
        if proof.rule_version == 0 {
            return Err(anyhow!("proof rule version must be non-zero"));
        }
        if proof.horizon_plies == 0 {
            return Err(anyhow!("proof horizon must be non-zero"));
        }
        if proof.max_nodes == 0 || proof.nodes > proof.max_nodes {
            return Err(anyhow!("invalid proof node bound"));
        }
        if proof
            .mate_distance
            .is_some_and(|distance| distance > proof.horizon_plies)
        {
            return Err(anyhow!("mate distance exceeds proof horizon"));
        }
    } else if matches!(
        record.source,
        LabelSource::ExactMate | LabelSource::ExactTactic
    ) {
        return Err(anyhow!("exact labels require bounded proof metadata"));
    }
    validate_position(&record.root)?;
    for preference in &record.preferences {
        validate_position(&preference.good_child)?;
        validate_position(&preference.bad_child)?;
        validate_preference_witness(record.source, &parsed, preference)?;
    }
    Ok(())
}

fn write_header(writer: &mut impl Write, header: &CurriculumHeader, records: u64) -> Result<()> {
    let metadata = serde_json::to_vec(header)?;
    let metadata_len =
        u32::try_from(metadata.len()).map_err(|_| anyhow!("HKCL header too large"))?;
    writer.write_all(MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;
    writer.write_all(&(HALFKP_HIDDEN as u32).to_le_bytes())?;
    writer.write_all(&(HALFKP_INPUTS as u32).to_le_bytes())?;
    writer.write_all(&records.to_le_bytes())?;
    writer.write_all(&metadata_len.to_le_bytes())?;
    writer.write_all(&metadata)?;
    Ok(())
}

fn read_header(reader: &mut impl Read) -> Result<(CurriculumHeader, u64)> {
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    let version = read_u32(reader)?;
    let hidden = read_u32(reader)? as usize;
    let inputs = read_u32(reader)? as usize;
    let records = read_u64(reader)?;
    let metadata_len = read_u32(reader)? as usize;
    if &magic != MAGIC || version != VERSION || hidden != HALFKP_HIDDEN || inputs != HALFKP_INPUTS {
        return Err(anyhow!("incompatible HKCL dataset header"));
    }
    if metadata_len > 16 * 1024 * 1024 {
        return Err(anyhow!("HKCL metadata is unreasonably large"));
    }
    let mut metadata = vec![0u8; metadata_len];
    reader.read_exact(&mut metadata)?;
    let header = serde_json::from_slice(&metadata)?;
    validate_header(&header)?;
    Ok((header, records))
}

fn write_record(writer: &mut impl Write, record: &CurriculumRecord) -> Result<()> {
    writer.write_all(&RECORD_MAGIC.to_le_bytes())?;
    writer.write_all(&[
        record.source as u8,
        split_code(record.split),
        record.phase,
        0,
    ])?;
    writer.write_all(&record.group_id.to_le_bytes())?;
    writer.write_all(&record.position_hash.to_le_bytes())?;
    writer.write_all(&record.family_hash)?;
    writer.write_all(&record.network_input_hash)?;
    let sfen = record.root_sfen.as_bytes();
    let sfen_len = u16::try_from(sfen.len()).map_err(|_| anyhow!("HKCL root SFEN too long"))?;
    writer.write_all(&sfen_len.to_le_bytes())?;
    writer.write_all(sfen)?;
    writer.write_all(&record.ply.to_le_bytes())?;
    writer.write_all(&record.opening_stratum.0.to_le_bytes())?;
    writer.write_all(&record.confidence.to_le_bytes())?;
    writer.write_all(&record.sample_weight.to_le_bytes())?;
    writer.write_all(&[
        encode_result(record.result)?,
        encode_proof_status(record.proof.as_ref()),
    ])?;
    writer.write_all(&(record.preferences.len() as u16).to_le_bytes())?;
    if let Some(proof) = &record.proof {
        writer.write_all(&proof.mate_distance.unwrap_or(u16::MAX).to_le_bytes())?;
        writer.write_all(&proof.horizon_plies.to_le_bytes())?;
        writer.write_all(&[proof.solver_mode as u8, 0, 0, 0])?;
        writer.write_all(&proof.max_nodes.to_le_bytes())?;
        writer.write_all(&proof.nodes.to_le_bytes())?;
        writer.write_all(&proof.rule_version.to_le_bytes())?;
    }
    write_position(writer, &record.root)?;
    for preference in &record.preferences {
        writer.write_all(&[preference.kind as u8, preference.margin_class, 0, 0])?;
        write_optional_short_string(writer, preference.good_move_usi.as_deref())?;
        write_optional_short_string(writer, preference.bad_move_usi.as_deref())?;
        write_position(writer, &preference.good_child)?;
        write_position(writer, &preference.bad_child)?;
    }
    Ok(())
}

fn read_record(reader: &mut impl Read) -> Result<CurriculumRecord> {
    if read_u32(reader)? != RECORD_MAGIC {
        return Err(anyhow!("invalid HKCL record marker"));
    }
    let source = decode_source(read_u8(reader)?)?;
    let split = decode_split(read_u8(reader)?)?;
    let phase = read_u8(reader)?;
    let _reserved = read_u8(reader)?;
    let group_id = read_u64(reader)?;
    let position_hash = read_u64(reader)?;
    let mut family_hash = [0u8; 32];
    reader.read_exact(&mut family_hash)?;
    let mut network_input_hash = [0u8; 32];
    reader.read_exact(&mut network_input_hash)?;
    let sfen_len = read_u16(reader)? as usize;
    let mut sfen = vec![0; sfen_len];
    reader.read_exact(&mut sfen)?;
    let root_sfen = String::from_utf8(sfen).map_err(|_| anyhow!("invalid UTF-8 in HKCL SFEN"))?;
    let ply = read_u16(reader)?;
    let opening_stratum = OpeningStratum(read_u16(reader)?);
    let confidence = read_f32(reader)?;
    let sample_weight = read_f32(reader)?;
    let result = decode_result(read_u8(reader)?)?;
    let proof_status = read_u8(reader)?;
    let preference_count = read_u16(reader)? as usize;
    let proof = if proof_status == u8::MAX {
        None
    } else {
        let status = decode_proof_status(proof_status)?;
        let mate_distance = read_u16(reader)?;
        let horizon_plies = read_u16(reader)?;
        let solver_mode = decode_proof_mode(read_u8(reader)?)?;
        let mut reserved = [0; 3];
        reader.read_exact(&mut reserved)?;
        let max_nodes = read_u64(reader)?;
        let nodes = read_u64(reader)?;
        Some(ProofLabel {
            status,
            mate_distance: (mate_distance != u16::MAX).then_some(mate_distance),
            horizon_plies,
            solver_mode,
            max_nodes,
            nodes,
            rule_version: read_u32(reader)?,
        })
    };
    let root = read_position(reader)?;
    let mut preferences = Vec::with_capacity(preference_count);
    for _ in 0..preference_count {
        let kind = decode_preference(read_u8(reader)?)?;
        let margin_class = read_u8(reader)?;
        let _reserved = read_u16(reader)?;
        let good_move_usi = read_optional_short_string(reader)?;
        let bad_move_usi = read_optional_short_string(reader)?;
        preferences.push(CurriculumPreference {
            good_child: read_position(reader)?,
            bad_child: read_position(reader)?,
            kind,
            margin_class,
            good_move_usi,
            bad_move_usi,
        });
    }
    Ok(CurriculumRecord {
        source,
        group_id,
        position_hash,
        family_hash,
        network_input_hash,
        root_sfen,
        split,
        ply,
        phase,
        opening_stratum,
        confidence,
        sample_weight,
        root,
        result,
        proof,
        preferences,
    })
}

fn write_position(writer: &mut impl Write, position: &PackedHalfKpPosition) -> Result<()> {
    validate_position(position)?;
    writer.write_all(&[
        u8::from(position.side_to_move == shogi_core::Color::White),
        position.features_black.len() as u8,
        position.features_white.len() as u8,
        0,
    ])?;
    writer.write_all(&position.material_black.to_le_bytes())?;
    writer.write_all(&position.material_white.to_le_bytes())?;
    for feature in position
        .features_black
        .iter()
        .chain(&position.features_white)
    {
        writer.write_all(&feature.to_le_bytes())?;
    }
    Ok(())
}

fn read_position(reader: &mut impl Read) -> Result<PackedHalfKpPosition> {
    let side = read_u8(reader)?;
    let black_len = read_u8(reader)? as usize;
    let white_len = read_u8(reader)? as usize;
    let _reserved = read_u8(reader)?;
    let material_black = read_f32(reader)?;
    let material_white = read_f32(reader)?;
    let mut features_black = Vec::with_capacity(black_len);
    let mut features_white = Vec::with_capacity(white_len);
    for index in 0..black_len + white_len {
        let feature = read_u32(reader)?;
        if index < black_len {
            features_black.push(feature);
        } else {
            features_white.push(feature);
        }
    }
    let position = PackedHalfKpPosition {
        side_to_move: match side {
            0 => shogi_core::Color::Black,
            1 => shogi_core::Color::White,
            _ => return Err(anyhow!("invalid HKCL side")),
        },
        features_black,
        features_white,
        material_black,
        material_white,
    };
    validate_position(&position)?;
    Ok(position)
}

fn validate_position(position: &PackedHalfKpPosition) -> Result<()> {
    if position.features_black.len() > u8::MAX as usize
        || position.features_white.len() > u8::MAX as usize
        || !position.material_black.is_finite()
        || !position.material_white.is_finite()
        || position
            .features_black
            .iter()
            .chain(&position.features_white)
            .any(|&feature| feature as usize >= HALFKP_INPUTS)
    {
        return Err(anyhow!("invalid packed HalfKP position in HKCL record"));
    }
    Ok(())
}

fn validate_preference_witness(
    source: LabelSource,
    root: &Position,
    preference: &CurriculumPreference,
) -> Result<()> {
    let exact = matches!(source, LabelSource::ExactMate | LabelSource::ExactTactic);
    if exact && (preference.good_move_usi.is_none() || preference.bad_move_usi.is_none()) {
        return Err(anyhow!("exact preference requires move witnesses"));
    }
    if preference.kind == PreferenceKind::SymmetryEqual
        && (preference.good_move_usi.is_some() || preference.bad_move_usi.is_some())
    {
        return Err(anyhow!("symmetry preference must not carry move witnesses"));
    }
    let (Some(good_text), Some(bad_text)) = (
        preference.good_move_usi.as_deref(),
        preference.bad_move_usi.as_deref(),
    ) else {
        return Ok(());
    };
    let good = crate::utils::parse_usi_move_for_color(good_text, root.side_to_move())
        .ok_or_else(|| anyhow!("invalid good-move witness"))?;
    let bad = crate::utils::parse_usi_move_for_color(bad_text, root.side_to_move())
        .ok_or_else(|| anyhow!("invalid bad-move witness"))?;
    let legal = root.legal_moves();
    if good == bad || !legal.contains(&good) || !legal.contains(&bad) {
        return Err(anyhow!(
            "preference move witnesses are not distinct legal moves"
        ));
    }
    let mut good_position = root.clone();
    good_position.do_move(good);
    let mut bad_position = root.clone();
    bad_position.do_move(bad);
    if PackedHalfKpPosition::from_position(&good_position).as_ref() != Some(&preference.good_child)
        || PackedHalfKpPosition::from_position(&bad_position).as_ref()
            != Some(&preference.bad_child)
    {
        return Err(anyhow!(
            "preference move witness does not match packed child"
        ));
    }
    Ok(())
}

fn write_optional_short_string(writer: &mut impl Write, value: Option<&str>) -> Result<()> {
    match value {
        None => writer.write_all(&[u8::MAX])?,
        Some(value) => {
            let bytes = value.as_bytes();
            let len = u8::try_from(bytes.len()).map_err(|_| anyhow!("HKCL witness too long"))?;
            if len == u8::MAX {
                return Err(anyhow!("HKCL witness too long"));
            }
            writer.write_all(&[len])?;
            writer.write_all(bytes)?;
        }
    }
    Ok(())
}

fn read_optional_short_string(reader: &mut impl Read) -> Result<Option<String>> {
    let len = read_u8(reader)?;
    if len == u8::MAX {
        return Ok(None);
    }
    let mut bytes = vec![0; len as usize];
    reader.read_exact(&mut bytes)?;
    Ok(Some(
        String::from_utf8(bytes).map_err(|_| anyhow!("invalid UTF-8 in HKCL witness"))?,
    ))
}

fn encode_result(value: Option<f32>) -> Result<u8> {
    match value {
        None => Ok(255),
        Some(0.0) => Ok(0),
        Some(0.5) => Ok(1),
        Some(1.0) => Ok(2),
        _ => Err(anyhow!("result must be loss, draw, win, or unknown")),
    }
}
fn decode_result(value: u8) -> Result<Option<f32>> {
    match value {
        0 => Ok(Some(0.0)),
        1 => Ok(Some(0.5)),
        2 => Ok(Some(1.0)),
        255 => Ok(None),
        _ => Err(anyhow!("invalid HKCL result")),
    }
}
fn encode_proof_status(value: Option<&ProofLabel>) -> u8 {
    value.map_or(u8::MAX, |proof| proof.status as u8)
}
fn decode_source(value: u8) -> Result<LabelSource> {
    match value {
        0 => Ok(LabelSource::ExactMate),
        1 => Ok(LabelSource::ExactTactic),
        2 => Ok(LabelSource::ExpertMove),
        3 => Ok(LabelSource::GameResult),
        4 => Ok(LabelSource::SelfSearch),
        5 => Ok(LabelSource::SelfPlay),
        6 => Ok(LabelSource::SymmetryAnchor),
        _ => Err(anyhow!("invalid HKCL label source")),
    }
}
fn decode_proof_status(value: u8) -> Result<ProofStatus> {
    match value {
        0 => Ok(ProofStatus::Unknown),
        1 => Ok(ProofStatus::NotProven),
        2 => Ok(ProofStatus::ProvenMate),
        3 => Ok(ProofStatus::NoForcedMateWithinHorizon),
        4 => Ok(ProofStatus::ResourceLimit),
        _ => Err(anyhow!("invalid HKCL proof status")),
    }
}
fn decode_proof_mode(value: u8) -> Result<ProofMode> {
    match value {
        0 => Ok(ProofMode::CheckingOnly),
        1 => Ok(ProofMode::AllLegal),
        _ => Err(anyhow!("invalid HKCL proof mode")),
    }
}
fn decode_preference(value: u8) -> Result<PreferenceKind> {
    match value {
        0 => Ok(PreferenceKind::ProvenOutcome),
        1 => Ok(PreferenceKind::ShorterMate),
        2 => Ok(PreferenceKind::LongerDefense),
        3 => Ok(PreferenceKind::ExpertMove),
        4 => Ok(PreferenceKind::Search),
        5 => Ok(PreferenceKind::SymmetryEqual),
        _ => Err(anyhow!("invalid HKCL preference kind")),
    }
}
fn split_code(value: DatasetSplit) -> u8 {
    match value {
        DatasetSplit::Train => 0,
        DatasetSplit::Valid => 1,
        DatasetSplit::Test => 2,
    }
}
fn decode_split(value: u8) -> Result<DatasetSplit> {
    match value {
        0 => Ok(DatasetSplit::Train),
        1 => Ok(DatasetSplit::Valid),
        2 => Ok(DatasetSplit::Test),
        _ => Err(anyhow!("invalid HKCL split")),
    }
}
fn append_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}
fn read_u8(reader: &mut impl Read) -> Result<u8> {
    let mut bytes = [0; 1];
    reader.read_exact(&mut bytes)?;
    Ok(bytes[0])
}
fn read_u16(reader: &mut impl Read) -> Result<u16> {
    let mut bytes = [0; 2];
    reader.read_exact(&mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}
fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}
fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}
fn read_f32(reader: &mut impl Read) -> Result<f32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_core::Color;

    fn header() -> CurriculumHeader {
        CurriculumHeader {
            lineage_id: "halfkp64-scratch-v1".into(),
            feature_profile: "halfkp64-mirror-file-v1".into(),
            producer_revision: "test-revision".into(),
            seed: 7,
            input_sha256: vec![],
            generator_config: serde_json::json!({"kind":"fixture"}),
            teacher_model_sha256: None,
            teacher_lineage_id: None,
        }
    }
    fn record() -> CurriculumRecord {
        let source = Position::default();
        let root = PackedHalfKpPosition::from_position(&source).expect("start features");
        let good_move = crate::utils::parse_usi_move_for_color("7g7f", Color::Black).unwrap();
        let bad_move = crate::utils::parse_usi_move_for_color("2g2f", Color::Black).unwrap();
        let mut good = source.clone();
        good.do_move(good_move);
        let mut bad = source.clone();
        bad.do_move(bad_move);
        CurriculumRecord {
            source: LabelSource::ExactMate,
            group_id: 11,
            position_hash: u64::from_le_bytes(
                root.network_input_hash()[..8]
                    .try_into()
                    .expect("hash width"),
            ),
            family_hash: [13; 32],
            network_input_hash: root.network_input_hash(),
            root_sfen: source.to_sfen_owned(),
            split: DatasetSplit::Train,
            ply: 5,
            phase: 0,
            opening_stratum: OpeningStratum::UNKNOWN,
            confidence: 1.0,
            sample_weight: 0.5,
            root,
            result: Some(1.0),
            proof: Some(ProofLabel {
                status: ProofStatus::ProvenMate,
                mate_distance: Some(3),
                horizon_plies: 3,
                solver_mode: ProofMode::CheckingOnly,
                max_nodes: 100,
                nodes: 99,
                rule_version: 1,
            }),
            preferences: vec![CurriculumPreference {
                good_child: PackedHalfKpPosition::from_position(&good).unwrap(),
                bad_child: PackedHalfKpPosition::from_position(&bad).unwrap(),
                kind: PreferenceKind::ProvenOutcome,
                margin_class: 2,
                good_move_usi: Some("7g7f".into()),
                bad_move_usi: Some("2g2f".into()),
            }],
        }
    }

    #[test]
    fn header_has_stable_golden_prefix() {
        let mut bytes = Vec::new();
        write_header(&mut bytes, &header(), 3).expect("header");
        assert_eq!(
            &bytes[..32],
            b"HKCL0003\x03\0\0\0\x40\0\0\0\x08\x9c\x01\0\x03\0\0\0\0\0\0\0\xb5\0\0\0"
        );
    }

    #[test]
    fn record_round_trip_and_corruption_rejection() {
        let mut bytes = Vec::new();
        write_record(&mut bytes, &record()).expect("write");
        let decoded = read_record(&mut bytes.as_slice()).expect("read");
        assert_eq!(decoded.source, LabelSource::ExactMate);
        assert_eq!(
            decoded.proof.expect("proof").status,
            ProofStatus::ProvenMate
        );
        bytes[0] ^= 1;
        assert!(read_record(&mut bytes.as_slice()).is_err());
    }

    #[test]
    fn exact_and_self_search_provenance_are_enforced() {
        let mut with_teacher = header();
        with_teacher.teacher_model_sha256 = Some("a".repeat(64));
        with_teacher.teacher_lineage_id = Some(with_teacher.lineage_id.clone());
        assert!(validate_record(&with_teacher, &record()).is_err());
        let mut self_search = record();
        self_search.source = LabelSource::SelfSearch;
        assert!(validate_record(&header(), &self_search).is_err());
    }

    #[test]
    fn network_hash_and_bounded_proof_semantics_are_enforced() {
        let mut invalid = record();
        invalid.network_input_hash = [99; 32];
        assert!(validate_record(&header(), &invalid).is_err());

        let mut invalid = record();
        invalid.proof.as_mut().expect("proof").horizon_plies = 2;
        assert!(validate_record(&header(), &invalid).is_err());

        let mut invalid = record();
        invalid.proof = None;
        assert!(validate_record(&header(), &invalid).is_err());
    }

    #[test]
    fn file_writer_patches_count_and_reader_checks_trailing_bytes() {
        let path = std::env::temp_dir().join(format!(
            "shogi-ai-hkcl-{}-{}.bin",
            std::process::id(),
            HALFKP_HIDDEN
        ));
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(append_suffix(&path, ".tmp")).ok();
        let mut writer = CurriculumWriter::create(&path, header()).expect("writer");
        writer.write_record(&record()).expect("record");
        assert_eq!(writer.finish().expect("finish"), 1);
        let mut reader = CurriculumReader::open(&path).expect("reader");
        assert_eq!(reader.remaining(), 1);
        assert!(reader.read_record().expect("first").is_some());
        assert!(reader.read_record().expect("eof").is_none());
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("append");
        file.write_all(&[0]).expect("trailing byte");
        drop(file);
        let mut reader = CurriculumReader::open(&path).expect("reader with trailing");
        reader.read_record().expect("record before trailing");
        assert!(reader.read_record().is_err());
        std::fs::remove_file(path).ok();
    }
}
