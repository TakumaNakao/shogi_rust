use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use shogi_ai::training_data::sha256_file;
use std::collections::HashSet;
use std::path::Path;

pub const SCRATCH_LINEAGE_ID: &str = "halfkp64-scratch-v1";
pub const DEFAULT_PROMOTION_REGISTRY: &str = "training/halfkp64-scratch-v1/promotion_registry.json";

#[derive(Debug, Deserialize)]
pub struct PromotionRegistry {
    schema: String,
    lineage_id: String,
    accepted: Vec<AcceptedModel>,
    #[serde(default)]
    rejected: Vec<RejectedModel>,
}

#[derive(Debug, Deserialize)]
struct AcceptedModel {
    model_sha256: String,
    generation: u32,
    gate_report_sha256: String,
    decision: String,
    accepted_at: String,
    decision_commit: String,
}

#[derive(Debug, Deserialize)]
struct RejectedModel {
    model_sha256: String,
    reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PromotionDecision {
    pub model_sha256: String,
    pub generation: u32,
    pub gate_report_sha256: String,
}

#[derive(Debug, Deserialize)]
struct ScratchModelManifest {
    lineage_id: String,
    generation: u32,
    model: ScratchModelArtifact,
    #[serde(default)]
    parent_model_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ScratchModelArtifact {
    sha256: String,
}

pub fn require_scratch_model_manifest(
    manifest_path: &Path,
    model_path: &Path,
    declared_generation: u32,
) -> Result<String> {
    let manifest: ScratchModelManifest = serde_json::from_slice(
        &std::fs::read(manifest_path)
            .with_context(|| format!("read model manifest {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parse model manifest {}", manifest_path.display()))?;
    let model_sha256 = sha256_file(model_path)?;
    if manifest.lineage_id != SCRATCH_LINEAGE_ID
        || manifest.generation != declared_generation
        || manifest.model.sha256 != model_sha256
    {
        return Err(anyhow!("model manifest does not match scratch artifact"));
    }
    if declared_generation > 0 {
        let parent = manifest
            .parent_model_sha256
            .as_deref()
            .ok_or_else(|| anyhow!("non-genesis model manifest has no parent"))?;
        validate_sha256(parent)?;
        if parent == model_sha256 {
            return Err(anyhow!("scratch model cannot be its own parent"));
        }
    }
    Ok(model_sha256)
}

pub fn require_accepted_parent(
    registry_path: &Path,
    model_path: &Path,
    gate_report_path: &Path,
    declared_generation: u32,
) -> Result<PromotionDecision> {
    let registry: PromotionRegistry = serde_json::from_slice(
        &std::fs::read(registry_path)
            .with_context(|| format!("read promotion registry {}", registry_path.display()))?,
    )
    .with_context(|| format!("parse promotion registry {}", registry_path.display()))?;
    validate_registry(&registry)?;
    let model_sha256 = sha256_file(model_path)?;
    if let Some(rejected) = registry
        .rejected
        .iter()
        .find(|entry| entry.model_sha256 == model_sha256)
    {
        return Err(anyhow!(
            "model {} is explicitly rejected: {}",
            model_path.display(),
            rejected.reason
        ));
    }
    let gate_report_sha256 = sha256_file(gate_report_path)?;
    let accepted = registry
        .accepted
        .iter()
        .find(|entry| {
            entry.model_sha256 == model_sha256
                && entry.gate_report_sha256 == gate_report_sha256
                && entry.generation == declared_generation
        })
        .ok_or_else(|| {
            anyhow!(
                "model is not accepted by {} with this Gate report",
                registry_path.display()
            )
        })?;
    Ok(PromotionDecision {
        model_sha256,
        generation: accepted.generation,
        gate_report_sha256,
    })
}

fn validate_registry(registry: &PromotionRegistry) -> Result<()> {
    if registry.schema != "halfkp_scratch_promotion_registry_v1"
        || registry.lineage_id != SCRATCH_LINEAGE_ID
    {
        return Err(anyhow!("incompatible scratch promotion registry"));
    }
    let mut hashes = HashSet::new();
    for entry in &registry.accepted {
        validate_sha256(&entry.model_sha256)?;
        validate_sha256(&entry.gate_report_sha256)?;
        if entry.decision != "accepted_genesis"
            && entry.decision != "accepted_gate"
            && entry.decision != "accepted_promotion_match"
            && entry.decision != "accepted_internal_scratch_parent"
        {
            return Err(anyhow!("invalid promotion decision"));
        }
        if entry.accepted_at.is_empty() || entry.decision_commit.is_empty() {
            return Err(anyhow!("incomplete promotion audit fields"));
        }
        if !hashes.insert(&entry.model_sha256) {
            return Err(anyhow!("duplicate accepted model hash"));
        }
    }
    for entry in &registry.rejected {
        validate_sha256(&entry.model_sha256)?;
        if entry.reason.is_empty() || hashes.contains(&entry.model_sha256) {
            return Err(anyhow!("invalid rejected model entry"));
        }
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(anyhow!("invalid SHA-256 in promotion registry"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn requires_matching_model_and_gate_report_hashes() {
        let directory = std::env::temp_dir().join(format!(
            "shogi-ai-promotion-registry-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).expect("directory");
        let model = directory.join("model.bin");
        let report = directory.join("report.json");
        let registry = directory.join("registry.json");
        std::fs::write(&model, b"scratch-model").expect("model");
        std::fs::write(&report, b"gate-report").expect("report");
        let model_hash = sha256_file(&model).expect("model hash");
        let report_hash = sha256_file(&report).expect("report hash");
        let value = serde_json::json!({
            "schema": "halfkp_scratch_promotion_registry_v1",
            "lineage_id": SCRATCH_LINEAGE_ID,
            "accepted": [{
                "model_sha256": model_hash,
                "generation": 0,
                "gate_report_sha256": report_hash,
                "decision": "accepted_promotion_match",
                "accepted_at": "2026-07-21T00:00:00+09:00",
                "decision_commit": "test"
            }],
            "rejected": []
        });
        let mut file = std::fs::File::create(&registry).expect("registry");
        serde_json::to_writer_pretty(&mut file, &value).expect("json");
        file.flush().expect("flush");
        assert!(require_accepted_parent(&registry, &model, &report, 0).is_ok());
        std::fs::write(&report, b"changed").expect("changed report");
        assert!(require_accepted_parent(&registry, &model, &report, 0).is_err());
        std::fs::remove_dir_all(directory).ok();
    }

    #[test]
    fn model_manifest_binds_generation_lineage_and_parent() {
        let directory =
            std::env::temp_dir().join(format!("shogi-ai-model-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&directory).expect("directory");
        let model = directory.join("model.bin");
        let manifest = directory.join("model.manifest.json");
        std::fs::write(&model, b"scratch-generation-two").expect("model");
        let model_hash = sha256_file(&model).expect("model hash");
        let value = serde_json::json!({
            "lineage_id": SCRATCH_LINEAGE_ID,
            "generation": 2,
            "parent_model_sha256": "1d773b4f41f5b485cda8fc94dd4b7131328fde332e33e5c6e7382fff33840ea2",
            "model": {"sha256": model_hash}
        });
        std::fs::write(&manifest, serde_json::to_vec(&value).expect("json")).expect("manifest");
        assert!(require_scratch_model_manifest(&manifest, &model, 2).is_ok());
        assert!(require_scratch_model_manifest(&manifest, &model, 1).is_err());
        std::fs::write(&model, b"changed").expect("changed model");
        assert!(require_scratch_model_manifest(&manifest, &model, 2).is_err());
        std::fs::remove_dir_all(directory).ok();
    }
}
