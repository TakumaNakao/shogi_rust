use super::halfkp_search_train::{
    project_center_mirror_sparse_gradient, Forward, Weights, TARGET_SCALE,
};
use anyhow::{anyhow, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use shogi_ai::evaluation::HALFKP_HIDDEN;
#[cfg(test)]
use shogi_ai::evaluation::HALFKP_INPUTS;
use shogi_ai::halfkp_training::PackedHalfKpPosition;
use shogi_ai::training_data::{artifact_metadata, capture_run_environment};
use shogi_ai::training_tools::scratch_lineage::{
    require_accepted_parent, require_scratch_model_manifest,
};
use shogi_ai::utils::{parse_usi_move_for_color, position_from_sfen_or_usi};
use shogi_core::Color;
use shogi_lib::Position;
use std::collections::{BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Train scalar HalfKP with all-legal expert-policy cross entropy")]
struct Args {
    #[arg(long, required = true)]
    train: Vec<PathBuf>,
    #[arg(long, required = true)]
    valid: Vec<PathBuf>,
    #[arg(long)]
    init: PathBuf,
    #[arg(long)]
    model_manifest: PathBuf,
    #[arg(long)]
    promotion_registry: PathBuf,
    #[arg(long)]
    gate_report: PathBuf,
    #[arg(long)]
    parent_generation: u32,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 2)]
    epochs: usize,
    #[arg(long, default_value_t = 128)]
    batch_size: usize,
    #[arg(long, default_value_t = 0.0001)]
    learning_rate: f32,
    #[arg(long, default_value_t = 0.00001)]
    output_learning_rate: f32,
    #[arg(long, default_value_t = 300.0)]
    temperature_cp: f32,
    #[arg(long)]
    max_train_records: Option<usize>,
    #[arg(long)]
    max_valid_records: Option<usize>,
    /// Re-evaluate the fixed training prefix after every epoch for small-set fit checks.
    #[arg(long, default_value_t = false)]
    report_train_metrics: bool,
    #[arg(long, default_value_t = 20260729)]
    seed: u64,
    #[arg(long, default_value_t = 1.0)]
    gradient_clip_norm: f32,
    #[arg(long, default_value_t = 0.20)]
    output_limit: f32,
    #[arg(long)]
    log: Option<PathBuf>,
}

#[derive(Deserialize)]
struct InputRecord {
    schema: String,
    game_key: String,
    opening_family: String,
    split: String,
    side_to_move: String,
    sfen: String,
    teacher_move: String,
    license_id: String,
    source_uri: String,
}

struct Sample {
    root: Position,
    teacher: shogi_core::Move,
}

struct Gradient {
    hidden_b: [f32; HALFKP_HIDDEN],
    out_w: [f32; HALFKP_HIDDEN * 2 + 1],
    out_b: f32,
    sparse: HashMap<usize, [f32; HALFKP_HIDDEN]>,
    records: usize,
}

impl Default for Gradient {
    fn default() -> Self {
        Self {
            hidden_b: [0.0; HALFKP_HIDDEN],
            out_w: [0.0; HALFKP_HIDDEN * 2 + 1],
            out_b: 0.0,
            sparse: HashMap::new(),
            records: 0,
        }
    }
}

struct Metrics {
    records: usize,
    legal_moves: usize,
    loss_sum: f64,
    top1: usize,
    margin_sum_cp: f64,
    improved: usize,
    worsened: usize,
    unchanged: usize,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            records: 0,
            legal_moves: 0,
            loss_sum: 0.0,
            top1: 0,
            margin_sum_cp: 0.0,
            improved: 0,
            worsened: 0,
            unchanged: 0,
        }
    }
}

#[derive(Serialize)]
struct MetricReport {
    records: usize,
    legal_moves: usize,
    mean_legal_moves: f64,
    cross_entropy: f64,
    top1_accuracy: f64,
    mean_teacher_margin_cp: f64,
    margin_improved: usize,
    margin_worsened: usize,
    margin_unchanged: usize,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let accepted = require_accepted_parent(
        &args.promotion_registry,
        &args.init,
        &args.gate_report,
        args.parent_generation,
    )?;
    let manifest_parent =
        require_scratch_model_manifest(&args.model_manifest, &args.init, args.parent_generation)?;
    if accepted.model_sha256 != manifest_parent {
        return Err(anyhow!("accepted parent and model manifest disagree"));
    }
    validate_dataset_splits(&args.train, &args.valid)?;

    let parent = Weights::load(&args.init)?;
    let mut weights = parent.clone();
    let mut adagrad = weights.zeros_like();
    let initial = evaluate(
        &weights,
        &parent,
        &args.valid,
        args.max_valid_records,
        args.temperature_cp,
    )?;
    print_metrics("initial", 0, &initial);
    if args.report_train_metrics {
        let initial_train = evaluate(
            &weights,
            &parent,
            &args.train,
            args.max_train_records,
            args.temperature_cp,
        )?;
        print_metrics("train_fixed", 0, &initial_train);
    }
    let mut log = create_log(args.log.as_deref())?;
    write_log(&mut log, 0, &initial)?;

    let mut best_loss = report(&initial).cross_entropy;
    let mut best_epoch = 0usize;
    weights.save(&args.output)?;
    for epoch in 1..=args.epochs {
        let mut gradient = Gradient::default();
        let mut seen = 0usize;
        let mut loss_sum = 0.0;
        for path in &args.train {
            visit_samples(
                path,
                args.max_train_records.map(|n| n.saturating_sub(seen)),
                |sample| {
                    let loss =
                        add_policy_gradient(&weights, &sample, args.temperature_cp, &mut gradient)?;
                    loss_sum += loss;
                    seen += 1;
                    if gradient.records == args.batch_size {
                        apply_batch(&mut weights, &mut adagrad, &mut gradient, &args);
                    }
                    Ok(())
                },
            )?;
            if args.max_train_records.is_some_and(|limit| seen >= limit) {
                break;
            }
        }
        if gradient.records > 0 {
            apply_batch(&mut weights, &mut adagrad, &mut gradient, &args);
        }
        let valid = evaluate(
            &weights,
            &parent,
            &args.valid,
            args.max_valid_records,
            args.temperature_cp,
        )?;
        if args.report_train_metrics {
            let train_fixed = evaluate(
                &weights,
                &parent,
                &args.train,
                args.max_train_records,
                args.temperature_cp,
            )?;
            print_metrics("train_fixed", epoch, &train_fixed);
        }
        print_metrics("valid", epoch, &valid);
        println!(
            "train epoch={epoch} records={seen} cross_entropy={:.8}",
            loss_sum / seen.max(1) as f64
        );
        write_log(&mut log, epoch, &valid)?;
        let valid_loss = report(&valid).cross_entropy;
        if valid_loss < best_loss {
            best_loss = valid_loss;
            best_epoch = epoch;
            weights.save(&args.output)?;
        }
    }
    write_manifest(&args, &accepted, best_epoch, best_loss, &initial)?;
    Ok(())
}

fn validate_args(args: &Args) -> Result<()> {
    if args.epochs == 0
        || args.batch_size == 0
        || args.learning_rate <= 0.0
        || args.output_learning_rate <= 0.0
        || args.temperature_cp <= 0.0
        || args.gradient_clip_norm <= 0.0
        || args.output_limit <= 0.0
    {
        return Err(anyhow!("invalid training hyperparameter"));
    }
    Ok(())
}

fn validate_dataset_splits(train: &[PathBuf], valid: &[PathBuf]) -> Result<()> {
    let mut train_families = BTreeSet::new();
    let mut valid_families = BTreeSet::new();
    for (paths, expected, families) in [
        (train, "train", &mut train_families),
        (valid, "valid", &mut valid_families),
    ] {
        for path in paths {
            for line in BufReader::new(File::open(path)?).lines() {
                let record: InputRecord = serde_json::from_str(&line?)?;
                validate_record_metadata(&record, expected)?;
                if record.game_key.is_empty() || record.opening_family.is_empty() {
                    return Err(anyhow!("missing WDoor split identity"));
                }
                families.insert(record.opening_family);
            }
        }
    }
    let overlap = train_families.intersection(&valid_families).count();
    if overlap != 0 {
        return Err(anyhow!("train/valid game overlap: {overlap}"));
    }
    println!(
        "dataset_split train_families={} valid_families={} overlap=0",
        train_families.len(),
        valid_families.len()
    );
    Ok(())
}

fn validate_record_metadata(record: &InputRecord, expected_split: &str) -> Result<()> {
    if record.schema != "rank_value_dataset_v3_position"
        || record.split != expected_split
        || record.license_id.is_empty()
        || record.source_uri.is_empty()
    {
        return Err(anyhow!("invalid WDoor policy record provenance"));
    }
    Ok(())
}

fn visit_samples(
    path: &Path,
    limit: Option<usize>,
    mut visitor: impl FnMut(Sample) -> Result<()>,
) -> Result<usize> {
    let mut count = 0usize;
    for line in BufReader::new(File::open(path)?).lines() {
        if limit.is_some_and(|limit| count >= limit) {
            break;
        }
        let record: InputRecord = serde_json::from_str(&line?)?;
        let position =
            position_from_sfen_or_usi(&record.sfen).ok_or_else(|| anyhow!("invalid SFEN"))?;
        let expected_side = if position.side_to_move() == Color::Black {
            "black"
        } else {
            "white"
        };
        if record.side_to_move != expected_side {
            return Err(anyhow!("side-to-move metadata mismatch"));
        }
        let legal = position.legal_moves();
        let teacher = parse_usi_move_for_color(&record.teacher_move, position.side_to_move())
            .filter(|mv| legal.contains(mv))
            .ok_or_else(|| anyhow!("illegal teacher move"))?;
        if !policy_eligible(legal.len()) {
            continue;
        }
        visitor(Sample {
            root: position,
            teacher,
        })?;
        count += 1;
    }
    Ok(count)
}

fn policy_eligible(legal_moves: usize) -> bool {
    legal_moves >= 2
}

fn children_and_utilities(
    weights: &Weights,
    sample: &Sample,
) -> Result<Vec<(shogi_core::Move, PackedHalfKpPosition, Forward, f32)>> {
    sample
        .root
        .legal_moves()
        .into_iter()
        .map(|mv| {
            let mut child = sample.root.clone();
            child.do_move(mv);
            let packed = PackedHalfKpPosition::from_position(&child)
                .ok_or_else(|| anyhow!("legal child lacks king"))?;
            let forward = weights.forward(&packed);
            let utility = -forward.raw * TARGET_SCALE;
            Ok((mv, packed, forward, utility))
        })
        .collect()
}

fn probabilities(utilities: &[f32], temperature_cp: f32) -> Vec<f32> {
    let maximum = utilities.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values = utilities
        .iter()
        .map(|&utility| ((utility - maximum) / temperature_cp).exp())
        .collect::<Vec<_>>();
    let sum = values.iter().sum::<f32>();
    for value in &mut values {
        *value /= sum;
    }
    values
}

fn add_policy_gradient(
    weights: &Weights,
    sample: &Sample,
    temperature_cp: f32,
    gradient: &mut Gradient,
) -> Result<f64> {
    let candidates = children_and_utilities(weights, sample)?;
    let teacher = candidates
        .iter()
        .position(|candidate| candidate.0 == sample.teacher)
        .ok_or_else(|| anyhow!("teacher missing from legal children"))?;
    let utilities = candidates
        .iter()
        .map(|candidate| candidate.3)
        .collect::<Vec<_>>();
    let probabilities = probabilities(&utilities, temperature_cp);
    for (index, (_, position, forward, _)) in candidates.iter().enumerate() {
        let target = f32::from(index == teacher);
        let d_raw = (target - probabilities[index]) * TARGET_SCALE / temperature_cp;
        add_position_gradient(weights, position, forward, d_raw, gradient);
    }
    gradient.records += 1;
    Ok(-f64::from(probabilities[teacher].max(1e-30).ln()))
}

fn add_position_gradient(
    weights: &Weights,
    position: &PackedHalfKpPosition,
    forward: &Forward,
    d_raw: f32,
    gradient: &mut Gradient,
) {
    let (stm, nstm, material) = if position.side_to_move == Color::Black {
        (&forward.black, &forward.white, position.material_black)
    } else {
        (&forward.white, &forward.black, position.material_white)
    };
    let mut stm_hidden = [0.0; HALFKP_HIDDEN];
    let mut nstm_hidden = [0.0; HALFKP_HIDDEN];
    for h in 0..HALFKP_HIDDEN {
        gradient.out_w[h] += d_raw * stm[h].clamp(0.0, 1.0);
        gradient.out_w[HALFKP_HIDDEN + h] += d_raw * nstm[h].clamp(0.0, 1.0);
        stm_hidden[h] = d_raw * weights.out_w[h] * active_derivative(stm[h]);
        nstm_hidden[h] = d_raw * weights.out_w[HALFKP_HIDDEN + h] * active_derivative(nstm[h]);
    }
    gradient.out_w[HALFKP_HIDDEN * 2] += d_raw * material / TARGET_SCALE;
    gradient.out_b += d_raw;
    let (black, white) = if position.side_to_move == Color::Black {
        (stm_hidden, nstm_hidden)
    } else {
        (nstm_hidden, stm_hidden)
    };
    for h in 0..HALFKP_HIDDEN {
        gradient.hidden_b[h] += black[h] + white[h];
    }
    for (&feature, hidden) in position
        .features_black
        .iter()
        .map(|feature| (feature, &black))
        .chain(
            position
                .features_white
                .iter()
                .map(|feature| (feature, &white)),
        )
    {
        let row = gradient
            .sparse
            .entry(feature as usize)
            .or_insert([0.0; HALFKP_HIDDEN]);
        for h in 0..HALFKP_HIDDEN {
            row[h] += hidden[h];
        }
    }
}

fn active_derivative(value: f32) -> f32 {
    f32::from(value > 0.0 && value < 1.0)
}

fn apply_batch(weights: &mut Weights, adagrad: &mut Weights, gradient: &mut Gradient, args: &Args) {
    project_center_mirror_sparse_gradient(&mut gradient.sparse);
    let scale = 1.0 / gradient.records.max(1) as f32;
    let mut norm_squared = (gradient.out_b * scale).powi(2);
    for value in gradient.hidden_b.iter_mut().chain(&mut gradient.out_w) {
        *value *= scale;
        norm_squared += *value * *value;
    }
    gradient.out_b *= scale;
    let mut features = gradient.sparse.keys().copied().collect::<Vec<_>>();
    features.sort_unstable();
    for feature in &features {
        for value in gradient.sparse.get_mut(feature).expect("feature") {
            *value *= scale;
            norm_squared += *value * *value;
        }
    }
    let clip = (args.gradient_clip_norm / norm_squared.sqrt()).min(1.0);
    for h in 0..HALFKP_HIDDEN {
        adagrad_scalar(
            &mut weights.hidden_b[h],
            &mut adagrad.hidden_b[h],
            gradient.hidden_b[h] * clip,
            args.learning_rate,
            1.0,
        );
    }
    for index in 0..weights.out_w.len() {
        adagrad_scalar(
            &mut weights.out_w[index],
            &mut adagrad.out_w[index],
            gradient.out_w[index] * clip,
            args.output_learning_rate,
            args.output_limit,
        );
    }
    // A common bias shifts every legal-child logit equally, so softmax CE is
    // exactly invariant to out_b. Keep it frozen instead of integrating f32
    // cancellation residue into a zero-state AdaGrad accumulator.
    for feature in features {
        let start = feature * HALFKP_HIDDEN;
        for h in 0..HALFKP_HIDDEN {
            adagrad_scalar(
                &mut weights.feature_emb[start + h],
                &mut adagrad.feature_emb[start + h],
                gradient.sparse[&feature][h] * clip,
                args.learning_rate,
                1.0,
            );
        }
    }
    *gradient = Gradient::default();
}

fn adagrad_scalar(
    weight: &mut f32,
    accumulator: &mut f32,
    gradient: f32,
    learning_rate: f32,
    limit: f32,
) {
    if gradient == 0.0 {
        return;
    }
    *accumulator += gradient * gradient;
    *weight =
        (*weight - learning_rate * gradient / (accumulator.sqrt() + 1e-8)).clamp(-limit, limit);
}

fn evaluate(
    weights: &Weights,
    baseline: &Weights,
    paths: &[PathBuf],
    limit: Option<usize>,
    temperature_cp: f32,
) -> Result<Metrics> {
    let mut metrics = Metrics::default();
    for path in paths {
        visit_samples(
            path,
            limit.map(|n| n.saturating_sub(metrics.records)),
            |sample| {
                let candidate = sample_metrics(weights, &sample, temperature_cp)?;
                let parent = sample_metrics(baseline, &sample, temperature_cp)?;
                metrics.records += 1;
                metrics.legal_moves += candidate.0;
                metrics.loss_sum += candidate.1;
                metrics.top1 += candidate.2;
                metrics.margin_sum_cp += f64::from(candidate.3);
                match candidate.3.total_cmp(&parent.3) {
                    std::cmp::Ordering::Greater => metrics.improved += 1,
                    std::cmp::Ordering::Less => metrics.worsened += 1,
                    std::cmp::Ordering::Equal => metrics.unchanged += 1,
                }
                Ok(())
            },
        )?;
        if limit.is_some_and(|limit| metrics.records >= limit) {
            break;
        }
    }
    if metrics.records == 0 {
        return Err(anyhow!("no usable policy records"));
    }
    Ok(metrics)
}

fn sample_metrics(
    weights: &Weights,
    sample: &Sample,
    temperature_cp: f32,
) -> Result<(usize, f64, usize, f32)> {
    let candidates = children_and_utilities(weights, sample)?;
    let teacher = candidates
        .iter()
        .position(|candidate| candidate.0 == sample.teacher)
        .ok_or_else(|| anyhow!("teacher missing"))?;
    let utilities = candidates
        .iter()
        .map(|candidate| candidate.3)
        .collect::<Vec<_>>();
    let probabilities = probabilities(&utilities, temperature_cp);
    let best = utilities
        .iter()
        .enumerate()
        .max_by(|lhs, rhs| lhs.1.total_cmp(rhs.1))
        .map(|(index, _)| index)
        .expect("legal candidates");
    let best_other = utilities
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != teacher)
        .map(|(_, &utility)| utility)
        .fold(f32::NEG_INFINITY, f32::max);
    Ok((
        candidates.len(),
        -f64::from(probabilities[teacher].max(1e-30).ln()),
        usize::from(best == teacher),
        utilities[teacher] - best_other,
    ))
}

fn report(metrics: &Metrics) -> MetricReport {
    let records = metrics.records.max(1) as f64;
    MetricReport {
        records: metrics.records,
        legal_moves: metrics.legal_moves,
        mean_legal_moves: metrics.legal_moves as f64 / records,
        cross_entropy: metrics.loss_sum / records,
        top1_accuracy: metrics.top1 as f64 / records,
        mean_teacher_margin_cp: metrics.margin_sum_cp / records,
        margin_improved: metrics.improved,
        margin_worsened: metrics.worsened,
        margin_unchanged: metrics.unchanged,
    }
}

fn print_metrics(label: &str, epoch: usize, metrics: &Metrics) {
    let value = report(metrics);
    println!(
        "{label} epoch={epoch} records={} legal_mean={:.2} ce={:.8} top1={:.4} margin={:.3}cp paired={}/{}/{}",
        value.records,
        value.mean_legal_moves,
        value.cross_entropy,
        value.top1_accuracy,
        value.mean_teacher_margin_cp,
        value.margin_improved,
        value.margin_worsened,
        value.margin_unchanged
    );
}

fn create_log(path: Option<&Path>) -> Result<Option<BufWriter<File>>> {
    path.map(|path| {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut writer = BufWriter::new(File::create(path)?);
        writeln!(
            writer,
            "epoch,records,mean_legal_moves,cross_entropy,top1_accuracy,mean_teacher_margin_cp,margin_improved,margin_worsened,margin_unchanged"
        )?;
        Ok(writer)
    })
    .transpose()
}

fn write_log(writer: &mut Option<BufWriter<File>>, epoch: usize, metrics: &Metrics) -> Result<()> {
    if let Some(writer) = writer {
        let value = report(metrics);
        writeln!(
            writer,
            "{epoch},{},{:.8},{:.8},{:.8},{:.8},{},{},{}",
            value.records,
            value.mean_legal_moves,
            value.cross_entropy,
            value.top1_accuracy,
            value.mean_teacher_margin_cp,
            value.margin_improved,
            value.margin_worsened,
            value.margin_unchanged
        )?;
        writer.flush()?;
    }
    Ok(())
}

fn write_manifest(
    args: &Args,
    accepted: &shogi_ai::training_tools::scratch_lineage::PromotionDecision,
    best_epoch: usize,
    best_loss: f64,
    initial: &Metrics,
) -> Result<()> {
    let manifest_path = PathBuf::from(format!("{}.manifest.json", args.output.display()));
    let temporary = PathBuf::from(format!("{}.tmp", manifest_path.display()));
    let model = artifact_metadata(&args.output, None)?;
    let value = serde_json::json!({
        "schema_version": 1,
        "lineage_id": "halfkp64-scratch-v1",
        "generation": args.parent_generation + 1,
        "parent_model_sha256": accepted.model_sha256,
        "parent_gate_report_sha256": accepted.gate_report_sha256,
        "stage": "halfkp_policy_train",
        "environment": capture_run_environment(),
        "command": std::env::args().collect::<Vec<_>>(),
        "inputs": {
            "train": args.train.iter().map(|path| artifact_metadata(path, None)).collect::<Result<Vec<_>>>()?,
            "valid": args.valid.iter().map(|path| artifact_metadata(path, None)).collect::<Result<Vec<_>>>()?
        },
        "init_model": artifact_metadata(&args.init, None)?,
        "model": model,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "output_learning_rate": args.output_learning_rate,
            "temperature_cp": args.temperature_cp,
            "max_train_records": args.max_train_records,
            "max_valid_records": args.max_valid_records,
            "report_train_metrics": args.report_train_metrics,
            "seed": args.seed,
            "gradient_clip_norm": args.gradient_clip_norm,
            "output_limit": args.output_limit
        },
        "best_epoch": best_epoch,
        "best_validation_cross_entropy": best_loss,
        "initial_validation": report(initial)
    });
    serde_json::to_writer_pretty(BufWriter::new(File::create(&temporary)?), &value)?;
    fs::rename(temporary, manifest_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(side_to_move: Color) -> (Weights, Sample) {
        let mut root = Position::default();
        if side_to_move == Color::White {
            let move_to_white = root.legal_moves()[0];
            root.do_move(move_to_white);
        }
        let teacher = root.legal_moves()[0];
        let weights = Weights {
            feature_emb: vec![0.0; HALFKP_INPUTS * HALFKP_HIDDEN],
            hidden_b: [0.5; HALFKP_HIDDEN],
            out_w: [0.01; HALFKP_HIDDEN * 2 + 1],
            out_b: 0.0,
        };
        (weights, Sample { root, teacher })
    }

    #[test]
    fn all_legal_ce_gradient_matches_output_finite_difference_for_both_sides() {
        for side_to_move in [Color::Black, Color::White] {
            let (mut weights, sample) = fixture(side_to_move);
            let mut gradient = Gradient::default();
            add_policy_gradient(&weights, &sample, 300.0, &mut gradient).expect("gradient");
            let output_index = if side_to_move == Color::Black {
                0
            } else {
                HALFKP_HIDDEN
            };
            let analytic = gradient.out_w[output_index];
            let epsilon = 1e-3;
            weights.out_w[output_index] += epsilon;
            let plus = sample_metrics(&weights, &sample, 300.0).expect("plus").1;
            weights.out_w[output_index] -= 2.0 * epsilon;
            let minus = sample_metrics(&weights, &sample, 300.0).expect("minus").1;
            weights.out_w[output_index] += epsilon;
            let numerical = ((plus - minus) / f64::from(2.0 * epsilon)) as f32;
            assert!(
                (analytic - numerical).abs() < 2e-4,
                "side={side_to_move:?} analytic={analytic} numerical={numerical}"
            );

            let (&feature, row) = gradient
                .sparse
                .iter()
                .find(|(_, row)| row.iter().any(|value| value.abs() > 1e-6))
                .expect("non-zero sparse gradient");
            let hidden = row
                .iter()
                .position(|value| value.abs() > 1e-6)
                .expect("non-zero hidden component");
            let feature_index = feature * HALFKP_HIDDEN + hidden;
            let analytic = row[hidden];
            weights.feature_emb[feature_index] += epsilon;
            let plus = sample_metrics(&weights, &sample, 300.0).expect("plus").1;
            weights.feature_emb[feature_index] -= 2.0 * epsilon;
            let minus = sample_metrics(&weights, &sample, 300.0).expect("minus").1;
            let numerical = ((plus - minus) / f64::from(2.0 * epsilon)) as f32;
            assert!(
                (analytic - numerical).abs() < 2e-4,
                "side={side_to_move:?} sparse analytic={analytic} numerical={numerical}"
            );

            assert!(
                gradient.out_b.abs() < 1e-4,
                "softmax common-bias gradient={}",
                gradient.out_b
            );
        }
    }

    #[test]
    fn forced_move_positions_are_excluded() {
        assert!(!policy_eligible(0));
        assert!(!policy_eligible(1));
        assert!(policy_eligible(2));
        let (weights, sample) = fixture(Color::Black);
        assert!(sample.root.legal_moves().len() > 1);
        assert!(add_policy_gradient(&weights, &sample, 300.0, &mut Gradient::default()).is_ok());
    }
}
