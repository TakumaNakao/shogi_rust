#!/usr/bin/env python3
"""Train a tiny sparse NNUE-style value model from nnue_feature_dump JSONL.

This is a lightweight prototype intentionally based only on NumPy.  It keeps the
data path usable on machines where PyTorch is not installed, while preserving a
model shape that can later be mirrored by a Rust evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


NNUE_NUM_KING_BUCKETS = 81 * 81
NNUE_NUM_BOARD_FEATURES = 14 * 81 * 2
NNUE_NUM_HAND_FEATURES = (18 + 4 + 4 + 4 + 4 + 2 + 2) * 2
NNUE_NUM_FEATURES = NNUE_NUM_BOARD_FEATURES + NNUE_NUM_HAND_FEATURES


@dataclass
class Sample:
    sfen: str
    king_bucket: int
    features: np.ndarray
    material: float
    target: float
    baseline: float | None
    root_index: int | None
    rank: int | None
    regret: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny NumPy NNUE prototype from JSONL feature dumps."
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--binary-output",
        type=Path,
        help="Optional Rust-friendly binary export path.",
    )
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rank-loss-weight", type=float, default=0.0)
    parser.add_argument("--rank-temperature-cp", type=float, default=20.0)
    parser.add_argument("--target-scale", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument(
        "--target-field",
        choices=["teacher_score", "static_eval"],
        default="teacher_score",
    )
    parser.add_argument(
        "--baseline-field",
        default="static_eval",
        help="Optional JSONL field to compare against the target before training.",
    )
    return parser.parse_args()


def load_jsonl(
    path: Path,
    target_field: str,
    target_scale: float,
    baseline_field: str | None,
) -> list[Sample]:
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if target_field not in record:
                continue
            target = record[target_field]
            if target is None or not math.isfinite(float(target)):
                continue
            king_bucket = int(record["king_bucket"])
            if not 0 <= king_bucket < NNUE_NUM_KING_BUCKETS:
                raise ValueError(f"{path}:{line_number}: king_bucket out of range")
            features = np.asarray(record["features"], dtype=np.int64)
            if features.size == 0:
                continue
            if int(features.min()) < 0 or int(features.max()) >= NNUE_NUM_FEATURES:
                raise ValueError(f"{path}:{line_number}: feature index out of range")
            sfen = record.get("sfen", record.get("child_sfen"))
            if sfen is None:
                continue
            samples.append(
                Sample(
                    sfen=str(sfen),
                    king_bucket=king_bucket,
                    features=features,
                    material=float(record["material"]),
                    target=float(target) / target_scale,
                    baseline=(
                        float(record[baseline_field]) / target_scale
                        if baseline_field and record.get(baseline_field) is not None
                        else None
                    ),
                    root_index=(
                        int(record["root_index"]) if record.get("root_index") is not None else None
                    ),
                    rank=(int(record["rank"]) if record.get("rank") is not None else None),
                    regret=(
                        float(record["regret"])
                        if record.get("regret") is not None
                        else None
                    ),
                )
            )
    if not samples:
        raise ValueError(f"no usable samples loaded from {path}")
    return samples


def init_model(hidden: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    scale = 0.01
    return {
        "feature_emb": rng.normal(0.0, scale, size=(NNUE_NUM_FEATURES, hidden)).astype(
            np.float32
        ),
        "king_emb": rng.normal(0.0, scale, size=(NNUE_NUM_KING_BUCKETS, hidden)).astype(
            np.float32
        ),
        "material_w": rng.normal(0.0, scale, size=(hidden,)).astype(np.float32),
        "hidden_b": np.zeros((hidden,), dtype=np.float32),
        "out_w": rng.normal(0.0, scale, size=(hidden,)).astype(np.float32),
        "out_b": np.zeros((), dtype=np.float32),
    }


def write_binary_model(
    path: Path,
    model: dict[str, np.ndarray],
    hidden: int,
    target_scale: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"TNNUE001")
        f.write(
            struct.pack(
                "<IIIIf",
                1,
                hidden,
                NNUE_NUM_FEATURES,
                NNUE_NUM_KING_BUCKETS,
                target_scale,
            )
        )
        for name in ["feature_emb", "king_emb", "material_w", "hidden_b", "out_w"]:
            array = np.ascontiguousarray(model[name], dtype="<f4")
            f.write(array.tobytes(order="C"))
        f.write(struct.pack("<f", float(model["out_b"])))


def forward(model: dict[str, np.ndarray], sample: Sample) -> tuple[float, np.ndarray, np.ndarray]:
    z = (
        model["feature_emb"][sample.features].sum(axis=0)
        + model["king_emb"][sample.king_bucket]
        + model["material_w"] * sample.material
        + model["hidden_b"]
    )
    hidden = np.clip(z, 0.0, 1.0)
    pred = float(hidden @ model["out_w"] + model["out_b"])
    return pred, z, hidden


def summarize_errors(
    predictions: Iterable[float], samples: list[Sample], target_scale: float
) -> tuple[float, float, float]:
    sq_error = 0.0
    abs_error = 0.0
    sign_matches = 0
    total = 0
    for pred, sample in zip(predictions, samples):
        error = pred - sample.target
        sq_error += error * error
        abs_error += abs(error)
        if (pred >= 0.0) == (sample.target >= 0.0):
            sign_matches += 1
        total += 1
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    rmse = math.sqrt(sq_error / total) * target_scale
    mae = (abs_error / total) * target_scale
    sign_accuracy = sign_matches / total
    return rmse, mae, sign_accuracy


def evaluate(
    model: dict[str, np.ndarray], samples: list[Sample], target_scale: float
) -> tuple[float, float, float]:
    predictions = (forward(model, sample)[0] for sample in samples)
    return summarize_errors(predictions, samples, target_scale)


def evaluate_baseline(samples: list[Sample], target_scale: float) -> tuple[float, float, float] | None:
    if any(sample.baseline is None for sample in samples):
        return None
    return summarize_errors((sample.baseline for sample in samples if sample.baseline is not None), samples, target_scale)


def evaluate_ranking(
    predictions: Iterable[float], samples: list[Sample]
) -> tuple[int, float, float, int] | None:
    root_records: dict[int, list[tuple[float, Sample]]] = defaultdict(list)
    for pred, sample in zip(predictions, samples):
        if sample.root_index is None or sample.rank is None or sample.regret is None:
            return None
        root_records[sample.root_index].append((pred, sample))
    if not root_records:
        return None

    top1_matches = 0
    selected_regret = 0.0
    bad_selected = 0
    for records in root_records.values():
        selected_pred, selected = max(records, key=lambda item: item[0])
        _ = selected_pred
        if selected.rank == 1:
            top1_matches += 1
        selected_regret += selected.regret
        if selected.regret > 300.0:
            bad_selected += 1
    roots = len(root_records)
    return roots, top1_matches / roots, selected_regret / roots, bad_selected


def evaluate_model_ranking(
    model: dict[str, np.ndarray], samples: list[Sample]
) -> tuple[int, float, float, int] | None:
    return evaluate_ranking((forward(model, sample)[0] for sample in samples), samples)


def zero_grads(model: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.zeros_like(value) for name, value in model.items()}


def clone_model(model: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in model.items()}


def add_prediction_grad(
    model: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    sample: Sample,
    dloss: float,
) -> None:
    _pred, z, hidden = forward(model, sample)
    grads["out_w"] += dloss * hidden
    grads["out_b"] += dloss
    dz = dloss * model["out_w"] * ((z > 0.0) & (z < 1.0))
    np.add.at(grads["feature_emb"], sample.features, dz)
    grads["king_emb"][sample.king_bucket] += dz
    grads["material_w"] += dz * sample.material
    grads["hidden_b"] += dz


def grouped_rank_samples(samples: list[Sample]) -> list[list[Sample]]:
    groups: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        if sample.root_index is None:
            return []
        groups[sample.root_index].append(sample)
    return [group for group in groups.values() if len(group) >= 2]


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def apply_rank_loss_epoch(
    model: dict[str, np.ndarray],
    groups: list[list[Sample]],
    moments1: dict[str, np.ndarray],
    moments2: dict[str, np.ndarray],
    step: int,
    lr: float,
    weight_decay: float,
    rank_loss_weight: float,
    rank_temperature: float,
) -> int:
    if rank_loss_weight <= 0.0 or not groups:
        return step
    for group in groups:
        preds = np.asarray([forward(model, sample)[0] for sample in group], dtype=np.float32)
        targets = np.asarray([sample.target for sample in group], dtype=np.float32)
        pred_probs = softmax(preds / rank_temperature)
        target_probs = softmax(targets / rank_temperature)
        pred_grads = rank_loss_weight * (pred_probs - target_probs) / rank_temperature

        grads = zero_grads(model)
        for sample, dloss in zip(group, pred_grads):
            add_prediction_grad(model, grads, sample, float(dloss))
        step += 1
        apply_adam(model, grads, moments1, moments2, step, lr, weight_decay)
    return step


def apply_adam(
    model: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    moments1: dict[str, np.ndarray],
    moments2: dict[str, np.ndarray],
    step: int,
    lr: float,
    weight_decay: float,
) -> None:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for name, param in model.items():
        grad = grads[name]
        if weight_decay > 0.0 and param.ndim > 0:
            grad = grad + weight_decay * param
        moments1[name] = beta1 * moments1[name] + (1.0 - beta1) * grad
        moments2[name] = beta2 * moments2[name] + (1.0 - beta2) * (grad * grad)
        m_hat = moments1[name] / (1.0 - beta1**step)
        v_hat = moments2[name] / (1.0 - beta2**step)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)


def batches(samples: list[Sample], batch_size: int) -> Iterable[list[Sample]]:
    for start in range(0, len(samples), batch_size):
        yield samples[start : start + batch_size]


def train(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    baseline_field = args.baseline_field or None
    train_samples = load_jsonl(
        args.train, args.target_field, args.target_scale, baseline_field
    )
    valid_samples = (
        load_jsonl(args.valid, args.target_field, args.target_scale, baseline_field)
        if args.valid is not None
        else train_samples
    )
    model = init_model(args.hidden, rng)
    best_model = clone_model(model)
    best_epoch = 0
    best_valid_rmse = float("inf")
    best_valid_mae = float("inf")
    best_valid_sign = 0.0
    moments1 = zero_grads(model)
    moments2 = zero_grads(model)
    step = 0
    rank_temperature = args.rank_temperature_cp / args.target_scale
    if args.rank_loss_weight < 0.0:
        raise SystemExit("--rank-loss-weight must be non-negative")
    if rank_temperature <= 0.0:
        raise SystemExit("--rank-temperature-cp must be greater than zero")
    train_rank_groups = grouped_rank_samples(train_samples)

    print(f"train samples: {len(train_samples)}")
    print(f"valid samples: {len(valid_samples)}")
    print(f"hidden: {args.hidden}")
    train_baseline = evaluate_baseline(train_samples, args.target_scale)
    valid_baseline = evaluate_baseline(valid_samples, args.target_scale)
    if train_baseline is not None:
        print(
            "baseline train "
            f"rmse={train_baseline[0]:.2f} mae={train_baseline[1]:.2f} "
            f"sign={train_baseline[2] * 100.0:.2f}%"
        )
    if valid_baseline is not None:
        print(
            "baseline valid "
            f"rmse={valid_baseline[0]:.2f} mae={valid_baseline[1]:.2f} "
            f"sign={valid_baseline[2] * 100.0:.2f}%"
        )
    train_rank = evaluate_model_ranking(model, train_samples)
    valid_rank = evaluate_model_ranking(model, valid_samples)
    if train_rank is not None:
        print(
            "initial rank train "
            f"roots={train_rank[0]} top1={train_rank[1] * 100.0:.2f}% "
            f"selected_regret={train_rank[2]:.2f} bad_selected={train_rank[3]}"
        )
    if valid_rank is not None:
        print(
            "initial rank valid "
            f"roots={valid_rank[0]} top1={valid_rank[1] * 100.0:.2f}% "
            f"selected_regret={valid_rank[2]:.2f} bad_selected={valid_rank[3]}"
        )
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_samples)
        for batch in batches(train_samples, args.batch_size):
            grads = zero_grads(model)
            inv_batch = 1.0 / len(batch)
            for sample in batch:
                pred, _, _ = forward(model, sample)
                dloss = (pred - sample.target) * inv_batch
                add_prediction_grad(model, grads, sample, dloss)
            step += 1
            apply_adam(model, grads, moments1, moments2, step, args.lr, args.weight_decay)
        step = apply_rank_loss_epoch(
            model,
            train_rank_groups,
            moments1,
            moments2,
            step,
            args.lr,
            args.weight_decay,
            args.rank_loss_weight,
            rank_temperature,
        )

        train_rmse, train_mae, train_sign = evaluate(model, train_samples, args.target_scale)
        valid_rmse, valid_mae, valid_sign = evaluate(model, valid_samples, args.target_scale)
        train_rank = evaluate_model_ranking(model, train_samples)
        valid_rank = evaluate_model_ranking(model, valid_samples)
        if valid_rmse < best_valid_rmse:
            best_model = clone_model(model)
            best_epoch = epoch
            best_valid_rmse = valid_rmse
            best_valid_mae = valid_mae
            best_valid_sign = valid_sign
        message = (
            f"epoch {epoch:03d} "
            f"train_rmse={train_rmse:.2f} train_mae={train_mae:.2f} "
            f"train_sign={train_sign * 100.0:.2f}% "
            f"valid_rmse={valid_rmse:.2f} valid_mae={valid_mae:.2f} "
            f"valid_sign={valid_sign * 100.0:.2f}%"
        )
        if train_rank is not None and valid_rank is not None:
            message += (
                f" train_top1={train_rank[1] * 100.0:.2f}%"
                f" train_sel_regret={train_rank[2]:.2f}"
                f" valid_top1={valid_rank[1] * 100.0:.2f}%"
                f" valid_sel_regret={valid_rank[2]:.2f}"
            )
        print(message)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **best_model)
    meta_path = args.output.with_suffix(args.output.suffix + ".json")
    meta = {
        "format": "tiny_nnue_numpy_v1",
        "checkpoint": "best_valid_rmse",
        "best_epoch": best_epoch,
        "best_valid_rmse": best_valid_rmse,
        "best_valid_mae": best_valid_mae,
        "best_valid_sign": best_valid_sign,
        "hidden": args.hidden,
        "num_features": NNUE_NUM_FEATURES,
        "num_king_buckets": NNUE_NUM_KING_BUCKETS,
        "target_scale": args.target_scale,
        "target_field": args.target_field,
        "rank_loss_weight": args.rank_loss_weight,
        "rank_temperature_cp": args.rank_temperature_cp,
        "train_samples": len(train_samples),
        "valid_samples": len(valid_samples),
    }
    best_rank = evaluate_model_ranking(best_model, valid_samples)
    if best_rank is not None:
        meta["best_valid_rank_roots"] = best_rank[0]
        meta["best_valid_rank_top1"] = best_rank[1]
        meta["best_valid_rank_selected_regret"] = best_rank[2]
        meta["best_valid_rank_bad_selected"] = best_rank[3]
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote model: {args.output}")
    print(f"wrote meta: {meta_path}")
    if args.binary_output is not None:
        write_binary_model(args.binary_output, best_model, args.hidden, args.target_scale)
        print(f"wrote binary model: {args.binary_output}")
    print(
        f"best epoch: {best_epoch} "
        f"valid_rmse={best_valid_rmse:.2f} "
        f"valid_mae={best_valid_mae:.2f} "
        f"valid_sign={best_valid_sign * 100.0:.2f}%"
    )


def main() -> None:
    args = parse_args()
    if args.hidden <= 0:
        raise SystemExit("--hidden must be greater than zero")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be greater than zero")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")
    if args.target_scale <= 0.0:
        raise SystemExit("--target-scale must be greater than zero")
    train(args)


if __name__ == "__main__":
    main()
