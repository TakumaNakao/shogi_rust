#!/usr/bin/env python3
"""CPU-friendly sparse trainer for the compact HalfKP model.

The trainer intentionally updates only active feature rows. It is suitable for
small laptop pilots and writes the HKP00001 format consumed by Rust.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


PIECE_STATES = 2344
KING_BUCKETS = 45
INPUTS = PIECE_STATES * KING_BUCKETS


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--valid", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--target", choices=["result", "static_eval"], default="result")
    p.add_argument("--target-scale", type=float, default=1000.0)
    p.add_argument("--wdl-scale", type=float, default=600.0)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--max-train", type=int)
    p.add_argument("--max-valid", type=int)
    p.add_argument("--seed", type=int, default=20260715)
    p.add_argument("--init-npz", type=Path,
                   help="initialize from a previous trainer .npz (for fine-tuning)")
    p.add_argument("--binary-output", type=Path)
    return p.parse_args()


def load(path: Path, target: str, limit: int | None) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(out) >= limit:
                break
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("features_black") or not rec.get("features_white"):
                continue
            if target == "static_eval" and rec.get("static_eval") is None:
                continue
            out.append(rec)
    if not out:
        raise ValueError(f"no usable records in {path}")
    return out


def init_model(hidden: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "feature_emb": rng.normal(0.0, 0.01, (INPUTS, hidden)).astype(np.float32),
        "hidden_b": np.full((hidden,), 0.5, dtype=np.float32),
        # Start with a neutral evaluator; large random output weights make the
        # first sparse AdaGrad updates saturate the WDL sigmoid.
        "out_w": np.zeros((hidden * 2 + 1,), dtype=np.float32),
        "out_b": np.zeros((), dtype=np.float32),
    }


def load_init(path: Path, hidden: int) -> dict[str, np.ndarray]:
    """Load and validate a checkpoint produced by this trainer."""
    with np.load(path) as data:
        required = ("feature_emb", "hidden_b", "out_w", "out_b")
        if any(name not in data for name in required):
            raise ValueError(f"checkpoint is missing one of {required}: {path}")
        model = {name: np.asarray(data[name], dtype=np.float32).copy() for name in required}
    expected = {
        "feature_emb": (INPUTS, hidden),
        "hidden_b": (hidden,),
        "out_w": (hidden * 2 + 1,),
        "out_b": (),
    }
    for name, shape in expected.items():
        if model[name].shape != shape:
            raise ValueError(f"checkpoint {name} shape {model[name].shape} != {shape}")
    return model


def active(rec: dict) -> tuple[list[int], list[int], float]:
    if rec["side_to_move"] == "black":
        return rec["features_black"], rec["features_white"], float(rec["material_black"])
    return rec["features_white"], rec["features_black"], float(rec["material_white"])


def forward(model: dict[str, np.ndarray], rec: dict) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    stm, nstm, material = active(rec)
    hidden_b = model["hidden_b"]
    a = hidden_b + model["feature_emb"][stm].sum(axis=0)
    b = hidden_b + model["feature_emb"][nstm].sum(axis=0)
    ha = np.clip(a, 0.0, 1.0)
    hb = np.clip(b, 0.0, 1.0)
    x = np.concatenate((ha, hb, np.asarray([material / 1000.0], dtype=np.float32)))
    return float(x @ model["out_w"] + model["out_b"]), a, b, x, material


def target(rec: dict, mode: str, scale: float) -> float:
    if mode == "result":
        return float(rec["result"])
    return float(rec["static_eval"]) / scale


def update(model: dict[str, np.ndarray], acc: dict[str, np.ndarray], rec: dict,
           mode: str, target_scale: float, wdl_scale: float, lr: float,
           decay: float) -> tuple[float, float]:
    pred, a, b, x, material = forward(model, rec)
    y = target(rec, mode, target_scale)
    if mode == "result":
        wdl_norm = wdl_scale / target_scale
        p = 1.0 / (1.0 + math.exp(float(np.clip(-pred / wdl_norm, -40.0, 40.0))))
        loss = -(y * math.log(max(p, 1e-7)) + (1.0 - y) * math.log(max(1.0 - p, 1e-7)))
        dscore = (p - y) / wdl_norm
    else:
        err = pred - y
        loss = 0.5 * err * err
        dscore = err
    dscore = float(np.clip(dscore, -5.0, 5.0))
    out_grad = dscore * x
    model["out_w"] -= lr * out_grad / np.sqrt(acc["out_w"] + 1e-8)
    acc["out_w"] += out_grad * out_grad
    np.clip(model["out_w"], -8.0, 8.0, out=model["out_w"])
    model["out_b"] -= lr * dscore / math.sqrt(float(acc["out_b"]) + 1e-8)
    acc["out_b"] += dscore * dscore
    model["out_b"][...] = np.clip(model["out_b"], -8.0, 8.0)

    hidden = model["hidden_b"].shape[0]
    da = dscore * model["out_w"][:hidden] * ((a > 0.0) & (a < 1.0))
    db = dscore * model["out_w"][hidden:2 * hidden] * ((b > 0.0) & (b < 1.0))
    dbias = da + db
    model["hidden_b"] -= lr * dbias / np.sqrt(acc["hidden_b"] + 1e-8)
    acc["hidden_b"] += dbias * dbias
    np.clip(model["hidden_b"], -1.0, 1.0, out=model["hidden_b"])
    for rows, grad in ((active(rec)[0], da), (active(rec)[1], db)):
        for row in rows:
            g = grad + decay * model["feature_emb"][row]
            model["feature_emb"][row] -= lr * g / np.sqrt(acc["feature_emb"][row] + 1e-8)
            acc["feature_emb"][row] += g * g
            np.clip(model["feature_emb"][row], -1.0, 1.0, out=model["feature_emb"][row])
    return loss, pred


def metrics(model: dict[str, np.ndarray], records: list[dict], mode: str,
            target_scale: float, wdl_scale: float) -> tuple[float, float, float]:
    losses = []
    abs_err = []
    correct = 0
    for rec in records:
        pred, *_ = forward(model, rec)
        y = target(rec, mode, target_scale)
        if mode == "result":
            wn = wdl_scale / target_scale
            p = 1.0 / (1.0 + math.exp(float(np.clip(-pred / wn, -40.0, 40.0))))
            losses.append(-(y * math.log(max(p, 1e-7)) + (1-y) * math.log(max(1-p, 1e-7))))
            correct += int((p >= 0.5) == (y >= 0.5))
        else:
            losses.append((pred - y) ** 2)
            correct += int((pred >= 0.0) == (y >= 0.0))
        abs_err.append(abs(pred - y) * target_scale)
    return float(np.mean(losses)), float(np.mean(abs_err)), correct / len(records)


def write_binary(path: Path, model: dict[str, np.ndarray], hidden: int, scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"HKP00001")
        f.write(struct.pack("<IIIII f", 1, hidden, INPUTS, KING_BUCKETS, PIECE_STATES, scale))
        for name in ("feature_emb", "hidden_b", "out_w"):
            f.write(np.ascontiguousarray(model[name], dtype="<f4").tobytes())
        f.write(struct.pack("<f", float(model["out_b"])))


def main() -> None:
    a = args()
    if a.hidden <= 0 or a.epochs <= 0 or a.lr <= 0:
        raise SystemExit("hidden, epochs and lr must be positive")
    train = load(a.train, a.target, a.max_train)
    valid = load(a.valid, a.target, a.max_valid)
    rng = np.random.default_rng(a.seed)
    model = load_init(a.init_npz, a.hidden) if a.init_npz else init_model(a.hidden, rng)
    acc = {name: np.zeros_like(value) for name, value in model.items()}
    print(f"train={len(train)} valid={len(valid)} hidden={a.hidden} target={a.target}")
    baseline_metric = metrics(model, valid, a.target, a.target_scale, a.wdl_scale)
    print("baseline valid", baseline_metric)
    best = {name: value.copy() for name, value in model.items()}
    best_metric = baseline_metric[0]
    for epoch in range(1, a.epochs + 1):
        rng.shuffle(train)
        losses = []
        for rec in train:
            loss, _ = update(model, acc, rec, a.target, a.target_scale, a.wdl_scale, a.lr, a.weight_decay)
            losses.append(loss)
        valid_metric = metrics(model, valid, a.target, a.target_scale, a.wdl_scale)
        print(f"epoch={epoch} train_loss={np.mean(losses):.6f} valid={valid_metric}")
        if valid_metric[0] < best_metric:
            best_metric = valid_metric[0]
            best = {name: value.copy() for name, value in model.items()}
    for name in model:
        model[name] = best[name]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **model)
    meta = {
        "format": "halfkp_numpy_v1", "hidden": a.hidden, "inputs": INPUTS,
        "king_buckets": KING_BUCKETS, "piece_states": PIECE_STATES,
        "target": a.target, "target_scale": a.target_scale,
        "wdl_scale": a.wdl_scale, "train": len(train), "valid": len(valid),
    }
    a.output.with_suffix(a.output.suffix + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    if a.binary_output:
        write_binary(a.binary_output, model, a.hidden, a.target_scale)
    print(f"wrote {a.output}")
    if a.binary_output:
        print(f"wrote {a.binary_output}")


if __name__ == "__main__":
    main()
