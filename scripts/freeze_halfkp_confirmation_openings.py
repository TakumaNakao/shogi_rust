#!/usr/bin/env python3
"""Freeze a confirmation suite after train and prior-match lineage checks."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-exclusions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    if args.pairs <= 0:
        raise SystemExit("--pairs must be positive")

    excluded_games, excluded_families = set(), set()
    with args.train_exclusions.open() as stream:
        for line in stream:
            value = json.loads(line)
            if value.get("game_key"):
                excluded_games.add(value["game_key"])
            if value.get("opening_family"):
                excluded_families.add(value["opening_family"])

    prior_families, prior_sfens, prior_records = set(), set(), 0
    for path in Path("runs/halfkp64-scratch-v1").glob("**/pairs/pair_*.json"):
        try:
            value = json.loads(path.read_text())
            opening = value["opening"]
        except (OSError, KeyError, json.JSONDecodeError):
            continue
        prior_records += 1
        prior_sfens.add(opening.get("sfen"))
        prior_families.update(str(opening.get("opening_family", "")).split("#"))

    candidates = []
    seen_games, seen_families = set(), set()
    with args.source.open() as stream:
        for line_number, line in enumerate(stream):
            value = json.loads(line)
            game = value.get("game_key")
            family = value.get("opening_family")
            sfen = value.get("sfen")
            if not game or not family or not sfen:
                continue
            if game in excluded_games or family in excluded_families:
                continue
            if family in prior_families or sfen in prior_sfens:
                continue
            if game in seen_games or family in seen_families:
                continue
            key = hashlib.sha256(f"{args.seed}:{game}:{family}:{sfen}".encode()).hexdigest()
            candidates.append((key, line_number, value))
            seen_games.add(game)
            seen_families.add(family)
    candidates.sort(key=lambda item: item[0])
    # The source is the training-shaped opening reserve.  The paired-match
    # loader deliberately accepts only records marked ``split=valid`` so that
    # a confirmation suite cannot accidentally be interpreted as train data.
    # Copy records before changing the marker; keep the original source
    # provenance in the manifest rather than mutating the source file.
    selected = []
    for _, _, value in candidates[: args.pairs]:
        frozen = dict(value)
        frozen["split"] = "valid"
        selected.append(frozen)
    if len(selected) != args.pairs:
        raise SystemExit(f"only {len(selected)} eligible openings; need {args.pairs}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in selected))
    output_sha = sha256(args.output)
    manifest = {
        "schema": "halfkp_confirmation_openings_v2",
        "seed": args.seed,
        "pairs": args.pairs,
        "records": len(selected),
        "source": {"path": str(args.source), "sha256": sha256(args.source)},
        "train_exclusions": {"path": str(args.train_exclusions), "sha256": sha256(args.train_exclusions)},
        "prior_match_opening_artifacts_scanned": prior_records,
        "excluded_train_games": len(excluded_games),
        "excluded_train_families": len(excluded_families),
        "eligible_before_selection": len(candidates),
        "output": {"path": str(args.output), "sha256": output_sha},
        "checks": {
            "selected_game_unique": len({value["game_key"] for value in selected}) == len(selected),
            "selected_family_unique": len({value["opening_family"] for value in selected}) == len(selected),
            "selected_sfen_unique": len({value["sfen"] for value in selected}) == len(selected),
            "prior_family_overlap": 0,
            "prior_sfen_overlap": 0,
            "train_game_overlap": 0,
            "train_family_overlap": 0,
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
