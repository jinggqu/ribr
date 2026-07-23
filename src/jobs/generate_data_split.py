"""Generate a project-owned RIBR split from the shared ultrasound catalog."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.datasets.manifest import DEFAULT_DATA_ROOT, DEFAULT_SPLIT_ROOT, DEFAULT_VIEW_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--data-view", type=Path, default=DEFAULT_VIEW_PATH)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def partition(groups: list[list[dict[str, str]]], ratios: dict[str, float], rng: random.Random) -> dict[str, str]:
    groups = list(groups)
    rng.shuffle(groups)
    total = sum(len(group) for group in groups)
    targets = {split: max(total * ratio, 1.0) for split, ratio in ratios.items()}
    counts = {split: 0 for split in ratios}
    assignments: dict[str, str] = {}
    for group in groups:
        split = min(ratios, key=lambda key: counts[key] / targets[key])
        for row in group:
            assignments[row["sample_id"]] = split
        counts[split] += len(group)
    return assignments


def stratified_partition(
    rows: list[dict[str, str]], ratios: dict[str, float], rng: random.Random
) -> dict[str, str]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    strata: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for group in grouped.values():
        labels = {row["class_label"] for row in group if row["class_label"]}
        strata[next(iter(labels)) if len(labels) == 1 else "unlabeled"].append(group)
    assignments: dict[str, str] = {}
    for stratum in sorted(strata):
        assignments.update(partition(strata[stratum], ratios, rng))
    return assignments


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    if not 0 <= args.test_ratio < 1 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("--test-ratio must be non-negative and val+test must be less than 1")

    data_root = args.data_root.expanduser().resolve()
    view_path = project_path(args.data_view)
    split_root = project_path(args.split_root)
    catalog = read_csv(data_root / "catalog" / "samples-v1.csv")
    view_rows = read_csv(view_path)
    view_ids = {
        row["sample_id"]
        for row in view_rows
        if row.get("segmentation") == "1" or row.get("classification") == "1"
    }
    rows = [row for row in catalog if row["sample_id"] in view_ids]
    if len(rows) != len(view_ids):
        raise ValueError("Project view contains sample IDs missing from the shared catalog")

    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    rng = random.Random(args.seed)
    assignments: dict[str, str] = {}
    fixed_test_datasets: list[str] = []
    for dataset in sorted(by_dataset):
        dataset_rows = by_dataset[dataset]
        fixed_test = [row for row in dataset_rows if row["official_split"] == "test"]
        if fixed_test:
            fixed_test_datasets.append(dataset)
            train_pool = [row for row in dataset_rows if row["official_split"] != "test"]
            assignments.update(
                stratified_partition(train_pool, {"train": 1 - args.val_ratio, "val": args.val_ratio}, rng)
            )
            assignments.update({row["sample_id"]: "test" for row in fixed_test})
        else:
            assignments.update(
                stratified_partition(
                    dataset_rows,
                    {
                        "train": 1 - args.val_ratio - args.test_ratio,
                        "val": args.val_ratio,
                        "test": args.test_ratio,
                    },
                    rng,
                )
            )

    split_root.mkdir(parents=True, exist_ok=True)
    split_path = split_root / f"seed-{args.seed}.csv"
    with split_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "dataset", "split"])
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["sample_id"]):
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "dataset": row["dataset"],
                    "split": assignments[row["sample_id"]],
                }
            )

    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row["dataset"]][assignments[row["sample_id"]]] += 1
    metadata = {
        "protocol": "ribr-v1",
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "view_sha256": sha256(view_path),
        "fixed_test_datasets": fixed_test_datasets,
        "counts": {dataset: dict(counter) for dataset, counter in sorted(counts.items())},
    }
    (split_root / f"seed-{args.seed}.json").write_text(json.dumps(metadata, indent=2) + "\n")

    lock_path = PROJECT_ROOT / "configs" / "data" / "ribr-v1.lock.json"
    if lock_path.is_file() and split_root == DEFAULT_SPLIT_ROOT:
        lock = json.loads(lock_path.read_text())
        lock.setdefault("splits", {})[str(args.seed)] = sha256(split_path)
        lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
