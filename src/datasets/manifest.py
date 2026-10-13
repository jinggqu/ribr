from __future__ import annotations

import csv
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = os.environ.get(
    "DATA_ROOT", os.environ.get("ULTRASOUND_DATA_ROOT", str(Path.home() / "datasets" / "ultrasound"))
)
DEFAULT_VIEW_PATH = PROJECT_ROOT / "data_views" / "ribr-v1.csv"
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "data_splits" / "ribr-v1"
DEFAULT_DERIVATIVE_MANIFEST = "derivatives/resize256-v1/manifest.csv"

DATASET_ALIASES = {
    "LymphUS-C1": "LymphUS-Center-1",
    "LymphUS-C2": "LymphUS-Center-2",
    "Breast-USG": "Breast-Lesions-USG",
}
INTERNAL_COUNTS = {
    "LN-INT": {"train": 1018, "val": 127, "test": 128},
    "BUSI": {"train": 517, "val": 64, "test": 66},
    "DDTI": {"train": 509, "val": 63, "test": 65},
    "TN3K": {"train": 2265, "val": 614, "test": 614},
    "Prostate": {"train": 1241, "val": 690, "test": 690},
}
EXTERNAL_COUNTS = {
    "LymphUS-Center-1": 180,
    "LymphUS-Center-2": 158,
    "Breast-Lesions-USG": 252,
    "BUS-UCLM": 174,
}


def canonical_dataset(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def check_paper_count(dataset: str, split: str, count: int) -> None:
    dataset = canonical_dataset(dataset)
    expected = EXTERNAL_COUNTS.get(dataset) if split == "external" else INTERNAL_COUNTS.get(dataset, {}).get(split)
    if expected is not None and count != expected:
        raise ValueError(
            f"Paper split mismatch: {dataset}/{split} has {count} images; expected {expected}. "
            "Check --split-root; use --allow-nonpaper-splits only for exploratory runs."
        )


def add_data_protocol_args(parser) -> None:
    parser.add_argument("--data-view", type=str, default=None)
    parser.add_argument("--split-root", type=str, default=None)
    parser.add_argument("--derivative-manifest", type=str, default=None)


def _project_path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    return Path(path).expanduser().resolve()


def _unique_rows(path: Path, required: set[str]) -> dict[str, dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if len(reader.fieldnames or []) != len(set(reader.fieldnames or [])):
            raise ValueError(f"Duplicate CSV column names in {path}")
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required CSV columns in {path}: {sorted(required)}")
        rows = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row in {path}")
            key = row["sample_id"]
            if not key or key in rows:
                raise ValueError(f"Empty or duplicate sample_id in {path}")
            rows[key] = row
    return rows


def load_records(
    data_root: str | Path,
    dataset: str,
    task: str,
    seed: int,
    splits: tuple[str, ...],
    *,
    view_path: str | Path | None = None,
    split_root: str | Path | None = None,
    derivative_manifest: str | Path | None = None,
) -> list[dict[str, str]]:
    root = Path(data_root).expanduser()
    catalog_path = root / "catalog" / "samples-v1.csv"
    derivative_path = Path(derivative_manifest or DEFAULT_DERIVATIVE_MANIFEST)
    if not derivative_path.is_absolute():
        derivative_path = root / derivative_path
    view_path = _project_path(view_path, DEFAULT_VIEW_PATH)
    split_path = _project_path(split_root, DEFAULT_SPLIT_ROOT) / f"seed-{seed}.csv"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Missing dataset catalog: {catalog_path}")
    if not derivative_path.is_file():
        raise FileNotFoundError(f"Missing derivative manifest: {derivative_path}")
    if not view_path.is_file():
        raise FileNotFoundError(f"Missing project data view: {view_path}")
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing project split: {split_path}")

    split_rows = _unique_rows(split_path, {"sample_id", "dataset", "split"})
    metadata_path = split_path.with_suffix(".json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("seed", seed) != seed:
            raise ValueError("Split metadata seed does not match the requested seed")
    view_ids = {key for key, row in _unique_rows(view_path, {"sample_id", task}).items() if row[task] == "1"}
    derivative_by_id = _unique_rows(derivative_path, {"sample_id", "image_path", "mask_path"})
    catalog = _unique_rows(catalog_path, {"sample_id", "dataset"})
    if not view_ids <= catalog.keys() or not view_ids <= split_rows.keys():
        raise ValueError("Selected view samples are missing from the catalog or seed split")
    group_splits: dict[str, str] = {}
    for sample_id in view_ids:
        row = catalog[sample_id]
        if row["dataset"] != dataset:
            continue
        assignment = split_rows[sample_id]
        if assignment["dataset"] != dataset or assignment["split"] not in {"train", "val", "test", "external"}:
            raise ValueError("Split and catalog disagree on dataset identity or use an unknown split")
        group = row.get("group_id")
        if group and group in group_splits and group_splits[group] != assignment["split"]:
            raise ValueError(f"Subject/group leakage across splits in {dataset}")
        if group:
            group_splits[group] = assignment["split"]
    requested = set(splits)
    records: list[dict[str, str]] = []
    for sample_id, row in catalog.items():
        if (
            row["dataset"] != dataset
            or sample_id not in view_ids
            or split_rows.get(sample_id, {}).get("split") not in requested
        ):
            continue
        derivative = derivative_by_id.get(sample_id)
        if derivative is None:
            raise ValueError(f"View sample has no protocol derivative: {sample_id}")
        if derivative.get("dataset", dataset) != dataset:
            raise ValueError("Derivative and catalog disagree on dataset identity")
        records.append({**row, **derivative})
    if not records:
        raise ValueError(f"No {task} samples for dataset={dataset}, seed={seed}, splits={','.join(splits)}")
    return records


def sample_path(data_root: str | Path, record: dict[str, str], field: str) -> Path:
    root = Path(data_root).expanduser().resolve()
    path = (root / record[field]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Manifest {field} must stay inside --data-root")
    return path
