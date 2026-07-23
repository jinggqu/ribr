from __future__ import annotations

import csv
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = os.environ.get(
    "ULTRASOUND_DATA_ROOT", str(Path.home() / "datasets" / "ultrasound")
)
DEFAULT_VIEW_PATH = PROJECT_ROOT / "data_views" / "ribr-v1.csv"
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "data_splits" / "ribr-v1"
DEFAULT_DERIVATIVE_MANIFEST = "derivatives/resize256-v1/manifest.csv"


def add_data_protocol_args(parser) -> None:
    parser.add_argument("--data-view", type=str, default=None)
    parser.add_argument("--split-root", type=str, default=None)
    parser.add_argument("--derivative-manifest", type=str, default=None)


def _project_path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


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
    root = Path(data_root)
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

    with split_path.open(newline="") as stream:
        split_by_id = {row["sample_id"]: row["split"] for row in csv.DictReader(stream)}
    with view_path.open(newline="") as stream:
        view_ids = {
            row["sample_id"]
            for row in csv.DictReader(stream)
            if row.get(task, "0") == "1"
        }
    with derivative_path.open(newline="") as stream:
        derivative_by_id = {row["sample_id"]: row for row in csv.DictReader(stream)}
    requested = set(splits)
    records: list[dict[str, str]] = []
    with catalog_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            sample_id = row["sample_id"]
            if (
                row["dataset"] != dataset
                or sample_id not in view_ids
                or split_by_id.get(sample_id) not in requested
            ):
                continue
            derivative = derivative_by_id.get(sample_id)
            if derivative is None:
                raise ValueError(f"View sample has no protocol derivative: {sample_id}")
            records.append({**row, **derivative})
    if not records:
        raise ValueError(
            f"No {task} samples for dataset={dataset}, seed={seed}, splits={','.join(splits)}"
        )
    return records


def sample_path(data_root: str | Path, record: dict[str, str], field: str) -> Path:
    return Path(data_root) / record[field]
