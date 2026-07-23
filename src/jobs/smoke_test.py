"""End-to-end synthetic smoke test for RIBR segmentation."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.methods.ribr import train_seg
from src.utils.logger import setup_logger
from src.utils.tools import set_seed, snapshot_paths


IMAGE_SIZE = 32
DATASET = "SMOKE"


def _parse_args(argv: list[str]):
    previous = sys.argv
    try:
        sys.argv = [previous[0], *argv]
        return train_seg.get_args()
    finally:
        sys.argv = previous


def _write_synthetic_data(data_root: Path, project_root: Path) -> tuple[Path, Path]:
    catalog_dir = data_root / "catalog"
    derivative_dir = data_root / "derivatives" / "resize256-v1"
    image_dir = derivative_dir / DATASET / "images"
    mask_dir = derivative_dir / DATASET / "masks"
    view_path = project_root / "data_views" / "ribr-v1.csv"
    split_dir = project_root / "data_splits" / "ribr-v1"
    for directory in (catalog_dir, image_dir, mask_dir, view_path.parent, split_dir):
        directory.mkdir(parents=True, exist_ok=True)

    catalog: list[dict[str, str]] = []
    derivatives: list[dict[str, str]] = []
    view: list[dict[str, str | int]] = []
    subsets: list[dict[str, str]] = []
    yy, xx = np.mgrid[:IMAGE_SIZE, :IMAGE_SIZE]
    for split in ("train", "val", "test"):
        for index in range(2):
            name = f"{split}-{index}.png"
            mask = ((xx - 11 - 9 * index) ** 2 + (yy - 16) ** 2 <= (6 + index) ** 2).astype(np.uint8) * 255
            image = np.repeat(np.linspace(25, 180, IMAGE_SIZE, dtype=np.float32)[None, :], IMAGE_SIZE, axis=0)
            image = np.clip(image + 55.0 * (mask > 0), 0, 255).astype(np.uint8)
            Image.fromarray(image).save(image_dir / name)
            Image.fromarray(mask).save(mask_dir / name)
            sample_id = f"{DATASET}:{Path(name).stem}"
            catalog.append({"sample_id": sample_id, "dataset": DATASET, "mask_status": "nonempty"})
            derivatives.append(
                {
                    "sample_id": sample_id,
                    "image_path": (image_dir / name).relative_to(data_root).as_posix(),
                    "mask_path": (mask_dir / name).relative_to(data_root).as_posix(),
                }
            )
            view.append({"sample_id": sample_id, "segmentation": 1, "classification": 0})
            subsets.append({"sample_id": sample_id, "dataset": DATASET, "split": split})
    with (catalog_dir / "samples-v1.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "dataset", "mask_status"])
        writer.writeheader()
        writer.writerows(catalog)
    with (derivative_dir / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "image_path", "mask_path"])
        writer.writeheader()
        writer.writerows(derivatives)
    with view_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "segmentation", "classification"])
        writer.writeheader()
        writer.writerows(view)
    with (split_dir / "seed-1.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "dataset", "split"])
        writer.writeheader()
        writer.writerows(subsets)
    return view_path, split_dir


def _run(data_root: Path, output_root: Path, view_path: Path, split_root: Path) -> dict[str, str]:
    args = _parse_args(
        [
            "--exp", "smoke_seg-1",
            "--device", "cpu",
            "--data-root", str(data_root),
            "--dataset", DATASET,
            "--img-size", str(IMAGE_SIZE),
            "--batch-size", "2",
            "--num-workers", "0",
            "--hidden-features", "16",
            "--hidden-layers", "1",
            "--latent-dim", "8",
            "--encoder-channels", "4",
            "--residual-hidden", "8",
            "--residual-layers", "1",
            "--amortized-epochs", "1",
            "--amortized-val-freq", "1",
            "--patience", "1",
            "--postprocess-min-component-area", "0",
            "--output-dir", str(output_root),
        ]
    )
    args.data_view = str(view_path)
    args.split_root = str(split_root)
    args.derivative_manifest = "derivatives/resize256-v1/manifest.csv"
    train_dir, test_dir = snapshot_paths(args.exp, args.dataset, args.img_size, args.output_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    args.train_snapshot_path = str(train_dir)
    args.test_snapshot_path = str(test_dir)
    stats = train_seg.train_amortized(args, setup_logger(args.exp, log_dir=train_dir))
    args.train_time_sec = stats["train_time_sec"]
    args.train_epochs_ran = stats["train_epochs_ran"]
    args.train_time_per_epoch_sec = stats["train_time_per_epoch_sec"]
    (train_dir / "train.complete").touch()
    args.checkpoint = str(train_dir / "best_model.pt")
    train_seg.test_amortized(args, setup_logger(args.exp, log_dir=test_dir))
    results = list(test_dir.glob("*/results.csv"))
    checkpoint = train_dir / "best_model.pt"
    if not checkpoint.exists() or len(results) != 1:
        raise RuntimeError("RIBR segmentation smoke test did not produce its checkpoint and results")
    with (results[0].parent / "per_sample.csv").open(newline="") as stream:
        if not all(row["file"].startswith("sample-") for row in csv.DictReader(stream)):
            raise RuntimeError("RIBR segmentation smoke test exposed source sample identifiers")
    return {"checkpoint": str(checkpoint), "results": str(results[0])}


def main() -> None:
    set_seed(1)
    with tempfile.TemporaryDirectory(prefix="ribr-smoke-") as temporary_dir:
        root = Path(temporary_dir)
        data_root = root / "store"
        view_path, split_root = _write_synthetic_data(data_root, root / "project")
        report = {"phase": "confirm", "segmentation": _run(data_root, root / "runs", view_path, split_root)}
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
