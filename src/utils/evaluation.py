"""Evaluation bookkeeping for segmentation experiments."""

from __future__ import annotations

import datetime as dt
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TQDM_BAR_FMT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"


def finite_values(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def mean_std(values: Iterable[float], *, scale: float = 1.0) -> tuple[float, float]:
    finite = finite_values(values)
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite) * scale), float(np.std(finite) * scale)


def nanmean_std(values: Iterable[float], *, scale: float = 1.0) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(arr) * scale), float(np.nanstd(arr) * scale)


def safe_percent(value: float) -> float:
    return float(value) * 100.0 if math.isfinite(float(value)) else float("nan")


@dataclass
class RuntimeTracker:
    """Track wall-clock test time and model-only inference time."""

    start_time: float = field(default_factory=time.perf_counter)
    inference_time_sec: float = 0.0
    num_samples: int = 0

    def add_batch(self, batch_size: int, inference_seconds: float) -> None:
        self.num_samples += int(batch_size)
        self.inference_time_sec += float(inference_seconds)

    @property
    def test_time_sec(self) -> float:
        return time.perf_counter() - self.start_time

    def summary(self) -> dict[str, float]:
        test_time_sec = self.test_time_sec
        n = max(self.num_samples, 0)
        return {
            "TestTimeSec": test_time_sec,
            "TestLatencyMsPerSample": test_time_sec * 1000.0 / n if n > 0 else float("nan"),
            "TestThroughputSamplesPerSec": n / test_time_sec if test_time_sec > 0 else float("nan"),
            "InferenceTimeSec": self.inference_time_sec,
            "InferenceLatencyMsPerSample": self.inference_time_sec * 1000.0 / n if n > 0 else float("nan"),
            "InferenceThroughputSamplesPerSec": n / self.inference_time_sec
            if self.inference_time_sec > 0
            else float("nan"),
        }

    def rows(self) -> list[list[float | str]]:
        return [[key, value, 0.0] for key, value in self.summary().items()]


@dataclass
class SegmentationEval:
    """Collect per-sample binary foreground metrics and empty-mask diagnostics."""

    dice: list[float] = field(default_factory=list)
    dice_all: list[float] = field(default_factory=list)
    iou: list[float] = field(default_factory=list)
    iou_all: list[float] = field(default_factory=list)
    hd95: list[float] = field(default_factory=list)
    asd: list[float] = field(default_factory=list)
    boundary_f1: list[float] = field(default_factory=list)
    surface_dice: list[float] = field(default_factory=list)
    gt_empty: list[bool] = field(default_factory=list)
    pred_empty: list[bool] = field(default_factory=list)
    gt_foreground_ratio: list[float] = field(default_factory=list)
    pred_foreground_ratio: list[float] = field(default_factory=list)

    def update(self, metrics: dict) -> None:
        self.dice.append(float(metrics["dice"]))
        self.dice_all.append(float(metrics["dice_all"]))
        self.iou.append(float(metrics.get("iou_binary", metrics["iou"])))
        self.iou_all.append(float(metrics["iou_all"]))
        self.hd95.append(float(metrics.get("hd95", float("nan"))))
        self.asd.append(float(metrics.get("asd", float("nan"))))
        self.boundary_f1.append(float(metrics.get("boundary_f1", float("nan"))))
        self.surface_dice.append(float(metrics.get("surface_dice", float("nan"))))
        self.gt_empty.append(bool(metrics["gt_empty"]))
        self.pred_empty.append(bool(metrics["pred_empty"]))
        self.gt_foreground_ratio.append(float(metrics["gt_foreground_ratio"]))
        self.pred_foreground_ratio.append(float(metrics["pred_foreground_ratio"]))

    @property
    def latest_display_dice(self) -> float:
        if not self.dice:
            return float("nan")
        return self.dice[-1] if np.isfinite(self.dice[-1]) else self.dice_all[-1]

    def empty_stats(self) -> dict[str, float | int]:
        gt_empty = np.asarray(self.gt_empty, dtype=bool)
        pred_empty = np.asarray(self.pred_empty, dtype=bool)
        if gt_empty.size == 0:
            return {
                "positive_count": 0,
                "empty_count": 0,
                "gt_empty_rate": float("nan"),
                "pred_empty_rate": float("nan"),
                "empty_specificity": float("nan"),
                "empty_false_positive_rate": float("nan"),
                "positive_miss_rate": float("nan"),
            }

        positive_gt = ~gt_empty
        empty_count = int(gt_empty.sum())
        positive_count = int(positive_gt.sum())
        return {
            "positive_count": positive_count,
            "empty_count": empty_count,
            "gt_empty_rate": float(gt_empty.mean()),
            "pred_empty_rate": float(pred_empty.mean()),
            "empty_specificity": float((pred_empty[gt_empty]).mean()) if empty_count > 0 else float("nan"),
            "empty_false_positive_rate": float((~pred_empty[gt_empty]).mean()) if empty_count > 0 else float("nan"),
            "positive_miss_rate": float((pred_empty[positive_gt]).mean()) if positive_count > 0 else float("nan"),
        }

    def rows(self) -> list[list[float | str]]:
        dice_mean, dice_std = mean_std(self.dice, scale=100.0)
        dice_all_mean, dice_all_std = nanmean_std(self.dice_all, scale=100.0)
        iou_mean, iou_std = mean_std(self.iou, scale=100.0)
        iou_all_mean, iou_all_std = nanmean_std(self.iou_all, scale=100.0)
        hd95_mean, hd95_std = mean_std(self.hd95)
        asd_mean, asd_std = mean_std(self.asd)
        boundary_f1_mean, boundary_f1_std = mean_std(self.boundary_f1, scale=100.0)
        surface_dice_mean, surface_dice_std = mean_std(self.surface_dice, scale=100.0)
        gt_fg_mean, gt_fg_std = nanmean_std(self.gt_foreground_ratio, scale=100.0)
        pred_fg_mean, pred_fg_std = nanmean_std(self.pred_foreground_ratio, scale=100.0)
        empty = self.empty_stats()

        return [
            ["Dice", dice_mean, dice_std],
            ["DiceAll", dice_all_mean, dice_all_std],
            ["IoU", iou_mean, iou_std],
            ["IoUAll", iou_all_mean, iou_all_std],
            ["HD95", hd95_mean, hd95_std],
            ["ASD", asd_mean, asd_std],
            ["BoundaryF1", boundary_f1_mean, boundary_f1_std],
            ["SurfaceDice", surface_dice_mean, surface_dice_std],
            ["PositiveSampleCount", empty["positive_count"], 0.0],
            ["EmptySampleCount", empty["empty_count"], 0.0],
            ["GTEmptyRate", safe_percent(empty["gt_empty_rate"]), 0.0],
            ["PredEmptyRate", safe_percent(empty["pred_empty_rate"]), 0.0],
            ["EmptySpecificity", safe_percent(empty["empty_specificity"]), 0.0],
            ["EmptyFalsePositiveRate", safe_percent(empty["empty_false_positive_rate"]), 0.0],
            ["PositiveMissRate", safe_percent(empty["positive_miss_rate"]), 0.0],
            ["MeanGTForegroundRatio", gt_fg_mean, gt_fg_std],
            ["MeanPredForegroundRatio", pred_fg_mean, pred_fg_std],
        ]

    def means(self) -> dict[str, float]:
        return {metric: mean for metric, mean, _ in self.rows()}

    def log_summary(self, extra: dict[str, tuple[float, float]] | None = None) -> str:
        values = {metric: (mean, std) for metric, mean, std in self.rows()}
        prefix = ""
        if extra:
            prefix = ", ".join(f"{name}={mean:.4f}+/-{std:.4f}" for name, (mean, std) in extra.items()) + ", "
        return (
            f"{prefix}"
            f"Dice={values['Dice'][0]:.2f}+/-{values['Dice'][1]:.2f}, "
            f"DiceAll={values['DiceAll'][0]:.2f}, "
            f"IoU={values['IoU'][0]:.2f}, "
            f"HD95={values['HD95'][0]:.2f}, ASD={values['ASD'][0]:.2f}, "
            f"BoundaryF1={values['BoundaryF1'][0]:.2f}, SurfaceDice={values['SurfaceDice'][0]:.2f}, "
            f"EmptySpecificity={values['EmptySpecificity'][0]:.2f}"
        )


def make_results_dir(base_dir: str | Path, *, score_name: str, score_value: float, timestamp_fmt: str) -> Path:
    safe_score = float(score_value) if math.isfinite(float(score_value)) else float("nan")
    timestamp = dt.datetime.now(tz=dt.UTC).astimezone().strftime(timestamp_fmt)
    folder = Path(base_dir) / f"{timestamp}-{score_name}{safe_score:.2f}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def metric_rows_to_frame(rows: list[list[float | str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Metric", "Mean", "Std"])


def write_metric_rows(rows: list[list[float | str]], output_path: str | Path) -> pd.DataFrame:
    df = metric_rows_to_frame(rows)
    df.to_csv(output_path, index=False, float_format="%.4f")
    return df
