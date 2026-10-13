"""Shared helpers for RIBR segmentation training and testing."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from monai.metrics import (
    compute_average_surface_distance as _asd,
)
from monai.metrics import (
    compute_hausdorff_distance as _hd95,
)
from monai.metrics import (
    compute_iou as _iou,
)
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from torchmetrics.functional.segmentation import generalized_dice_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_coords2d(height: int, width: int) -> torch.Tensor:
    x = torch.linspace(-1.0, 1.0, width)
    y = torch.linspace(-1.0, 1.0, height)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


def snapshot_paths(exp: str, dataset: str, img_size: int, output_dir: str | Path = "runs") -> tuple[Path, Path]:
    for name in (exp, dataset):
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("Experiment and dataset names must be single directory names")
    base = Path(output_dir) / exp / dataset / str(img_size)
    return base / "train", base / "test"


def convert_tensor_to_onehot(x: torch.Tensor | np.ndarray, num_classes: int) -> torch.Tensor:
    x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    onehot = torch.zeros(size=(x.shape[0], x.shape[1], num_classes), dtype=torch.int32, device=x.device)
    for class_idx in range(num_classes):
        onehot[..., class_idx] = x == class_idx
    return onehot


def multiclass_dice_score(
    segmentation: torch.Tensor,
    images: torch.Tensor,
    num_classes: int,
    include_background: bool = False,
) -> torch.Tensor:
    if segmentation.shape != images.shape:
        raise ValueError(
            f"Segmentation and ground truth must have the same shape: {segmentation.shape=} {images.shape=}"
        )
    if segmentation.shape[-1] != num_classes:
        raise ValueError(f"Segmentation last dimension must equal {num_classes}: {segmentation.shape=}")
    if segmentation.ndim == 3:
        segmentation = segmentation[None, ...]
    if images.ndim == 3:
        images = images[None, ...]
    segmentation = segmentation.permute(0, 3, 1, 2)
    images = images.permute(0, 3, 1, 2)
    return generalized_dice_score(
        segmentation,
        images,
        include_background=include_background,
        num_classes=num_classes,
        weight_type="simple",
    )


def binary_foreground_sample_metrics(pred_hw: np.ndarray, gt_hw: np.ndarray) -> dict[str, float | bool]:
    pred = np.asarray(pred_hw) > 0
    gt = np.asarray(gt_hw) > 0
    pred_empty = not bool(pred.any())
    gt_empty = not bool(gt.any())

    if gt_empty:
        dice = 1.0 if pred_empty else 0.0
        iou = 1.0 if pred_empty else 0.0
        dice_all = 1.0 if pred_empty else 0.0
        iou_all = 1.0 if pred_empty else 0.0
    else:
        intersection = float(np.logical_and(pred, gt).sum())
        pred_sum = float(pred.sum())
        gt_sum = float(gt.sum())
        union = float(np.logical_or(pred, gt).sum())
        dice = 2.0 * intersection / (pred_sum + gt_sum)
        iou = intersection / union
        dice_all = dice
        iou_all = iou

    return {
        "dice": float(dice),
        "iou": float(iou),
        "dice_all": float(dice_all),
        "iou_all": float(iou_all),
        "gt_empty": gt_empty,
        "pred_empty": pred_empty,
        "gt_foreground_ratio": float(gt.mean()),
        "pred_foreground_ratio": float(pred.mean()),
    }


def binary_boundary(mask_hw: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_hw) > 0
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask)
    return np.logical_xor(mask, eroded)


def boundary_f1_score(pred_hw: np.ndarray, gt_hw: np.ndarray, tolerance: int = 2, eps: float = 1e-8) -> float:
    pred_boundary = binary_boundary(pred_hw)
    gt_boundary = binary_boundary(gt_hw)
    pred_empty = not bool(pred_boundary.any())
    gt_empty = not bool(gt_boundary.any())
    if pred_empty and gt_empty:
        return 1.0
    if pred_empty or gt_empty:
        return 0.0
    structure = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
    gt_match_band = binary_dilation(gt_boundary, structure=structure)
    pred_match_band = binary_dilation(pred_boundary, structure=structure)
    precision = np.logical_and(pred_boundary, gt_match_band).sum() / (pred_boundary.sum() + eps)
    recall = np.logical_and(gt_boundary, pred_match_band).sum() / (gt_boundary.sum() + eps)
    return float(2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0


def surface_dice_score(pred_hw: np.ndarray, gt_hw: np.ndarray, tolerance: float = 2.0) -> float:
    pred_boundary = binary_boundary(pred_hw)
    gt_boundary = binary_boundary(gt_hw)
    pred_empty = not bool(pred_boundary.any())
    gt_empty = not bool(gt_boundary.any())
    if pred_empty and gt_empty:
        return 1.0
    if pred_empty or gt_empty:
        return 0.0
    pred_to_gt = distance_transform_edt(~gt_boundary)[pred_boundary]
    gt_to_pred = distance_transform_edt(~pred_boundary)[gt_boundary]
    pred_close = float((pred_to_gt <= tolerance).sum())
    gt_close = float((gt_to_pred <= tolerance).sum())
    return float((pred_close + gt_close) / (pred_boundary.sum() + gt_boundary.sum()))


def compute_seg_sample_metrics(seg_oh: torch.Tensor, mask_oh: torch.Tensor) -> dict:
    seg = seg_oh.permute(2, 0, 1).unsqueeze(0).float()
    gt = mask_oh.permute(2, 0, 1).unsqueeze(0).float()
    pred_foreground = seg_oh[..., 1:].sum(dim=-1).detach().cpu().numpy() > 0
    gt_foreground = mask_oh[..., 1:].sum(dim=-1).detach().cpu().numpy() > 0
    binary_metrics = binary_foreground_sample_metrics(pred_foreground, gt_foreground)
    iou = float(_iou(seg, gt, include_background=False).cpu().item())
    hd95 = float(_hd95(seg, gt, include_background=False, percentile=95).cpu().item())
    asd = float(_asd(seg, gt, include_background=False).cpu().item())
    return {
        "dice": binary_metrics["dice"],
        "iou": iou,
        "iou_binary": binary_metrics["iou"],
        "dice_all": binary_metrics["dice_all"],
        "iou_all": binary_metrics["iou_all"],
        "gt_empty": binary_metrics["gt_empty"],
        "pred_empty": binary_metrics["pred_empty"],
        "gt_foreground_ratio": binary_metrics["gt_foreground_ratio"],
        "pred_foreground_ratio": binary_metrics["pred_foreground_ratio"],
        "hd95": hd95,
        "asd": asd,
        "boundary_f1": boundary_f1_score(pred_foreground, gt_foreground),
        "surface_dice": surface_dice_score(pred_foreground, gt_foreground),
    }
