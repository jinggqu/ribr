from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_fill_holes, label
from torch import nn
from torchinfo import summary
from tqdm import tqdm

from src.datasets import segmentation as dataset_seg
from src.datasets.manifest import DEFAULT_DATA_ROOT, EXTERNAL_COUNTS, add_data_protocol_args, canonical_dataset
from src.methods.ribr import RIBRSegmentationModel
from src.utils.evaluation import (
    TQDM_BAR_FMT,
    RuntimeTracker,
    SegmentationEval,
    make_results_dir,
    write_metric_rows,
)
from src.utils.logger import setup_logger
from src.utils.losses import (
    BoundaryBandSDFLoss,
    BoundarySemanticLoss,
    CoarseSemanticLoss,
    DiceSemanticLoss,
    FocalSemanticLoss,
    LossFunction,
    ResidualMagnitudeLoss,
    SpeckleAwareSmoothnessLoss,
)
from src.utils.tensors import coords_for_shape, expand_coords, flat_images_to_bchw
from src.utils.tools import (
    compute_seg_sample_metrics,
    convert_tensor_to_onehot,
    multiclass_dice_score,
    set_seed,
    snapshot_paths,
)

RIBR_SURFACE_DEFAULTS = {
    "patience": 20,
    "checkpoint_score": "balanced_seg",
    "boundary_loss_alpha": 0.6,
    "boundary_factor": 5.0,
    "gate_tau": 0.65,
    "gate_gamma": 12.0,
    "risk_route_scale": 0.25,
    "band_sdf_alpha": 0.18,
    "speckle_smooth_alpha": 0.03,
    "residual_l2_alpha": 0.06,
    "amortized_epochs": 120,
    "amortized_lr": 1.5e-4,
    "amortized_val_freq": 2,
}

# BRI, GSBR, RRC, matching ablation rows (a)-(f).
VARIANTS = {
    "conv-only": (False, False, False),
    "gsbr-only": (False, True, False),
    "bri-only": (True, False, False),
    "bri-gsbr": (True, True, False),
    "bri-rrc": (True, False, True),
    "full": (True, True, True),
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RIBR ultrasound segmentation training")

    parser.add_argument("--exp", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument(
        "--allow-nonpaper-splits", action="store_true", help="Allow exploratory splits; not paper reproduction."
    )

    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    add_data_protocol_args(parser)
    parser.add_argument("--dataset", type=str, default="LN-INT")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--strong-augs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--weak-augs", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--encoder-channels", type=int, default=10)
    parser.add_argument("--residual-hidden", type=int, default=32)
    parser.add_argument("--residual-layers", type=int, default=2)

    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument(
        "--checkpoint-score",
        type=str,
        default=None,
        choices=["dice", "balanced_seg"],
        help="Validation score used for checkpoint selection.",
    )
    parser.add_argument("--seg-dice-alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--seg-alpha", type=float, default=1.0)
    parser.add_argument("--coarse-loss-alpha", type=float, default=0.5)
    parser.add_argument("--boundary-loss-alpha", type=float, default=None)
    parser.add_argument("--boundary-factor", type=float, default=None)
    parser.add_argument("--boundary-kernel-size", type=int, default=5)
    parser.add_argument("--sdf-truncation", type=float, default=16.0)

    parser.add_argument("--gate-tau", type=float, default=None)
    parser.add_argument("--gate-gamma", type=float, default=None)
    parser.add_argument("--risk-route-scale", type=float, default=None)
    parser.add_argument("--local-stat-window", type=int, default=5)
    parser.add_argument("--band-sdf-alpha", type=float, default=None)
    parser.add_argument("--band-sdf-width", type=float, default=4.0)
    parser.add_argument("--speckle-smooth-alpha", type=float, default=None)
    parser.add_argument("--speckle-smooth-lambda", type=float, default=4.0)
    parser.add_argument("--residual-l2-alpha", type=float, default=None)
    parser.add_argument("--boundary-residual-scale", type=float, default=1.0)

    parser.add_argument("--amortized-epochs", type=int, default=None)
    parser.add_argument("--amortized-lr", type=float, default=None)
    parser.add_argument("--amortized-val-freq", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to load for testing; defaults to this run's train/best_model.pt.",
    )

    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--test-all-splits", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--postprocess-largest-component", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--postprocess-min-component-area", type=int, default=0)
    parser.add_argument("--postprocess-fill-holes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--export-sample-identifiers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write source file stems to outputs instead of deterministic pseudonyms.",
    )

    args = parser.parse_args(sys.argv[1:])
    args.dataset = canonical_dataset(args.dataset)
    if args.exp is None:
        args.exp = f"ribr-{args.variant}-{args.seed}"
    _apply_method_defaults(args)
    for key in ("local_stat_window", "boundary_kernel_size"):
        if getattr(args, key) < 1 or getattr(args, key) % 2 == 0:
            parser.error(f"--{key.replace('_', '-')} must be positive and odd")
    for key in (
        "batch_size",
        "amortized_epochs",
        "amortized_val_freq",
        "patience",
    ):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.hidden_features < 8 or args.hidden_features % 8:
        parser.error("--hidden-features must be a positive multiple of eight")
    if args.num_workers < 0 or args.postprocess_min_component_area < 0:
        parser.error("worker count and component area must be non-negative")
    if args.dataset in EXTERNAL_COUNTS and (not args.test or args.checkpoint is None):
        parser.error("External cohorts are test-only and require --test --checkpoint from LN-INT or BUSI")
    return args


def _apply_method_defaults(args: argparse.Namespace) -> None:
    for key, value in RIBR_SURFACE_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def build_model(args: argparse.Namespace) -> nn.Module:
    model_config = {
        "hidden_features": args.hidden_features,
        "in_channels": args.in_channels,
        "latent_dim": args.latent_dim,
        "encoder_channels": args.encoder_channels,
        "num_classes": args.num_classes,
        "use_relative_coords": True,
        "use_probe_coords": True,
        "probe_coord_mode": "hybrid",
        "use_coarse_context": True,
        "use_anatomy_style_split": True,
        "style_latent_dim": 16,
        "style_local_channels": 10,
        "anatomy_feature_source": "d1",
        "coarse_head_source": "d1",
    }
    segmentation_config = {
        "hidden_features": [args.hidden_features],
        "output_features": args.num_classes,
        "use_occlusion_field": True,
        "refine_segmentation": True,
    }
    common_kwargs = {
        "inr_config": model_config,
        "segmentation_config": segmentation_config,
        "img_size": args.img_size,
        "add_coarse_logits_to_segmentation": True,
        "coarse_segmentation_residual_scale": 1.0,
    }
    model = RIBRSegmentationModel(
        **common_kwargs,
        residual_hidden=args.residual_hidden,
        residual_layers=args.residual_layers,
        gate_tau=args.gate_tau,
        gate_gamma=args.gate_gamma,
        risk_route_scale=args.risk_route_scale,
        boundary_residual_scale=args.boundary_residual_scale,
        local_stat_window=args.local_stat_window,
        first_omega_0=20.0,
        hidden_omega_0=15.0,
        use_bri=VARIANTS[args.variant][0],
        use_gsbr=VARIANTS[args.variant][1],
        use_rrc=VARIANTS[args.variant][2],
    )
    return model.float().to(args.device)


def build_loss(args: argparse.Namespace) -> LossFunction:
    loss_terms = {
        "focal_loss": FocalSemanticLoss(alpha=args.seg_alpha, gamma=args.gamma),
        "dice_loss": DiceSemanticLoss(
            num_classes=args.num_classes,
            alpha=args.seg_dice_alpha,
        ),
    }
    if args.coarse_loss_alpha > 0:
        loss_terms["coarse_loss"] = CoarseSemanticLoss(num_classes=args.num_classes, alpha=args.coarse_loss_alpha)
    if VARIANTS[args.variant][1] and args.boundary_loss_alpha > 0:
        loss_terms["boundary_loss"] = BoundarySemanticLoss(
            num_classes=args.num_classes,
            alpha=args.boundary_loss_alpha,
            boundary_factor=args.boundary_factor,
            kernel_size=args.boundary_kernel_size,
        )
    if VARIANTS[args.variant][1]:
        loss_terms["band_sdf_loss"] = BoundaryBandSDFLoss(
            num_classes=args.num_classes,
            alpha=args.band_sdf_alpha,
            truncation=args.sdf_truncation,
            band_pixels=args.band_sdf_width,
        )
        loss_terms["speckle_smooth_loss"] = SpeckleAwareSmoothnessLoss(
            num_classes=args.num_classes,
            alpha=args.speckle_smooth_alpha,
            boundary_kernel_size=args.boundary_kernel_size,
            speckle_lambda=args.speckle_smooth_lambda,
            window=args.local_stat_window,
        )
    if VARIANTS[args.variant][2] and args.residual_l2_alpha > 0:
        loss_terms["residual_l2_loss"] = ResidualMagnitudeLoss(alpha=args.residual_l2_alpha)
    return LossFunction(loss_terms)


def _checkpoint_score(metrics: dict[str, float], args: argparse.Namespace) -> float:
    if args.checkpoint_score == "dice":
        return metrics["dice"]
    hd95_penalty = metrics["hd95"] / max(float(args.img_size), 1.0)
    asd_penalty = metrics["asd"] / max(float(args.img_size) / 4.0, 1.0)
    return metrics["dice"] + metrics["iou"] - hd95_penalty - asd_penalty


def _should_save_checkpoint(best_metrics: dict[str, float] | None, val_metrics: dict[str, float]) -> tuple[bool, str]:
    if not np.isfinite(val_metrics["score"]):
        raise ValueError("Validation score is non-finite; refusing to select a checkpoint")
    if best_metrics is None:
        return True, "initial checkpoint"
    if val_metrics["score"] > best_metrics["score"] + 1e-6:
        return True, "better validation score"
    return False, ""


def train_amortized(args: argparse.Namespace, logger: logging.Logger):
    train_start = time.perf_counter()
    dm = dataset_seg.USSegmentationDataModule(args)
    model = build_model(args)
    model.train()
    try:
        summary(model)
    except (RuntimeError, TypeError, ValueError) as error:
        logger.debug("Model summary unavailable: %s", error)

    coords_cache: dict[tuple[int, int], torch.Tensor] = {}
    trainloader, valloader = dm.train_dataloader(), dm.val_dataloader()
    loss_fn = build_loss(args)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.amortized_lr,
        weight_decay=0.01,
    )

    logger.info(
        "[RIBR] variant=%s | hidden=%s | epochs=%s | %s",
        args.variant,
        args.hidden_features,
        args.amortized_epochs,
        args.device,
    )

    best_metrics: dict[str, float] | None = None
    if (Path(args.train_snapshot_path) / "best_model.pt").exists():
        raise FileExistsError("Training directory already has a checkpoint; use a new --exp or --test")
    no_improve = 0
    epochs_ran = 0
    for epoch in range(args.amortized_epochs):
        epochs_ran = epoch + 1
        epoch_loss = 0.0
        epoch_loss_terms: dict[str, float] = {}
        model.train()
        for source_images, source_masks, _ in trainloader:
            images, masks = source_images.to(args.device), source_masks.to(args.device)
            images_bchw = flat_images_to_bchw(images, in_channels=args.in_channels)
            height, width = images_bchw.shape[-2:]
            coords = expand_coords(coords_for_shape(height, width, args.device, coords_cache), images.shape[0])

            output = model(coords, images_bchw)
            loss, loss_info = loss_fn({"images": images, "masks": masks, "output": output})
            for key, value in loss_info.items():
                epoch_loss_terms[key] = epoch_loss_terms.get(key, 0.0) + float(value)
            optimizer.zero_grad()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        epoch_loss /= max(len(trainloader), 1)
        term_log = ", ".join(
            f"{key}={value / max(len(trainloader), 1):.4f}" for key, value in sorted(epoch_loss_terms.items())
        )
        logger.info("E%s/%s | Loss=%.4f | %s", epoch + 1, args.amortized_epochs, epoch_loss, term_log)

        do_val = (epoch + 1) % args.amortized_val_freq == 0 or epoch == args.amortized_epochs - 1
        if not do_val:
            continue

        model.eval()
        val_metric_rows = []
        with torch.no_grad():
            for source_images, source_masks, _ in valloader:
                v_img, v_mask = source_images.to(args.device), source_masks.to(args.device)
                v_img_bchw = flat_images_to_bchw(v_img, in_channels=args.in_channels)
                height, width = v_img_bchw.shape[-2:]
                coords = expand_coords(coords_for_shape(height, width, args.device, coords_cache), v_img.shape[0])
                output = model(coords, v_img_bchw)
                seg_argmax = nn.functional.softmax(output["segmentation_output"], dim=-1).argmax(dim=-1)
                for b in range(v_img.shape[0]):
                    seg_oh = convert_tensor_to_onehot(seg_argmax[b].reshape(height, width), args.num_classes)
                    mask_oh = convert_tensor_to_onehot(v_mask[b].reshape(height, width).long(), args.num_classes)
                    if args.checkpoint_score == "dice":
                        val_metric_rows.append(
                            {
                                "dice": float(
                                    multiclass_dice_score(
                                        seg_oh,
                                        mask_oh,
                                        args.num_classes,
                                        include_background=False,
                                    )
                                ),
                                "iou": 0.0,
                                "hd95": float("nan"),
                                "asd": float("nan"),
                            }
                        )
                    else:
                        metrics = compute_seg_sample_metrics(seg_oh, mask_oh)
                        val_metric_rows.append(
                            {
                                "dice": float(metrics["dice"]),
                                "iou": float(metrics.get("iou_binary", metrics["iou"])),
                                "hd95": float(metrics.get("hd95", float("nan"))),
                                "asd": float(metrics.get("asd", float("nan"))),
                            }
                        )

        val_metrics = {}
        for metric in ["dice", "iou", "hd95", "asd"]:
            values = np.asarray([row[metric] for row in val_metric_rows], dtype=float)
            finite = values[np.isfinite(values)]
            val_metrics[metric] = float(np.mean(finite)) if len(finite) else float("nan")
        if not np.isfinite(val_metrics["iou"]):
            val_metrics["iou"] = 0.0
        if not np.isfinite(val_metrics["hd95"]):
            val_metrics["hd95"] = float(args.img_size)
        if not np.isfinite(val_metrics["asd"]):
            val_metrics["asd"] = float(args.img_size)
        val_metrics["score"] = _checkpoint_score(val_metrics, args)
        should_save, save_reason = _should_save_checkpoint(best_metrics, val_metrics)
        logger.info(
            "Val | Dice=%.2f IoU=%.2f HD95=%.2f ASD=%.2f Score=%.4f | CheckpointBy=%s",
            val_metrics["dice"] * 100,
            val_metrics["iou"] * 100,
            val_metrics["hd95"],
            val_metrics["asd"],
            val_metrics["score"],
            args.checkpoint_score,
        )
        if should_save:
            best_metrics = dict(val_metrics)
            no_improve = 0
            torch.save(model.state_dict(), Path(args.train_snapshot_path) / "best_model.pt")
            logger.info("Saved best model | reason=%s", save_reason)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info("Early stopping")
                break

    train_time_sec = time.perf_counter() - train_start
    logger.info("Timing | TrainTimeSec=%.2f", train_time_sec)
    return {
        "train_time_sec": train_time_sec,
        "train_epochs_ran": epochs_ran,
        "train_time_per_epoch_sec": train_time_sec / epochs_ran if epochs_ran > 0 else float("nan"),
    }


@torch.inference_mode()
def test_amortized(args: argparse.Namespace, logger: logging.Logger):
    model = build_model(args)
    model.eval()
    coords_cache: dict[tuple[int, int], torch.Tensor] = {}
    dm = dataset_seg.USSegmentationDataModule(args)
    testloader = dm.test_dataloader()

    save_best = Path(args.checkpoint) if args.checkpoint else Path(args.train_snapshot_path) / "best_model.pt"
    if not save_best.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {save_best}. Pass --checkpoint for external evaluation.")
    logger.info("Testing, loading from %s", save_best)
    model.load_state_dict(torch.load(save_best, map_location=args.device, weights_only=True))

    seg_eval = SegmentationEval()
    runtime = RuntimeTracker()

    per_sample_rows: list[dict] = []
    pbar = tqdm(
        enumerate(testloader),
        total=len(testloader),
        desc="Test",
        ncols=100,
        file=sys.stderr,
        bar_format=TQDM_BAR_FMT,
    )
    for _, (source_images, source_masks, file_names) in pbar:
        images, masks = source_images.to(args.device), source_masks.to(args.device)
        images_bchw = flat_images_to_bchw(images, in_channels=args.in_channels)
        height, width = images_bchw.shape[-2:]
        coords = expand_coords(coords_for_shape(height, width, args.device, coords_cache), images.shape[0])

        _synchronize_device(args.device)
        infer_start = time.perf_counter()
        output = model(coords, images_bchw)
        _synchronize_device(args.device)
        runtime.add_batch(images.shape[0], time.perf_counter() - infer_start)

        seg_probs = nn.functional.softmax(output["segmentation_output"], dim=-1)
        seg_argmax = seg_probs.argmax(dim=-1)
        for b in range(images.shape[0]):
            pred_hw = seg_argmax[b].reshape(height, width).detach().cpu().numpy()
            pred_hw = _postprocess_binary_prediction(pred_hw, args)
            pred_tensor = torch.as_tensor(pred_hw, device=masks.device)
            seg_oh = convert_tensor_to_onehot(pred_tensor, args.num_classes)
            mask_oh = convert_tensor_to_onehot(masks[b].reshape(height, width).long(), args.num_classes)
            seg_metrics = compute_seg_sample_metrics(seg_oh, mask_oh)
            seg_eval.update(seg_metrics)

            source_file_stem = file_names[b]
            if args.export_sample_identifiers:
                sample_key = source_file_stem
            else:
                digest = hashlib.sha256(f"{args.dataset}:{source_file_stem}".encode()).hexdigest()[:16]
                sample_key = f"sample-{digest}"
            per_sample_rows.append(
                {
                    "file": sample_key,
                    "dice": float(seg_metrics["dice"]),
                    "iou": float(seg_metrics.get("iou_binary", seg_metrics["iou"])),
                    "hd95": float(seg_metrics.get("hd95", float("nan"))),
                    "asd": float(seg_metrics.get("asd", float("nan"))),
                    "boundary_f1": float(seg_metrics.get("boundary_f1", float("nan"))),
                    "surface_dice": float(seg_metrics.get("surface_dice", float("nan"))),
                    "gt_empty": bool(seg_metrics["gt_empty"]),
                    "pred_empty": bool(seg_metrics["pred_empty"]),
                    "gt_foreground_ratio": float(seg_metrics["gt_foreground_ratio"]),
                    "pred_foreground_ratio": float(seg_metrics["pred_foreground_ratio"]),
                }
            )

        pbar.set_postfix_str(f"Dice={seg_eval.latest_display_dice:.4f}")

    train_time_sec = float(getattr(args, "train_time_sec", float("nan")))
    train_epochs_ran = float(getattr(args, "train_epochs_ran", float("nan")))
    train_time_per_epoch_sec = float(getattr(args, "train_time_per_epoch_sec", float("nan")))
    logger.info("Test | %s", seg_eval.log_summary())

    seg_means = seg_eval.means()
    backup_folder = make_results_dir(
        args.test_snapshot_path,
        score_name="dice",
        score_value=seg_means["Dice"],
        timestamp_fmt="%y%m%d-%H%M%S",
    )
    timing = runtime.summary()
    metric_rows = (
        seg_eval.rows()
        + [
            ["TrainTimeSec", train_time_sec, 0.0],
            ["TrainEpochsRan", train_epochs_ran, 0.0],
            ["TrainTimePerEpochSec", train_time_per_epoch_sec, 0.0],
        ]
        + [[key, value, 0.0] for key, value in timing.items()]
    )
    write_metric_rows(metric_rows, backup_folder / "results.csv")

    if per_sample_rows:
        pd.DataFrame(per_sample_rows).to_csv(backup_folder / "per_sample.csv", index=False, float_format="%.6f")

    with (backup_folder / "conf.json").open("w") as f:
        json.dump(args.__dict__, f, indent=2)

    shutil.copy(save_best, backup_folder / "best_model.pt")
    return "Testing Finished!"


def _synchronize_device(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    elif torch.device(device).type == "mps":
        torch.mps.synchronize()


def _postprocess_binary_prediction(pred_hw: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.num_classes != 2:
        return pred_hw
    foreground = pred_hw.astype(bool)
    if args.postprocess_largest_component and foreground.any():
        components, count = label(foreground)
        if count > 1:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            foreground = components == int(sizes.argmax())
    elif args.postprocess_min_component_area > 0 and foreground.any():
        components, count = label(foreground)
        if count > 0:
            sizes = np.bincount(components.ravel())
            keep = sizes >= args.postprocess_min_component_area
            keep[0] = False
            foreground = keep[components]
    if args.postprocess_fill_holes and foreground.any():
        foreground = binary_fill_holes(foreground)
    return foreground.astype(pred_hw.dtype)


def main():
    args = get_args()
    if args.device is None:
        raise ValueError("Device is required")
    set_seed(args.seed)

    train_dir, test_dir = snapshot_paths(args.exp, args.dataset, args.img_size, args.output_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    args.train_snapshot_path = str(train_dir)
    args.test_snapshot_path = str(test_dir)

    if not args.test:
        logger = setup_logger(args.exp, log_dir=args.train_snapshot_path)
        train_stats = train_amortized(args, logger)
        args.train_time_sec = train_stats["train_time_sec"]
        args.train_epochs_ran = train_stats["train_epochs_ran"]
        args.train_time_per_epoch_sec = train_stats["train_time_per_epoch_sec"]
        (Path(args.train_snapshot_path) / "train.complete").touch()

    logger = setup_logger(args.exp, log_dir=args.test_snapshot_path)
    test_amortized(args, logger)


if __name__ == "__main__":
    main()
