"""Small tensor-shape helpers shared by training entrypoints."""

from __future__ import annotations

import numpy as np
import torch

from src.utils.tools import get_coords2d


def probe_coordinate_feature_dim(mode: str) -> int:
    if mode in {"polar", "ray"}:
        return 3
    if mode == "hybrid":
        return 5
    raise ValueError(f"Unsupported probe coordinate mode: {mode}")


def infer_square_spatial_shape(num_points: int) -> tuple[int, int]:
    side = int(round(num_points**0.5))
    if side * side != num_points:
        raise ValueError(f"Expected a square flattened image, got {num_points} pixels")
    return side, side


def expand_coords(coords: torch.Tensor, batch_size: int) -> torch.Tensor:
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    if coords.shape[0] != batch_size:
        coords = coords.expand(batch_size, -1, -1)
    return coords


def coords_for_shape(
    height: int,
    width: int,
    device: torch.device | str,
    cache: dict[tuple[int, int], torch.Tensor],
) -> torch.Tensor:
    key = (height, width)
    coords = cache.get(key)
    if coords is None:
        coords = get_coords2d(height, width).float().to(device)
        cache[key] = coords
    return coords


def probe_coordinate_features_map(
    height: int,
    width: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    mode: str = "polar",
) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    axial = (grid_y + 1.0) * 0.5
    radius = torch.sqrt(grid_x.square() + axial.square() + 1e-6)
    if mode == "polar":
        angle = torch.atan2(grid_x, axial + 1e-4) / np.pi
        features = [axial, radius, angle]
    elif mode == "ray":
        beam_cos = axial / radius.clamp_min(1e-4)
        beam_sin = grid_x / radius.clamp_min(1e-4)
        features = [axial, beam_cos, beam_sin]
    elif mode == "hybrid":
        angle = torch.atan2(grid_x, axial + 1e-4) / np.pi
        beam_cos = axial / radius.clamp_min(1e-4)
        beam_sin = grid_x / radius.clamp_min(1e-4)
        features = [axial, radius, angle, beam_cos, beam_sin]
    else:
        raise ValueError(f"Unsupported probe coordinate mode: {mode}")
    return torch.stack(features, dim=0).unsqueeze(0)


def flat_images_to_bchw(
    images: torch.Tensor,
    *,
    in_channels: int,
    img_size: int | None = None,
) -> torch.Tensor:
    """Convert either (B, N, C) or (B, C, H, W) images to BCHW."""
    if images.dim() == 4:
        return images
    if images.dim() != 3:
        raise ValueError(f"Unsupported image shape: {tuple(images.shape)}")

    if img_size is None:
        height, width = infer_square_spatial_shape(images.shape[1])
    else:
        height = width = img_size
    return images.permute(0, 2, 1).contiguous().view(-1, in_channels, height, width)
