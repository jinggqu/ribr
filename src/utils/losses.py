import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch import nn


def _spatial_size_from_flat(num_points: int) -> int:
    spatial_size = round(num_points**0.5)
    if spatial_size * spatial_size != num_points:
        raise ValueError(f"Expected square flattened mask/logits, got {num_points} points")
    return spatial_size


def _foreground_boundary_map(masks_hw, num_classes: int, kernel_size: int):
    masks_oh = F.one_hot(masks_hw.long(), num_classes).permute(0, 3, 1, 2).float()
    foreground = masks_oh[:, 1:].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    padding = kernel_size // 2
    dilated = F.max_pool2d(foreground, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-foreground, kernel_size=kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp(0.0, 1.0).squeeze(1)


def _signed_distance_targets(masks_hw: torch.Tensor, truncation: float) -> torch.Tensor:
    targets = []
    masks_np = masks_hw.detach().cpu().numpy() > 0
    for mask_np in masks_np:
        if mask_np.all():
            signed = torch.ones(mask_np.shape, dtype=torch.float32)
        elif not mask_np.any():
            signed = -torch.ones(mask_np.shape, dtype=torch.float32)
        else:
            inside = distance_transform_edt(mask_np)
            outside = distance_transform_edt(~mask_np)
            signed_np = inside - outside
            signed_np = signed_np.clip(-truncation, truncation) / max(truncation, 1e-6)
            signed = torch.from_numpy(signed_np).float()
        targets.append(signed)
    return torch.stack(targets, dim=0).to(device=masks_hw.device, dtype=torch.float32)


class DiceSemanticLoss(nn.Module):
    def __init__(self, num_classes=2, alpha=1.0, smoothing=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.smoothing = smoothing

    def forward(self, data_packet):
        masks = data_packet["masks"]
        logits = data_packet["output"]["segmentation_output"]
        probs = F.softmax(logits, dim=-1)
        masks_oh = F.one_hot(masks.long(), self.num_classes).float()
        intersection = (probs * masks_oh).sum(dim=1)
        union = probs.sum(dim=1) + masks_oh.sum(dim=1)
        dice = (2 * intersection + self.smoothing) / (union + self.smoothing)
        return (1 - dice.mean()) * self.alpha


class FocalSemanticLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, data_packet):
        masks = data_packet["masks"]
        segmentation_output = data_packet["output"]["segmentation_output"]
        seg_prob_log = nn.functional.log_softmax(segmentation_output, dim=-1)

        grouped_seg_log_probs = torch.gather(seg_prob_log, dim=-1, index=masks.unsqueeze(-1)).reshape(-1)

        scaled_grouped_probs = (1 - torch.exp(grouped_seg_log_probs)) ** self.gamma
        loss = -1 * scaled_grouped_probs * grouped_seg_log_probs

        return loss.mean() * self.alpha


class CoarseSemanticLoss(nn.Module):
    def __init__(self, num_classes=2, alpha=1.0, smoothing=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.smoothing = smoothing

    def forward(self, data_packet):
        masks = data_packet["masks"]
        coarse_logits = data_packet["output"]["coarse_segmentation_output"]
        probs = F.softmax(coarse_logits, dim=-1)
        masks_oh = F.one_hot(masks.long(), self.num_classes).float()

        ce_loss = F.cross_entropy(coarse_logits.reshape(-1, self.num_classes), masks.reshape(-1).long())
        intersection = (probs * masks_oh).sum(dim=1)
        union = probs.sum(dim=1) + masks_oh.sum(dim=1)
        dice = (2 * intersection + self.smoothing) / (union + self.smoothing)
        dice_loss = 1 - dice.mean()
        return (ce_loss + dice_loss) * self.alpha


class BoundarySemanticLoss(nn.Module):
    def __init__(
        self,
        num_classes=2,
        alpha=1.0,
        boundary_factor=2.0,
        kernel_size=5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.boundary_factor = boundary_factor
        self.kernel_size = kernel_size

    def forward(self, data_packet):
        masks = data_packet["masks"]
        logits = data_packet["output"]["segmentation_output"]
        batch_size, num_points, _ = logits.shape
        spatial_size = _spatial_size_from_flat(num_points)
        masks_hw = masks.view(batch_size, spatial_size, spatial_size)
        logits_hw = logits.view(batch_size, spatial_size, spatial_size, self.num_classes).permute(0, 3, 1, 2)

        boundary = _foreground_boundary_map(masks_hw, self.num_classes, self.kernel_size)
        weights = 1.0 + self.boundary_factor * boundary
        ce_map = F.cross_entropy(logits_hw, masks_hw.long(), reduction="none")
        loss = (ce_map * weights).sum() / weights.sum().clamp_min(1.0)
        return loss * self.alpha


class BoundaryBandSDFLoss(nn.Module):
    """Signed-distance supervision restricted to a band around the GT contour.

    A dedicated geometry head is supervised with an L1 SDF target only where
    |GT_SDF| < band, so geometry learning is decoupled from domain-specific
    speckle statistics that dominate far-from-boundary regions.
    """

    def __init__(self, num_classes=2, alpha=1.0, truncation=16.0, band_pixels=4.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.truncation = float(truncation)
        self.band = float(band_pixels) / max(float(truncation), 1.0)

    def forward(self, data_packet):
        masks = data_packet["masks"]
        sdf_pred = data_packet["output"].get("sdf_output")
        if sdf_pred is None:
            return torch.zeros((), device=masks.device, dtype=torch.float32)
        batch_size, num_points, _ = sdf_pred.shape
        spatial_size = _spatial_size_from_flat(num_points)
        masks_hw = masks.view(batch_size, spatial_size, spatial_size)
        sdf_target = _signed_distance_targets(masks_hw, truncation=self.truncation)
        sdf_pred_hw = sdf_pred.view(batch_size, spatial_size, spatial_size)
        band = (sdf_target.abs() < self.band).float()
        denom = band.sum().clamp_min(1.0)
        loss = (sdf_pred_hw - sdf_target).abs()
        return (loss * band).sum() / denom * self.alpha


class SpeckleAwareSmoothnessLoss(nn.Module):
    """Penalize foreground-probability oscillation away from the boundary band.

    Weights decrease with local speckle strength (image intensity std). Uniform regions are forced
    smooth; high-speckle regions are left alone so the model is not punished for texture.
    """

    def __init__(self, num_classes=2, alpha=1.0, boundary_kernel_size=5, speckle_lambda=4.0, window=5):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.boundary_kernel_size = boundary_kernel_size
        self.speckle_lambda = float(speckle_lambda)
        self.window = int(window)

    def forward(self, data_packet):
        masks = data_packet["masks"]
        logits = data_packet["output"]["segmentation_output"]
        images = data_packet["images"]
        batch_size, num_points, _ = logits.shape
        spatial_size = _spatial_size_from_flat(num_points)
        masks_hw = masks.view(batch_size, spatial_size, spatial_size)
        probs = F.softmax(logits, dim=-1)
        foreground_prob = probs[..., 1:].sum(dim=-1).view(batch_size, spatial_size, spatial_size)
        boundary = _foreground_boundary_map(masks_hw, self.num_classes, self.boundary_kernel_size)
        non_boundary = 1.0 - boundary

        img_hw = images.view(batch_size, spatial_size, spatial_size, -1)[..., 0].unsqueeze(1)
        k = self.window
        pad = k // 2
        mean = F.avg_pool2d(img_hw, kernel_size=k, stride=1, padding=pad)
        sq = F.avg_pool2d(img_hw * img_hw, kernel_size=k, stride=1, padding=pad)
        local_std = (sq - mean * mean).clamp_min(0.0).sqrt().squeeze(1)
        w_smooth = torch.sigmoid(-self.speckle_lambda * local_std)

        dx = (foreground_prob[:, :, 1:] - foreground_prob[:, :, :-1]).square()
        wx = torch.minimum(w_smooth[:, :, 1:], w_smooth[:, :, :-1]) * torch.minimum(
            non_boundary[:, :, 1:], non_boundary[:, :, :-1]
        )
        dy = (foreground_prob[:, 1:, :] - foreground_prob[:, :-1, :]).square()
        wy = torch.minimum(w_smooth[:, 1:, :], w_smooth[:, :-1, :]) * torch.minimum(
            non_boundary[:, 1:, :], non_boundary[:, :-1, :]
        )
        loss_x = (dx * wx).sum() / wx.sum().clamp_min(1.0)
        loss_y = (dy * wy).sum() / wy.sum().clamp_min(1.0)
        return 0.5 * (loss_x + loss_y) * self.alpha


class ResidualMagnitudeLoss(nn.Module):
    """Mean square of the ungated residual over batch, pixels, and both classes."""

    def __init__(self, alpha=0.06):
        super().__init__()
        self.alpha = alpha

    def forward(self, data_packet):
        residual = data_packet["output"]["boundary_residual_output"]
        if residual is None:
            raise ValueError("Residual magnitude supervision requires an active BRI branch")
        return residual.square().mean() * self.alpha


class LossFunction(nn.Module):
    def __init__(self, loss_funcs):
        super().__init__()
        self.loss_funcs = loss_funcs

    def forward(self, data_packet):
        total_loss = 0.0
        loss_dict = {}
        for key, loss_func in self.loss_funcs.items():
            loss_val = loss_func(data_packet)
            total_loss += loss_val
            loss_dict[key] = float(loss_val.item())

        return total_loss, loss_dict
