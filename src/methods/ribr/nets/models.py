import numpy as np
import torch
from torch import nn

from src.utils.tensors import probe_coordinate_feature_dim, probe_coordinate_features_map


class SineLayer(nn.Module):
    """SIREN activation layer. See Sitzmann et al., NeurIPS 2020, Sec 3.2."""

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, init_weights=True):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if init_weights:
            self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0
                )

    def forward(self, inputs):
        return torch.sin(self.omega_0 * self.linear(inputs))


def _coords_from_shape(height, width, device, dtype):
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack([grid_x, grid_y], dim=-1)
    return coords.view(1, height * width, 2)


def _foreground_moments_from_probs(coarse_probs):
    height, width = coarse_probs.shape[-2:]
    base_coords = _coords_from_shape(height, width, coarse_probs.device, coarse_probs.dtype)
    base_coords = base_coords.expand(coarse_probs.shape[0], -1, -1)
    foreground = (1.0 - coarse_probs[:, :1]).reshape(coarse_probs.shape[0], -1, 1)
    mass = foreground.sum(dim=1, keepdim=True).clamp_min(1e-4)
    center = (foreground * base_coords).sum(dim=1, keepdim=True) / mass
    centered = base_coords - center
    return foreground, mass, center, centered


def _compute_isotropic_relative_coords(coords, foreground, mass, center, centered):
    """Historical name: the scale is the probability-weighted SD of each axis."""
    scale = torch.sqrt((foreground * centered.square()).sum(dim=1, keepdim=True) / mass + 1e-4).clamp_min(0.20)
    relative = (coords - center) / scale
    return relative.clamp(-4.0, 4.0)


def _compute_relative_coords_from_probs(coords, coarse_probs):
    foreground, mass, center, centered = _foreground_moments_from_probs(coarse_probs)
    return _compute_isotropic_relative_coords(coords, foreground, mass, center, centered)


def _confidence_from_probs(probs):
    eps = 1e-6
    max_entropy = float(np.log(max(probs.shape[-1] if probs.dim() == 3 else probs.shape[1], 2)))
    if probs.dim() == 4:
        entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=1, keepdim=True)
        return 1.0 - entropy / max_entropy
    entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1, keepdim=True)
    return 1.0 - entropy / max_entropy


class SpatialSegHead(nn.Module):
    """Conv-based segmentation head with multi-scale feature fusion and residual connections."""

    def __init__(self, in_features, hidden_features, output_features, img_size):
        super().__init__()
        self.img_size = img_size
        self.fuse = nn.Linear(in_features, hidden_features)
        self.res_block = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, 1, 1),
            nn.GroupNorm(8, hidden_features),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_features, hidden_features, 3, 1, 1),
            nn.GroupNorm(8, hidden_features),
        )
        self.act = nn.LeakyReLU()
        self.conv_down = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features // 2, 3, 1, 1),
            nn.GroupNorm(4, hidden_features // 2),
            nn.LeakyReLU(),
        )
        self.head = nn.Conv2d(hidden_features // 2, output_features, 1)

    def forward(self, features, spatial_shape=None):
        batch = features.shape[0]
        if spatial_shape is None:
            height = width = self.img_size
        else:
            height, width = spatial_shape
        x = self.fuse(features)
        x = x.reshape(batch, height, width, -1).permute(0, 3, 1, 2)
        x = self.act(self.res_block(x) + x)
        x = self.conv_down(x)
        x = self.head(x)
        return x.permute(0, 2, 3, 1).reshape(batch, height * width, -1)


def _group_count(num_channels: int, max_groups: int = 8) -> int:
    for group_count in range(min(max_groups, num_channels), 0, -1):
        if num_channels % group_count == 0:
            return group_count
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + residual)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualConvBlock(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UpsampleFuseBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, skip_channels, 1, bias=False)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.block = ResidualConvBlock(skip_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(self.proj(x))
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ImageEncoder(nn.Module):
    """U-Net style encoder-decoder that also exposes coarse semantics for INR conditioning."""

    def __init__(
        self,
        in_channels=1,
        base_channels=16,
        latent_dim=64,
        num_classes=2,
        use_anatomy_style_split=False,
        style_latent_dim=16,
        style_local_channels=None,
        anatomy_feature_source="d0",
        coarse_head_source="d0",
    ):
        super().__init__()
        ch0 = base_channels
        ch1 = base_channels * 2
        ch2 = base_channels * 4
        ch3 = base_channels * 8
        self.use_anatomy_style_split = use_anatomy_style_split
        self.anatomy_feature_source = anatomy_feature_source
        self.coarse_head_source = coarse_head_source

        self.stem = ResidualConvBlock(in_channels, ch0)
        self.down1 = DownsampleBlock(ch0, ch1)
        self.down2 = DownsampleBlock(ch1, ch2)
        self.down3 = DownsampleBlock(ch2, ch3)
        self.up2 = UpsampleFuseBlock(ch3, ch2, ch2)
        self.up1 = UpsampleFuseBlock(ch2, ch1, ch1)
        self.up0 = UpsampleFuseBlock(ch1, ch0, ch0)

        self.local_channels = ch1
        source_channels = {"d0": ch0, "d1": ch1, "d2": ch2, "d3": ch3}
        if self.anatomy_feature_source not in source_channels:
            raise ValueError(f"Unsupported anatomy feature source: {self.anatomy_feature_source}")
        if self.coarse_head_source not in source_channels:
            raise ValueError(f"Unsupported coarse head source: {self.coarse_head_source}")
        self.local_proj = nn.Conv2d(source_channels[self.anatomy_feature_source], self.local_channels, 1)
        self.coarse_head = nn.Conv2d(source_channels[self.coarse_head_source], num_classes, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch3, latent_dim)
        if self.use_anatomy_style_split:
            if style_local_channels is None:
                style_local_channels = max(self.local_channels // 2, ch0)
            self.style_local_channels = style_local_channels
            self.style_local_proj = nn.Sequential(
                nn.Conv2d(ch0 + ch1, self.style_local_channels, 3, stride=1, padding=1, bias=False),
                nn.GroupNorm(_group_count(self.style_local_channels), self.style_local_channels),
                nn.LeakyReLU(inplace=True),
            )
            self.style_fc = nn.Linear(ch1 + ch3, style_latent_dim)
        else:
            self.style_local_channels = 0
            self.style_local_proj = None
            self.style_fc = None

    def forward(self, images, return_feature_map=False):
        x0 = self.stem(images)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        d2 = self.up2(x3, x2)
        d1 = self.up1(d2, x1)
        d0 = self.up0(d1, x0)

        source_maps = {"d0": d0, "d1": d1, "d2": d2, "d3": x3}
        latent = self.fc(self.pool(x3).flatten(1))
        # When anatomy/style split is enabled, deeper anatomy/coarse sources reduce shallow texture leakage.
        feature_map = self.local_proj(source_maps[self.anatomy_feature_source])
        if feature_map.shape[-2:] != d0.shape[-2:]:
            feature_map = nn.functional.interpolate(
                feature_map, size=d0.shape[-2:], mode="bilinear", align_corners=True
            )
        coarse_logits = self.coarse_head(source_maps[self.coarse_head_source])
        if coarse_logits.shape[-2:] != d0.shape[-2:]:
            coarse_logits = nn.functional.interpolate(
                coarse_logits,
                size=d0.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        encoder_out = {
            "latent": latent,
            "feature_map": feature_map,
            "coarse_logits": coarse_logits,
            "anatomy_latent": latent,
            "anatomy_feature_map": feature_map,
            "style_latent": None,
            "style_feature_map": None,
        }
        if self.use_anatomy_style_split:
            d1_up = nn.functional.interpolate(d1, size=d0.shape[-2:], mode="bilinear", align_corners=True)
            style_feature_map = self.style_local_proj(torch.cat([d0, d1_up], dim=1))
            style_latent = self.style_fc(torch.cat([self.pool(d1).flatten(1), self.pool(x3).flatten(1)], dim=1))
            encoder_out["style_latent"] = style_latent
            encoder_out["style_feature_map"] = style_feature_map
        if return_feature_map:
            return encoder_out
        return latent


class RIBRConvBackbone(nn.Module):
    """Convolutional coarse segmentation backbone used by RIBR."""

    def __init__(
        self,
        inr_config,
        segmentation_config,
        img_size,
        add_coarse_logits_to_segmentation=True,
        coarse_segmentation_residual_scale=1.0,
    ):
        super().__init__()
        self.use_relative_coords = inr_config.get("use_relative_coords", False)
        self.use_coarse_context = inr_config.get("use_coarse_context", True)
        self.use_anatomy_style_split = inr_config.get("use_anatomy_style_split", False)
        self.use_probe_coords = inr_config.get("use_probe_coords", False)
        self.probe_coord_mode = inr_config.get("probe_coord_mode", "polar")
        self.add_coarse_logits_to_segmentation = add_coarse_logits_to_segmentation
        self.coarse_segmentation_residual_scale = coarse_segmentation_residual_scale
        self.use_occlusion_field = segmentation_config.get("use_occlusion_field", False)
        self.refine_segmentation = segmentation_config.get("refine_segmentation", False)
        self.encoder = ImageEncoder(
            in_channels=inr_config["in_channels"],
            base_channels=inr_config["encoder_channels"],
            latent_dim=inr_config["latent_dim"],
            num_classes=inr_config["num_classes"],
            use_anatomy_style_split=self.use_anatomy_style_split,
            style_latent_dim=inr_config.get("style_latent_dim", 16),
            style_local_channels=inr_config.get("style_local_channels", 8),
            anatomy_feature_source=inr_config.get("anatomy_feature_source", "d0"),
            coarse_head_source=inr_config.get("coarse_head_source", "d0"),
        )
        hidden_features = inr_config["hidden_features"]
        self.global_context_channels = max(hidden_features // 4, 8)
        self.global_proj = nn.Linear(inr_config["latent_dim"], self.global_context_channels)
        if self.use_anatomy_style_split:
            self.style_global_proj = nn.Linear(inr_config.get("style_latent_dim", 16), self.global_context_channels)
        else:
            self.style_global_proj = None

        coarse_context_channels = 2 if self.use_coarse_context else 0
        relative_context_channels = 2 if self.use_relative_coords else 0
        probe_context_channels = probe_coordinate_feature_dim(self.probe_coord_mode) if self.use_probe_coords else 0
        feature_in_channels = (
            self.encoder.local_channels
            + self.global_context_channels
            + coarse_context_channels
            + relative_context_channels
            + probe_context_channels
        )
        if self.use_anatomy_style_split:
            feature_in_channels += self.encoder.style_local_channels + self.global_context_channels

        self.feature_stem = ResidualConvBlock(feature_in_channels, hidden_features)

        seg_in_features = hidden_features + coarse_context_channels
        visibility_hidden = max(hidden_features // 2, 16)
        if self.use_occlusion_field:
            self.occlusion_head = nn.Sequential(
                nn.Linear(seg_in_features, visibility_hidden),
                nn.LeakyReLU(inplace=True),
                nn.Linear(visibility_hidden, 1),
                nn.Sigmoid(),
            )
        else:
            self.occlusion_head = None
        seg_in_features = seg_in_features + (1 if self.use_occlusion_field else 0)
        self.segmentation_head = SpatialSegHead(
            in_features=seg_in_features,
            hidden_features=segmentation_config["hidden_features"][0],
            output_features=segmentation_config["output_features"],
            img_size=img_size,
        )
        if self.refine_segmentation:
            refine_in_channels = (
                segmentation_config["output_features"] + self.encoder.local_channels + inr_config["num_classes"]
            )
            if self.use_coarse_context:
                refine_in_channels += 1
            if self.use_occlusion_field:
                refine_in_channels += 1
            if self.use_anatomy_style_split:
                refine_in_channels += self.encoder.style_local_channels
            refine_hidden = max(segmentation_config["hidden_features"][0] // 2, 16)
            self.refine_head = nn.Sequential(
                ResidualConvBlock(refine_in_channels, refine_hidden),
                ResidualConvBlock(refine_hidden, refine_hidden),
                nn.Conv2d(refine_hidden, segmentation_config["output_features"], 1),
            )
        else:
            self.refine_head = None

    @staticmethod
    def _flat_to_map(flattened, spatial_shape):
        height, width = spatial_shape
        return flattened.reshape(flattened.shape[0], height, width, -1).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _flatten_map(feature_map):
        return feature_map.permute(0, 2, 3, 1).reshape(feature_map.shape[0], -1, feature_map.shape[1])

    @staticmethod
    def _broadcast_latent(latent, projector, height, width):
        context = projector(latent).unsqueeze(-1).unsqueeze(-1)
        return context.expand(-1, -1, height, width)

    def forward(self, coords, images):
        del coords
        encoder_out = self.encoder(images, return_feature_map=True)
        feature_map = encoder_out["feature_map"]
        coarse_logits = encoder_out["coarse_logits"]
        coarse_probs = nn.functional.softmax(coarse_logits, dim=1)
        height, width = images.shape[-2:]

        feature_inputs = [feature_map, self._broadcast_latent(encoder_out["latent"], self.global_proj, height, width)]
        relative_coords = None
        if self.use_relative_coords:
            base_coords = _coords_from_shape(height, width, images.device, images.dtype)
            base_coords = base_coords.expand(images.shape[0], -1, -1)
            relative_coords = _compute_relative_coords_from_probs(base_coords, coarse_probs)
            relative_map = relative_coords.reshape(images.shape[0], height, width, 2).permute(0, 3, 1, 2).contiguous()
            feature_inputs.append(relative_map)
        probe_feature_map = None
        if self.use_probe_coords:
            probe_feature_map = probe_coordinate_features_map(
                height,
                width,
                device=images.device,
                dtype=images.dtype,
                mode=self.probe_coord_mode,
            ).expand(images.shape[0], -1, -1, -1)
            feature_inputs.append(probe_feature_map)
        coarse_foreground = None
        confidence_map = None
        if self.use_coarse_context:
            coarse_foreground = 1.0 - coarse_probs[:, :1]
            confidence_map = _confidence_from_probs(coarse_probs)
            feature_inputs.extend([coarse_foreground, confidence_map])

        style_feature_map = encoder_out["style_feature_map"]
        style_context = None
        if self.use_anatomy_style_split and style_feature_map is not None:
            style_context = self._broadcast_latent(
                encoder_out["style_latent"],
                self.style_global_proj,
                height,
                width,
            )
            feature_inputs.extend([style_feature_map, style_context])

        ribr_features_map = self.feature_stem(torch.cat(feature_inputs, dim=1))
        coarse_output = self._flatten_map(coarse_logits)
        ribr_features = self._flatten_map(ribr_features_map)
        segmentation_features = [ribr_features]
        confidence_output = None
        if coarse_foreground is not None and confidence_map is not None:
            confidence_output = self._flatten_map(confidence_map)
            segmentation_features.append(self._flatten_map(torch.cat([coarse_foreground, confidence_map], dim=1)))
        segmentation_features = torch.cat(segmentation_features, dim=-1)
        occlusion_output = self.occlusion_head(segmentation_features) if self.occlusion_head is not None else None
        seg_features = (
            torch.cat([segmentation_features, occlusion_output], dim=-1)
            if occlusion_output is not None
            else segmentation_features
        )
        segmentation_output = self.segmentation_head(seg_features, spatial_shape=(height, width))
        if self.add_coarse_logits_to_segmentation:
            segmentation_output = segmentation_output + self.coarse_segmentation_residual_scale * coarse_output
        if self.refine_head is not None:
            refine_inputs = [
                self._flat_to_map(segmentation_output, (height, width)),
                encoder_out["anatomy_feature_map"],
                coarse_logits,
            ]
            if confidence_map is not None:
                refine_inputs.append(confidence_map)
            if occlusion_output is not None:
                refine_inputs.append(self._flat_to_map(occlusion_output, (height, width)))
            if style_feature_map is not None:
                refine_inputs.append(style_feature_map)
            refine_delta = self.refine_head(torch.cat(refine_inputs, dim=1))
            segmentation_output = self._flatten_map(
                self._flat_to_map(segmentation_output, (height, width)) + refine_delta
            )
        return {
            "ribr_features": ribr_features,
            "occlusion_output": occlusion_output,
            "segmentation_output": segmentation_output,
            "coarse_segmentation_output": coarse_output,
            "coarse_segmentation_map": coarse_logits,
            "coarse_foreground_map": coarse_foreground,
            "confidence_output": confidence_output,
            "confidence_map": confidence_map,
            "relative_coords": relative_coords,
            "probe_features": self._flatten_map(probe_feature_map) if probe_feature_map is not None else None,
            "anatomy_feature_map": encoder_out["anatomy_feature_map"],
            "style_feature_map": encoder_out["style_feature_map"],
        }


class BoundaryRefinementImplicitResidual(nn.Module):
    """Module B: Boundary-Refinement Implicit Residual.

    The convolutional decoder predicts the coarse mask. This module predicts only an
    implicit residual and writes it back through a boundary/risk gate.
    """

    def __init__(
        self,
        feature_channels,
        num_classes,
        residual_hidden=32,
        residual_layers=2,
        gate_tau=0.65,
        gate_gamma=12.0,
        first_omega_0=20.0,
        hidden_omega_0=15.0,
    ):
        super().__init__()
        self.gate_tau = float(gate_tau)
        self.gate_gamma = float(gate_gamma)
        residual_in = 2 + 2 + feature_channels + 2 + 1 + 1
        self.residual_layers = nn.ModuleList()
        self.residual_layers.append(SineLayer(residual_in, residual_hidden, is_first=True, omega_0=first_omega_0))
        for _ in range(max(residual_layers - 1, 0)):
            self.residual_layers.append(
                SineLayer(residual_hidden, residual_hidden, is_first=False, omega_0=hidden_omega_0)
            )
        self.residual_out = nn.Linear(residual_hidden, num_classes)
        with torch.no_grad():
            const = np.sqrt(6 / residual_hidden) / max(hidden_omega_0, 1e-12)
            self.residual_out.weight.uniform_(-const, const)

    @staticmethod
    def spatial_to_flat(feature_map):
        return feature_map.permute(0, 2, 3, 1).reshape(feature_map.shape[0], -1, feature_map.shape[1])

    def forward(
        self,
        coords_flat,
        relative_coords,
        feature_map,
        local_stats,
        coarse_foreground,
        confidence,
        risk_score,
    ):
        feature_flat = self.spatial_to_flat(feature_map)
        local_flat = self.spatial_to_flat(local_stats)
        fg_flat = self.spatial_to_flat(coarse_foreground)
        conf_flat = self.spatial_to_flat(confidence)

        gate = torch.sigmoid(self.gate_gamma * (risk_score - self.gate_tau))
        residual_in = torch.cat([coords_flat, relative_coords, feature_flat, local_flat, fg_flat, conf_flat], dim=-1)
        x = residual_in
        for layer in self.residual_layers:
            x = layer(x)
        residual = self.residual_out(x)
        return residual, self.spatial_to_flat(gate)


class RiskRoutedResidualControl(nn.Module):
    """Module C: Risk-Routed Residual Control.

    Detached head inputs adjust an uncertainty prior whose direct path is differentiable.
    The residual L2 term is added in training as the companion magnitude control.
    """

    def __init__(self, route_scale=0.25):
        super().__init__()
        self.route_scale = float(route_scale)
        self.risk_route_head = nn.Conv2d(6, 1, 1)
        with torch.no_grad():
            self.risk_route_head.weight.zero_()
            self.risk_route_head.bias.zero_()

    @staticmethod
    def _image_gradient(images_bchw):
        gray = images_bchw.mean(dim=1, keepdim=True)
        dx = torch.zeros_like(gray)
        dy = torch.zeros_like(gray)
        dx[:, :, :, 1:] = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs()
        dy[:, :, 1:, :] = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs()
        grad = torch.linalg.vector_norm(torch.cat([dx, dy], dim=1), dim=1, keepdim=True)
        scale = grad.detach().amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return grad / scale

    def forward(self, images, boundary_proximity, coarse_foreground, confidence, local_stats):
        image_grad = self._image_gradient(images)
        route_features = torch.cat([boundary_proximity, coarse_foreground, confidence, local_stats, image_grad], dim=1)
        risk_delta = torch.tanh(self.risk_route_head(route_features.detach())) * self.route_scale
        risk_score = (boundary_proximity + risk_delta).clamp(0.0, 1.0)
        return risk_score, risk_delta


class GeometrySpeckleBoundaryRegularizationHead(nn.Module):
    """Module D: Geometry- and Speckle-Aware Boundary Regularization head."""

    def __init__(self, feature_channels):
        super().__init__()
        self.sdf_head = nn.Conv2d(feature_channels, 1, 1)

    def forward(self, feature_map):
        return torch.tanh(self.sdf_head(feature_map))


class RIBRSegmentationModel(RIBRConvBackbone):
    """RIBR: Risk-Routed Implicit Boundary Refinement.

    The model is organized as:
    B. Boundary-Refinement Implicit Residual.
    C. Risk-Routed Residual Control.
    D. Geometry- and Speckle-Aware Boundary Regularization.
    """

    def __init__(
        self,
        inr_config,
        segmentation_config,
        img_size,
        add_coarse_logits_to_segmentation=True,
        coarse_segmentation_residual_scale=1.0,
        residual_hidden=32,
        residual_layers=2,
        gate_tau=0.65,
        gate_gamma=12.0,
        risk_route_scale=0.25,
        boundary_residual_scale=1.0,
        local_stat_window=5,
        first_omega_0=20.0,
        hidden_omega_0=15.0,
        use_bri=True,
        use_gsbr=True,
        use_rrc=True,
    ):
        if inr_config["in_channels"] != 1 or inr_config["num_classes"] != 2:
            raise ValueError("RIBR implements single-channel binary segmentation")
        if use_rrc and not use_bri:
            raise ValueError("RRC requires BRI residuals")
        if local_stat_window < 1 or local_stat_window % 2 == 0:
            raise ValueError("local_stat_window must be positive and odd")
        if residual_hidden < 1 or residual_layers < 1:
            raise ValueError("The SIREN needs positive width and at least one hidden layer")
        super().__init__(
            inr_config,
            segmentation_config,
            img_size,
            add_coarse_logits_to_segmentation,
            coarse_segmentation_residual_scale,
        )
        self.local_stat_window = int(local_stat_window)
        self.use_bri, self.use_gsbr, self.use_rrc = use_bri, use_gsbr, use_rrc
        self.boundary_residual_scale = float(boundary_residual_scale)
        feature_channels = inr_config["hidden_features"]
        self.boundary_refinement_residual = (
            BoundaryRefinementImplicitResidual(
                feature_channels=feature_channels,
                num_classes=inr_config["num_classes"],
                residual_hidden=residual_hidden,
                residual_layers=residual_layers,
                gate_tau=gate_tau,
                gate_gamma=gate_gamma,
                first_omega_0=first_omega_0,
                hidden_omega_0=hidden_omega_0,
            )
            if use_bri
            else None
        )
        self.geometry_speckle_boundary_regularizer = (
            GeometrySpeckleBoundaryRegularizationHead(feature_channels) if use_gsbr else None
        )
        self.risk_routed_control = RiskRoutedResidualControl(risk_route_scale) if use_rrc else None

    def _local_stats(self, images_bchw):
        k = self.local_stat_window
        pad = k // 2
        mean = nn.functional.avg_pool2d(images_bchw, kernel_size=k, stride=1, padding=pad)
        sq_mean = nn.functional.avg_pool2d(images_bchw * images_bchw, kernel_size=k, stride=1, padding=pad)
        std = (sq_mean - mean * mean).clamp_min(0.0).sqrt()
        return torch.cat([mean, std], dim=1)

    def forward(self, coords, images):
        if (
            images.ndim != 4
            or images.shape[1] != 1
            or min(images.shape[-2:]) < 8
            or any(side % 8 for side in images.shape[-2:])
        ):
            raise ValueError("Expected (B, 1, H, W) images with H and W positive multiples of eight")
        out = super().forward(coords, images)
        height, width = images.shape[-2:]
        batch = images.shape[0]
        device, dtype = images.device, images.dtype

        feature_map = self._flat_to_map(out["ribr_features"], (height, width))
        out["sdf_output"] = (
            self._flatten_map(self.geometry_speckle_boundary_regularizer(feature_map)) if self.use_gsbr else None
        )
        if not self.use_bri:
            out["boundary_residual_output"] = None
            out["gate_output"] = None
            return out
        local_stats = self._local_stats(images)

        coords_flat = coords.expand(batch, -1, -1)
        relative = out["relative_coords"]
        if relative is None:
            relative = torch.zeros(batch, height * width, 2, device=device, dtype=dtype)
        coarse_fg = out["coarse_foreground_map"]
        if coarse_fg is None:
            coarse_fg = torch.zeros(batch, 1, height, width, device=device, dtype=dtype)
        confidence = out["confidence_map"]
        if confidence is None:
            confidence = torch.zeros(batch, 1, height, width, device=device, dtype=dtype)

        # Predictive uncertainty is a soft prior, not geometric contour distance.
        # All pixels receive a residual evaluation; the sigmoid weights its contribution.
        fg = coarse_fg.clamp(0.0, 1.0)
        boundary_proximity = (1.0 - (fg - 0.5).abs() * 2.0).clamp(0.0, 1.0)
        if self.use_rrc:
            risk_score, risk_delta = self.risk_routed_control(images, boundary_proximity, fg, confidence, local_stats)
        else:
            risk_score, risk_delta = torch.zeros_like(fg), torch.zeros_like(fg)

        residual, alpha_flat = self.boundary_refinement_residual(
            coords_flat=coords_flat,
            relative_coords=relative,
            feature_map=feature_map,
            local_stats=local_stats,
            coarse_foreground=fg,
            confidence=confidence,
            risk_score=risk_score,
        )

        if not self.use_rrc:
            alpha_flat = torch.ones_like(alpha_flat)

        hybrid_logits = out["segmentation_output"] + self.boundary_residual_scale * alpha_flat * residual

        out["segmentation_output"] = hybrid_logits
        out["boundary_residual_output"] = residual
        out["gate_output"] = alpha_flat
        out["risk_route_delta_output"] = self._flatten_map(risk_delta)
        out["risk_route_score_output"] = self._flatten_map(risk_score)
        out["boundary_proximity_map"] = boundary_proximity
        return out
