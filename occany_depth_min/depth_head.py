"""PromptDA-style DPT head and dense metric-depth loss."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualConvUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + residual


class _FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        prompt_depth_enabled: bool = False,
        prompt_depth_channels: int = 2,
    ) -> None:
        super().__init__()
        self.res1 = _ResidualConvUnit(channels)
        self.res2 = _ResidualConvUnit(channels)
        self.prompt_depth_enabled = bool(prompt_depth_enabled)
        if self.prompt_depth_enabled:
            self.res_prompt_depth = nn.Sequential(
                nn.Conv2d(int(prompt_depth_channels), channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            )
            nn.init.zeros_(self.res_prompt_depth[-1].weight)
            nn.init.zeros_(self.res_prompt_depth[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
        prompt_depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
            x = x + self.res1(skip)
        x = self.res2(x)
        if self.prompt_depth_enabled and prompt_depth is not None:
            prompt = F.interpolate(
                prompt_depth,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            x = x + self.res_prompt_depth(prompt)
        return x


class _PromptDAResidualConvUnit(nn.Module):
    """Residual unit used by PromptDA's released refinement blocks."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        # PromptDA keeps this activation out-of-place so the skip branch remains
        # the unmodified input tensor.
        self.activation = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.activation(x))
        out = self.conv2(self.activation(out))
        return out + x


class _PromptDAFeatureFusionBlock(nn.Module):
    """PromptDA refinement: fuse, inject depth, upsample, then 1x1 project."""

    def __init__(
        self,
        channels: int,
        prompt_depth_enabled: bool = False,
        prompt_depth_channels: int = 1,
    ) -> None:
        super().__init__()
        self.res1 = _PromptDAResidualConvUnit(channels)
        self.res2 = _PromptDAResidualConvUnit(channels)
        self.prompt_depth_enabled = bool(prompt_depth_enabled)
        if self.prompt_depth_enabled:
            activation = nn.ReLU(inplace=False)
            self.res_prompt_depth = nn.Sequential(
                nn.Conv2d(int(prompt_depth_channels), channels, kernel_size=3, padding=1),
                activation,
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            )
            nn.init.zeros_(self.res_prompt_depth[-1].weight)
            nn.init.zeros_(self.res_prompt_depth[-1].bias)
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
        prompt_depth: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                raise RuntimeError(
                    "PromptDA refinement inputs must already share a spatial size; "
                    f"got x={tuple(x.shape[-2:])}, skip={tuple(skip.shape[-2:])}."
                )
            x = x + self.res1(skip)
        x = self.res2(x)
        if self.prompt_depth_enabled and prompt_depth is not None:
            prompt = F.interpolate(
                prompt_depth,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            x = x + self.res_prompt_depth(prompt)

        resize_kwargs = {"scale_factor": 2.0} if size is None else {"size": size}
        x = F.interpolate(
            x,
            **resize_kwargs,
            mode="bilinear",
            align_corners=True,
        )
        return self.out_conv(x)


class SingleScaleDPTDepthHead(nn.Module):
    """DPT-style dense depth head for one or four post-fusion token maps.

    Depth-Anything-3's DPT head fuses four transformer feature levels. This
    head keeps the same projection/resize/fusion idea. The legacy interface
    synthesizes the pyramid from one ``(B, N, H_t, W_t, C)`` token map, while
    the multi-layer interface maps four ordered transformer outputs directly
    to the four pyramid branches.
    """

    def __init__(
        self,
        token_dim: int = 768,
        patch_size: int = 16,
        features: int = 128,
        out_channels: Tuple[int, int, int, int] = (96, 192, 384, 384),
        initial_depth: float = 10.0,
        prompt_depth_enabled: bool = False,
        prompt_depth_scale: str = "log",
        prompt_depth_min: float = 1e-3,
        prompt_depth_max: float = 120.0,
        depth_output_mode: str = "metric",
        refinement_style: str = "baseline",
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.prompt_depth_enabled = bool(prompt_depth_enabled)
        self.prompt_depth_scale = str(prompt_depth_scale)
        self.prompt_depth_min = float(prompt_depth_min)
        self.prompt_depth_max = float(prompt_depth_max)
        self.depth_output_mode = str(depth_output_mode)
        self.refinement_style = str(refinement_style)
        valid_refinement_styles = ("baseline", "promptda")
        if self.refinement_style not in valid_refinement_styles:
            raise ValueError(
                f"refinement_style must be one of {valid_refinement_styles}, "
                f"got {self.refinement_style!r}."
            )
        valid_output_modes = ("metric", "prompt_minmax", "normalized_sigmoid")
        if self.depth_output_mode not in valid_output_modes:
            raise ValueError(
                f"depth_output_mode must be one of {valid_output_modes}, "
                f"got {self.depth_output_mode!r}."
            )
        if self.prompt_depth_enabled:
            valid_scales = ("log", "linear", "per_frame_max", "per_frame_minmax")
            if self.prompt_depth_scale not in valid_scales:
                raise ValueError(
                    f"prompt_depth_scale must be one of {valid_scales}, "
                    f"got {self.prompt_depth_scale!r}."
                )
            if self.prompt_depth_min <= 0.0:
                raise ValueError("prompt_depth_min must be > 0.")
            if self.prompt_depth_max <= self.prompt_depth_min:
                raise ValueError("prompt_depth_max must be > prompt_depth_min.")
        if self.depth_output_mode == "prompt_minmax" and not (
            self.prompt_depth_enabled and self.prompt_depth_scale == "per_frame_minmax"
        ):
            raise ValueError(
                "depth_output_mode='prompt_minmax' requires prompt_depth_enabled=True "
                "and prompt_depth_scale='per_frame_minmax'."
            )
        self.norm = nn.LayerNorm(int(token_dim))
        self.projects = nn.ModuleList(
            [nn.Conv2d(int(token_dim), int(c), kernel_size=1) for c in out_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )
        self.adapters = nn.ModuleList(
            [nn.Conv2d(int(c), int(features), kernel_size=3, padding=1) for c in out_channels]
        )
        # Cached KNN prompts are dense and need no validity-mask channel.  The
        # older sparse/log/linear modes retain their original depth+mask input
        # contract for backward compatibility.
        prompt_depth_channels = 1 if self.prompt_depth_scale == "per_frame_minmax" else 2
        refinement_block = (
            _PromptDAFeatureFusionBlock
            if self.refinement_style == "promptda"
            else _FeatureFusionBlock
        )
        self.refinenet1 = refinement_block(
            int(features), self.prompt_depth_enabled, prompt_depth_channels
        )
        self.refinenet2 = refinement_block(
            int(features), self.prompt_depth_enabled, prompt_depth_channels
        )
        self.refinenet3 = refinement_block(
            int(features), self.prompt_depth_enabled, prompt_depth_channels
        )
        self.refinenet4 = refinement_block(
            int(features), self.prompt_depth_enabled, prompt_depth_channels
        )
        self.output_conv1 = nn.Conv2d(int(features), int(features) // 2, kernel_size=3, padding=1)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(int(features) // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        if self.depth_output_mode == "metric":
            nn.init.constant_(self.output_conv2[-1].bias, float(initial_depth))

    def _fuse_after_refinenet3(
        self,
        path: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
        image_hw: torch.Tensor,
        fusion_context: Optional[Dict[str, object]],
    ) -> torch.Tensor:
        """Optional DPT-mid fusion hook; the base head remains unchanged."""
        del batch_size, num_frames, image_hw, fusion_context
        return path

    def forward(
        self,
        tokens: torch.Tensor | Sequence[torch.Tensor],
        image_hw: torch.Tensor,
        prompt_depth: Optional[torch.Tensor] = None,
        fusion_context: Optional[Dict[str, object]] = None,
    ) -> torch.Tensor:
        if isinstance(tokens, torch.Tensor):
            stage_tokens = (tokens,) * 4
        else:
            stage_tokens = tuple(tokens)
            if len(stage_tokens) != 4:
                raise RuntimeError(
                    "Multi-layer DPT input must contain exactly four feature "
                    f"maps, got {len(stage_tokens)}."
                )
            if not all(isinstance(value, torch.Tensor) for value in stage_tokens):
                raise RuntimeError("Every multi-layer DPT feature must be a tensor.")

        reference = stage_tokens[0]
        if reference.ndim != 5:
            raise RuntimeError(
                "DPT tokens must be (B,N,H_t,W_t,C), got "
                f"{tuple(reference.shape)}."
            )
        B, N, H_t, W_t, C = reference.shape
        for index, stage in enumerate(stage_tokens[1:], start=1):
            if stage.shape != reference.shape:
                raise RuntimeError(
                    "All four DPT feature layers must share a shape; layer 0 is "
                    f"{tuple(reference.shape)} and layer {index} is "
                    f"{tuple(stage.shape)}."
                )
        prompt_min = None
        prompt_max = None
        if prompt_depth is None:
            prompt = None
        elif self.prompt_depth_scale == "per_frame_minmax":
            prompt, prompt_min, prompt_max = self._normalize_dense_prompt_minmax(
                prompt_depth, B, N
            )
        else:
            prompt = self._normalize_prompt_depth(prompt_depth, B, N)

        if self.depth_output_mode == "prompt_minmax" and prompt is None:
            raise RuntimeError(
                "depth_output_mode='prompt_minmax' requires a dense prompt_depth "
                "for every forward pass."
            )

        feats = []
        for stage, project, resize, adapter in zip(
            stage_tokens,
            self.projects,
            self.resize_layers,
            self.adapters,
        ):
            x = self.norm(stage)
            x = x.reshape(B * N, H_t, W_t, C).permute(0, 3, 1, 2).contiguous()
            feats.append(adapter(resize(project(x))))

        if self.refinement_style == "promptda":
            # Match PromptDA's released decoder ordering: every block upsamples
            # its result and applies a learned 1x1 projection before the next
            # skip feature is fused.  The final block performs its default 2x
            # upsample as in the original implementation.
            path = self.refinenet4(
                feats[3],
                prompt_depth=prompt,
                size=feats[2].shape[-2:],
            )
            path = self.refinenet3(
                path,
                feats[2],
                prompt_depth=prompt,
                size=feats[1].shape[-2:],
            )
            path = self._fuse_after_refinenet3(
                path,
                batch_size=B,
                num_frames=N,
                image_hw=image_hw,
                fusion_context=fusion_context,
            )
            path = self.refinenet2(
                path,
                feats[1],
                prompt_depth=prompt,
                size=feats[0].shape[-2:],
            )
            path = self.refinenet1(path, feats[0], prompt_depth=prompt)
        else:
            path = self.refinenet4(feats[3], prompt_depth=prompt)
            path = self.refinenet3(path, feats[2], prompt_depth=prompt)
            path = self._fuse_after_refinenet3(
                path,
                batch_size=B,
                num_frames=N,
                image_hw=image_hw,
                fusion_context=fusion_context,
            )
            path = self.refinenet2(path, feats[1], prompt_depth=prompt)
            path = self.refinenet1(path, feats[0], prompt_depth=prompt)

        H_img = int(image_hw[0, 0].item())
        W_img = int(image_hw[0, 1].item())
        if not bool((image_hw[:, 0] == H_img).all().item()) or not bool(
            (image_hw[:, 1] == W_img).all().item()
        ):
            raise RuntimeError("SingleScaleDPTDepthHead expects same image_hw within a batch.")

        path = self.output_conv1(path)
        path = F.interpolate(path, size=(H_img, W_img), mode="bilinear", align_corners=True)
        logits = self.output_conv2(path)
        if self.depth_output_mode == "prompt_minmax":
            if prompt_min is None or prompt_max is None:
                raise RuntimeError("Prompt min/max bounds were not computed.")
            # PromptDA predicts a normalized depth in [0, 1], then restores
            # metric scale from the original per-frame prompt bounds.
            normalized_depth = torch.sigmoid(logits.float())
            depth = normalized_depth * (prompt_max - prompt_min) + prompt_min
        elif self.depth_output_mode == "normalized_sigmoid":
            # Keep range restoration outside the head so non-prompt sources
            # (for example projected LiDAR bounds) can share this decoder.
            depth = torch.sigmoid(logits.float())
        else:
            depth = F.softplus(logits.float()).to(dtype=reference.dtype) + 1e-3
        return depth.view(B, N, H_img, W_img)

    @staticmethod
    def _normalize_dense_prompt_minmax(
        prompt_depth: torch.Tensor,
        B: int,
        N: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prompt_depth.ndim != 4 or prompt_depth.shape[:2] != (B, N):
            raise RuntimeError(
                "dense min-max prompt_depth must be (B,N,H,W) and match "
                f"tokens B,N={(B, N)}; got {tuple(prompt_depth.shape)}."
            )
        depth = prompt_depth.reshape(
            B * N, 1, prompt_depth.shape[-2], prompt_depth.shape[-1]
        ).float()
        if not bool(torch.isfinite(depth).all().item()) or not bool(
            (depth > 0.0).all().item()
        ):
            raise RuntimeError(
                "dense min-max prompt_depth must be finite and strictly positive."
            )
        flat = depth.flatten(1)
        min_value = flat.amin(dim=1).view(B * N, 1, 1, 1)
        max_value = flat.amax(dim=1).view(B * N, 1, 1, 1)
        depth_range = max_value - min_value
        if not bool((depth_range > 1e-6).all().item()):
            raise RuntimeError(
                "dense min-max prompt_depth must have a non-degenerate "
                "depth range in every frame."
            )
        normalized = ((depth - min_value) / depth_range).clamp(0.0, 1.0)
        return normalized, min_value, max_value

    def _normalize_prompt_depth(
        self,
        prompt_depth: torch.Tensor,
        B: int,
        N: int,
    ) -> torch.Tensor:
        if self.prompt_depth_scale == "per_frame_minmax":
            normalized, _, _ = self._normalize_dense_prompt_minmax(prompt_depth, B, N)
            return normalized

        if prompt_depth.ndim == 4:
            if prompt_depth.shape[:2] != (B, N):
                raise RuntimeError(
                    f"prompt_depth must match tokens B,N={(B, N)}, got {tuple(prompt_depth.shape)}."
                )
            depth = prompt_depth
            valid = torch.isfinite(depth) & (depth > 0.0)
        elif prompt_depth.ndim == 5:
            if prompt_depth.shape[:3] != (B, N, 2):
                raise RuntimeError(
                    "prompt_depth must be (B,N,2,H,W) when a mask channel is provided; "
                    f"got {tuple(prompt_depth.shape)}."
                )
            depth = prompt_depth[:, :, 0]
            valid = prompt_depth[:, :, 1] > 0.5
            valid = valid & torch.isfinite(depth) & (depth > 0.0)
        else:
            raise RuntimeError(
                "prompt_depth must be (B,N,H,W) or (B,N,2,H,W); "
                f"got {tuple(prompt_depth.shape)}."
            )

        depth = depth.reshape(B * N, 1, depth.shape[-2], depth.shape[-1]).float()
        valid = valid.reshape(B * N, 1, valid.shape[-2], valid.shape[-1])
        depth = torch.where(valid, depth, torch.zeros_like(depth))

        if self.prompt_depth_scale == "per_frame_max":
            max_val = depth.flatten(1).amax(dim=1).view(B * N, 1, 1, 1).clamp_min(1e-6)
            depth_norm = depth / max_val
        elif self.prompt_depth_scale == "linear":
            depth_clamped = depth.clamp(min=self.prompt_depth_min, max=self.prompt_depth_max)
            depth_norm = (depth_clamped - self.prompt_depth_min) / (
                self.prompt_depth_max - self.prompt_depth_min
            )
        else:
            depth_clamped = depth.clamp(min=self.prompt_depth_min, max=self.prompt_depth_max)
            denom = math.log(self.prompt_depth_max) - math.log(self.prompt_depth_min)
            depth_norm = (torch.log(depth_clamped) - math.log(self.prompt_depth_min)) / denom

        depth_norm = torch.where(valid, depth_norm.clamp(0.0, 1.0), torch.zeros_like(depth_norm))
        return torch.cat([depth_norm, valid.to(dtype=depth_norm.dtype)], dim=1)


def dense_metric_depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    frame_mask: Optional[torch.Tensor] = None,
    loss_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked metric dense-depth loss.

    Uses log-depth L1 plus relative L1 over finite positive depth pixels. Frames
    with no dense depth are skipped by ``frame_mask`` or naturally by the valid
    pixel mask.
    """
    if pred_depth.shape != gt_depth.shape:
        raise RuntimeError(
            f"pred_depth shape {tuple(pred_depth.shape)} != gt_depth {tuple(gt_depth.shape)}"
        )

    device_type = pred_depth.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        pred = pred_depth.float().clamp(min=1e-3, max=120.0)
        gt = gt_depth.to(device=pred.device, dtype=torch.float32)
        valid = torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0.0)
        if frame_mask is not None:
            fm = frame_mask.to(device=pred.device, dtype=torch.bool).view(
                pred.shape[0], pred.shape[1], 1, 1
            )
            valid = valid & fm

        valid_count = valid.sum()
        frame_count = valid.view(pred.shape[0], pred.shape[1], -1).any(dim=-1).sum()
        if not bool(valid_count.item()):
            zero = pred.sum() * 0.0
            return zero, zero.detach(), valid_count.float(), frame_count.float()

        pred_v = pred[valid]
        gt_v = gt[valid].clamp(min=1e-3, max=120.0)
        log_l1 = F.l1_loss(torch.log(pred_v), torch.log(gt_v), reduction="mean")
        rel_l1 = (pred_v - gt_v).abs().div(gt_v.clamp(min=1.0)).mean()
        raw_loss = log_l1 + rel_l1
        weighted_loss = float(loss_weight) * raw_loss
    return weighted_loss, raw_loss.detach(), valid_count.float(), frame_count.float()
