"""Patch/depth-bin LiDAR token encoder used before DA3."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


def _sinusoidal_pe_3d(coords: torch.Tensor, num_freqs: int) -> torch.Tensor:
    freqs = (2.0 ** torch.arange(num_freqs, device=coords.device, dtype=coords.dtype)) * math.pi
    phase = (coords / 50.0).unsqueeze(-1) * freqs
    return torch.cat([phase.sin(), phase.cos()], dim=-1).flatten(-2)


class PatchDepthBinFeatureEncoder(nn.Module):
    """Encode z-buffered sparse depth as patch-by-depth-bin tokens.

    Every valid sparse-depth pixel is backprojected with the matching camera
    intrinsics.  Backprojected points are grouped by
    ``(frame, patch_y, patch_x, floor(depth / depth_bin_size))`` and reduced by
    a PointNet-style max pool.  Unlike :class:`VoxelFeatureEncoder`, this path
    intentionally consumes geometry only; LiDAR intensity is not carried
    through the sparse-depth map.
    """

    def __init__(
        self,
        *,
        d_out: int,
        H_t: int,
        W_t: int,
        patch_size: int,
        depth_bin_size: float = 4.0,
        vox_origin: Tuple[float, float, float] = (-25.6, -2.0, 0.0),
        vox_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
        vox_grid: Tuple[int, int, int] = (128, 16, 128),
        d_token: int = 128,
        hidden: int = 64,
        pe_num_freqs: int = 8,
        dynamic_image_size: bool = False,
        force_fp32_backprojection: bool = False,
    ) -> None:
        super().__init__()
        if int(H_t) <= 0 or int(W_t) <= 0 or int(patch_size) <= 0:
            raise ValueError(
                "H_t, W_t, and patch_size must be positive; got "
                f"{(H_t, W_t, patch_size)}."
            )
        if not math.isfinite(float(depth_bin_size)) or float(depth_bin_size) <= 0:
            raise ValueError(
                f"depth_bin_size must be finite and positive, got {depth_bin_size}."
            )
        self.d_out = int(d_out)
        self.H_t = int(H_t)
        self.W_t = int(W_t)
        self.patch_size = int(patch_size)
        self.dynamic_image_size = bool(dynamic_image_size)
        self.force_fp32_backprojection = bool(force_fp32_backprojection)
        self.depth_bin_size = float(depth_bin_size)
        self.vox_grid = tuple(int(value) for value in vox_grid)
        self.register_buffer(
            "vox_origin",
            torch.tensor(vox_origin, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "vox_size",
            torch.tensor(vox_size, dtype=torch.float32),
            persistent=False,
        )

        # Pure geometry: absolute camera xyz plus offset from the group mean.
        self.point_mlp = nn.Sequential(
            nn.Linear(6, int(hidden)),
            nn.LayerNorm(int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(d_token)),
        )
        self.token_norm = nn.LayerNorm(int(d_token))
        self.token_proj = nn.Linear(int(d_token), self.d_out)
        self.pe_num_freqs = int(pe_num_freqs)
        self.pe_proj = nn.Linear(3 * 2 * self.pe_num_freqs, self.d_out)

    @staticmethod
    def _per_sample_grid_value(
        value: Optional[torch.Tensor],
        default: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if value is None:
            return default.to(device=device, dtype=torch.float32).view(1, 3).expand(
                batch_size, 3
            )
        resolved = value.to(device=device, dtype=torch.float32)
        if resolved.ndim == 1:
            resolved = resolved.view(1, 3).expand(batch_size, 3)
        if tuple(resolved.shape) != (batch_size, 3):
            raise RuntimeError(
                f"{name} must be (3,) or (B,3); got {tuple(resolved.shape)}."
            )
        return resolved

    def forward(
        self,
        sparse_depth: torch.Tensor,
        sparse_mask: torch.Tensor,
        K_per_frame: torch.Tensor,
        *,
        vox_origin: Optional[torch.Tensor] = None,
        vox_size: Optional[torch.Tensor] = None,
        vox_grid: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Return token features, mean xyz, frame ids, patch rows, and columns."""
        if sparse_depth.ndim != 4:
            raise RuntimeError(
                "sparse_depth must be (B,N,H,W); got "
                f"{tuple(sparse_depth.shape)}."
            )
        if sparse_mask.shape != sparse_depth.shape:
            raise RuntimeError(
                "sparse_mask must match sparse_depth; got "
                f"{tuple(sparse_mask.shape)} vs {tuple(sparse_depth.shape)}."
            )
        batch_size, num_frames, height, width = sparse_depth.shape
        expected_hw = (self.H_t * self.patch_size, self.W_t * self.patch_size)
        if self.dynamic_image_size:
            if height % self.patch_size or width % self.patch_size:
                raise RuntimeError(
                    f"sparse depth resolution {(height, width)} must be divisible "
                    f"by patch_size={self.patch_size} in dynamic-image-size mode."
                )
        elif (height, width) != expected_hw:
            raise RuntimeError(
                f"sparse depth resolution {(height, width)} must equal {expected_hw}."
            )

        intrinsics = K_per_frame.to(device=sparse_depth.device, dtype=torch.float32)
        if intrinsics.ndim == 3 and tuple(intrinsics.shape) == (
            batch_size,
            3,
            3,
        ):
            intrinsics = intrinsics[:, None].expand(batch_size, num_frames, 3, 3)
        if tuple(intrinsics.shape) != (batch_size, num_frames, 3, 3):
            raise RuntimeError(
                "K_per_frame must be (B,3,3) or (B,N,3,3); got "
                f"{tuple(intrinsics.shape)}."
            )

        depth = sparse_depth.to(dtype=torch.float32)
        valid = sparse_mask.to(dtype=torch.bool) & torch.isfinite(depth) & (depth > 0)
        pixel_index = torch.nonzero(valid, as_tuple=False)
        if pixel_index.numel() == 0:
            return None, None, None, None, None

        sample_idx = pixel_index[:, 0]
        frame_in_sample = pixel_index[:, 1]
        pixel_y = pixel_index[:, 2]
        pixel_x = pixel_index[:, 3]
        frame_idx = sample_idx * num_frames + frame_in_sample
        point_depth = depth[
            sample_idx,
            frame_in_sample,
            pixel_y,
            pixel_x,
        ]

        pixels = torch.stack(
            [
                pixel_x.to(dtype=torch.float32),
                pixel_y.to(dtype=torch.float32),
                torch.ones_like(point_depth),
            ],
            dim=-1,
        )
        if self.force_fp32_backprojection:
            # The projected-snake layout uses sub-pixel coordinates derived
            # from these centers.  Keep its inverse/batched matmul out of the
            # caller's bf16 autocast so image-edge pixels remain geometrically
            # stable; the learned MLP/projection below still follows autocast.
            with torch.autocast(
                device_type=sparse_depth.device.type,
                enabled=False,
            ):
                inv_k = torch.linalg.inv(intrinsics.reshape(-1, 3, 3))
                rays = torch.bmm(
                    inv_k[frame_idx], pixels.unsqueeze(-1)
                ).squeeze(-1)
        else:
            inv_k = torch.linalg.inv(intrinsics.reshape(-1, 3, 3))
            rays = torch.bmm(
                inv_k[frame_idx], pixels.unsqueeze(-1)
            ).squeeze(-1)
        xyz = rays * point_depth.unsqueeze(-1)

        origins = self._per_sample_grid_value(
            vox_origin,
            self.vox_origin,
            batch_size=batch_size,
            device=sparse_depth.device,
            name="vox_origin",
        )
        sizes = self._per_sample_grid_value(
            vox_size,
            self.vox_size,
            batch_size=batch_size,
            device=sparse_depth.device,
            name="vox_size",
        )
        Gx, Gy, Gz = self.vox_grid if vox_grid is None else tuple(
            int(value) for value in vox_grid
        )
        grid_idx = torch.floor(
            (xyz - origins[sample_idx]) / sizes[sample_idx]
        ).to(dtype=torch.long)
        in_grid = (
            torch.isfinite(xyz).all(dim=-1)
            & (grid_idx[:, 0] >= 0)
            & (grid_idx[:, 0] < Gx)
            & (grid_idx[:, 1] >= 0)
            & (grid_idx[:, 1] < Gy)
            & (grid_idx[:, 2] >= 0)
            & (grid_idx[:, 2] < Gz)
        )
        if not bool(in_grid.any().item()):
            return None, None, None, None, None

        xyz = xyz[in_grid]
        point_depth = point_depth[in_grid]
        frame_idx = frame_idx[in_grid]
        pixel_y = pixel_y[in_grid]
        pixel_x = pixel_x[in_grid]

        patch_y = torch.div(pixel_y, self.patch_size, rounding_mode="floor")
        patch_x = torch.div(pixel_x, self.patch_size, rounding_mode="floor")
        depth_bin = torch.floor(point_depth / self.depth_bin_size).to(dtype=torch.long)
        group_key = torch.stack([frame_idx, patch_y, patch_x, depth_bin], dim=-1)
        groups, inverse = torch.unique(
            group_key,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        token_count = int(groups.shape[0])

        counts = xyz.new_zeros((token_count,))
        counts.scatter_add_(0, inverse, torch.ones_like(point_depth))
        centers = xyz.new_zeros((token_count, 3))
        centers.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, 3), xyz)
        centers = centers / counts.clamp_min(1.0).unsqueeze(-1)

        point_features = torch.cat([xyz, xyz - centers[inverse]], dim=-1)
        encoded_points = self.point_mlp(point_features)
        encoded_dim = int(encoded_points.shape[-1])
        neg_inf = torch.finfo(encoded_points.dtype).min
        pooled = torch.full(
            (token_count, encoded_dim),
            neg_inf,
            dtype=encoded_points.dtype,
            device=encoded_points.device,
        )
        pooled.scatter_reduce_(
            0,
            inverse.unsqueeze(-1).expand(-1, encoded_dim),
            encoded_points,
            reduce="amax",
            include_self=False,
        )
        pooled = pooled.masked_fill(pooled == neg_inf, 0.0)
        token_features = self.token_proj(self.token_norm(pooled))
        position = _sinusoidal_pe_3d(
            centers,
            self.pe_num_freqs,
        ).to(dtype=token_features.dtype)
        token_features = token_features + self.pe_proj(position)
        return (
            token_features,
            centers,
            groups[:, 0].to(dtype=torch.long),
            groups[:, 1].to(dtype=torch.long),
            groups[:, 2].to(dtype=torch.long),
        )


# ============================================================================
# Windowed Attention
# ============================================================================
