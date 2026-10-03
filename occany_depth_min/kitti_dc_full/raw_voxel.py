"""Camera-frame RAW xyz/intensity VFE and visible image-grid token projection.

Retained from the full KITTI experiment; the sparse-depth z-buffer is a
separate input and never changes RAW VFE's intensity or voxel visibility.
"""
from __future__ import annotations
import math
from contextlib import nullcontext
from typing import List, Optional, Tuple
import torch
from torch import nn

class VoxelFeatureEncoder(nn.Module):
    """PointPillars-style PointNet on cam-frame voxels.

    Inputs are raw points in the velodyne frame of the same timestep;
    a ``T_cam_from_velo`` (4x4) transforms them into cam coords prior to
    voxelization. We voxelize at the cam-frame grid defined by
    ``(vox_origin, vox_size, vox_grid)`` (all in cam coords:
    x→right, y→down, z→forward).

    Output:
      - ``voxel_feat``: (V, d_out) features for the non-empty voxels.
      - ``voxel_center_cam``: (V, 3) cam-frame center coordinate per voxel.
    """

    def __init__(
        self,
        vox_origin: Tuple[float, float, float] = (-25.6, -2.0, 0.0),
        vox_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
        vox_grid: Tuple[int, int, int] = (128, 16, 128),
        d_voxel: int = 128,
        d_out: int = 768,
        hidden: int = 64,
        pe_num_freqs: int = 8,
        point_mlp: Optional[nn.Module] = None,
        force_fp32_geometry: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "vox_origin", torch.tensor(vox_origin, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "vox_size", torch.tensor(vox_size, dtype=torch.float32), persistent=False
        )
        self.vox_grid: Tuple[int, int, int] = tuple(int(v) for v in vox_grid)
        self.force_fp32_geometry = bool(force_fp32_geometry)

        # Per-point input features: (x, y, z, intensity, dx_c, dy_c, dz_c) = 7.
        # If a shared module is provided it must output d_voxel-dim features.
        if point_mlp is not None:
            self.point_mlp = point_mlp
        else:
            self.point_mlp = nn.Sequential(
                nn.Linear(7, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, d_voxel),
            )
        self.voxel_norm = nn.LayerNorm(d_voxel)
        self.voxel_proj = nn.Linear(d_voxel, d_out)

        # Sinusoidal 3D PE on voxel center (cam coords), projected to d_out and added.
        self.pe_num_freqs = int(pe_num_freqs)
        pe_dim = 3 * 2 * self.pe_num_freqs
        self.pe_proj = nn.Linear(pe_dim, d_out)

    @staticmethod
    def _sinusoidal_pe_3d(coords: torch.Tensor, num_freqs: int) -> torch.Tensor:
        """coords: (..., 3) → (..., 3*2*num_freqs)."""
        device = coords.device
        dtype = coords.dtype
        # Use geometric base-2 freqs, scaled by pi.
        freqs = (2.0 ** torch.arange(num_freqs, device=device, dtype=dtype)) * math.pi
        # Normalize coordinate magnitudes loosely so high freqs don't alias too
        # fast — divide by an "effective range" of 50 m.
        scaled = coords / 50.0
        x = scaled.unsqueeze(-1) * freqs  # (..., 3, F)
        sin_part = x.sin()
        cos_part = x.cos()
        return torch.cat([sin_part, cos_part], dim=-1).flatten(-2)  # (..., 6F)

    def forward(
        self,
        points_velo: torch.Tensor,       # (P, 4) (x, y, z, intensity), float32
        T_cam_from_velo: torch.Tensor,   # (4, 4), float32
        vox_origin: Optional[torch.Tensor] = None,
        vox_size: Optional[torch.Tensor] = None,
        vox_grid: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns (voxel_feat, voxel_center_cam) or (None, None) if no voxel survives.

        Computed in input dtype (float32 by default); caller can cast as needed.
        """
        if points_velo.shape[0] == 0:
            return None, None

        # Transform velo→cam: (P, 3) using the rigid transform.
        # Up-cast points to float for the geometry to avoid precision loss under bf16.
        T = T_cam_from_velo.to(dtype=torch.float32)
        p_velo = points_velo[:, :3].to(dtype=torch.float32)
        intensity = points_velo[:, 3:4].to(dtype=torch.float32)
        R = T[:3, :3]
        t = T[:3, 3]
        with (torch.autocast(device_type=p_velo.device.type, enabled=False)
              if self.force_fp32_geometry else nullcontext()):
            p_cam = p_velo @ R.T + t  # (P, 3)

        vox_origin = (
            self.vox_origin
            if vox_origin is None
            else vox_origin.to(device=p_cam.device, dtype=torch.float32)
        )
        vox_size = (
            self.vox_size
            if vox_size is None
            else vox_size.to(device=p_cam.device, dtype=torch.float32)
        )
        Gx, Gy, Gz = self.vox_grid if vox_grid is None else tuple(int(v) for v in vox_grid)

        idx_f = (p_cam - vox_origin) / vox_size
        idx = idx_f.floor().long()  # (P, 3)

        valid = (
            (idx[:, 0] >= 0) & (idx[:, 0] < Gx)
            & (idx[:, 1] >= 0) & (idx[:, 1] < Gy)
            & (idx[:, 2] >= 0) & (idx[:, 2] < Gz)
        )
        if int(valid.sum().item()) == 0:
            return None, None

        idx = idx[valid]
        p_cam = p_cam[valid]
        intensity = intensity[valid]

        voxel_center = (idx.to(p_cam.dtype) + 0.5) * vox_size + vox_origin  # (P_valid, 3)
        rel = p_cam - voxel_center

        point_feat = torch.cat([p_cam, intensity, rel], dim=-1)  # (P_valid, 7)

        # Linear voxel index for grouping.
        lin = (idx[:, 0] * Gy + idx[:, 1]) * Gz + idx[:, 2]
        # Compact to dense [0..V) ids via unique.
        uniq_lin, inverse = torch.unique(lin, return_inverse=True)
        V = int(uniq_lin.shape[0])

        # Per-point MLP (cast to model dtype via autocast if active).
        h = self.point_mlp(point_feat)  # (P_valid, d_voxel)
        d_voxel = h.shape[-1]

        # Per-voxel max pool via scatter_reduce.
        neg_inf = torch.finfo(h.dtype).min
        voxel_feat = torch.full(
            (V, d_voxel), neg_inf, dtype=h.dtype, device=h.device
        )
        voxel_feat.scatter_reduce_(
            0,
            inverse.unsqueeze(-1).expand(-1, d_voxel),
            h,
            reduce="amax",
            include_self=False,
        )
        # Any voxel with no contributing point (shouldn't happen due to inverse) → zero.
        voxel_feat = voxel_feat.masked_fill(voxel_feat == neg_inf, 0.0)

        # Voxel centers (decompose uniq_lin).
        u_x = uniq_lin // (Gy * Gz)
        u_y = (uniq_lin // Gz) % Gy
        u_z = uniq_lin % Gz
        uniq_idx = torch.stack([u_x, u_y, u_z], dim=-1).to(p_cam.dtype)
        voxel_center_cam = (uniq_idx + 0.5) * vox_size + vox_origin  # (V, 3)

        # Project to d_out + 3D positional encoding.
        voxel_feat = self.voxel_norm(voxel_feat)
        voxel_proj = self.voxel_proj(voxel_feat)
        pe = self._sinusoidal_pe_3d(voxel_center_cam, self.pe_num_freqs).to(voxel_proj.dtype)
        voxel_proj = voxel_proj + self.pe_proj(pe)

        return voxel_proj, voxel_center_cam

    def forward_frames(
        self,
        points_per_frame: List[List[torch.Tensor]],
        T_cam_from_velo: torch.Tensor,
        vox_origin: Optional[torch.Tensor] = None,
        vox_size: Optional[torch.Tensor] = None,
        vox_grid: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Voxelize every sample/frame with one grouped reduction.

        ``forward`` handles one frame at a time. Calling it in a Python loop
        repeats ``unique`` and ``scatter_reduce`` for every frame. Here a frame
        id is folded into the linear voxel key, so all frames share one point
        MLP, one grouping pass, and one max-pool without mixing their voxels.

        Returns ``(voxel_feat, voxel_center_cam, voxel_frame_idx)`` where the
        frame index is flattened as ``sample_idx * num_frames + frame_idx``.
        """
        batch_size = len(points_per_frame)
        if batch_size == 0:
            return None, None, None
        num_frames = len(points_per_frame[0])
        if num_frames == 0:
            return None, None, None
        if any(len(per_sample) != num_frames for per_sample in points_per_frame):
            raise RuntimeError("points_per_frame must have the same frame count per sample.")

        if T_cam_from_velo.ndim == 3:
            if T_cam_from_velo.shape[0] != batch_size:
                raise RuntimeError(
                    "T_cam_from_velo batch size does not match points_per_frame: "
                    f"{T_cam_from_velo.shape[0]} vs {batch_size}."
                )
            T_cam_from_velo = T_cam_from_velo[:, None].expand(
                batch_size, num_frames, 4, 4
            )
        elif T_cam_from_velo.ndim != 4 or T_cam_from_velo.shape[:2] != (
            batch_size,
            num_frames,
        ):
            raise RuntimeError(
                "T_cam_from_velo must be (B,4,4) or (B,N,4,4); got "
                f"{tuple(T_cam_from_velo.shape)} for B={batch_size}, N={num_frames}."
            )

        device = T_cam_from_velo.device
        non_empty_cam_points: List[torch.Tensor] = []
        non_empty_intensity: List[torch.Tensor] = []
        non_empty_frame_idx: List[torch.Tensor] = []
        for b, per_sample in enumerate(points_per_frame):
            for f, points in enumerate(per_sample):
                if points.shape[0] == 0:
                    continue
                points = points.to(device=device, non_blocking=True)
                transform = T_cam_from_velo[b, f].to(dtype=torch.float32)
                p_velo = points[:, :3].to(dtype=torch.float32)
                with (torch.autocast(device_type=device.type, enabled=False)
                      if self.force_fp32_geometry else nullcontext()):
                    non_empty_cam_points.append(
                        p_velo @ transform[:3, :3].T + transform[:3, 3]
                    )
                non_empty_intensity.append(points[:, 3:4].to(dtype=torch.float32))
                non_empty_frame_idx.append(
                    torch.full(
                        (points.shape[0],),
                        b * num_frames + f,
                        dtype=torch.long,
                        device=device,
                    )
                )
        if not non_empty_cam_points:
            return None, None, None

        p_cam = torch.cat(non_empty_cam_points, dim=0)
        intensity = torch.cat(non_empty_intensity, dim=0)
        point_frame_idx = torch.cat(non_empty_frame_idx, dim=0)

        def _per_sample_grid_value(
            value: Optional[torch.Tensor], default: torch.Tensor, name: str
        ) -> torch.Tensor:
            if value is None:
                return default.to(device=device, dtype=torch.float32).view(1, 3).expand(
                    batch_size, 3
                )
            value = value.to(device=device, dtype=torch.float32)
            if value.ndim == 1:
                value = value.view(1, 3).expand(batch_size, 3)
            if value.shape != (batch_size, 3):
                raise RuntimeError(
                    f"{name} must be (3,) or (B,3); got {tuple(value.shape)}."
                )
            return value

        origins = _per_sample_grid_value(vox_origin, self.vox_origin, "vox_origin")
        sizes = _per_sample_grid_value(vox_size, self.vox_size, "vox_size")
        point_sample_idx = point_frame_idx // num_frames
        point_origins = origins[point_sample_idx]
        point_sizes = sizes[point_sample_idx]
        Gx, Gy, Gz = self.vox_grid if vox_grid is None else tuple(int(v) for v in vox_grid)
        voxels_per_frame = Gx * Gy * Gz

        idx = ((p_cam - point_origins) / point_sizes).floor().long()
        valid = (
            (idx[:, 0] >= 0)
            & (idx[:, 0] < Gx)
            & (idx[:, 1] >= 0)
            & (idx[:, 1] < Gy)
            & (idx[:, 2] >= 0)
            & (idx[:, 2] < Gz)
        )
        if int(valid.sum().item()) == 0:
            return None, None, None

        idx = idx[valid]
        p_cam = p_cam[valid]
        intensity = intensity[valid]
        point_frame_idx = point_frame_idx[valid]
        point_sample_idx = point_sample_idx[valid]
        point_origins = origins[point_sample_idx]
        point_sizes = sizes[point_sample_idx]

        point_voxel_center = (idx.to(p_cam.dtype) + 0.5) * point_sizes + point_origins
        point_feat = torch.cat(
            [p_cam, intensity, p_cam - point_voxel_center], dim=-1
        )
        local_lin = (idx[:, 0] * Gy + idx[:, 1]) * Gz + idx[:, 2]
        global_lin = point_frame_idx * voxels_per_frame + local_lin
        uniq_global, inverse = torch.unique(global_lin, return_inverse=True)

        h = self.point_mlp(point_feat)
        d_voxel = h.shape[-1]
        neg_inf = torch.finfo(h.dtype).min
        voxel_feat = torch.full(
            (uniq_global.shape[0], d_voxel),
            neg_inf,
            dtype=h.dtype,
            device=device,
        )
        voxel_feat.scatter_reduce_(
            0,
            inverse.unsqueeze(-1).expand(-1, d_voxel),
            h,
            reduce="amax",
            include_self=False,
        )
        voxel_feat = voxel_feat.masked_fill(voxel_feat == neg_inf, 0.0)

        voxel_frame_idx = uniq_global // voxels_per_frame
        uniq_lin = uniq_global % voxels_per_frame
        u_x = uniq_lin // (Gy * Gz)
        u_y = (uniq_lin // Gz) % Gy
        u_z = uniq_lin % Gz
        uniq_idx = torch.stack([u_x, u_y, u_z], dim=-1).to(p_cam.dtype)
        voxel_sample_idx = voxel_frame_idx // num_frames
        voxel_center_cam = (
            (uniq_idx + 0.5) * sizes[voxel_sample_idx] + origins[voxel_sample_idx]
        )

        voxel_feat = self.voxel_norm(voxel_feat)
        voxel_proj = self.voxel_proj(voxel_feat)
        pe = self._sinusoidal_pe_3d(voxel_center_cam, self.pe_num_freqs).to(
            voxel_proj.dtype
        )
        voxel_proj = voxel_proj + self.pe_proj(pe)
        return voxel_proj, voxel_center_cam, voxel_frame_idx
def encode_projected_raw_voxel_tokens(
    encoder: VoxelFeatureEncoder, images, points_per_frame, T_cam_from_velo,
    K_per_frame, image_valid_mask, *, patch_size, vox_origin=None,
    vox_size=None, vox_grid=None,
):
    """Return features, centers, frame ids, patch rows/cols without a z-buffer.

    All occupied voxels in the configured camera-frame grid are encoded.
    Only tokens whose centers project into the unpadded image are fused.
    """
    batch_size, num_views, _, height, width = images.shape
    if points_per_frame is None or T_cam_from_velo is None or image_valid_mask is None:
        raise ValueError("RAW VFE requires points_per_frame, T_cam_from_velo and image_valid_mask")
    if len(points_per_frame) != batch_size or any(len(frames) != num_views for frames in points_per_frame):
        raise ValueError("RAW VFE points must have shape [batch][view]")
    if tuple(image_valid_mask.shape) != (batch_size, num_views, height, width):
        raise ValueError("RAW VFE image_valid_mask must match (B,V,H,W)")
    if height % patch_size or width % patch_size:
        raise ValueError("RAW VFE image dimensions must be divisible by patch_size")
    intrinsics = K_per_frame.to(device=images.device, dtype=torch.float32)
    if intrinsics.shape == (batch_size, 3, 3):
        intrinsics = intrinsics[:, None].expand(batch_size, num_views, 3, 3)
    if tuple(intrinsics.shape) != (batch_size, num_views, 3, 3):
        raise ValueError("RAW VFE K_per_frame must be (B,3,3) or (B,V,3,3)")
    features, centers, frame_ids = encoder.forward_frames(
        points_per_frame, T_cam_from_velo.to(device=images.device, dtype=torch.float32),
        vox_origin=vox_origin, vox_size=vox_size, vox_grid=vox_grid,
    )
    if features is None:
        return None, None, None, None, None
    with torch.autocast(device_type=images.device.type, enabled=False):
        centers = centers.float()
        projected = torch.bmm(intrinsics.reshape(-1, 3, 3)[frame_ids], centers.unsqueeze(-1)).squeeze(-1)
        positive_z = centers[:, 2] > 0.1
        uv = projected[:, :2] / torch.where(positive_z, centers[:, 2], 1.).unsqueeze(-1)
        valid = positive_z & torch.isfinite(uv).all(-1)
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        pixels = uv.nan_to_num().floor().long()
        pixel_x = pixels[:, 0].clamp(0, width - 1)
        pixel_y = pixels[:, 1].clamp(0, height - 1)
        masks = image_valid_mask.to(device=images.device, dtype=torch.bool).reshape(-1, height, width)
        valid &= masks[frame_ids, pixel_y, pixel_x]
    return (features[valid], centers[valid], frame_ids[valid],
            pixel_y[valid] // patch_size, pixel_x[valid] // patch_size)
