"""LiDAR projection helpers shared by metric-depth models."""
from __future__ import annotations

from typing import List, Tuple

import torch


def projected_lidar_sparse_depth(
    points_per_frame: List[List[torch.Tensor]],
    T_cam_from_velo: torch.Tensor,
    K_per_frame: torch.Tensor,
    image_hw: torch.Tensor,
    *,
    device: torch.device,
    allow_empty_single_view: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project LiDAR sweeps to one or more sparse camera-depth maps.

    A sample may provide one shared sweep (broadcast to every camera) or one
    sweep per view. Each cloud is projected with the corresponding camera
    transform/intrinsics, rounded to the nearest pixel, and z-buffered. Invalid
    pixels in the returned ``(B,V,H,W)`` float32 tensor are zero. A camera with
    no projected points is represented by an all-zero depth/all-false mask;
    malformed geometry still raises an error. Single-view callers retain the
    legacy non-empty requirement unless ``allow_empty_single_view=True``.
    """
    batch_size = len(points_per_frame)
    if batch_size == 0:
        raise RuntimeError("points_per_frame must contain at least one sample.")
    if K_per_frame.ndim != 4 or K_per_frame.shape[0] != batch_size:
        raise RuntimeError(
            "K_per_frame must be (B,V,3,3); "
            f"got {tuple(K_per_frame.shape)}."
        )
    if tuple(K_per_frame.shape[-2:]) != (3, 3):
        raise RuntimeError(f"K_per_frame must end in (3,3), got {tuple(K_per_frame.shape)}.")
    num_views = int(K_per_frame.shape[1])
    if num_views <= 0:
        raise RuntimeError("K_per_frame must contain at least one view.")

    if tuple(T_cam_from_velo.shape) == (batch_size, 4, 4):
        transforms = T_cam_from_velo[:, None].expand(-1, num_views, -1, -1)
    elif (
        T_cam_from_velo.ndim == 4
        and T_cam_from_velo.shape[0] == batch_size
        and tuple(T_cam_from_velo.shape[-2:]) == (4, 4)
        and T_cam_from_velo.shape[1] in (1, num_views)
    ):
        transforms = T_cam_from_velo
        if transforms.shape[1] == 1 and num_views > 1:
            transforms = transforms.expand(-1, num_views, -1, -1)
    else:
        raise RuntimeError(
            "T_cam_from_velo must be (B,4,4), (B,1,4,4), or (B,V,4,4); got "
            f"{tuple(T_cam_from_velo.shape)} for B={batch_size}, V={num_views}."
        )

    if tuple(image_hw.shape) == (batch_size, 2):
        image_hw_per_view = image_hw[:, None].expand(-1, num_views, -1)
    elif tuple(image_hw.shape) == (batch_size, num_views, 2):
        image_hw_per_view = image_hw
    else:
        raise RuntimeError(
            "image_hw must be (B,2) or (B,V,2); got "
            f"{tuple(image_hw.shape)} for B={batch_size}, V={num_views}."
        )

    transforms = transforms.to(device=device, dtype=torch.float32)
    intrinsics = K_per_frame.to(device=device, dtype=torch.float32)
    image_hw_per_view = image_hw_per_view.to(device=device)
    expected_height = int(image_hw_per_view[0, 0, 0].item())
    expected_width = int(image_hw_per_view[0, 0, 1].item())
    if expected_height <= 0 or expected_width <= 0:
        raise RuntimeError(
            f"image_hw must be positive, got {(expected_height, expected_width)}."
        )
    if not bool((image_hw_per_view[..., 0] == expected_height).all().item()) or not bool(
        (image_hw_per_view[..., 1] == expected_width).all().item()
    ):
        raise RuntimeError(
            "Projected LiDAR sparse-depth batches require identical image_hw; "
            f"got {image_hw_per_view.tolist()}."
        )

    sparse_depths: List[torch.Tensor] = []
    sparse_masks: List[torch.Tensor] = []
    for batch_index, sample_points in enumerate(points_per_frame):
        if len(sample_points) not in (1, num_views):
            raise RuntimeError(
                "Projected LiDAR depth requires one shared sweep or one sweep per view; "
                f"sample {batch_index} has {len(sample_points)} for V={num_views}."
            )
        view_depths: List[torch.Tensor] = []
        view_masks: List[torch.Tensor] = []
        for view_index in range(num_views):
            points = sample_points[0 if len(sample_points) == 1 else view_index].to(
                device=device, dtype=torch.float32
            )
            if points.ndim != 2 or points.shape[1] < 3:
                raise RuntimeError(
                    f"LiDAR points must be (P,>=3), got {tuple(points.shape)}."
                )

            height = expected_height
            width = expected_width
            transform = transforms[batch_index, view_index]
            xyz_cam = points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
            depth = xyz_cam[:, 2]
            valid_xyz = torch.isfinite(xyz_cam).all(dim=1) & (depth > 1e-6)
            xyz_cam = xyz_cam[valid_xyz]
            depth = depth[valid_xyz]
            if (
                depth.numel() == 0
                and num_views == 1
                and not allow_empty_single_view
            ):
                raise RuntimeError(
                    f"Sample {batch_index} has no positive finite LiDAR depth."
                )

            if depth.numel() > 0:
                K = intrinsics[batch_index, view_index]
                u = xyz_cam[:, 0] / depth * K[0, 0] + K[0, 2]
                v = xyz_cam[:, 1] / depth * K[1, 1] + K[1, 2]
                finite_uv = torch.isfinite(u) & torch.isfinite(v)
                u = u[finite_uv]
                v = v[finite_uv]
                depth = depth[finite_uv]
                x = torch.round(u).to(dtype=torch.long)
                y = torch.round(v).to(dtype=torch.long)
                in_image = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                x = x[in_image]
                y = y[in_image]
                depth = depth[in_image]
                if (
                    depth.numel() == 0
                    and num_views == 1
                    and not allow_empty_single_view
                ):
                    raise RuntimeError(
                        f"Sample {batch_index} has no LiDAR points in the image."
                    )

            z_buffer = torch.full(
                (height * width,),
                float("inf"),
                device=device,
                dtype=torch.float32,
            )
            if depth.numel() > 0:
                ranks = y * width + x
                z_buffer.scatter_reduce_(
                    0,
                    ranks,
                    depth,
                    reduce="amin",
                    include_self=True,
                )
            projected_mask = torch.isfinite(z_buffer)
            projected_depth = torch.where(
                projected_mask,
                z_buffer,
                torch.zeros_like(z_buffer),
            )
            view_depths.append(projected_depth.view(height, width))
            view_masks.append(projected_mask.view(height, width))

        sparse_depths.append(torch.stack(view_depths, dim=0))
        sparse_masks.append(torch.stack(view_masks, dim=0))

    return torch.stack(sparse_depths, dim=0), torch.stack(sparse_masks, dim=0)
