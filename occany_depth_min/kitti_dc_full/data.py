"""Full KITTI depth completion, left camera, with synchronized RAW LiDAR.

The official train/val directories define the split. Images retain their full
field of view; native GT is kept separately from the resized training target.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from ..data import _collate_views
from ..knn import knn_complete_sparse_depth


def normalize_rgb(rgb: np.ndarray) -> torch.Tensor:
    """Match the original ToTensor + Normalize(0.5, 0.5) arithmetic exactly."""
    image = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    return torch.from_numpy(image).float().div_(255.0).sub_(0.5).div_(0.5)


def project_velodyne_to_sparse_depth(
    points: np.ndarray,
    T_cam_from_velo: np.ndarray,
    intrinsics: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    """Project a Velodyne sweep and z-buffer the closest point per pixel."""
    height, width = (int(image_hw[0]), int(image_hw[1]))
    points = np.asarray(points, dtype=np.float32)
    transform = np.asarray(T_cam_from_velo, dtype=np.float32)
    K = np.asarray(intrinsics, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must be (P,>=3), got {points.shape}.")
    if transform.shape != (4, 4) or K.shape != (3, 3):
        raise ValueError(f"Expected T=(4,4), K=(3,3), got {transform.shape}, {K.shape}.")

    xyz = points[:, :3]
    xyz_cam = xyz @ transform[:3, :3].T + transform[:3, 3]
    depth = xyz_cam[:, 2]
    valid = np.isfinite(xyz_cam).all(axis=1) & (depth > 1e-6)
    xyz_cam = xyz_cam[valid]
    depth = depth[valid]
    if depth.size == 0:
        return np.zeros((height, width), dtype=np.float32)

    u = xyz_cam[:, 0] / depth * K[0, 0] + K[0, 2]
    v = xyz_cam[:, 1] / depth * K[1, 1] + K[1, 2]
    x = np.rint(u).astype(np.int64)
    y = np.rint(v).astype(np.int64)
    valid = (
        np.isfinite(u)
        & np.isfinite(v)
        & (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )
    x, y, depth = x[valid], y[valid], depth[valid]
    sparse = np.zeros((height, width), dtype=np.float32)
    if depth.size == 0:
        return sparse

    ranks = y * width + x
    order = np.lexsort((depth, ranks))
    ranks, x, y, depth = ranks[order], x[order], y[order], depth[order]
    keep = np.ones(ranks.shape, dtype=bool)
    keep[1:] = ranks[1:] != ranks[:-1]
    sparse[y[keep], x[keep]] = depth[keep].astype(np.float32)
    return sparse


PROTOCOL = "kitti_dc_full_left_raw_lidar_letterbox_halfpixel_v1"
EXPECTED_COUNTS = {"train": 42949, "val": 3426}


def read_calibration(path: Path) -> dict[str, np.ndarray]:
    result = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition(":")
        if separator and key not in ("calib_time", "corner_dist"):
            result[key] = np.array([float(x) for x in value.split()], dtype=np.float64)
    return result


def raw_left_geometry(date_root: Path) -> tuple[np.ndarray, np.ndarray]:
    camera = read_calibration(date_root / "calib_cam_to_cam.txt")
    velo = read_calibration(date_root / "calib_velo_to_cam.txt")
    projection = camera["P_rect_02"].reshape(3, 4)
    intrinsics = projection[:, :3].copy()
    raw = np.eye(4)
    raw[:3, :3] = velo["R"].reshape(3, 3)
    raw[:3, 3] = velo["T"]
    rectification = np.eye(4)
    rectification[:3, :3] = camera["R_rect_00"].reshape(3, 3)
    camera_shift = np.eye(4)
    camera_shift[:3, 3] = np.linalg.solve(intrinsics, projection[:, 3])
    return intrinsics, camera_shift @ rectification @ raw


def letterbox_full_image(rgb, depth, intrinsics, long_side=1232, patch_size=14):
    h, w = depth.shape
    if rgb.shape != (h, w, 3):
        raise ValueError(f"RGB/GT dimensions disagree: {rgb.shape}, {depth.shape}")
    if long_side <= 0 or patch_size <= 0:
        raise ValueError("long_side and patch_size must be positive")
    scale = min(1.0, float(long_side) / max(h, w))
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    ph, pw = math.ceil(nh / patch_size) * patch_size, math.ceil(nw / patch_size) * patch_size
    top, left = (ph - nh) // 2, (pw - nw) // 2
    resized_rgb = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.Resampling.LANCZOS))
    clean_depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
    resized_depth = cv2.resize(clean_depth, (nw, nh), interpolation=cv2.INTER_NEAREST_EXACT)
    image_out = np.zeros((ph, pw, 3), dtype=np.uint8)
    depth_out = np.zeros((ph, pw), dtype=np.float32)
    image_mask = np.zeros((ph, pw), dtype=bool)
    region = np.s_[top:top + nh, left:left + nw]
    image_out[region], depth_out[region], image_mask[region] = resized_rgb, resized_depth, True
    K = np.array(intrinsics, dtype=np.float32, copy=True)
    sx, sy = nw / w, nh / h
    K[0, 0] *= sx
    K[0, 1] *= sx
    K[1, 0] *= sy
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5 + left
    K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5 + top
    metadata = dict(native_hw=(h, w), resized_hw=(nh, nw), padded_hw=(ph, pw),
                    pad_tblr=(top, ph - nh - top, left, pw - nw - left), scale_xy=(sx, sy))
    return image_out, depth_out, image_mask, K, metadata


class KITTIDepthCompletionDataset(Dataset):
    def __init__(self, raw_root, split, *, input_long_side=1232, patch_size=14,
                 use_lidar=True, scaled=False, require_full=True, return_raw_points=False):
        if split not in EXPECTED_COUNTS:
            raise ValueError(f"Unsupported split: {split}")
        self.root = Path(raw_root).resolve()
        self.split, self.input_long_side, self.patch_size = split, input_long_side, patch_size
        self.use_lidar, self.scaled = bool(use_lidar), bool(scaled)
        self.return_raw_points = bool(return_raw_points)
        if self.return_raw_points and not self.use_lidar:
            raise ValueError("Returning RAW points requires use_lidar=True")
        if scaled and not use_lidar:
            raise ValueError("Scaled models require RAW LiDAR")
        self.samples = sorted((self.root / split).glob("*/proj_depth/groundtruth/image_02/*.png"))
        if not self.samples or (require_full and len(self.samples) != EXPECTED_COUNTS[split]):
            raise ValueError(f"{split}: found {len(self.samples)} left GT images, expected {EXPECTED_COUNTS[split]}")
        self.calibrations = {}
        for path in self.samples:
            drive = path.parts[-5]
            date = drive[:10]
            if date not in self.calibrations:
                self.calibrations[date] = raw_left_geometry(self.root / date)
            required = [self.root / date / drive / "image_02/data" / path.name]
            if self.use_lidar:
                required.append(self.root / date / drive / "velodyne_points/data" / f"{path.stem}.bin")
            for source in required:
                if not source.is_file():
                    raise FileNotFoundError(source)
        drives = {p.parts[-5] for p in self.samples}
        other_split = "val" if split == "train" else "train"
        other_root = self.root / other_split
        other_drives = {p.name for p in other_root.iterdir() if p.is_dir()} if other_root.is_dir() else set()
        if drives & other_drives:
            raise ValueError(f"Train/val drives overlap: {sorted(drives & other_drives)}")
        digest = hashlib.sha256()
        for path in self.samples:
            digest.update((str(path.relative_to(self.root)) + "\n").encode())
        for date in sorted(self.calibrations):
            for name in ("calib_cam_to_cam.txt", "calib_velo_to_cam.txt"):
                digest.update((self.root / date / name).read_bytes())
        self.manifest_sha256 = digest.hexdigest()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        loss_scale, padding = 1.0, False
        if isinstance(index, tuple):
            index, loss_scale, padding = index
        path = self.samples[index]
        drive, date = path.parts[-5], path.parts[-5][:10]
        rgb_path = self.root / date / drive / "image_02/data" / path.name
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        with Image.open(path) as image:
            native_depth = np.asarray(image, dtype=np.float32) / 256.0
        native_K, transform = self.calibrations[date]
        rgb, depth, image_mask, K, metadata = letterbox_full_image(
            rgb, native_depth, native_K, self.input_long_side, self.patch_size)
        h, w = depth.shape
        sparse = np.zeros((h, w), dtype=np.float32)
        sample_id = f"{drive}/{path.stem}/image_02"
        if self.use_lidar:
            point_path = self.root / date / drive / "velodyne_points/data" / f"{path.stem}.bin"
            points = np.fromfile(point_path, dtype=np.float32)
            if not points.size or points.size % 4:
                raise ValueError(f"Malformed RAW LiDAR: {point_path}")
            points = points.reshape(-1, 4)
            sparse = project_velodyne_to_sparse_depth(points, transform, K, (h, w))
            sparse[~image_mask] = 0
        sample = dict(
            views=[dict(img=normalize_rgb(rgb), true_shape=np.array((h, w), dtype=np.int32),
                        camera_intrinsics=K, camera_pose=np.eye(4, dtype=np.float32),
                        label=sample_id)],
            K_per_frame=torch.from_numpy(K)[None],
            T_cam_from_velo=torch.from_numpy(transform.astype(np.float32)),
            image_hw=torch.tensor((h, w)),
            dense_depth=torch.from_numpy(depth)[None],
            dense_depth_pixel_mask=torch.from_numpy(image_mask & (depth > 0))[None],
            dense_depth_frame_mask=torch.tensor([bool((depth > 0).any())]),
            sparse_depth=torch.from_numpy(sparse)[None],
            sparse_depth_mask=torch.from_numpy(sparse > 0)[None],
            image_valid_mask=torch.from_numpy(image_mask)[None],
            native_dense_depth=torch.from_numpy(native_depth),
            native_valid_mask=torch.from_numpy(np.isfinite(native_depth) & (native_depth > 0)),
            resize_metadata=metadata, sample_id=sample_id, dataset_name="kitti",
            ddp_loss_scale=torch.tensor(loss_scale, dtype=torch.float32),
            sampler_padding=torch.tensor(padding),
        )
        if self.return_raw_points:
            sample["points_per_frame"] = [torch.from_numpy(points)]
        if self.scaled:
            top, _, left, _ = metadata["pad_tblr"]
            nh, nw = metadata["resized_hw"]
            try:
                prompt = knn_complete_sparse_depth(sparse[top:top + nh, left:left + nw])
            except (ValueError, RuntimeError) as exc:
                raise RuntimeError(f"{sample_id}: {exc}") from exc
            sample["knn_depth"] = torch.from_numpy(np.pad(prompt, (
                (top, h - top - nh), (left, w - left - nw)), mode="edge"))[None]
        return sample


def collate_kitti_dc_full(batch):
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    result = {"views": _collate_views(batch)}
    for key in batch[0]:
        if key == "views":
            continue
        values = [sample[key] for sample in batch]
        if key in ("native_dense_depth", "native_valid_mask", "resize_metadata", "sample_id", "dataset_name", "points_per_frame"):
            result[key] = values
        elif isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            raise TypeError(f"Unexpected batch field: {key}")
    return result


class FullCoverageDistributedSampler(Sampler):
    """Visit each real sample once; zero-weight slots complete the last DDP step."""
    def __init__(self, dataset, *, num_replicas=1, rank=0, seed=0, max_samples=None):
        if not len(dataset) or num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("Invalid dataset or distributed rank")
        self.length = len(dataset) if max_samples is None else int(max_samples)
        if not 0 < self.length <= len(dataset):
            raise ValueError("max_samples must be positive and no larger than the dataset")
        self.num_replicas, self.rank, self.seed, self.epoch = num_replicas, rank, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return math.ceil(self.length / self.num_replicas)

    def __iter__(self) -> Iterator:
        order = torch.randperm(self.length, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        for start in range(0, self.length, self.num_replicas):
            real = min(self.num_replicas, self.length - start)
            if self.rank < real:
                yield order[start + self.rank], self.num_replicas / real, False
            else:
                yield order[start], 0.0, True
