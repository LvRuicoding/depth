"""Unified single-frame sparse-depth datasets used by the six-domain run.

The module deliberately performs all depth decoding and geometric transforms
online.  Split manifests contain paths/indices only; no dense or sparse depth
map is materialised as a cache.

All adapters return the same model-facing contract::

    views                    list with one normalized RGB view
    points_per_frame         list with one ``(P, 4)`` camera/LiDAR cloud
    T_cam_from_velo          ``(1, 4, 4)``
    K_per_frame              ``(1, 3, 3)``
    image_hw                 ``(2,)`` (padded H, W)
    dense_depth              ``(1, H, W)`` metres
    dense_depth_pixel_mask   ``(1, H, W)`` bool
    sparse_depth/_mask       ``(1, H, W)`` (diagnostic; model reprojects points)

Images are aspect-preservingly resized so the long side is ``input_long_side``
and then symmetrically padded to a multiple of ``patch_size``.  Padding is
never included in supervision or metrics.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .knn import online_knn4_depth_from_sparse


DOMAIN_NAMES: Tuple[str, ...] = (
    "kitti",
    "ddad",
    "7scenes",
    "nyuv2",
    "sunrgbd",
    "void",
)

EXPECTED_SPLIT_COUNTS: Mapping[str, Mapping[str, int]] = {
    # Every available official training sample is used for training. The
    # released evaluation split is validation; no local test split is invented.
    # KITTI records without ``dense_depthmap`` are not eligible.
    "kitti": {"train": 3659, "val": 815},
    "ddad": {"train": 12650, "val": 3950},
    "7scenes": {"train": 26000, "val": 17000},
    "nyuv2": {"train": 795, "val": 654},
    "sunrgbd": {"train": 5285, "val": 5050},
    "void": {"train": 48259, "val": 800},
}

DEPTH_RANGES: Mapping[str, Tuple[float, float]] = {
    "kitti": (1.0e-3, 80.0),
    "ddad": (1.0e-3, 120.0),
    "7scenes": (0.1, 4.0),
    "nyuv2": (1.0e-3, 10.0),
    "sunrgbd": (0.0, 10.0),
    "void": (0.2, 5.0),
    # Evaluation-only domain.  VKITTI2 is deliberately not added to
    # DOMAIN_NAMES, so the historical six-domain training/checkpoint contract
    # remains unchanged.
    "vkitti2": (1.0e-3, 120.0),
}

DEPTH_RANGE_INCLUSIVITY: Mapping[str, Tuple[bool, bool]] = {
    "kitti": (True, True),
    "ddad": (True, True),
    "7scenes": (True, True),
    "nyuv2": (True, True),
    "sunrgbd": (False, False),
    "void": (False, False),
    "vkitti2": (True, True),
}

NYUV2_RGB_INTRINSICS = np.array(
    [
        [518.85790117450188, 0.0, 325.58244941119034],
        [0.0, 519.46961112127485, 253.73616633400465],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

SEVEN_SCENES_INTRINSICS = np.array(
    [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)

MANIFEST_SCHEMA = "occany_unified_depth_split_v3"


def _canonical_domain_name(name: str) -> str:
    value = str(name).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "kitti": "kitti",
        "ddad": "ddad",
        "7scenes": "7scenes",
        "sevenscenes": "7scenes",
        "nyu": "nyuv2",
        "nyuv2": "nyuv2",
        "sun": "sunrgbd",
        "sunrgbd": "sunrgbd",
        "void": "void",
        "void500": "void",
    }
    if value not in aliases:
        raise ValueError(f"Unknown unified-depth dataset {name!r}; expected one of {DOMAIN_NAMES}.")
    return aliases[value]


def _stable_uint64(*parts: object) -> int:
    digest = hashlib.sha256("\0".join(str(v) for v in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_rgb(image: np.ndarray) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise RuntimeError(f"RGB image must have shape (H,W,3), got {array.shape}.")
    contiguous = np.ascontiguousarray(array.transpose(2, 0, 1))
    return torch.from_numpy(contiguous).float().div_(127.5).sub_(1.0)


@dataclass(frozen=True)
class ResizeMetadata:
    native_hw: Tuple[int, int]
    resized_hw: Tuple[int, int]
    padded_hw: Tuple[int, int]
    pad_tblr: Tuple[int, int, int, int]
    scale_xy: Tuple[float, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "native_hw": self.native_hw,
            "resized_hw": self.resized_hw,
            "padded_hw": self.padded_hw,
            "pad_tblr": self.pad_tblr,
            "scale_xy": self.scale_xy,
        }


def compute_letterbox_metadata(
    native_hw: Sequence[int],
    *,
    input_long_side: int = 518,
    patch_size: int = 14,
) -> ResizeMetadata:
    """Return aspect-preserving resize + symmetric patch padding geometry."""
    native_h, native_w = (int(native_hw[0]), int(native_hw[1]))
    if native_h <= 0 or native_w <= 0:
        raise ValueError(f"native_hw must be positive, got {(native_h, native_w)}.")
    if input_long_side <= 0 or patch_size <= 0:
        raise ValueError("input_long_side and patch_size must be positive.")

    scale = float(input_long_side) / float(max(native_h, native_w))
    resized_h = max(1, int(round(native_h * scale)))
    resized_w = max(1, int(round(native_w * scale)))
    if native_h >= native_w:
        resized_h = int(input_long_side)
    else:
        resized_w = int(input_long_side)
    padded_h = int(math.ceil(resized_h / patch_size) * patch_size)
    padded_w = int(math.ceil(resized_w / patch_size) * patch_size)
    pad_top = (padded_h - resized_h) // 2
    pad_bottom = padded_h - resized_h - pad_top
    pad_left = (padded_w - resized_w) // 2
    pad_right = padded_w - resized_w - pad_left
    return ResizeMetadata(
        native_hw=(native_h, native_w),
        resized_hw=(resized_h, resized_w),
        padded_hw=(padded_h, padded_w),
        pad_tblr=(pad_top, pad_bottom, pad_left, pad_right),
        scale_xy=(resized_w / native_w, resized_h / native_h),
    )


def _resize_array_nearest(array: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")
    return np.asarray(pil.resize(size_wh, resample=Image.Resampling.NEAREST), dtype=np.float32)


def letterbox_rgb_depth(
    image: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    *,
    input_long_side: int = 518,
    patch_size: int = 14,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ResizeMetadata]:
    """Resize RGB/depth without cropping and update camera intrinsics."""
    rgb = np.asarray(image, dtype=np.uint8)
    depth_f32 = np.asarray(depth, dtype=np.float32)
    if rgb.shape[:2] != depth_f32.shape:
        raise RuntimeError(f"RGB/depth shape mismatch: {rgb.shape[:2]} vs {depth_f32.shape}.")
    metadata = compute_letterbox_metadata(
        depth_f32.shape, input_long_side=input_long_side, patch_size=patch_size
    )
    resized_h, resized_w = metadata.resized_hw
    padded_h, padded_w = metadata.padded_hw
    pad_top, _pad_bottom, pad_left, _pad_right = metadata.pad_tblr

    rgb_resized = np.asarray(
        Image.fromarray(rgb).resize((resized_w, resized_h), resample=Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    finite_depth = np.isfinite(depth_f32) & (depth_f32 > 0.0)
    clean_depth = np.where(finite_depth, depth_f32, 0.0).astype(np.float32)
    depth_resized = _resize_array_nearest(clean_depth, (resized_w, resized_h))
    valid_resized = _resize_array_nearest(finite_depth.astype(np.float32), (resized_w, resized_h)) > 0.5
    depth_resized = np.where(valid_resized, depth_resized, 0.0).astype(np.float32)

    rgb_out = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
    depth_out = np.zeros((padded_h, padded_w), dtype=np.float32)
    image_mask = np.zeros((padded_h, padded_w), dtype=bool)
    ys = slice(pad_top, pad_top + resized_h)
    xs = slice(pad_left, pad_left + resized_w)
    rgb_out[ys, xs] = rgb_resized
    depth_out[ys, xs] = depth_resized
    image_mask[ys, xs] = True

    K = np.asarray(intrinsics, dtype=np.float32).copy()
    if K.shape != (3, 3):
        raise RuntimeError(f"Camera intrinsics must be (3,3), got {K.shape}.")
    scale_x, scale_y = metadata.scale_xy
    K[0, :] *= scale_x
    K[1, :] *= scale_y
    K[0, 2] += float(pad_left)
    K[1, 2] += float(pad_top)
    K[2, :] = np.asarray(intrinsics, dtype=np.float32)[2, :]
    return rgb_out, depth_out, image_mask, K, metadata


def _map_native_pixels_to_letterbox(
    y: np.ndarray, x: np.ndarray, metadata: ResizeMetadata
) -> Tuple[np.ndarray, np.ndarray]:
    scale_x, scale_y = metadata.scale_xy
    pad_top, _pad_bottom, pad_left, _pad_right = metadata.pad_tblr
    # Pixel-centre mapping followed by the same nearest-integer convention as
    # the model's projection helper (torch.round / numpy.rint).
    out_x = np.rint((x.astype(np.float64) + 0.5) * scale_x - 0.5).astype(np.int64)
    out_y = np.rint((y.astype(np.float64) + 0.5) * scale_y - 0.5).astype(np.int64)
    out_x += int(pad_left)
    out_y += int(pad_top)
    out_y = np.clip(out_y, 0, metadata.padded_hw[0] - 1)
    out_x = np.clip(out_x, 0, metadata.padded_hw[1] - 1)
    return out_y, out_x


def sample_unique_sparse_depth(
    native_depth: np.ndarray,
    native_valid: np.ndarray,
    metadata: ResizeMetadata,
    *,
    sample_id: str,
    point_count: int = 500,
    sampling_seed: int = 0,
    return_native_mask: bool = False,
) -> Any:
    """Sample deterministic native GT pixels, refilling resize collisions."""
    depth = np.asarray(native_depth, dtype=np.float32)
    valid = np.asarray(native_valid, dtype=bool) & np.isfinite(depth) & (depth > 0.0)
    y, x = np.nonzero(valid)
    if len(y) < int(point_count):
        raise RuntimeError(
            f"{sample_id}: requested {point_count} sparse points but only {len(y)} valid GT pixels exist."
        )
    rng = np.random.default_rng(_stable_uint64("sparse", sampling_seed, sample_id))
    order = rng.permutation(len(y))
    out_h, out_w = metadata.padded_hw
    sparse = np.zeros((out_h, out_w), dtype=np.float32)
    mask = np.zeros((out_h, out_w), dtype=bool)
    native_selected = np.zeros(depth.shape, dtype=bool)
    mapped_y, mapped_x = _map_native_pixels_to_letterbox(y, x, metadata)
    selected = 0
    for index in order:
        yy, xx = int(mapped_y[index]), int(mapped_x[index])
        if mask[yy, xx]:
            continue
        sparse[yy, xx] = float(depth[y[index], x[index]])
        mask[yy, xx] = True
        native_selected[y[index], x[index]] = True
        selected += 1
        if selected == int(point_count):
            break
    if selected != int(point_count):
        raise RuntimeError(
            f"{sample_id}: resize collisions leave only {selected} unique pixels; "
            f"cannot provide the required {point_count}."
        )
    if return_native_mask:
        return sparse, mask, native_selected
    return sparse, mask


def resize_native_sparse_depth(
    native_sparse: np.ndarray,
    native_valid: np.ndarray,
    metadata: ResizeMetadata,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project a native sparse map through letterbox geometry with z-buffering."""
    depth = np.asarray(native_sparse, dtype=np.float32)
    valid = np.asarray(native_valid, dtype=bool) & np.isfinite(depth) & (depth > 0.0)
    y, x = np.nonzero(valid)
    out_h, out_w = metadata.padded_hw
    sparse = np.zeros((out_h, out_w), dtype=np.float32)
    mask = np.zeros((out_h, out_w), dtype=bool)
    out_y, out_x = _map_native_pixels_to_letterbox(y, x, metadata)
    for source_index in range(len(y)):
        yy, xx = int(out_y[source_index]), int(out_x[source_index])
        value = float(depth[y[source_index], x[source_index]])
        if not mask[yy, xx] or value < sparse[yy, xx]:
            sparse[yy, xx] = value
            mask[yy, xx] = True
    return sparse, mask


def sparse_depth_to_camera_points(sparse_depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Unproject sparse depth into ``(x,y,z,intensity)`` camera points."""
    depth = np.asarray(sparse_depth, dtype=np.float32)
    K = np.asarray(intrinsics, dtype=np.float32)
    mask = np.isfinite(depth) & (depth > 0.0)
    y, x = np.nonzero(mask)
    z = depth[y, x]
    px = (x.astype(np.float32) - K[0, 2]) / K[0, 0] * z
    py = (y.astype(np.float32) - K[1, 2]) / K[1, 1] * z
    intensity = np.zeros_like(z, dtype=np.float32)
    return np.ascontiguousarray(np.stack((px, py, z, intensity), axis=1), dtype=np.float32)


def project_points_to_sparse_depth(
    points: np.ndarray,
    T_cam_from_points: np.ndarray,
    intrinsics: np.ndarray,
    image_hw: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Numpy mirror of the model's rounded-pixel, nearest-depth projection."""
    pts = np.asarray(points, dtype=np.float32)
    transform = np.asarray(T_cam_from_points, dtype=np.float32)
    K = np.asarray(intrinsics, dtype=np.float32)
    height, width = int(image_hw[0]), int(image_hw[1])
    xyz = pts[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    z = xyz[:, 2]
    valid = np.isfinite(xyz).all(axis=1) & (z > 1.0e-6)
    xyz, z = xyz[valid], z[valid]
    u = xyz[:, 0] / z * K[0, 0] + K[0, 2]
    v = xyz[:, 1] / z * K[1, 1] + K[1, 2]
    x = np.rint(u).astype(np.int64)
    y = np.rint(v).astype(np.int64)
    inside = np.isfinite(u) & np.isfinite(v) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, z = x[inside], y[inside], z[inside]
    sparse = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    for xx, yy, value in zip(x.tolist(), y.tolist(), z.tolist()):
        if not mask[yy, xx] or value < sparse[yy, xx]:
            sparse[yy, xx] = float(value)
            mask[yy, xx] = True
    return sparse, mask


def decode_sunrgbd_depth(raw_depth: np.ndarray) -> np.ndarray:
    """Decode SUN RGB-D's 16-bit rotated depth representation to metres."""
    raw = np.asarray(raw_depth, dtype=np.uint16)
    decoded = np.bitwise_or(np.right_shift(raw, 3), np.left_shift(raw, 13)).astype(np.uint16)
    return decoded.astype(np.float32) / 1000.0


def decode_png_depth(path: str | Path, divisor: float) -> np.ndarray:
    raw = np.asarray(Image.open(path), dtype=np.float32)
    return raw / float(divisor)


def _resolve_subroot(root: str | Path, candidates: Sequence[str], marker: str) -> Path:
    base = Path(root).expanduser().resolve()
    for suffix in ("", *candidates):
        candidate = base / suffix if suffix else base
        if (candidate / marker).exists():
            return candidate
    tried = [str(base / suffix) if suffix else str(base) for suffix in ("", *candidates)]
    raise FileNotFoundError(f"Could not resolve dataset root containing {marker!r}; tried {tried}.")


def load_split_manifest(
    manifest_dir: str | Path,
    dataset_name: str,
    split: str,
    *,
    strict_count: bool = True,
) -> List[Dict[str, Any]]:
    """Load a checked JSONL split and verify its metadata/checksum."""
    domain = _canonical_domain_name(dataset_name)
    directory = Path(manifest_dir)
    path = directory / f"{domain}_{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing unified-depth split manifest: {path}. Run "
            "ft/kitti_stage1_5f/tools/prepare_unified_depth_splits.py; "
            "the dataset never creates manifests or depth caches implicitly."
        )
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} is not a JSON object.")
            records.append(value)

    expected = EXPECTED_SPLIT_COUNTS.get(domain, {}).get(split)
    if strict_count and expected is not None and len(records) != expected:
        raise RuntimeError(f"{path} contains {len(records)} records; expected {expected}.")

    metadata_path = directory / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema") != MANIFEST_SCHEMA:
            raise RuntimeError(f"Unsupported split metadata schema in {metadata_path}.")
        entry = metadata.get("manifests", {}).get(path.name)
        if entry is None:
            raise RuntimeError(f"{metadata_path} has no checksum entry for {path.name}.")
        if int(entry.get("count", -1)) != len(records):
            raise RuntimeError(f"Count checksum mismatch for {path}.")
        if str(entry.get("sha256", "")) != _sha256_file(path):
            raise RuntimeError(f"SHA256 checksum mismatch for {path}.")
    return records


class _UnifiedDepthBase(Dataset):
    dataset_name: str

    def __init__(
        self,
        *,
        input_long_side: int = 518,
        patch_size: int = 14,
        synthetic_sparse_points: int = 500,
        sampling_seed: int = 0,
    ) -> None:
        super().__init__()
        self.input_long_side = int(input_long_side)
        self.patch_size = int(patch_size)
        self.synthetic_sparse_points = int(synthetic_sparse_points)
        self.sampling_seed = int(sampling_seed)
        if self.input_long_side <= 0 or self.patch_size <= 0:
            raise ValueError("input_long_side and patch_size must be positive.")
        if self.synthetic_sparse_points <= 1:
            raise ValueError("synthetic_sparse_points must be greater than one.")
        self.depth_min, self.depth_max = DEPTH_RANGES[self.dataset_name]

    def _protocol_depth_mask(self, depth: np.ndarray) -> np.ndarray:
        finite = np.isfinite(depth) & (depth > 0.0)
        include_min, include_max = DEPTH_RANGE_INCLUSIVITY[self.dataset_name]
        lower = depth >= float(self.depth_min) if include_min else depth > float(self.depth_min)
        upper = depth <= float(self.depth_max) if include_max else depth < float(self.depth_max)
        return finite & lower & upper

    def _finish_sample(
        self,
        *,
        sample_id: str,
        image: np.ndarray,
        dense_depth: np.ndarray,
        intrinsics: np.ndarray,
        cam2world: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
        T_cam_from_points: Optional[np.ndarray] = None,
        native_sparse_depth: Optional[np.ndarray] = None,
        native_sparse_valid: Optional[np.ndarray] = None,
        synthetic_sparse: bool = False,
        sequence: Optional[str] = None,
        frame_id: int = 0,
    ) -> Dict[str, Any]:
        raw_native_depth = np.asarray(dense_depth, dtype=np.float32)
        raw_native_valid = np.isfinite(raw_native_depth) & (raw_native_depth > 0.0)
        # Keep raw finite-positive GT intact for sparse-point construction and
        # native-grid evaluation.  Dataset protocol ranges are supervision /
        # metric masks only and must never alter the sparse min/max scale.
        native_depth = np.where(raw_native_valid, raw_native_depth, 0.0).astype(np.float32)
        native_valid = self._protocol_depth_mask(native_depth)
        rgb, depth, image_mask, K, metadata = letterbox_rgb_depth(
            image,
            native_depth,
            intrinsics,
            input_long_side=self.input_long_side,
            patch_size=self.patch_size,
        )
        pixel_mask = image_mask & self._protocol_depth_mask(depth)
        depth = np.where(pixel_mask, depth, 0.0).astype(np.float32)

        if synthetic_sparse:
            sparse_depth, sparse_mask, native_sparse_mask = sample_unique_sparse_depth(
                native_depth,
                raw_native_valid,
                metadata,
                sample_id=sample_id,
                point_count=self.synthetic_sparse_points,
                sampling_seed=self.sampling_seed,
                return_native_mask=True,
            )
            native_sparse = np.where(native_sparse_mask, native_depth, 0.0).astype(np.float32)
            points_array = sparse_depth_to_camera_points(sparse_depth, K)
            transform = np.eye(4, dtype=np.float32)
        elif native_sparse_depth is not None:
            if native_sparse_valid is None:
                native_sparse_valid = np.asarray(native_sparse_depth) > 0
            sparse_depth, sparse_mask = resize_native_sparse_depth(
                native_sparse_depth, native_sparse_valid, metadata
            )
            native_sparse_mask = (
                np.asarray(native_sparse_valid, dtype=bool)
                & np.isfinite(native_sparse_depth)
                & (np.asarray(native_sparse_depth) > 0.0)
            )
            native_sparse = np.where(
                native_sparse_mask, np.asarray(native_sparse_depth, dtype=np.float32), 0.0
            ).astype(np.float32)
            points_array = sparse_depth_to_camera_points(sparse_depth, K)
            transform = np.eye(4, dtype=np.float32)
        else:
            if points is None or T_cam_from_points is None:
                raise RuntimeError(f"{sample_id}: native point inputs are missing.")
            points_array = np.ascontiguousarray(points, dtype=np.float32)
            transform = np.asarray(T_cam_from_points, dtype=np.float32)
            sparse_depth, sparse_mask = project_points_to_sparse_depth(
                points_array, transform, K, metadata.padded_hw
            )
            native_sparse, native_sparse_mask = project_points_to_sparse_depth(
                points_array, transform, intrinsics, native_depth.shape
            )
        projected = sparse_depth[sparse_mask]
        if projected.size < 2 or float(projected.max() - projected.min()) <= 1.0e-6:
            raise RuntimeError(
                f"{sample_id}: projected sparse input needs at least two distinct positive depths."
            )

        camera_to_world = (
            np.eye(4, dtype=np.float32)
            if cam2world is None
            else np.asarray(cam2world, dtype=np.float32)
        )
        padded_h, padded_w = metadata.padded_hw
        view = {
            "img": _normalize_rgb(rgb),
            "true_shape": np.asarray((padded_h, padded_w), dtype=np.int32),
            "camera_pose": np.eye(4, dtype=np.float32),
            "camera_intrinsics": K.astype(np.float32),
            "cam2world": camera_to_world,
            "timestep": 0,
            "is_raymap": False,
            "is_metric_scale": True,
            "frame_id": int(frame_id),
            "label": sample_id,
            "image_pixel_mask": torch.from_numpy(image_mask.copy()),
        }
        grid = _grid_tensors()
        return {
            "views": [view],
            "voxel_label": torch.zeros((1, 1, 1), dtype=torch.long),
            "T_target_from_refcam": torch.eye(4, dtype=torch.float32),
            "voxel_origin": grid["voxel_origin"],
            "voxel_size": grid["voxel_size"],
            "grid_size": grid["grid_size"],
            "half_voxel_origin": grid["half_voxel_origin"],
            "half_voxel_size": grid["half_voxel_size"],
            "half_grid_size": grid["half_grid_size"],
            "fusion_vox_origin": grid["fusion_vox_origin"],
            "fusion_vox_size": grid["fusion_vox_size"],
            "fusion_vox_grid": grid["fusion_vox_grid"],
            "dense_depth": torch.from_numpy(depth[None]),
            "dense_depth_pixel_mask": torch.from_numpy(pixel_mask[None].copy()),
            "dense_depth_frame_mask": torch.tensor([bool(pixel_mask.any())], dtype=torch.bool),
            "sparse_depth": torch.from_numpy(sparse_depth[None]),
            "sparse_depth_mask": torch.from_numpy(sparse_mask[None].copy()),
            "points_per_frame": [torch.from_numpy(points_array)],
            "T_cam_from_velo": torch.from_numpy(transform[None]),
            "K_per_frame": torch.from_numpy(K[None]),
            "image_hw": torch.tensor((padded_h, padded_w), dtype=torch.int32),
            "native_dense_depth": torch.from_numpy(native_depth),
            "native_valid_mask": torch.from_numpy(native_valid.copy()),
            "native_sparse_depth": torch.from_numpy(native_sparse),
            "native_sparse_depth_mask": torch.from_numpy(native_sparse_mask.copy()),
            "resize_metadata": metadata.as_dict(),
            "depth_range": torch.tensor((self.depth_min, self.depth_max), dtype=torch.float32),
            "dataset_name": self.dataset_name,
            "sample_id": sample_id,
            "sequence": sequence or self.dataset_name,
            "target_frame_id": int(frame_id),
            "frame_ids": (int(frame_id),),
            "num_frames": 1,
            "num_views": 1,
            "view_layout": "unified_single_frame",
        }



def _grid_tensors() -> Dict[str, torch.Tensor]:
    return {
        "grid_size": torch.tensor((600, 600, 50), dtype=torch.long),
        "voxel_origin": torch.tensor((0.0, -60.0, -5.0), dtype=torch.float32),
        "voxel_size": torch.tensor((0.2, 0.2, 0.2), dtype=torch.float32),
        "half_grid_size": torch.tensor((300, 300, 25), dtype=torch.long),
        "half_voxel_origin": torch.tensor((0.0, -60.0, -5.0), dtype=torch.float32),
        "half_voxel_size": torch.tensor((0.4, 0.4, 0.4), dtype=torch.float32),
        "fusion_vox_origin": torch.tensor((-60.0, -5.0, 0.0), dtype=torch.float32),
        "fusion_vox_size": torch.tensor((0.4, 0.4, 0.4), dtype=torch.float32),
        "fusion_vox_grid": torch.tensor((300, 25, 300), dtype=torch.long),
    }


def _parse_kitti_calib(path: Path) -> Dict[str, np.ndarray]:
    raw: Dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            key, values = line.split(":", 1)
            raw[key] = np.asarray([float(value) for value in values.split()], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :4] = raw["Tr"].reshape(3, 4)
    return {"P2": raw["P2"].reshape(3, 4), "Tr": transform}


def _kitti_cam2_from_velo(calib: Mapping[str, np.ndarray]) -> np.ndarray:
    projection = calib["P2"]
    rectified = np.eye(4, dtype=np.float64)
    rectified[:3, 3] = np.linalg.inv(projection[:, :3]) @ projection[:, 3]
    return (rectified @ calib["Tr"]).astype(np.float32)


class KITTISingleFrameUnifiedDepthDataset(_UnifiedDepthBase):
    dataset_name = "kitti"

    def __init__(
        self,
        processed_root: str,
        split: str = "train",
        *,
        manifest_dir: str,
        strict_count: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.root = Path(processed_root).expanduser().resolve()
        self.split = split
        self.records = load_split_manifest(
            manifest_dir, self.dataset_name, split, strict_count=strict_count
        )
        self._calibration: Dict[str, Dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _sequence_dir(self, sequence: str) -> Path:
        source = "train" if sequence != "08" else "val"
        return self.root / f"{source}_{sequence}"

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        sequence, frame = str(record["sequence"]), int(record["frame_id"])
        sequence_dir = self._sequence_dir(sequence)
        path = sequence_dir / f"{frame:06d}_0.npz"
        with np.load(path) as npz:
            image = np.asarray(npz["image"])
            intrinsics = np.asarray(npz["intrinsics"], dtype=np.float32)
            cam2world = np.asarray(npz["cam2world"], dtype=np.float32)
            if "dense_depthmap" not in npz.files:
                raise RuntimeError(f"KITTI record has no dense depth: {path}")
            depth = np.asarray(npz["dense_depthmap"], dtype=np.float32)
        if sequence not in self._calibration:
            self._calibration[sequence] = _parse_kitti_calib(sequence_dir / "calib.txt")
        transform = _kitti_cam2_from_velo(self._calibration[sequence])
        points = np.fromfile(
            sequence_dir / "lidar" / f"{frame:06d}.bin", dtype=np.float32
        ).reshape(-1, 4)
        sample = self._finish_sample(
            sample_id=str(record["sample_id"]),
            image=image,
            dense_depth=depth,
            intrinsics=intrinsics,
            cam2world=cam2world,
            points=points,
            T_cam_from_points=transform,
            sequence=sequence,
            frame_id=frame,
        )
        sample["views"][0]["camera_name"] = "image_02"
        sample["camera_indices"], sample["camera_names"] = (0,), ("image_02",)
        return sample


def _quaternion_matrix(q: Mapping[str, float]) -> np.ndarray:
    values = np.asarray([q["qw"], q["qx"], q["qy"], q["qz"]], dtype=np.float64)
    values /= np.linalg.norm(values)
    w, x, y, z = values
    return np.asarray([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _pose_matrix(pose: Mapping[str, Any]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _quaternion_matrix(pose["rotation"])
    translation = pose["translation"]
    result[:3, 3] = [translation["x"], translation["y"], translation["z"]]
    return result


class DDADCamera01UnifiedDepthDataset(_UnifiedDepthBase):
    dataset_name = "ddad"

    def __init__(
        self,
        processed_root: str,
        split: str = "train",
        *,
        manifest_dir: str,
        strict_count: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.root = Path(processed_root).expanduser().resolve()
        self.metadata_root = self.root / "metadata"
        self.split = split
        self.records = load_split_manifest(
            manifest_dir, self.dataset_name, split, strict_count=strict_count
        )
        self._scene_files: Optional[Dict[str, List[str]]] = None
        self._pose_cache: Dict[str, List[np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _metadata_files(self) -> Dict[str, List[str]]:
        if self._scene_files is None:
            with (self.metadata_root / "ddad.json").open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            splits = metadata.get("scene_splits", {})
            self._scene_files = {
                "train": [str(value) for value in splits.get("0", {}).get("filenames", [])],
                "val": [str(value) for value in splits.get("1", {}).get("filenames", [])],
            }
        return self._scene_files

    def _lidar_poses(self, scene: str) -> List[np.ndarray]:
        if scene in self._pose_cache:
            return self._pose_cache[scene]
        split, index_text = scene.rsplit("_", 1)
        path = self.metadata_root / self._metadata_files()[split][int(index_text)]
        with path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        data = {item["key"]: item for item in metadata.get("data", [])}
        poses = []
        for sample in metadata.get("samples", []):
            lidar = next(
                data[key]
                for key in sample.get("datum_keys", [])
                if data.get(key, {}).get("id", {}).get("name") == "LIDAR"
            )
            pose = lidar["datum"]["point_cloud"]["pose"]
            poses.append(_pose_matrix(pose).astype(np.float32))
        self._pose_cache[scene] = poses
        return poses

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        scene, frame = str(record["scene"]), int(record["frame_id"])
        scene_dir = self.root / scene
        path = scene_dir / f"{frame:06d}_0.npz"
        with np.load(path) as npz:
            image = np.asarray(npz["image"])
            intrinsics = np.asarray(npz["intrinsics"], dtype=np.float32)
            cam2world = np.asarray(npz["cam2world"], dtype=np.float32)
            depth = np.asarray(npz["depthmap"], dtype=np.float32)
        point_path = scene_dir / "point_cloud" / "LIDAR" / f"{frame:06d}.npz"
        with np.load(point_path) as npz:
            key = "data" if "data" in npz.files else npz.files[0]
            points = np.asarray(npz[key], dtype=np.float32)[:, :4]
        transform = (
            np.linalg.inv(cam2world.astype(np.float64)) @ self._lidar_poses(scene)[frame]
        ).astype(np.float32)
        sample = self._finish_sample(
            sample_id=str(record["sample_id"]),
            image=image,
            dense_depth=depth,
            intrinsics=intrinsics,
            cam2world=cam2world,
            points=points,
            T_cam_from_points=transform,
            sequence=scene,
            frame_id=frame,
        )
        sample["views"][0]["camera_name"] = "CAMERA_01"
        sample["camera_indices"], sample["camera_names"] = (0,), ("CAMERA_01",)
        return sample


class _ManifestDepthDataset(_UnifiedDepthBase):
    def __init__(
        self,
        root: str,
        split: str,
        *,
        manifest_dir: str,
        strict_count: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.records = load_split_manifest(
            manifest_dir, self.dataset_name, split, strict_count=strict_count
        )

    def __len__(self) -> int:
        return len(self.records)


class SevenScenesUnifiedDepthDataset(_ManifestDepthDataset):
    dataset_name = "7scenes"

    def __init__(self, root: str, *args: Any, **kwargs: Any) -> None:
        root = str(_resolve_subroot(root, ("raw",), "chess/TrainSplit.txt"))
        super().__init__(root, *args, **kwargs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image = np.asarray(Image.open(self.root / record["image"]).convert("RGB"))
        raw_depth = np.asarray(Image.open(self.root / record["depth"]), dtype=np.uint16)
        valid = (raw_depth > 0) & (raw_depth < np.iinfo(np.uint16).max)
        depth = np.where(valid, raw_depth.astype(np.float32) / 1000.0, 0.0)
        pose_path = self.root / record["pose"]
        cam2world = np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
        return self._finish_sample(
            sample_id=str(record["sample_id"]),
            image=image,
            dense_depth=depth,
            intrinsics=SEVEN_SCENES_INTRINSICS,
            cam2world=cam2world,
            synthetic_sparse=True,
            sequence=str(record["sequence"]),
            frame_id=int(record["frame_id"]),
        )


class NYUv2UnifiedDepthDataset(_ManifestDepthDataset):
    dataset_name = "nyuv2"

    def __init__(self, root: str, *args: Any, **kwargs: Any) -> None:
        root_path = _resolve_subroot(root, (), "nyu_depth_v2_labeled.mat")
        self.mat_path = root_path / "nyu_depth_v2_labeled.mat"
        self._h5: Any = None
        super().__init__(str(root_path), *args, **kwargs)

    def _file(self) -> Any:
        if self._h5 is None:
            try:
                import h5py
            except ImportError as exc:
                raise ImportError("NYUv2 labeled MAT loading requires h5py.") from exc
            self._h5 = h5py.File(self.mat_path, "r")
        return self._h5

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self) -> None:
        handle = getattr(self, "_h5", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        mat_index = int(record["mat_index"])
        handle = self._file()
        # MATLAB v7.3 stores these arrays as (C,W,H)/(W,H).
        image = np.asarray(handle["images"][mat_index], dtype=np.uint8).transpose(2, 1, 0)
        depth = np.asarray(handle["depths"][mat_index], dtype=np.float32).T
        return self._finish_sample(
            sample_id=str(record["sample_id"]),
            image=image,
            dense_depth=depth,
            intrinsics=NYUV2_RGB_INTRINSICS,
            synthetic_sparse=True,
            sequence=str(record.get("scene", "nyuv2")),
            frame_id=mat_index,
        )


class SUNRGBDUnifiedDepthDataset(_ManifestDepthDataset):
    dataset_name = "sunrgbd"

    def __init__(self, root: str, *args: Any, **kwargs: Any) -> None:
        root = str(
            _resolve_subroot(
                root,
                ("OpenDataLab___SUN_RGB-D/raw/SUNRGBD", "raw/SUNRGBD", "SUNRGBD"),
                "kv1",
            )
        )
        super().__init__(root, *args, **kwargs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image = np.asarray(Image.open(self.root / record["image"]).convert("RGB"))
        encoded = np.asarray(Image.open(self.root / record["depth"]), dtype=np.uint16)
        depth = decode_sunrgbd_depth(encoded)
        K = np.loadtxt(self.root / record["intrinsics"], dtype=np.float32).reshape(3, 3)
        return self._finish_sample(
            sample_id=str(record["sample_id"]),
            image=image,
            dense_depth=depth,
            intrinsics=K,
            synthetic_sparse=True,
            sequence=str(record.get("sequence", record["sample_id"])),
            frame_id=int(record.get("frame_id", index)),
        )


def _resolve_void_path(root: Path, manifest_value: str) -> Path:
    value = str(manifest_value).replace("\\", "/").lstrip("./")
    candidates = [root / value]
    if value.startswith("void_500/data/"):
        candidates.append(root / value[len("void_500/data/") :])
    if value.startswith("void_500/"):
        candidates.append(root.parent / value)
    if value.startswith("data/"):
        candidates.append(root / value[len("data/") :])
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"VOID manifest path {manifest_value!r} did not resolve under {root}.")


class VOID500UnifiedDepthDataset(_ManifestDepthDataset):
    dataset_name = "void"

    def __init__(self, root: str, *args: Any, **kwargs: Any) -> None:
        root_path = _resolve_subroot(root, ("void_500",), "train_image.txt")
        super().__init__(str(root_path), *args, **kwargs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image = np.asarray(Image.open(_resolve_void_path(self.root, record["image"])).convert("RGB"))
        sparse = decode_png_depth(_resolve_void_path(self.root, record["sparse_depth"]), 256.0)
        validity = np.asarray(Image.open(_resolve_void_path(self.root, record["validity_map"]))) > 0
        depth = decode_png_depth(_resolve_void_path(self.root, record["ground_truth"]), 256.0)
        K = np.loadtxt(_resolve_void_path(self.root, record["intrinsics"]), dtype=np.float32).reshape(3, 3)
        pose_values = np.loadtxt(
            _resolve_void_path(self.root, record["absolute_pose"]), dtype=np.float32
        )
        if pose_values.size == 12:
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :4] = pose_values.reshape(3, 4)
        elif pose_values.size == 16:
            pose = pose_values.reshape(4, 4)
        else:
            raise RuntimeError(
                f"VOID pose must contain 12 or 16 values, got {pose_values.size}."
            )
        return self._finish_sample(
            sample_id=str(record["sample_id"]),
            image=image,
            dense_depth=depth,
            intrinsics=K,
            cam2world=pose,
            native_sparse_depth=sparse,
            native_sparse_valid=validity,
            sequence=str(record["sequence"]),
            frame_id=int(record.get("frame_id", index)),
        )


def build_unified_depth_dataset(
    dataset_name: str,
    root: str,
    split: str,
    *,
    manifest_dir: Optional[str] = None,
    input_long_side: int = 518,
    patch_size: int = 14,
    synthetic_sparse_points: int = 500,
    sampling_seed: int = 0,
    strict_count: bool = True,
) -> Dataset:
    """Build one domain adapter without creating any data/cache files."""
    domain = _canonical_domain_name(dataset_name)
    common = dict(
        input_long_side=input_long_side,
        patch_size=patch_size,
        synthetic_sparse_points=synthetic_sparse_points,
        sampling_seed=sampling_seed,
        strict_count=strict_count,
    )
    if not manifest_dir:
        raise ValueError(f"manifest_dir is required for {domain}.")
    if domain == "kitti":
        return KITTISingleFrameUnifiedDepthDataset(
            root, split, manifest_dir=manifest_dir, **common
        )
    if domain == "ddad":
        return DDADCamera01UnifiedDepthDataset(
            root, split, manifest_dir=manifest_dir, **common
        )
    cls = {
        "7scenes": SevenScenesUnifiedDepthDataset,
        "nyuv2": NYUv2UnifiedDepthDataset,
        "sunrgbd": SUNRGBDUnifiedDepthDataset,
        "void": VOID500UnifiedDepthDataset,
    }[domain]
    return cls(root, split, manifest_dir=manifest_dir, **common)


class UnifiedSixDataset(Dataset):
    """Natural concatenation of domains, with explicit indices for balancing."""

    def __init__(self, datasets: Mapping[str, Dataset]) -> None:
        canonical: Dict[str, Dataset] = {_canonical_domain_name(k): v for k, v in datasets.items()}
        missing = [name for name in DOMAIN_NAMES if name not in canonical]
        if missing:
            raise ValueError(f"UnifiedSixDataset is missing domains: {missing}.")
        self.domain_names = DOMAIN_NAMES
        self.datasets = tuple(canonical[name] for name in self.domain_names)
        self.offsets: List[int] = []
        self.domain_indices: Dict[str, range] = {}
        offset = 0
        for name, dataset in zip(self.domain_names, self.datasets):
            self.offsets.append(offset)
            self.domain_indices[name] = range(offset, offset + len(dataset))
            offset += len(dataset)
        self.total_length = offset

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        domain_index = int(np.searchsorted(self.offsets, index, side="right") - 1)
        return self.datasets[domain_index][index - self.offsets[domain_index]]


def build_unified_six_dataset(
    roots: Mapping[str, str],
    split: str,
    *,
    manifest_dir: str,
    **kwargs: Any,
) -> UnifiedSixDataset:
    canonical_roots = {_canonical_domain_name(k): v for k, v in roots.items()}
    missing = [name for name in DOMAIN_NAMES if name not in canonical_roots]
    if missing:
        raise ValueError(f"Missing unified-depth roots for {missing}.")
    datasets = {
        name: build_unified_depth_dataset(
            name,
            canonical_roots[name],
            split,
            manifest_dir=manifest_dir,
            **kwargs,
        )
        for name in DOMAIN_NAMES
    }
    return UnifiedSixDataset(datasets)


class DomainBalancedDistributedSampler(Sampler[int]):
    """Equal-domain sampling with deterministic full-run coverage.

    ``samples_per_epoch=0`` derives a per-domain quota from the largest domain
    and ``coverage_epochs``.  Each domain is then traversed through persistent,
    independently shuffled no-replacement cycles, so every training sample is
    seen at least once during the run while every logical epoch remains exactly
    domain-balanced.  DDP-only padding rotates between domains by epoch.
    """

    def __init__(
        self,
        dataset: UnifiedSixDataset,
        *,
        samples_per_epoch: int = 0,
        coverage_epochs: int = 20,
        seed: int = 0,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ) -> None:
        if not isinstance(dataset, UnifiedSixDataset):
            raise TypeError("DomainBalancedDistributedSampler requires UnifiedSixDataset.")
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if int(num_replicas) <= 0 or not 0 <= int(rank) < int(num_replicas):
            raise ValueError(f"Invalid num_replicas/rank: {num_replicas}/{rank}.")
        if int(coverage_epochs) <= 0:
            raise ValueError(f"coverage_epochs must be positive, got {coverage_epochs}.")
        self.domain_lengths = tuple(len(domain_dataset) for domain_dataset in dataset.datasets)
        if any(length <= 0 for length in self.domain_lengths):
            empty = [
                name
                for name, length in zip(dataset.domain_names, self.domain_lengths)
                if length <= 0
            ]
            raise RuntimeError(f"Unified depth domains must be non-empty, got {empty}.")
        self.coverage_epochs = int(coverage_epochs)
        if int(samples_per_epoch) == 0:
            samples_per_domain = int(
                math.ceil(max(self.domain_lengths) / self.coverage_epochs)
            )
            samples_per_epoch = samples_per_domain * len(dataset.domain_names)
        if int(samples_per_epoch) < 0 or int(samples_per_epoch) % len(dataset.domain_names) != 0:
            raise ValueError(
                "samples_per_epoch must be zero (automatic coverage) or positive "
                f"and divisible by {len(dataset.domain_names)}."
            )
        self.dataset = dataset
        self.logical_samples_per_epoch = int(samples_per_epoch)
        self.samples_per_domain = self.logical_samples_per_epoch // len(dataset.domain_names)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.guarantees_full_domain_coverage = (
            self.samples_per_domain * self.coverage_epochs >= max(self.domain_lengths)
        )
        self.num_samples = int(math.ceil(self.logical_samples_per_epoch / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _draw_domain(self, indices: Sequence[int], count: int, domain_index: int) -> List[int]:
        if len(indices) == 0:
            raise RuntimeError(f"Domain {self.dataset.domain_names[domain_index]} is empty.")
        result: List[int] = []
        position = self.epoch * count
        while len(result) < count:
            cycle, offset = divmod(position, len(indices))
            if self.shuffle:
                generator = torch.Generator()
                generator.manual_seed(
                    _stable_uint64("domain-cycle", self.seed, domain_index, cycle)
                    % (2**63 - 1)
                )
                order = torch.randperm(len(indices), generator=generator).tolist()
            else:
                order = list(range(len(indices)))
            take = min(count - len(result), len(order) - offset)
            result.extend(int(indices[order[i]]) for i in range(offset, offset + take))
            position += take
        return result

    def global_indices(self) -> List[int]:
        logical: List[int] = []
        per_domain_draws: List[List[int]] = []
        for domain_index, name in enumerate(self.dataset.domain_names):
            draws = self._draw_domain(
                self.dataset.domain_indices[name], self.samples_per_domain, domain_index
            )
            per_domain_draws.append(draws)
            logical.extend(draws)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(_stable_uint64("interleave", self.seed, self.epoch) % (2**63 - 1))
            order = torch.randperm(len(logical), generator=generator).tolist()
            logical = [logical[i] for i in order]

        padding = self.total_size - len(logical)
        for offset in range(padding):
            domain_index = (self.epoch * max(1, padding) + offset) % len(self.dataset.domain_names)
            candidates = per_domain_draws[domain_index]
            logical.append(candidates[(self.epoch + offset) % len(candidates)])
        return logical

    def __iter__(self) -> Iterator[int]:
        indices = self.global_indices()[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError(f"Sampler rank {self.rank} produced {len(indices)} != {self.num_samples}.")
        return iter(indices)


def _collate_views(batch: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if any(len(item["views"]) != 1 for item in batch):
        raise RuntimeError("Unified depth collate accepts exactly one view per sample.")
    per_view = [item["views"][0] for item in batch]
    result: Dict[str, Any] = {}
    for key in per_view[0]:
        values = [view[key] for view in per_view]
        first = values[0]
        if isinstance(first, torch.Tensor):
            result[key] = torch.stack(values, dim=0)
        elif isinstance(first, np.ndarray):
            result[key] = torch.from_numpy(np.stack(values, axis=0))
        elif isinstance(first, (int, float, bool)):
            result[key] = torch.tensor(values)
        else:
            result[key] = values
    return [result]


def collate_unified_depth(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate unified samples; mixed shapes intentionally require batch size 1."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    shapes = {tuple(item["dense_depth"].shape) for item in batch}
    if len(shapes) != 1:
        raise RuntimeError(
            "Dynamic unified-depth inputs must use batch_size=1 unless all samples "
            f"share a shape; got {sorted(shapes)}."
        )
    tensor_keys = (
        "voxel_label",
        "T_target_from_refcam",
        "voxel_origin",
        "voxel_size",
        "grid_size",
        "half_voxel_origin",
        "half_voxel_size",
        "half_grid_size",
        "fusion_vox_origin",
        "fusion_vox_size",
        "fusion_vox_grid",
        "dense_depth",
        "dense_depth_pixel_mask",
        "dense_depth_frame_mask",
        "sparse_depth",
        "sparse_depth_mask",
        "T_cam_from_velo",
        "K_per_frame",
        "image_hw",
        "depth_range",
    )
    out: Dict[str, Any] = {"views": _collate_views(batch)}
    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    optional_tensor_keys = ("ddp_loss_scale", "sampler_padding")
    for key in optional_tensor_keys:
        present = [key in item for item in batch]
        if any(present) and not all(present):
            raise RuntimeError(f"Unified batch only partially defines optional key {key!r}.")
        if all(present):
            out[key] = torch.stack([item[key] for item in batch], dim=0)
    out["points_per_frame"] = [item["points_per_frame"] for item in batch]
    for key in (
        "native_dense_depth",
        "native_valid_mask",
        "native_sparse_depth",
        "native_sparse_depth_mask",
        "resize_metadata",
        "dataset_name",
        "sample_id",
        "sequence",
        "target_frame_id",
        "frame_ids",
        "num_frames",
        "num_views",
        "view_layout",
    ):
        out[key] = [item[key] for item in batch]
    return out


def collate_unified_depth_online_knn4(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Unified collate with an in-memory KNN map used only for output scale."""

    out = collate_unified_depth(batch)
    out["knn_depth"] = online_knn4_depth_from_sparse(
        out["sparse_depth"],
        sample_ids=out["sample_id"],
    )
    return out


__all__ = [
    "DOMAIN_NAMES",
    "EXPECTED_SPLIT_COUNTS",
    "DEPTH_RANGES",
    "DEPTH_RANGE_INCLUSIVITY",
    "MANIFEST_SCHEMA",
    "ResizeMetadata",
    "compute_letterbox_metadata",
    "letterbox_rgb_depth",
    "sample_unique_sparse_depth",
    "resize_native_sparse_depth",
    "sparse_depth_to_camera_points",
    "project_points_to_sparse_depth",
    "decode_sunrgbd_depth",
    "load_split_manifest",
    "KITTISingleFrameUnifiedDepthDataset",
    "DDADCamera01UnifiedDepthDataset",
    "SevenScenesUnifiedDepthDataset",
    "NYUv2UnifiedDepthDataset",
    "SUNRGBDUnifiedDepthDataset",
    "VOID500UnifiedDepthDataset",
    "UnifiedSixDataset",
    "DomainBalancedDistributedSampler",
    "build_unified_depth_dataset",
    "build_unified_six_dataset",
    "collate_unified_depth",
    "online_knn4_depth_from_sparse",
    "collate_unified_depth_online_knn4",
]
